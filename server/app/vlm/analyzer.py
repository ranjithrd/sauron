"""
analyzer.py — structured drone analysis via litellm + OpenRouter.

Uses google/gemini-flash-1.5-8b (cheap, fast, good vision) by default.
Override with VLM_MODEL env var.  Any OpenRouter-supported vision model works.

Returns a JSON dict with:
  is_drone    — bool
  drone_type  — str, ≤3 words ("DJI Phantom 4", "unknown")
  action      — str, single phrase ("hovering", "ascending fast", "circling")
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert UAV / drone analyst reviewing a security camera frame.
Respond ONLY with a valid JSON object — no markdown, no explanation, no extra text.
JSON schema (strict):
{
  "is_drone": <bool>,
  "drone_type": "<brand model variant, max 3 words, or 'unknown' if not a drone>",
  "action": "<what the drone is doing, max 5 words>"
}
"""

_USER_TEMPLATE = """\
Analyze this security camera frame.

Speed context (from radar/triangulation):
  horizontal speed : {speed_ms:.1f} m/s  ({speed_kmh:.1f} km/h)
  altitude estimate: {altitude_m}
  classification   : speed < 0.5 m/s = hovering, 0.5–3 = slow movement, > 3 = active flight

Classify the most prominent airborne object visible:
(a) Is it a drone? (true/false)
(b) Best guess drone type — brand/model in ≤3 words (e.g. "DJI Phantom 4", "FPV racing quad", "fixed wing", "unknown")
(c) What is it doing? One short phrase based on the image and speed data above.

Return ONLY the JSON object.
"""


def _download_s3_image(bucket: str, key: str, region: str) -> bytes:
    import boto3  # type: ignore[import]

    kwargs: dict = {"region_name": region} if region else {}
    s3 = boto3.client("s3", **kwargs)
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    return buf.getvalue()


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of model output and parse it."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(text[start:end])


async def analyze_snapshot(
    s3_key: str,
    device_id: Optional[str],
    bucket: str,
    region: str,
    model: str,
    api_key: str,
    vel_lat: float = 0.0,
    vel_lon: float = 0.0,
    altitude_m: Optional[float] = None,
) -> tuple[str, int]:
    """
    Download image from S3, send to VLM, return (json_result_str, duration_ms).

    The result string is a JSON object with keys: is_drone, drone_type, action.
    Raises on fatal error.
    """
    try:
        import litellm  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "litellm is not installed. Run: pip install litellm"
        ) from exc

    t0 = time.monotonic()

    image_bytes = _download_s3_image(bucket, s3_key, region)
    b64 = base64.b64encode(image_bytes).decode("ascii")

    # Compute speed in m/s from vel_lat / vel_lon (degrees/s)
    import math
    speed_ms = math.sqrt(vel_lat**2 + vel_lon**2) * 111_320.0
    speed_kmh = speed_ms * 3.6
    alt_str = f"{altitude_m:.1f} m" if altitude_m is not None else "unknown"

    user_text = _USER_TEMPLATE.format(
        speed_ms=speed_ms,
        speed_kmh=speed_kmh,
        altitude_m=alt_str,
    )

    # litellm with OpenRouter: model = "openrouter/google/gemini-flash-1.5-8b"
    # api_key = OpenRouter API key (sk-or-...)
    response = await litellm.acompletion(
        model=model,
        api_key=api_key or None,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        max_tokens=150,
        temperature=0.1,
    )

    duration_ms = int((time.monotonic() - t0) * 1000)
    raw = response.choices[0].message.content or ""

    # Parse and validate the JSON response
    try:
        parsed = _extract_json(raw)
        # Normalise fields
        result = {
            "is_drone": bool(parsed.get("is_drone", False)),
            "drone_type": str(parsed.get("drone_type", "unknown"))[:64],
            "action": str(parsed.get("action", "unknown"))[:128],
        }
        result_str = json.dumps(result)
    except Exception as parse_exc:
        logger.warning(
            "VLM: could not parse JSON response (%s) — raw: %r", parse_exc, raw
        )
        # Fall back to storing raw text so it's still useful
        result_str = json.dumps({
            "is_drone": None,
            "drone_type": "parse error",
            "action": raw[:200],
        })

    logger.info(
        "VLM: %s via %s in %dms — result=%s",
        s3_key,
        model,
        duration_ms,
        result_str,
    )
    return result_str, duration_ms
