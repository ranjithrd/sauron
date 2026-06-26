"""
analyzer.py — provider-agnostic VLM image analysis via litellm.

Configure via environment variables:
  VLM_MODEL   — litellm model string, e.g. "gpt-4o", "claude-3-5-sonnet-20241022",
                 "gemini/gemini-pro-vision", "ollama/llava"
  VLM_API_KEY — provider API key (OpenAI, Anthropic, Google, etc.)
  AWS_S3_BUCKET / AWS_REGION — for downloading the snapshot from S3

The function is a stub: the prompt is minimal and can be replaced with
any scene-analysis or object-description prompt for the real demo.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _download_s3_image(bucket: str, key: str, region: str) -> bytes:
    import boto3  # type: ignore[import]

    kwargs: dict = {"region_name": region} if region else {}
    s3 = boto3.client("s3", **kwargs)
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    return buf.getvalue()


async def analyze_snapshot(
    s3_key: str,
    device_id: Optional[str],
    bucket: str,
    region: str,
    model: str,
    api_key: str,
) -> tuple[str, int]:
    """
    Download image from S3 and send it to the configured VLM.

    Returns (result_text, duration_ms).  Raises on error.
    """
    try:
        import litellm  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "litellm is not installed. Add it to requirements.txt and reinstall."
        ) from exc

    t0 = time.monotonic()

    image_bytes = _download_s3_image(bucket, s3_key, region)
    b64 = base64.b64encode(image_bytes).decode("ascii")

    response = await litellm.acompletion(
        model=model,
        api_key=api_key or None,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are a surveillance analyst. "
                            "Briefly describe what you see in this camera frame: "
                            "any people, vehicles, or notable activity. "
                            "Be concise (2-3 sentences)."
                        ),
                    },
                ],
            }
        ],
        max_tokens=200,
    )

    duration_ms = int((time.monotonic() - t0) * 1000)
    result = response.choices[0].message.content or ""
    logger.info(
        "VLM: analyzed %s via %s in %dms — %d chars",
        s3_key,
        model,
        duration_ms,
        len(result),
    )
    return result, duration_ms
