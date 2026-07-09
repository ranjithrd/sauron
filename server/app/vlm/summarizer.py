"""
summarizer.py — LLM-based situational summary of current tracking state.

Uses a cheap text-only model (Qwen3 8B via OpenRouter) to produce a concise
operator-style situation report from live track + camera data.
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 30.0

_cache: Dict[str, Any] = {"result": None, "error": None, "ts": 0.0, "duration_ms": 0}


def _cache_fresh() -> bool:
    return bool(_cache["result"]) and (time.time() - _cache["ts"] < _CACHE_TTL_S)


def cached_summary() -> Dict[str, Any]:
    return dict(_cache)


async def generate_summary(
    tracks: list,
    cameras: list,
    vlm_history: list,
    model: str,
    api_key: str,
    force: bool = False,
) -> Dict[str, Any]:
    if not force and _cache_fresh():
        return dict(_cache)

    t0 = time.monotonic()
    try:
        result = await _call_llm(tracks, cameras, vlm_history, model, api_key)
        _cache.update({"result": result, "error": None, "ts": time.time(),
                       "duration_ms": int((time.monotonic() - t0) * 1000)})
    except Exception as exc:
        err = str(exc)
        logger.error("Summarizer: LLM call failed: %s", err)
        _cache.update({"result": None, "error": err, "ts": time.time(),
                       "duration_ms": int((time.monotonic() - t0) * 1000)})

    return dict(_cache)


async def _call_llm(tracks: list, cameras: list, vlm_history: list, model: str, api_key: str) -> str:
    import litellm  # type: ignore[import]

    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Build compact data blob for the prompt
    track_lines = []
    for t in tracks:
        speed_ms = math.sqrt((t.get("vel_lat") or 0) ** 2 + (t.get("vel_lon") or 0) ** 2) * 111_320
        alt = f"{t['altitude_m']:.0f}m" if t.get("altitude_m") is not None else "?"
        track_lines.append(
            f"  - {t['object_id']}: lat={t['lat']:.5f} lon={t['lon']:.5f} "
            f"alt={alt} speed={speed_ms:.1f}m/s sources={t.get('source_cameras', [])}"
        )

    cam_lines = []
    for c in cameras:
        age_s = int(time.time() - time.mktime(time.strptime(c["last_seen"][:19], "%Y-%m-%dT%H:%M:%S"))) if c.get("last_seen") else "?"
        cam_lines.append(
            f"  - {c['device_id']}: heading={c.get('last_heading', '?'):.0f}° "
            f"pitch={c.get('last_pitch', '?'):.0f}° last_seen={age_s}s ago"
            if isinstance(age_s, int) else
            f"  - {c['device_id']}: heading={c.get('last_heading', '?')} last_seen=unknown"
        )

    vlm_lines = []
    for entry in vlm_history[:2]:
        if entry.get("result"):
            try:
                r = json.loads(entry["result"])
                vlm_lines.append(
                    f"  - {entry.get('device_id','?')} @ {time.strftime('%H:%MZ', time.gmtime(entry['run_ts']))}: "
                    f"drone={r.get('is_drone')} type={r.get('drone_type')} action={r.get('action')}"
                )
            except Exception:
                pass
        elif entry.get("error"):
            vlm_lines.append(f"  - VLM error: {entry['error'][:80]}")

    data_block = f"""UTC: {now_utc}

ACTIVE TRACKS ({len(tracks)}):
{chr(10).join(track_lines) if track_lines else '  none'}

CAMERAS ({len(cameras)}):
{chr(10).join(cam_lines) if cam_lines else '  none'}

RECENT VLM ANALYSES:
{chr(10).join(vlm_lines) if vlm_lines else '  none'}"""

    prompt = f"""You are a concise UAV surveillance operator writing a quick situation report.
Given the live tracking data below, write 2-4 short sentences covering:
- what is being tracked and its behaviour
- camera coverage and health
- any notable observations or concerns
- overall threat assessment (low / medium / high)

Be direct and specific. Use numbers. No bullet points — flowing prose only.

DATA:
{data_block}"""

    response = await litellm.acompletion(
        model=model,
        api_key=api_key or None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4,
    )
    return (response.choices[0].message.content or "").strip()
