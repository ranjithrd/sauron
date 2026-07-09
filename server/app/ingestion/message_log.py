"""In-memory ring buffer for the last N MQTT messages."""
from __future__ import annotations

import datetime
from collections import deque
from typing import Any

_MAX = 60
_log: deque[dict[str, Any]] = deque(maxlen=_MAX)


def record(topic: str, device_id: str, payload: dict[str, Any] | None) -> None:
    _log.appendleft({
        "time": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "topic": topic,
        "device_id": device_id,
        "payload": payload,
    })


def get_recent(n: int = 30) -> list[dict[str, Any]]:
    return list(_log)[:n]
