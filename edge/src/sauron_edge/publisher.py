"""
AWS IoT Core publisher via Greengrass v2 IPC.

When running as a Greengrass component, the nucleus injects the IPC socket
path and auth token into the environment automatically.  The awsiotsdk
greengrasscoreipc module discovers these and connects without any explicit
configuration.

Device ID is taken from the AWS_IOT_THING_NAME environment variable, which
Greengrass also sets automatically for every component.  Falls back to the
machine hostname for local development/testing.

Topic convention (matches server ingestion):
    devices/{device_id}/telemetry
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Optional

logger = logging.getLogger(__name__)

_THING_NAME_ENV = "AWS_IOT_THING_NAME"
_PUBLISH_TIMEOUT_S = 10.0

# Reconnect backoff: starts at 5 s, doubles on each consecutive failure,
# caps at 60 s.  Resets to the initial value on a successful connection.
_RECONNECT_BACKOFF_INITIAL_S = 5.0
_RECONNECT_BACKOFF_MAX_S = 60.0


def resolve_device_id() -> str:
    """
    Resolve this device's identifier.

    Greengrass v2 automatically sets AWS_IOT_THING_NAME for every component
    it launches.  If that variable is absent (local dev/test), we fall back
    to the machine hostname.
    """
    thing_name = os.environ.get(_THING_NAME_ENV, "").strip()
    if thing_name:
        logger.info(
            "Publisher: device_id = %r (from %s env var)", thing_name, _THING_NAME_ENV
        )
        return thing_name

    hostname = socket.gethostname()
    logger.warning(
        "Publisher: %s env var not set — using hostname %r as device_id. "
        "This is expected only during local testing outside Greengrass.",
        _THING_NAME_ENV,
        hostname,
    )
    return hostname


class GreengrassPublisher:
    """
    Publishes JSON telemetry payloads to an AWS IoT Core topic via the
    Greengrass v2 IPC client (awsiot.greengrasscoreipc).

    The IPC connection is established lazily on the first call to publish()
    (or explicitly via connect()).  If publishing fails, the connection is
    marked dirty and re-established on the next attempt.
    """

    def __init__(self) -> None:
        self._device_id: str = resolve_device_id()
        self._topic: str = f"devices/{self._device_id}/telemetry"
        self._client: Optional[object] = None
        self._connected: bool = False
        self._publish_count: int = 0
        self._failure_count: int = 0

        # Reconnect backoff state
        self._reconnect_backoff_s: float = _RECONNECT_BACKOFF_INITIAL_S
        self._next_reconnect_at: float = 0.0

        # Publish-level backoff — set when an UnauthorizedError is received
        self._publish_backoff_until: float = 0.0

        logger.info(
            "Publisher: initialised — device_id=%s topic=%s",
            self._device_id,
            self._topic,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self._connected and self._client is not None:
            logger.debug("Publisher: already connected — skipping connect()")
            return True

        now = time.monotonic()
        if now < self._next_reconnect_at:
            remaining = self._next_reconnect_at - now
            logger.debug(
                "Publisher: reconnect backoff active — %.1f s remaining", remaining
            )
            return False

        # Close any previous (dirty/stale) connection before opening a new one —
        # otherwise every reconnect leaks a native IPC connection/thread on both
        # this process and the Nucleus side.
        if self._client is not None:
            try:
                self._client.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._client = None

        logger.info(
            "Publisher: connecting to Greengrass nucleus IPC... (backoff=%.1fs)",
            self._reconnect_backoff_s,
        )
        try:
            import awsiot.greengrasscoreipc as gg_ipc  # type: ignore[import]

            self._client = gg_ipc.connect()
            self._connected = True
            self._reconnect_backoff_s = _RECONNECT_BACKOFF_INITIAL_S
            self._next_reconnect_at = 0.0
            logger.info("Publisher: Greengrass IPC connection established")
            return True

        except ImportError:
            logger.error(
                "Publisher: awsiotsdk is not installed — cannot connect to Greengrass IPC. "
                "Install it with: uv pip install awsiotsdk"
            )
            self._connected = False
            return False

        except Exception as exc:
            logger.error(
                "Publisher: failed to connect to Greengrass IPC — %s. "
                "Next attempt in %.1f s.",
                exc,
                self._reconnect_backoff_s,
            )
            self._client = None
            self._connected = False
            self._next_reconnect_at = time.monotonic() + self._reconnect_backoff_s
            self._reconnect_backoff_s = min(
                self._reconnect_backoff_s * 2.0, _RECONNECT_BACKOFF_MAX_S
            )
            return False

    def disconnect(self) -> None:
        """Close the IPC connection gracefully."""
        if self._client is not None:
            try:
                self._client.close()  # type: ignore[union-attr]
                logger.info("Publisher: IPC connection closed")
            except Exception as exc:
                logger.debug("Publisher: error during disconnect (ignored) — %s", exc)

        self._client = None
        self._connected = False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, payload: dict) -> bool:
        now = time.monotonic()
        if now < self._publish_backoff_until:
            remaining = self._publish_backoff_until - now
            logger.debug(
                "Publisher: publish suppressed — authorization backoff active (%.0f s remaining)",
                remaining,
            )
            return False

        if not self._connected:
            logger.info(
                "Publisher: not connected — attempting reconnect before publish"
            )
            if not self.connect():
                self._failure_count += 1
                return False

        try:
            payload_json = json.dumps(payload)
            payload_bytes = payload_json.encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.error(
                "Publisher: payload serialisation failed — %s", exc, exc_info=True
            )
            self._failure_count += 1
            return False

        logger.debug(
            "Publisher: sending %d bytes to topic %r (msg #%d)",
            len(payload_bytes),
            self._topic,
            self._publish_count + 1,
        )

        try:
            from awsiot.greengrasscoreipc.model import (  # type: ignore[import]
                QOS,
                PublishToIoTCoreRequest,
                UnauthorizedError,
            )

            request = PublishToIoTCoreRequest(
                topic_name=self._topic,
                payload=payload_bytes,
                qos=QOS.AT_LEAST_ONCE,
            )
            operation = self._client.new_publish_to_iot_core()  # type: ignore[union-attr]
            operation.activate(request)
            operation.get_response().result(timeout=_PUBLISH_TIMEOUT_S)

            self._publish_count += 1
            n_detections = len(payload.get("ncoords", []))
            logger.info(
                "Publisher: telemetry published — topic=%s msg#=%d detections=%d bytes=%d "
                "total_failures=%d",
                self._topic,
                self._publish_count,
                n_detections,
                len(payload_bytes),
                self._failure_count,
            )
            return True

        except UnauthorizedError:
            self._failure_count += 1
            logger.error(
                "Publisher: UnauthorizedError publishing to %s (total_failures=%d). "
                "Check ComponentAccessControl in the Greengrass recipe. "
                "Suppressing publish attempts for %.0f s.",
                self._topic,
                self._failure_count,
                _RECONNECT_BACKOFF_MAX_S,
            )
            self._publish_backoff_until = time.monotonic() + _RECONNECT_BACKOFF_MAX_S
            return False

        except Exception as exc:
            self._failure_count += 1
            logger.error(
                "Publisher: publish failed (total_failures=%d) — %s. "
                "Marking connection as dirty for reconnect.",
                self._failure_count,
                exc,
                exc_info=True,
            )
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def device_id(self) -> str:
        return self._device_id
