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
from typing import Optional

logger = logging.getLogger(__name__)

_THING_NAME_ENV = "AWS_IOT_THING_NAME"
_PUBLISH_TIMEOUT_S = 10.0


def resolve_device_id() -> str:
    """
    Resolve this device's identifier.

    Greengrass v2 automatically sets AWS_IOT_THING_NAME for every component
    it launches.  If that variable is absent (local dev/test), we fall back
    to the machine hostname.
    """
    thing_name = os.environ.get(_THING_NAME_ENV, "").strip()
    if thing_name:
        logger.info("Publisher: device_id = %r (from %s env var)", thing_name, _THING_NAME_ENV)
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

        logger.info(
            "Publisher: initialised — device_id=%s topic=%s",
            self._device_id, self._topic,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish (or re-establish) the Greengrass IPC connection.
        Returns True on success, False on failure.
        """
        if self._connected and self._client is not None:
            logger.debug("Publisher: already connected — skipping connect()")
            return True

        logger.info("Publisher: connecting to Greengrass nucleus IPC...")
        try:
            import awsiot.greengrasscoreipc as gg_ipc  # type: ignore[import]

            self._client = gg_ipc.connect()
            self._connected = True
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
                "Messages will be dropped until reconnection succeeds.",
                exc,
                exc_info=True,
            )
            self._client = None
            self._connected = False
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
        """
        Serialise `payload` as JSON and publish it to the telemetry topic
        via Greengrass IPC.

        Automatically attempts to reconnect if not currently connected.

        Args:
            payload: Dict that will be JSON-serialised.  Must be compatible
                     with json.dumps().

        Returns:
            True if the publish succeeded, False otherwise.
        """
        # Ensure we have a live connection
        if not self._connected:
            logger.info("Publisher: not connected — attempting reconnect before publish")
            if not self.connect():
                logger.error(
                    "Publisher: reconnect failed — dropping telemetry message #%d",
                    self._publish_count + 1,
                )
                self._failure_count += 1
                return False

        try:
            payload_json = json.dumps(payload)
            payload_bytes = payload_json.encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.error("Publisher: payload serialisation failed — %s", exc, exc_info=True)
            self._failure_count += 1
            return False

        logger.debug(
            "Publisher: sending %d bytes to topic %r (msg #%d)",
            len(payload_bytes), self._topic, self._publish_count + 1,
        )

        try:
            from awsiot.greengrasscoreipc.model import (  # type: ignore[import]
                PublishToIoTCoreRequest,
                QOS,
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

        except Exception as exc:
            self._failure_count += 1
            logger.error(
                "Publisher: publish failed (total_failures=%d) — %s. "
                "Marking connection as dirty for reconnect.",
                self._failure_count, exc, exc_info=True,
            )
            # Mark connection dirty — will reconnect on next publish()
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
