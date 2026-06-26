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
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_THING_NAME_ENV = "AWS_IOT_THING_NAME"
_PUBLISH_TIMEOUT_S = 10.0

# Reconnect backoff: starts at 5 s, doubles on each consecutive failure,
# caps at 60 s.  Resets to the initial value on a successful connection.
_RECONNECT_BACKOFF_INITIAL_S = 5.0
_RECONNECT_BACKOFF_MAX_S = 60.0

_COMMANDS_TOPIC = "devices/all/commands"


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

        # Reconnect backoff state — prevents hammering the nucleus with thread
        # creation requests when IPC is unavailable (AWS_ERROR_THREAD_INSUFFICIENT_RESOURCE).
        self._reconnect_backoff_s: float = _RECONNECT_BACKOFF_INITIAL_S
        self._next_reconnect_at: float = (
            0.0  # epoch seconds; 0 = may connect immediately
        )

        # Publish-level backoff — set when an UnauthorizedError is received so
        # that we stop hammering the nucleus with doomed publish requests while
        # the IPC socket itself remains open.
        self._publish_backoff_until: float = 0.0

        # Telemetry toggle — can be flipped by the command listener thread
        self.telemetry_enabled: bool = True

        self._cmd_thread: Optional[threading.Thread] = None
        self._cmd_stop: threading.Event = threading.Event()

        logger.info(
            "Publisher: initialised — device_id=%s topic=%s",
            self._device_id,
            self._topic,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish (or re-establish) the Greengrass IPC connection.

        Respects exponential backoff — returns False immediately (without
        attempting a connection) if the backoff window has not elapsed.
        This prevents hammering the nucleus with EventLoopGroup allocations
        when resources are scarce (AWS_ERROR_THREAD_INSUFFICIENT_RESOURCE).

        Returns True on success, False on failure or while backing off.
        """
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

        logger.info(
            "Publisher: connecting to Greengrass nucleus IPC... (backoff=%.1fs)",
            self._reconnect_backoff_s,
        )
        try:
            import awsiot.greengrasscoreipc as gg_ipc  # type: ignore[import]

            self._client = gg_ipc.connect()
            self._connected = True
            # Reset backoff on successful connection
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
            # Don't backoff for import errors — it's a permanent config issue
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
            # Advance backoff (exponential, capped)
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
        # If a previous UnauthorizedError put us in a publish backoff, honour it.
        now = time.monotonic()
        if now < self._publish_backoff_until:
            remaining = self._publish_backoff_until - now
            logger.debug(
                "Publisher: publish suppressed — authorization backoff active (%.0f s remaining)",
                remaining,
            )
            return False

        # Ensure we have a live connection
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
            # Authorization is enforced by the Greengrass nucleus IPC policy
            # (ComponentAccessControl in the recipe) and the IoT Core thing
            # policy (iot:Publish on the topic ARN).  Reconnecting will not
            # fix this — back off and log a clear action item instead.
            self._failure_count += 1
            logger.error(
                "Publisher: UnauthorizedError publishing to %s (total_failures=%d). "
                "This will not resolve itself automatically. "
                "Check: (1) ComponentAccessControl in the Greengrass recipe grants "
                "aws.greengrass#PublishToIoTCore on 'devices/*/telemetry', "
                "(2) the Greengrass thing's IoT Core policy allows iot:Publish on "
                "arn:aws:iot:REGION:ACCOUNT:topic/devices/*/telemetry. "
                "Suppressing publish attempts for %.0f s.",
                self._topic,
                self._failure_count,
                _RECONNECT_BACKOFF_MAX_S,
            )
            # Do NOT mark connection dirty — the IPC socket itself is fine.
            # Gate all further publish calls until the backoff window passes.
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
            # Mark connection dirty — will reconnect on next publish()
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Command listener (telemetry toggle)
    # ------------------------------------------------------------------

    def start_command_listener(self) -> None:
        """Start a daemon thread that subscribes to the global commands topic."""
        if self._cmd_thread and self._cmd_thread.is_alive():
            return
        self._cmd_stop.clear()
        self._cmd_thread = threading.Thread(
            target=self._command_listener_loop,
            name="cmd-listener",
            daemon=True,
        )
        self._cmd_thread.start()
        logger.info("Publisher: command listener thread started (topic=%s)", _COMMANDS_TOPIC)

    def stop_command_listener(self) -> None:
        self._cmd_stop.set()
        if self._cmd_thread:
            self._cmd_thread.join(timeout=3.0)

    def _command_listener_loop(self) -> None:
        while not self._cmd_stop.is_set():
            try:
                import awsiot.greengrasscoreipc as gg_ipc  # type: ignore[import]
                from awsiot.greengrasscoreipc.model import (  # type: ignore[import]
                    QOS,
                    SubscribeToIoTCoreRequest,
                )

                client = gg_ipc.connect()
                request = SubscribeToIoTCoreRequest(
                    topic_name=_COMMANDS_TOPIC,
                    qos=QOS.AT_LEAST_ONCE,
                )
                operation = client.new_subscribe_to_iot_core()
                operation.activate(request)

                while not self._cmd_stop.is_set():
                    event = operation.get_response().result(timeout=5.0)
                    if event and event.message and event.message.payload:
                        try:
                            msg = json.loads(event.message.payload.decode("utf-8"))
                            if msg.get("action") == "set_telemetry":
                                enabled = bool(msg.get("enabled", True))
                                self.telemetry_enabled = enabled
                                logger.info(
                                    "Publisher: telemetry %s via command",
                                    "ENABLED" if enabled else "DISABLED",
                                )
                        except (json.JSONDecodeError, AttributeError) as exc:
                            logger.warning("Publisher: malformed command message: %s", exc)

            except ImportError:
                logger.warning("Publisher: awsiotsdk not available — command listener disabled")
                break
            except Exception as exc:
                if not self._cmd_stop.is_set():
                    logger.warning(
                        "Publisher: command listener error — %s. Retrying in 10s.", exc
                    )
                    self._cmd_stop.wait(timeout=10.0)

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
