from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List

from pydantic import ValidationError

from app.config import settings
from app.db.writer import upsert_camera
from app.ingestion.schemas import TelemetryPayload

try:
    from awscrt import io, mqtt
    from awsiot import mqtt_connection_builder
except ImportError:  # pragma: no cover - exercised only when dependency missing
    io = None
    mqtt = None
    mqtt_connection_builder = None

logger = logging.getLogger(__name__)

_TOPIC_FILTER = "devices/+/telemetry"
_RECONNECT_DELAY_S = 5.0


@dataclass
class Detection:
    device_id: str
    object_id: str        # generated from payload position, not tracked across frames
    xnorm: float
    ynorm: float
    timestamp: float
    camera_lat: float
    camera_lon: float
    camera_heading: float
    camera_pitch: float
    camera_roll: float
    camera_fov: float


AsyncCallback = Callable[[Detection], Awaitable[None]]


class MQTTClient:
    """Subscribe to AWS IoT telemetry and emit one Detection per ncoord."""

    def __init__(self) -> None:
        self._callbacks: List[AsyncCallback] = []
        self._stop_event = asyncio.Event()
        self._connection_lost = asyncio.Event()
        self._run_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connection: Any | None = None
        self._bootstrap_resources: tuple[Any, Any, Any] | None = None

    def on_detection(self, callback: AsyncCallback) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the MQTT client loop as a background task."""
        if self._run_task and not self._run_task.done():
            return
        self._stop_event.clear()
        self._run_task = asyncio.create_task(self.run(), name="mqtt-client-loop")

    async def run(self) -> None:
        """Run forever until stop() is called, reconnecting on all failures."""
        if io is None or mqtt is None or mqtt_connection_builder is None:
            raise RuntimeError(
                "awsiotsdk is not installed. Install dependencies from requirements.txt."
            )

        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()

        while not self._stop_event.is_set():
            try:
                await self._connect_and_subscribe()
                logger.info("MQTTClient: subscribed to %s", _TOPIC_FILTER)

                while not self._stop_event.is_set() and not self._connection_lost.is_set():
                    await asyncio.sleep(0.5)

                if self._connection_lost.is_set() and not self._stop_event.is_set():
                    logger.warning("MQTTClient: connection interrupted, reconnecting")
            except ValueError as exc:
                logger.error("MQTTClient: configuration error: %s", exc)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("MQTTClient: run loop error: %s", exc, exc_info=exc)
            finally:
                await self._disconnect_current()

            if not self._stop_event.is_set():
                await asyncio.sleep(_RECONNECT_DELAY_S)

    async def stop(self) -> None:
        """Request shutdown and close the active MQTT connection."""
        self._stop_event.set()

        if self._run_task and not self._run_task.done():
            await self._run_task

        await self._disconnect_current()

    async def _connect_and_subscribe(self) -> None:
        endpoint = self._normalize_iot_endpoint(settings.IOT_ENDPOINT)
        self._validate_iot_endpoint(endpoint)

        self._connection = self._build_connection(endpoint)
        self._connection_lost.clear()

        connect_future = self._connection.connect()
        await asyncio.to_thread(connect_future.result)

        subscribe_result = self._connection.subscribe(
            topic=_TOPIC_FILTER,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=self._on_message,
        )
        subscribe_future = (
            subscribe_result[0] if isinstance(subscribe_result, tuple) else subscribe_result
        )
        await asyncio.to_thread(subscribe_future.result)

    async def _disconnect_current(self) -> None:
        connection = self._connection
        self._connection = None

        if connection is None:
            return

        try:
            disconnect_future = connection.disconnect()
            await asyncio.to_thread(disconnect_future.result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("MQTTClient: disconnect skipped/failed: %s", exc)

    @staticmethod
    def _normalize_iot_endpoint(raw_endpoint: str) -> str:
        endpoint = raw_endpoint.strip()
        for prefix in ("mqtts://", "https://", "mqtt://"):
            if endpoint.startswith(prefix):
                endpoint = endpoint[len(prefix):]
        endpoint = endpoint.split("/", 1)[0]
        return endpoint

    @staticmethod
    def _validate_iot_endpoint(endpoint: str) -> None:
        if not endpoint:
            raise ValueError("IOT_ENDPOINT is empty. Configure it in .env.")
        if "xxxxxx" in endpoint.lower():
            raise ValueError(
                "IOT_ENDPOINT still contains placeholder value 'xxxxxx'. "
                "Set it to your real AWS IoT endpoint, e.g. '<id>-ats.iot.<region>.amazonaws.com'."
            )
        if " " in endpoint or "." not in endpoint:
            raise ValueError(
                "IOT_ENDPOINT appears invalid. Expected a host name like "
                "'<id>-ats.iot.<region>.amazonaws.com' (without protocol or path)."
            )

    def _build_connection(self, endpoint: str) -> Any:
        # Keep references alive for the life of the active connection.
        event_loop_group = io.EventLoopGroup(1)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        self._bootstrap_resources = (event_loop_group, host_resolver, client_bootstrap)

        return mqtt_connection_builder.mtls_from_path(
            endpoint=endpoint,
            cert_filepath=settings.IOT_CERT_PATH,
            pri_key_filepath=settings.IOT_KEY_PATH,
            ca_filepath=settings.IOT_CA_PATH,
            client_bootstrap=client_bootstrap,
            client_id=settings.IOT_CLIENT_ID,
            clean_session=False,
            keep_alive_secs=30,
            on_connection_interrupted=self._on_connection_interrupted,
            on_connection_resumed=self._on_connection_resumed,
        )

    def _on_connection_interrupted(self, connection: Any, error: Exception, **_: Any) -> None:
        logger.warning("MQTTClient: connection interrupted: %s", error)
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._connection_lost.set)

    def _on_connection_resumed(
        self,
        connection: Any,
        return_code: Any,
        session_present: bool,
        **_: Any,
    ) -> None:
        logger.info(
            "MQTTClient: connection resumed (session_present=%s, return_code=%s)",
            session_present,
            return_code,
        )
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._connection_lost.clear)

    def _on_message(
        self,
        topic: str,
        payload: bytes,
        dup: bool,
        qos: Any,
        retain: bool,
        **_: Any,
    ) -> None:
        if self._loop is None or self._loop.is_closed():
            return

        def _schedule() -> None:
            asyncio.create_task(
                self._handle_message(topic=topic, payload=payload),
                name="mqtt-message-handler",
            )

        self._loop.call_soon_threadsafe(_schedule)

    async def _handle_message(self, topic: str, payload: bytes) -> None:
        try:
            device_id = topic.split("/")[1]
        except IndexError:
            logger.warning("MQTTClient: malformed topic '%s'", topic)
            return

        try:
            payload_model = TelemetryPayload.model_validate_json(payload.decode("utf-8"))
        except (ValidationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("MQTTClient: invalid telemetry payload from %s: %s", device_id, exc)
            return

        try:
            await upsert_camera(
                device_id=device_id,
                lat=payload_model.cameras.gps.lat,
                lon=payload_model.cameras.gps.lon,
                heading=payload_model.cameras.imu.heading,
                pitch=payload_model.cameras.imu.pitch,
                roll=payload_model.cameras.imu.roll,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "MQTTClient: failed to upsert camera metadata for %s: %s",
                device_id,
                exc,
                exc_info=exc,
            )

        for coord in payload_model.ncoords:
            detection = Detection(
                device_id=device_id,
                object_id=f"{device_id}_obj_{coord.xnorm:.3f}_{coord.ynorm:.3f}",
                xnorm=coord.xnorm,
                ynorm=coord.ynorm,
                timestamp=payload_model.timestamp,
                camera_lat=payload_model.cameras.gps.lat,
                camera_lon=payload_model.cameras.gps.lon,
                camera_heading=payload_model.cameras.imu.heading,
                camera_pitch=payload_model.cameras.imu.pitch,
                camera_roll=payload_model.cameras.imu.roll,
                camera_fov=payload_model.cameras.fov,
            )
            await self._emit_detection(detection)

    async def _emit_detection(self, detection: Detection) -> None:
        for callback in self._callbacks:
            try:
                await callback(detection)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "MQTTClient: detection callback %s raised: %s",
                    callback,
                    exc,
                    exc_info=exc,
                )
