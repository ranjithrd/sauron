"""
mqtt_mock.py — synthetic telemetry publisher for AWS IoT Core.

Publishes payloads to:
    devices/cam_01/telemetry
    devices/cam_02/telemetry

The object moves in a circle between two camera positions and is projected into
camera-frame normalized coordinates.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any

try:
    from awscrt import io, mqtt  # type: ignore[import-not-found]
    from awsiot import mqtt_connection_builder  # type: ignore[import-not-found]
except ImportError:
    io = None
    mqtt = None
    mqtt_connection_builder = None

IOT_ENDPOINT = os.getenv("IOT_ENDPOINT", "")
IOT_CERT_PATH = os.getenv("IOT_CERT_PATH", "certs/certificate.pem.crt")
IOT_KEY_PATH = os.getenv("IOT_KEY_PATH", "certs/private.pem.key")
IOT_CA_PATH = os.getenv("IOT_CA_PATH", "certs/AmazonRootCA1.pem")
IOT_CLIENT_ID = os.getenv("IOT_CLIENT_ID", "iotel-mqtt-mock")

PUBLISH_HZ = 10.0
TICK_S = 1.0 / PUBLISH_HZ

CAMERAS: list[dict[str, Any]] = [
    {
        "device_id": "cam_01",
        "lat": 12.880,
        "lon": 77.550,
        "heading": 45.0,
        "fov": 60.0,
        "base_pitch": -6.0,
        "base_roll": 0.0,
    },
    {
        "device_id": "cam_02",
        "lat": 12.880,
        "lon": 77.552,
        "heading": 315.0,
        "fov": 60.0,
        "base_pitch": -6.5,
        "base_roll": 0.0,
    },
]


def _project_to_camera(obj_lat: float, obj_lon: float, cam: dict[str, Any]) -> tuple[float, float] | None:
    """Project world point into one camera's normalized frame coordinates."""
    c_lat = cam["lat"]
    c_lon = cam["lon"]

    north_m = (obj_lat - c_lat) * 111_320.0
    east_m = (obj_lon - c_lon) * 111_320.0 * math.cos(math.radians(c_lat))

    abs_bearing = math.degrees(math.atan2(east_m, north_m)) % 360.0
    rel = (abs_bearing - cam["heading"] + 540.0) % 360.0 - 180.0

    half_fov = cam["fov"] / 2.0
    if abs(rel) > half_fov:
        return None

    xnorm = (rel + half_fov) / cam["fov"]

    # Keep y in a plausible range and weakly dependent on distance.
    dist_m = math.hypot(east_m, north_m)
    ynorm = min(0.95, max(0.05, 0.45 + (dist_m / 150.0) * 0.1))
    return xnorm, ynorm


def _build_payload(cam: dict[str, Any], object_lat: float, object_lon: float) -> dict[str, Any]:
    projected = _project_to_camera(object_lat, object_lon, cam)
    ncoords = []
    if projected is not None:
        xnorm, ynorm = projected
        ncoords.append({"xnorm": xnorm, "ynorm": ynorm})

    pitch = cam["base_pitch"] + random.gauss(0.0, 0.25)
    roll = cam["base_roll"] + random.gauss(0.0, 0.2)

    return {
        "cameras": {
            "gps": {"lat": cam["lat"], "lon": cam["lon"]},
            "imu": {
                "heading": cam["heading"],
                "pitch": pitch,
                "roll": roll,
            },
            "fov": cam["fov"],
        },
        "ncoords": ncoords,
        "timestamp": time.time(),
    }


def main() -> None:
    if io is None or mqtt is None or mqtt_connection_builder is None:
        raise RuntimeError("awsiotsdk is not installed. Install requirements first.")

    if not IOT_ENDPOINT:
        raise RuntimeError("IOT_ENDPOINT is empty. Set AWS IoT Core endpoint in env vars.")

    event_loop_group = io.EventLoopGroup(1)
    host_resolver = io.DefaultHostResolver(event_loop_group)
    client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

    connection = mqtt_connection_builder.mtls_from_path(
        endpoint=IOT_ENDPOINT,
        cert_filepath=IOT_CERT_PATH,
        pri_key_filepath=IOT_KEY_PATH,
        ca_filepath=IOT_CA_PATH,
        client_bootstrap=client_bootstrap,
        client_id=IOT_CLIENT_ID,
        clean_session=False,
        keep_alive_secs=30,
    )

    print(f"Connecting to AWS IoT Core endpoint {IOT_ENDPOINT} ...")
    connection.connect().result()
    print("Connected. Publishing telemetry at 10Hz...")

    t = 0.0
    center_lat = 12.881
    center_lon = 77.551
    radius_deg = 0.0002

    try:
        while True:
            loop_start = time.time()

            object_lat = center_lat + radius_deg * math.sin(t)
            object_lon = center_lon + radius_deg * math.cos(t)

            for cam in CAMERAS:
                payload = _build_payload(cam, object_lat, object_lon)
                topic = f"devices/{cam['device_id']}/telemetry"
                connection.publish(
                    topic=topic,
                    payload=json.dumps(payload),
                    qos=mqtt.QoS.AT_LEAST_ONCE,
                )

            t += 0.05

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, TICK_S - elapsed))
    except KeyboardInterrupt:
        print("Stopping publisher...")
    finally:
        connection.disconnect().result()


if __name__ == "__main__":
    main()
