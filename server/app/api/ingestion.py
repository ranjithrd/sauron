"""
ingestion.py — POST /ingest/device router.

Registers a camera with the StreamManager and persists it to the DB.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.db.writer import upsert_camera
from app.ingestion.schemas import DevicePayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/device", status_code=200)
async def register_device(payload: DevicePayload, request: Request) -> dict:
    """
    Accept a DevicePayload from an edge device.
    - Registers / updates the RTSP stream in the StreamManager.
    - Upserts the camera metadata into the DB registry.
    """
    stream_manager = request.app.state.stream_manager
    await stream_manager.register_camera(payload)

    await upsert_camera(
        device_id=payload.device_id,
        rtsp_url=payload.rtsp_url,
        lat=payload.camera.lat,
        lon=payload.camera.lon,
        bearing_deg=payload.camera.bearing_deg,
        fov_deg=payload.camera.fov_deg,
    )

    logger.info("Registered device %s", payload.device_id)
    return {"status": "registered", "device_id": payload.device_id}
