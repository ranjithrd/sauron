"""
dashboard.py — API routes for the frontend visualizer.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from app.db.queries import get_all_cameras, get_live_tracks

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/cameras")
async def api_get_cameras() -> List[dict]:
    """Return a list of all registered cameras."""
    return await get_all_cameras()


@router.get("/live_tracks")
async def api_get_live_tracks(within_seconds: int = 5) -> List[dict]:
    """Return the most recent smoothed position of each active object."""
    return await get_live_tracks(within_seconds=within_seconds)
