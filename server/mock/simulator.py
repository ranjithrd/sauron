"""
simulator.py — synthetic RTSP stream generator.
Pushes video frames to mediamtx via FFmpeg.
Registers the simulated cameras with the FastAPI backend.
"""

import math
import os
import subprocess
import time
from typing import Dict, List

import cv2
import numpy as np
import requests

SERVER_URL = os.getenv("SERVER_URL", "http://host.docker.internal:8000")
RTSP_HOST = os.getenv("RTSP_HOST", "rtsp://mediamtx:8554")

CAMERAS = [
    {
        # Camera 1: placed West of scene, pointing Northeast (bearing=45)
        "device_id": "cam_01",
        "rtsp_url_internal": f"{RTSP_HOST}/cam_01",
        "rtsp_url_external": "rtsp://localhost:8554/cam_01",
        "camera": {
            "lat": 12.880,
            "lon": 77.550,
            "bearing_deg": 45.0,   # North-East
            "fov_deg": 60.0
        }
    },
    {
        # Camera 2: placed East of scene, pointing Northwest (bearing=315)
        "device_id": "cam_02",
        "rtsp_url_internal": f"{RTSP_HOST}/cam_02",
        "rtsp_url_external": "rtsp://localhost:8554/cam_02",
        "camera": {
            "lat": 12.880,
            "lon": 77.552,
            "bearing_deg": 315.0,  # North-West
            "fov_deg": 60.0
        }
    }
]

FPS = 30
WIDTH, HEIGHT = 640, 480

def start_ffmpeg_pipe(rtsp_url: str) -> subprocess.Popen:
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        rtsp_url
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def register_cameras():
    for cam in CAMERAS:
        payload = {
            "device_id": cam["device_id"],
            "timestamp": time.time(),
            "rtsp_url": cam["rtsp_url_external"],
            "camera": cam["camera"]
        }
        try:
            res = requests.post(f"{SERVER_URL}/ingest/device", json=payload, timeout=5)
            print(f"Registered {cam['device_id']}: {res.status_code}")
        except Exception as e:
            print(f"Failed to register {cam['device_id']}: {e}")

def project_to_camera(obj_lat: float, obj_lon: float, cam: dict) -> float | None:
    """Project object lat/lon into camera x_norm. Returns None if outside FOV."""
    c_lat = cam["camera"]["lat"]
    c_lon = cam["camera"]["lon"]
    c_bearing = cam["camera"]["bearing_deg"]
    c_fov = cam["camera"]["fov_deg"]

    # Flat earth approx (simplified for simulator)
    lat_m = (obj_lat - c_lat) * 111320.0
    lon_m = (obj_lon - c_lon) * 111320.0 * math.cos(math.radians(c_lat))

    # Angle from camera to object (North=0, East=90)
    angle_rad = math.atan2(lon_m, lat_m)
    angle_deg = math.degrees(angle_rad) % 360

    # Relative angle
    rel_angle = (angle_deg - c_bearing + 360) % 360
    if rel_angle > 180:
        rel_angle -= 360

    if abs(rel_angle) > c_fov / 2:
        return None # outside FOV
        
    x_norm = (rel_angle + c_fov / 2) / c_fov
    return x_norm

def main():
    print("Waiting for servers to start...")
    time.sleep(5)
    
    # Keep registering until success (wait for FastAPI to boot)
    while True:
        try:
            requests.get(f"{SERVER_URL}/health", timeout=2)
            break
        except:
            time.sleep(2)
            print("Waiting for FastAPI...")
            
    register_cameras()

    pipes = {cam["device_id"]: start_ffmpeg_pipe(cam["rtsp_url_internal"]) for cam in CAMERAS}
    
    t = 0.0
    # Object moves in a small circle centred between the two cameras
    # cam_01 is at (12.880, 77.550), cam_02 is at (12.880, 77.552)
    # Midpoint is (12.881, 77.551) — verified visible from both cameras
    center_lat = 12.881
    center_lon = 77.551
    radius_deg = 0.0002  # ~22 meters radius

    print("Starting simulation loop...")
    while True:
        loop_start = time.time()
        
        # Object position
        obj_lat = center_lat + radius_deg * math.sin(t)
        obj_lon = center_lon + radius_deg * math.cos(t)
        
        for cam in CAMERAS:
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            x_norm = project_to_camera(obj_lat, obj_lon, cam)

            if x_norm is not None:
                x_px = int(x_norm * WIDTH)
                y_px = HEIGHT // 2
                cv2.circle(frame, (x_px, y_px), 20, (255, 255, 255), -1)
                if int(t * 100) % 30 == 0:  # log every ~3 seconds
                    print(f"  [{cam['device_id']}] visible: x_norm={x_norm:.3f}  x_px={x_px}")
            else:
                if int(t * 100) % 30 == 0:
                    print(f"  [{cam['device_id']}] NOT visible (outside FOV)")
                
            pipe = pipes[cam["device_id"]]
            pipe.stdin.write(frame.tobytes())
            
        t += 0.01
        
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, (1.0 / FPS) - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
