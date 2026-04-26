# End-to-End Visual Trial & Bug Fixes

This plan outlines the steps to build a real-time web dashboard for your tracking server, construct a synthetic RTSP simulator to test the pipeline without edge devices, and fix the two critical bugs identified during the codebase review.

## User Review Required

> [!IMPORTANT]
> **Docker requirement for Simulator**
> The synthetic simulator will require `ffmpeg` to push generated frames to the RTSP server. To avoid asking you to install FFmpeg locally on Windows, I propose adding the simulator as a Docker service inside your `docker-compose.yml`. It will run alongside TimescaleDB and MediaMTX.

## Proposed Changes

### 1. Fix Critical Bugs in `StreamManager`

The current video stream manager has two major flaws: it never gets started, and it incorrectly opens/closes the RTSP connection for every single frame.

#### [MODIFY] `app/main.py`
- Update the `lifespan` to properly run the StreamManager in the background using `asyncio.create_task(stream_manager.run())`.

#### [MODIFY] `app/video/stream_manager.py`
- Rewrite `_camera_loop` to use a dedicated, persistent background thread for each camera.
- The thread will keep `cv2.VideoCapture` open continuously, read frames, and push them to an `asyncio.Queue`.
- The async loop will consume frames from the queue and run the blob detector, solving the RTSP performance bottleneck.

---

### 2. Infrastructure & Simulation Environment

We need a way to mock edge cameras without hardware.

#### [MODIFY] `docker-compose.yml`
- Add `mediamtx`, a lightweight RTSP server to host our local streams.
- Add `simulator`, a custom Python service that builds an image with `ffmpeg` installed.

#### [NEW] `mock/simulator.py`
- A script that runs inside the `simulator` docker container.
- **Math engine**: Simulates an object moving in a circle in the real world (lat/lon). It mathematically projects this object's position onto the field of view of two mock cameras.
- **Video Generator**: Uses OpenCV to draw a white dot on a black frame corresponding to the object's position in each camera's FOV.
- **Streaming**: Pipes the raw frames into an `ffmpeg` subprocess to stream to `rtsp://mediamtx:8554/cam1` and `cam2`.
- **Registration**: Automatically sends `POST /ingest/device` to the FastAPI server to register the simulated cameras.

#### [NEW] `mock/Dockerfile`
- Simple Dockerfile using `python:3.11-slim` with `ffmpeg` and `opencv-python-headless` installed for the simulator service.

---

### 3. API Layer for the Dashboard

The dashboard needs endpoints to fetch the live tracks and camera metadata.

#### [NEW] `app/api/dashboard.py`
- Define two GET routes:
  - `GET /api/live_tracks`: calls `get_live_tracks()`
  - `GET /api/cameras`: calls `get_all_cameras()`
- Expose the static files.

#### [MODIFY] `app/main.py`
- Include the new `dashboard` router.
- Mount a `StaticFiles` directory so the server can host the HTML/JS dashboard natively.

---

### 4. The Visual Dashboard

A single-page web app to visualize the system in real-time.

#### [NEW] `app/static/index.html`
- A full-screen Map container (using Leaflet.js with a dark-mode tile layer).

#### [NEW] `app/static/app.js`
- **Initial Load**: Fetches `/api/cameras`. Plots camera markers and draws semi-transparent polygons to represent their Field of View (bearing and FOV angle).
- **Polling Loop**: Fetches `/api/live_tracks` every 500ms.
- **Visualization**: 
  - Plots moving dots on the map for each tracked `object_id`.
  - Draws velocity vectors (small lines extending from the dot based on `vel_lat` and `vel_lon`) to visualize the Kalman filter's predictions.

## Verification Plan

1. Run `docker-compose up -d --build`. This starts TimescaleDB, MediaMTX, and the Simulator.
2. The Simulator will immediately begin streaming synthetic video and will automatically register `cam1` and `cam2` with the FastAPI server.
3. Start the FastAPI server locally (`uvicorn app.main:app`).
4. Open `http://localhost:8000/static/index.html` in your browser.
5. **Expected Result**: You should see the map, the two cameras with their overlapping FOV cones, and a tracked object dot moving smoothly in a circle through the intersection zone, verifying the entire end-to-end pipeline (Blob Detection → Correlation → Triangulation → Kalman Filter → Database).
