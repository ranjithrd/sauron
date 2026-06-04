# Sauron Edge

AWS Greengrass v2 component that:
1. Captures video from a Raspberry Pi camera (`raspi`) or RTSP stream (`rtsp`)
2. Runs MOG2 background subtraction to find moving objects
3. Reads GPS (UART/NMEA) and IMU (BNO055 over I2C) sensors
4. Publishes `TelemetryPayload` messages to `devices/{thing_name}/telemetry` via Greengrass IPC

Python version is guaranteed to be **3.11.2**. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

---

## Repository layout

```
edge/
├── src/
│   └── sauron_edge/
│       ├── __init__.py
│       ├── main.py        # Entrypoint — orchestration loop, signal handling, stats
│       ├── config.py      # Config file management (mirrors Go agent pattern)
│       ├── schemas.py     # TelemetryPayload model (kept in sync with server)
│       ├── camera.py      # Camera source abstraction (raspi / rtsp)
│       ├── detection.py   # MOG2 background subtractor
│       ├── sensors.py     # GPS + IMU reader thread
│       └── publisher.py   # Greengrass IPC publisher
├── pyproject.toml
├── config.example.yaml    # Copy to ~/.config/sauron/config.yaml
├── .env.example           # Copy to .env in working directory
├── recipe.example.yaml    # Greengrass component recipe template
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11.2 | Pinned. Greengrass platform guarantees this. |
| [uv](https://astral.sh/uv) | Package manager. Install once per device. |
| AWS Greengrass v2 nucleus ≥ 2.9.0 | Provides IPC socket for IoT Core publishing. |
| Raspberry Pi camera OR RTSP stream | Configured via `camera_source` in config. |
| GPS module on UART (`/dev/serial0`) | Optional — zeros published if absent. |
| BNO055 IMU on I2C | Optional — zeros published if absent. |

---

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart your shell
```

### 2. Clone / deploy the edge directory onto the device

```bash
# From the sauron repo root:
scp -r edge/ pi@<device-ip>:~/sauron-edge/
```

### 3. Install dependencies

```bash
cd ~/sauron-edge
uv sync --python 3.11.2
```

This creates a `.venv` inside the directory and installs all pinned dependencies.

### 4. Create your config file

```bash
mkdir -p ~/.config/sauron
cp config.example.yaml ~/.config/sauron/config.yaml
nano ~/.config/sauron/config.yaml   # edit as needed
```

Key values to set:

| Field | Default | Description |
|---|---|---|
| `camera_source` | `raspi` | `raspi` for Pi camera, `rtsp` for network stream |
| `camera_rtsp_url` | *(empty)* | Required when `camera_source: rtsp` |
| `camera_dimensions` | `640x480` | Frame resolution (WxH) |
| `camera_fov` | `90.0` | Horizontal FOV of your lens in degrees |
| `publish_rate_hz` | `5.0` | Telemetry messages per second |
| `detection.min_contour_area` | `200.0` | Tune up to reduce false positives |
| `sensors.imu_enabled` | `true` | Set `false` if BNO055 is not fitted |

### 5. Create your .env file (local testing only)

```bash
cp .env.example .env
# Edit .env to set AWS_IOT_THING_NAME if running outside Greengrass
```

### 6. Run locally (outside Greengrass)

```bash
# Inside the edge/ directory:
uv run sauron-edge
```

> **Note:** Without a live Greengrass nucleus, IPC connection will fail and messages will be dropped. The rest of the pipeline (camera, detection, sensors) runs normally. Use this mode to verify camera and sensor connectivity before deploying as a component.

---

## Deploying as a Greengrass component

### 1. Package the component

```bash
cd ~/sauron-edge
zip -r sauron-edge.zip . -x ".venv/*" -x "*.pyc" -x "__pycache__/*"
```

### 2. Upload to S3

```bash
aws s3 cp sauron-edge.zip s3://YOUR_BUCKET/sauron-edge/sauron-edge.zip
```

### 3. Create the recipe

Copy `recipe.example.yaml`, replace `REPLACE_WITH_YOUR_BUCKET`, then register with Greengrass:

```bash
aws greengrassv2 create-component-version \
  --inline-recipe fileb://recipe.example.yaml
```

### 4. Deploy to device

Use the AWS Greengrass console or CLI to deploy `com.sauron.edge` to your device.  
The nucleus will inject `AWS_IOT_THING_NAME` and the IPC socket path automatically.

### 5. Required IAM / IoT Core policy

The Greengrass device role must allow:

```json
{
  "Effect": "Allow",
  "Action": "iot:Publish",
  "Resource": "arn:aws:iot:REGION:ACCOUNT:topic/devices/*/telemetry"
}
```

The component recipe (`recipe.example.yaml`) already includes the IPC access control policy that allows `PublishToIoTCore` on `devices/*/telemetry`.

---

## Configuration reference

See [config.example.yaml](config.example.yaml) for the full annotated reference.

The config file path is resolved in this order:
1. `CONFIGURATION_FILE_PATH` environment variable
2. `~/.config/sauron/config.yaml` (default)

If the file doesn't exist, the component creates it with defaults.

---

## Telemetry message format

Published as JSON to `devices/{thing_name}/telemetry`. Schema matches `server/app/ingestion/schemas.py`:

```json
{
  "cameras": {
    "gps":  { "lat": 1.234567, "lon": 103.987654 },
    "imu":  { "heading": 270.0, "pitch": -2.1, "roll": 0.5 },
    "fov":  90.0
  },
  "ncoords": [
    { "xnorm": 0.412, "ynorm": 0.651 },
    { "xnorm": 0.789, "ynorm": 0.302 }
  ],
  "timestamp": 1748908800.123
}
```

`ncoords` contains one entry per detected foreground blob. Coordinates are normalised to `[0.0, 1.0]` (origin = top-left).

---

## Logging

All output goes to stdout at `DEBUG` level. Greengrass routes stdout to the component log file at:

```
/greengrass/v2/logs/com.sauron.edge.log
```

To tail logs on the device:

```bash
tail -f /greengrass/v2/logs/com.sauron.edge.log
```

Key log signatures:

| Pattern | Meaning |
|---|---|
| `MOG2Detector: N detection(s)` | Objects detected in frame |
| `Telemetry publish #N` | Full publish attempt with sensor readings |
| `Publisher: telemetry published` | Successful IoT Core delivery |
| `Publisher: publish failed` | IPC error — will auto-retry next interval |
| `SensorReader: heartbeat` | Sensor thread alive (every 200 loops ≈ 10 s) |
| `RaspiCamera: camera opened` | Camera init succeeded |
| `Camera capture thread is no longer alive` | Hardware failure — component will exit |

---

## Tuning MOG2 detection

| Symptom | Adjustment |
|---|---|
| Too many false positives (noise, shadows) | Increase `min_contour_area` or `var_threshold` |
| Slow/distant objects not detected | Decrease `min_contour_area` or `var_threshold` |
| Many noise blobs in windy conditions | Increase `morph_kernel_size` |
| Background model adapts too slowly to lighting | Increase `learning_rate` toward `0.01` |
| Background model forgets static background too fast | Decrease `learning_rate` toward `0.001` |

---

## Troubleshooting

**Camera fails to open**
```
RaspiCamera: failed to open /dev/video0
```
Check that the camera is enabled (`raspi-config → Interface Options → Camera`) and the user has `/dev/video0` access: `sudo usermod -aG video $USER`.

**GPS shows zeros**
Normal until the module acquires a satellite fix. Can take 30-90 seconds outdoors. Check wiring if it never locks — verify with `cat /dev/serial0`.

**IMU shows zeros / not calibrated**
BNO055 needs a calibration period after power-on. Slowly rotate the device through several orientations. `imu_calibration_status` in the logs shows progress toward `Sys:3 G:3 A:3 M:3`.

**IPC connection refused outside Greengrass**
Expected. The `AWS_GG_NUCLEUS_DOMAIN_SOCKET_FILEPATH_FOR_COMPONENT` socket only exists when the Greengrass nucleus is running and the component was launched by it.

**`adafruit_bno055` import error**
The adafruit library requires the `libgpiod` system library:
```bash
sudo apt-get install -y libgpiod2 python3-libgpiod
```
