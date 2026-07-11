import time
import threading
import serial
import pynmea2
import board
import busio
import adafruit_bno055
import cv2
from flask import Flask, Response, jsonify

app = Flask(__name__)

# --- GLOBAL SENSOR STATE ---
sensor_data = {
    "heading": 0.0, "roll": 0.0, "pitch": 0.0,
    "lat": 0.0, "lon": 0.0, "sats": 0,
    "calibration": "Sys:0 G:0 A:0 M:0",
    "gps_status": "Waiting for fix..."
}

# --- 1. SENSOR BACKGROUND THREAD ---
def read_sensors():
    print("[INIT] Starting Sensor Thread...")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_bno055.BNO055_I2C(i2c)
        imu_active = True
    except Exception as e:
        print(f"[ERROR] IMU Failed: {e}")
        imu_active = False

    try:
        ser = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
        gps_active = True
    except Exception as e:
        print(f"[ERROR] GPS Failed: {e}")
        gps_active = False

    while True:
        if imu_active:
            try:
                euler = sensor.euler
                sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
                sensor_data["calibration"] = f"Sys:{sys_cal} G:{gyro_cal} A:{accel_cal} M:{mag_cal}"
                if euler.count(None) == 0:
                    sensor_data["heading"] = round(euler[0] % 360, 2)
                    sensor_data["roll"] = round(euler[1], 2)
                    sensor_data["pitch"] = round(euler[2], 2)
            except:
                pass

        if gps_active and ser.in_waiting > 0:
            try:
                newdata = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # --- NEW: Mirror ALL raw NMEA data to a file for debugging ---
                if newdata:
                    with open("/tmp/gps_debug.log", "a") as f:
                        f.write(newdata + "\n")
                # -------------------------------------------------------------

                if newdata.startswith('$GPGGA'):
                    msg = pynmea2.parse(newdata)
                    if msg.is_valid:
                        sensor_data["lat"] = round(msg.latitude, 6)
                        sensor_data["lon"] = round(msg.longitude, 6)
                        sensor_data["sats"] = int(msg.num_sats)
                        sensor_data["gps_status"] = "Locked"
                    else:
                        sensor_data["gps_status"] = "Searching..."
            except:
                pass
        time.sleep(0.05)

threading.Thread(target=read_sensors, daemon=True).start()

# --- 2. DEDICATED CAMERA THREAD ---
class CameraStream:
    def __init__(self):
        print("[INIT] Starting Camera Thread...")
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # Force the buffer to hold only 1 frame to kill latency
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 24)

        if not self.camera.isOpened():
            print("[ERROR] Camera not found.")

        self.grabbed, self.frame = self.camera.read()
        self.stopped = False
        # Boot the background reader
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        # This loop does NOTHING but read the camera instantly. No processing.
        while not self.stopped:
            self.grabbed, self.frame = self.camera.read()

    def read(self):
        return self.frame

# Initialize the global camera stream
cam_stream = CameraStream()

# --- 3. THE FLASK STREAMER ---
def generate_frames():
    # Pre-allocate encode parameters to save CPU cycles
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

    while True:
        frame = cam_stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # --- NEW: Flip the frame 180 degrees to fix the upside-down physical mount ---
        frame = cv2.flip(frame, -1)
        # -----------------------------------------------------------------------------

        # Encode frame at 70% quality (visually identical, massively faster)
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Yield control slightly so we don't spam the network and choke the Pi
        time.sleep(0.03)

# --- WEB ROUTES ---
@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>Pi Telemetry Deck</title>
            <style>
                body { background-color: #111; color: #0f0; font-family: monospace; text-align: center; margin-top: 20px;}
                img { border: 2px solid #0f0; border-radius: 5px; width: 100%; max-width: 1280px; }
                .data-box { border: 1px solid #333; padding: 15px; margin: 10px auto; width: 600px; text-align: left; background: #000; }
                h1 { color: #fff; }
            </style>
        </head>
        <body>
            <h1>Pi Telemetry Deck</h1>
            <img src="/video_feed" />
            <div class="data-box">
                <h3>IMU Dynamics</h3>
                <p>Heading (Yaw): <span id="heading">0</span>&deg;</p>
                <p>Roll: <span id="roll">0</span>&deg; | Pitch: <span id="pitch">0</span>&deg;</p>
                <p>Calibration: <span id="cal">Unknown</span></p>
            </div>
            <div class="data-box">
                <h3>GPS Status: <span id="status">Waiting...</span></h3>
                <p>Latitude: <span id="lat">0</span></p>
                <p>Longitude: <span id="lon">0</span></p>
                <p>Satellites: <span id="sats">0</span></p>
            </div>

            <script>
                setInterval(() => {
                    fetch('/sensor_data')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('heading').innerText = data.heading;
                            document.getElementById('roll').innerText = data.roll;
                            document.getElementById('pitch').innerText = data.pitch;
                            document.getElementById('cal').innerText = data.calibration;

                            document.getElementById('status').innerText = data.gps_status;
                            document.getElementById('lat').innerText = data.lat;
                            document.getElementById('lon').innerText = data.lon;
                            document.getElementById('sats').innerText = data.sats;
                        });
                }, 500);
            </script>
        </body>
    </html>
    """
    return html

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensor_data')
def get_sensor_data():
    return jsonify(sensor_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    
