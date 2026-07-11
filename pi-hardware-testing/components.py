import time
import serial
import pynmea2
import board
import busio
import adafruit_bno055

print("Initializing Sensors...")

# --- SETUP GPS (UART) ---
gps_port = "/dev/serial0"
gps_baud = 9600
try:
    ser = serial.Serial(gps_port, baudrate=gps_baud, timeout=1)
    gps_active = True
    print("[OK] GPS Serial Port Opened")
except Exception as e:
    print(f"[FAIL] GPS Error: {e}")
    gps_active = False

# --- SETUP IMU (I2C) ---
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    imu_active = True
    print("[OK] IMU I2C Connection Established")
except Exception as e:
    print(f"[FAIL] IMU Error: {e}")
    imu_active = False

print("\n--- Starting Data Stream (Press Ctrl+C to stop) ---\n")
time.sleep(1)

while True:
    try:
        output_string = ""

        # Read IMU
        if imu_active:
            # Get absolute orientation (Euler angles: Heading, Roll, Pitch)
            euler = sensor.euler
            if euler.count(None) == 0:
                output_string += f"IMU (Heading, Roll, Pitch): {euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f} | "
            else:
                output_string += "IMU: Calibrating... | "

        # Read GPS
        if gps_active:
            if ser.in_waiting > 0:
                # Read a line of serial data
                newdata = ser.readline().decode('utf-8', errors='ignore').strip()
                # Parse NMEA sentences (looking for GPGGA - location data)
                if newdata.startswith('$GPGGA'):
                    try:
                        msg = pynmea2.parse(newdata)
                        if msg.is_valid:
                            output_string += f"GPS: Lat {msg.latitude:.6f}, Lon {msg.longitude:.6f}, Sats: {msg.num_sats}"
                        else:
                            output_string += "GPS: No Fix (Searching for Satellites...)"
                    except pynmea2.ParseError:
                        output_string += "GPS: Parsing Error"
            else:
                if not output_string: # If IMU is off and no GPS data is waiting
                    output_string = "GPS: Waiting for serial data..."

        # Print the combined output if we have anything
        if output_string:
            print(output_string)
            
        time.sleep(0.5) # Read twice a second

    except KeyboardInterrupt:
        print("\nStopping data stream.")
        if gps_active:
            ser.close()
        break
    except Exception as e:
        print(f"Loop Error: {e}")
        time.sleep(1)
