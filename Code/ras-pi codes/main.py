import os, csv, time, serial, uuid
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue
from smbus2 import SMBus
from supabase import create_client

# Pothole detection imports
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from picamera2 import Picamera2
from collections import deque

# ========= Supabase Config =========
SUPABASE_URL = "https://ghtqafnlnijxvsmzdnmh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdodHFhZm5sbmlqeHZzbXpkbm1oIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk3NDQyNjcsImV4cCI6MjA3NTMyMDI2N30.Q1LGQP8JQdWn6rJJ1XRYT8rfo9b2Q5YfWUytrzQEsa0"
# NOTE: keep your key secure. I left it as-is from your snippet but consider env vars.

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========= Hardware Config =========
BUS_NUM = 1
ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
IMU_SAMPLE_RATE = 104  # 104 samples per second for IMU

# ========= LSTM Event Detection Config =========
LSTM_MODEL_PATH = "best_model.pth"
LSTM_WINDOW_SIZE = 40  # Must match training n_time_steps
LSTM_STRIDE = 10  # How often to run inference (every N samples)

# ========= Globals =========
running = False
rider_id = None
current_file_id = None
current_folder = None
pothole_images_folder = None
pothole_log_file = None
pothole_log_writer = None
events_log_file = None
events_log_writer = None

# LSTM event detection globals
imu_buffer = deque(maxlen=LSTM_WINDOW_SIZE)  # Rolling buffer for LSTM input
imu_buffer_lock = Lock()
event_model = None
event_scaler_mean = None
event_scaler_scale = None
event_classes = None

gps_lock = Lock()
latest_gps = {
    "utc": "N/A", "lat": "N/A", "ns": "N/A",
    "lon": "N/A", "ew": "N/A", "speed": "N/A",
    "course": "N/A", "date": "N/A", "valid": "N/A"
}

supabase_queue = Queue(maxsize=1000)
gps_thread_obj = None
imu_thread_obj = None
supabase_thread_obj = None
pothole_thread_obj = None
event_thread_obj = None
stop_event = Event()

# ========== Pothole model & preprocessing ==========
# Will be loaded the first time the pothole thread runs (or you can load here)
MODEL_PATH = "pothole_model.pth"
classes = ['Plain', 'Pothole']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Toggle preview window (user said SHOW)
show_preview = True

# ========== LSTM Event Detection Model ==========
class BidirectionalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rates, l2_reg=1e-4):
        super(BidirectionalLSTMModel, self).__init__()
        self.l2_reg = l2_reg
        
        # First Bidirectional LSTM (128 units each direction = 256 total output)
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0] * 2)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # Second Bidirectional LSTM (96 units each direction = 192 total output)
        self.lstm2 = nn.LSTM(hidden_sizes[0] * 2, hidden_sizes[1], batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1] * 2)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_sizes[1] * 2, 96)
        self.bn3 = nn.BatchNorm1d(96)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        self.fc2 = nn.Linear(96, 48)
        self.dropout4 = nn.Dropout(dropout_rates[3])
        
        self.fc3 = nn.Linear(48, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # First LSTM layer
        out, _ = self.lstm1(x)
        # out shape: (batch, seq_len, hidden*2)
        # Take only the last timestep for batch norm
        out_last = out[:, -1, :]  # (batch, hidden*2)
        out_last = self.bn1(out_last)
        out_last = self.dropout1(out_last)
        # Expand back to sequence
        out = out_last.unsqueeze(1)  # (batch, 1, hidden*2)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take last timestep (batch, hidden*2)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Dense layers
        out = torch.relu(self.fc1(out))
        out = self.bn3(out)
        out = self.dropout3(out)
        
        out = torch.relu(self.fc2(out))
        out = self.dropout4(out)
        
        out = self.fc3(out)
        return out

# ---------- Helper: IMU read ----------
def read_word(bus, addr, reg):
    try:
        high = bus.read_byte_data(addr, reg)
        low  = bus.read_byte_data(addr, reg+1)
        val = (high << 8) | low
        if val & 0x8000:
            val -= 0x10000
        return val
    except Exception as e:
        print(f"IMU read error: {e}")
        return 0

# ---------- Helper: setup folder ----------
def setup_data_folder():
    global pothole_images_folder, pothole_log_file, pothole_log_writer, events_log_file, events_log_writer
    
    main_data_dir = "data"
    os.makedirs(main_data_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(main_data_dir, f"ride{n:02d}")):
        n += 1
    session_folder = os.path.join(main_data_dir, f"ride{n:02d}")
    os.makedirs(session_folder, exist_ok=True)
    
    # Create pothole_image subfolder
    pothole_images_folder = os.path.join(session_folder, "pothole_image")
    os.makedirs(pothole_images_folder, exist_ok=True)
    
    # Create pothole_log.csv
    pothole_log_path = os.path.join(session_folder, "pothole_log.csv")
    pothole_log_file = open(pothole_log_path, "w", newline="")
    pothole_log_writer = csv.writer(pothole_log_file)
    pothole_log_writer.writerow(["timestamp", "epoch_time", "image_filename", "confidence_percent"])
    print(f"📋 Pothole log created: {pothole_log_path}")
    
    # Create events.csv with time range format
    events_log_path = os.path.join(session_folder, "events.csv")
    events_log_file = open(events_log_path, "w", newline="")
    events_log_writer = csv.writer(events_log_file)
    events_log_writer.writerow(["start_time", "end_time", "event_type", "confidence_percent", "duration_seconds"])
    events_log_file.flush()
    print(f"📋 Events log created: {events_log_path}")
    
    return session_folder

# ---------- Helper: parse GPS ----------
def parse_gprmc(line):
    try:
        parts = line.split(",")
        if len(parts) >= 10:
            return {
                "utc": parts[1] or "N/A",
                "valid": parts[2] or "N/A",
                "lat": parts[3] or "N/A",
                "ns": parts[4] or "N/A",
                "lon": parts[5] or "N/A",
                "ew": parts[6] or "N/A",
                "speed": parts[7] or "N/A",
                "course": parts[8] or "N/A",
                "date": parts[9] or "N/A"
            }
    except:
        pass
        return None

# ---------- Helper: Load LSTM model ----------
def load_event_model():
    """Load the LSTM event detection model from best_model.pth"""
    global event_model, event_scaler_mean, event_scaler_scale, event_classes
    
    try:
        print("🧠 Loading LSTM event detection model...")
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location='cpu')
        
        # Extract metadata
        event_scaler_mean = np.array(checkpoint['scaler_mean'], dtype=np.float32)
        event_scaler_scale = np.array(checkpoint['scaler_scale'], dtype=np.float32)
        event_classes = checkpoint['classes']
        
        num_classes = len(event_classes)
        n_features = len(event_scaler_mean)  # Should be 7 (Speed, Ax, Ay, Az, Gx, Gy, Gz)
        
        # Initialize model
        event_model = BidirectionalLSTMModel(
            input_size=n_features,
            hidden_sizes=[128, 96],
            num_classes=num_classes,
            dropout_rates=[0.35, 0.35, 0.25, 0.2],
            l2_reg=1e-4
        )
        
        event_model.load_state_dict(checkpoint['model_state_dict'])
        event_model.eval()
        
        print(f"✅ LSTM model loaded successfully")
        print(f"   Classes: {event_classes}")
        print(f"   Features: {n_features}, Window size: {LSTM_WINDOW_SIZE}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load LSTM model: {e}")
        return False# ========= Supabase Upload Thread =========
def supabase_upload_thread():
    print("☁️  Supabase upload thread started...")
    while not stop_event.is_set() or not supabase_queue.empty():
        try:
            data = supabase_queue.get(timeout=1)
            try:
                supabase.table("riderdata").insert(data).execute()
            except Exception as e:
                # Optional: print upload error for debugging
                print(f"Supabase upload error: {e}")
            supabase_queue.task_done()
        except Exception:
            pass
    print("☁️  Supabase upload thread exiting...")

# ========= Command Listener Thread =========
def command_listener():
    global running, current_file_id, rider_id, current_folder
    global gps_thread_obj, imu_thread_obj, supabase_thread_obj, pothole_thread_obj, event_thread_obj, stop_event
    global pothole_log_file, pothole_log_writer, events_log_file, events_log_writer
    global imu_buffer

    last_command_id = None
    print("👂 Command listener started - waiting for commands...")

    while True:
        try:
            result = supabase.table("rider_commands")\
                .select("*")\
                .eq("status", "pending")\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()

            if result.data:
                latest = result.data[0]
                if latest["id"] != last_command_id:
                    last_command_id = latest["id"]
                    command = latest["command"].lower()
                    rider_id = latest["rider_id"]

                    if command == "start" and not running:
                        running = True
                        stop_event.clear()
                        
                        # Clear IMU buffer
                        with imu_buffer_lock:
                            imu_buffer.clear()
                        
                        # Clear Supabase queue
                        while not supabase_queue.empty():
                            try:
                                supabase_queue.get_nowait()
                            except:
                                break
                        
                        # Load LSTM model
                        if not load_event_model():
                            print("❌ Failed to load event model. Events will not be detected.")

                        current_folder = setup_data_folder()

                        file_res = supabase.table("riderfiles").insert({
                            "rider_id": rider_id,
                            "filename": f"ride_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "filepath": current_folder
                        }).execute()
                        try:
                            current_file_id = file_res.data[0]["id"]
                        except:
                            current_file_id = None

                        print(f"\n{'='*60}")
                        print(f"🚀 START command received!")
                        print(f"👤 Rider ID: {rider_id}")
                        print(f"📁 Data folder: {current_folder}")
                        print(f"{'='*60}\n")

                        # Start threads
                        gps_thread_obj = Thread(target=gps_thread, daemon=False)
                        gps_thread_obj.start()
                        print("📡 GPS thread STARTED")

                        supabase_thread_obj = Thread(target=supabase_upload_thread, daemon=False)
                        supabase_thread_obj.start()
                        print("☁️  Supabase upload thread STARTED")

                        imu_thread_obj = Thread(target=imu_thread, daemon=False)
                        imu_thread_obj.start()
                        print("📊 IMU thread STARTED (104 Hz)")

                        # Start Pothole detection thread
                        pothole_thread_obj = Thread(target=pothole_thread, daemon=False)
                        pothole_thread_obj.start()
                        print("📷 Pothole detection thread STARTED")
                        
                        # Start Event prediction thread
                        if event_model is not None:
                            event_thread_obj = Thread(target=event_prediction_thread, daemon=False)
                            event_thread_obj.start()
                            print("🧠 Event prediction thread STARTED")
                        else:
                            print("⚠️ Event prediction thread NOT started (model not loaded)")

                        # Mark command as executed
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                    elif command == "stop" and running:
                        running = False
                        stop_event.set()

                        print(f"\n{'='*60}")
                        print(f"🛑 STOP command received!")
                        print(f"👤 Rider ID: {rider_id}")
                        print(f"ℹ️  Stopping threads...")
                        print(f"{'='*60}\n")

                        # Wait for threads to finish
                        if gps_thread_obj and gps_thread_obj.is_alive():
                            gps_thread_obj.join(timeout=5)
                            print("✅ GPS thread STOPPED")
                        if imu_thread_obj and imu_thread_obj.is_alive():
                            imu_thread_obj.join(timeout=5)
                            print("✅ IMU thread STOPPED")
                        if supabase_thread_obj and supabase_thread_obj.is_alive():
                            supabase_thread_obj.join(timeout=10)
                            print("✅ Supabase upload thread STOPPED")
                        if pothole_thread_obj and pothole_thread_obj.is_alive():
                            pothole_thread_obj.join(timeout=10)
                            print("✅ Pothole detection thread STOPPED")
                        if event_thread_obj and event_thread_obj.is_alive():
                            event_thread_obj.join(timeout=10)
                            print("✅ Event prediction thread STOPPED")

                        # Close CSV files
                        try:
                            if pothole_log_file:
                                pothole_log_file.close()
                                print("💾 Pothole log closed")
                        except:
                            pass
                        
                        try:
                            if events_log_file:
                                events_log_file.close()
                                print("💾 Events log closed")
                        except:
                            pass

                        print(f"💾 Data saved in: {current_folder}\n")
                        
                        # Calculate safety index and create index.csv
                        try:
                            print("🛡️  Calculating safety index...")
                            import subprocess
                            env = os.environ.copy()
                            env["RIDER_ID"] = str(rider_id)
                            env["FILE_ID"] = str(current_file_id)

                            result = subprocess.run(
                                            ['python3', 'formula.py', current_folder],
                                            env=env,
                                            capture_output=True,
                                            text=True,
                                            timeout=30
                            )
                            if result.returncode == 0:
                                print("✅ Safety index calculated successfully")
                                # Extract the score from output
                                for line in result.stdout.split('\n'):
                                    if 'OVERALL SAFETY INDEX:' in line:
                                        print(f"   {line.strip()}")
                            else:
                                print(f"⚠️  Safety index calculation had issues: {result.stderr}")
                        except subprocess.TimeoutExpired:
                            print("⚠️  Safety index calculation timed out")
                        except Exception as e:
                            print(f"⚠️  Could not calculate safety index: {e}")

                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                        # Reset session variables
                        current_file_id = None
                        current_folder = None
                        gps_thread_obj = None
                        imu_thread_obj = None
                        supabase_thread_obj = None
                        pothole_thread_obj = None
                        event_thread_obj = None
                        pothole_log_file = None
                        pothole_log_writer = None
                        events_log_file = None
                        events_log_writer = None
                        
                        # Clear IMU buffer
                        with imu_buffer_lock:
                            imu_buffer.clear()

                        print("🔄 System ready for next ride\n")

        except Exception as e:
            print(f"❌ Command listener error: {e}")

        time.sleep(1)

# ========= GPS Thread =========
def gps_thread():
    print("📡 GPS collection starting...")
    retry_count = 0
    max_retries = 5
    
    while not stop_event.is_set() and retry_count < max_retries:
        try:
            print(f"📡 Opening GPS port {GPS_PORT} at {GPS_BAUD} baud...")
            with serial.Serial(GPS_PORT, GPS_BAUD, timeout=2) as ser:
                print("✅ GPS port opened successfully")
                retry_count = 0  # Reset retry counter on successful connection
                line_count = 0
                
                while not stop_event.is_set():
                    try:
                        line = ser.readline().decode("ascii", errors="ignore").strip()
                        
                        if line:  # Got some data
                            line_count += 1
                            if line_count % 100 == 0:
                                print(f"📡 GPS active: {line_count} lines read")
                            
                            if line.startswith("$GPRMC"):
                                gps_data = parse_gprmc(line)
                                if gps_data:
                                    with gps_lock:
                                        latest_gps.update(gps_data)
                                    if line_count % 50 == 0:
                                        print(f"📡 GPS updated: {gps_data['lat']}{gps_data['ns']} {gps_data['lon']}{gps_data['ew']} Speed:{gps_data['speed']}kn")
                        else:
                            # No data received in timeout period - this is normal
                            pass
                            
                    except serial.SerialException as e:
                        print(f"⚠️ GPS serial exception: {e}")
                        break  # Exit inner loop to reconnect
                    except Exception as e:
                        if not stop_event.is_set():
                            print(f"⚠️ GPS parse error: {e}")
                        time.sleep(0.1)
                        
        except serial.SerialException as e:
            retry_count += 1
            print(f"❌ GPS port error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries and not stop_event.is_set():
                print(f"⏳ Retrying GPS connection in 3 seconds...")
                time.sleep(3)
        except Exception as e:
            print(f"❌ GPS fatal error: {e}")
            break
    
    if retry_count >= max_retries:
        print("❌ GPS failed after maximum retries")
    print("📡 GPS thread exiting...")

# ========= IMU Thread =========
def imu_thread():
    print("📊 IMU initialization starting...")
    bus = None
    try:
        bus = SMBus(BUS_NUM)
        bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        print("✅ IMU initialized successfully (104 Hz sampling)")
    except Exception as e:
        print(f"❌ IMU initialization error: {e}")
        return

    csv_file = None
    csv_writer = None
    sample_count = 0

    try:
        csv_path = os.path.join(current_folder, "sensor_data.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "epoch_time",
            "ax_g", "ay_g", "az_g",
            "gx_dps", "gy_dps", "gz_dps",
            "gps_utc", "gps_lat", "gps_ns", "gps_lon", "gps_ew",
            "gps_speed_kn", "gps_course_deg", "gps_valid"
        ])
        print(f"📝 CSV file created: {csv_path}")

        sample_interval = 1.0 / IMU_SAMPLE_RATE
        next_sample_time = time.time()

        while not stop_event.is_set():
            current_time = time.time()
            if current_time >= next_sample_time:
                now = datetime.now()
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                epoch_time = current_time

                ax = read_word(bus, ADDR, ACCEL_XOUT_H)
                ay = read_word(bus, ADDR, ACCEL_XOUT_H+2)
                az = read_word(bus, ADDR, ACCEL_XOUT_H+4)
                gx = read_word(bus, ADDR, GYRO_XOUT_H)
                gy = read_word(bus, ADDR, GYRO_XOUT_H+2)
                gz = read_word(bus, ADDR, GYRO_XOUT_H+4)

                ax_g = ax / 16384.0
                ay_g = ay / 16384.0
                az_g = az / 16384.0
                gx_dps = gx / 131.0
                gy_dps = gy / 131.0
                gz_dps = gz / 131.0

                # Parse GPS speed and convert knots to km/h
                with gps_lock:
                    gps_copy = latest_gps.copy()
                
                speed_val = 0.0
                if gps_copy["speed"] != "N/A":
                    try:
                        speed_knots = float(gps_copy["speed"])
                        speed_val = speed_knots * 1.852  # Convert knots to km/h
                    except:
                        speed_val = 0.0
                
                # Add to LSTM buffer (Speed, Ax, Ay, Az, Gx, Gy, Gz)
                # This must match the training data column order
                with imu_buffer_lock:
                    imu_buffer.append([speed_val, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps])

                csv_writer.writerow([
                    timestamp_str, epoch_time,
                    round(ax_g, 4), round(ay_g, 4), round(az_g, 4),
                    round(gx_dps, 3), round(gy_dps, 3), round(gz_dps, 3),
                    gps_copy["utc"], gps_copy["lat"], gps_copy["ns"],
                    gps_copy["lon"], gps_copy["ew"],
                    gps_copy["speed"], gps_copy["course"], gps_copy["valid"]
                ])

                sample_count += 1
                if sample_count % 100 == 0:
                    csv_file.flush()

                speed_val = None
                course_val = None
                if gps_copy["speed"] != "N/A":
                    try:
                        speed_knots = float(gps_copy["speed"])
                        speed_val = speed_knots * 1.852  # Convert knots to km/h
                    except:
                        pass
                if gps_copy["course"] != "N/A":
                    try:
                        course_val = float(gps_copy["course"])
                    except:
                        pass

                # Upload to Supabase at 10 Hz (every 10th sample = ~10 times per second)
                # Note: All 104 samples/sec are still saved to local CSV
                if sample_count % 10 == 0:
                    try:
                        supabase_queue.put_nowait({
                            "rider_id": rider_id,
                            "file_id": current_file_id,
                            "timestamp": now.isoformat(),
                            "ax": round(ax_g, 4),
                            "ay": round(ay_g, 4),
                            "az": round(az_g, 4),
                            "gx": round(gx_dps, 3),
                            "gy": round(gy_dps, 3),
                            "gz": round(gz_dps, 3),
                            "gps_utc": gps_copy["utc"],
                            "gps_lat": gps_copy["lat"],
                            "gps_ns": gps_copy["ns"],
                            "gps_lon": gps_copy["lon"],
                            "gps_ew": gps_copy["ew"],
                            "gps_speed_kn": speed_val,
                            "gps_course_deg": course_val,
                            "gps_valid": gps_copy["valid"]
                        })
                    except:
                        pass

                if sample_count % 10 == 0:
                    print(f"✅ [{timestamp_str}] IMU: ax={ax_g:.3f}g ay={ay_g:.3f}g az={az_g:.3f}g | gx={gx_dps:.1f}°/s gy={gy_dps:.1f}°/s gz={gz_dps:.1f}°/s | speed={speed_val if speed_val else 0.0:.1f}km/h")
                next_sample_time += sample_interval
            else:
                time.sleep(0.0001)

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ IMU main loop error: {e}")
    finally:
        try:
            if csv_file:
                csv_file.close()
                print("💾 CSV file closed")
        except Exception as e:
            print(f"CSV cleanup error: {e}")
        try:
            if bus:
                bus.close()
                print("🔌 IMU bus closed")
        except Exception as e:
            print(f"IMU cleanup error: {e}")
    print("📊 IMU thread exiting...")

# ========= Event Prediction Thread =========
def event_prediction_thread():
    """Run LSTM inference on buffered IMU data and log event time ranges to events.csv"""
    global imu_buffer, event_model, events_log_writer, events_log_file
    
    print("🧠 Event prediction thread starting...")
    
    if event_model is None:
        print("❌ Event model not loaded. Exiting event thread.")
        return
    
    sample_counter = 0
    current_event = None
    event_start_time = None
    event_start_timestamp = None
    event_confidence_sum = 0.0
    event_confidence_count = 0
    
    try:
        while not stop_event.is_set():
            time.sleep(0.01)  # Check every 10ms
            
            with imu_buffer_lock:
                if len(imu_buffer) < LSTM_WINDOW_SIZE:
                    continue
                
                # Get last LSTM_WINDOW_SIZE samples
                window_data = list(imu_buffer)[-LSTM_WINDOW_SIZE:]
            
            sample_counter += 1
            
            # Run inference every LSTM_STRIDE samples
            if sample_counter % LSTM_STRIDE != 0:
                continue
            
            try:
                # Convert to numpy array (shape: LSTM_WINDOW_SIZE x 7)
                window_array = np.array(window_data, dtype=np.float32)
                
                # Standardize using saved scaler params
                window_scaled = (window_array - event_scaler_mean) / event_scaler_scale
                
                # Convert to tensor (batch_size=1, seq_len=LSTM_WINDOW_SIZE, features=7)
                window_tensor = torch.from_numpy(window_scaled).unsqueeze(0).float()
                
                # Run inference
                with torch.no_grad():
                    output = event_model(window_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred = torch.max(probs, 1)
                    event_label = event_classes[pred.item()]
                    conf_percent = confidence.item() * 100
                
                # Only process events with confidence > 60%
                if conf_percent > 60:
                    if event_label != current_event:
                        # Event transition detected
                        now = datetime.now()
                        current_time = time.time()
                        
                        # If there was a previous event, log it with time range
                        if current_event is not None and event_start_time is not None:
                            end_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            duration = current_time - event_start_time
                            avg_confidence = event_confidence_sum / max(event_confidence_count, 1)
                            
                            # Write to events.csv with time range
                            if events_log_writer:
                                try:
                                    events_log_writer.writerow([
                                        event_start_timestamp,
                                        end_timestamp,
                                        current_event,
                                        round(avg_confidence, 1),
                                        round(duration, 2)
                                    ])
                                    events_log_file.flush()
                                    print(f"📝 Event logged: {event_start_timestamp} → {end_timestamp} | {current_event} ({avg_confidence:.1f}%) | {duration:.1f}s")
                                except Exception as e:
                                    print(f"❌ Error writing to events.csv: {e}")
                                try:
                                    supabase.table("ride_events").insert({
                                        "rider_id": rider_id,
                                         "file_id": current_file_id,
                                         "event_type": current_event, 
                                         "start_time": event_start_timestamp,
                                          "end_time": end_timestamp,
                                          "confidence_percent": round(avg_confidence, 1),
                                         "duration_seconds": round(duration, 2)
                                    }).execute()
                                    print("☁️ Uploaded event to Supabase")
                                except Exception as e:
                                     print("❌ Supabase event upload error:", e)

                        # Start tracking new event
                        current_event = event_label
                        event_start_time = current_time
                        event_start_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        event_confidence_sum = conf_percent
                        event_confidence_count = 1
                        print(f"🎯 New event started: {event_label} ({conf_percent:.1f}%)")
                    else:
                        # Same event continuing, accumulate confidence
                        event_confidence_sum += conf_percent
                        event_confidence_count += 1
                
            except Exception as e:
                if not stop_event.is_set():
                    print(f"❌ Event prediction error: {e}")
        
        # Log final event if exists when thread stops
        if current_event is not None and event_start_time is not None:
            now = datetime.now()
            end_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            duration = time.time() - event_start_time
            avg_confidence = event_confidence_sum / max(event_confidence_count, 1)
            
            if events_log_writer:
                try:
                    events_log_writer.writerow([
                        event_start_timestamp,
                        end_timestamp,
                        current_event,
                        round(avg_confidence, 1),
                        round(duration, 2)
                    ])
                    events_log_file.flush()
                    print(f"📝 Final event logged: {event_start_timestamp} → {end_timestamp} | {current_event} ({avg_confidence:.1f}%) | {duration:.1f}s")
                except Exception as e:
                    print(f"❌ Error writing final event: {e}")
                try:
                    supabase.table("ride_events").insert({
                                        "rider_id": rider_id,
                                         "file_id": current_file_id,
                                         "event_type": current_event, 
                                         "start_time": event_start_timestamp,
                                          "end_time": end_timestamp,
                                          "confidence_percent": round(avg_confidence, 1),
                                         "duration_seconds": round(duration, 2)
                    }).execute()
                    print("☁️ Uploaded event to Supabase")
                except Exception as e:
                    print("❌ Supabase event upload error:", e)
    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Event thread main loop error: {e}")
    
    print("🧠 Event prediction thread exiting...")

# ========= Pothole Detection Thread =========
def pothole_thread():
    """Runs camera, model inference, shows preview, saves pothole snaps."""
    global pothole_images_folder, rider_id, current_file_id, pothole_log_writer, pothole_log_file, events_log_writer, events_log_file, current_folder

    print("📷 Pothole thread initializing...")

    # Load model
    try:
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print("✅ Pothole model loaded")
    except Exception as e:
        print(f"❌ Failed to load pothole model: {e}")
        return

    # Start camera
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("✅ Picamera2 started")
    except Exception as e:
        print(f"❌ Picamera2 start error: {e}")
        return

    # Setup video writer
    video_writer = None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"ride_video_{timestamp}.mp4"
        video_path = os.path.join(current_folder, video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0  # Same as camera
        frame_size = (640, 480)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        
        if video_writer.isOpened():
            print(f"🎥 Video recording started: {video_path}")
        else:
            print("❌ Failed to open video writer")
            video_writer = None
    except Exception as e:
        print(f"❌ Video writer setup error: {e}")
        video_writer = None

    frame_count = 0
    pothole_count = 0

    try:
        while not stop_event.is_set():

            frame = picam2.capture_array()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame to video
            if video_writer and video_writer.isOpened():
                video_writer.write(frame_bgr)

            # Model inference
            img = Image.fromarray(frame)
            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img_t)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                label = classes[pred.item()]
                conf_percent = confidence.item() * 100

            # Draw debug text
            color = (0, 255, 0) if label == "Plain" else (0, 0, 255)
            text = f"{label}: {conf_percent:.1f}%"
            cv2.putText(frame_bgr, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ============================
            #   POTHOLE DETECTED
            # ============================
            if label == "Pothole" and conf_percent > 70:
                pothole_count += 1

                # Save local snapshot in pothole_image folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                epoch_time = time.time()
                filename = f"pothole_{timestamp}.jpg"
                filepath = os.path.join(pothole_images_folder, filename)
                cv2.imwrite(filepath, frame_bgr)

                print(f"🕳️ Pothole #{pothole_count} detected ({conf_percent:.1f}%)")
                print(f"💾 Snapshot saved: {filepath}")

                # Log to pothole_log.csv
                if pothole_log_writer:
                    try:
                        pothole_log_writer.writerow([timestamp, epoch_time, filename, round(conf_percent, 2)])
                        pothole_log_file.flush()
                        print(f"📝 Pothole logged to pothole_log.csv")
                    except Exception as e:
                        print(f"❌ Error writing to pothole_log.csv: {e}")

                # Check if rider_id and file_id exist
                if rider_id is None or current_file_id is None:
                    print("❌ ERROR: rider_id or file_id is None. Cannot upload pothole event!")
                else:
                    # Upload to Supabase
                    try:
                        response = supabase.table("pothole_events").insert({
                            "rider_id": rider_id,
                            "file_id": current_file_id,
                            "detected_at": datetime.now().isoformat()
                        }).execute()

                        # Check server response
                        if hasattr(response, "error") and response.error:
                            print("❌ Supabase Insert Error:", response.error)
                        else:
                            print("☁️ Pothole event uploaded:", response.data)

                    except Exception as e:
                        print("❌ Exception uploading pothole event:", e)

            # Show preview feed
            if show_preview:
                cv2.imshow("Pothole Detection (Press 'q' to stop)", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("⌨️ 'q' pressed → stopping")
                    stop_event.set()
                    break

            time.sleep(0.001)

    except Exception as e:
        if not stop_event.is_set():
            print("❌ Pothole thread error:", e)

    finally:
        try:
            picam2.stop()
        except:
            pass

        if video_writer:
            try:
                video_writer.release()
                print("🎥 Video recording saved")
            except:
                pass

        if show_preview:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        print(f"📷 Pothole thread exiting. Frames={frame_count}, potholes={pothole_count}")

# ========= Main =========
if __name__ == "__main__":
    print("=" * 60)
    print("   IMU (104 Hz) + GPS + Event Detection + Pothole Detection")
    print("=" * 60)
    print(f"☁️  Supabase: Real-time updates enabled")
    print(f"📊 IMU Sampling Rate: {IMU_SAMPLE_RATE} Hz")
    print(f"🧠 LSTM Model: {LSTM_MODEL_PATH}")
    print(f"🧠 Event Window: {LSTM_WINDOW_SIZE} samples, stride: {LSTM_STRIDE}")
    print("-" * 60)

    try:
        t_cmd = Thread(target=command_listener, daemon=True)
        print("🚀 Starting command listener...")
        t_cmd.start()

        print("\n⏳ WAITING FOR START COMMAND FROM WEBSITE...")
        print("   GPS, IMU, Event Detection and Pothole Detection will start when START is received")
        print("=" * 60)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⛔ Received stop signal...")
        stop_event.set()
        print("👋 Exiting program...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("✅ Program ended.")