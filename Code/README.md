# Ride Safety Analysis System - Code Documentation

## 📁 Project Structure

```
Code/
├── ml/                          # Machine Learning Training Scripts
│   ├── LSTM-ESW-updated.ipynb  # LSTM event detection model training
│   └── pothole.py              # Pothole detection model training
│
└── ras-pi codes/               # Raspberry Pi Runtime Code
    ├── main.py                 # Main application (run this!)
    ├── formula.py              # Safety index calculator
    ├── gps.py                  # GPS logging utility
    ├── check_video.py          # Camera test script
    ├── best_model.pth          # Trained LSTM model for event detection
    └── pothole_model.pth       # Trained pothole detection model
```

---

## 🌐 Website Dashboard

**Website URL:** [https://rider9600.github.io/Safety-index/](https://rider9600.github.io/Safety-index/)

### How the System Works (Website + Raspberry Pi)

The system uses a **web dashboard** to control the Raspberry Pi remotely via **Supabase cloud database**:

```
┌──────────────┐         ┌──────────────┐         ┌─────────────────┐
│   Website    │  ────►  │   Supabase   │  ────►  │  Raspberry Pi   │
│  (Browser)   │  ◄────  │  (Database)  │  ◄────  │   (main.py)     │
└──────────────┘         └──────────────┘         └─────────────────┘
   User clicks              Cloud stores              Detects command
   START/STOP               commands & data           Starts/stops ride
```

### 🔄 Complete Workflow: Website to Raspberry Pi

#### **STEP 1: Start Raspberry Pi Code (MUST DO FIRST)**

**⚠️ IMPORTANT:** Before using the website, you MUST have `main.py` running on the Raspberry Pi!

```bash
# SSH into your Raspberry Pi
ssh pi@raspberrypi.local

# Navigate to project folder
cd "Code/ras-pi codes"

# Start the main program (it will wait for commands)
python3 main.py
```

**You will see:**
```
============================================================
   IMU (104 Hz) + GPS + Event Detection + Pothole Detection
============================================================
☁️  Supabase: Real-time updates enabled
📊 IMU Sampling Rate: 104 Hz
🧠 LSTM Model: best_model.pth
------------------------------------------------------------
🚀 Starting command listener...

⏳ WAITING FOR START COMMAND FROM WEBSITE...
   GPS, IMU, Event Detection and Pothole Detection will start when START is received
============================================================
```

**✅ The Raspberry Pi is now ready and listening for commands from the website!**

---

#### **STEP 2: Open Website and Select User**

1. **Open the website:** [https://rider9600.github.io/Safety-index/](https://rider9600.github.io/Safety-index/)

2. **Homepage Options:**
   - **Select User** - Choose your rider profile (login/select from dropdown)
   - **View Dashboard** - See past ride history and analytics

3. **Select your Rider ID** (example: Rider 1, Rider 2, etc.)

---

#### **STEP 3: User Interface - Control Panel**

After selecting a user, you'll see the **User Interface** with:

**📱 Control Buttons:**
- **🟢 START Button** - Begin ride recording
- **🔴 STOP Button** - End ride recording

**📊 Real-time Display:**
- **Live GPS Data** - Current location, speed, heading
- **IMU Sensor Data** - Acceleration and gyroscope readings (updated 10 times per second)
- **Ride Status** - Active/Inactive
- **Event Feed** - Live events (turns, braking, potholes detected)

---

#### **STEP 4: Click START on Website**

**What happens when you click START:**

1. **Website** → Sends START command to Supabase database
   ```
   Table: rider_commands
   {
     rider_id: 1,
     command: "start",
     status: "pending",
     timestamp: "2025-12-02 10:30:00"
   }
   ```

2. **Raspberry Pi** → Detects new command (within 1 second)
   ```
   ============================================================
   🚀 START command received!
   👤 Rider ID: 1
   📁 Data folder: data/ride01
   ============================================================
   
   📡 GPS thread STARTED
   ☁️  Supabase upload thread STARTED
   📊 IMU thread STARTED (104 Hz)
   📷 Pothole detection thread STARTED
   🧠 Event prediction thread STARTED
   ```

3. **System Begins Recording:**
   - ✅ GPS tracking (location, speed, heading)
   - ✅ IMU sensors (acceleration, rotation at 104 Hz)
   - ✅ Camera recording video + pothole detection
   - ✅ LSTM event detection (turns, braking, acceleration)
   - ✅ Real-time upload to Supabase (website shows live data)

4. **Website Updates:**
   - Status changes to **"Ride Active"**
   - Live sensor data starts streaming
   - Map shows current location
   - Events appear as they're detected

---

#### **STEP 5: During the Ride**

**On Raspberry Pi (running automatically):**
```
✅ [2025-12-02 10:31:15] IMU: ax=0.123g ay=-0.045g az=0.987g
📡 GPS updated: 12.9716°N 77.5946°E Speed:25.3km/h
🕳️  Pothole #1 detected (87.3%)
💾 Snapshot saved: data/ride01/pothole_image/pothole_20251202_103125.jpg
🎯 New event started: LEFT_TURN (92.1%)
```

**On Website (you see in real-time):**
- 📍 GPS location updating on map
- 📊 Speed graph updating
- ⚠️ "Pothole detected at 10:31:15"
- 🔄 "Left turn detected"
- 📈 Sensor data graphs updating

---

#### **STEP 6: Click STOP on Website**

**What happens when you click STOP:**

1. **Website** → Sends STOP command to Supabase
   ```
   Table: rider_commands
   {
     rider_id: 1,
     command: "stop",
     status: "pending"
   }
   ```

2. **Raspberry Pi** → Detects STOP command
   ```
   ============================================================
   🛑 STOP command received!
   👤 Rider ID: 1
   ℹ️  Stopping threads...
   ============================================================
   
   ✅ GPS thread STOPPED
   ✅ IMU thread STOPPED
   ✅ Supabase upload thread STOPPED
   ✅ Pothole detection thread STOPPED
   ✅ Event prediction thread STOPPED
   💾 Pothole log closed
   💾 Events log closed
   🎥 Video recording saved
   💾 Data saved in: data/ride01
   
   🛡️  Calculating safety index...
   ✅ Safety index calculated successfully
      OVERALL SAFETY INDEX: 7.85/10 (GOOD)
   📊 Safety index saved: data/ride01/index.csv (Score: 7.85/10)
   ☁️ Safety Index uploaded to Supabase successfully!
   
   🔄 System ready for next ride
   ```

3. **Safety Analysis Runs Automatically:**
   - Analyzes all collected data
   - Calculates safety score (0-10)
   - Uploads to Supabase

4. **Website Shows Results:**
   - Ride summary appears
   - Safety score displayed
   - Component scores breakdown
   - Pothole locations on map
   - Event timeline visualization

---

#### **STEP 7: View Results on Website**

After stopping, the website dashboard shows:

**📊 Ride Summary:**
- Total distance traveled
- Average speed
- Ride duration
- Route map with GPS trace

**🛡️ Safety Analysis:**
- **Overall Score:** 7.85/10 (GOOD)
- **Acceleration Score:** 8.2/10
- **Gyroscope Score:** 7.8/10
- **Event Smoothness:** 8.5/10
- **Pothole Encounters:** 6.9/10
- **Speed Consistency:** 7.9/10

**⚠️ Detected Events:**
- 3 Left Turns
- 2 Right Turns
- 1 Hard Braking
- 5 Potholes Detected (with images)

**📥 Download Options:**
- Download sensor data CSV
- Download pothole images
- Download ride video
- Download full report JSON

---

## 🚀 Quick Start Guide

### Prerequisites

**Hardware Required:**
- Raspberry Pi 4/5
- MPU6050 IMU sensor (I2C connection)
- GPS Module (UART connection at /dev/serial0)
- Raspberry Pi Camera Module 3
- Internet connection (for Supabase cloud sync)

**Software Required:**
- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- I2C and Serial interfaces enabled

---

## 📦 Installation

### 1. Enable Hardware Interfaces

```bash
# Enable I2C (for IMU)
sudo raspi-config
# Navigate to: Interface Options → I2C → Enable

# Enable Serial (for GPS)
sudo raspi-config
# Navigate to: Interface Options → Serial Port
# - Login shell over serial: NO
# - Serial port hardware enabled: YES

# Reboot
sudo reboot
```

### 2. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-opencv i2c-tools git

# Verify I2C connection (should show address 0x68 for MPU6050)
sudo i2cdetect -y 1
```

### 3. Install Python Dependencies

```bash
cd "Code/ras-pi codes"

# Install all required Python packages
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install smbus2 pyserial supabase picamera2 pillow opencv-python numpy pandas
```

---

## ▶️ How to Run

### Main Application (Real-time Ride Monitoring)

**This is the primary script that runs on the Raspberry Pi during rides.**

```bash
cd "Code/ras-pi codes"
python3 main.py
```

**What it does:**
1. **Waits for commands** from Supabase database (start/stop via website)
2. **When START command received:**
   - Creates new ride folder: `data/rideXX/`
   - Starts GPS thread (real-time location tracking)
   - Starts IMU thread (104 Hz sensor data logging)
   - Starts Pothole detection (camera-based AI detection)
   - Starts Event detection (LSTM-based riding event classification)
   - Uploads data to Supabase cloud in real-time
3. **When STOP command received:**
   - Stops all threads gracefully
   - Saves all data locally
   - Calculates safety index
   - Uploads final report to cloud

**Output Files (in `data/rideXX/`):**
- `sensor_data.csv` - IMU + GPS data at 104 Hz
- `events.csv` - Detected riding events (turns, braking, etc.)
- `pothole_log.csv` - Pothole detection log
- `pothole_image/` - Captured pothole images
- `ride_video_YYYYMMDD_HHMMSS.mp4` - Recorded ride video
- `safety_report.json` - Detailed safety metrics
- `index.csv` - Overall safety score (0-10)

---

### Safety Index Calculator (Post-Ride Analysis)

**Calculates safety score from collected ride data.**

```bash
cd "Code/ras-pi codes"

# Basic report
python3 formula.py data/ride01

# Detailed report with metrics
python3 formula.py data/ride01 --detailed
```

**What it analyzes:**
- ✅ Acceleration patterns (harsh braking/acceleration)
- ✅ Gyroscope patterns (aggressive turns)
- ✅ Event smoothness (riding style consistency)
- ✅ Pothole encounters (route hazard level)
- ✅ Speed consistency (erratic speed changes)

**Output:**
- Safety Index: 0-10 scale
- Rating: EXCELLENT / GOOD / FAIR / POOR / DANGEROUS
- Component scores breakdown
- Recommendations for improvement

---

### Camera Test (Verify Camera Setup)

**Test if camera is working properly before running main.py**

```bash
cd "Code/ras-pi codes"
python3 check_video.py
```

**What it does:**
- Opens live camera feed
- Displays 1280x720 preview
- Press 'q' to quit

---

### GPS Test (Verify GPS Connection)

**Test GPS module and log data to CSV**

```bash
cd "Code/ras-pi codes"
python3 gps.py
```

**What it does:**
- Reads NMEA sentences from GPS module
- Logs position, speed, course to CSV
- Press Ctrl+C to stop

---

## 🧠 Machine Learning Models

### Pothole Detection Model Training

**Location:** `Code/ml/pothole.py`

**Dataset Structure:**
```
train/
  ├── Plain/      # Normal road images
  └── Pothole/    # Pothole images
test/
  ├── Plain/
  └── Pothole/
```

**Training:**
```bash
cd Code/ml
python3 pothole.py
```

**Output:** `pothole_model.pth` (copy to `ras-pi codes/`)

---

### LSTM Event Detection Model Training

**Location:** `Code/ml/LSTM-ESW-updated.ipynb`

**Dataset:** IMU sensor data with labeled riding events

**Training:**
```bash
jupyter notebook LSTM-ESW-updated.ipynb
# Run all cells
```

**Output:** `best_model.pth` (copy to `ras-pi codes/`)

---

## 🔧 Configuration

### Supabase Setup

Edit `main.py` and `formula.py` to update your Supabase credentials:

```python
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

**Database Tables Required:**
- `rider_commands` - Start/stop commands from website
- `riderdata` - Real-time sensor data
- `riderfiles` - Ride session metadata
- `pothole_events` - Pothole detection events
- `ride_events` - LSTM-detected riding events
- `ride_safety_index` - Calculated safety scores

---

### Hardware Connections

**MPU6050 IMU (I2C):**
```
VCC  → 3.3V
GND  → GND
SDA  → GPIO 2 (Pin 3)
SCL  → GPIO 3 (Pin 5)
```

**GPS Module (UART):**
```
VCC  → 5V
GND  → GND
TX   → GPIO 15 (RXD, Pin 10)
RX   → GPIO 14 (TXD, Pin 8)
```

**Camera:** Connect via CSI cable to camera port

---

## 📊 Data Flow

```
┌─────────────────┐
│  Raspberry Pi   │
│   (main.py)     │
└────────┬────────┘
         │
         ├─► GPS Thread        → sensor_data.csv
         ├─► IMU Thread        → sensor_data.csv
         ├─► Pothole Thread    → pothole_log.csv + images
         ├─► Event Thread      → events.csv
         └─► Supabase Upload   → Cloud Database
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │   Website    │
                                  │  (Dashboard) │
                                  └──────────────┘
```

---

## 🐛 Troubleshooting

### IMU Not Detected
```bash
# Check I2C connection
sudo i2cdetect -y 1
# Should show 0x68

# If not shown, check wiring and enable I2C
sudo raspi-config
```

### GPS No Data
```bash
# Test serial connection
cat /dev/serial0
# Should see NMEA sentences

# Check if enabled
ls -l /dev/serial0
```

### Camera Not Working
```bash
# Check camera detection
vcgencmd get_camera
# Should show: supported=1 detected=1

# Enable camera
sudo raspi-config
# Interface Options → Camera → Enable
```

### Torch/PyTorch Issues
```bash
# Reinstall PyTorch (CPU version for Raspberry Pi)
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 📝 Notes

- **Sampling Rate:** IMU runs at 104 Hz (all data saved locally), uploads to cloud at 10 Hz
- **Storage:** Each ride creates ~50-100 MB of data (CSV + images + video)
- **Network:** Requires internet for Supabase sync (data saved locally if offline)
- **Performance:** Tested on Raspberry Pi 4 (4GB RAM recommended)

---

## 🎯 Typical Workflow (Quick Summary)

### First Time Setup:
1. **Install:** Follow installation steps above (one-time setup)
2. **Connect:** Wire up IMU, GPS, Camera to Raspberry Pi
3. **Configure:** Update Supabase credentials in `main.py` and `formula.py`
4. **Test:** Run `check_video.py` and `gps.py` to verify hardware

### Every Ride:
1. **Start Raspberry Pi:** 
   ```bash
   ssh pi@raspberrypi.local
   cd "Code/ras-pi codes"
   python3 main.py
   ```
   ⏳ Wait for "WAITING FOR START COMMAND FROM WEBSITE..."

2. **Open Website:** [https://rider9600.github.io/Safety-index/](https://rider9600.github.io/Safety-index/)

3. **Select User:** Choose your rider profile from homepage

4. **Click START:** Website sends command → Raspberry Pi starts recording

5. **Ride:** System automatically logs everything (GPS, sensors, video, events)

6. **Click STOP:** Website sends command → Raspberry Pi stops and calculates safety score

7. **View Results:** Dashboard shows ride summary, safety score, and detected events

---

## ❓ Frequently Asked Questions

### Q: What if I click START on website but nothing happens on Raspberry Pi?

**A:** Check these:
- ✅ Is `main.py` running on Raspberry Pi? (You should see "WAITING FOR START COMMAND")
- ✅ Is Raspberry Pi connected to internet?
- ✅ Are Supabase credentials correct in `main.py`?
- ✅ Check Raspberry Pi terminal for error messages

### Q: Can I use multiple Raspberry Pis with the same website?

**A:** Yes! Each Raspberry Pi can be assigned a different `rider_id`. The website dashboard shows data for the selected rider.

### Q: What if internet connection drops during ride?

**A:** All data is saved locally on Raspberry Pi in `data/rideXX/` folder. Only real-time cloud upload will pause. Data remains safe.

### Q: How do I download the recorded video?

**A:** After ride ends:
- On Raspberry Pi: Check `data/rideXX/ride_video_YYYYMMDD_HHMMSS.mp4`
- Copy via SCP: `scp pi@raspberrypi.local:"Code/ras-pi\ codes/data/ride01/*.mp4" .`
- Or use FileZilla/WinSCP to transfer files

### Q: Can I stop the ride without using website?

**A:** Yes, press `Ctrl+C` in the Raspberry Pi terminal where `main.py` is running. This will gracefully stop all threads and save data.

### Q: Where is all the ride data stored?

**A:** 
- **Locally on Raspberry Pi:** `data/rideXX/` folders (CSV files, images, videos)
- **Cloud (Supabase):** Real-time sensor data, events, safety scores (viewable on website)

---

## 📄 License

This project is part of an Embedded Systems Workshop assignment.

---

## 👥 Support

For issues or questions, please refer to the project documentation or contact the development team