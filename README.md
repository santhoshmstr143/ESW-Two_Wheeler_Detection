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
 