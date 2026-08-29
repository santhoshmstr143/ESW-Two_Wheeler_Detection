"""Sample runtime configuration for the ride-safety project.

This file is intentionally kept simple and does not change the current pipeline.
It is meant to document the settings currently embedded in the Raspberry Pi scripts.
"""

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
SUPABASE_SERVICE_KEY = "your-service-role-key"

GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
IMU_SAMPLE_RATE = 104

MODEL_DIR = "Code/ras-pi codes"
RIDE_DATA_DIR = "data"

# Model file names
LSTM_MODEL_PATH = "best_model.pth"
POTHOLE_MODEL_PATH = "pothole_model.pth"
