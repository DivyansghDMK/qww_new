"""
Constants and configuration values for ECG processing
"""

# ECG Lead Names
ECG_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Sampling Rates
DEFAULT_SAMPLING_RATE = 80
ALTERNATIVE_SAMPLING_RATE = 250

# Buffer Configuration
DEFAULT_BUFFER_SIZE = 1000
MAX_BUFFER_SIZE = 2000

# Timer Intervals (milliseconds)
GUI_UPDATE_INTERVAL = 50  # 20 Hz
ECG_UPDATE_INTERVAL = 100  # 10 Hz
OVERLAY_UPDATE_INTERVAL = 100  # 10 Hz
RECORDING_UPDATE_INTERVAL = 33  # ~30 Hz

# Hardware Configuration
MAX_READINGS_PER_CYCLE = 20
DEFAULT_BAUD_RATE = 9600
DEFAULT_TIMEOUT = 1.0

# Signal Processing
BANDPASS_LOW = 0.5  # Hz
BANDPASS_HIGH = 40.0  # Hz
NOTCH_FREQUENCY = 50.0  # Hz

# Heart Rate Calculation
MIN_HEART_RATE = 30  # BPM
MAX_HEART_RATE = 200  # BPM
DEFAULT_HEART_RATE = 72  # BPM

# ECG Metrics Ranges
PR_INTERVAL_MIN = 120  # ms
PR_INTERVAL_MAX = 200  # ms
QRS_DURATION_MIN = 80  # ms
QRS_DURATION_MAX = 120  # ms
QT_INTERVAL_MIN = 350  # ms
QT_INTERVAL_MAX = 450  # ms

# File Paths
ASSETS_DIR = "assets"
CONFIG_FILE = "ecg_settings.json"
USERS_FILE = "users.json"
LOG_FILE = "ecg_app.log"

# UI Configuration
DEFAULT_WINDOW_SIZE = (1200, 800)
DEFAULT_FONT_FAMILY = "Arial"
DEFAULT_FONT_SIZE = 12
DASHBOARD_REFRESH_RATE = 20  # Hz

# Audio Configuration
HEARTBEAT_SOUND_FILE = "heartbeat.wav"
MAX_AUDIO_VOLUME = 100

# Error Messages
ERROR_MESSAGES = {
    "import_error": "❌ Core module import error: {}",
    "ecg_import_warning": "⚠️ ECG module import warning: {}",
    "config_load_error": "Warning: Could not load config file {}: {}",
    "config_save_error": "Error saving config file: {}",
    "serial_connection_error": "Error connecting to ECG device: {}",
    "data_processing_error": "Error processing ECG data: {}",
    "file_not_found": "File not found: {}",
    "invalid_config": "Invalid configuration value: {}"
}

# Success Messages
SUCCESS_MESSAGES = {
    "modules_loaded": "✅ Core modules imported successfully",
    "ecg_modules_loaded": "✅ ECG modules imported successfully",
    "config_loaded": "✅ Configuration loaded successfully",
    "device_connected": "✅ ECG device connected successfully",
    "data_processed": "✅ ECG data processed successfully"
}

# File Extensions
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.ogg']
SUPPORTED_DATA_FORMATS = ['.csv', '.txt', '.json']

# ECG Analysis Parameters
PAN_TOMPKINS_THRESHOLD = 0.3
R_PEAK_MIN_DISTANCE = 0.2  # seconds
NOISE_THRESHOLD = 0.1

# Display Configuration
ECG_LINE_WIDTH = 0.5
ECG_COLOR = '#2E8B57'  # Sea Green
GRID_COLOR = '#CCCCCC'
BACKGROUND_COLOR = '#FFFFFF'
TEXT_COLOR = '#000000'

# Performance Settings
MAX_CONCURRENT_THREADS = 4
MEMORY_CLEANUP_INTERVAL = 60  # seconds
LOG_ROTATION_SIZE = 10 * 1024 * 1024  # 10MB
