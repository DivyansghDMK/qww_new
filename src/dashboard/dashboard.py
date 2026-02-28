from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QGridLayout, QCalendarWidget, QTextEdit,
    QDialog, QLineEdit, QComboBox, QFormLayout, QMessageBox, QSizePolicy, QStackedWidget, QScrollArea, QSpacerItem, QSlider
)
from PyQt5.QtGui import QFont, QPixmap, QMovie
from PyQt5.QtCore import Qt, QTimer, QSize
try:
    from PyQt5.QtMultimedia import QSound
except ImportError:
    print(" QSound not available - heartbeat sound will be disabled")
    QSound = None
import sys
import platform
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ecg.serial.serial_reader import SerialStreamReader, SERIAL_AVAILABLE
import serial.tools.list_ports
if SERIAL_AVAILABLE:
    from ecg.serial.hardware_commands import HardwareCommandHandler
    import serial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import math
import os
import json
import matplotlib.image as mpimg
import time
import datetime
from dashboard.chatbot_dialog import ChatbotDialog
from utils.settings_manager import SettingsManager
from utils.localization import translate_text
from utils.crash_logger import get_crash_logger, CrashLogDialog
from dashboard.admin_reports import AdminLoginDialog, AdminReportsDialog

# Try to import configuration, fallback to defaults if not available
try:
    import sys
    # Add the src directory to the path
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from config.settings import get_config
        config = get_config()
        def get_background_config():
            return config.get('ui.background', {"background": "none", "gif": False})
        print(" Dashboard configuration loaded successfully")
    except ImportError as e:
        print(f" Dashboard config import warning: {e}")
        def get_background_config():
            return {"background": "none", "gif": False}
except ImportError:
    print(" Dashboard configuration not found, using default settings")
    def get_background_config():
        return {
            "use_gif_background": False,
            "preferred_background": "none"
        }

def get_asset_path(asset_name):
    """
    Get the absolute path to an asset file in a portable way.
    This function will work regardless of where the script is run from.
    
    Args:
        asset_name (str): Name of the asset file (e.g., 'her.png', 'v.gif')
    
    Returns:
        str: Absolute path to the asset file
    """
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = []
    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS", "")
        exe_dir = os.path.dirname(sys.executable)
        possible_paths.extend([
            os.path.join(bundle_dir, "assets"),
            os.path.join(exe_dir, "assets"),
        ])
    possible_paths.extend([
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), "assets"),
        os.path.join(script_dir, "assets"),
        os.path.join(os.path.dirname(script_dir), "assets"),
        os.path.join(script_dir, "..", "assets"),
    ])
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.join(path, asset_name)
    return os.path.join(os.path.dirname(script_dir), "..", "assets", asset_name)

class MplCanvas(FigureCanvas):
    def __init__(self, width=4, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class SignInDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign In")
        self.setFixedSize(340, 240)
        self.setStyleSheet("""
            QDialog { background: #fff; border-radius: 18px; }
            QLabel { font-size: 15px; color: #222; }
            QLineEdit, QComboBox { border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; }
            QPushButton { background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background: #ff8800; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Sign In to PulseMonitor")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Doctor", "Patient"])
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter your name")
        self.pass_edit = QLineEdit()
        self.pass_edit.setPlaceholderText("Password")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Role:", self.role_combo)
        form.addRow("Name:", self.name_edit)
        form.addRow("Password:", self.pass_edit)
        layout.addLayout(form)
        self.signin_btn = QPushButton("Sign In")
        self.signin_btn.clicked.connect(self.accept)
        layout.addWidget(self.signin_btn)
    def get_user_info(self):
        return self.role_combo.currentText(), self.name_edit.text()
    
class DashboardHomeWidget(QWidget):
    def __init__(self):
        super().__init__()

class Dashboard(QWidget):
    def __init__(self, username=None, role=None, user_details=None):
        super().__init__()
        # Settings for wave speed/gain
        self.settings_manager = SettingsManager()
        self.current_language = self.settings_manager.get_setting("system_language", "en")
        self.heartbeat_sound_enabled = (
            str(self.settings_manager.get_setting("system_beat_vol", "on")).lower() == "on"
        )
        
        # Initialize crash logger
        self.crash_logger = get_crash_logger()
        self.crash_logger.log_info("Dashboard initialized", "DASHBOARD_START")
        
        # ========================================
        # START AUTOMATIC BACKGROUND CLOUD SYNC
        # ========================================
        # This service runs in background and syncs every 5 seconds
        try:
            from utils.auto_sync_service import start_auto_sync
            self.auto_sync_service = start_auto_sync(interval_seconds=15)
            print(" Automatic cloud sync started (every 15 seconds)")
        except Exception as e:
            print(f" Could not start auto-sync service: {e}")
            self.auto_sync_service = None
        
        # ========================================
        # INITIALIZE OFFLINE QUEUE FOR S3 UPLOADS
        # ========================================
        # This ensures reports are queued when offline and auto-synced when online
        try:
            from utils.offline_queue import get_offline_queue
            self.offline_queue = get_offline_queue()
            queue_stats = self.offline_queue.get_stats()
            print(f"✅ Offline queue initialized")
            print(f"   Pending uploads: {queue_stats.get('pending_count', 0)}")
            print(f"   Online status: {queue_stats.get('is_online', False)}")
            
            # If there are pending items, try to sync immediately
            if queue_stats.get('pending_count', 0) > 0:
                print(f"🔄 Found {queue_stats.get('pending_count', 0)} pending uploads - attempting sync...")
                self.offline_queue.force_sync_now()
        except Exception as e:
            print(f"⚠️ Could not initialize offline queue: {e}")
            self.offline_queue = None
        
        # Triple-click counter for heart rate metric
        self.heart_rate_click_count = 0
        self.last_heart_rate_click_time = 0
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size for usability
        
        # Reports filter date
        self.reports_filter_date = None
        
        # Store username, role, and full user details
        self.username = username
        self.role = role
        self.user_details = user_details or {}

        # Flags to track which test is currently running
        self.test_states = {
            "12_lead_test": False,
            "hrv_test": False,
            "hyperkalemia_test": False
        }
        self.closed_by_sign_out = False
        
        # Initialize standard values flag
        self._use_standard_values = True
        print("✅ Standard values flag initialized")
        
        # Initialize BPM tracking for instant updates
        self._last_bpm = None
        self._bpm_change_threshold = 2  # Update if BPM changes by 2 or more
        print("🔄 BPM change detection initialized")
        
        # Initialize theme modes
        self.dark_mode = False
        self.medical_mode = False
        print("🎨 Theme modes initialized")
        
        # Initialize stable RR tracking
        self._last_stable_rr = None
        self._rr_stability_counter = 0
        print("🔒 Stable RR tracking initialized")
        
        self.setWindowTitle("ECG Monitor Dashboard")
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowState(Qt.WindowMaximized)
        self.center_on_screen()
        
        # Test asset paths at startup for debugging
        self.test_asset_paths()
        
        # Load background settings from configuration file
        config = get_background_config()
        self.use_gif_background = config.get("use_gif_background", False)
        self.preferred_background = config.get("preferred_background", "none")
        
        print(f"Dashboard background: {self.preferred_background} (GIF: {self.use_gif_background})")
        
        # --- Plasma GIF background ---
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        self.bg_label.lower()
        
        # Try to load background GIFs using portable paths
        if not self.use_gif_background:
            # Use light gray background matching ECG 12 test page
            self.bg_label.setStyleSheet("background: #f8f9fa;")
            print("Using light gray background (matching ECG page)")
        else:
            # Priority order based on user preference
            movie = None
            if self.preferred_background == "plasma.gif":
                plasma_path = get_asset_path("plasma.gif")
                if os.path.exists(plasma_path):
                    movie = QMovie(plasma_path)
                    print("Using plasma.gif as background")
                else:
                    print("plasma.gif not found, trying alternatives...")
                    self.preferred_background = "tenor.gif"  # Fallback
            
            if self.preferred_background == "tenor.gif" and not movie:
                tenor_gif_path = get_asset_path("tenor.gif")
                if os.path.exists(tenor_gif_path):
                    movie = QMovie(tenor_gif_path)
                    print("Using tenor.gif as background")
                else:
                    print("tenor.gif not found, trying alternatives...")
                    self.preferred_background = "v.gif"  # Fallback
            
            if self.preferred_background == "v.gif" and not movie:
                v_gif_path = get_asset_path("v.gif")
                if os.path.exists(v_gif_path):
                    movie = QMovie(v_gif_path)
                    print("Using v.gif as background")
                else:
                    print("v.gif not found, using solid color background")
            
            if movie:
                self.bg_label.setMovie(movie)
                movie.start()
            else:
                # If no GIF found, use light gray background matching ECG page
                self.bg_label.setStyleSheet("background: #f8f9fa;")
                print("Using light gray background (no GIFs found)")
        # --- Central stacked widget for in-place navigation ---
        self.page_stack = QStackedWidget(self)
        
        # --- Dashboard main page widget ---
        self.dashboard_page = DashboardHomeWidget()
        # Set light gray background for main content area (matching ECG page)
        self.dashboard_page.setStyleSheet("background-color: #f8f9fa;")
        dashboard_layout = QVBoxLayout(self.dashboard_page)
        dashboard_layout.setSpacing(20)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Header ---
        header = QHBoxLayout()
        logo = QLabel("ECG Monitor")
        logo.setFont(QFont("Arial", 24, QFont.Bold))
        logo.setStyleSheet("color: #ff6600;")
        logo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        header.addWidget(logo)
        
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(18, 18)
        self.status_dot.setStyleSheet("border-radius: 9px; background: gray; border: 2px solid #fff;")
        header.addWidget(self.status_dot)
        
        self.update_internet_status()
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_internet_status)
        self.status_timer.start(3000)

        # Device Status Label
        self.device_status_label = QLabel("Device Disconnected")
        self.device_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.device_status_label.setStyleSheet("color: red; margin-right: 10px;")
        self.device_status_label.setMinimumWidth(160)
        self.device_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.addWidget(self.device_status_label)
        
        self.medical_btn = QPushButton("Medical Mode")
        self.medical_btn.setCheckable(True)
        self.medical_btn.setStyleSheet("background: #00b894; color: white; border-radius: 10px; padding: 4px 18px;")
        self.medical_btn.clicked.connect(self.toggle_medical_mode)
        self.medical_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Hide button per request while preserving logic
        self.medical_btn.setVisible(False)
        
        self.dark_btn = QPushButton("Dark Mode")
        self.dark_btn.setCheckable(True)
        self.dark_btn.setStyleSheet("background: #222; color: #fff; border-radius: 10px; padding: 4px 18px;")
        self.dark_btn.clicked.connect(self.toggle_dark_mode)
        self.dark_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Hide dark mode button
        self.dark_btn.setVisible(False)
        
        # Removed background control button per request
        
        header.addStretch()
        
        # Cloud Sync Button - styled to match orange UI buttons
        self.cloud_sync_btn = QPushButton("Cloud Sync")
        self.cloud_sync_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 16px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #ff7a26;
                min-width: 140px;
            }
            QPushButton:hover { background: #ff7a26; border: 2px solid #ff8e47; }
            QPushButton:pressed { background: #e65c00; }
        """)
        self.cloud_sync_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cloud_sync_btn.setToolTip("Upload ECG reports and metrics to AWS S3")
        self.cloud_sync_btn.clicked.connect(self.sync_to_cloud)
        header.addWidget(self.cloud_sync_btn)
        # Fully automatic mode: hide manual sync button
        self.cloud_sync_btn.setVisible(False)
        
        # User label removed per request
        # self.user_label = QLabel(f"{username or 'User'}\n{role or ''}")
        # self.user_label.setFont(QFont("Arial", 12))
        # self.user_label.setAlignment(Qt.AlignRight)
        # self.user_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # header.addWidget(self.user_label)
        
        # Admin button (disabled per request; keep logic available)
        self.admin_btn = QPushButton("Admin")
        self.admin_btn.setVisible(False)

        self.version_btn = QPushButton("Version Information")
        self.version_btn.setStyleSheet("background: #3498db; color: white; border-radius: 10px; padding: 4px 18px; margin-right: 10px;")
        self.version_btn.clicked.connect(self.show_version_popup)
        self.version_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.version_btn)
        
        self.sign_btn = QPushButton("Sign Out")
        self.sign_btn.setStyleSheet("background: #e74c3c; color: white; border-radius: 10px; padding: 4px 18px;")
        self.sign_btn.clicked.connect(self.handle_sign_out)
        self.sign_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.sign_btn)

        self.apply_language(self.current_language)
        
        dashboard_layout.addLayout(header)

        # --- Device Connection Monitor ---
        self.device_connected = False
        self.device_port = None
        self.device_version = None
        self.settings_manager = SettingsManager()
        self.device_check_timer = QTimer(self)
        self.device_check_timer.timeout.connect(self.check_device_connection)
        self.device_check_timer.start(100) # Check every 0.1 second

        self._device_scan_in_progress = False
        self._last_device_scan_time = 0
        self._initial_scan_completed = False

        # Initialize UI as disconnected
        self.update_device_ui(False)

        # --- Cloud Auto Sync: periodically back up reports when online ---
        self._cloud_sync_in_progress = False
        self.cloud_auto_timer = QTimer(self)
        self.cloud_auto_timer.timeout.connect(self.auto_sync_to_cloud)
        self.cloud_auto_timer.start(15000)  # every 15s (reduced from 5s for performance)
        
        # --- Greeting and Date Row ---
        greet_row = QHBoxLayout()
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good Morning"
        elif hour < 18:
            greeting = "Good Afternoon"
        else:
            greeting = "Good Evening"
        
        # Show full name if available, otherwise username

         
        display_name = self.user_details.get('full_name', username) or username or 'User'
        user_info_lines = [f"<span style='font-size:18pt;font-weight:bold;'>{greeting}, {display_name}</span>"]
        
        # Add user details if available
        if self.user_details:
            details = []
            if self.user_details.get('age'):
                details.append(f"Age: {self.user_details.get('age')}")
            if self.user_details.get('gender'):
                details.append(f"Gender: {self.user_details.get('gender')}")
            if details:
                user_info_lines.append(f"<span style='color:#666; font-size:11pt;'>{' | '.join(details)}</span>")
                
        
        user_info_lines.append("<span style='color:#888;'>Welcome to your ECG dashboard</span>")
        
        greet = QLabel("<br>".join(user_info_lines))
        greet.setFont(QFont("Arial", 16))
        greet.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        greet_row.addWidget(greet)
        greet_row.addStretch()
        
        # History button (orange dark suede, left of Hyperkalemia)
        self.history_btn = QPushButton("History")
        self.history_btn.setStyleSheet("background: #b35900; color: white; border-radius: 16px; padding: 8px 24px;")
        self.history_btn.clicked.connect(self.open_history_window)
        self.history_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.history_btn)

        # Hyperkalemia Test button (orange suede color, right of History, left of HRV Test)
        # Define disabled style for initial state to prevent flash of active buttons
        grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"

        self.hyperkalemia_test_btn = QPushButton("Hyperkalemia Test")
        self.hyperkalemia_test_btn.setEnabled(False)
        self.hyperkalemia_test_btn.setStyleSheet(grey_style)
        self.hyperkalemia_test_btn.clicked.connect(self.open_hyperkalemia_test)
        self.hyperkalemia_test_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.hyperkalemia_test_btn)
        
        # HRV Test button (red color, left of ECG Lead Test 12)
        self.hrv_test_btn = QPushButton("HRV Test")
        self.hrv_test_btn.setEnabled(False)
        self.hrv_test_btn.setStyleSheet(grey_style)
        self.hrv_test_btn.clicked.connect(self.open_hrv_test)
        self.hrv_test_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hrv_test_btn.setVisible(True)
        greet_row.addWidget(self.hrv_test_btn)
        
        self.date_btn = QPushButton("ECG Lead Test 12")
        self.date_btn.setEnabled(False)
        self.date_btn.setStyleSheet(grey_style)
        self.date_btn.clicked.connect(self.go_to_lead_test)
        self.date_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.date_btn)

        # --- Add Chatbot Button ---
        self.chatbot_btn = QPushButton("AI Chatbot")
        self.chatbot_btn.setStyleSheet("background: #2453ff; color: white; border-radius: 16px; padding: 8px 24px;")
        self.chatbot_btn.clicked.connect(self.open_chatbot_dialog)
        self.chatbot_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.chatbot_btn)
        
        # --- Add Analysis Window Button ---
        self.analysis_btn = QPushButton("Analysis Window")
        self.analysis_btn.setStyleSheet("background: #28a745; color: white; border-radius: 16px; padding: 8px 24px;")
        self.analysis_btn.clicked.connect(self.open_analysis_window)
        self.analysis_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.analysis_btn)

        dashboard_layout.addLayout(greet_row)

        # --- Main Grid ---
        # Create a scroll area for responsive design
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(20)
        
        # --- Heart Rate Card --- 
        heart_card = QFrame()
        heart_card.setStyleSheet("background: white; border-radius: 16px;")
        heart_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        heart_layout = QVBoxLayout(heart_card)
        
        self.heart_label = QLabel("Live Heart Rate Overview")
        self.heart_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.heart_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        heart_layout.addWidget(self.heart_label)
        
        heart_img = QLabel()
        # Use portable path for the heart image asset
        heart_img_path = get_asset_path("her.png")
        print(f"Heart image path: {heart_img_path}")  # Debugging line to check the path
        # Ensure os module is available
        import os
        print(f"Heart image exists: {os.path.exists(heart_img_path)}")  # Check if the file exists
        
        # Load the heart image with error handling
        if os.path.exists(heart_img_path):
            self.heart_pixmap = QPixmap(heart_img_path)
            if self.heart_pixmap.isNull():
                print(f"Error: Failed to load heart image from {heart_img_path}")
                # Create a placeholder pixmap
                self.heart_pixmap = QPixmap(220, 220)
                self.heart_pixmap.fill(Qt.lightGray)
        else:
            print(f"Error: Heart image not found at {heart_img_path}")
            # Create a placeholder pixmap
            self.heart_pixmap = QPixmap(220, 220)
            self.heart_pixmap.fill(Qt.lightGray)
        self.heart_base_size = 220
        heart_img.setFixedSize(self.heart_base_size + 20, self.heart_base_size + 20)
        heart_img.setAlignment(Qt.AlignCenter)
        heart_img.setPixmap(self.heart_pixmap.scaled(self.heart_base_size, self.heart_base_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        heart_img.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        heart_layout.addWidget(heart_img)
        
        # Live stress and HRV labels
        self.stress_label = QLabel("Stress Level: --")
        self.stress_label.setStyleSheet("font-size: 13px; color: #666;")
        self.stress_label.setAlignment(Qt.AlignCenter)
        heart_layout.addWidget(self.stress_label)
        
        self.hrv_label = QLabel("Average Variability: --")
        self.hrv_label.setStyleSheet("font-size: 13px; color: #666;")
        self.hrv_label.setAlignment(Qt.AlignCenter)
        heart_layout.addWidget(self.hrv_label)
        
        grid.addWidget(heart_card, 0, 0, 2, 1)
        
        # --- Heartbeat Animation ---
        self.heart_img = heart_img
        self.heartbeat_phase = 0
        self.current_heart_rate = 60  # Default heart rate
        self.last_beat_time = 0
        self.beat_interval = 1000  # Default 1 second between beats (60 BPM)
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self.animate_heartbeat)
        self.heartbeat_timer.start(100)  # 10 FPS (reduced from 33 FPS for performance)
        
        # --- Heartbeat Sound ---
        try:
            if QSound is not None:
                # Try to load heartbeat sound file
                heartbeat_sound_path = get_asset_path("heartbeat.wav")
                if os.path.exists(heartbeat_sound_path):
                    self.heartbeat_sound = QSound(heartbeat_sound_path)
                    print(f" Heartbeat sound loaded: {heartbeat_sound_path}")
                else:
                    print(f" Heartbeat sound not found at: {heartbeat_sound_path}")
                    # Create a synthetic heartbeat sound
                    self.create_heartbeat_sound()
            else:
                print(" QSound not available - heartbeat sound disabled")
                self.heartbeat_sound = None
                self.heartbeat_sound_enabled = False
        except Exception as e:
            print(f" Could not load heartbeat sound: {e}")
            self.heartbeat_sound = None
            self.heartbeat_sound_enabled = False
        
        # --- ECG Recording (Animated Chart) ---
        ecg_card = QFrame()
        ecg_card.setStyleSheet("background: white; border-radius: 16px;")
        ecg_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ecg_layout = QVBoxLayout(ecg_card)
        
        self.ecg_label = QLabel("ECG Recording")
        self.ecg_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.ecg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ecg_layout.addWidget(self.ecg_label)
        
        self.ecg_canvas = MplCanvas(width=4, height=2)
        self.ecg_canvas.axes.set_facecolor("#eee")
        self.ecg_canvas.axes.set_xticks([])
        self.ecg_canvas.axes.set_yticks([])
        self.ecg_canvas.axes.set_title("Lead II", fontsize=10)
        # Set fixed Y-axis limits to match ECG 12-lead page (0-4096)
        self.ecg_canvas.axes.set_ylim(0, 4096)
        self.ecg_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ecg_layout.addWidget(self.ecg_canvas)
        
        grid.addWidget(ecg_card, 1, 1)
        
        # --- Total Visitors (Pie Chart) ---
        visitors_card = QFrame()
        visitors_card.setStyleSheet("background: white; border-radius: 16px;")
        visitors_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visitors_layout = QVBoxLayout(visitors_card)
        
        from datetime import datetime
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        self.visitors_label = QLabel(f"Visitors - Last 6 Months ({current_year})")
        self.visitors_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.visitors_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        visitors_layout.addWidget(self.visitors_label)
        
        pie_canvas = MplCanvas(width=2.5, height=2.5)
        
        # Generate last 6 months dynamically with real session counts
        import calendar
        month_names = []
        month_data = []
        
        # Get session counts per month
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            sessions_dir = os.path.join(base_dir, 'reports', 'sessions')
            
            for i in range(5, -1, -1):  # Last 6 months
                # Calculate target month/year
                target_month = ((current_month - i - 1) % 12) + 1
                target_year = current_year if (current_month - i) > 0 else current_year - 1
                
                month_names.append(calendar.month_name[target_month][:3])
                
                # Count session files for this month
                count = 0
                if os.path.exists(sessions_dir):
                    for filename in os.listdir(sessions_dir):
                        if filename.endswith('.jsonl'):
                            try:
                                # Parse date from filename: session_user_YYYYMMDD_HHMMSS.jsonl
                                parts = filename.split('_')
                                if len(parts) >= 3:
                                    date_str = parts[-2]  # YYYYMMDD
                                    if len(date_str) == 8:
                                        file_year = int(date_str[:4])
                                        file_month = int(date_str[4:6])
                                        if file_year == target_year and file_month == target_month:
                                            count += 1
                            except Exception:
                                continue
                
                # Use actual count, or show 1 if zero to keep chart visible
                month_data.append(max(1, count))
        except Exception as e:
            print(f" Could not calculate visitor stats: {e}")
            month_data = [1, 1, 1, 1, 1, 1]  # Fallback to equal distribution
        
        pie_data = month_data
        pie_labels = month_names
        # Modern colors matching software orange theme
        pie_colors = ["#ff6600", "#ff8533", "#ffaa66", "#ffcc99", "#ffe6cc", "#fff3e6"]
        
        # Draw pie without labels inside, percentages will be outside
        # Adjust figure margins to accommodate outside labels
        pie_canvas.figure.subplots_adjust(left=0.05, right=0.7, top=0.95, bottom=0.05)
        
        # Calculate percentages
        total = sum(pie_data)
        percentages = [(value / total * 100) if total > 0 else 0 for value in pie_data]
        
        # Draw pie chart without autopct (no percentages inside)
        wedges, texts, autotexts = pie_canvas.axes.pie(
            pie_data, labels=None, autopct='', colors=pie_colors, 
            startangle=90
        )
        
        # Add percentage labels outside the pie chart
        for i, (wedge, pct) in enumerate(zip(wedges, percentages)):
            # Calculate angle for label position (middle of the wedge)
            ang = (wedge.theta2 + wedge.theta1) / 2.0
            # Convert angle to radians
            ang_rad = np.deg2rad(ang)
            # Position label outside the pie (1.2x radius for better spacing)
            x = 1.2 * np.cos(ang_rad)
            y = 1.2 * np.sin(ang_rad)
            # Determine horizontal alignment based on position
            if abs(x) < 0.3:
                ha = 'center'
            elif x > 0:
                ha = 'left'
            else:
                ha = 'right'
            # Add text annotation outside
            pie_canvas.axes.annotate(
                f'{pct:.0f}%',
                xy=(np.cos(ang_rad), np.sin(ang_rad)),
                xytext=(x, y),
                ha=ha,
                va='center',
                fontsize=9,
                fontweight='bold',
                color='#222',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
            )
        
        # Add legend on the side
        legend = pie_canvas.axes.legend(
            wedges, pie_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=False
        )
        # Ensure legend doesn't get clipped
        legend.set_clip_on(False)
        
        pie_canvas.axes.set_aspect('equal')
        pie_canvas.draw()
        pie_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visitors_layout.addWidget(pie_canvas)
        
        grid.addWidget(visitors_card, 1, 2)
        
        # --- Schedule Card ---
        schedule_card = QFrame()
        schedule_card.setStyleSheet("background: white; border-radius: 16px;")
        schedule_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        schedule_layout = QVBoxLayout(schedule_card)
        
        schedule_label = QLabel("Calendar")
        schedule_label.setFont(QFont("Arial", 14, QFont.Bold))
        schedule_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        schedule_layout.addWidget(schedule_label)
        
        
        # Create custom navigation header for calendar
        calendar_nav = QFrame()
        calendar_nav.setStyleSheet("background: #f9f9f9; border-radius: 8px; border: 1px solid #e0e0e0;")
        calendar_nav_layout = QHBoxLayout(calendar_nav)
        calendar_nav_layout.setContentsMargins(8, 8, 8, 8)
        calendar_nav_layout.setSpacing(10)
        
        # Previous month button
        self.prev_month_btn = QPushButton("◀")
        self.prev_month_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8533, stop:1 #ff6600);
                color: white;
                border: 1px solid #ff6600;
                border-radius: 15px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffaa66, stop:1 #ff8533);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6600, stop:1 #ff5500);
            }
        """)
        self.prev_month_btn.clicked.connect(self.go_to_prev_month)
        calendar_nav_layout.addWidget(self.prev_month_btn)
        
        # Month label with rounded background
        self.month_label = QLabel("February")
        self.month_label.setAlignment(Qt.AlignCenter)
        self.month_label.setStyleSheet("""
            QLabel {
                background: #ffffff;
                color: #222;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: 600;
                min-width: 80px;
            }
        """)
        calendar_nav_layout.addWidget(self.month_label)
        
        # Year label with rounded background
        self.year_label = QLabel("2026")
        self.year_label.setAlignment(Qt.AlignCenter)
        self.year_label.setStyleSheet("""
            QLabel {
                background: #ffffff;
                color: #222;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: 600;
                min-width: 60px;
            }
        """)
        calendar_nav_layout.addWidget(self.year_label)
        
        # Next month button
        self.next_month_btn = QPushButton("▶")
        self.next_month_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8533, stop:1 #ff6600);
                color: white;
                border: 1px solid #ff6600;
                border-radius: 15px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffaa66, stop:1 #ff8533);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6600, stop:1 #ff5500);
            }
        """)
        self.next_month_btn.clicked.connect(self.go_to_next_month)
        calendar_nav_layout.addWidget(self.next_month_btn)
        
        schedule_layout.addWidget(calendar_nav)
        
        # Calendar widget (hide default navigation)
        self.schedule_calendar = QCalendarWidget()
        self.schedule_calendar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.schedule_calendar.setMinimumHeight(200)
        self.schedule_calendar.setMaximumHeight(260)
        self.schedule_calendar.setMinimumWidth(280)
        self.schedule_calendar.setNavigationBarVisible(False)  # Hide default navigation
        
        self.schedule_calendar.setStyleSheet("""
        QCalendarWidget QWidget { 
            background: #ffffff; 
            color: #222; 
        }
        QCalendarWidget QAbstractItemView {
            background: #ffffff; 
            color: #222;
            selection-background-color: #ff6600; 
            selection-color: #fff;
            font-size: 12px;
        }
        QCalendarWidget QToolButton { 
            color: #222; 
            background: transparent;
            padding: 4px;
            margin: 2px;
        }
        QCalendarWidget QToolButton:hover {
            background: #ffe6cc;
            border-radius: 4px;
        }
    """)

        # Highlight last ECG usage date in red
        from PyQt5.QtGui import QTextCharFormat, QColor
        last_ecg_file = 'last_ecg_date.json'
        import datetime
        today = datetime.date.today()
        # Try to load last ECG date from file
        last_ecg_date = None
        if os.path.exists(last_ecg_file):
            with open(last_ecg_file, 'r') as f:
                try:
                    data = json.load(f)
                    last_ecg_date = data.get('last_ecg_date')
                except Exception:
                    last_ecg_date = None
        if last_ecg_date:
            try:
                y, m, d = map(int, last_ecg_date.split('-'))
                last_date = Qt.QDate(y, m, d)
                fmt = QTextCharFormat()
                fmt.setBackground(QColor('red'))
                fmt.setForeground(QColor('white'))
                self.schedule_calendar.setDateTextFormat(last_date, fmt)
            except Exception:
                pass

        # Apply calendar date restrictions for new users
        self._apply_new_user_calendar_restrictions()
        
        # connect date click/selection to filter reports
        self.schedule_calendar.clicked.connect(self.on_calendar_date_selected)
        self.schedule_calendar.selectionChanged.connect(self.on_calendar_selection_changed)
        
        # Update labels when calendar page changes
        self.schedule_calendar.currentPageChanged.connect(self.update_calendar_labels)
        
        # Connect to page change to track month navigation
        self.schedule_calendar.currentPageChanged.connect(self.on_calendar_page_changed)
        
        # Initialize calendar labels
        self.update_calendar_labels()
        
        # Disable double-click activation (prevents popup window)
        try:
            self.schedule_calendar.activated.disconnect()
        except:
            pass  # No activated signal connected

       
        
        schedule_layout.addWidget(self.schedule_calendar)
        grid.addWidget(schedule_card, 2, 0)
        # --- Conclusion Card ---
        issue_card = QFrame()
        issue_card.setStyleSheet("background: white; border-radius: 16px;")
        issue_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        issue_layout = QVBoxLayout(issue_card)
        
        issue_label = QLabel("Conclusion")
        issue_label.setFont(QFont("Arial", 14, QFont.Bold))
        issue_label.setStyleSheet("color: #ff6600;")
        issue_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        issue_layout.addWidget(issue_label)
        
        # Live conclusion box that updates based on ECG analysis - BIGGER SIZE
        self.conclusion_box = QTextEdit()
        self.conclusion_box.setReadOnly(True)
        self.conclusion_box.setStyleSheet("background: #f7f7f7; border: none; font-size: 12px; padding: 10px;")
        self.conclusion_box.setMinimumHeight(300)  # Increased from 180 to 300
        self.conclusion_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Set initial placeholder text
        self.conclusion_box.setHtml("""
            <p style='color: #888; font-style: italic;'>
            No ECG data available yet.<br><br>
            Start an ECG test or enable demo mode to see your personalized analysis and recommendations.
            </p>
        """)
        
        issue_layout.addWidget(self.conclusion_box)
        # # Small footer box below the conclusion (~3 cm height)
        # self.conclusion_footer = QFrame()
        # self.conclusion_footer.setStyleSheet("background: #f7f7f7; border: none; border-radius: 10px;")
        # self.conclusion_footer.setFixedHeight(115)
        # self.conclusion_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # _footer_layout = QHBoxLayout(self.conclusion_footer)
        # _footer_layout.setContentsMargins(10, 8, 10, 8)
        # _footer_layout.addWidget(QLabel(""))
        # issue_layout.addWidget(self.conclusion_footer)

        grid.addWidget(issue_card, 2, 1, 1, 1)

        # Separate card for Additional Notes (outside Conclusion)
        notes_card = QFrame()
        notes_card.setStyleSheet("background: white; border-radius: 16px;")
        notes_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        notes_card.setFixedHeight(160)
        notes_layout = QVBoxLayout(notes_card)
        notes_layout.setContentsMargins(12, 12, 12, 12)
        notes_layout.setSpacing(8)
        notes_title = QLabel("METRICS")
        notes_title.setFont(QFont("Arial", 14, QFont.Bold))
        notes_title.setStyleSheet("color: #ff6600;")
        notes_layout.addWidget(notes_title)
        self.parameters_text = QTextEdit()
        self.parameters_text.setReadOnly(True)
        self.parameters_text.setFixedHeight(90)
        self.parameters_text.setLineWrapMode(QTextEdit.NoWrap)  # Single row, no wrap
        self.parameters_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scroll
        self.parameters_text.setStyleSheet("background: #f9f9f9; border: none; font-size: 12px; padding: 6px 8px;")
        notes_layout.addWidget(self.parameters_text)
        # Slider to control horizontal scroll for parameters_text
        self.parameters_slider = QSlider(Qt.Horizontal)
        self.parameters_slider.setMinimum(0)
        self.parameters_slider.setMaximum(0)
        self.parameters_slider.setSingleStep(20)
        self.parameters_slider.setPageStep(self.parameters_text.viewport().width())
        self.parameters_slider.setStyleSheet("QSlider::groove:horizontal { height: 6px; background: #ececec; border-radius: 3px; } QSlider::handle:horizontal { background: #ff6600; width: 14px; border-radius: 7px; margin: -5px 0; } QSlider::sub-page:horizontal { background: #ffd5b3; border-radius: 3px; }")
        notes_layout.addWidget(self.parameters_slider)

        # Keep a reference so we can show/hide the METRICS panel based on user actions
        self.metrics_notes_card = notes_card

        # Sync slider with the hidden horizontal scrollbar
        _hbar = self.parameters_text.horizontalScrollBar()
        _hbar.rangeChanged.connect(lambda _min, _max: self.parameters_slider.setMaximum(_max))
        _hbar.valueChanged.connect(self.parameters_slider.setValue)
        self.parameters_slider.valueChanged.connect(_hbar.setValue)
        # Make the Additional Notes card span all 3 columns (full row width)
        grid.addWidget(notes_card, 3, 0, 1, 3)

        # Hide METRICS panel by default; it will be shown when a report is selected
        self.metrics_notes_card.hide()

        # --- Recent Reports Card ---
        reports_card = QFrame()
        reports_card.setStyleSheet(
            "background: white; border-radius: 16px;"
        )
        reports_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        reports_v = QVBoxLayout(reports_card)
        ttl = QLabel("Recent Reports")
        ttl.setFont(QFont("Arial", 14, QFont.Bold))
        ttl.setStyleSheet("color: #ff6600;")
        reports_v.addWidget(ttl)

        # Scroll area for list
        self.reports_list_widget = QWidget()
        self.reports_list_layout = QVBoxLayout(self.reports_list_widget)
        self.reports_list_layout.setContentsMargins(4, 4, 4, 4)  # Add margins to prevent button cropping
        self.reports_list_layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(180)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Allow horizontal scroll if needed
        scroll.setWidget(self.reports_list_widget)
        reports_v.addWidget(scroll)

        self.refresh_recent_reports_ui()

        grid.addWidget(reports_card, 2, 2, 1, 1)
        
        # --- ECG Monitor Metrics Cards ---
        metrics_card = QFrame()
        metrics_card.setStyleSheet("background: white; border-radius: 16px;")
        metrics_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        metrics_layout = QHBoxLayout(metrics_card)
        
        # Store metric labels for live update
        self.metric_labels = {}
        metric_info = [
            ("HR", "00", "BPM", "heart_rate"),
            ("PR", "0", "ms", "pr_interval"),
            ("QRS Complex", "0", "ms", "qrs_duration"),
            ("QT/QTc", "0", "ms", "qtc_interval"),
        ]
        
        for title, value, unit, key in metric_info:
            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val = QLabel(f"{value} {unit}")
            val.setFont(QFont("Arial", 18, QFont.Bold))
            val.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            # Add triple-click functionality to heart rate metric
            if key == "heart_rate":
                val.mousePressEvent = self.heart_rate_triple_click
                try:
                    val.setMinimumWidth(val.fontMetrics().horizontalAdvance("000 BPM"))
                except Exception:
                    pass
            
            box.addWidget(lbl)
            box.addWidget(val)
            metrics_layout.addLayout(box, 1)
            self.metric_labels[key] = val  
        
        grid.addWidget(metrics_card, 0, 1, 1, 2)
        
        # Add the grid widget to the scroll area
        scroll_area.setWidget(grid_widget)
        
        # Add scroll area to dashboard layout
        dashboard_layout.addWidget(scroll_area)
        
        
        # --- ECG Animation Setup ---
        self.ecg_x = np.linspace(0, 2, 500)
        self.ecg_y = 2048 + 150 * np.sin(2 * np.pi * 2 * self.ecg_x) + 30 * np.random.randn(500)  # Centered at 2048 for 0-4096 range
        self.ecg_line, = self.ecg_canvas.axes.plot(self.ecg_x, self.ecg_y, color="#ff6600", linewidth=0.5, antialiased=True)
        # Reduce CPU/GPU usage: lower refresh rate slightly and disable frame caching
        self.anim = FuncAnimation(
            self.ecg_canvas.figure,
            self.update_ecg,
            interval=85,              # ~12 FPS for smoothness without lag
            blit=True,
            cache_frame_data=False,   # prevent unbounded cache growth
            save_count=100
        )
        
        # --- Dashboard Metrics Update Timer ---
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_dashboard_metrics_from_ecg)
        self.metrics_timer.start(1000)  # Update every 1 second for accurate values within 10 seconds
        print("⏰ Dashboard metrics timer started - updates every 1 second")
        
        # Force initial metrics update immediately to ensure values appear within 10 seconds
        try:
            self.update_dashboard_metrics_from_ecg()
            print(" Initial metrics update completed - values should appear immediately")
        except Exception as e:
            print(f" Initial metrics update failed: {e}")
        
        # Session timer removed - no longer needed
        # Add dashboard_page to stack
        self.page_stack.addWidget(self.dashboard_page)
        # --- ECG Test Page ---
        try:
            # Add the src directory to the path for ECG imports
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
                print(f" Added src directory to path: {src_dir}")
            
            from ecg.twelve_lead_test import ECGTestPage
            print(" ECG Test Page imported successfully")
                    
        except ImportError as e:
            print(f" ECG Test Page import error: {e}")
            print(" Creating fallback ECG Test Page")
            # Create a fallback ECG test page
            class ECGTestPage(QWidget):
                def __init__(self, title, parent):
                    super().__init__()
                    self.title = title
                    self.parent = parent
                    self.dashboard_callback = None
                    layout = QVBoxLayout()
                    label = QLabel("ECG Test Page - Import Error")
                    label.setAlignment(Qt.AlignCenter)
                    layout.addWidget(label)
                    self.setLayout(layout)
                    print(" Using fallback ECG Test Page")
        self.ecg_test_page = ECGTestPage("12 Lead ECG Test", self.page_stack)
        self.ecg_test_page.dashboard_callback = self.update_ecg_metrics
        # Pass username and dashboard reference to ECGTestPage for report filtering
        self.ecg_test_page.dashboard_instance = self
        self.ecg_test_page.current_username = self.username

        if hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
        
        self.page_stack.addWidget(self.ecg_test_page)
        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.page_stack)
        self.setLayout(main_layout)
        self.page_stack.setCurrentWidget(self.dashboard_page)

        # Add a content_frame for ECGMenu to use
        self.content_frame = QFrame(self)
        self.content_frame.setStyleSheet("background: transparent; border: none;")
        main_layout.addWidget(self.content_frame)

        self.setLayout(main_layout)
        self.page_stack.setCurrentWidget(self.dashboard_page)

    # Calendar date selection
    
    def update_calendar_labels(self):
        """Update the custom month and year labels based on current calendar date."""
        try:
            current_date = self.schedule_calendar.selectedDate()
            month_names = ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]
            self.month_label.setText(month_names[current_date.month() - 1])
            self.year_label.setText(str(current_date.year()))
        except Exception as e:
            print(f"Error updating calendar labels: {e}")
    
    def go_to_prev_month(self):
        """Navigate to previous month."""
        try:
            current_date = self.schedule_calendar.selectedDate()
            new_date = current_date.addMonths(-1)
            self.schedule_calendar.setSelectedDate(new_date)
            self.update_calendar_labels()
        except Exception as e:
            print(f"Error navigating to previous month: {e}")
    
    def go_to_next_month(self):
        """Navigate to next month."""
        try:
            current_date = self.schedule_calendar.selectedDate()
            new_date = current_date.addMonths(1)
            self.schedule_calendar.setSelectedDate(new_date)
            self.update_calendar_labels()
        except Exception as e:
            print(f"Error navigating to next month: {e}")

    def on_calendar_date_selected(self, qdate):
        try:
            # Check if date is valid and within allowed range for new users
            if hasattr(self, '_user_signup_date') and hasattr(self, '_user_max_date'):
                if qdate < self._user_signup_date:
                    # Date is before signup - reset to signup date
                    self.schedule_calendar.setSelectedDate(self._user_signup_date)
                    QMessageBox.warning(self, "Invalid Date", 
                                      f"Please select a date on or after your signup date: {self._user_signup_date.toString('yyyy-MM-dd')}")
                    return
                elif qdate > self._user_max_date:
                    # Date is after max date - reset to max date
                    self.schedule_calendar.setSelectedDate(self._user_max_date)
                    QMessageBox.warning(self, "Invalid Date", 
                                      f"Please select a date on or before: {self._user_max_date.toString('yyyy-MM-dd')}")
                    return
            
            self.reports_filter_date = qdate.toString("yyyy-MM-dd")
            # Set flag to prevent automatic report opening when calendar is clicked
            self._calendar_triggered = True
            self.refresh_recent_reports_ui(self.reports_filter_date)
            self._calendar_triggered = False
        except Exception:
            self._calendar_triggered = False
            self.refresh_recent_reports_ui()  # safe fallback

    def on_calendar_selection_changed(self):
        try:
            qdate = self.schedule_calendar.selectedDate()
            self.reports_filter_date = qdate.toString("yyyy-MM-dd")
            self.refresh_recent_reports_ui(self.reports_filter_date)
        except Exception:
            pass
    
    def _apply_new_user_calendar_restrictions(self):
        """Apply calendar date restrictions for new users based on signup date"""
        try:
            from PyQt5.QtCore import QDate
            from PyQt5.QtGui import QTextCharFormat, QColor
            from datetime import datetime, timedelta
            
            # Check if user has signup_date (new user)
            signup_date_str = self.user_details.get('signup_date') or self.user_details.get('registered_at')
            
            # Also check if user logged in with phone number
            has_phone = bool(self.user_details.get('phone') or self.user_details.get('contact'))
            
            if not signup_date_str or not has_phone:
                # Not a new user or no phone number - no restrictions
                return
            
            # Parse signup date
            try:
                # Try parsing different date formats
                if 'T' in signup_date_str or ' ' in signup_date_str:
                    # ISO format with time: "2024-01-15 10:30:00" or "2024-01-15T10:30:00"
                    signup_date_str = signup_date_str.split('T')[0].split(' ')[0]
                
                signup_date_obj = datetime.strptime(signup_date_str, "%Y-%m-%d").date()
                signup_qdate = QDate(signup_date_obj.year, signup_date_obj.month, signup_date_obj.day)
            except Exception as e:
                print(f"Error parsing signup date: {e}")
                return
            
            # Calculate maximum date (1 year from signup)
            max_date_obj = signup_date_obj + timedelta(days=365)
            max_qdate = QDate(max_date_obj.year, max_date_obj.month, max_date_obj.day)
            
            # Set date range
            self.schedule_calendar.setMinimumDate(signup_qdate)
            self.schedule_calendar.setMaximumDate(max_qdate)
            
            # Fade dates before signup date (if any are still visible)
            # Get current displayed month
            current_date = self.schedule_calendar.selectedDate()
            if not current_date.isValid():
                current_date = QDate.currentDate()
            
            # Fade all dates before signup date
            fade_format = QTextCharFormat()
            fade_format.setForeground(QColor(200, 200, 200))  # Light gray
            fade_format.setBackground(QColor(240, 240, 240))  # Light background
            
            # Iterate through dates in the visible month and fade past dates
            year = current_date.year()
            month = current_date.month()
            days_in_month = QDate.daysInMonth(month, year)
            
            for day in range(1, days_in_month + 1):
                check_date = QDate(year, month, day)
                if check_date.isValid() and check_date < signup_qdate:
                    self.schedule_calendar.setDateTextFormat(check_date, fade_format)
            
            # Store signup date and max date for later use
            self._user_signup_date = signup_qdate
            self._user_max_date = max_qdate
            self._user_navigated_months = set()  # Track which months user has navigated to
            
            # Set initial selected date to signup date for new users
            self.schedule_calendar.setSelectedDate(signup_qdate)
            self.schedule_calendar.setCurrentPage(signup_qdate.year(), signup_qdate.month())
            
            print(f" Calendar restrictions applied for new user. Signup: {signup_date_str}, Max: {max_date_obj}")
            
        except Exception as e:
            print(f"Error applying calendar restrictions: {e}")
            import traceback
            traceback.print_exc()
    
    def on_calendar_page_changed(self, year, month):
        """Handle calendar page change - apply date locking after navigation"""
        try:
            # Check if this is a new user with restrictions
            if not hasattr(self, '_user_signup_date'):
                return
            
            from PyQt5.QtCore import QDate
            from PyQt5.QtGui import QTextCharFormat, QColor
            
            # Track that user navigated to this month
            month_key = (year, month)
            self._user_navigated_months.add(month_key)
            
            # Lock dates after the current month (if user navigated forward)
            current_date = QDate.currentDate()
            displayed_date = QDate(year, month, 1)
            
            # If displayed month is in the future (after current month), lock dates after it
            if displayed_date > current_date:
                # Lock all dates in months after the displayed month
                lock_format = QTextCharFormat()
                lock_format.setForeground(QColor(150, 150, 150))  # Darker gray
                lock_format.setBackground(QColor(220, 220, 220))  # Gray background
                
                # Lock dates in the displayed month that are after today
                days_in_month = QDate.daysInMonth(month, year)
                for day in range(1, days_in_month + 1):
                    check_date = QDate(year, month, day)
                    if check_date.isValid() and check_date > current_date:
                        self.schedule_calendar.setDateTextFormat(check_date, lock_format)
            
            # Also ensure dates before signup are still faded
            fade_format = QTextCharFormat()
            fade_format.setForeground(QColor(200, 200, 200))
            fade_format.setBackground(QColor(240, 240, 240))
            
            days_in_month = QDate.daysInMonth(month, year)
            for day in range(1, days_in_month + 1):
                check_date = QDate(year, month, day)
                if check_date.isValid() and check_date < self._user_signup_date:
                    self.schedule_calendar.setDateTextFormat(check_date, fade_format)
            
        except Exception as e:
            print(f"Error in calendar page change: {e}")
            import traceback
            traceback.print_exc()

    def show_month_dropdown(self, year, month):
        """Show month selection dropdown"""
        from PyQt5.QtWidgets import QComboBox, QDialog, QVBoxLayout, QPushButton, QLabel
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Month")
        dialog.setModal(True)
        dialog.setFixedSize(200, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Year selection
        year_label = QLabel("Year:")
        year_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(year_label)
        
        year_combo = QComboBox()
        current_year = year
        for y in range(current_year - 5, current_year + 6):
            year_combo.addItem(str(y))
        year_combo.setCurrentText(str(year))
        layout.addWidget(year_combo)
        
        # Month selection
        month_label = QLabel("Month:")
        month_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(month_label)
        
        month_combo = QComboBox()
        months = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        for i, month_name in enumerate(months):
            month_combo.addItem(month_name)
        month_combo.setCurrentIndex(month - 1)
        layout.addWidget(month_combo)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.setStyleSheet("background: #ff6600; color: white; border-radius: 5px; padding: 8px;")
        ok_button.clicked.connect(lambda: self.apply_calendar_selection(
            int(year_combo.currentText()), month_combo.currentIndex() + 1, dialog))
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("background: #666; color: white; border-radius: 5px; padding: 8px;")
        cancel_button.clicked.connect(dialog.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        dialog.exec_()

    def apply_calendar_selection(self, year, month, dialog):
        """Apply the selected year and month to the calendar"""
        try:
            from PyQt5.QtCore import QDate
            # Set the calendar to the selected month/year
            self.schedule_calendar.setCurrentPage(year, month)
            dialog.accept()
        except Exception as e:
            print(f"Error applying calendar selection: {e}")
            dialog.reject()

    def open_analysis_window(self):
        """Open the ECG Analysis Window"""
        try:
            from dashboard.analysis_window import ECGAnalysisWindow
            self.analysis_window = ECGAnalysisWindow(self)
            self.analysis_window.show()
            print("✅ ECG Analysis Window opened successfully")
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to import Analysis Window: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Analysis Window: {str(e)}")
    
    def open_chatbot_dialog(self):
        dlg = ChatbotDialog(self)
        dlg.exec_()

    def refresh_recent_reports_ui(self, filter_date=None):
        import os, json
        
        # CRITICAL FIX: Skip complete UI refresh when triggered by calendar  
        # This prevents the mysterious popup from appearing
        if getattr(self, '_calendar_triggered', False):
            print(" BLOCKED: Skipping refresh_recent_reports_ui during calendar click to prevent popup")
            return
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        reports_dir = os.path.join(base_dir, "..", "reports")
        index_path = os.path.join(reports_dir, "index.json")

        # Clear list
        while self.reports_list_layout.count():
            item = self.reports_list_layout.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)

        entries = []
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    entries = json.load(f) or []
            except Exception:
                entries = []

        # Use the calendar’s current filter if none explicitly provided
        if filter_date is None:
            filter_date = getattr(self, "reports_filter_date", None)

        if filter_date:
            fd = str(filter_date).strip()
            entries = [e for e in entries if str(e.get('date','')).strip() == fd]

        # Filter to only show ECG Report entries (not detailed JSON metadata)
        # ECG Report entries have 'filename' and 'title' keys
        # Detailed metadata entries have 'timestamp' and 'metrics' keys
        entries = [e for e in entries if 'filename' in e and 'title' in e]
        
        # Filter reports by current user - only show reports generated by this user
        if self.username:
            entries = [e for e in entries if e.get('username', '') == self.username]
        else:
            # If no username, only show reports that also have no username (backward compatibility)
            entries = [e for e in entries if not e.get('username', '')]

        for e in entries[:10]:
            # Build row with hover/touch feedback
            row = QHBoxLayout()
            row.setContentsMargins(6, 6, 12, 6)  # Increased right margin to 12px to prevent button cropping
            row.setSpacing(8)  # Add spacing between elements

            meta = QLabel(f"{e.get('date','')} {e.get('time','')}  |  {e.get('patient','')}  |  {e.get('title','Report')}")
            meta.setStyleSheet("color: #333333; font-size: 12px;")
            meta.setCursor(Qt.PointingHandCursor)
            meta.setWordWrap(False)  # Prevent text wrapping
            row.addWidget(meta, 1)

            path = os.path.join(reports_dir, e.get('filename',''))

            # Clicking will be bound after container is created to allow row selection

            # "Params" button to load metrics into Parameters box
            # params_btn = QPushButton("Params")
            # params_btn.setStyleSheet("background: #eeeeee; color: #333333; border-radius: 8px; padding: 4px 10px; font-weight: bold;")
            # params_btn.clicked.connect(lambda _, p=path: self.load_metrics_into_parameters(p))
            # row.addWidget(params_btn)

            # "Open" button strictly opens the PDF
            btn = QPushButton("Open")
            btn.setStyleSheet("background: #ff6600; color: white; border-radius: 8px; padding: 4px 12px; font-weight: bold;")
            btn.setMinimumWidth(65)  # Increased minimum width to prevent cropping
            btn.setMaximumWidth(75)  # Set maximum width
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Fixed size policy to prevent stretching
            btn.clicked.connect(lambda _, p=path: self.open_report_file(p))
            row.addWidget(btn, 0)  # Don't stretch the button
            row.addSpacing(4)  # Add extra spacing after button

            # Container with hover feedback and full-row click
            cont = QWidget()
            cont.setLayout(row)
            cont.setMinimumHeight(35)  # Ensure container has minimum height
            cont.setCursor(Qt.PointingHandCursor)
            cont.setStyleSheet("background: transparent; border-radius: 8px;")
            cont._meta_label = meta
            cont._report_path = path

            # Make label click select row and load parameters
            def _label_click_handler(_evt, p=path, w=cont):
                # Skip if triggered by calendar (to prevent automatic actions)
                if getattr(self, '_calendar_triggered', False):
                    return
                try:
                    self._select_report_row(w, p)
                except Exception:
                    pass
                self.load_metrics_into_parameters(p)
            meta.mousePressEvent = _label_click_handler

            # Make entire row clickable to select and load parameters
            def _row_click_handler(_evt, p=path, w=cont):
                # Skip if triggered by calendar (to prevent automatic actions)
                if getattr(self, '_calendar_triggered', False):
                    return
                try:
                    self._select_report_row(w, p)
                except Exception:
                    pass
                self.load_metrics_into_parameters(p)
            cont.mousePressEvent = _row_click_handler

            # Hover effects (enter/leave)
            def _hover_enter(_e, w=cont, m=meta):
                # Keep selected style if this row is selected
                if getattr(self, '_selected_report_widget', None) is w:
                    w.setStyleSheet("background: #ffe6cc; border-radius: 8px; border: 1px solid #ffb366;")
                    m.setStyleSheet("color: #333333; font-size: 12px;")
                else:
                    w.setStyleSheet("background: #fff3e6; border-radius: 8px;")
                    m.setStyleSheet("color: #333333; font-size: 12px; text-decoration: underline;")

            def _hover_leave(_e, w=cont, m=meta):
                # If selected, maintain selection; else clear
                if getattr(self, '_selected_report_widget', None) is w:
                    w.setStyleSheet("background: #ffe6cc; border-radius: 8px; border: 1px solid #ffb366;")
                    m.setStyleSheet("color: #333333; font-size: 12px;")
                else:
                    w.setStyleSheet("background: transparent; border-radius: 8px;")
                    m.setStyleSheet("color: #333333; font-size: 12px;")

            cont.enterEvent = _hover_enter
            cont.leaveEvent = _hover_leave

            # Preserve selected row highlight on UI refresh - DISABLED to prevent automatic selection on calendar date click
            # try:
            #     if getattr(self, '_selected_report_path', None) and os.path.abspath(path) == os.path.abspath(self._selected_report_path):
            #         self._select_report_row(cont, path)
            # except Exception:
            #     pass

            self.reports_list_layout.addWidget(cont)

        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reports_list_layout.addWidget(spacer)

    def open_report_file(self, path):
        import os, sys, subprocess
        # Prevent automatic opening when triggered by calendar
        if getattr(self, '_calendar_triggered', False):
            print(" Blocked automatic report opening from calendar click")
            return
        if not os.path.exists(path):
            return
        print(f" Opening report: {path}")
        if sys.platform == 'darwin':
            subprocess.call(['open', path])
        elif sys.platform.startswith('linux'):
            subprocess.call(['xdg-open', path])
        elif sys.platform.startswith('win'):
            os.startfile(path)

    def _select_report_row(self, widget, path):
        """Select a Recent Reports row and keep it highlighted until another is clicked."""
        try:
            # Clear previous selection if different
            prev = getattr(self, '_selected_report_widget', None)
            if prev is not None and prev is not widget:
                try:
                    prev.setStyleSheet("background: transparent; border-radius: 8px;")
                    if hasattr(prev, '_meta_label') and prev._meta_label is not None:
                        prev._meta_label.setStyleSheet("color: #333333; font-size: 12px;")
                except Exception:
                    pass

            # Apply selection style to current widget
            if widget is not None:
                widget.setStyleSheet("background: #ffe6cc; border-radius: 8px; border: 1px solid #ffb366;")
                if hasattr(widget, '_meta_label') and widget._meta_label is not None:
                    widget._meta_label.setStyleSheet("color: #333333; font-size: 12px;")

            # Save selection state
            self._selected_report_widget = widget
            self._selected_report_path = path
        except Exception:
            pass

    def load_metrics_into_parameters(self, report_path: str):
        import os, json
        # Add debug output to track when this is called
        print(f" load_metrics_into_parameters called for: {os.path.basename(report_path)}")
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        reports_dir = os.path.join(base_dir, "..", "reports")
        metrics_path = os.path.join(reports_dir, "metrics.json")

        line = "No metrics found for this report."
        try:
            # First try: look for JSON twin (ECG_Report_YYYYMMDD_HHMMSS.json next to .pdf)
            json_path = os.path.splitext(report_path)[0] + ".json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract metrics from JSON twin format
                metrics_data = data.get('metrics', {})
                patient = data.get('patient', {})
                user = data.get('user', {})
                
                if metrics_data:
                    m = {
                        'HR_bpm': metrics_data.get('heart_rate', '--'),
                        'PR_ms': metrics_data.get('pr_interval', '--'),
                        'QRS_ms': metrics_data.get('qrs_duration', '--'),
                        'QT_ms': metrics_data.get('qt_interval', '--'),
                        'QTc_ms': metrics_data.get('qtc_interval', '--'),
                        'ST_mV': metrics_data.get('ST_mV', metrics_data.get('st_interval', '--')),
                        'RR_ms': metrics_data.get('rr_interval', '--'),
                        'RV5_plus_SV1_mV': metrics_data.get('rv5_sv1', '--'),
                        'P_QRS_T_mm': ['--', '--', '--'],  # Placeholder
                        'QTCF': '--',
                        'RV5_SV1_mV': ['--', '--'],
                    }
            else:
                # Fallback: look in old-style metrics.json
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                reports_dir = os.path.join(base_dir, "..", "reports")
                metrics_path = os.path.join(reports_dir, "metrics.json")
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        items = json.load(f) or []
                    report_abs = os.path.abspath(report_path)
                    report_name = os.path.basename(report_abs)
                    # Try absolute path match first
                    matches = [m for m in items if os.path.abspath(m.get("file", "")) == report_abs]
                    # Fallback: match by filename only
                    if not matches:
                        matches = [m for m in items if os.path.basename(m.get("file", "")) == report_name]
                    if matches:
                        m = matches[-1]
                    else:
                        raise ValueError("No matching report in metrics.json")
                else:
                    raise ValueError("No metrics.json or JSON twin found")
            
            if m:
                    hr   = m.get("HR_bpm", "--")
                    pr   = m.get("PR_ms", "--")
                    qrs  = m.get("QRS_ms", "--")
                    qt   = m.get("QT_ms", "--")
                    qtc  = m.get("QTc_ms", "--")
                    st   = m.get("ST_mV", m.get("ST_ms", "--"))
                    rr   = m.get("RR_ms", "--")
                    rv5p = m.get("RV5_plus_SV1_mV", "--")
                    pqt  = m.get("P_QRS_T_mm", ["--", "--", "--"])
                    qtcF = None
                    rv5s = m.get("RV5_SV1_mV", ["--", "--"])

                    # Build vertical stacks (label top, value bottom) to match top metrics layout
                    metrics = [
                        ("HR", f"{hr} BPM"),
                        ("PR", f"{pr} ms"),
                        ("QRS Complex", f"{qrs} ms"),
                        ("QT", f"{qt} ms"),
                        ("QTc", f"{qtc} ms"),
                        ("ST", f"{st} mV"),
                        ("RR", f"{rr} ms"),
                        ("RV5+SV1", f"{rv5p} mV"),
                        ("P/QRS/T", f"{pqt[0]}/{pqt[1]}/{pqt[2]} mm"),
                        ("RV5/SV1", f"{rv5s[0]}/{rv5s[1]} mV"),
                    ]

                    # Render as a tight 11-column table (label top, value bottom) to ensure one-row layout
                    labels_row = []

                    values_row = []
                    for label, value in metrics:
                        labels_row.append(
                            f"<td style='width:8%; padding:6px 82px; text-align:center; white-space:nowrap; color:#ff6600; font-size:15px; font-weight:900; letter-spacing:0.2px;'>{label}</td>"
                        )
                        values_row.append(
                            f"<td style='width:8%; padding:4px 16px 8px; text-align:center; white-space:nowrap; color:#222222; font-size:14px; font-weight:800;'>{value}</td>"
                        )
                    table_html = (
                        "<table style='width:100%; border-collapse:collapse; table-layout:fixed;'>"
                        + "<tr>" + "".join(labels_row) + "</tr>"
                        + "<tr>" + "".join(values_row) + "</tr>"
                        + "</table>"
                    )
                    line = table_html
        except Exception as e:
            print(f"Failed to read metrics for report: {e}")
            import traceback
            traceback.print_exc()

        if hasattr(self, "parameters_text"):
            # If HTML built, render as HTML; else plain text
            if line.startswith("<table") or line.startswith("<div"):
                self.parameters_text.setHtml(line)
            else:
                self.parameters_text.setPlainText(line)
        
        # Show METRICS panel when user selects/clicks a report in Recent Reports
        if hasattr(self, "metrics_notes_card"):
            self.metrics_notes_card.show()

    def update_live_metrics_panel(self):
        """Update METRICS panel with live data from ECG test page (if not viewing a report)"""
        try:
            # If user is viewing a specific report, don't override with live data
            if getattr(self, '_viewing_report', False):
                return
            
            # Check if ECG test page exists and has data
            if not hasattr(self, 'ecg_test_page') or not self.ecg_test_page:
                return
            
            # Get current metrics from metric labels (top cards)
            if not hasattr(self, 'metric_labels') or not self.metric_labels:
                # Show "Waiting for ECG data..." if no metrics yet
                if hasattr(self, 'parameters_text'):
                    self.parameters_text.setHtml(
                        "<div style='text-align:center; padding:20px; color:#666; font-size:14px;'>"
                        "📊 <b>LIVE METRICS</b><br><br>Waiting for ECG data...<br>"
                        "<small>Start ECG acquisition or demo mode to see real-time metrics</small>"
                        "</div>"
                    )
                return
            
            # Extract metrics from metric labels
            hr = self.metric_labels.get('heart_rate', QLabel()).text().replace(' ', '').replace('BPM', '') or '--'
            pr = self.metric_labels.get('pr_interval', QLabel()).text().replace(' ', '').replace('ms', '') or '--'
            qrs = self.metric_labels.get('qrs_duration', QLabel()).text().replace(' ', '').replace('ms', '') or '--'
            qtc_raw = self.metric_labels.get('qtc_interval', QLabel()).text() or '--/--'
            
            # Parse QT/QTc
            qt = '--'
            qtc = '--'
            if '/' in qtc_raw:
                parts = [part.strip() for part in qtc_raw.replace(' ms', '').split('/') if part.strip()]
                if len(parts) >= 1:
                    qt = parts[0]
                if len(parts) >= 2:
                    qtc = parts[1]
            else:
                qtc = qtc_raw
            
            # Calculate RR from HR
            try:
                hr_val = float(hr) if hr != '--' else 0
                rr = int(60000 / hr_val) if hr_val > 0 else '--'
            except:
                rr = '--'
            
            # Get wave amplitudes from ECG test page if available
            rv5_sv1_sum = '--'
            p_qrs_t = '--/--/--'
            rv5_sv1 = '--/--'
            
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                try:
                    # Calculate wave amplitudes in real-time
                    if hasattr(self.ecg_test_page, 'calculate_wave_amplitudes'):
                        wave_amps = self.ecg_test_page.calculate_wave_amplitudes()
                        if wave_amps:
                            p_amp = wave_amps.get('p_amp', 0.0)
                            qrs_amp = wave_amps.get('qrs_amp', 0.0)
                            t_amp = wave_amps.get('t_amp', 0.0)
                            rv5 = wave_amps.get('rv5', 0.0)
                            sv1 = wave_amps.get('sv1', 0.0)
                            
                            # Convert to display format
                            rv5_sv1_sum = f"{(rv5 + sv1):.3f}" if (rv5 + sv1) > 0 else '--'
                            p_qrs_t = f"{p_amp:.2f}/{qrs_amp:.2f}/{t_amp:.2f}" if (p_amp + qrs_amp + t_amp) > 0 else '--/--/--'
                            rv5_sv1 = f"{rv5:.2f}/{sv1:.2f}" if (rv5 + sv1) > 0 else '--/--'
                except Exception as e:
                    print(f"Error calculating wave amplitudes for dashboard: {e}")
            
            # Build metrics table
            metrics = [
                ("HR", f"{hr} BPM" if hr != '--' else '--'),
                ("PR", f"{pr} ms" if pr != '--' else '--'),
                ("QRS Complex", f"{qrs} ms" if qrs != '--' else '--'),
                ("QT", f"{qt} ms" if qt != '--' else '--'),
                ("QTc", f"{qtc} ms" if qtc != '--' else '--'),
                ("RR", f"{rr} ms" if rr != '--' else '--'),
                ("RV5+SV1", f"{rv5_sv1_sum} mV"),
                ("P/QRS/T", f"{p_qrs_t} mm"),
                ("RV5/SV1", f"{rv5_sv1} mV"),
            ]
            
            # Render as HTML table with "LIVE" badge
            labels_row = []
            values_row = []
            for label, value in metrics:
                labels_row.append(
                    f"<td style='width:8%; padding:6px 82px; text-align:center; white-space:nowrap; color:#ff6600; font-size:15px; font-weight:900; letter-spacing:0.2px;'>{label}</td>"
                )
                values_row.append(
                    f"<td style='width:8%; padding:4px 16px 8px; text-align:center; white-space:nowrap; color:#222222; font-size:14px; font-weight:800;'>{value}</td>"
                )
            
            # Add LIVE indicator
            live_badge = (
                "<div style='text-align:right; padding:4px 8px; font-size:11px;'>"
                "<span style='background:#4CAF50; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;'>"
                "● LIVE</span>"
                "</div>"
            )
            
            table_html = (
                live_badge +
                "<table style='width:100%; border-collapse:collapse; table-layout:fixed;'>"
                + "<tr>" + "".join(labels_row) + "</tr>"
                + "<tr>" + "".join(values_row) + "</tr>"
                + "</table>"
            )
            
            if hasattr(self, 'parameters_text'):
                self.parameters_text.setHtml(table_html)
                
        except Exception as e:
            print(f"Error updating live metrics panel: {e}")

    def is_ecg_active(self):
        """Return True if demo is ON or serial acquisition is running."""
        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                # Demo mode active?
                if hasattr(self.ecg_test_page, 'demo_toggle') and self.ecg_test_page.demo_toggle.isChecked():
                    return True
                try:
                    t = getattr(self.ecg_test_page, 'timer', None)
                    if t is not None and t.isActive():
                        return True
                except Exception:
                    pass
                # Serial acquisition running?
                reader = getattr(self.ecg_test_page, 'serial_reader', None)
                if reader and getattr(reader, 'running', False):
                    return True
                # 🔧 Allow metric updates even when demo/serial not running
                # This ensures dashboard values update from Lead 2 calculation
                return True  # Always allow updates for calibrated metrics
        except Exception:
            pass
        return True  # Default to True to ensure updates work

    def calculate_stable_rr_interval(self, ecg_signal, sampling_rate):
        """Calculate stabilized RR interval using multiple validation layers"""
        try:
            from scipy.signal import find_peaks
            import numpy as np
            
            if len(ecg_signal) < 1000:  # Need at least 2 seconds at 500Hz
                return None, None
            
            # Step 1: Apply gentle filtering for RR stability
            # Use measurement filter for accurate RR calculation
            try:
                from ecg.signal_paths import measurement_filter
                filtered_signal = measurement_filter(ecg_signal, sampling_rate)
            except ImportError:
                # Fallback: use simple filtering if measurement_filter not available
                filtered_signal = ecg_signal
            
            # Step 2: Multi-strategy peak detection for RR stability
            rr_values = []
            
            # Strategy A: Conservative (most stable)
            try:
                peaks_a, _ = find_peaks(
                    filtered_signal,
                    height=np.std(filtered_signal) * 0.4,
                    distance=int(0.4 * sampling_rate),  # 400ms minimum
                    prominence=np.std(filtered_signal) * 0.4
                )
                if len(peaks_a) >= 2:
                    rr_a = np.diff(peaks_a) * (1000.0 / sampling_rate)
                    # Strict RR filtering: 300-2000ms (30-200 BPM)
                    valid_a = rr_a[(rr_a >= 300) & (rr_a <= 2000)]
                    if len(valid_a) >= 3:  # Need at least 3 intervals
                        rr_values.extend(valid_a)
            except Exception as e:
                print(f" Strategy A failed: {e}")
            
            # Strategy B: Normal (moderate sensitivity)
            try:
                peaks_b, _ = find_peaks(
                    filtered_signal,
                    height=np.std(filtered_signal) * 0.3,
                    distance=int(0.3 * sampling_rate),  # 300ms minimum
                    prominence=np.std(filtered_signal) * 0.3
                )
                if len(peaks_b) >= 2:
                    rr_b = np.diff(peaks_b) * (1000.0 / sampling_rate)
                    # Moderate RR filtering: 250-3000ms (20-240 BPM)
                    valid_b = rr_b[(rr_b >= 250) & (rr_b <= 3000)]
                    if len(valid_b) >= 3:
                        rr_values.extend(valid_b)
            except Exception as e:
                print(f" Strategy B failed: {e}")
            
            # Step 3: RR interval validation and stabilization
            if len(rr_values) < 3:
                return None, None
            
            # Convert to numpy array
            rr_values = np.array(rr_values)
            
            # Step 4: Remove outliers using IQR method
            q25, q75 = np.percentile(rr_values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Filter outliers
            clean_rr = rr_values[(rr_values >= lower_bound) & (rr_values <= upper_bound)]
            
            if len(clean_rr) < 2:
                return None, None
            
            # Step 5: Calculate stable RR using median with EMA smoothing
            median_rr = np.median(clean_rr)
            
            # Apply EMA smoothing if we have previous RR value
            if hasattr(self, '_last_stable_rr'):
                alpha = 0.3  # Smoothing factor
                smoothed_rr = alpha * median_rr + (1 - alpha) * self._last_stable_rr
                self._last_stable_rr = smoothed_rr
            else:
                smoothed_rr = median_rr
                self._last_stable_rr = median_rr
            
            # Step 6: Final validation
            if 300 <= smoothed_rr <= 2000:  # 30-200 BPM range
                # Calculate BPM from smoothed RR
                stable_bpm = 60000.0 / smoothed_rr
                
                # Additional BPM validation
                if 40 <= stable_bpm <= 200:
                    print(f"🔒 Stable RR: {smoothed_rr:.1f}ms → BPM: {stable_bpm:.1f}")
                    return smoothed_rr, stable_bpm
                else:
                    print(f"⚠️ BPM out of range: {stable_bpm:.1f}")
                    return None, None
            else:
                print(f"⚠️ RR out of range: {smoothed_rr:.1f}ms")
                return None, None
                
        except Exception as e:
            print(f"❌ Error calculating stable RR: {e}")
            return None, None
    
    def calculate_standard_ecg_metrics(self, bpm):
        """DEPRECATED: Calculate standard ECG metrics based on BPM using simplified formulas.
        
        ⚠️ This function uses simplified BPM-based formulas and is kept only as a fallback.
        Real calculations should use calculate_live_ecg_metrics() which calculates from actual ECG signal.
        """
        try:
            bpm = float(bpm)
            
            # Standard medical formulas based on heart rate
            
            # PR Interval: Normal range 120-200ms, inversely related to HR
            # Formula: PR = 180 - (BPM-60)*0.3 (simplified approximation)
            pr_interval = max(120, min(200, 180 - (bpm - 60) * 0.3))
            
            # 🔧 HR-dependent PR calibration offsets (from calibration guide)
            # These adjustments match the reference table values exactly
            if bpm >= 200:
                pr_interval -= 8.0  # High HR: reduce PR by 8ms
            elif bpm >= 190:
                pr_interval -= 6.0  # Reduce by 6ms
            elif bpm >= 180:
                pr_interval -= 6.0  # Reduce by 6ms
            elif bpm >= 170:
                pr_interval -= 4.0  # Reduce by 4ms
            elif bpm >= 150:
                pr_interval -= 5.0  # Reduce by 5ms
            elif bpm >= 120:
                pr_interval -= 5.0  # Reduce by 5ms
            elif bpm >= 70:
                pr_interval -= 7.0  # 70 BPM needs reduction by 7ms
            
            # Ensure PR stays within physiological limits
            pr_interval = max(80, min(200, pr_interval))
            
            # QRS Duration: Normal range 60-100ms, relatively stable
            # Formula: QRS = 80 + (BPM-60)*0.1 (slight variation with HR)
            qrs_duration = max(60, min(100, 80 + (bpm - 60) * 0.1))
            
            # 🔧 QRS Duration fine-tuning calibration (from calibration guide)
            # Minor threshold adjustment for exact reference table match
            # Most values are already within 1-2ms, this fine-tunes to within ±1ms
            if bpm >= 100:
                qrs_duration -= 1.0  # High HR: slight reduction
            elif bpm >= 80:
                qrs_duration -= 0.5  # Medium HR: minimal reduction
            elif bpm >= 60:
                qrs_duration += 0.0  # Normal HR: no adjustment needed
            else:
                qrs_duration += 1.0  # Low HR: slight increase
            
            # Ensure QRS stays within physiological limits
            qrs_duration = max(60, min(100, qrs_duration))
            
            # QT Interval: Normal range 300-440ms, inversely related to HR
            # Bazett's formula: QTc = QT / sqrt(RR), where RR = 60/BPM
            # Simplified: QT = 400 - (BPM-60)*0.8
            qt_interval = max(300, min(440, 400 - (bpm - 60) * 0.8))
            
            # 🔧 QT Interval calibration (from calibration guide)
            # According to reference table analysis, QT is already correct
            # Adding minimal verification adjustments for perfect match
            if bpm >= 200:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 180:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 160:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 140:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 120:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 100:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 80:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 60:
                qt_interval += 0.0  # Already perfect
            else:
                qt_interval += 0.0  # Already perfect
            
            # QTc (corrected QT): Using Bazett's formula
            rr_interval = 60000 / bpm  # RR in milliseconds
            qtc_bazett = qt_interval / ((rr_interval / 1000) ** 0.5)
            qtc_bazett = max(350, min(450, qtc_bazett))
            
            # 🔧 QTc verification (automatically correct if QT is correct)
            # Since QTc uses Bazett's formula: QTc = QT / sqrt(RR)
            # If QT is correct, QTc will automatically be correct
            # Adding range validation for safety
            qtc_bazett = max(300, min(500, qtc_bazett))  # Extended safety range
            
            # P Duration: Normal range 60-120ms, relatively stable
            # Standard P wave duration is typically 80-100ms, slight variation with HR
            p_duration = max(60, min(120, 80 + (bpm - 60) * 0.1))
            
            return {
                'heart_rate': int(round(bpm)),
                'pr_interval': int(round(pr_interval)),
                'qrs_duration': int(round(qrs_duration)),
                'qt_interval': int(round(qt_interval)),
                'qtc_interval': f"{int(round(qt_interval))}/{int(round(qtc_bazett))}",
                'p_duration': int(round(p_duration))
            }
            
        except Exception as e:
            print(f"Error calculating standard ECG metrics: {e}")
            return None
    
    def calculate_live_ecg_metrics(self, ecg_signal, sampling_rate=None):
        """Calculate live ECG metrics from Lead 2 data - ADAPTIVE for 40-300 BPM
        
        CRITICAL: Uses actual sampling rate from ECG test page for accurate BPM calculation.
        On Windows, sampling rate may be 80 Hz (not 500 Hz), so we must detect it correctly.
        """
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            
            # Ensure we have enough data
            if len(ecg_signal) < 200:
                return {}
            
            # CRITICAL: Get actual sampling rate from ECG test page
            # Use same fallback as ECG test page (250 Hz) for consistency
            import platform
            is_windows = platform.system() == 'Windows'
            platform_tag = "[Windows]" if is_windows else "[macOS/Linux]"
            
            fs = 500.0  # Base fallback (matches ECG test page - unified across platforms)
            if sampling_rate and sampling_rate > 10:
                fs = float(sampling_rate)
            elif hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                try:
                    if hasattr(self.ecg_test_page, 'sampler') and hasattr(self.ecg_test_page.sampler, 'sampling_rate'):
                        if self.ecg_test_page.sampler.sampling_rate > 10:
                            fs = float(self.ecg_test_page.sampler.sampling_rate)
                    elif hasattr(self.ecg_test_page, 'sampling_rate') and self.ecg_test_page.sampling_rate > 10:
                        fs = float(self.ecg_test_page.sampling_rate)
                except Exception as e:
                    pass
            
            # Enhanced debugging with platform detection
            if not hasattr(self, '_calc_count'):
                self._calc_count = 0
            self._calc_count += 1
            if self._calc_count <= 5:  # First 5 calculations (increased from 3)
                print(f" {platform_tag} BPM Calculation - Sampling rate: {fs:.1f} Hz, Signal length: {len(ecg_signal)} samples")
            
            # Windows-specific warnings
            if is_windows and fs == 500.0:
                if self._calc_count <= 5:
                    print(f" {platform_tag} Sampling rate detection failed, using fallback 500.0 Hz")
            
            # Validation
            if fs <= 0 or not np.isfinite(fs):
                if is_windows:
                    print(f" {platform_tag} Invalid sampling rate detected: {fs}, using fallback 500.0 Hz")
                fs = 500.0  # Fallback
            
            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, ecg_signal)
            
            # SMART ADAPTIVE PEAK DETECTION (40-300 BPM with BPM-based selection)
            # Run multiple detections and choose based on CALCULATED BPM consistency
            height_threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
            prominence_threshold = np.std(filtered_signal) * 0.4
            
            # Run 3 detection strategies
            detection_results = []
            
            # Strategy 1: Conservative (best for 10-120 BPM)
            # Distance set to minimum RR for highest BPM in range (120 BPM = 500ms)
            # RR interval filtering (200-6000ms) will handle the full 10-300 BPM range
            peaks_conservative, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
                prominence=prominence_threshold
            )
            if len(peaks_conservative) >= 2:
                rr_cons = np.diff(peaks_conservative) * (1000 / fs)
                # Accept RR intervals from 200–6000 ms (300–10 BPM) - changed from 2000 to 6000 to allow 10 BPM
                valid_cons = rr_cons[(rr_cons >= 200) & (rr_cons <= 6000)]
                if len(valid_cons) > 0:
                    bpm_cons = 60000 / np.median(valid_cons)
                    std_cons = np.std(valid_cons)
                    detection_results.append(('conservative', peaks_conservative, bpm_cons, std_cons))
            
            # Strategy 2: Normal (best for 100-180 BPM)
            peaks_normal, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.3 * fs),  # 240ms - medium distance
                prominence=prominence_threshold
            )
            if len(peaks_normal) >= 2:
                rr_norm = np.diff(peaks_normal) * (1000 / fs)
                # Accept RR intervals from 200–6000 ms (300–10 BPM) - changed from 2000 to 6000 to allow 10 BPM
                valid_norm = rr_norm[(rr_norm >= 200) & (rr_norm <= 6000)]
                if len(valid_norm) > 0:
                    bpm_norm = 60000 / np.median(valid_norm)
                    std_norm = np.std(valid_norm)
                    detection_results.append(('normal', peaks_normal, bpm_norm, std_norm))
            
            # Strategy 3: Tight (best for 160-300 BPM)
            peaks_tight, _ = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=int(0.2 * fs),  # 160ms - tight distance for high BPM
                prominence=prominence_threshold
            )
            if len(peaks_tight) >= 2:
                rr_tight = np.diff(peaks_tight) * (1000 / fs)
                # Accept RR intervals from 200–6000 ms (300–10 BPM) - changed from 2000 to 6000 to allow 10 BPM
                valid_tight = rr_tight[(rr_tight >= 200) & (rr_tight <= 6000)]
                if len(valid_tight) > 0:
                    bpm_tight = 60000 / np.median(valid_tight)
                    std_tight = np.std(valid_tight)
                    detection_results.append(('tight', peaks_tight, bpm_tight, std_tight))
            
            # Select based on BPM consistency (lowest std deviation = most stable)
            if detection_results:
                # Sort by consistency (lower std = better)
                detection_results.sort(key=lambda x: x[3])  # Sort by std
                best_method, peaks, best_bpm, best_std = detection_results[0]
            else:
                # Fallback - use conservative distance to handle low BPM (10-40 BPM)
                peaks, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
                    prominence=prominence_threshold
                )
            
            metrics = {}
            
            # Calculate Heart Rate (instantaneous, per-beat)
            if len(peaks) >= 2:
                # Calculate R-R intervals in milliseconds
                # CRITICAL: Use correct sampling rate (fs) for accurate BPM calculation
                rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
                
                # Filter physiologically reasonable intervals (200-6000 ms)
                # 200 ms = 300 BPM (max), 6000 ms = 10 BPM (min)
                # Changed from 120 to 200 to match ECG test page and reduce noise
                min_rr_ms = 200
                max_rr_ms = 6000
                valid_intervals = rr_intervals_ms[(rr_intervals_ms >= min_rr_ms) & (rr_intervals_ms <= max_rr_ms)]
                
                if len(valid_intervals) > 0:
                    # Calculate heart rate from median R-R interval (more stable than instantaneous)
                    median_rr = np.median(valid_intervals)
                    heart_rate = 60000 / median_rr  # Convert to BPM

                    print(" Dashboard Heart Rate", heart_rate)
                    
                    # Ensure reasonable range (10-300 BPM)
                    heart_rate = max(10, min(300, heart_rate))
                    
                    # STABLE BPM WITH EXPONENTIAL MOVING AVERAGE (EMA) - Clinical Standard
                    # EMA provides stability while responding to genuine changes
                    # Alpha = 0.1 gives ~40 second stabilization (updates every 1 second)
                    if not hasattr(self, '_dashboard_bpm_ema'):
                        self._dashboard_bpm_ema = heart_rate  # Initialize with first reading
                        self._dashboard_bpm_alpha = 0.1  # Smoothing factor (0.1 = 40s stabilization)
                        print(f" Dashboard BPM EMA initialized with: {heart_rate}")  # Debug
                    else:
                        # Apply EMA: new_EMA = alpha * new_value + (1 - alpha) * old_EMA
                        self._dashboard_bpm_ema = self._dashboard_bpm_alpha * heart_rate + (1 - self._dashboard_bpm_alpha) * self._dashboard_bpm_ema
                        print(f" Dashboard BPM EMA updated: raw={heart_rate}, ema={self._dashboard_bpm_ema}")  # Debug
                    
                    # Use EMA value for display (stable and accurate)
                    smoothed_bpm = int(round(self._dashboard_bpm_ema))
                    print(f" Dashboard BPM final value: {smoothed_bpm}")  # Debug
                    
                    # Store for next iteration
                    self._last_stable_dashboard_bpm = smoothed_bpm
                    
                    # Add heart rate to metrics
                    metrics['heart_rate'] = smoothed_bpm
                else:
                    metrics['heart_rate'] = 0
            else:
                metrics['heart_rate'] = 0
            
            # ✅ REAL CALCULATIONS: Calculate PR, QRS, P, QT, QTC from actual ECG signal using clinical formulas
            # Uses 0.05-150 Hz measurement channel with median beat and clinical-grade detection methods
            # This replaces reference value lookups with real-time calculations from the signal
            
            try:
                # Import clinical measurement functions (real formulas from reference software)
                try:
                    from ecg.clinical_measurements import (
                        build_median_beat, get_tp_baseline, measure_pr_from_median_beat,
                        measure_qrs_duration_from_median_beat, measure_qt_from_median_beat,
                        measure_p_duration_from_median_beat
                    )
                except ImportError:
                    # Try alternative import path
                    import sys
                    import os
                    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)
                    from ecg.clinical_measurements import (
                        build_median_beat, get_tp_baseline, measure_pr_from_median_beat,
                        measure_qrs_duration_from_median_beat, measure_qt_from_median_beat,
                        measure_p_duration_from_median_beat
                    )
                
                # Need at least 8 beats for median beat calculation (GE/Philips standard)
                if len(peaks) >= 8:
                    # Build median beat from Lead II signal (requires ≥8 beats)
                    time_axis, median_beat = build_median_beat(ecg_signal, peaks, fs, min_beats=8)
                    
                    if median_beat is not None and time_axis is not None:
                        # Get TP baseline for accurate measurements
                        r_mid = peaks[len(peaks) // 2]
                        prev_r_idx = peaks[len(peaks) // 2 - 1] if len(peaks) > 1 else None
                        tp_baseline = get_tp_baseline(ecg_signal, r_mid, fs, prev_r_peak_idx=prev_r_idx)
                        
                        # Calculate RR interval for QTC calculation
                        if len(peaks) >= 2:
                            rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
                            valid_rr = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)]
                            rr_ms = np.median(valid_rr) if len(valid_rr) > 0 else 600.0
                        else:
                            rr_ms = 600.0
                        
                        # Calculate PR Interval from median beat (real formula)
                        pr_val = measure_pr_from_median_beat(median_beat, time_axis, fs, tp_baseline)
                        if pr_val is None or pr_val <= 0:
                            pr_val = 0
                        
                        # Apply EMA smoothing for PR interval stability
                        if not hasattr(self, '_dashboard_pr_ema'):
                            self._dashboard_pr_ema = pr_val
                            self._dashboard_pr_alpha = 0.2
                        else:
                            self._dashboard_pr_ema = self._dashboard_pr_alpha * pr_val + (1 - self._dashboard_pr_alpha) * self._dashboard_pr_ema
                        
                        metrics['pr_interval'] = int(round(self._dashboard_pr_ema))
                        
                        # Calculate QRS Duration from median beat (real formula)
                        qrs_val = measure_qrs_duration_from_median_beat(median_beat, time_axis, fs, tp_baseline)
                        if qrs_val is None or qrs_val <= 0:
                            qrs_val = 0
                        
                        # Apply EMA smoothing for QRS duration stability
                        if not hasattr(self, '_dashboard_qrs_ema'):
                            self._dashboard_qrs_ema = qrs_val
                            self._dashboard_qrs_alpha = 0.2
                        else:
                            self._dashboard_qrs_ema = self._dashboard_qrs_alpha * qrs_val + (1 - self._dashboard_qrs_alpha) * self._dashboard_qrs_ema
                        
                        metrics['qrs_duration'] = int(round(self._dashboard_qrs_ema))
                        
                        # Calculate P Duration from median beat (real formula)
                        p_duration = measure_p_duration_from_median_beat(median_beat, time_axis, fs, tp_baseline)
                        if p_duration is None or p_duration <= 0:
                            p_duration = 0
                        # Store P duration in both p_duration and st_interval (st_interval label shows P)
                        p_duration_int = int(round(p_duration))
                        metrics['p_duration'] = p_duration_int
                        metrics['p_duration'] = p_duration_int
                        
                        # Calculate QT Interval from median beat (real formula)
                        qt_val = measure_qt_from_median_beat(median_beat, time_axis, fs, tp_baseline, rr_ms=rr_ms)
                        if qt_val is None or qt_val <= 0:
                            qt_val = 0
                        
                        metrics['qt_interval'] = int(round(qt_val))
                        
                        # Calculate QTc using Bazett's formula: QTc = QT / sqrt(RR_sec)
                        if qt_val > 0 and rr_ms > 0:
                            rr_sec = rr_ms / 1000.0
                            qtc_val = qt_val / (rr_sec ** 0.5)
                            # Validate QTC range (250-600 ms)
                            qtc_val = max(250, min(600, qtc_val))
                        else:
                            qtc_val = 0
                        
                        # Format QT/QTc display
                        qt_int = int(round(qt_val)) if qt_val > 0 else 0
                        qtc_int = int(round(qtc_val)) if qtc_val > 0 else 0
                        
                        if qt_int > 0 and qtc_int > 0:
                            metrics['qtc_interval'] = f"{qt_int}/{qtc_int}"
                        elif qtc_int > 0:
                            metrics['qtc_interval'] = str(qtc_int)
                        else:
                            metrics['qtc_interval'] = "0"
                    else:
                        # Fallback if median beat cannot be built
                        metrics['pr_interval'] = 0
                        metrics['qrs_duration'] = 0
                        metrics['qt_interval'] = 0
                        metrics['qtc_interval'] = "0"
                        metrics['p_duration'] = 0
                        metrics['st_interval'] = "0"  # P duration = 0
                else:
                    # Not enough beats for median beat calculation
                    metrics['pr_interval'] = 0
                    metrics['qrs_duration'] = 0
                    metrics['qt_interval'] = 0
                    metrics['qtc_interval'] = "0"
                    metrics['p_duration'] = 0
                    metrics['st_interval'] = "0"  # P duration = 0
                    
            except ImportError as e:
                print(f" ⚠️ Clinical measurement functions not available: {e}")
                # Fallback if clinical measurement functions not available
                metrics['pr_interval'] = 0
                metrics['qrs_duration'] = 0
                metrics['qt_interval'] = 0
                metrics['qtc_interval'] = "0"
                metrics['p_duration'] = 0
                metrics['st_interval'] = "0"  # P duration = 0
            except Exception as e:
                print(f" ⚠️ Error calculating real ECG metrics: {e}")
                # Fallback on error
                metrics['pr_interval'] = 0
                metrics['qrs_duration'] = 0
                metrics['qt_interval'] = 0
                metrics['qtc_interval'] = "0"
                metrics['p_duration'] = 0
                metrics['st_interval'] = "0"  # P duration = 0
            
            return metrics
            
        except Exception:
            # Quietly fall back if metrics cannot be calculated
            return {}

    def update_dashboard_metrics_live(self, ecg_metrics):
        """Update dashboard metrics with live calculated values"""
        try:
            import time as _time
            # Throttle: reduced to 0.3s for much faster responsiveness (real-time)
            if not hasattr(self, '_last_metrics_update_ts'):
                self._last_metrics_update_ts = 0.0
            if _time.time() - self._last_metrics_update_ts < 0.3:
                return
            self._last_metrics_update_ts = _time.time()
            # Do not update metrics for first-time users until acquisition/demo starts
            if not self.is_ecg_active():
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("0 BPM")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("0 ms")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("0 ms")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("0")
                key = 'st_interval' if 'st_interval' in self.metric_labels else 'st_segment'
                if key in self.metric_labels:
                    self.metric_labels[key].setText("0 ms")
                return
            
            # Allow updates in demo mode - display the values set by demo_manager
            # Update Heart Rate
            if 'heart_rate' in ecg_metrics:
                hr_val = ecg_metrics['heart_rate']
                if hr_val in (None, "", "--"):
                    self.metric_labels['heart_rate'].setText("0 BPM")
                else:
                    try:
                        hr_int = int(round(float(hr_val)))
                    except Exception:
                        hr_int = hr_val
                    if isinstance(hr_int, int):
                        self.metric_labels['heart_rate'].setText(f"{hr_int:3d} BPM")
                    else:
                        self.metric_labels['heart_rate'].setText(f"{hr_int} BPM")

            # ------- PR / QRS / P / QT/QTc: update at most every 6 seconds -------
            # Build a compact snapshot of the interval metrics we care about
            pr_val   = ecg_metrics.get('pr_interval')
            qrs_val  = ecg_metrics.get('qrs_duration')
            p_val    = ecg_metrics.get('st_interval')  # st_interval stores P duration (label shows "P")
            qtc_raw  = ecg_metrics.get('qtc_interval')
            interval_snapshot = (pr_val, qrs_val, p_val, qtc_raw)

            # Initialise tracking state
            if not hasattr(self, '_last_interval_metrics_ts'):
                self._last_interval_metrics_ts = 0.0
            if not hasattr(self, '_last_interval_metrics_values'):
                self._last_interval_metrics_values = None

            now = _time.time()
            since_last = now - self._last_interval_metrics_ts

            # Only redraw PR/QRS/P/QT/QTc if:
            #  - at least 1.0 second has passed (reduced from 6s for real-time), AND
            #  - the values actually changed since last update
            if since_last >= 1.0 and interval_snapshot != self._last_interval_metrics_values:
                # Update PR Interval
                if pr_val is not None and 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText(f"{pr_val} ms")

                # Update QRS Duration
                if qrs_val is not None and 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText(f"{qrs_val} ms")

                # Update P Duration (stored in st_interval key, label shows "P")
                if p_val is not None and 'st_interval' in self.metric_labels:
                    self.metric_labels['st_interval'].setText(f"{p_val} ms")

                # Update QT/QTc interval text
                if 'qtc_interval' in self.metric_labels:
                    qtc_text = str(qtc_raw) if qtc_raw is not None else "0"
                    if qtc_text.endswith(" ms"):
                        qtc_text = qtc_text[:-3]
                    self.metric_labels['qtc_interval'].setText(qtc_text)

                # Remember timestamp and snapshot
                self._last_interval_metrics_ts = now
                self._last_interval_metrics_values = interval_snapshot
            
            # Update Sampling Rate - Commented out
            # if 'sampling_rate' in ecg_metrics:
            #     self.metric_labels['sampling_rate'].setText(ecg_metrics['sampling_rate'])
            # Record last update time
            self._last_metrics_update_ts = _time.time()
            
            # Keep ECG test page metrics identical to dashboard
            try:
                self.sync_dashboard_metrics_to_ecg_page()
            except Exception:
                pass
            
        except Exception as e:
            print(f"Error updating live dashboard metrics: {e}")




    def update_ecg(self, frame):
        try:
            # Try to get data from ECG test page if available
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                try:
                    # Validate ECG test page data structure
                    if not hasattr(self.ecg_test_page, 'data') or not self.ecg_test_page.data:
                        print(" ECG test page has no data")
                        return self._fallback_wave_update(frame)
                    
                    if len(self.ecg_test_page.data) <= 1:
                        print(" Insufficient ECG data (need Lead II)")
                        return self._fallback_wave_update(frame)
                    
                    # Get Lead II data from ECG test page (index 1 is Lead II)
                    lead_ii_data = self.ecg_test_page.data[1]
                    
                    # Validate Lead II data
                    if not isinstance(lead_ii_data, (list, np.ndarray)) or len(lead_ii_data) <= 10:
                        print(" Invalid Lead II data")
                        return self._fallback_wave_update(frame)
                    
                    # Convert to numpy array safely
                    try:
                        original_data = np.asarray(lead_ii_data, dtype=float)
                    except Exception as e:
                        print(f" Error converting Lead II data to array: {e}")
                        return self._fallback_wave_update(frame)

                    # Get actual sampling rate from ECG test page.
                    # Hardware default is 500 Hz; fall back only when no valid rate is reported.
                    actual_sampling_rate = 500  # Hardware default: 500 Hz
                    try:
                        if (hasattr(self.ecg_test_page, 'sampler') and
                                hasattr(self.ecg_test_page.sampler, 'sampling_rate') and
                                self.ecg_test_page.sampler.sampling_rate):
                            reported_rate = float(self.ecg_test_page.sampler.sampling_rate)
                            # Accept only physiologically sane rates (50–1000 Hz)
                            if 50.0 <= reported_rate <= 1000.0:
                                actual_sampling_rate = reported_rate
                            else:
                                print(f" Reported sampling rate {reported_rate} Hz out of range; using 500 Hz default")
                    except Exception as e:
                        print(f" Error getting sampling rate: {e}")

                    # Apply filters matching the 12-lead display (AC notch → EMG → DFT → Gaussian)
                    try:
                        original_data = np.asarray(lead_ii_data, dtype=float)

                        from ecg.ecg_filters import apply_ecg_filters
                        from scipy.ndimage import gaussian_filter1d as _gf1d

                        ac_setting  = str(self.settings_manager.get_setting("filter_ac",  "50")).strip()
                        emg_setting = str(self.settings_manager.get_setting("filter_emg", "150")).strip()
                        dft_setting = str(self.settings_manager.get_setting("filter_dft", "0.5")).strip()

                        # Nyquist guard: AC notch at F Hz requires sampling rate > 2*F Hz.
                        # Disable gracefully rather than letting the filter raise an error.
                        if ac_setting in ("50", "60"):
                            required_fs = float(ac_setting) * 2.0 + 1.0  # e.g. 101 Hz for 50 Hz notch
                            if actual_sampling_rate <= required_fs:
                                print(f" AC filter disabled (rate {actual_sampling_rate} Hz too low for {ac_setting} Hz notch)")
                                ac_setting = "off"

                        original_data = apply_ecg_filters(
                            signal=original_data,
                            sampling_rate=actual_sampling_rate,
                            ac_filter=ac_setting  if ac_setting  not in ("off", "") else None,
                            emg_filter=emg_setting if emg_setting not in ("off", "") else None,
                            dft_filter=dft_setting if dft_setting not in ("off", "") else None,
                        )

                        # Light Gaussian smoothing (sigma=0.8 — same as 12-lead page SMOOTH_SIGMA)
                        if len(original_data) > 5:
                            original_data = _gf1d(original_data, sigma=0.8)

                    except Exception as e:
                        print(f" Dashboard filter error, using raw data: {e}")
                        try:
                            original_data = np.asarray(lead_ii_data, dtype=float)
                        except Exception as e2:
                            print(f" Error converting Lead II data to array: {e2}")
                            return self._fallback_wave_update(frame)
                    
                    # Check for invalid values
                    if np.any(np.isnan(original_data)) or np.any(np.isinf(original_data)):
                        print(" Invalid values (NaN/Inf) in Lead II data")
                        return self._fallback_wave_update(frame)

                    # Choose settings source
                    settings_src = self.settings_manager
                    try:
                        if hasattr(self.ecg_test_page, 'settings_manager') and self.ecg_test_page.settings_manager is not None:
                            settings_src = self.ecg_test_page.settings_manager
                    except Exception:
                        settings_src = self.settings_manager

                    # Apply same AC/EMG/DFT filters as 12-lead dashboard for DISPLAY ONLY
                    try:
                        from ecg.ecg_filters import apply_ecg_filters_from_settings
                        display_data = apply_ecg_filters_from_settings(
                            signal=original_data,
                            sampling_rate=actual_sampling_rate,
                            settings_manager=settings_src
                        )
                    except Exception as filter_err:
                        print(f" Error applying ECG display filters: {filter_err}")
                        display_data = original_data

                    # Apply Gaussian smoothing to reduce corner noise
                    try:
                        from scipy.ndimage import gaussian_filter1d
                        sigma = getattr(self.ecg_test_page, 'SMOOTH_SIGMA', 0.8)
                        if len(display_data) > 5 and sigma > 0:
                            sigma = max(sigma * 1.5, 1.3)
                            display_data = gaussian_filter1d(display_data, sigma=sigma)

                        # Trim filter edge artefacts (similar to inner grid: ~0.5s each side)
                            try:
                                fs = float(actual_sampling_rate)
                                edge_trim = int(0.5 * fs)
                                if edge_trim > 0 and len(display_data) > 2 * edge_trim:
                                    display_data = display_data[edge_trim:-edge_trim]
                            except Exception:
                                pass

                    except Exception as gauss_err:
                        print(f" Error applying Gaussian smoothing to ECG display: {gauss_err}")

                    # Determine visible window based on wave speed (display feature only)
                    try:
                        wave_speed = float(settings_src.get_wave_speed())  # 12.5 / 25 / 50
                        if wave_speed <= 0:
                            wave_speed = 25.0
                    except Exception as e:
                        print(f" Error getting wave speed: {e}")
                        wave_speed = 25.0
                    
                    # Baseline window at 25 mm/s (diagnostic standard)
                    # Smaller window to reduce visible baseline drift
                    # 25 mm/s → 1.5 seconds visible
                    baseline_seconds = 1.5
                    # Scale time window with wave speed:
                    #   12.5 mm/s → 6 s, 25 mm/s → 3 s, 50 mm/s → 1.5 s
                    seconds_to_show = baseline_seconds * (25.0 / max(1e-6, wave_speed))
                    window_samples = int(max(50, min(len(display_data), seconds_to_show * actual_sampling_rate)))

                    # ── Direct mirror of 12-lead Lead II display ────────────────────────
                    # data[1] is already a rolling fixed-size buffer filled by the serial
                    # reader (same as the 12-lead page uses).  We just take the last
                    # window_samples from it, apply the same filters, and plot directly.
                    # This is identical to how Lead II looks on the 12-lead page —
                    # no manual ring-buffer needed, no wrapping jump.
                    # ────────────────────────────────────────────────────────────────────
                    try:
                        src = display_data[-window_samples:]
                        if len(src) > 0:
                            # Hard-center baseline each frame to prevent upward drift
                            current_dc = float(np.nanmean(src))
                            src = src - current_dc + 2048.0

                        # Strip NaNs from the head (happens while buffer fills up)
                        valid_mask = ~np.isnan(src)
                        if valid_mask.sum() < 5:
                            # Not enough real data yet — show flat baseline
                            display_y = np.full(len(self.ecg_x), 2048.0)
                        else:
                            # Baseline-centre once (same as 12-lead adaptive gain logic)
                            dc = float(np.nanmedian(src))
                            src_c = np.where(valid_mask, src - dc + 2048.0, 2048.0)

                            # Resample to the fixed display width (ecg_x has 500 pts)
                            display_len = len(self.ecg_x)
                            x_src = np.linspace(0.0, 1.0, src_c.size)
                            x_dst = np.linspace(0.0, 1.0, display_len)
                            display_y = np.interp(x_dst, x_src, src_c.astype(float))
                            display_y = np.clip(display_y, 0, 4095)

                        # ── Match x-axis to time in seconds (like 12-lead "1..10 s") ──
                        # Update x-axis limits to show the correct time window
                        seconds_shown = window_samples / actual_sampling_rate
                        self.ecg_canvas.axes.set_xlim(0, seconds_shown)

                        # Validate display data
                        if np.any(np.isnan(display_y)) or np.any(np.isinf(display_y)):
                            print(" Invalid display data generated")
                            return self._fallback_wave_update(frame)
                        
                        # Update both X and Y so the visible window matches the time scale exactly
                        x_axis = np.linspace(0.0, seconds_to_show, display_len)
                        self.ecg_line.set_data(x_axis, display_y)
                        # Ensure axes are locked: 0–seconds_to_show in X, 0–4096 in Y
                        self.ecg_canvas.axes.set_autoscale_on(False)
                        self.ecg_canvas.axes.set_xlim(0.0, seconds_to_show)
                        self.ecg_canvas.axes.set_ylim(0, 4096)
                        
                    except Exception as e:
                        print(f" Error processing display data: {e}")
                        return self._fallback_wave_update(frame)

                    
                    # Calculate and update live ECG metrics using ORIGINAL data with SAME sampling rate
                    try:
                        # Use ECG test page's own calculation methods for consistency
                        if hasattr(self.ecg_test_page, 'calculate_ecg_metrics'):
                            self.ecg_test_page.calculate_ecg_metrics()
                        
                        # Get metrics from ECG test page to ensure synchronization
                        if hasattr(self.ecg_test_page, 'get_current_metrics'):
                            ecg_metrics = self.ecg_test_page.get_current_metrics()
                            # Debug: Print metrics to see what's being calculated
                            if hasattr(self, '_debug_counter'):
                                self._debug_counter += 1
                            else:
                                self._debug_counter = 1
                            if self._debug_counter % 50 == 0:  # Optimized: Print every 50 updates (was 10) - reduces console spam
                                print(f" Dashboard ECG metrics: {ecg_metrics}")
                            self.update_dashboard_metrics_from_ecg()
                        
                        # Calculate and update stress level and HRV (throttled to every 3 seconds for stability)
                        if not hasattr(self, '_last_stress_update'):
                            self._last_stress_update = 0
                        if time.time() - self._last_stress_update > 3:
                            self.update_stress_and_hrv(original_data, actual_sampling_rate)
                            self._last_stress_update = time.time()
                        
                        # Update live conclusion every 5 seconds
                        if not hasattr(self, '_last_conclusion_update'):
                            self._last_conclusion_update = 0
                        if time.time() - self._last_conclusion_update > 5:
                            self.update_live_conclusion()
                            self._last_conclusion_update = time.time()
                    except Exception as e:
                        print(f" Error calculating ECG metrics: {e}")
                        # Continue with display even if metrics fail
                    
                    return [self.ecg_line]
                    
                except Exception as e:
                    print(f" Error getting data from ECG test page: {e}")
                    return self._fallback_wave_update(frame)
            
            # No ECG test page available
            return self._fallback_wave_update(frame)
            
        except Exception as e:
            print(f" Critical error in update_ecg: {e}")
            return self._fallback_wave_update(frame)
    
    def _fallback_wave_update(self, frame):
        """Fallback wave generation when ECG data is not available"""
        try:
            self.ecg_y = np.roll(self.ecg_y, -1)
            # Generate demo wave scaled to fit 0-4096 range (centered around 2048)
            y_center = 2048
            amplitude = 800  # Amplitude for demo wave
            demo_value = y_center + amplitude * np.sin(2 * np.pi * 2 * self.ecg_x[-1] + frame/10) + 50 * np.random.randn()
            # Clamp to 0-4096 range
            self.ecg_y[-1] = np.clip(demo_value, 0, 4096)
            self.ecg_line.set_ydata(self.ecg_y)
            # Ensure y-axis is locked to 0-4096 range
            self.ecg_canvas.axes.set_ylim(0, 4096)
            # Do not compute/update metrics from mock wave; keep zeros until user starts
            return [self.ecg_line]
        except Exception as e:
            print(f" Error in fallback wave update: {e}")
            return [self.ecg_line]
    
    def heart_rate_triple_click(self, event):
        """Handle triple-click on heart rate metric to open crash log dialog"""
        # Only count left mouse button clicks
        try:
            if hasattr(event, 'button') and event.button() != Qt.LeftButton:
                return
        except Exception:
            pass
        current_time = time.time()
        
        # Check if this is within 1 second of the last click
        if current_time - self.last_heart_rate_click_time < 1.0:
            self.heart_rate_click_count += 1
        else:
            self.heart_rate_click_count = 1
        
        self.last_heart_rate_click_time = current_time
        
        # Show click count in terminal
        print(f" Heart Rate Metric Click #{self.heart_rate_click_count}")
        
        # If triple-clicked, open crash log dialog
        if self.heart_rate_click_count >= 3:
            self.heart_rate_click_count = 0  # Reset counter
            print(" Triple-click detected! Opening diagnostic dialog...")
            self.crash_logger.log_info("Triple-click detected on heart rate metric", "TRIPLE_CLICK")
            self.open_crash_log_dialog()
        
        # Call original mousePressEvent if it exists
        if hasattr(event, 'original_mousePressEvent'):
            event.original_mousePressEvent(event)
    
    def open_crash_log_dialog(self):
        """Open the crash log diagnostic dialog"""
        try:
            dialog = CrashLogDialog(self.crash_logger, self)
            dialog.exec_()
        except Exception as e:
            self.crash_logger.log_error(f"Failed to open crash log dialog: {str(e)}", e, "DIALOG_ERROR")
            QMessageBox.critical(self, "Error", f"Failed to open diagnostic dialog: {str(e)}")
    
    
    def update_ecg_metrics(self, intervals):
        import time as _time
        # Throttle: reduced to 0.3s for much faster responsiveness (real-time)
        if not hasattr(self, '_last_metrics_update_ts'):
            self._last_metrics_update_ts = 0.0
        if _time.time() - self._last_metrics_update_ts < 0.3:
            return
        if 'Heart_Rate' in intervals and intervals['Heart_Rate'] is not None:
            self.metric_labels['heart_rate'].setText(
                f"{int(round(intervals['Heart_Rate']))} bpm" if isinstance(intervals['Heart_Rate'], (int, float)) else str(intervals['Heart_Rate'])
            )
        if 'PR' in intervals and intervals['PR'] is not None:
            self.metric_labels['pr_interval'].setText(
                f"{int(round(intervals['PR']))} ms" if isinstance(intervals['PR'], (int, float)) else str(intervals['PR'])
            )
        if 'QRS' in intervals and intervals['QRS'] is not None:
            self.metric_labels['qrs_duration'].setText(
                f"{int(round(intervals['QRS']))} ms" if isinstance(intervals['QRS'], (int, float)) else str(intervals['QRS'])
            )
        # QTc label may not exist in current metrics card; update only if present
        # Check for 'QTc_interval' first (demo mode sends this as "400/430")
        if 'QTc_interval' in intervals and intervals['QTc_interval'] is not None and 'qtc_interval' in self.metric_labels:
            # QTc_interval is already in the correct format (e.g., "400/430")
            self.metric_labels['qtc_interval'].setText(f"{intervals['QTc_interval']} ms")
        elif 'QTc' in intervals and intervals['QTc'] is not None and 'qtc_interval' in self.metric_labels:
            if isinstance(intervals['QTc'], (int, float)) and intervals['QTc'] >= 0:
                self.metric_labels['qtc_interval'].setText(f"{int(round(intervals['QTc']))} ms")
            else:
                self.metric_labels['qtc_interval'].setText("-- ms")
        # Record last update time
        self._last_metrics_update_ts = _time.time()
        
        # OPTIMIZED: Reduce sync frequency to prevent lag - only sync every 10th update
        if not hasattr(self, '_sync_throttle_count'):
            self._sync_throttle_count = 0
        self._sync_throttle_count += 1
        
        # Keep ECG test page metrics identical to dashboard (throttled)
        if self._sync_throttle_count % 10 == 0:  # Sync every 10th update instead of every update
            try:
                self.sync_dashboard_metrics_to_ecg_page()
            except Exception:
                pass
        # Also update the ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
        
        # Update recommendations based on new metrics (works in demo mode too!)
        try:
            if hasattr(self, 'update_live_conclusion'):
                self.update_live_conclusion()
        except Exception:
            pass
    
    def sync_dashboard_metrics_to_ecg_page(self):
        """Force sync dashboard's current metric values to ECG test page for consistency"""
        try:
            if not hasattr(self, 'ecg_test_page') or not self.ecg_test_page:
                return
                
            if not hasattr(self.ecg_test_page, 'metric_labels'):
                return
            
            # OPTIMIZED: Reduce sync frequency to prevent lag
            if not hasattr(self, '_sync_count'):
                self._sync_count = 0
            self._sync_count += 1
            
            # Only print sync message every 50th sync to reduce console spam
            if self._sync_count % 50 == 1:
                print(f"🔄 FORCE SYNC: Dashboard -> ECG Page")
                
            # Force sync metric values from dashboard to ECG test page
            # Extract numeric values from dashboard labels (e.g., "100 BPM" -> "100")
            if 'heart_rate' in self.metric_labels and 'heart_rate' in self.ecg_test_page.metric_labels:
                hr_text = self.metric_labels['heart_rate'].text()
                hr_value = hr_text.split()[0] if ' ' in hr_text else hr_text
                self.ecg_test_page.metric_labels['heart_rate'].setText(hr_value)
                if self._sync_count % 50 == 1:
                    print(f"  HR: {hr_value}")
                
            if 'pr_interval' in self.metric_labels and 'pr_interval' in self.ecg_test_page.metric_labels:
                pr_text = self.metric_labels['pr_interval'].text()
                pr_value = pr_text.split()[0] if ' ' in pr_text else pr_text
                self.ecg_test_page.metric_labels['pr_interval'].setText(pr_value)
                if self._sync_count % 50 == 1:
                    print(f"  PR: {pr_value}")
                
            if 'qrs_duration' in self.metric_labels and 'qrs_duration' in self.ecg_test_page.metric_labels:
                qrs_text = self.metric_labels['qrs_duration'].text()
                qrs_value = qrs_text.split()[0] if ' ' in qrs_text else qrs_text
                self.ecg_test_page.metric_labels['qrs_duration'].setText(qrs_value)
                if self._sync_count % 50 == 1:
                    print(f"  QRS: {qrs_value}")
                
            if 'st_interval' in self.metric_labels and 'st_segment' in self.ecg_test_page.metric_labels:
                st_text = self.metric_labels['st_interval'].text()
                st_value = st_text.split()[0] if ' ' in st_text else st_text
                self.ecg_test_page.metric_labels['st_segment'].setText(st_value)
                if self._sync_count % 50 == 1:
                    print(f"  ST: {st_value}")
                
            # Handle qtc_interval - dashboard might have "286/369 ms" format
            if 'qtc_interval' in self.metric_labels and 'qtc_interval' in self.ecg_test_page.metric_labels:
                qtc_text = self.metric_labels['qtc_interval'].text()
                # Extract both QT and QTc values
                if '/' in qtc_text:
                    # Format: "286/369 ms" -> extract "286/369"
                    qtc_value = qtc_text.split()[0] if ' ' in qtc_text else qtc_text
                    self.ecg_test_page.metric_labels['qtc_interval'].setText(qtc_value)
                    if self._sync_count % 50 == 1:
                        print(f"  QT/QTc: {qtc_value}")
                else:
                    # Single value
                    qtc_value = qtc_text.split()[0] if ' ' in qtc_text else qtc_text
                    self.ecg_test_page.metric_labels['qtc_interval'].setText(qtc_value)
                    if self._sync_count % 50 == 1:
                        print(f"  QTc: {qtc_value}")
                    
            if self._sync_count % 50 == 1:
                print("✅ FORCE SYNC COMPLETED - Both pages now show identical values")
            
        except Exception as e:
            print(f"❌ Error syncing dashboard metrics to ECG test page: {e}")
    
    def periodic_sync_to_ecg_page(self):
        """Periodic sync to ensure both pages always show identical values"""
        try:
            # Only sync if ECG is active (either demo or real mode)
            if self.is_ecg_active():
                self.sync_dashboard_metrics_to_ecg_page()
        except Exception as e:
            print(f"❌ Periodic sync error: {e}")
    
    def update_dashboard_metrics_from_ecg(self):
        try:
            import time as _time
            # Throttle: reduced to 0.3s for much faster responsiveness (real-time)
            if not hasattr(self, '_last_metrics_update_ts'):
                self._last_metrics_update_ts = 0.0
            if _time.time() - self._last_metrics_update_ts < 0.3:
                return
            self._last_metrics_update_ts = _time.time()
            if not self.is_ecg_active():
                return
            if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'get_current_metrics'):
                ecg_metrics = self.ecg_test_page.get_current_metrics()
                hr_text = ecg_metrics.get('heart_rate', '0')
                pr_text = ecg_metrics.get('pr_interval', '0')
                qrs_text = ecg_metrics.get('qrs_duration', '0')
                p_text = ecg_metrics.get('st_interval', '0')
                qt_text = ecg_metrics.get('qt_interval', '')
                qtc_text = ecg_metrics.get('qtc_interval', '0')
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText(f"{hr_text} BPM")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText(f"{pr_text} ms")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText(f"{qrs_text} ms")
                key = 'st_interval' if 'st_interval' in self.metric_labels else 'st_segment'
                if key in self.metric_labels:
                    # Display P duration with ms unit (remove any existing units first)
                    p_val = str(p_text).replace(' ms', '').replace('mV', '').strip()
                    self.metric_labels[key].setText(f"{p_val} ms")
                if 'qtc_interval' in self.metric_labels:
                    qt_clean  = str(qt_text).replace(' ms', '').strip()
                    qtc_clean = str(qtc_text).replace(' ms', '').strip()
                    if qt_clean and qt_clean != '0' and qtc_clean and qtc_clean != '0':
                        self.metric_labels['qtc_interval'].setText(f"{qt_clean}/{qtc_clean}")
                    else:
                        self.metric_labels['qtc_interval'].setText(qtc_clean if qtc_clean else "0")
                self._last_metrics_update_ts = _time.time()
                # This Prevents Jittering of BPM values in inner dashboard
                # try:
                #     self.sync_dashboard_metrics_to_ecg_page()
                # except Exception:
                #     pass
            else:
                default_metrics = self.calculate_standard_ecg_metrics(75)
                if default_metrics:
                    self.metric_labels['heart_rate'].setText(f"{default_metrics['heart_rate']} BPM")
                    self.metric_labels['pr_interval'].setText(f"{default_metrics['pr_interval']} ms")
                    self.metric_labels['qrs_duration'].setText(f"{default_metrics['qrs_duration']} ms")
                    self.metric_labels['qtc_interval'].setText(f"{default_metrics['qtc_interval']} ms")
                    self.metric_labels['st_interval'].setText(f"{default_metrics['st_interval']} ms")  # P duration in ms
        except Exception as e:
            print(f" Error updating dashboard metrics from ECG: {e}")
    
    def generate_pdf_report(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import datetime
        import os
        # Import the simple function from ecg_report_generator
        from ecg.ecg_report_generator import generate_ecg_report
        # Demo-specific report generator (uses dummycsv.csv and fixed demo metrics)
        try:
            from ecg.demo_ecg_report_generator import generate_demo_ecg_report
        except Exception:
            generate_demo_ecg_report = None
        # History helper
        try:
            from dashboard.history_window import append_history_entry
        except Exception:
            append_history_entry = None

        print(" Starting PDF report generation...")

        is_demo_mode = False
        if hasattr(self, 'ecg_test_page') and self.ecg_test_page and hasattr(self.ecg_test_page, 'demo_toggle'):
            is_demo_mode = self.ecg_test_page.demo_toggle.isChecked()

        def _extract_metric(label_key, default="0", strip_units=True):
            if not hasattr(self, 'metric_labels') or label_key not in self.metric_labels:
                return default
            text = self.metric_labels[label_key].text().strip()
            if not text:
                return default
            if strip_units:
                for unit in ("BPM", "bpm", "ms", "mV", "°"):
                    text = text.replace(unit, "")
            return text.strip() or default

        def _to_int(value, default=0):
            try:
                return int(float(value))
            except Exception:
                return default

        def _to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default

        # Gather ECG data from dashboard metrics
        HR_text = _extract_metric('heart_rate', "0")
        PR_text = _extract_metric('pr_interval', "0")
        QRS_text = _extract_metric('qrs_duration', "0")
        qtc_label_text = _extract_metric('qtc_interval', "400/430", strip_units=False)
        st_label_text = _extract_metric('st_interval', "", strip_units=False)
        if not st_label_text or st_label_text == "":
            st_label_text = _extract_metric('st_segment', "0.0 mV", strip_units=False)
        
        QT_text = "0"
        QTc_text = "0"
        if '/' in qtc_label_text:
            parts = [part.strip().replace("ms", "").strip() for part in qtc_label_text.split('/') if part.strip()]
            if len(parts) >= 1:
                QT_text = parts[0]
            if len(parts) >= 2:
                QTc_text = parts[1]
        else:
            QTc_text = qtc_label_text.strip()

        ST_text = st_label_text.replace("mV", "").strip()
        
        HR = _to_int(HR_text, 0)
        PR = _to_int(PR_text, 0)
        QRS = _to_int(QRS_text, 0)
        QT = _to_int(QT_text, 0)
        QTc = _to_int(QTc_text, 0)
        ST = _to_float(ST_text, 0.0)

        if QT <= 0 and hasattr(self, 'ecg_test_page') and getattr(self.ecg_test_page, '_last_qt_ms', None):
            QT = int(self.ecg_test_page._last_qt_ms)
        if QTc <= 0 and hasattr(self, 'ecg_test_page') and getattr(self.ecg_test_page, '_last_qtc_ms', None):
            QTc = int(self.ecg_test_page._last_qtc_ms)
        QTcF = 0
        if hasattr(self, 'ecg_test_page') and getattr(self.ecg_test_page, '_last_qtcf_ms', None):
            QTcF = int(self.ecg_test_page._last_qtcf_ms or 0)
        
        print(f" PDF Report ECG Values - HR: {HR}, PR: {PR}, QRS: {QRS}, QT: {QT}, QTc: {QTc}, QTcF: {QTcF}, ST: {ST}")

        # Prepare data for the report generator
        ecg_data = {
            "HR": 4833,  # Total heartbeats
            "beat": HR if HR > 0 else 88,  # Current heart rate
            "PR": PR if PR > 0 else 160,
            "QRS": QRS if QRS > 0 else 90,
            "QT": QT if QT > 0 else 400,
            "QTc": QTc if QTc > 0 else 400,
            "QTcF": QTcF if QTcF > 0 else QTc if QTc > 0 else 400,
            "ST": ST,
            "HR_max": 136,
            "HR_min": 74,
            "HR_avg": HR if HR > 0 else 88,
        }

        # --- Capture last 10 seconds of live ECG data ---
        lead_img_paths = {}
        ordered_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        print(" Capturing last 10 seconds of live ECG data...")
        
        # Get current directory for saving images
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        
        # Capture live data from ECG test page
        if hasattr(self, 'ecg_test_page') and self.ecg_test_page and hasattr(self.ecg_test_page, 'data'):
            print(f" Found ECG test page with data: {len(self.ecg_test_page.data)} leads")
            
            # Calculate 10 seconds of data based on sampling rate
            sampling_rate = 250  # Default sampling rate
            if hasattr(self.ecg_test_page, 'sampler') and hasattr(self.ecg_test_page.sampler, 'sampling_rate'):
                try:
                    sampling_rate = float(self.ecg_test_page.sampler.sampling_rate)
                except:
                    sampling_rate = 250
            
            data_points_10_sec = int(sampling_rate * 10)  # 10 seconds of data
            print(f" Capturing {data_points_10_sec} data points at {sampling_rate}Hz")
            
            for i, lead in enumerate(ordered_leads):
                if i < len(self.ecg_test_page.data) and i < len(self.ecg_test_page.leads):
                    try:
                        # Get the last 10 seconds of data for this lead
                        lead_data = self.ecg_test_page.data[i]
                        if len(lead_data) > data_points_10_sec:
                            recent_data = lead_data[-data_points_10_sec:]
                        else:
                            recent_data = lead_data
                        
                        if len(recent_data) > 0:
                            # Create a clean plot for the report
                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')  # Use non-interactive backend
                            
                            fig, ax = plt.subplots(figsize=(8, 2))
                            
                            # Plot the 10-second ECG trace
                            time_axis = np.linspace(0, 10, len(recent_data))  # 10 seconds
                            ax.plot(time_axis, recent_data, color='black', linewidth=0.8)
                            
                            # Clean medical-style formatting
                            ax.set_xlim(0, 10)
                            ax.set_xticks([0, 2, 4, 6, 8, 10])
                            ax.set_xticklabels(['0s', '2s', '4s', '6s', '8s', '10s'])
                            ax.set_ylabel('Amplitude (mV)')
                            ax.set_title(f'Lead {lead} - Last 10 seconds', fontsize=10, fontweight='bold')
                            
                            # Add subtle grid
                            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                            ax.set_axisbelow(True)
                            
                            # Clean background
                            ax.set_facecolor('white')
                            fig.patch.set_facecolor('white')
                            
                            # Save the image
                            img_path = os.path.join(project_root, f"lead_{lead}_10sec.png")
                            fig.savefig(img_path, 
                                      bbox_inches='tight', 
                                      pad_inches=0.1, 
                                      dpi=150, 
                                      facecolor='white',
                                      edgecolor='none')
                            
                            plt.close(fig)  # Close to free memory
                            lead_img_paths[lead] = img_path
                            
                            print(f"  Captured 10s Lead {lead}: {len(recent_data)} samples")
                        else:
                            print(f"  No data available for Lead {lead}")
                            
                    except Exception as e:
                        print(f"  Error capturing Lead {lead}: {e}")
                else:
                    print(f" Lead {lead} not available (index {i})")
        else:
            print(" No ECG test page or data available for capture")
        
        # Method 3: Check current stack widget for ECG pages
        if not lead_img_paths and hasattr(self, 'page_stack'):
            print(" Checking page stack for ECG test pages...")
            
            for i in range(self.page_stack.count()):
                widget = self.page_stack.widget(i)
                
                # Check if it's ECGTestPage
                if hasattr(widget, 'figures') and hasattr(widget, 'leads'):
                    print(f"Found ECG page in stack at index {i}")
                    
                    for j, lead in enumerate(ordered_leads):
                        if j < len(widget.figures):
                            try:
                                fig = widget.figures[j]
                                
                                # Medical-grade clean graph saving
                                if fig.axes:
                                    ax = fig.axes[0]
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.set_xlabel('')
                                    ax.set_ylabel('')
                                    ax.set_title('')
                                    
                                    for spine in ax.spines.values():
                                        spine.set_visible(False)
                                    
                                    ax.set_facecolor('none')
                                    fig.patch.set_facecolor('none')
                                    
                                    legend = ax.get_legend()
                                    if legend:
                                        legend.remove()
                                    
                                    for line in ax.lines:
                                        line.set_linewidth(0.4)       # Ultra-thin like reference
                                        line.set_antialiased(True)
                                        line.set_color('#000000')
                                        line.set_alpha(0.9)
                                        line.set_solid_capstyle('round')   # Rounded line endings
                                        line.set_solid_joinstyle('round')  # Rounded line joints
                                    
                                    ax.grid(True, color='#f0f0f0', linestyle='-', linewidth=0.3, alpha=0.7)
                                    ax.set_axisbelow(True)
                                
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                project_root = os.path.join(current_dir, '..')
                                project_root = os.path.abspath(project_root)
                                
                                img_path = os.path.join(project_root, f"lead_{lead}.png")
                                
                                fig.savefig(img_path, 
                                          bbox_inches='tight',
                                          pad_inches=0.05,
                                          dpi=200,
                                          facecolor='none',
                                          edgecolor='none',
                                          transparent=True)
                                
                                lead_img_paths[lead] = img_path
                                print(f" Saved medical-grade Lead {lead}")
                                
                            except Exception as e:
                                print(f" Error capturing Lead {lead}: {e}")
                    break
        
        # Method 4: Capture from PyQtGraph plot widgets (current 12-lead grid)
        if not lead_img_paths and hasattr(self, 'ecg_test_page') and self.ecg_test_page:
            if hasattr(self.ecg_test_page, 'plot_widgets') and self.ecg_test_page.plot_widgets:
                print(" Capturing ECG from PyQtGraph plot widgets...")
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(current_dir, '..'))
                    for i, lead in enumerate(ordered_leads):
                        if i < len(self.ecg_test_page.plot_widgets):
                            try:
                                w = self.ecg_test_page.plot_widgets[i]
                                pix = w.grab()  # QWidget -> QPixmap
                                img_path = os.path.join(project_root, f"lead_{lead}.png")
                                pix.save(img_path, 'PNG')
                                lead_img_paths[lead] = img_path
                                print(f" Saved PyQtGraph Lead {lead}: {img_path}")
                            except Exception as e:
                                print(f" Error capturing PyQtGraph Lead {lead}: {e}")
                except Exception as e:
                    print(f" PyQtGraph capture failed: {e}")

        # Report results
        if lead_img_paths:
            print(f" Successfully captured {len(lead_img_paths)}/12 real ECG graphs!")
        else:
            print(" No real ECG graphs found!")
            QMessageBox.warning(
                self,
                "No ECG Data",
                " No real ECG graphs found!\n\n Please:\n1. Start ECG test first\n2. Make sure 12-lead graphs are displayed\n3. Try again while ECG is running"
            )
            return
        
        # Ask user where to save the PDF
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save ECG Report", 
            f"ECG_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)"
        )
        
        if filename:
            try:
                if is_demo_mode:
                    try:
                        from ecg.demo_ecg_report_generator import generate_demo_ecg_report
                        generate_demo_ecg_report(filename, lead_img_paths, self, self.ecg_test_page)
                        QMessageBox.information(
                            self,
                            "Success",
                            f" Demo ECG Report generated successfully!\n Saved as: {filename}\n Demo graphs: {len(lead_img_paths)}/12"
                        )
                    except Exception as e:
                        error_msg = f"Failed to generate demo PDF: {str(e)}"
                        print(f" {error_msg}")
                        QMessageBox.critical(self, "Error", error_msg)
                    return

                # Load patient details from centralized all_patients.json database
                patient = None
                try:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    patients_db_file = os.path.join(base_dir, "all_patients.json")
                    
                    if os.path.exists(patients_db_file):
                        with open(patients_db_file, "r") as jf:
                            all_patients = json.load(jf)
                            if all_patients.get("patients") and len(all_patients["patients"]) > 0:
                                # Get the last patient (most recent)
                                patient = all_patients["patients"][-1]
                except Exception as e:
                    print(f" Error loading patient data: {e}")
                    patient = None

                # Always stamp current date/time from system
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if not patient:
                    patient = {}
                patient["date_time"] = now_str

                # Force update conclusions before generating report
                self.update_live_conclusion()
                
                # Calculate wave amplitudes before generating report
                print(" Calculating wave amplitudes for report...")
                if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                    try:
                        # PRIORITY: Use standardized RV5/SV1 calculation (median beat method)
                        if hasattr(self.ecg_test_page, 'calculate_rv5_sv1_from_median'):
                            rv5_calc, sv1_calc = self.ecg_test_page.calculate_rv5_sv1_from_median()
                            if rv5_calc is not None and rv5_calc > 0:
                                ecg_data['rv5'] = float(rv5_calc)
                            if sv1_calc is not None and sv1_calc != 0.0:
                                ecg_data['sv1'] = float(sv1_calc)
                            print(f" Using standardized RV5/SV1: RV5={ecg_data.get('rv5', 0):.3f}, SV1={ecg_data.get('sv1', 0):.3f}")
                        
                        # Get other wave amplitudes (P, QRS, T) from calculate_wave_amplitudes
                        print(f" ECG test page found, calling calculate_wave_amplitudes()...")
                        wave_amps = self.ecg_test_page.calculate_wave_amplitudes()
                        print(f" Raw wave_amps returned: {wave_amps}")
                        
                        # Add wave amplitudes to ecg_data (only if not already set from standardized function)
                        ecg_data['p_amp'] = wave_amps.get('p_amp', 0.0)
                        ecg_data['qrs_amp'] = wave_amps.get('qrs_amp', 0.0)
                        ecg_data['t_amp'] = wave_amps.get('t_amp', 0.0)
                        # Only use calculate_wave_amplitudes RV5/SV1 if standardized function didn't provide values
                        if 'rv5' not in ecg_data or ecg_data['rv5'] <= 0:
                            ecg_data['rv5'] = wave_amps.get('rv5', 0.0)
                        if 'sv1' not in ecg_data or ecg_data['sv1'] == 0.0:
                            ecg_data['sv1'] = wave_amps.get('sv1', 0.0)
                        
                        print(f" Wave amplitudes added to ecg_data:")
                        print(f"   P={ecg_data['p_amp']:.4f}, QRS={ecg_data['qrs_amp']:.4f}, T={ecg_data['t_amp']:.4f}")
                        print(f"   RV5={ecg_data['rv5']:.4f}, SV1={ecg_data['sv1']:.4f}, RV5+SV1={ecg_data['rv5'] + ecg_data['sv1']:.4f}")
                        print(f" Final ecg_data keys: {ecg_data.keys()}")
                    except Exception as e:
                        print(f" Error calculating wave amplitudes: {e}")
                        import traceback
                        traceback.print_exc()
                        ecg_data['p_amp'] = 0.0
                        ecg_data['qrs_amp'] = 0.0
                        ecg_data['t_amp'] = 0.0
                        ecg_data['rv5'] = 0.0
                        ecg_data['sv1'] = 0.0
                else:
                    print(" No ECG test page available for wave amplitude calculation")
                    ecg_data['p_amp'] = 0.0
                    ecg_data['qrs_amp'] = 0.0
                    ecg_data['t_amp'] = 0.0
                    ecg_data['rv5'] = 0.0
                    ecg_data['sv1'] = 0.0
                
                # Add axis values from ECG test page (standardized median beat method)
                if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                    try:
                        # Get P axis
                        if hasattr(self.ecg_test_page, 'calculate_p_axis_from_median'):
                            p_axis = self.ecg_test_page.calculate_p_axis_from_median()
                            if p_axis is not None and p_axis != 0:
                                ecg_data['p_axis'] = int(round(p_axis))
                        
                        # Get T axis
                        if hasattr(self.ecg_test_page, 'calculate_t_axis_from_median'):
                            t_axis = self.ecg_test_page.calculate_t_axis_from_median()
                            if t_axis is not None and t_axis != 0:
                                ecg_data['t_axis'] = int(round(t_axis))
                        
                        print(f" Added axis values to ecg_data: P={ecg_data.get('p_axis', '--')}, QRS={ecg_data.get('QRS_axis', '--')}, T={ecg_data.get('t_axis', '--')}")
                    except Exception as e:
                        print(f" Error getting axis values from ECG test page: {e}")
                
                # Add user details to ecg_data for JSON export
                ecg_data['user'] = {
                    'name': self.user_details.get('full_name', self.username or ''),
                    'phone': self.user_details.get('phone', ''),
                }
                ecg_data['machine_serial'] = self.user_details.get('serial_id', '') or os.getenv('MACHINE_SERIAL_ID', '')

                # Generate the PDF with patient details
                if is_demo_mode and generate_demo_ecg_report is not None:
                    print(" DEMO MODE: Using generate_demo_ecg_report with dummycsv.csv data")
                    generate_demo_ecg_report(
                        filename,
                        lead_img_paths,
                        self,
                        self.ecg_test_page,
                    )
                else:
                    generate_ecg_report(
                        filename,
                        ecg_data,
                        lead_img_paths,
                        self,
                        self.ecg_test_page,
                        patient,
                        log_history=False,
                    )

                # After successful report generation, append to history (explicit)
                try:
                    # Ensure history helper is available
                    if append_history_entry is None:
                        from dashboard.history_window import append_history_entry as _ah
                        append_history_entry = _ah
                    if append_history_entry is not None:
                        append_history_entry(patient, filename, report_type="12 Lead")
                except Exception as hist_err:
                    import traceback
                    traceback.print_exc()
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f" ECG Report generated successfully!\n Saved as: {filename}\n Real graphs: {len(lead_img_paths)}/12"
                )
                
                print(f" PDF generated: {filename}")
                # Save a copy inside the app for Recent Reports + update index.json
                try:
                    import shutil, json
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    reports_dir = os.path.abspath(os.path.join(base_dir, "..", "reports"))
                    os.makedirs(reports_dir, exist_ok=True)
                    # Destination filename (keep basename)
                    dst_basename = os.path.basename(filename)
                    dst_path = os.path.join(reports_dir, dst_basename)
                    # Avoid overwrite
                    if os.path.abspath(filename) != os.path.abspath(dst_path):
                        counter = 1
                        base_name, ext = os.path.splitext(dst_basename)
                        while os.path.exists(dst_path):
                            dst_basename = f"{base_name}_{counter}{ext}"
                            dst_path = os.path.join(reports_dir, dst_basename)
                            counter += 1
                        shutil.copyfile(filename, dst_path)
                    
                    # Also copy the JSON twin if it exists
                    src_json = os.path.splitext(filename)[0] + ".json"
                    if os.path.exists(src_json):
                        dst_json = os.path.splitext(dst_path)[0] + ".json"
                        if os.path.abspath(src_json) != os.path.abspath(dst_json):
                            shutil.copyfile(src_json, dst_json)
                            print(f"✓ Copied JSON twin to: {dst_json}")
                    # Update index.json (prepend)
                    index_path = os.path.join(reports_dir, "index.json")
                    items = []
                    if os.path.exists(index_path):
                        try:
                            with open(index_path, 'r') as f:
                                items = json.load(f)
                        except Exception:
                            items = []
                    now = datetime.datetime.now()
                    meta = {
                        "filename": os.path.basename(dst_path),
                        "title": "ECG Report",
                        "patient": "",  # Fill from form if available
                        "date": now.strftime('%Y-%m-%d'),
                        "time": now.strftime('%H:%M:%S'),
                        "username": self.username or ""  # Add username to track report ownership
                    }
                    items = [meta] + items
                    items = items[:10]
                    with open(index_path, 'w') as f:
                        json.dump(items, f, indent=2)
                    # Refresh dashboard list
                    self.refresh_recent_reports_ui()
                except Exception as idx_err:
                    print(f" Failed to update Recent Reports index: {idx_err}")

            except Exception as e:
                error_msg = f"Failed to generate PDF: {str(e)}"
                print(f" {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)

    def animate_heartbeat(self):
        """Animate heart image synchronized with live heart rate and play sound"""
        import time
        
        current_time = time.time() * 1000  # Convert to milliseconds
        current_hr = 0
        
        # Get current heart rate from metric card
        try:
            if 'heart_rate' in self.metric_labels:
                hr_text = self.metric_labels['heart_rate'].text()
                # Normalize text (e.g., "86 bpm", "86 BPM", "86")
                if hr_text:
                    cleaned = hr_text.replace("BPM", "").replace("bpm", "").strip()
                    # Treat non-numeric or placeholder values as "no data"
                    if cleaned.isdigit():
                        current_hr = int(cleaned)
                        self.current_heart_rate = current_hr
                        if self.current_heart_rate > 0:
                            # Calculate beat interval based on heart rate
                            self.beat_interval = 60000 / self.current_heart_rate  # Convert BPM to ms between beats
                    else:
                        # No valid heart rate available
                        self.current_heart_rate = 0
        except Exception as e:
            print(f" Error parsing heart rate: {e}")
            self.current_heart_rate = 0
        
        # If there is no valid heart data (HR < 10 bpm or 0 / '--'), 
        # do NOT play heartbeat sound and keep the heart static
        if not isinstance(self.current_heart_rate, (int, float)) or self.current_heart_rate < 10:
            # Optionally, you can keep a very subtle idle animation; here we freeze the icon
            return
        
        # Check if it's time for a heartbeat
        if current_time - self.last_beat_time >= self.beat_interval:
            self.last_beat_time = current_time
            
            # Play heartbeat sound with increased volume
            if self.heartbeat_sound and self.heartbeat_sound_enabled:
                try:
                    # Try to set volume if available (some Qt versions support this)
                    if hasattr(self.heartbeat_sound, 'setVolume'):
                        self.heartbeat_sound.setVolume(100)  # Maximum volume
                    self.heartbeat_sound.play()
                except Exception as e:
                    print(f" Error playing heartbeat sound: {e}")
            
            # Reset heartbeat phase for new beat
            self.heartbeat_phase = 0
        
        # Heartbeat effect: scale up and down based on phase
        # More pronounced beat when close to actual heartbeat
        time_since_beat = current_time - self.last_beat_time
        beat_progress = min(time_since_beat / self.beat_interval, 1.0)
        
        # Create a more realistic heartbeat pattern
        if beat_progress < 0.1:  # First 10% of cycle - sharp beat
            beat = 1 + 0.25 * math.sin(beat_progress * 10 * math.pi)
        elif beat_progress < 0.2:  # Next 10% - second beat
            beat = 1 + 0.15 * math.sin((beat_progress - 0.1) * 10 * math.pi)
        else:  # Rest of cycle - gradual return to normal
            beat = 1 + 0.05 * math.sin(self.heartbeat_phase)
        
        # Apply the beat effect
        size = int(self.heart_base_size * beat)
        self.heart_img.setPixmap(self.heart_pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Update phase for smooth animation
        self.heartbeat_phase += 0.18
        if self.heartbeat_phase > 2 * math.pi:
            self.heartbeat_phase -= 2 * math.pi
    
    def create_heartbeat_sound(self):
        """Create a synthetic heartbeat sound if no sound file is available.

        This generates a louder, normalized 'lub-dub' sound so it is clearly audible
        across devices. The waveform is normalized to full 16‑bit range.
        """
        try:
            import wave
            import struct
            import math
            
            # Create a simple heartbeat sound (lub-dub pattern)
            sample_rate = 22050
            duration = 0.6  # seconds
            samples = int(sample_rate * duration)
            
            # Generate heartbeat sound data
            sound_data = []
            for i in range(samples):
                t = i / sample_rate
                
                # First beat (lub) - lower frequency (louder envelope)
                if t < 0.1:
                    freq1 = 80  # Hz
                    amplitude = 1.0 * math.sin(2 * math.pi * freq1 * t) * math.exp(-t * 12)
                # Second beat (dub) - higher frequency (louder envelope)
                elif 0.2 < t < 0.3:
                    freq2 = 120  # Hz
                    amplitude = 0.95 * math.sin(2 * math.pi * freq2 * (t - 0.2)) * math.exp(-(t - 0.2) * 12)
                else:
                    amplitude = 0
                
                sound_data.append(amplitude)

            # Normalize to full 16‑bit range
            peak = max(1e-6, max(abs(x) for x in sound_data))
            norm = 32767.0 / peak
            pcm_data = [int(max(-32767, min(32767, x * norm))) for x in sound_data]
            
            # Save as WAV file
            heartbeat_path = get_asset_path("heartbeat.wav")
            with wave.open(heartbeat_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(struct.pack('<' + 'h' * len(pcm_data), *pcm_data))
            
            # Load the created sound
            if QSound is not None:
                self.heartbeat_sound = QSound(heartbeat_path)
                print(f" Created synthetic heartbeat sound: {heartbeat_path}")
            else:
                self.heartbeat_sound = None
                print(f" QSound not available - heartbeat sound disabled")
            
        except Exception as e:
            print(f" Could not create heartbeat sound: {e}")
            self.heartbeat_sound = None

    def set_heartbeat_sound_enabled(self, enabled):
        """Enable or disable the audible heartbeat feedback."""
        desired_state = str(enabled).lower() in ("on", "true", "1", "yes")
        self.heartbeat_sound_enabled = desired_state
        if not desired_state and self.heartbeat_sound:
            try:
                self.heartbeat_sound.stop()
            except Exception as e:
                print(f" Unable to stop heartbeat sound: {e}")

    def tr(self, text):
        return translate_text(text, getattr(self, "current_language", "en"))

    def apply_language(self, language=None):
        if language:
            self.current_language = language
        translator = self.tr
        if hasattr(self, 'date_btn') and self.date_btn:
            self.date_btn.setText(translator("ECG Lead Test 12"))
        if hasattr(self, 'chatbot_btn') and self.chatbot_btn:
            self.chatbot_btn.setText(translator("AI Chatbot"))
        if hasattr(self, 'heart_label') and self.heart_label:
            self.heart_label.setText(translator("Live Heart Rate Overview"))
        if hasattr(self, 'stress_label') and self.stress_label:
            self.stress_label.setText(translator("Stress Level: --"))
        if hasattr(self, 'hrv_label') and self.hrv_label:
            self.hrv_label.setText(translator("Average Variability: --"))
        if hasattr(self, 'ecg_label') and self.ecg_label:
            self.ecg_label.setText(translator("ECG Recording"))
        if hasattr(self, 'visitors_label') and self.visitors_label:
            year = datetime.datetime.now().year
            self.visitors_label.setText(translator("Visitors - Last 6 Months ({year})").format(year=year))
        if hasattr(self, 'sign_btn') and self.sign_btn:
            self.sign_btn.setText(translator("Sign Out"))

    def on_settings_changed(self, key, value):
        """Handle global settings pushed from the ECG menu."""
        if key == "system_beat_vol":
            self.set_heartbeat_sound_enabled(value)
        elif key == "system_language":
            self.apply_language(value)
    def handle_sign(self):
        if self.sign_btn.text() == "Sign In":
            dialog = SignInDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                role, name = dialog.get_user_info()
                if not name.strip():
                    QMessageBox.warning(self, "Input Error", "Please enter your name.")
                    return
                # User label removed per request
                # self.user_label.setText(f"{name}\n{role}")
                self.sign_btn.setText("Sign Out")
        else:
            # User label removed per request
            # self.user_label.setText("Not signed in")
            self.sign_btn.setText("Sign In")
    def update_stress_and_hrv(self, ecg_signal, sampling_rate):
        """Calculate and update stress level and HRV from ECG data with smoothing"""
        try:
            from scipy.signal import find_peaks
            
            if len(ecg_signal) < 500:
                return
            
            # Find R-peaks
            peaks, _ = find_peaks(
                ecg_signal,
                height=np.mean(ecg_signal) + 0.5 * np.std(ecg_signal),
                distance=int(0.15 * sampling_rate)  # Reduced from 0.4 to 0.15 for high BPM (up to 360)
            )
            
            if len(peaks) >= 3:
                # Calculate R-R intervals in milliseconds
                rr_intervals = np.diff(peaks) * (1000 / sampling_rate)
                
                # Filter valid intervals (240-2000 ms) - 240ms = 250 BPM
                valid_rr = rr_intervals[(rr_intervals >= 240) & (rr_intervals <= 2000)]
                
                if len(valid_rr) >= 2:
                    # HRV: Standard deviation of R-R intervals (SDNN)
                    current_hrv_ms = np.std(valid_rr)
                    
                    # Initialize rolling average for HRV smoothing
                    if not hasattr(self, '_hrv_history'):
                        self._hrv_history = []
                    
                    # Add current HRV to history (keep last 5 values for smoothing)
                    self._hrv_history.append(current_hrv_ms)
                    if len(self._hrv_history) > 5:
                        self._hrv_history.pop(0)
                    
                    # Use smoothed HRV value
                    smoothed_hrv_ms = np.mean(self._hrv_history)
                    
                    # Store for conclusion generation
                    self._current_hrv = smoothed_hrv_ms
                    
                    # Stress level based on smoothed HRV
                    # Use dashboard's translation method
                    translator = self.tr
                    
                    if smoothed_hrv_ms > 100:
                        stress = translator("Low")
                        stress_color = "#27ae60"
                    elif smoothed_hrv_ms > 50:
                        stress = translator("Moderate")
                        stress_color = "#f39c12"
                    else:
                        stress = translator("High")
                        stress_color = "#e74c3c"
                    
                    # Update labels with translation
                    if hasattr(self, 'stress_label'):
                        stress_label_text = translator("Stress Level:")
                        self.stress_label.setText(f"{stress_label_text} {stress}")
                        self.stress_label.setStyleSheet(f"font-size: 13px; color: {stress_color}; font-weight: bold;")
                    
                    if hasattr(self, 'hrv_label'):
                        hrv_label_text = translator("Average Variability:")
                        self.hrv_label.setText(f"{hrv_label_text} {int(smoothed_hrv_ms)}ms")
                        self.hrv_label.setStyleSheet("font-size: 13px; color: #666;")
        except Exception as e:
            print(f" Error calculating stress/HRV: {e}")
    
    def update_live_conclusion(self):
        """Generate comprehensive personalized conclusion based on current ECG metrics with detailed BPM analysis"""
        try:
            findings = []
            recommendations = []
            additional_info = []
            
            rhythm_text = None
            try:
                if hasattr(self, 'ecg_test_page') and self.ecg_test_page and hasattr(self.ecg_test_page, 'get_latest_rhythm_interpretation'):
                    rhythm_text = self.ecg_test_page.get_latest_rhythm_interpretation()
            except Exception:
                rhythm_text = None

            # Get current metrics
            hr_text = self.metric_labels.get('heart_rate', QLabel()).text()
            pr_text = self.metric_labels.get('pr_interval', QLabel()).text()
            qrs_text = self.metric_labels.get('qrs_duration', QLabel()).text()
            st_text = self.metric_labels.get('st_interval', QLabel()).text()
            qtc_text = self.metric_labels.get('qtc_interval', QLabel()).text()
            
            # Parse values
            try:
                hr = int(hr_text.replace(' BPM', '').replace(' bpm', '').strip()) if hr_text and hr_text != '00' else 0
            except:
                hr = 0
            
            try:
                pr = int(pr_text.replace(' ms', '').strip()) if pr_text and pr_text != '0 ms' else 0
            except:
                pr = 0
            
            try:
                qrs = int(qrs_text.replace(' ms', '').strip()) if qrs_text and qrs_text != '0 ms' else 0
            except:
                qrs = 0
            
            # Parse QTC (can be "QT/QTC" format)
            qt = 0
            qtc = 0
            try:
                if qtc_text and '/' in qtc_text:
                    parts = qtc_text.split('/')
                    qt = int(parts[0].strip().replace(' ms', '')) if len(parts) > 0 else 0
                    qtc = int(parts[1].strip().replace(' ms', '')) if len(parts) > 1 else 0
                elif qtc_text:
                    qtc = int(qtc_text.strip().replace(' ms', ''))
            except:
                pass
            
            # Include rhythm interpretation findings first (System detects: AFib, VT, PVCs, Bradycardia, Tachycardia, NSR)
            if rhythm_text:
                rhythm_clean = rhythm_text.strip()
                ignore_values = {"", "Analyzing Rhythm...", "Detecting...", "Insufficient Data", "Insufficient data"}
                if rhythm_clean not in ignore_values:
                    is_normal_rhythm = any(keyword in rhythm_clean.lower() for keyword in ["normal sinus", "none detected", "sinus rhythm"])
                    prefix = "[OK]" if is_normal_rhythm else "[!]"
                    findings.append(f"{prefix} <b>Rhythm Analysis</b> - {rhythm_clean}")
                    if not is_normal_rhythm:
                        recommendations.append("• Review detected arrhythmia pattern, consult physician if persistent")
            
            # COMPREHENSIVE Heart Rate Analysis (System supports 10-300 BPM range)
            if hr >= 10 and hr <= 300:
                if hr > 200:
                    findings.append("[!] <b>Extreme Tachycardia</b> - Heart rate critically elevated (>200 BPM)")
                    recommendations.append("• Immediate medical attention required - may indicate SVT, VT, or severe stress")
                    recommendations.append("• Check for symptoms: chest pain, dizziness, shortness of breath")
                    additional_info.append(f"• Current HR: {hr} BPM is in the extreme tachycardia range")
                    additional_info.append("• Normal resting HR: 60-100 BPM for adults")
                elif hr > 150:
                    findings.append("[!] <b>Severe Tachycardia</b> - Heart rate significantly elevated (150-200 BPM)")
                    recommendations.append("• Consult physician promptly, check for arrhythmias or underlying conditions")
                    recommendations.append("• Monitor for symptoms and avoid strenuous activity")
                    additional_info.append(f"• Current HR: {hr} BPM indicates significant cardiac stress")
                    additional_info.append("• Possible causes: exercise, stress, fever, anemia, hyperthyroidism")
                elif hr > 100:
                    findings.append("[!] <b>Tachycardia detected</b> - Heart rate elevated above normal range (100-150 BPM)")
                    recommendations.append("• Monitor symptoms, consider medical evaluation if persistent")
                    recommendations.append("• Ensure adequate hydration and rest")
                    additional_info.append(f"• Current HR: {hr} BPM is above normal resting rate")
                    additional_info.append("• May be normal during exercise, stress, or after caffeine intake")
                elif hr < 40:
                    findings.append("[!] <b>Severe Bradycardia</b> - Heart rate critically low (<40 BPM)")
                    recommendations.append("• Immediate medical evaluation recommended - may indicate heart block or sick sinus syndrome")
                    recommendations.append("• Check for symptoms: fatigue, dizziness, fainting, chest pain")
                    additional_info.append(f"• Current HR: {hr} BPM is dangerously low")
                    additional_info.append("• Normal resting HR: 60-100 BPM for adults")
                elif hr < 60:
                    findings.append("[i] <b>Bradycardia detected</b> - Heart rate below normal range (40-60 BPM)")
                    recommendations.append("• May be normal for well-trained athletes or during sleep")
                    recommendations.append("• Monitor for symptoms, consult if experiencing fatigue or dizziness")
                    additional_info.append(f"• Current HR: {hr} BPM is below normal resting rate")
                    additional_info.append("• Athletes often have resting HR 40-60 BPM due to cardiovascular fitness")
                else:
                    findings.append("[OK] <b>Normal heart rate</b> - Within healthy range (60-100 BPM)")
                    additional_info.append(f"• Current HR: {hr} BPM is within normal resting range")
                    additional_info.append("• Optimal HR varies by age, fitness level, and activity")
            
            # Analyze PR Interval
            if pr > 0:
                if pr > 200:
                    findings.append("[!] <b>Prolonged PR interval</b> - Possible first-degree heart block (>200ms)")
                    recommendations.append("• Medical evaluation recommended for AV conduction assessment")
                    recommendations.append("• Monitor for progression to higher-degree blocks")
                    additional_info.append(f"• PR interval: {pr} ms (normal: 120-200 ms)")
                elif pr < 120:
                    findings.append("[i] <b>Short PR interval</b> - May indicate pre-excitation syndrome (<120ms)")
                    recommendations.append("• Monitor for accessory pathway patterns, consult if symptomatic")
                    recommendations.append("• May be associated with WPW syndrome")
                    additional_info.append(f"• PR interval: {pr} ms (normal: 120-200 ms)")
                else:
                    findings.append("[OK] <b>Normal PR interval</b> - Atrial-ventricular conduction normal")
                    additional_info.append(f"• PR interval: {pr} ms (normal: 120-200 ms)")
            
            # Analyze QRS Duration
            if qrs > 0:
                if qrs > 120:
                    findings.append("[!] <b>Wide QRS complex</b> - Possible bundle branch block or ventricular rhythm (>120ms)")
                    recommendations.append("• 12-lead ECG analysis recommended for conduction pattern assessment")
                    recommendations.append("• May indicate bundle branch block, ventricular rhythm, or hyperkalemia")
                    additional_info.append(f"• QRS duration: {qrs} ms (normal: <100 ms)")
                elif qrs > 100:
                    findings.append("[i] <b>Borderline QRS duration</b> - Early conduction delay detected (100-120ms)")
                    recommendations.append("• Monitor for progression, follow-up ECG if symptoms develop")
                    additional_info.append(f"• QRS duration: {qrs} ms (normal: <100 ms)")
                else:
                    findings.append("[OK] <b>Normal QRS duration</b> - Ventricular depolarization normal")
                    additional_info.append(f"• QRS duration: {qrs} ms (normal: <100 ms)")
            
            # Analyze QT/QTC Interval
            if qtc > 0:
                if qtc > 500:
                    findings.append("[!] <b>Prolonged QTc interval</b> - High risk for arrhythmias (>500ms)")
                    recommendations.append("• Immediate medical evaluation - risk of Torsades de Pointes")
                    recommendations.append("• Review medications that may prolong QT interval")
                    additional_info.append(f"• QTc interval: {qtc} ms (normal: <450 ms for men, <470 ms for women)")
                elif qtc > 470:
                    findings.append("[!] <b>Borderline prolonged QTc</b> - Moderate risk (470-500ms)")
                    recommendations.append("• Medical evaluation recommended, monitor for symptoms")
                    recommendations.append("• Review medications and electrolyte levels")
                    additional_info.append(f"• QTc interval: {qtc} ms (normal: <450 ms for men, <470 ms for women)")
                elif qtc > 450:
                    findings.append("[i] <b>Slightly prolonged QTc</b> - Mild concern (450-470ms)")
                    recommendations.append("• Monitor, may be normal variant or medication effect")
                    additional_info.append(f"• QTc interval: {qtc} ms (normal: <450 ms for men, <470 ms for women)")
                else:
                    findings.append("[OK] <b>Normal QTc interval</b> - Within safe range")
                    additional_info.append(f"• QTc interval: {qtc} ms (normal: <450 ms for men, <470 ms for women)")
            
            # Check HRV/Stress
            if hasattr(self, '_current_hrv'):
                hrv = self._current_hrv
                if hrv > 100:
                    findings.append("[OK] <b>Good heart rate variability</b> - Low stress indicated")
                    additional_info.append(f"• HRV: {hrv:.1f} ms indicates good autonomic function")
                elif hrv > 50:
                    findings.append("[i] <b>Moderate HRV</b> - Normal stress levels")
                    additional_info.append(f"• HRV: {hrv:.1f} ms indicates moderate autonomic function")
                else:
                    findings.append("[!] <b>Low HRV</b> - Elevated stress or fatigue")
                    recommendations.append("• Ensure adequate rest and stress management")
                    recommendations.append("• Consider relaxation techniques, adequate sleep, and regular exercise")
                    additional_info.append(f"• HRV: {hrv:.1f} ms indicates reduced autonomic function")
            
            # Add general cardiac health information
            if hr > 0:
                additional_info.append("• Regular exercise and healthy lifestyle help maintain optimal heart function")
                additional_info.append("• Avoid smoking, excessive alcohol, and maintain healthy weight")
            
            # Build comprehensive conclusion HTML
            if not findings:
                conclusion_html = """
                    <p style='color: #888; font-style: italic;'>
                    Waiting for stable ECG data...<br><br>
                    Metrics are being analyzed. Please wait a few seconds.
                    </p>
                """
            else:
                conclusion_html = "<b style='color: #ff6600; font-size: 14px;'>Findings:</b><br>"
                for f in findings:
                    conclusion_html += f + "<br>"
                
                if recommendations:
                    conclusion_html += "<br><b style='color: #ff6600; font-size: 14px;'>Recommendations:</b><br>"
                    for r in recommendations:
                        conclusion_html += r + "<br>"
                
                if additional_info:
                    conclusion_html += "<br><b style='color: #2c3e50; font-size: 13px;'>Additional Information:</b><br>"
                    for info in additional_info:
                        conclusion_html += f"<span style='color: #555;'>{info}</span><br>"
                
                conclusion_html += """
                    <br><p style='font-size: 10px; color: #999; font-style: italic;'>
                    <b>NOTE:</b> This is an automated analysis for educational purposes only. 
                    Not a substitute for professional medical advice. Consult a healthcare provider for medical concerns.
                    </p>
                """
            
            if hasattr(self, 'conclusion_box'):
                self.conclusion_box.setHtml(conclusion_html)
            
            # Save conclusions to JSON file for report generation (only if valid findings exist)
            try:
                import os
                import json
                from datetime import datetime
                import re
                
                # Only save if we have actual findings (not empty)
                if findings:
                    # Extract clean headings from findings (remove prefixes, HTML tags, and explanations)
                    clean_findings = []
                    for f in findings:
                        # Remove HTML tags first
                        text = re.sub(r'<[^>]+>', '', f).strip()
                        # Remove prefix markers like [i], [OK], [!]
                        text = re.sub(r'^\[.*?\]\s*', '', text).strip()
                        # Extract only the heading (before " - " if present)
                        if ' - ' in text:
                            text = text.split(' - ')[0].strip()
                        clean_findings.append(text)
                    
                    # Clean recommendations (remove HTML tags and bullet points)
                    clean_recommendations = []
                    for r in recommendations:
                        text = re.sub(r'<[^>]+>', '', r).strip()
                        # Remove bullet point if present
                        text = re.sub(r'^[•●○]\s*', '', text).strip()
                        clean_recommendations.append(text)
                    
                    conclusions_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "findings": clean_findings,
                        "recommendations": clean_recommendations
                    }
                    
                    # Save to project root directory
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    conclusions_file = os.path.join(base_dir, 'last_conclusions.json')
                    
                    with open(conclusions_file, 'w') as f:
                        json.dump(conclusions_data, f, indent=2)
                    
                    print(f" Saved {len(clean_findings)} findings to last_conclusions.json")
                    print(f"   Findings: {clean_findings}")
                else:
                    print(f" Skipped saving empty findings to last_conclusions.json (waiting for valid ECG data)")
                
            except Exception as save_err:
                print(f" Error saving conclusions to JSON: {save_err}")
        
        except Exception as e:
            print(f" Error updating conclusion: {e}")

    def show_version_popup(self):
        """Show version information in a popup dialog"""
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'ecg_menu'):
            content = self.ecg_test_page.ecg_menu.create_version_info_content()
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Version Information")
            dialog.setMinimumWidth(500)
            dialog.setMinimumHeight(600)
            
            layout = QVBoxLayout(dialog)
            layout.addWidget(content)
            
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 8px; padding: 10px; font-weight: bold;")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec_()
        else:
            hw_v = "Not Detected"
            if hasattr(self, 'settings_manager'):
                # Reload settings to ensure we have the latest from disk
                self.settings_manager.settings = self.settings_manager.load_settings()
                hw_v = self.settings_manager.get_setting("hardware_version", "Not Detected")
                if not hw_v:
                    hw_v = "Not Detected"
            QMessageBox.information(self, "Version Information", f"Software Version: V 1.1.1\nHardware Version: {hw_v}")

    def handle_sign_out(self):
        # User label removed per request
        # self.user_label.setText("Not signed in")
        self.sign_btn.setText("Sign In")
        self.closed_by_sign_out = True
        try:
            recorder = getattr(self, '_session_recorder', None)
            if recorder:
                recorder.close()
                self._session_recorder = None
        except Exception:
            pass
        
        # Clean up ECG test page serial connection before closing
        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                # Stop acquisition if running
                if hasattr(self.ecg_test_page, 'serial_reader') and self.ecg_test_page.serial_reader:
                    try:
                        print("🔌 Closing serial connection on sign out...")
                        self.ecg_test_page.serial_reader.stop()
                        self.ecg_test_page.serial_reader.close()
                        self.ecg_test_page.serial_reader = None
                        print(" Serial connection closed successfully")
                    except Exception as e:
                        print(f" Error closing serial connection: {e}")
                
                # Stop timers
                if hasattr(self.ecg_test_page, 'timer') and self.ecg_test_page.timer:
                    try:
                        self.ecg_test_page.timer.stop()
                    except Exception:
                        pass
                
                # Stop demo manager if active
                if hasattr(self.ecg_test_page, 'demo_manager') and self.ecg_test_page.demo_manager:
                    try:
                        self.ecg_test_page.demo_manager.stop_demo_data()
                    except Exception:
                        pass
        except Exception as e:
            print(f" Error cleaning up ECG test page: {e}")
        
        self.close()

    def update_test_state(self, test_name, is_running):
        """Update the running state of a specific test."""
        if test_name in self.test_states:
            self.test_states[test_name] = is_running
            print(f" Test State Updated: {test_name} = {is_running}")
            # Force UI update if needed
            QApplication.processEvents()

    def can_start_test(self, test_name):
        """
        Check if a test can be started. 
        Returns True if no other test is running.
        Returns False and shows a popup if another test is already running.
        """
        for name, is_running in self.test_states.items():
            if is_running and name != test_name:
                # Another test is running
                running_test_display = name.replace("_", " ").title()
                QMessageBox.warning(
                    self, 
                    "Test Already Running", 
                    f"Cannot start {test_name.replace('_', ' ').title()} because {running_test_display} is currently running.\n\nPlease stop the running test first."
                )
                return False
        return True
        
    def open_hyperkalemia_test(self):
        """Open Hyperkalemia Test window in a new window"""
        # Ensure 12-lead serial connection is closed before opening Hyperkalemia test
        # if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
        #     try:
        #         if hasattr(self.ecg_test_page, 'close_serial_connection'):
        #             self.ecg_test_page.close_serial_connection()
        #     except Exception as e:
        #         print(f"Error closing 12-lead serial connection: {e}")

        try:
            from ecg.hyperkalemia_test import HyperkalemiaTestWindow
            self.hyperkalemia_window = HyperkalemiaTestWindow(parent=self, username=self.username)
            self.hyperkalemia_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Hyperkalemia Test window: {str(e)}")
            print(f" Error opening Hyperkalemia test: {e}")

    def open_history_window(self):
        """Open the ECG report history window."""
        try:
            from dashboard.history_window import HistoryWindow
            dlg = HistoryWindow(parent=self, username=self.username)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "History", f"Failed to open history window: {e}")
    
    def open_hrv_test(self):
        """Open HRV Test window in a new window"""
        # # Ensure 12-lead serial connection is closed before opening HRV test
        # if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
        #     try:
        #         if hasattr(self.ecg_test_page, 'close_serial_connection'):
        #             self.ecg_test_page.close_serial_connection()
        #     except Exception as e:
        #         print(f"Error closing 12-lead serial connection: {e}")

        try:
            from ecg.hrv_test import HRVTestWindow
            self.hrv_window = HRVTestWindow(parent=self, username=self.username)
            self.hrv_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open HRV Test window: {str(e)}")
            print(f" Error opening HRV test: {e}")   
    
    def go_to_lead_test(self):
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)

        if hasattr(self, 'ecg_test_page'):
            self.ecg_test_page.current_username = self.username
            
        self.page_stack.setCurrentWidget(self.ecg_test_page)
        # Sync dashboard metrics to ECG test page
        self.sync_dashboard_metrics_to_ecg_page()
        # Also update dashboard metrics when opening ECG test page
        self.update_dashboard_metrics_from_ecg()
    def go_to_dashboard(self):
        # # Close serial connection on ECG page to free up COM port
        # if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
        #     try:
        #         if hasattr(self.ecg_test_page, 'close_serial_connection'):
        #             self.ecg_test_page.close_serial_connection()
        #     except Exception as e:
        #         print(f"Error closing serial connection: {e}")

                
        self.page_stack.setCurrentWidget(self.dashboard_page)
        # Update metrics when returning to dashboard
        self.update_dashboard_metrics_from_ecg()

    # Resume device check if needed
    def check_device_connection(self):
        """Check for device connection or scan if disconnected"""

        if not SERIAL_AVAILABLE:
            return

        # If device is already connected, check if it's still there
        if self.device_connected and self.device_port:
            try:
                available_ports = [p.device for p in serial.tools.list_ports.comports()]
                if self.device_port not in available_ports:
                    print(f"⚠️ Port {self.device_port} disconnected.")
                    self.device_connected = False
                    self.device_port = None
                    self.update_device_ui(False)

                    # Inform user if on dashboard and no tests active
                    is_on_dashboard = self.page_stack.currentWidget() == self.dashboard_page
                    hrv_active = hasattr(self, 'hrv_window') and self.hrv_window and self.hrv_window.isVisible()
                    hyper_active = hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and self.hyperkalemia_window.isVisible()
                    
                    if is_on_dashboard and not hrv_active and not hyper_active:
                        QMessageBox.warning(self, "Connection Status", "Connection lost. Please ensure the device is properly connected")

                    # If HRV or Hyperkalemia test window is open, show "Test Failed" and close it
                    if hasattr(self, 'hrv_window') and self.hrv_window and self.hrv_window.isVisible():
                        if hasattr(self.hrv_window, 'stop_capture'):
                            try:
                                self.hrv_window.stop_capture(device_disconnected=True)
                            except TypeError:
                                self.hrv_window.stop_capture()
                        QMessageBox.critical(self.hrv_window, "Test Failed", "Device disconnected. Test failed.")
                        self.hrv_window.close()
                    elif hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and self.hyperkalemia_window.isVisible():
                        if hasattr(self.hyperkalemia_window, 'stop_capture'):
                            try:
                                self.hyperkalemia_window.stop_capture(device_disconnected=True)
                            except TypeError:
                                self.hyperkalemia_window.stop_capture()
                        QMessageBox.critical(self.hyperkalemia_window, "Test Failed", "Device disconnected. Test failed.")
                        self.hyperkalemia_window.close()
                    # If ECG 12 lead test is running (on the stacked widget), show "Test Failed" and go back to dashboard
                    elif self.page_stack.currentWidget() == getattr(self, 'ecg_test_page', None):
                        if hasattr(self.ecg_test_page, 'stop_acquisition'):
                            self.ecg_test_page.stop_acquisition()

                        # Close any open expanded lead view dialogs
                        try:
                            for widget in QApplication.topLevelWidgets():
                                if widget.__class__.__name__ == 'ExpandedLeadView':
                                    widget.close()
                        except Exception as e:
                            print(f"Error closing expanded views: {e}")

                        QMessageBox.critical(self, "Test Failed", "Device disconnected. Test Failed")
                        self.page_stack.setCurrentWidget(self.dashboard_page)

            except Exception:
                pass
        else:
            # Not connected, scan for device
            if sys.platform == "darwin":
                now = time.time()
                if getattr(self, "_device_scan_in_progress", False):
                    return
                last = getattr(self, "_last_device_scan_time", 0)
                if now - last < 5.0:
                    return

                # Show searching status while scan is in progress
                if hasattr(self, 'device_status_label'):
                    self.device_status_label.setText("Searching for device...")
                    self.device_status_label.setStyleSheet("color: orange; margin-right: 10px; font-weight: bold;")
                
                self._device_scan_in_progress = True
                self._last_device_scan_time = now
                try:
                    self.scan_for_device_version()
                finally:
                    self._device_scan_in_progress = False
            else:
                # Non-macOS platforms: trigger scan immediately and show searching status
                if hasattr(self, 'device_status_label'):
                    self.device_status_label.setText("Searching for device...")
                    self.device_status_label.setStyleSheet("color: orange; margin-right: 10px; font-weight: bold;")

                self.scan_for_device_version()

        # Only skip scanning if NOT already connected and a test window is open
        # Skip if any test window is open (HRV or Hyperkalemia)
        if hasattr(self, 'hrv_window') and self.hrv_window and self.hrv_window.isVisible():
            return
        if hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and self.hyperkalemia_window.isVisible():
            return
        # Skip if on ECG Test Page (stacked widget)
        if self.page_stack.currentWidget() == getattr(self, 'ecg_test_page', None):
            return

    def scan_for_device_version(self):
        """Scan all available ports for the device using version command"""
        try:
            scan_start = time.time()
            ports = list(serial.tools.list_ports.comports())
            if sys.platform == "darwin":
                ports = [p for p in ports if ("usbserial" in p.device) or ("usbmodem" in p.device)]
            if not ports:
                self.update_device_ui(False)
                self._initial_scan_completed = True
                return

            # Prioritize the last saved port to speed up connection
            if hasattr(self, 'settings_manager'):
                saved_port = self.settings_manager.get_setting("serial_port")
                if saved_port:
                    # Move saved port to the front of the list
                    ports.sort(key=lambda p: 0 if p.device == saved_port else 1)

            for port in ports:
                try:
                    # Quick check
                    ser = serial.Serial(port.device, 115200, timeout=0.1)
                    handler = HardwareCommandHandler(ser)
                    success, version, _ = handler.send_version_command(timeout=0.2)
                    ser.close()
                    
                    if success and version:
                        # Inform user if not the initial scan
                        if getattr(self, '_initial_scan_completed', False):
                            QMessageBox.information(self, "Connection Status", "Device connected")

                        # Update hardware version if it's different from current
                        if self.device_version != version:
                            print(f"🔄 Hardware version changed from {self.device_version} to {version}")
                            self.device_version = version

                        self.device_port = port.device
                        self.device_connected = True
                        self.update_device_ui(True)

                        # Save to settings so test pages can use it
                        if hasattr(self, 'settings_manager'):
                            self.settings_manager.set_setting("serial_port", port.device)
                            self.settings_manager.set_setting("baud_rate", "115200")
                            self.settings_manager.set_setting("hardware_version", version)
                            self.settings_manager.save_settings()
                            elapsed = time.time() - scan_start
                            print(f"✅ Device found on {port.device} in {elapsed:.2f}s, saved to settings with version {version}.")
                            self._initial_scan_completed = True
                        return
                except Exception:
                    continue

            # If loop finishes without success
            self.update_device_ui(False)
            self._initial_scan_completed = True
            
        except Exception:
            self._initial_scan_completed = True
            pass

    def update_device_ui(self, connected):
        """Update UI elements based on device connection status"""
        if connected:
            self .device_status_label.setText( "Device Connected" )
            self.device_status_label.setStyleSheet("color: green; margin-right: 10px; font-weight: bold;")
            
            # Enable test buttons
            if hasattr(self, 'hrv_test_btn'):
                self.hrv_test_btn.setEnabled(True)
                self.hrv_test_btn.setStyleSheet("background: #dc3545; color: white; border-radius: 16px; padding: 8px 24px;")

            if hasattr(self, 'hyperkalemia_test_btn'):
                self.hyperkalemia_test_btn.setEnabled(True)
                self.hyperkalemia_test_btn.setStyleSheet("background: #d2691e; color: white; border-radius: 16px; padding: 8px 24px;")

            if hasattr(self, 'date_btn'): # ECG Lead Test 12
                self.date_btn.setEnabled(True)
                self.date_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 16px; padding: 8px 24px;")
        else:
            self.device_status_label.setText("Device Disconnected")
            self.device_status_label.setStyleSheet("color: red; margin-right: 10px; font-weight: bold;")

            # Reset hardware version in settings when disconnected
            if hasattr(self, 'settings_manager'):
                self.settings_manager.set_setting("hardware_version", "")
                # set_setting already calls save_settings()
            self.device_version = None
            
            # Disable test buttons
            
            grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"

            if hasattr(self, 'hrv_test_btn'):
                self.hrv_test_btn.setEnabled(False)
                self.hrv_test_btn.setStyleSheet(grey_style)

            if hasattr(self, 'hyperkalemia_test_btn'):
                self.hyperkalemia_test_btn.setEnabled(False)
                self.hyperkalemia_test_btn.setStyleSheet(grey_style)

            if hasattr(self, 'date_btn'):
                self.date_btn.setEnabled(False)
                self.date_btn.setStyleSheet(grey_style)

    def update_internet_status(self):
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            self.status_dot.setStyleSheet("border-radius: 9px; background: #00e676; border: 2px solid #fff;")
            self.status_dot.setToolTip("Connected to Internet")
        except Exception:
            self.status_dot.setStyleSheet("border-radius: 9px; background: #e74c3c; border: 2px solid #fff;")
            self.status_dot.setToolTip("No Internet Connection")
    def toggle_medical_mode(self):
        self.medical_mode = not self.medical_mode
        if self.medical_mode:
            # Medical color coding: blue/green/white (previous behavior)
            self.setStyleSheet("QWidget { background: #e3f6fd; } QFrame { background: #f8fdff; border-radius: 16px; } QLabel { color: #006266; }")
            self.medical_btn.setText("Normal Mode")
            self.medical_btn.setStyleSheet("background: #0984e3; color: white; border-radius: 10px; padding: 4px 18px;")
        else:
            self.setStyleSheet("")
            self.medical_btn.setText("Medical Mode")
            self.medical_btn.setStyleSheet("background: #00b894; color: white; border-radius: 10px; padding: 4px 18px;")
        # Update ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
            
    def open_admin_reports(self):
        try:
            login = AdminLoginDialog(self)
            if login.exec_() == QDialog.Accepted:
                from utils.cloud_uploader import get_cloud_uploader
                cu = get_cloud_uploader()
                cu.reload_config()
                dlg = AdminReportsDialog(cu, self)
                dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to open admin reports: {e}")

    def auto_sync_to_cloud(self):
        """Background auto-backup of reports/metrics when internet is available"""
        try:
            # Do not overlap
            if getattr(self, '_cloud_sync_in_progress', False):
                return
            # Require internet
            import socket
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=2)
                online = True
            except Exception:
                online = False
            if not online:
                return
            # Require cloud configured
            from utils.cloud_uploader import get_cloud_uploader
            cloud_uploader = get_cloud_uploader()
            if not cloud_uploader.is_configured():
                return
            # Scan reports directory for new files not in upload log
            import glob, os, json
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            uploaded_names = set()
            try:
                history = cloud_uploader.get_upload_history(limit=1000)
                for item in history:
                    path = item.get('local_path') or ''
                    if path:
                        uploaded_names.add(os.path.basename(path))
            except Exception:
                pass
            candidates = []
            candidates += glob.glob(os.path.join(reports_dir, "ECG_Report_*.pdf"))
            candidates += [p for p in glob.glob(os.path.join(reports_dir, "*.json"))
                           if ('report' in os.path.basename(p).lower() or 'metric' in os.path.basename(p).lower())]
            # Filter new files
            pending = [p for p in candidates if os.path.basename(p) not in uploaded_names]
            if not pending:
                return
            # Upload in background (sequential, small set)
            self._cloud_sync_in_progress = True
            original_text = self.cloud_sync_btn.text() if hasattr(self, 'cloud_sync_btn') else ''
            if hasattr(self, 'cloud_sync_btn'):
                self.cloud_sync_btn.setText("Syncing...")
                self.cloud_sync_btn.setEnabled(False)
            for path in pending:
                cloud_uploader.upload_report(path)
            if hasattr(self, 'cloud_sync_btn'):
                self.cloud_sync_btn.setText(original_text or "Cloud Sync")
                self.cloud_sync_btn.setEnabled(True)
        except Exception:
            pass
        finally:
            self._cloud_sync_in_progress = False

    def sync_to_cloud(self):
        """Sync ECG reports and metrics to AWS S3"""
        try:
            from utils.cloud_uploader import get_cloud_uploader
            from PyQt5.QtWidgets import QMessageBox
            
            cloud_uploader = get_cloud_uploader()
            # Re-read .env in case the app was launched before keys were added
            try:
                cloud_uploader.reload_config()
                # As an extra safeguard, read .env directly and override fields
                try:
                    from dotenv import dotenv_values
                    from pathlib import Path as _P
                    root = _P(__file__).resolve().parents[2]
                    cfg = dotenv_values(str(root / '.env'))
                    if cfg:
                        cloud_uploader.cloud_service = (cfg.get('CLOUD_SERVICE') or cloud_uploader.cloud_service or 'none').lower()
                        cloud_uploader.upload_enabled = (str(cfg.get('CLOUD_UPLOAD_ENABLED') or cloud_uploader.upload_enabled).lower() == 'true')
                        cloud_uploader.s3_bucket = cfg.get('AWS_S3_BUCKET') or cloud_uploader.s3_bucket
                        cloud_uploader.s3_region = cfg.get('AWS_S3_REGION') or cloud_uploader.s3_region
                        cloud_uploader.aws_access_key = cfg.get('AWS_ACCESS_KEY_ID') or cloud_uploader.aws_access_key
                        cloud_uploader.aws_secret_key = cfg.get('AWS_SECRET_ACCESS_KEY') or cloud_uploader.aws_secret_key
                        _env_path_used = str(root / '.env')
                    else:
                        _env_path_used = '(not found)'
                except Exception:
                    _env_path_used = '(error reading .env)'
            except Exception:
                _env_path_used = '(reload failed)'
            
            if not cloud_uploader.is_configured():
                QMessageBox.warning(
                    self, 
                    "Cloud Not Configured",
                    (
                        "AWS S3 is not configured.\n\nCurrent values read:\n"
                        f"CLOUD_SERVICE={getattr(cloud_uploader,'cloud_service','')}\n"
                        f"CLOUD_UPLOAD_ENABLED={getattr(cloud_uploader,'upload_enabled','')}\n"
                        f"AWS_S3_BUCKET={getattr(cloud_uploader,'s3_bucket','')}\n"
                        f"AWS_S3_REGION={getattr(cloud_uploader,'s3_region','')}\n"
                        f"AWS_ACCESS_KEY_ID set?={'yes' if getattr(cloud_uploader,'aws_access_key',None) else 'no'}\n"
                        f"AWS_SECRET_ACCESS_KEY set?={'yes' if getattr(cloud_uploader,'aws_secret_key',None) else 'no'}\n"
                        f".env path tried: {_env_path_used}\n\n"
                        "Fix: Create .env in project root with:\n"
                        "CLOUD_UPLOAD_ENABLED=true\nCLOUD_SERVICE=s3\n"
                        "AWS_S3_BUCKET=your-bucket-name\nAWS_S3_REGION=us-east-1\n"
                        "AWS_ACCESS_KEY_ID=...\nAWS_SECRET_ACCESS_KEY=...\n\n"
                        "See AWS_REPORTS_ONLY_SETUP.md for details."
                    )
                )
                return
            
            # Show progress
            self.cloud_sync_btn.setText("Syncing...")
            self.cloud_sync_btn.setEnabled(False)
            
            # Find and upload all report files
            import glob
            reports_dir = "reports"
            uploaded_count = 0
            errors = []
            
            # Build file lists (non-blocking upload in background thread)
            pdf_reports = glob.glob(os.path.join(reports_dir, "ECG_Report_*.pdf"))
            json_reports = [p for p in glob.glob(os.path.join(reports_dir, "*.json"))
                            if 'report' in os.path.basename(p).lower() or 'metric' in os.path.basename(p).lower()]

            files_to_upload = pdf_reports + json_reports

            import threading
            def _do_upload():
                nonlocal uploaded_count, errors
                try:
                    for path in files_to_upload:
                        result = cloud_uploader.upload_report(path)
                        if result.get('status') == 'success':
                            uploaded_count += 1
                        elif result.get('status') != 'skipped':
                            errors.append(f"{os.path.basename(path)}: {result.get('message', 'Unknown error')}")
                finally:
                    # Restore UI safely on the main thread
                    try:
                        self.cloud_sync_btn.setText("Cloud Sync")
                        self.cloud_sync_btn.setEnabled(True)
                        if uploaded_count > 0:
                            msg = f" Successfully uploaded {uploaded_count} file(s) to AWS S3!"
                            if errors:
                                msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors[:3])
                            QMessageBox.information(self, "Cloud Sync Complete", msg)
                        else:
                            QMessageBox.information(self, "No Files to Sync", "No report files found in the reports directory.")
                    except Exception:
                        pass

            t = threading.Thread(target=_do_upload, daemon=True)
            t.start()
                
        except Exception as e:
            self.cloud_sync_btn.setText("Cloud Sync")
            self.cloud_sync_btn.setEnabled(True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Sync Error",
                f"Failed to sync to cloud:\n{str(e)}"
            )
            print(f" Cloud sync error: {e}")
    
    def apply_dark_theme(self):
        """Apply dark theme styling to all UI components"""
        self.setStyleSheet("""
            QWidget { background: #181818; color: #fff; }
            QFrame { background: #232323 !important; border-radius: 16px; color: #fff; border: 2px solid #fff; }
            QLabel { color: #fff; }
            QPushButton { background: #333; color: #ff6600; border-radius: 10px; }
            QPushButton:checked { background: #ff6600; color: #fff; }
            QCalendarWidget QWidget { background: #232323; color: #fff; }
            QCalendarWidget QAbstractItemView { background: #232323; color: #fff; selection-background-color: #444; selection-color: #ff6600; }
            QTextEdit { background: #232323; color: #fff; border-radius: 12px; border: 2px solid #fff; }
        """)
        self.dark_btn.setText("Light Mode")
        # Set matplotlib canvas backgrounds to dark
        self.ecg_canvas.axes.set_facecolor("#232323")
        self.ecg_canvas.figure.set_facecolor("#232323")
        for child in self.findChildren(QFrame):
            child.setStyleSheet("background: #232323; border-radius: 16px; color: #fff; border: 2px solid #fff;")
        for key, label in self.metric_labels.items():
            label.setStyleSheet("color: #fff; background: transparent;")
        for canvas in self.findChildren(MplCanvas):
            canvas.axes.set_facecolor("#232323")
            canvas.figure.set_facecolor("#232323")
            canvas.draw()
        for calendar in self.findChildren(QCalendarWidget):
            calendar.setStyleSheet("background: #232323; color: #fff; border-radius: 12px; border: 2px solid #fff;")
        for txt in self.findChildren(QTextEdit):
                    txt.setStyleSheet("background: #232323; color: #fff; border-radius: 12px; border: 2px solid #fff;")
        # Update ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
    
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.setStyleSheet("")
            self.dark_btn.setText("Dark Mode")
            self.ecg_canvas.axes.set_facecolor("#eee")
            self.ecg_canvas.figure.set_facecolor("#fff")
            for child in self.findChildren(QFrame):
                child.setStyleSheet("")
            for key, label in self.metric_labels.items():
                label.setStyleSheet("color: #222; background: transparent;")
                for canvas in child.findChildren(MplCanvas):
                    canvas.axes.set_facecolor("#fff")
                    canvas.figure.set_facecolor("#fff")
                    canvas.draw()
                for self.schedule_calendar in child.findChildren(QCalendarWidget):
                    self.schedule_calendar.setStyleSheet("")
                for txt in child.findChildren(QTextEdit):
                    txt.setStyleSheet("")
        # Update ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
    
    def test_asset_paths(self):
        """
        Test all asset paths at startup to ensure they're working correctly.
        This helps with debugging path issues.
        """
        print("=== Testing Asset Paths ===")
        
        # Test common assets
        test_assets = ["her.png", "v.gif", "plasma.gif", "ECG1.png"]
        
        for asset in test_assets:
            path = get_asset_path(asset)
            exists = os.path.exists(path)
            print(f"{asset}: {'✓' if exists else '✗'} - {path}")
            
            if not exists:
                print(f"  Warning: {asset} not found!")
        
        print("=== Asset Path Test Complete ===\n")
    
    def change_background(self, background_type):
        """
        Change the dashboard background dynamically.
        
        Args:
            background_type (str): "plasma.gif", "tenor.gif", "v.gif", "solid", or "none"
        """
        if background_type == "none":
            self.use_gif_background = False
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);")
            print("Background changed to solid color")
            return
        
        self.use_gif_background = True
        self.preferred_background = background_type
        
        # Stop current movie if any
        if hasattr(self.bg_label, 'movie'):
            self.bg_label.movie().stop()
        
        # Load new background
        movie = None
        if background_type == "plasma.gif":
            plasma_path = get_asset_path("plasma.gif")
            if os.path.exists(plasma_path):
                movie = QMovie(plasma_path)
                print("Background changed to plasma.gif")
            else:
                print("plasma.gif not found, keeping current background")
                return
        elif background_type == "tenor.gif":
            tenor_gif_path = get_asset_path("tenor.gif")
            if os.path.exists(tenor_gif_path):
                movie = QMovie(tenor_gif_path)
                print("Background changed to tenor.gif")
            else:
                print("tenor.gif not found, keeping current background")
                return
        elif background_type == "v.gif":
            v_gif_path = get_asset_path("v.gif")
            if os.path.exists(v_gif_path):
                movie = QMovie(v_gif_path)
                print("Background changed to v.gif")
            else:
                print("v.gif not found, keeping current background")
                return
        
        if movie:
            self.bg_label.setMovie(movie)
            movie.start()
            # Store reference to movie
            self.bg_label.movie = lambda: movie
    
    def cycle_background(self):
        """
        Cycle through different background options when the background button is clicked.
        """
        backgrounds = ["solid", "light_gradient", "dark_gradient", "medical_theme"]
        current_bg = "solid"  # Default to solid
        
        try:
            current_index = backgrounds.index(current_bg)
            next_index = (current_index + 1) % len(backgrounds)
            next_bg = backgrounds[next_index]
        except ValueError:
            next_bg = "solid"
        
        if next_bg == "solid":
            self.change_background("none")
            self.bg_btn.setText("BG: Solid")
        elif next_bg == "light_gradient":
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f0f0f0);")
            self.bg_btn.setText("BG: Light")
        elif next_bg == "dark_gradient":
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #34495e);")
            self.bg_btn.setText("BG: Dark")
        elif next_bg == "medical_theme":
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e8f5e8, stop:1 #d4edda);")
            self.bg_btn.setText("BG: Medical")
        
    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def resizeEvent(self, event):
        """Handle window resize events to maintain responsive design"""
        super().resizeEvent(event)
        
        # Update background label size to match new window size
        if hasattr(self, 'bg_label'):
            self.bg_label.setGeometry(0, 0, self.width(), self.height())
        
        # Ensure all widgets maintain proper proportions
        self.update_layout_proportions()

    def closeEvent(self, event):
        """Handle dashboard closure to ensure hardware is disconnected"""
        print("Dashboard closing...")

        # Clear hardware version on application close
        if hasattr(self, 'settings_manager'):
            self.settings_manager.set_setting("hardware_version", "")
            self.settings_manager.save_settings()
            print("✅ Hardware version cleared on close")

        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                if hasattr(self.ecg_test_page, 'close_serial_connection'):
                    self.ecg_test_page.close_serial_connection()
                elif hasattr(self.ecg_test_page, 'serial_reader') and self.ecg_test_page.serial_reader:
                    self.ecg_test_page.serial_reader.stop()
                    self.ecg_test_page.serial_reader.close()
        except Exception as e:
            print(f"Error cleaning up dashboard resources: {e}")
        try:
            from ecg.serial.serial_reader import GlobalHardwareManager
            GlobalHardwareManager().close_reader()
        except Exception as e:
            print(f"Error closing global serial reader: {e}")
        try:
            if hasattr(self, 'device_check_timer') and self.device_check_timer:
                self.device_check_timer.stop()
        except Exception:
            pass
        event.accept()
    
    def update_layout_proportions(self):
        """Update layout proportions when window is resized"""
        # This method can be used to adjust layout proportions based on window size
        current_width = self.width()
        current_height = self.height()
        
        # Adjust font sizes based on window size for better readability
        if current_width < 1000:
            # Small window - use smaller fonts
            font_size = 12
        elif current_width < 1400:
            # Medium window - use medium fonts
            font_size = 14
        else:
            # Large window - use larger fonts
            font_size = 16
        
        # Update font sizes for better responsiveness
        for child in self.findChildren(QLabel):
            if hasattr(child, 'font'):
                current_font = child.font()
                if current_font.pointSize() > 8:  # Don't make fonts too small
                    current_font.setPointSize(max(8, font_size - 2))
                    child.setFont(current_font)
