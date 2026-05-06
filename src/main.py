import sys
import os
import shutil

# ── BUG-05 FIX: Force software OpenGL rendering ──────────────────────────────
# MUST be set BEFORE any Qt/PyQtGraph import.
# This fixes blank waves on laptops with Intel HD, AMD integrated, or no GPU.
os.environ['QT_OPENGL'] = 'software'
os.environ['PYOPENGL_PLATFORM'] = 'win32'
os.environ['QT_SCALE_FACTOR'] = '1'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
# ─────────────────────────────────────────────────────────────────────────────

import json
from dotenv import load_dotenv

def _prepare_runtime_workspace() -> str:
    """
    Ensure a writable runtime directory for packaged installs.
    This avoids permission issues on systems where app is installed in Program Files.
    """
    use_runtime = bool(getattr(sys, "frozen", False)) or (
        str(os.getenv("ECG_FORCE_RUNTIME_DIR", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )
    if not use_runtime:
        return os.getcwd()

    base_dir = os.getenv("ECG_RUNTIME_DIR", "").strip()
    if not base_dir:
        local_appdata = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
        base_dir = os.path.join(local_appdata, "Deckmount", "ECGMonitor")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Ensure required runtime folders exist.
    for rel in ("reports", "logs", "offline_queue", "temp", "src"):
        os.makedirs(os.path.join(base_dir, rel), exist_ok=True)

    # Seed essential config files from bundle/app folder if missing in runtime dir.
    source_roots = []
    if getattr(sys, "frozen", False):
        source_roots.append(os.path.dirname(sys.executable))
        if hasattr(sys, "_MEIPASS"):
            source_roots.append(sys._MEIPASS)
    else:
        source_roots.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    seed_files = [
        ".env",
        "customer_channels.json",
        "users.json",
        "ecg_settings.json",
        "last_conclusions.json",
        os.path.join("src", "users.json"),
        os.path.join("src", "ecg_settings.json"),
    ]
    for rel in seed_files:
        dst = os.path.join(base_dir, rel)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for root in source_roots:
            src = os.path.join(root, rel)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
                break

    os.environ["ECG_RUNTIME_DIR"] = base_dir
    os.chdir(base_dir)
    return base_dir


_RUNTIME_DIR = _prepare_runtime_workspace()

# Load environment variables from .env file.
# Priority: runtime dir (.env) -> executable dir -> _MEIPASS
runtime_env = os.path.join(os.getcwd(), ".env")
if os.path.exists(runtime_env):
    load_dotenv(runtime_env, override=False)
else:
    load_dotenv(override=False)
if getattr(sys, "frozen", False):
    app_env = os.path.join(os.path.dirname(sys.executable), ".env")
    if os.path.exists(app_env):
        load_dotenv(app_env, override=False)
if hasattr(sys, '_MEIPASS'):
    meipass_env = os.path.join(sys._MEIPASS, '.env')
    if os.path.exists(meipass_env):
        load_dotenv(meipass_env, override=False)

from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, 
    QMessageBox, QStackedWidget, QWidget, QInputDialog, QSizePolicy, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
from utils.crash_logger import get_crash_logger
from utils.session_recorder import SessionRecorder
from PyQt5.QtGui import QFont, QPixmap, QIntValidator
from utils.ecg_auth_api import get_ecg_auth_api
from utils.offline_queue import get_offline_queue

try:
    from version import APP_VERSION, UPDATE_CHANNEL, GITHUB_REPOSITORY
except Exception:
    APP_VERSION = "0.0.0"
    UPDATE_CHANNEL = "stable"
    GITHUB_REPOSITORY = ""

# Import core modules  
try:
    from core.logging_config import get_logger, log_function_call
    from core.exceptions import ECGError, ECGConfigError
    from config.settings import get_config, resource_path
    from core.constants import SUCCESS_MESSAGES, ERROR_MESSAGES
    logger_available = True
except ImportError as e:
    print(f" Core modules not available: {e}")
    print(" Using fallback logging")
    logger_available = False
    
    # Fallback logging
    class FallbackLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}") #msg is messagin g for the self
    
    def log_function_call(func):
        return func
    
    def get_config():
        return type('Config', (), {'get': lambda x, y=None: y})()
    
    def resource_path(relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
    
    SUCCESS_MESSAGES = {"modules_loaded": " Core modules imported successfully"}
    ERROR_MESSAGES = {"import_error": " Core module import error: {}"}

# Initialize logger
if logger_available:
    logger = get_logger("MainApp")
else:
    logger = FallbackLogger()

# Import application modules with proper error handling
def get_auth_modules():
    try:
        from auth.sign_in import SignIn
        from auth.sign_out import SignOut
        return SignIn, SignOut
    except ImportError as e:
        logger.error(ERROR_MESSAGES["import_error"].format(e))
        logger.error("💡 Make sure you're running from the src directory")
        logger.error("💡 Try: cd src && python main.py")
        sys.exit(1)

def get_dashboard_module():
    try:
        from dashboard.dashboard import Dashboard
        return Dashboard
    except ImportError as e:
        logger.error(ERROR_MESSAGES["import_error"].format(e))
        return None

# Import ECG modules with fallback
def get_ecg_modules():
    try:
        from ecg.pan_tompkins import pan_tompkins
        logger.info(SUCCESS_MESSAGES["ecg_modules_loaded"])
        return pan_tompkins
    except ImportError as e:
        if "ecg_import_warning" in ERROR_MESSAGES:
            logger.warning(ERROR_MESSAGES["ecg_import_warning"].format(e))
        else:
            logger.warning(f"ECG module import warning: {e}")
        logger.warning("💡 ECG analysis features may be limited")
        # Create a dummy function to prevent errors
        def pan_tompkins(ecg, fs=500):
            return []
        return pan_tompkins

# Get configuration
config = get_config()
USER_DATA_FILE = resource_path("users.json")


@log_function_call
def load_users():
    """Load user data from file with error handling"""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r") as f:
                users = json.load(f)
                logger.info(f"Loaded {len(users)} users from {USER_DATA_FILE}")
                return users
        else:
            logger.info(f"User file {USER_DATA_FILE} not found, creating empty user database")
            return {}
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading users: {e}")
        logger.error("Creating empty user database")
        return {}


@log_function_call
def save_users(users):
    """Save user data to file with error handling"""
    try:
        with open(USER_DATA_FILE, "w") as f:
            json.dump(users, f, indent=2)
        logger.info(f"Saved {len(users)} users to {USER_DATA_FILE}")
    except IOError as e:
        logger.error(f"Error saving users: {e}")
        raise ECGError(f"Failed to save user data: {e}")


# Login/Register Dialog
class LoginRegisterDialog(QDialog):
    def __init__(self):
        super().__init__()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size for usability
        
        # Set window properties for better responsiveness
        self.setWindowTitle("CardioX by Deckmount - Sign In / Sign Up")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Initialize sign-in logic
        SignIn, _ = get_auth_modules()
        self.sign_in_logic = SignIn()

        # Resize according to current screen size (~90% of available geometry)
        try:
            screen_geom = QApplication.primaryScreen().availableGeometry()
            target_w = max(int(screen_geom.width() * 0.9), self.minimumWidth())
            target_h = max(int(screen_geom.height() * 0.9), self.minimumHeight())
            self.resize(target_w, target_h)
        except Exception:
            pass
        
        try:
            self.setWindowState(Qt.WindowMaximized)
        except Exception:
            pass
        
        self.init_ui()
        self.result = False
        self.username = None
        self.user_details = {}

    def init_ui(self):
        # Set up GIF background
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        self.bg_label.lower()
        
        # Try multiple possible paths for the v.gif file
        possible_gif_paths = [
            resource_path('assets/v.gif'),
            resource_path('../assets/v.gif'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'v.gif'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'v.gif')
        ]
        
        gif_path = None
        for path in possible_gif_paths:
            if os.path.exists(path):
                gif_path = path
                print(f" Found v.gif at: {gif_path}")
                break
        
        if gif_path and os.path.exists(gif_path):
            try:
                from PyQt5.QtGui import QMovie
                movie = QMovie(gif_path)
                if movie.isValid():
                    self.bg_label.setMovie(movie)
                    movie.start()
                    print(" v.gif background started successfully")
                else:
                    print(" Invalid GIF file")
                    # Set fallback background
                    self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:1 #16213e);")
            except Exception as e:
                print(f" Error loading v.gif: {e}")
                # Set fallback background
                self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:1 #16213e);")
        else:
            print(" v.gif not found in any expected location")
            print(f"Tried paths: {possible_gif_paths}")
            # Set fallback background
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:1 #16213e);")
        
        self.bg_label.setScaledContents(True)
        # --- Title and tagline above glass ---
        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1)
        # Title (outside glass) - logo style
        title = QLabel("CardioX by Deckmount")
        title.setFont(QFont("Arial", 42, QFont.Black))
        title.setStyleSheet("""
            color: #ffb347;
            letter-spacing: 2px;
            margin-bottom: 0px;
            padding-top: 0px;
            padding-bottom: 0px;
            font-weight: 900;
            border-radius: 18px;
        """)
        title.setAlignment(Qt.AlignHCenter)
        main_layout.addWidget(title)
        # Tagline (outside glass)
        tagline = QLabel("Built to Detect. Designed to Last.")
        tagline.setFont(QFont("Arial", 16, QFont.Bold))
        tagline.setStyleSheet("color: #ff7a12; margin-bottom: 20px; margin-top: 2px; background: transparent;")
        tagline.setAlignment(Qt.AlignHCenter)
        main_layout.addWidget(tagline)
        # --- Glass effect container in center ---
        row = QHBoxLayout()
        row.addStretch(1)
        glass = QWidget(self)
        glass.setObjectName("Glass")
        glass.setStyleSheet("""
            QWidget#Glass {
                background: rgba(255,255,255,0.14);
                border-radius: 30px;
                border: 1px solid rgba(255,255,255,0.26);
            }
        """)
        glass.setMinimumSize(560, 500)
        # Create stacked widget and login/register widgets BEFORE using stacked_col
        self.stacked = QStackedWidget(glass)
        self.login_widget = self.create_login_widget()
        self.register_widget = self.create_register_widget()
        self.stacked.addWidget(self.login_widget)
        self.stacked.addWidget(self.register_widget)
        glass_layout = QHBoxLayout(glass)
        glass_layout.setContentsMargins(28, 28, 28, 24)
        glass_layout.setSpacing(12)
        # Login/Register stacked widget only, centered like the reference
        stacked_col = QVBoxLayout()
        stacked_col.setSpacing(14)
        stacked_col.addWidget(self.stacked, 1)
        # Add sign up/login prompt below
        signup_row = QHBoxLayout()
        signup_row.addStretch(1)
        signup_lbl = QLabel("Don't have an account?")
        signup_lbl.setStyleSheet("color: rgba(255,255,255,0.82); font-size: 14px;")
        signup_btn = QPushButton("Sign up")
        signup_btn.setStyleSheet("color: #ff8d2b; background: transparent; border: none; font-size: 14px; font-weight: bold; text-decoration: underline;")
        signup_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        signup_row.addWidget(signup_lbl)
        signup_row.addWidget(signup_btn)
        signup_row.addStretch(1)
        stacked_col.addLayout(signup_row)
        glass_layout.addLayout(stacked_col, 1)
        row.addWidget(glass, 0, Qt.AlignHCenter)
        row.addStretch(1)
        main_layout.addLayout(row)
        main_layout.addStretch(1)   
        self.setLayout(main_layout)
        # Make glass and all widgets expand responsively
        glass.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Resize background with window
        self.resizeEvent = self._resize_bg
        
        # Ensure background is always visible
        self.ensure_background_visible()


    def _resize_bg(self, event):
        """Handle window resize to maintain background coverage"""
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        # Ensure the background stays behind all other widgets
        self.bg_label.lower()
        event.accept()
    
    def ensure_background_visible(self):
        """Ensure the background is always visible and properly positioned"""
        try:
            # Make sure the background label is at the bottom of the widget stack
            self.bg_label.lower()
            # Ensure it covers the entire window
            self.bg_label.setGeometry(0, 0, self.width(), self.height())
            # Make sure it's visible
            self.bg_label.setVisible(True)
            logger.info(" Background visibility ensured")
        except Exception as e:
            logger.warning(f"Background visibility issue: {e}")

    def create_login_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        input_style = """
            QLineEdit {
                border: 1px solid rgba(255, 156, 64, 0.85);
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 15px;
                background: rgba(255,255,255,0.92);
                color: #1f1f1f;
                selection-background-color: #ff8a1f;
            }
            QLineEdit:focus {
                border: 2px solid #ff8a1f;
                background: rgba(255,255,255,0.98);
            }
        """
        otp_input_style = """
            QLineEdit {
                border: 1px solid rgba(75, 190, 134, 0.78);
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 15px;
                background: rgba(255,255,255,0.92);
                color: #1f1f1f;
            }
            QLineEdit:focus {
                border: 2px solid #2fa66f;
                background: rgba(255,255,255,0.98);
            }
        """
        primary_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
                color: white;
                border-radius: 14px;
                padding: 12px 0;
                font-size: 16px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff8a26, stop:1 #ffab31); }
            QPushButton:pressed { background: #e96a00; }
        """
        secondary_button_style = """
            QPushButton {
                background: rgba(58,58,58,0.62);
                color: #ffbe63;
                border: 1px solid rgba(255, 179, 71, 0.42);
                border-radius: 14px;
                padding: 11px 14px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background: rgba(88,88,88,0.78); }
            QPushButton:pressed { background: rgba(108,108,108,0.92); }
        """

        section_title = QLabel("Sign in to continue")
        section_title.setStyleSheet("color: white; font-size: 30px; font-weight: bold;")

        section_subtitle = QLabel("Use your account password or request an OTP on your phone.")
        section_subtitle.setWordWrap(True)
        section_subtitle.setStyleSheet("color: rgba(255,255,255,0.78); font-size: 13px;")

        password_header = QLabel("ACCOUNT LOGIN")
        password_header.setStyleSheet("color: #ffb347; font-size: 12px; font-weight: bold; letter-spacing: 1px;")

        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Full Name or Phone Number")
        self.login_email.setMinimumHeight(44)

        password_row = QHBoxLayout()
        password_row.setSpacing(10)
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setMinimumHeight(44)
        password_row.addWidget(self.login_password)

        self.login_eye_btn = QPushButton("View")
        self.login_eye_btn.setFixedSize(72, 46)
        self.login_eye_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            font-size: 13px;
            font-weight: bold;
            border: none;
        """)
        self.login_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.login_password, self.login_eye_btn))
        password_row.addWidget(self.login_eye_btn)

        login_btn = QPushButton("Login")
        login_btn.setObjectName("LoginBtn")
        login_btn.setMinimumHeight(46)
        login_btn.setStyleSheet(primary_button_style)
        login_btn.clicked.connect(self.handle_login)

        phone_btn = QPushButton("Send OTP")
        phone_btn.setObjectName("SignUpBtn")
        phone_btn.setMinimumHeight(44)
        phone_btn.setMinimumWidth(170)
        phone_btn.setStyleSheet(secondary_button_style)
        phone_btn.clicked.connect(self.handle_phone_login)
        self.phone_btn = phone_btn

        self.login_phone = QLineEdit()
        self.login_phone.setPlaceholderText("Phone number (10 digits)")
        self.login_phone.setMinimumHeight(44)
        self.login_phone.setMaxLength(10)
        self.login_phone.setValidator(QIntValidator(0, 2147483647, self))

        phone_row = QHBoxLayout()
        phone_row.setSpacing(10)
        phone_row.addWidget(self.login_phone, 3)
        phone_row.addWidget(phone_btn, 0)

        self.login_otp = QLineEdit()
        self.login_otp.setPlaceholderText("Enter 4-digit OTP")
        self.login_otp.setMaxLength(4)
        self.login_otp.setMinimumHeight(44)
        self.login_otp.setValidator(QIntValidator(0, 9999, self))

        self.verify_otp_btn = QPushButton("Verify OTP")
        self.verify_otp_btn.setMinimumHeight(44)
        self.verify_otp_btn.setMinimumWidth(132)
        self.verify_otp_btn.setEnabled(False)
        self.verify_otp_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f704f, stop:1 #2fa66f);
                color: white;
                border-radius: 14px;
                padding: 11px 14px;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background: #2a9b67; }
            QPushButton:pressed { background: #1f7b52; }
            QPushButton:disabled {
                background: rgba(35, 139, 92, 0.30);
                color: rgba(255,255,255,0.65);
            }
        """)
        self.verify_otp_btn.clicked.connect(self.verify_phone_otp)

        self._otp_cooldown_seconds = 60
        self._otp_lockout_seconds = 300
        self._otp_resend_available_at = 0.0
        self._otp_lockout_until = 0.0
        self._otp_failed_attempts = 0
        self._otp_timer = QTimer(self)
        self._otp_timer.timeout.connect(self._refresh_otp_controls)

        otp_row = QHBoxLayout()
        otp_row.setSpacing(10)
        otp_row.addWidget(self.login_otp, 3)
        otp_row.addWidget(self.verify_otp_btn, 1)

        phone_card = QWidget()
        phone_card.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(255,255,255,0.10),
                stop:1 rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 18px;
        """)
        phone_card_layout = QVBoxLayout(phone_card)
        phone_card_layout.setContentsMargins(14, 12, 14, 12)
        phone_card_layout.setSpacing(10)
        phone_card_title = QLabel("Phone Login")
        phone_card_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.12);
        """)
        phone_card_desc = QLabel("Enter your mobile number, request an OTP, then verify it to sign in.")
        phone_card_desc.setWordWrap(True)
        phone_card_desc.setStyleSheet("color: rgba(255,255,255,0.74); font-size: 12px;")
        phone_card_layout.addWidget(phone_card_title)
        phone_card_layout.addWidget(phone_card_desc)
        phone_card_layout.addLayout(phone_row)
        phone_card_layout.addLayout(otp_row)

        for w in [self.login_email, self.login_password, self.login_phone, self.login_otp, login_btn, phone_btn, self.verify_otp_btn]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.login_email.setStyleSheet(input_style)
        self.login_password.setStyleSheet(input_style)
        self.login_phone.setStyleSheet(input_style)
        self.login_otp.setStyleSheet(otp_input_style)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background: rgba(255,255,255,0.10); max-height: 1px; min-height: 1px; border: none;")

        self.login_email.returnPressed.connect(self.handle_login)
        self.login_password.returnPressed.connect(self.handle_login)
        self.login_phone.returnPressed.connect(self.handle_phone_login)
        self.login_otp.returnPressed.connect(self.verify_phone_otp)
        self.login_otp.textChanged.connect(self._update_verify_otp_button)

        layout.addWidget(section_title)
        layout.addWidget(section_subtitle)
        layout.addWidget(password_header)
        layout.addWidget(self.login_email)
        layout.addLayout(password_row)
        layout.addWidget(login_btn)
        layout.addSpacing(4)
        layout.addWidget(divider)
        layout.addSpacing(4)
        layout.addWidget(phone_card)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(10)

        class NavHome(QWidget):
            def __init__(self): super().__init__(); self.setWindowTitle("Home")
        class NavAbout(QWidget):
            def __init__(self): super().__init__(); self.setWindowTitle("About")
        class NavBlog(QWidget):
            def __init__(self): super().__init__(); self.setWindowTitle("Blog")
        class NavPricing(QWidget):
            def __init__(self): super().__init__(); self.setWindowTitle("Pricing")

        nav_links = [
            ("Home", NavHome),
            ("About us", NavAbout),
            ("Blog", NavBlog),
            ("Pricing", NavPricing)
        ]
        self.nav_stack = QStackedWidget()
        self.nav_pages = {}

        def show_nav_page(page_name):
            self.nav_stack.setCurrentWidget(self.nav_pages[page_name])
            self.nav_stack.setVisible(True)

        nav_row.addStretch(1)
        for text, NavClass in nav_links:
            nav_btn = QPushButton(text)
            nav_btn.setStyleSheet("""
                color: #ff9a3b;
                background: transparent;
                border: none;
                font-size: 14px;
                font-weight: bold;
                padding: 4px 8px;
            """)
            page = NavClass()
            self.nav_stack.addWidget(page)
            self.nav_pages[text] = page
            if text == "Pricing":
                def show_pricing_dialog():
                    QMessageBox.information(self, "Pricing", "Pricing information not available.")
                nav_btn.clicked.connect(lambda checked, p=self: show_pricing_dialog())
            else:
                nav_btn.clicked.connect(lambda checked, t=text: show_nav_page(t))
            nav_row.addWidget(nav_btn)
        nav_row.addStretch(1)

        layout.addLayout(nav_row)
        layout.addWidget(self.nav_stack)
        self.nav_stack.setVisible(False)
        self._refresh_otp_controls()
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def create_register_widget(self):
        # Scroll area prevents layout compression on smaller screens, which was cropping the org buttons.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        register_input_style = """
            QLineEdit {
                border: 1px solid rgba(255, 156, 64, 0.82);
                border-radius: 14px;
                padding: 11px 14px;
                font-size: 15px;
                background: rgba(255,255,255,0.92);
                color: #1f1f1f;
            }
            QLineEdit:focus {
                border: 2px solid #ff8a1f;
                background: rgba(255,255,255,0.98);
            }
        """
        self.reg_serial = QLineEdit()
        self.reg_serial.setPlaceholderText("Machine Serial ID")
        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText("Full Name")
        self.reg_age = QLineEdit()
        self.reg_age.setPlaceholderText("Age")
        self.reg_gender = QLineEdit()
        self.reg_gender.setPlaceholderText("Gender")
        self.reg_address = QLineEdit()
        self.reg_address.setPlaceholderText("Address")
        self.reg_phone = QLineEdit()
        self.reg_phone.setPlaceholderText("Phone Number")
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Password")
        self.reg_password.setEchoMode(QLineEdit.Password)
        
        self.reg_confirm = QLineEdit()
        self.reg_confirm.setPlaceholderText("Confirm Password")
        self.reg_confirm.setEchoMode(QLineEdit.Password)
        
        register_btn = QPushButton("Sign Up")
        register_btn.setObjectName("SignUpBtn")
        register_btn.clicked.connect(self.handle_register)
        
        for w in [self.reg_serial, self.reg_name, self.reg_age, self.reg_gender, self.reg_address, self.reg_phone, self.reg_password, self.reg_confirm]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            w.setMinimumHeight(44)
        
        for w in [self.reg_serial, self.reg_name, self.reg_age, self.reg_gender, self.reg_address, self.reg_phone, self.reg_password, self.reg_confirm]:
            w.setStyleSheet(register_input_style)
        
        register_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            padding: 11px 0;
            font-size: 16px;
            font-weight: bold;
            border: none;
        """)
        register_btn.setMinimumHeight(46)
        
        # Create password field with eye toggle
        password_row = QHBoxLayout()
        password_row.setSpacing(10)
        password_row.addWidget(self.reg_password)
        self.password_eye_btn = QPushButton("View")
        self.password_eye_btn.setFixedSize(72, 46)
        self.password_eye_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            font-size: 13px;
            font-weight: bold;
            border: none;
        """)
        self.password_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.reg_password, self.password_eye_btn))
        password_row.addWidget(self.password_eye_btn)
        
        # Create confirm password field with eye toggle
        confirm_row = QHBoxLayout()
        confirm_row.setSpacing(10)
        confirm_row.addWidget(self.reg_confirm)
        self.confirm_eye_btn = QPushButton("View")
        self.confirm_eye_btn.setFixedSize(72, 46)
        self.confirm_eye_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            font-size: 13px;
            font-weight: bold;
            border: none;
        """)
        self.confirm_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.reg_confirm, self.confirm_eye_btn))
        confirm_row.addWidget(self.confirm_eye_btn)
        
        # Organization buttons (imported from organization module)
        import importlib
        organization_module = importlib.import_module('organization')
        create_organization_buttons_layout = getattr(organization_module, 'create_organization_buttons_layout')
        self.org_buttons_layout, self.new_org_handler, self.existing_org_handler = create_organization_buttons_layout(self)
        
        layout.addLayout(self.org_buttons_layout)
        layout.addWidget(self.reg_serial)
        layout.addWidget(self.reg_name)
        layout.addWidget(self.reg_age)
        layout.addWidget(self.reg_gender)
        layout.addWidget(self.reg_address)
        layout.addWidget(self.reg_phone)
        layout.addLayout(password_row)
        layout.addLayout(confirm_row)
        layout.addWidget(register_btn)
        layout.addStretch(1)

        # Login prompt inside the register page (so it scrolls with the form content).
        login_row = QHBoxLayout()
        login_row.addStretch(1)
        login_lbl = QLabel("Already have an account?")
        login_lbl.setStyleSheet("color: rgba(255,255,255,0.82); font-size: 14px;")
        login_btn = QPushButton("Login")
        login_btn.setStyleSheet(
            "color: #ff8d2b; background: transparent; border: none; font-size: 14px; "
            "font-weight: bold; text-decoration: underline; padding: 2px 6px;"
        )
        login_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        login_row.addWidget(login_lbl)
        login_row.addWidget(login_btn)
        login_row.addStretch(1)
        layout.addSpacing(12)
        layout.addLayout(login_row)

        scroll.setWidget(widget)
        return scroll

    def handle_login(self):
        identifier = self.login_email.text()  # Can be full name, username, or phone
        password_or_serial = self.login_password.text()
        # Users can be created while the app is running (e.g., by Doctor/HCP head flows).
        # Refresh from disk before validating so new accounts can log in immediately.
        try:
            self.sign_in_logic.users = self.sign_in_logic.load_users()
        except Exception:
            pass
        # BUG-31 FIX: Admin credentials loaded from environment variable, not hardcoded
        try:
            admin_user = os.environ.get('CARDIOX_ADMIN_USER', 'admin')
            admin_pass = os.environ.get('CARDIOX_ADMIN_PASS', '')  # empty = disabled unless set in .env
            if admin_pass and identifier.strip().lower() == admin_user and password_or_serial == admin_pass:
                self.result = True
                self.username = 'admin'
                self.user_details = {'is_admin': True}
                self.accept()
                return
        except Exception:
            pass
        if self.sign_in_logic.sign_in_user_allow_serial(identifier, password_or_serial):
            # Get the actual user record for details
            found = self.sign_in_logic._find_user_record(identifier)
            if found:
                username, record = found
                self.result = True
                self.username = username
                self.user_details = record  # Store full user details
                self.accept()
            else:
                self.result = True
                self.username = identifier
                self.user_details = {}
                self.accept()
        else:
            QMessageBox.warning(self, "Error", "Invalid credentials. Please check your full name and password.")

    def _upsert_phone_login_user(self, phone: str, token: str):
        from datetime import datetime

        # Use the same user store as normal password login (auth/sign_in.py),
        # otherwise OTP-created passwords can be saved to a different users.json
        # and then fail validation at next login.
        try:
            users = self.sign_in_logic.load_users()
        except Exception:
            users = load_users()
        user_key = phone
        user_record = None
        source_key = None

        for username, record in users.items():
            if str(record.get('phone', '')).strip() == phone:
                user_key = phone
                source_key = username
                user_record = dict(record)
                break

        if not isinstance(user_record, dict):
            user_record = {}

        if not user_record.get('signup_date'):
            user_record['signup_date'] = datetime.now().strftime("%Y-%m-%d")

        user_record['phone'] = phone
        user_record['contact'] = phone
        user_record['username'] = phone
        user_record['login_username'] = phone
        user_record['login_identifier'] = phone
        user_record['canonical_username'] = phone
        user_record['master_phone'] = phone
        user_record['auth_provider'] = 'ecg_otp_backend'
        user_record['jwt_token'] = token
        user_record['last_phone_login_at'] = datetime.now().isoformat()

        users[user_key] = user_record
        if source_key and source_key != user_key and source_key in users:
            try:
                del users[source_key]
            except Exception:
                pass
        try:
            self.sign_in_logic.users = users
            self.sign_in_logic.save_users()
        except Exception:
            save_users(users)
        return user_key, user_record

    def _save_phone_user_password(self, username: str, password: str):
        from datetime import datetime
        canonical_username = self._get_inline_phone_number() or username

        try:
            users = self.sign_in_logic.load_users()
        except Exception:
            users = load_users()
        record = users.get(canonical_username, users.get(username, {}))
        if not isinstance(record, dict):
            record = {}
        record['username'] = canonical_username
        record['login_username'] = canonical_username
        record['login_identifier'] = canonical_username
        record['canonical_username'] = canonical_username
        record['master_phone'] = canonical_username
        if username and username != canonical_username and username in users:
            try:
                del users[username]
            except Exception:
                pass
        record['password'] = password
        record['password_set_via'] = 'phone_otp'
        record['password_set_at'] = datetime.now().isoformat()
        users[canonical_username] = record
        try:
            self.sign_in_logic.users = users
            self.sign_in_logic.save_users()
        except Exception:
            save_users(users)
        return record

    def _prompt_phone_password_setup(self, phone: str) -> str:
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Password")
        dialog.setModal(True)
        dialog.setMinimumWidth(420)
        dialog.setStyleSheet("""
            QDialog { background: #141a2c; border-radius: 16px; }
            QLabel { color: white; font-size: 13px; }
            QLineEdit {
                border: 1px solid rgba(255, 156, 64, 0.82);
                border-radius: 12px;
                padding: 10px 12px;
                font-size: 14px;
                background: rgba(255,255,255,0.94);
                color: #1f1f1f;
            }
            QPushButton {
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel(f"Phone {phone} verified successfully.")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        desc = QLabel("Create a password once so next time you can log in directly with your phone number and password.")
        desc.setWordWrap(True)

        password_input = QLineEdit()
        password_input.setPlaceholderText("Create Password")
        password_input.setEchoMode(QLineEdit.Password)

        confirm_input = QLineEdit()
        confirm_input.setPlaceholderText("Confirm Password")
        confirm_input.setEchoMode(QLineEdit.Password)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        skip_btn = QPushButton("Skip")
        skip_btn.setStyleSheet("background: rgba(255,255,255,0.12); color: #ffd2a3;")
        save_btn = QPushButton("Save Password")
        save_btn.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f); color: white;")

        btn_row.addWidget(skip_btn)
        btn_row.addWidget(save_btn)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(password_input)
        layout.addWidget(confirm_input)
        layout.addLayout(btn_row)

        result = {"password": ""}

        def _skip():
            dialog.reject()

        def _save():
            password = password_input.text().strip()
            confirm = confirm_input.text().strip()
            if len(password) < 4:
                QMessageBox.warning(dialog, "Password Required", "Password must be at least 4 characters long.")
                return
            if password != confirm:
                QMessageBox.warning(dialog, "Password Mismatch", "Password and confirm password must match.")
                return
            result["password"] = password
            dialog.accept()

        skip_btn.clicked.connect(_skip)
        save_btn.clicked.connect(_save)
        password_input.returnPressed.connect(_save)
        confirm_input.returnPressed.connect(_save)

        dialog.exec_()
        return result["password"]

    def _ensure_phone_user_password(self, username: str, user_record: dict, phone: str):
        if not isinstance(user_record, dict):
            user_record = {}
        if str(user_record.get('password', '')).strip():
            return user_record

        reply = QMessageBox.question(
            self,
            "Create Password",
            "Do you want to create a password for this phone login?\n\nYou can use it next time with phone number + password.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return user_record

        new_password = self._prompt_phone_password_setup(phone)
        if not new_password:
            return user_record

        updated_record = self._save_phone_user_password(username, new_password)
        QMessageBox.information(
            self,
            "Password Saved",
            "Password created successfully. Next time you can sign in using your phone number and password.",
        )
        return updated_record

    def _get_inline_phone_number(self) -> str:
        auth_api = get_ecg_auth_api()
        raw_phone = self.login_phone.text().strip() if hasattr(self, "login_phone") else ""
        return auth_api.normalize_phone(raw_phone)

    def _update_verify_otp_button(self):
        otp = self.login_otp.text().strip() if hasattr(self, "login_otp") else ""
        if hasattr(self, "verify_otp_btn"):
            locked = self._is_otp_locked()
            self.verify_otp_btn.setEnabled(len(otp) == 4 and otp.isdigit() and not locked)

    def _is_otp_locked(self) -> bool:
        import time
        now = time.time()
        return now < getattr(self, "_otp_lockout_until", 0.0)

    def _otp_resend_cooldown_remaining(self) -> int:
        import time
        remaining = int(getattr(self, "_otp_resend_available_at", 0.0) - time.time())
        return max(0, remaining)

    def _refresh_otp_controls(self):
        if not hasattr(self, "phone_btn") or not hasattr(self, "verify_otp_btn"):
            return

        resend_remaining = self._otp_resend_cooldown_remaining()
        locked = self._is_otp_locked()

        if locked:
            import time
            remaining = int(getattr(self, "_otp_lockout_until", 0.0) - time.time())
            remaining = max(0, remaining)
            self.phone_btn.setEnabled(False)
            self.phone_btn.setText(f"Wait {remaining}s")
            self.verify_otp_btn.setEnabled(False)
            self.verify_otp_btn.setText(f"Verify OTP ({remaining}s)")
            if remaining == 0:
                self._otp_lockout_until = 0.0
                self._otp_failed_attempts = 0
                self.phone_btn.setEnabled(True)
                self.phone_btn.setText("Send OTP")
                self.verify_otp_btn.setText("Verify OTP")
                self._update_verify_otp_button()
            return

        if resend_remaining > 0:
            self.phone_btn.setEnabled(False)
            self.phone_btn.setText(f"Wait {resend_remaining}s")
        else:
            self.phone_btn.setEnabled(True)
            self.phone_btn.setText("Send OTP")
            self.verify_otp_btn.setText("Verify OTP")
            if not self._otp_timer.isActive():
                self._otp_timer.stop()

        self._update_verify_otp_button()

    def handle_phone_login(self):
        # Check internet connection first
        try:
            if not get_offline_queue().is_online():
                QMessageBox.warning(self, "No Internet Connection", "An active internet connection is required to send and verify OTP. Please check your network and try again.")
                return
        except Exception as e:
            logger.warning(f"Failed to check connectivity: {e}")

        auth_api = get_ecg_auth_api()
        normalized_phone = self._get_inline_phone_number()
        if len(normalized_phone) != 10:
            QMessageBox.warning(self, "Invalid Phone Number", "Phone number must be exactly 10 digits.")
            return

        if self._is_otp_locked():
            QMessageBox.warning(self, "OTP Locked", "Too many failed OTP attempts. Please wait before trying again.")
            self._refresh_otp_controls()
            return

        resend_remaining = self._otp_resend_cooldown_remaining()
        if resend_remaining > 0:
            QMessageBox.information(self, "Please Wait", f"You can request another OTP in {resend_remaining} seconds.")
            self._refresh_otp_controls()
            return

        try:
            auth_api.send_otp(normalized_phone)
            QMessageBox.information(self, "OTP Sent", f"OTP sent successfully to +91 {normalized_phone}.")
            import time
            self._otp_resend_available_at = time.time() + getattr(self, "_otp_cooldown_seconds", 60)
            self._otp_timer.start(1000)
            self._refresh_otp_controls()
        except Exception as e:
            logger.error(f"OTP send failed for {normalized_phone}: {e}")
            QMessageBox.warning(self, "OTP Failed", f"Could not send OTP: {e}")
            return

    def verify_phone_otp(self):
        # Check internet connection first
        try:
            if not get_offline_queue().is_online():
                QMessageBox.warning(self, "No Internet Connection", "An active internet connection is required to verify OTP. Please check your network and try again.")
                return
        except Exception as e:
            logger.warning(f"Failed to check connectivity: {e}")
            
        normalized_phone = self._get_inline_phone_number()
        if len(normalized_phone) != 10:
            QMessageBox.warning(self, "Invalid Phone Number", "Phone number must be exactly 10 digits.")
            return

        otp = self.login_otp.text().strip() if hasattr(self, "login_otp") else ""
        if len(otp) != 4 or not otp.isdigit():
            QMessageBox.warning(self, "OTP Required", "OTP must be exactly 4 digits.")
            return

        if self._is_otp_locked():
            QMessageBox.warning(self, "OTP Locked", "Too many failed OTP attempts. Please wait before trying again.")
            self._refresh_otp_controls()
            return

        auth_api = get_ecg_auth_api()
        try:
            verify_result = auth_api.verify_otp(normalized_phone, otp)
            token = verify_result.get('token', '')
            if not token:
                raise ValueError("JWT token missing from verify OTP response.")

            try:
                from utils.backend_api import get_backend_api
                get_backend_api().set_token(token)
            except Exception as token_error:
                logger.warning(f"Could not propagate JWT token to backend API helper: {token_error}")

            username, user_record = self._upsert_phone_login_user(normalized_phone, token)
            user_record = self._ensure_phone_user_password(username, user_record, normalized_phone)
            self._otp_failed_attempts = 0
            self._otp_lockout_until = 0.0
            self._otp_resend_available_at = 0.0
            if hasattr(self, "_otp_timer"):
                self._otp_timer.stop()
            self._refresh_otp_controls()
            self.result = True
            self.username = username
            self.user_details = user_record
            QMessageBox.information(self, "Phone Login", f"OTP verified for {normalized_phone}.")
            self.accept()
        except Exception as e:
            logger.error(f"OTP verification failed for {normalized_phone}: {e}")
            error_text = str(e).lower()
            if "otp" in error_text or "invalid" in error_text or "incorrect" in error_text:
                self._otp_failed_attempts = getattr(self, "_otp_failed_attempts", 0) + 1
                if self._otp_failed_attempts >= 3:
                    import time
                    self._otp_lockout_until = time.time() + getattr(self, "_otp_lockout_seconds", 300)
                    self._otp_failed_attempts = 0
                    self._otp_timer.start(1000)
                    self._refresh_otp_controls()
                QMessageBox.warning(
                    self,
                    "Incorrect OTP",
                    "Incorrect OTP. Please enter the 4-digit OTP again.\n"
                    "After 3 failed attempts, OTP verification will pause for a cooling period.",
                )
            else:
                QMessageBox.warning(self, "Verification Failed", f"Could not verify OTP: {e}")

    def handle_register(self):
        serial_id = self.reg_serial.text()
        name = self.reg_name.text()
        age = self.reg_age.text()
        gender = self.reg_gender.text()
        address = self.reg_address.text()
        phone = self.reg_phone.text().strip()
        password = self.reg_password.text()
        confirm = self.reg_confirm.text()
        if not all([serial_id, name, age, gender, address, phone, password, confirm]):
            QMessageBox.warning(self, "Error", "All fields are required, including Machine Serial ID.")
            return
        # Enforce numeric phone number with length up to 10 digits
        if not phone.isdigit() or len(phone) > 10:
            QMessageBox.warning(self, "Error", "Phone number must be numbers only and at most 10 digits.")
            return
        if password != confirm:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        # Use phone as username for registration, enforce uniqueness on serial/fullname/phone
        ok, msg = self.sign_in_logic.register_user_with_details(
            username=phone,
            password=password,
            full_name=name,
            phone=phone,
            serial_id=serial_id,
            email="",
            extra={"age": age, "gender": gender, "address": address}
        )
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        
        # Upload user signup details to cloud with all patient information
        try:
            from utils.cloud_uploader import get_cloud_uploader
            from datetime import datetime
            
            uploader = get_cloud_uploader()
            user_data = {
                'username': phone,
                'full_name': name,
                'age': age,
                'gender': gender,
                'phone': phone,
                'address': address,
                'serial_number': serial_id,
                'serial_id': serial_id,  # Include both for compatibility
                'machine_serial_id': serial_id,  # Include machine serial ID
                'registered_at': datetime.now().isoformat()
            }
            upload_result = uploader.upload_user_signup(user_data)
            print(f" Signup upload status: {upload_result.get('status', 'unknown')}")

        except Exception as e:
            print(f" Error uploading user signup: {e}")
        
        QMessageBox.information(self, "Success", "Registration successful! You can now sign in.")
        self.stacked.setCurrentIndex(0)
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("Hide")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("View")

    def _show_nav_window(self, NavClass, text):
        nav_win = NavClass()
        nav_win.setWindowTitle(text)
        nav_win.setMinimumSize(400, 300)
        nav_win.show()
        if not hasattr(self, '_nav_windows'):
            self._nav_windows = []
        self._nav_windows.append(nav_win)


@log_function_call
def main():
    """Main application entry point with proper error handling"""
    try:
        # Initialize crash logger first
        crash_logger = get_crash_logger()
        crash_logger.log_info("Application starting", "APP_START")
        
        logger.info("Starting ECG Monitor Application")

        # =========================================================
        # START BACKGROUND UPLOAD SERVICE (GLOBAL)
        # =========================================================
        # Start this immediately so uploads happen even at login screen
        # and regardless of which user logs in.
        try:
            from utils.auto_sync_service import start_auto_sync
            # Start auto-sync service (runs every 15s)
            # This will:
            # 1. Scan for new/modified reports
            # 2. Initialize CloudUploader
            # 3. Initialize OfflineQueue (which handles connectivity changes)
            start_auto_sync(interval_seconds=15)
            logger.info("✅ Global background upload service started")
            
            # Also force an immediate check of the offline queue
            try:
                from utils.offline_queue import get_offline_queue
                offline_queue = get_offline_queue()
                if offline_queue:
                    stats = offline_queue.get_stats()
                    if stats.get('pending_count', 0) > 0:
                        logger.info(f"Found {stats.get('pending_count')} pending uploads - starting sync")
            except Exception as e:
                logger.warning(f"Could not check offline queue: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start background services: {e}")

        app = QApplication(sys.argv)
        app.setApplicationName("ECG Monitor")
        app.setApplicationVersion(APP_VERSION)

        try:
            from utils.update_manager import check_and_install_update, report_update_completion

            if check_and_install_update(parent=None, quiet=True):
                logger.info(
                    f"Update launched for channel={UPDATE_CHANNEL}, repo={GITHUB_REPOSITORY or 'unset'}"
                )
                return

            report_update_completion(APP_VERSION, async_mode=True)
        except Exception as e:
            logger.warning(f"Update check failed: {e}")

        # ── Pre-warm heavy imports in background ──────────────────────
        # matplotlib, scipy, pyqtgraph take 2-5s on first import
        # Start loading now so by the time user types password → cached
        def _prewarm():
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot
                import scipy.signal
                import scipy.ndimage
                import pyqtgraph
            except Exception:
                pass
        import threading
        threading.Thread(target=_prewarm, daemon=True, name="Prewarm").start()
        # ──────────────────────────────────────────────────────────────

        # ── License Gate ─────────────────────────────────────────────────────
        # Validate license BEFORE showing login. On success the result is cached
        # locally (HMAC-protected) so subsequent starts are instant / offline.
        try:
            from utils.license_manager import check_license, load_stored_key
            from utils.license_dialog import LicenseDialog

            _stored_key = load_stored_key()
            _license_ok = False

            if _stored_key:
                _res = check_license(_stored_key)
                _license_ok = bool(_res.get("valid"))
                if _license_ok:
                    logger.info(
                        f"[License] Valid — tier={_res.get('tier',0)}, "
                        f"source={_res.get('source','?')}"
                    )

            if not _license_ok:
                _dlg = LicenseDialog()
                if _dlg.exec_() != QDialog.Accepted:
                    logger.info("[License] Activation cancelled — exiting.")
                    sys.exit(0)
                logger.info(f"[License] Activated — {_dlg.get_license_result()}")

        except Exception as _lic_err:
            # If license system fails to import (e.g. first install), log and continue.
            logger.warning(f"[License] Check skipped due to error: {_lic_err}")
        # ─────────────────────────────────────────────────────────────────────

        # Initialize login dialog
        login = LoginRegisterDialog()

        # Main application loop
        while True:
            try:
                if login.exec_() == QDialog.Accepted and login.result:
                    logger.info(f"User {login.username} logged in successfully")
                    # Attach machine serial ID to crash logger for email subject/body   tagging
                    try:
                        users = load_users()
                        record = None
                        if isinstance(users, dict) and login.username in users:
                            record = users.get(login.username)
                        else:
                            # Fallback: search by phone/contact stored under 'phone'    
                            for uname, rec in (users or {}).items():
                                try:
                                    if str(rec.get('phone', '')) == str(login.username):
                                        record = rec
                                        break
                                except Exception:
                                    continue
                        serial_id = ''
                        if isinstance(record, dict):
                            serial_id = str(record.get('serial_id', ''))
                            
                        if serial_id:
                            crash_logger.set_machine_serial_id(serial_id)
                            os.environ['MACHINE_SERIAL_ID'] = serial_id
                            logger.info(f"Machine serial ID set for crash reporting: {serial_id}")
                    except Exception as e:
                        logger.warning(f"Could not set machine serial ID for crash reporting: {e}")
                    
                    # If admin, open Admin Reports UI instead of dashboard
                    if isinstance(login.user_details, dict) and login.user_details.get('is_admin'):
                        try:
                            from utils.cloud_uploader import get_cloud_uploader
                            from dashboard.admin_reports import AdminReportsDialog
                            cu = get_cloud_uploader()
                            cu.reload_config()
                            dlg = AdminReportsDialog(cu)
                            dlg.exec_()
                        except Exception as e:
                            QMessageBox.critical(None, "Admin", f"Failed to open admin reports: {e}")
                        # After admin dialog closes, show login again
                        login = LoginRegisterDialog()
                        continue
                    # ── Show splash while Dashboard imports + constructs ──────
                    # On first run / slow disk, matplotlib+scipy imports take 2-5s
                    # Without splash: window appears frozen → user thinks crash
                    try:
                        from PyQt5.QtWidgets import QSplashScreen
                        from PyQt5.QtGui import QPixmap, QColor
                        from PyQt5.QtCore import Qt
                        _splash_pix = QPixmap(420, 180)
                        _splash_pix.fill(QColor("#1a1a2e"))
                        _splash = QSplashScreen(_splash_pix,
                                                Qt.WindowStaysOnTopHint)
                        _splash.showMessage(
                            "  Loading ECG Monitor…  Please wait",
                            Qt.AlignCenter | Qt.AlignBottom,
                            QColor("#ff6600"))
                        _splash.show()
                        app.processEvents()
                    except Exception:
                        _splash = None

                    # Create and show dashboard with user details
                    Dashboard = get_dashboard_module()
                    if Dashboard is None:
                        QMessageBox.critical(None, "Error", "Failed to load Dashboard module. Please check logs.")
                        break
                    dashboard = Dashboard(username=login.username, role=None, user_details=login.user_details)
                    # Attach a session recorder for this user
                    try:
                        user_record = None
                        users = load_users()
                        if isinstance(users, dict) and login.username in users:
                            user_record = users.get(login.username)
                        else:
                            for uname, rec in (users or {}).items():
                                try:
                                    if str(rec.get('phone', '')) == str(login.username):
                                        user_record = rec
                                        break
                                except Exception:
                                    continue
                        dashboard._session_recorder = SessionRecorder(username=login.username, user_record=user_record or {})
                    except Exception as e:
                        logger.warning(f"Session recorder init failed: {e}")
                    # Close splash and show dashboard
                    if _splash is not None:
                        try:
                            _splash.finish(dashboard)
                        except Exception:
                            pass

                    dashboard.show()

                    # Run application
                    app.exec_()
                    
                    if getattr(dashboard, "closed_by_sign_out", False):
                        logger.info(f"User {login.username} logged out")
                        # After dashboard closes via sign out, show login again
                        login = LoginRegisterDialog()
                    else:
                        logger.info("Application closed by user from dashboard")
                        break
                else:
                    logger.info("Application closed by user")
                    break
                    
            except Exception as e:
                logger.error(f"Error in main application loop: {e}")
                QMessageBox.critical(None, "Application Error", 
                                    f"An error occurred: {e}\nThe application will continue.")
                # Continue with new login dialog
                login = LoginRegisterDialog()
                
    except Exception as e:
        logger.critical(f"Fatal error in main application: {e}")
        crash_logger.log_crash(f"Fatal application error: {str(e)}", e, "MAIN_APPLICATION")
        QMessageBox.critical(None, "Fatal Error", 
                           f"A fatal error occurred: {e}\nThe application will exit.")
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
