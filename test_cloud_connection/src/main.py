import sys
import os

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

# Load environment variables from .env file
# When running as an executable, .env is bundled in the same directory
if hasattr(sys, '_MEIPASS'):
    env_path = os.path.join(sys._MEIPASS, '.env')
    load_dotenv(env_path)
else:
    load_dotenv()

from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, 
    QMessageBox, QStackedWidget, QWidget, QInputDialog, QSizePolicy
)
from PyQt5.QtCore import Qt
from utils.crash_logger import get_crash_logger
from utils.session_recorder import SessionRecorder
from PyQt5.QtGui import QFont, QPixmap, QIntValidator

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
        title.setFont(QFont("Arial", 52, QFont.Black))
        title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff6600, stop:1 #ffb347);
            letter-spacing: 4px;
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
        tagline.setFont(QFont("Arial", 18, QFont.Bold))
        tagline.setStyleSheet("color: #ff6600; margin-bottom: 18px; margin-top: 0px; background: rgba(255,255,255,0.1);")
        tagline.setAlignment(Qt.AlignHCenter)
        main_layout.addWidget(tagline)
        # --- Glass effect container in center ---
        row = QHBoxLayout()
        row.addStretch(1)
        glass = QWidget(self)
        glass.setObjectName("Glass")
        glass.setStyleSheet("""
            QWidget#Glass {

                background: rgba(255,255,255,0.18);
                border-radius: 24px;
                border: 2px solid rgba(255,255,255,0.35);zx
            }
        """)
        glass.setMinimumSize(600, 520)
        # Create stacked widget and login/register widgets BEFORE using stacked_col
        self.stacked = QStackedWidget(glass)
        self.login_widget = self.create_login_widget()
        self.register_widget = self.create_register_widget()
        self.stacked.addWidget(self.login_widget)
        self.stacked.addWidget(self.register_widget)
        glass_layout = QHBoxLayout(glass)
        glass_layout.setContentsMargins(32, 32, 32, 32)
        # ECG image inside glass, left side (larger)
        ecg_img = QLabel()
        ecg_pix = QPixmap(resource_path('assets/v1.png'))
        if not ecg_pix.isNull():
            ecg_img.setPixmap(ecg_pix.scaled(400, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            ecg_img.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            ecg_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            ecg_img.setStyleSheet("margin: 0px 32px 0px 0px; border-radius: 24px; background: transparent;")
        # Wrap image in a layout to center vertically
        img_col = QVBoxLayout()
        img_col.addStretch(1)
        img_col.addWidget(ecg_img, alignment=Qt.AlignHCenter)
        img_col.addStretch(1)
        glass_layout.addLayout(img_col, 2)
        # Login/Register stacked widget (vertical)
        stacked_col = QVBoxLayout()
        stacked_col.addStretch(1)
        stacked_col.addWidget(self.stacked, 2)
        # Add sign up/login prompt below
        signup_row = QHBoxLayout()
        signup_row.addStretch(1)
        signup_lbl = QLabel("Don't have an account?")
        signup_lbl.setStyleSheet("color: #fff; font-size: 15px;")
        signup_btn = QPushButton("Sign up")
        signup_btn.setStyleSheet("color: #ff6600; background: transparent; border: none; font-size: 15px; font-weight: bold; text-decoration: underline;")
        signup_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        signup_row.addWidget(signup_lbl)
        signup_row.addWidget(signup_btn)
        signup_row.addStretch(1)
        stacked_col.addSpacing(10)
        stacked_col.addLayout(signup_row)
        # Add login prompt to register widget
        login_row = QHBoxLayout()
        
        login_row.addStretch(1)
        login_lbl = QLabel("Already have an account?")
        login_lbl.setStyleSheet("color: #fff; font-size: 15px;")
        login_btn = QPushButton("Login")
        login_btn.setStyleSheet("color: #ff6600; background: transparent; border: none; font-size: 15px; font-weight: bold; text-decoration: underline;")
        login_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        login_row.addWidget(login_lbl)
        login_row.addWidget(login_btn)
        login_row.addStretch(1)
        # Insert login_row at the bottom of the register widget
        self.register_widget.layout().addSpacing(10)
        self.register_widget.layout().addLayout(login_row)
        stacked_col.addStretch(1)
        glass_layout.addLayout(stacked_col, 3)
        glass_layout.setSpacing(0)
        row.addWidget(glass, 1)
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
        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Full Name")

        password_row = QHBoxLayout()
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        password_row.addWidget(self.login_password)
        
        # Add eye toggle button
        self.login_eye_btn = QPushButton("👁")
        self.login_eye_btn.setFixedSize(36, 36)
        self.login_eye_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 8px; font-size: 16px;")
        self.login_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.login_password, self.login_eye_btn))
        password_row.addWidget(self.login_eye_btn)

        login_btn = QPushButton("Login")
        login_btn.setObjectName("LoginBtn")
        login_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold;")
        login_btn.clicked.connect(self.handle_login)
        phone_btn = QPushButton("Login with Phone Number")
        phone_btn.setObjectName("SignUpBtn")
        phone_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold;")
        phone_btn.clicked.connect(self.handle_phone_login)

        for w in [self.login_email, self.login_password, login_btn, phone_btn]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.login_email.setStyleSheet("border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; color: #222;")
        self.login_password.setStyleSheet("border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; color: #222;")

         # Add Enter key functionality to both fields
        self.login_email.returnPressed.connect(self.handle_login)
        self.login_password.returnPressed.connect(self.handle_login)
        
        layout.addWidget(self.login_email)
        layout.addLayout(password_row)
        layout.addWidget(login_btn)
        layout.addWidget(phone_btn)
        # Add nav links under phone_btn
        nav_row = QHBoxLayout()
        # Navigation modules - using simple placeholder classes
        # (Original nav modules were moved to clutter directory)
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
        for text, NavClass in nav_links:
            nav_btn = QPushButton(text)
            nav_btn.setStyleSheet("color: #ff6600; background: transparent; border: none; font-size: 15px; font-weight: bold; text-decoration: underline;")
            page = NavClass()
            self.nav_stack.addWidget(page)
            self.nav_pages[text] = page
            if text == "Pricing":
                # Pricing dialog - using simple fallback
                def show_pricing_dialog():
                    QMessageBox.information(self, "Pricing", "Pricing information not available.")
                nav_btn.clicked.connect(lambda checked, p=self: show_pricing_dialog())
            else:
                nav_btn.clicked.connect(lambda checked, t=text: show_nav_page(t))
            nav_row.addWidget(nav_btn)
        layout.addLayout(nav_row)
        layout.addWidget(self.nav_stack)
        self.nav_stack.setVisible(False)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def create_register_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
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
        
        # Apply dashboard color coding
        for w in [self.reg_serial, self.reg_name, self.reg_age, self.reg_gender, self.reg_address, self.reg_phone, self.reg_password, self.reg_confirm]:
            w.setStyleSheet("border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; color: #222;")
        
        register_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold;")
        register_btn.setMinimumHeight(36)
        
        # Create password field with eye toggle
        password_row = QHBoxLayout()
        password_row.addWidget(self.reg_password)
        self.password_eye_btn = QPushButton("👁")
        self.password_eye_btn.setFixedSize(36, 36)
        self.password_eye_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 8px; font-size: 16px;")
        self.password_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.reg_password, self.password_eye_btn))
        password_row.addWidget(self.password_eye_btn)
        
        # Create confirm password field with eye toggle
        confirm_row = QHBoxLayout()
        confirm_row.addWidget(self.reg_confirm)
        self.confirm_eye_btn = QPushButton("👁")
        self.confirm_eye_btn.setFixedSize(36, 36)
        self.confirm_eye_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 8px; font-size: 16px;")
        self.confirm_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.reg_confirm, self.confirm_eye_btn))
        confirm_row.addWidget(self.confirm_eye_btn)
        
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
        widget.setLayout(layout)
        return widget

    def handle_login(self):
        identifier = self.login_email.text()  # Can be full name, username, or phone
        password_or_serial = self.login_password.text()
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

    def handle_phone_login(self):
        # Create a custom input dialog so we can enforce numeric-only input
        dlg = QInputDialog(self)
        dlg.setWindowTitle("Login with Phone Number")
        dlg.setLabelText("Enter your phone number:")
        dlg.setInputMode(QInputDialog.TextInput)

        # Apply an integer validator to restrict input to digits only
        line_edit = dlg.findChild(QLineEdit)
        if line_edit is not None:
            # Limit to 10-digit phone numbers:
            # QIntValidator uses 32-bit ints, so max must be <= 2147483647.
            # We also cap length to 10 characters to enforce 10 digits.
            line_edit.setValidator(QIntValidator(0, 2147483647, self))
            line_edit.setMaxLength(10)

        if dlg.exec_() == QDialog.Accepted:
            phone = dlg.textValue().strip()
        
            # Extra safety: ensure only digits are accepted and length is <= 10
            if not phone.isdigit():
                QMessageBox.warning(self, "Invalid Input", "Please enter digits only for the phone number.")
                return
            if len(phone) > 10:
                QMessageBox.warning(self, "Invalid Phone Number", "Phone number must be at most 10 digits.")
                return

            # Check if this is a new phone number (not in users)
            users = load_users()
            is_new_user = True
            user_record = None
            
            # Check if phone number exists in users
            for username, record in users.items():
                if str(record.get('phone', '')) == str(phone):
                    is_new_user = False
                    user_record = record
                    self.username = username
                    break
            
            # If new user, create a record with signup date
            if is_new_user:
                from datetime import datetime
                user_record = {
                    'phone': phone,
                    'contact': phone,
                    'signup_date': datetime.now().strftime("%Y-%m-%d")
                }
                # Save new user to users.json
                users[phone] = user_record
                save_users(users)
                self.username = phone
                QMessageBox.information(self, "Phone Login", f"New user registered with phone: {phone}")
            else:
                # Existing user - check if signup_date exists, if not add it
                if 'signup_date' not in user_record or not user_record.get('signup_date'):
                    from datetime import datetime
                    user_record['signup_date'] = datetime.now().strftime("%Y-%m-%d")
                    users[self.username] = user_record
                    save_users(users)
                QMessageBox.information(self, "Phone Login", f"Logged in with phone: {phone}")
            
            self.result = True
            self.user_details = user_record
            self.accept()

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
            

        except Exception as e:
            print(f" Error uploading user signup: {e}")
        
        QMessageBox.information(self, "Success", "Registration successful! You can now sign in.")
        self.stacked.setCurrentIndex(0)
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("🔒")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("👁")

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
        app.setApplicationVersion("1.3")

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
    main()