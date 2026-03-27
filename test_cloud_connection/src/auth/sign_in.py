import os
import json
from typing import Dict, Any, Optional, Tuple
from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QStackedWidget, QWidget, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


def get_asset_path(asset_name):
    """
    Get the absolute path to an asset file in a portable way.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), "assets"),
        os.path.join(script_dir, "assets"),
        os.path.join(os.path.dirname(script_dir), "assets"),
        os.path.join(script_dir, "..", "assets"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.join(path, asset_name)
    
    # Fallback
    return os.path.join(script_dir, "..", "assets", asset_name)


USER_DATA_FILE = os.path.join(os.path.dirname(__file__), '../../users.json')


class SignIn:
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = self.load_users()

    def _migrate_legacy_format(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        # Legacy format: {username: password}
        if isinstance(raw, dict):
            sample_values = list(raw.values())
            if len(sample_values) == 0:
                return {}
            if isinstance(sample_values[0], str):
                return {u: {"password": p} for u, p in raw.items()}
            # Already in new structured format
            return raw
        # Unknown/invalid -> start fresh
        return {}

    def load_users(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(USER_DATA_FILE):
            try:
                with open(USER_DATA_FILE, "r") as f:
                    data = json.load(f)
                return self._migrate_legacy_format(data)
            except Exception:
                return {}
        return {}

    def save_users(self) -> None:
        with open(USER_DATA_FILE, "w") as f:
            json.dump(self.users, f, indent=2)

    def _find_user_record(self, identifier: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Identifier may be username (dict key), phone, or full_name
        if identifier in self.users:
            return identifier, self.users[identifier]
        ident_norm = str(identifier).strip().lower()
        for uname, record in self.users.items():
            phone = str(record.get("phone", ""))
            fullname = str(record.get("full_name", ""))
            if ident_norm == str(phone).strip().lower():
                return uname, record
            if fullname and ident_norm == fullname.strip().lower():
                return uname, record
            if str(record.get("full_name", "")) == str(identifier):
                return uname, record
        return None

    def sign_in_user(self, username: str, password: str) -> bool:
        # Backward-compatible: validate using password or fallback serial
        return self.validate_credentials(username, password)

    def sign_in_user_allow_serial(self, identifier: str, secret: str) -> bool:
        # Try normal login first
        found = self._find_user_record(identifier)
        if not found:
            # Check if both fields are serial IDs (hidden recovery method)
            for uname, record in self.users.items():
                serial_id = str(record.get("serial_id", ""))
                if serial_id and str(identifier) == serial_id and str(secret) == serial_id:
                    return True
            return False
        _, record = found
        stored_password = str(record.get("password", ""))
        serial_id = str(record.get("serial_id", ""))
        # Accept password, or serial ID in password field, or serial in both fields
        return str(secret) == stored_password or (serial_id and str(secret) == serial_id)

    def validate_credentials(self, username: str, password: str) -> bool:
        found = self._find_user_record(username)
        if not found:
            return False
        _, record = found
        stored_password = str(record.get("password", "")) if isinstance(record, dict) else str(record)
        if str(password) == stored_password:
            return True
        # Also allow serial as password (forgot-password convenience)
        serial_id = str(record.get("serial_id", "")) if isinstance(record, dict) else ""
        return bool(serial_id) and str(password) == serial_id

    def _is_unique(self, key: str, value: str) -> bool:
        if not value:
            return True
        for uname, rec in self.users.items():
            if str(rec.get(key, "")) == str(value):
                return False
        return True

    def register_user_with_details(
        self,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        phone: Optional[str] = None,
        serial_id: Optional[str] = None,
        email: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        # Username uniqueness
        if username in self.users:
            return False, "Username already exists."
        # Enforce uniqueness: machine serial ID, full name, phone number
        if serial_id and not self._is_unique("serial_id", serial_id):
            return False, "Machine Serial ID already registered."
        if full_name and not self._is_unique("full_name", full_name):
            return False, "Full Name already registered."
        if phone and not self._is_unique("phone", phone):
            return False, "Phone number already registered."
        from datetime import datetime
        record: Dict[str, Any] = {
            "password": password,
            "full_name": full_name or "",
            "phone": phone or "",
            "serial_id": serial_id or "",
            "email": email or "",
            "signup_date": datetime.now().strftime("%Y-%m-%d"),  # Store signup date for new users
        }
        if isinstance(extra, dict):
            # Only include simple JSON-serializable values
            for k, v in extra.items():
                if k not in record:
                    record[k] = v
        self.users[username] = record
        self.save_users()
        
        # ========================================
        # AUTOMATIC CLOUD UPLOAD - USER SIGNUP
        # ========================================
        # Upload user signup data to cloud automatically
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.cloud_uploader import get_cloud_uploader
            from datetime import datetime
            
            cloud_uploader = get_cloud_uploader()
            if cloud_uploader.is_configured():
                # Prepare user data for cloud upload
                user_data = {
                    "username": username,
                    "full_name": full_name or "",
                    "phone": phone or "",
                    "serial_id": serial_id or "",
                    "email": email or "",
                    "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Add extra fields if present
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if k not in user_data:
                            user_data[k] = str(v) if v is not None else ""
                
                # Upload to cloud
                result = cloud_uploader.upload_user_signup(user_data)
                if result.get('status') == 'success':
                    print(f" User signup automatically uploaded to cloud")
                else:
                    print(f"  Cloud upload failed: {result.get('message', 'Unknown error')}")
            else:
                print("  Cloud not configured - user saved locally only")
        except Exception as e:
            print(f"  Cloud upload error: {e}")
            # Don't fail registration if cloud upload fails
        
        return True, "Registration successful."

    def register_user(self, username: str, password: str) -> bool:
        # Maintain legacy API, without extra details
        ok, _ = self.register_user_with_details(username=username, password=password)
        return ok


class LoginRegisterDialog(QDialog):
    def __init__(self):
        super().__init__()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(600, 400)  # Minimum size for usability
        
        self.setWindowTitle("ECG Monitor - Sign In / Sign Up")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setStyleSheet("""
            QDialog { background: #fff; border-radius: 18px; }
            QLabel { font-size: 15px; color: #222; }
            QLineEdit { border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; }
            QPushButton { background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background: #ff8800; }
        """)
        self.sign_in_logic = SignIn()
        self.init_ui()
        self.result = False
        self.username = None

    def init_ui(self):
        from PyQt5.QtWidgets import QSizePolicy
        from PyQt5.QtGui import QMovie, QPixmap
        
        main_layout = QHBoxLayout()
        
        # Left: Logo/Image with plasma effect background
        logo_widget = QWidget()
        logo_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        logo_layout = QVBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)
        
        # Clean background instead of plasma effect
        bg_label = QLabel()
        bg_label.setFixedSize(260, 260)
        bg_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        bg_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 130px;
        """)
        
        # ECG image on top of clean background
        logo_label = QLabel()
        logo_path = get_asset_path('ECG1.png')
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path).scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setStyleSheet("background: transparent;")
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            logo_label.setText("<b>PulseMonitor</b>")
            logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Stack background and ECG image
        bg_container = QVBoxLayout()
        bg_container.setAlignment(Qt.AlignCenter)
        bg_container.addWidget(bg_label, alignment=Qt.AlignCenter)
        bg_container.addWidget(logo_label, alignment=Qt.AlignCenter)
        
        logo_layout.addStretch(1)
        logo_layout.addLayout(bg_container)
        logo_layout.addStretch(1)
        logo_widget.setLayout(logo_layout)
        
        # Right: Form
        form_widget = QWidget()
        form_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)
        self.stacked = QStackedWidget(self)
        self.stacked.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.login_widget = self.create_login_widget()
        self.register_widget = self.create_register_widget()
        self.stacked.addWidget(self.login_widget)
        self.stacked.addWidget(self.register_widget)
        
        btn_layout = QHBoxLayout()
        self.login_tab = QPushButton("Sign In")
        self.signup_tab = QPushButton("Sign Up")
        
        # Make buttons responsive
        for btn in [self.login_tab, self.signup_tab]:
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setMinimumWidth(100)
        
        self.login_tab.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.signup_tab.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        btn_layout.addWidget(self.login_tab)
        btn_layout.addWidget(self.signup_tab)
        
        title = QLabel("<span style='font-family:cursive;font-size:32px;color:#222;'>PulseMonitor</span>")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form_layout.addWidget(title)
        form_layout.addLayout(btn_layout)
        form_layout.addWidget(self.stacked)
        form_widget.setLayout(form_layout)
        
        # Add to main layout (image left, form right) with responsive proportions
        main_layout.addWidget(logo_widget, 2)  # Logo takes 2 parts
        main_layout.addWidget(form_widget, 3)  # Form takes 3 parts (more space)
        self.setLayout(main_layout)

    def create_login_widget(self):
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("Username")
        self.login_username.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                padding: 12px 15px;
                font-size: 16px;
                color: white;
                font-weight: bold;
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.7);
            }
            QLineEdit:focus {
                border: 2px solid rgba(255, 255, 255, 0.6);
                background: rgba(255, 255, 255, 0.25);
            }
        """)
        
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setStyleSheet(self.login_username.styleSheet())
        
        login_btn = QPushButton("Sign In")
        login_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.25);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.4);
                border-radius: 12px;
                padding: 12px 0;
                font-size: 18px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.6);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.45);
            }
        """)
        login_btn.clicked.connect(self.handle_login)
        
        layout.addWidget(self.login_username)
        layout.addWidget(self.login_password)
        layout.addWidget(login_btn)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def create_register_widget(self):
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Create all input fields with glass morphism style
        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText("Username (will be your phone)")
        self.reg_username.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_serial = QLineEdit()
        self.reg_serial.setPlaceholderText("Machine Serial ID")
        self.reg_serial.setStyleSheet(self.login_username.styleSheet())

        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Password")
        self.reg_password.setEchoMode(QLineEdit.Password)
        self.reg_password.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_confirm = QLineEdit()
        self.reg_confirm.setPlaceholderText("Confirm Password")
        self.reg_confirm.setEchoMode(QLineEdit.Password)
        self.reg_confirm.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_fullname = QLineEdit()
        self.reg_fullname.setPlaceholderText("Full Name")
        self.reg_fullname.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_age = QLineEdit()
        self.reg_age.setPlaceholderText("Age")
        self.reg_age.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_gender = QLineEdit()
        self.reg_gender.setPlaceholderText("Gender")
        self.reg_gender.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_contact = QLineEdit()
        self.reg_contact.setPlaceholderText("Contact Number")
        self.reg_contact.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("Email Address")
        self.reg_email.setStyleSheet(self.login_username.styleSheet())
        
        register_btn = QPushButton("Sign Up")
        register_btn.setStyleSheet(self.login_btn.styleSheet())
        register_btn.clicked.connect(self.handle_register)
        
        # Set size policy for all widgets
        for w in [self.reg_username, self.reg_serial, self.reg_password, self.reg_confirm, self.reg_fullname, 
                  self.reg_age, self.reg_gender, self.reg_contact, self.reg_email, register_btn]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Add widgets to layout
        layout.addWidget(self.reg_username)
        layout.addWidget(self.reg_serial)
        layout.addWidget(self.reg_password)
        layout.addWidget(self.reg_confirm)
        layout.addWidget(self.reg_fullname)
        layout.addWidget(self.reg_age)
        layout.addWidget(self.reg_gender)
        layout.addWidget(self.reg_contact)
        layout.addWidget(self.reg_email)
        layout.addWidget(register_btn)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def handle_login(self):
        username = self.login_username.text()
        password_or_serial = self.login_password.text()
        if self.sign_in_logic.sign_in_user_allow_serial(username, password_or_serial):
            self.result = True
            self.username = username
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Invalid username or password/serial.")

    def handle_register(self):
        username = self.reg_username.text()
        password = self.reg_password.text()
        confirm = self.reg_confirm.text()
        serial_id = self.reg_serial.text()
        fullname = self.reg_fullname.text()
        age = self.reg_age.text()
        gender = self.reg_gender.text()
        contact = self.reg_contact.text()
        email = self.reg_email.text()
        if not username or not password or not serial_id:
            QMessageBox.warning(self, "Error", "Username, password and machine serial ID are required.")
            return
        if password != confirm:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        if not fullname or not age or not gender or not contact or not email:
            QMessageBox.warning(self, "Error", "All details are required.")
            return
        ok, msg = self.sign_in_logic.register_user_with_details(
            username=username,
            password=password,
            full_name=fullname,
            phone=contact,
            serial_id=serial_id,
            email=email,
            extra={"age": age, "gender": gender}
        )
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        QMessageBox.information(self, "Success", "Registration successful! You can now sign in.")
        self.stacked.setCurrentIndex(0)