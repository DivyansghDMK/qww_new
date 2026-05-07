"""
utils/license_dialog.py
=======================
PyQt5 License Activation Dialog for CardioX.

Shows at application startup (before login) when no valid license is cached.
Users enter their license key, which is validated against the server.
"""

from __future__ import annotations

import platform
import sys
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy,
    QScrollArea, QVBoxLayout, QWidget,
)

try:
    from utils.license_manager import (
        check_license,
        activate_with_server,
        deactivate,
        format_key,
        get_hardware_fingerprint,
        load_stored_key,
        remember_valid_license,
        save_stored_key,
        tier_name,
        SOFTWARE_VERSION,
    )
except ImportError:
    from license_manager import (  # type: ignore
        check_license,
        activate_with_server,
        deactivate,
        format_key,
        get_hardware_fingerprint,
        load_stored_key,
        remember_valid_license,
        save_stored_key,
        tier_name,
        SOFTWARE_VERSION,
    )


# â”€â”€ Background validation worker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _ValidateWorker(QThread):
    result = pyqtSignal(dict)

    def __init__(self, license_key: str):
        super().__init__()
        self._key = license_key

    def run(self):
        try:
            fingerprint = get_hardware_fingerprint()
            # First try activation. If the key is already bound to this machine,
            # the server may still accept it; otherwise fall back to validation.
            res = activate_with_server(self._key, fingerprint)
            if res.get("success"):
                res["valid"] = True
                res["source"] = "server"
                res["message"] = res.get("message", "License activated.")
            else:
                # Already-activated or activation-rejected keys should still be
                # checked through the normal validation path.
                res = check_license(self._key)
        except Exception as e:
            res = {"valid": False, "message": str(e)}
        self.result.emit(res)


# â”€â”€ License Dialog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class LicenseDialog(QDialog):
    """
    Full-screen license activation dialog.
    Accepted  â†’ license is valid, store it, proceed to app.
    Rejected  â†’ user cancelled / quit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CardioX â€” License Activation")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(560, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._worker: Optional[_ValidateWorker] = None
        self._license_result: dict = {}

        self._build_ui()
        self._fit_to_screen()

        # Pre-fill stored key if available
        stored = load_stored_key()
        if stored:
            self._key_input.setText(format_key(stored))

    def _fit_to_screen(self):
        """Size the dialog to the active screen so it never gets clipped."""
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                self.resize(680, 520)
                return
            geom = screen.availableGeometry()
            width = min(max(int(geom.width() * 0.70), self.minimumWidth()), 900)
            height = min(max(int(geom.height() * 0.72), self.minimumHeight()), 700)
            self.resize(width, height)
            self.move(
                geom.left() + max(0, (geom.width() - width) // 2),
                geom.top() + max(0, (geom.height() - height) // 2),
            )
        except Exception:
            self.resize(680, 520)

    # â”€â”€ UI construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # â”€â”€ Header bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        header = QWidget()
        header.setFixedHeight(72)
        header.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a1a2e, stop:1 #16213e);"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(28, 0, 28, 0)

        logo_lbl = QLabel("ðŸ«€ CardioX")
        logo_lbl.setFont(QFont("Arial", 20, QFont.Bold))
        logo_lbl.setStyleSheet("color: #ff8c00; background: transparent;")

        ver_lbl = QLabel(f"v{SOFTWARE_VERSION}")
        ver_lbl.setStyleSheet("color: rgba(255,255,255,0.5); font-size:12px; background:transparent;")
        ver_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        h_layout.addWidget(logo_lbl)
        h_layout.addStretch()
        h_layout.addWidget(ver_lbl)
        root.addWidget(header)

        # â”€â”€ Main card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        card = QWidget()
        card.setStyleSheet("background: #f5f6fa;")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(40, 30, 40, 24)
        c_layout.setSpacing(16)

        # Title
        title = QLabel("Software License Activation")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #1a1a2e;")
        title.setWordWrap(True)
        c_layout.addWidget(title)

        # Subtitle
        sub = QLabel(
            "Enter the license key provided by Deckmount to activate this software.\n"
            "An internet connection is required for the initial activation."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #555; font-size: 13px;")
        c_layout.addWidget(sub)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("background: #dde; max-height: 1px; border: none;")
        c_layout.addWidget(div)

        # Key input
        key_lbl = QLabel("License Key")
        key_lbl.setStyleSheet("color: #333; font-weight: bold; font-size: 13px;")
        c_layout.addWidget(key_lbl)

        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("XXXXX-XXXXX-XXXXX-XXXXX")
        self._key_input.setMinimumHeight(48)
        self._key_input.setFont(QFont("Courier New", 15, QFont.Bold))
        self._key_input.setMaxLength(23)  # 20 chars + 3 hyphens
        self._key_input.setAlignment(Qt.AlignCenter)
        self._key_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ccd;
                border-radius: 10px;
                padding: 8px 16px;
                background: white;
                color: #1a1a2e;
                letter-spacing: 2px;
            }
            QLineEdit:focus {
                border: 2px solid #ff8c00;
                background: #fffbf5;
            }
        """)
        self._key_input.textChanged.connect(self._on_key_typed)
        self._key_input.returnPressed.connect(self._on_activate)
        c_layout.addWidget(self._key_input)

        # System info (shown to user for transparency)
        fp = get_hardware_fingerprint()
        sys_info = QLabel(
            f"Machine ID:\n{fp}\n"
            f"OS: {platform.system()} {platform.release()}  |  Host: {platform.node()}"
        )
        sys_info.setWordWrap(True)
        sys_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sys_info.setStyleSheet("color: #999; font-size: 10px; line-height: 1.2;")
        sys_info.setAlignment(Qt.AlignCenter)
        c_layout.addWidget(sys_info)
        # Status label (feedback)
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(32)
        self._status.setStyleSheet("font-size: 12px; color: #888;")
        c_layout.addWidget(self._status)

        c_layout.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._quit_btn = QPushButton("Quit")
        self._quit_btn.setFixedHeight(44)
        self._quit_btn.setStyleSheet("""
            QPushButton {
                background: #eee;
                color: #555;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 14px;
                padding: 0 24px;
            }
            QPushButton:hover { background: #e0e0e0; }
        """)
        self._quit_btn.clicked.connect(self.reject)

        self._activate_btn = QPushButton("Activate")
        self._activate_btn.setFixedHeight(44)
        self._activate_btn.setDefault(True)
        self._activate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff7a12, stop:1 #ff950f);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 0 32px;
            }
            QPushButton:hover { background: #e86f00; }
            QPushButton:disabled { background: #ffb87a; }
        """)
        self._activate_btn.clicked.connect(self._on_activate)

        btn_row.addWidget(self._quit_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._activate_btn)
        c_layout.addLayout(btn_row)

        # Help link
        help_lbl = QLabel(
            "Need a license key? Contact "
            "<a href='mailto:cardiocare@deckmount.in' style='color:#ff8c00;'>"
            "cardiocare@deckmount.in</a>"
        )
        help_lbl.setOpenExternalLinks(True)
        help_lbl.setAlignment(Qt.AlignCenter)
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        c_layout.addWidget(help_lbl)

        scroll.setWidget(card)
        root.addWidget(scroll, 1)

    # â”€â”€ Slots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_key_typed(self, text: str):
        """Auto-format key as user types: insert hyphens every 5 chars."""
        # Strip everything except base-32 chars
        clean = "".join(
            c for c in text.upper()
            if c in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        )
        # Insert hyphens
        parts = [clean[i:i+5] for i in range(0, min(len(clean), 20), 5)]
        formatted = "-".join(parts)

        # Update without triggering recursive signal
        self._key_input.blockSignals(True)
        self._key_input.setText(formatted)
        self._key_input.setCursorPosition(len(formatted))
        self._key_input.blockSignals(False)

        self._activate_btn.setEnabled(len(clean) == 20)
        self._set_status("", "")

    def _set_status(self, msg: str, color: str = "#888"):
        self._status.setText(msg)
        self._status.setStyleSheet(f"font-size: 12px; color: {color};")

    def _on_activate(self):
        key_text = self._key_input.text().strip().upper().replace(" ", "")
        clean = key_text.replace("-", "")
        if len(clean) != 20:
            self._set_status("âš  Please enter a complete 20-character license key.", "#e67e22")
            return

        self._set_status("ðŸ”„ Contacting license serverâ€¦", "#2980b9")
        self._activate_btn.setEnabled(False)
        self._quit_btn.setEnabled(False)

        self._worker = _ValidateWorker(key_text)
        self._worker.result.connect(self._on_validation_result)
        self._worker.start()

    def _on_validation_result(self, result: dict):
        self._activate_btn.setEnabled(True)
        self._quit_btn.setEnabled(True)

        if result.get("valid"):
            self._license_result = result
            tier = result.get("tier", 0)
            exp  = result.get("expires", 0)
            exp_str = "Perpetual" if exp == 0 else __import__("datetime").datetime.fromtimestamp(exp).strftime("%Y-%m-%d")

            self._set_status(
                f"âœ… Activated â€” {tier_name(tier)} License  |  Expires: {exp_str}",
                "#27ae60",
            )
            key_text = self._key_input.text().strip()
            save_stored_key(key_text)
            remember_valid_license(key_text, get_hardware_fingerprint(), result)

            # Brief pause so user sees success message, then accept
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1200, self.accept)
        else:
            msg = result.get("message", "Validation failed.")
            if result.get("revoked") or "revoked" in msg.lower():
                QMessageBox.critical(
                    self,
                    "License Revoked",
                    f"{msg}\n\nContact support to restore access.",
                )
                self.reject()
                return
            self._set_status(f"âœ— {msg}", "#e74c3c")

    def get_license_result(self) -> dict:
        return self._license_result

    def get_license_key(self) -> str:
        return self._key_input.text().strip()


# â”€â”€ Standalone test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = LicenseDialog()
    if dlg.exec_() == QDialog.Accepted:
        print("Activated:", dlg.get_license_result())
    else:
        print("Cancelled.")
    sys.exit(0)


