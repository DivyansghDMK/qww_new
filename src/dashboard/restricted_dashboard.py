from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import Qt, QTimer

from dashboard.dashboard import Dashboard


class RestrictedDashboard(Dashboard):
    """Top-level view-only dashboard for users created by Doctor/HCP heads."""

    def __init__(
        self,
        username=None,
        role=None,
        user_details=None,
        return_on_sign_out=None,
        root_login_dialog=None,
        parent=None,
    ):
        self.return_on_sign_out = return_on_sign_out
        self.root_login_dialog = root_login_dialog
        self._parent_flow_restored = False
        super().__init__(username=username, role=role, user_details=user_details, parent=parent)
        self._apply_view_only_mode()

    def _apply_view_only_mode(self):
        """Restrict created users to report/history viewing only."""
        grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"
        self.device_connected = False
        self.device_port = None

        if hasattr(self, 'device_status_label'):
            self.device_status_label.setText("View Only Access")
            self.device_status_label.setStyleSheet("color: #8c8c8c; margin-right: 10px; font-weight: bold;")

        for button_name in ['hrv_test_btn', 'hyperkalemia_test_btn', 'date_btn', 'holter_btn']:
            button = getattr(self, button_name, None)
            if button is not None:
                button.setEnabled(False)
                button.setStyleSheet(grey_style)

    def handle_sign_out(self):
        self._restore_parent_flow()
        super().handle_sign_out()

    def closeEvent(self, event):
        self._restore_parent_flow()
        super().closeEvent(event)

    def _restore_parent_flow(self):
        if self._parent_flow_restored:
            return
        self._parent_flow_restored = True

        root_login_dialog = getattr(self, 'root_login_dialog', None)
        return_window = getattr(self, 'return_on_sign_out', None)

        if root_login_dialog is not None:
            try:
                root_login_dialog.setEnabled(True)
            except Exception:
                pass

        if return_window is not None:
            try:
                # Make restore robust after hide/show on Windows: re-apply geometry + maximized state
                # on the next event-loop tick (Qt can drop maximized geometry during show()).
                return_window.show()
                return_window.setWindowState(return_window.windowState() & ~Qt.WindowMinimized)

                def _force_maximize():
                    # Resize to available geometry (without moving) to avoid "cropped" restores.
                    # Do not call setGeometry(availableGeometry) here, as that can shift the window upward.
                    try:
                        screen = None
                        try:
                            screen = return_window.screen()
                        except Exception:
                            screen = None
                        if screen is None:
                            screen = QApplication.primaryScreen()
                        if screen is not None:
                            geom = screen.availableGeometry()
                            if geom and geom.isValid():
                                return_window.resize(geom.size())
                    except Exception:
                        pass
                    try:
                        return_window.setWindowState(return_window.windowState() | Qt.WindowMaximized)
                    except Exception:
                        pass

                QTimer.singleShot(0, _force_maximize)
                return_window.raise_()
                return_window.activateWindow()
            except Exception:
                pass

    def open_holter_from_dashboard(self):
        QMessageBox.information(self, "Access Restricted", "This account can view reports and history only.")

    def open_hyperkalemia_test(self):
        QMessageBox.information(self, "Access Restricted", "This account can view reports and history only.")

    def open_hrv_test(self):
        QMessageBox.information(self, "Access Restricted", "This account can view reports and history only.")

    def go_to_lead_test(self):
        QMessageBox.information(self, "Access Restricted", "This account can view reports and history only.")

    def check_device_connection(self):
        self._apply_view_only_mode()

    def update_device_ui(self, connected):
        self._apply_view_only_mode()
