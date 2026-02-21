from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QSizePolicy,
    QApplication,
    QFileDialog,
    QLineEdit,
    QComboBox,
    QLabel,
    QDateEdit,
    QGridLayout,
    QFrame,
    QFrame,
    QScrollArea,
    QInputDialog,
    QProgressDialog,
)
import sys
import os
try:
    from utils.cloud_uploader import get_cloud_uploader
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from utils.cloud_uploader import get_cloud_uploader
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import json
import datetime
import shutil
import requests
import webbrowser


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HISTORY_FILE = os.path.join(BASE_DIR, "ecg_history.json")
ECG_DATA_FILE = os.path.join(BASE_DIR, "ecg_data.txt")
REPORTS_INDEX_FILE = os.path.join(BASE_DIR, "reports", "index.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Backend API configuration
BACKEND_API_URL = "https://your-backend-api.com/api/reports"  # Replace with actual backend URL
API_TIMEOUT = 30  # seconds
PUBLIC_REVIEWED_REPORTS_URL = "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/public/reviewed-reports"


class UploadWorker(QThread):
    """Worker thread for background report upload to prevent UI freeze."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, uploader, file_path, doctor_name, metadata=None):
        super().__init__()
        self.uploader = uploader
        self.file_path = file_path
        self.doctor_name = doctor_name
        self.metadata = metadata

    def run(self):
        try:
            result = self.uploader.send_for_doctor_review(
                self.file_path, 
                self.doctor_name, 
                metadata=self.metadata
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class HistoryWindow(QDialog):
    """ECG reports history: shows one row per generated report with basic patient details."""

    def __init__(self, parent=None, username=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Report History")
        self.username = username
        self.all_history_entries = []  # Store all entries for filtering
        self._cloud_preview_map = {}   # Cache reviewed report preview URLs by (patient,date)
        
        # Make window responsive to screen size
        screen = QApplication.desktop().screenGeometry()
        window_width = int(screen.width() * 0.85)
        window_height = int(screen.height() * 0.75)
        self.resize(window_width, window_height)
        
        # Set minimum size for usability
        self.setMinimumSize(900, 500)
        
        # Enable responsive design with enhanced calendar styling
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTableWidget {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                gridline-color: #e9ecef;
                selection-background-color: #007bff;
                selection-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QLineEdit {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
            QComboBox {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                background-color: white;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid #007bff;
                border-top: 5px solid transparent;
                border-bottom: 5px solid transparent;
            }
            QDateEdit {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                background-color: white;
                min-width: 120px;
            }
            QDateEdit:focus {
                border-color: #007bff;
            }
            QDateEdit::drop-down {
                border: none;
                width: 20px;
                image: none;
            }
            QCalendarWidget {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 5px;
            }
            QCalendarWidget QToolButton {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 4px 8px;
                margin: 2px;
                min-width: 30px;
                min-height: 30px;
                font-weight: bold;
                color: #495057;
            }
            QCalendarWidget QToolButton:hover {
                background-color: #e9ecef;
                border-color: #007bff;
                color: #007bff;
            }
            QCalendarWidget QToolButton:pressed {
                background-color: #007bff;
                color: white;
            }
            QCalendarWidget QToolButton::menu-indicator {
                width: 0px;
                height: 0px;
            }
            QCalendarWidget QSpinBox {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                color: #495057;
            }
            QCalendarWidget QSpinBox:focus {
                border-color: #007bff;
            }
            QCalendarWidget QAbstractItemView {
                background-color: white;
                selection-background-color: #007bff;
                selection-color: white;
                outline: none;
            }
            QCalendarWidget QAbstractItemView::item {
                padding: 8px;
                border-radius: 4px;
                margin: 1px;
            }
            QCalendarWidget QAbstractItemView::item:hover {
                background-color: #e9ecef;
                color: #007bff;
            }
            QCalendarWidget QAbstractItemView::item:selected {
                background-color: #007bff;
                color: white;
            }
            QCalendarWidget QTableView {
                background-color: white;
                gridline-color: #e9ecef;
                selection-background-color: #007bff;
                selection-color: white;
            }
            QCalendarWidget QHeaderView {
                background-color: #f8f9fa;
                border: none;
                border-bottom: 2px solid #dee2e6;
                padding: 4px;
                font-weight: bold;
                color: #495057;
            }
            QCalendarWidget QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 8px;
                font-weight: bold;
                color: #495057;
            }
            QLabel {
                font-weight: bold;
                color: #495057;
                margin: 2px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Enhanced search section with date filtering
        search_frame = QFrame()
        search_frame.setFrameStyle(QFrame.Box)
        search_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
            }
        """)
        
        search_layout = QGridLayout(search_frame)
        search_layout.setSpacing(10)
        
        # First row: Search type selector
        search_type_label = QLabel("Search By:")
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["Patient Name", "Date Range", "Single Date"])
        self.search_type_combo.currentTextChanged.connect(self.on_search_type_changed)
        
        search_layout.addWidget(search_type_label, 0, 0)
        search_layout.addWidget(self.search_type_combo, 0, 1)
        
        # Second row: Name search (default visible)
        self.name_search_label = QLabel("Patient Name:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter patient name to search...")
        self.search_input.textChanged.connect(self.filter_table)
        
        search_layout.addWidget(self.name_search_label, 1, 0)
        search_layout.addWidget(self.search_input, 1, 1)
        
        # Third row: Date range search (initially hidden)
        self.date_range_label = QLabel("Date Range:")
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-30))  # Default: last 30 days
        self.start_date_edit.dateChanged.connect(self.filter_table)
        
        self.to_label = QLabel("to")
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.dateChanged.connect(self.filter_table)
        
        search_layout.addWidget(self.date_range_label, 2, 0)
        search_layout.addWidget(self.start_date_edit, 2, 1)
        search_layout.addWidget(self.to_label, 2, 2)
        search_layout.addWidget(self.end_date_edit, 2, 3)
        
        # Fourth row: Single date search (initially hidden)
        self.single_date_label = QLabel("Date:")
        self.single_date_edit = QDateEdit()
        self.single_date_edit.setCalendarPopup(True)
        self.single_date_edit.setDate(QDate.currentDate())
        self.single_date_edit.dateChanged.connect(self.filter_table)
        
        search_layout.addWidget(self.single_date_label, 3, 0)
        search_layout.addWidget(self.single_date_edit, 3, 1)
        
        # Initially hide date search options
        self.date_range_label.hide()
        self.start_date_edit.hide()
        self.to_label.hide()
        self.end_date_edit.hide()
        self.single_date_label.hide()
        self.single_date_edit.hide()
        
        layout.addWidget(search_frame)

        self.table = QTableWidget()
        self.table.setColumnCount(11)  # Added Review Status column
        self.table.setHorizontalHeaderLabels(
            [
                "Date",
                "Time",
                "Org.",
                "Doctor",
                "Patient Name",
                "Age",
                "Gender",
                "Height (cm)",
                "Weight (kg)",
                "Report Type",
                "Review Status",  # New column
            ]
        )
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        
        # Make table expand to fill available space
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Set column stretch factors for proportional expansion
        # Date and Time: smaller, Patient Name and Doctor: larger
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(self.table.horizontalHeader().Stretch)
        
        layout.addWidget(self.table, 1)

        # Connect double-click signal to open report
        self.table.cellDoubleClicked.connect(self.on_row_double_clicked)
        
        # Connect single-click signal for status column
        self.table.cellClicked.connect(self.on_status_cell_clicked)

        # Enhanced buttons row with responsive design
        buttons_frame = QFrame()
        buttons_frame.setFrameStyle(QFrame.NoFrame)
        btn_row = QHBoxLayout(buttons_frame)
        btn_row.setSpacing(10)
        
        # Create responsive buttons with icons and better styling
        self.open_btn = QPushButton(" Open Report")
        self.open_btn.setMinimumHeight(40)
        self.open_btn.clicked.connect(self.open_selected_report)
        btn_row.addWidget(self.open_btn)

        self.export_all_btn = QPushButton(" Export All")
        self.export_all_btn.setMinimumHeight(40)
        self.export_all_btn.clicked.connect(self.export_all_reports)
        btn_row.addWidget(self.export_all_btn)

        self.send_review_btn = QPushButton(" Send for Review")
        self.send_review_btn.setMinimumHeight(40)
        self.send_review_btn.clicked.connect(self.send_report_for_review)
        btn_row.addWidget(self.send_review_btn)

        self.refresh_reviewed_btn = QPushButton(" Refresh Reviewed")
        self.refresh_reviewed_btn.setMinimumHeight(40)
        self.refresh_reviewed_btn.clicked.connect(self.refresh_reviewed_reports)
        btn_row.addWidget(self.refresh_reviewed_btn)

        self.open_cloud_preview_btn = QPushButton(" Open Cloud Preview")
        self.open_cloud_preview_btn.setMinimumHeight(40)
        self.open_cloud_preview_btn.clicked.connect(self.open_cloud_preview_selected)
        btn_row.addWidget(self.open_cloud_preview_btn)

        btn_row.addStretch(1)

        self.close_btn = QPushButton(" Close")
        self.close_btn.setMinimumHeight(40)
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)

        layout.addWidget(buttons_frame)
        # Pre-fetch doctor list in background for faster access
        try:
            import threading
            threading.Thread(target=self._prefetch_doctors, daemon=True).start()
        except Exception as e:
            print(f"⚠️ Could not start doctor prefetch thread: {e}")
            
        self.load_history()
        
        # Connect resize event for responsive design
        self.resizeEvent = self.on_resize_event

    def _prefetch_doctors(self):
        """Fetch doctor list in background to populate cache."""
        try:
            uploader = get_cloud_uploader()
            uploader.get_available_doctors()
        except Exception:
            pass  # Silent fail for prefetch

    def select_doctor_from_list(self, doctors, current_doctor=""):
        """Show a custom scrollable dialog to select a doctor."""
        from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Doctor")
        dialog.setMinimumWidth(350)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Select doctor to send report to:")
        label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(label)
        
        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f1f3f5;
            }
            QListWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        
        for doc in doctors:
            item = QListWidgetItem(doc)
            list_widget.addItem(item)
            if doc == current_doctor:
                item.setSelected(True)
                list_widget.setCurrentItem(item)
                
        layout.addWidget(list_widget)
        
        # Search box for fast filtering
        search_box = QLineEdit()
        search_box.setPlaceholderText("Search doctor name...")
        search_box.textChanged.connect(lambda text: self._filter_doctor_list(list_widget, text))
        layout.addWidget(search_box)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Select")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #6c757d;")
        
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        selected_doctor = [None]
        
        def on_ok():
            current = list_widget.currentItem()
            if current:
                selected_doctor[0] = current.text()
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Selection", "Please select a doctor.")
                
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        list_widget.itemDoubleClicked.connect(on_ok)
        
        if dialog.exec_() == QDialog.Accepted:
            return selected_doctor[0]
        return None

    def _filter_doctor_list(self, list_widget, text):
        """Filter doctor list items based on search text."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def on_resize_event(self, event):
        """Handle window resize for responsive design."""
        super().resizeEvent(event)
        
        # Adjust column widths based on window size
        width = event.size().width()
        
        if width < 1200:
            # Small screens: compact layout
            self.table.horizontalHeader().setDefaultSectionSize(80)
        elif width < 1600:
            # Medium screens: balanced layout
            self.table.horizontalHeader().setDefaultSectionSize(100)
        else:
            # Large screens: spacious layout
            self.table.horizontalHeader().setDefaultSectionSize(120)
        
        # Ensure table fills available space
        self.table.horizontalHeader().setSectionResizeMode(self.table.horizontalHeader().Stretch)

    def load_history(self):
        """Load history entries from both ecg_history.json and reports/index.json into the table."""
        self.table.setRowCount(0)

        history_entries = []

        # Preferred source: rich per-report history
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_entries = json.load(f)
                if not isinstance(history_entries, list):
                    history_entries = []
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load history from ecg_history.json: {e}")
                history_entries = []

        # Also load from reports/index.json (same source as Recent Reports)
        reports_index_entries = []
        if os.path.exists(REPORTS_INDEX_FILE):
            try:
                with open(REPORTS_INDEX_FILE, "r", encoding="utf-8") as f:
                    reports_index_entries = json.load(f)
                if not isinstance(reports_index_entries, list):
                    reports_index_entries = []
            except Exception as e:
                print(f"Failed to load reports index: {e}")
                reports_index_entries = []

        # Convert reports index entries to history format
        for entry in reports_index_entries:
            # Only include ECG Report entries (not detailed metadata)
            if 'filename' in entry and 'title' in entry:
                # Determine report type from title or filename
                title = entry.get('title', 'ECG Report')
                filename = entry.get('filename', '').lower()
                
                if 'hyper' in filename:
                    report_type = "Hyperkalemia"
                elif 'hrv' in filename:
                    report_type = "HRV"
                else:
                    report_type = "ECG"
                
                # Only include ECG, HRV, and Hyperkalemia reports
                if report_type in ["ECG", "HRV", "Hyperkalemia"]:
                    history_entry = {
                        "date": entry.get('date', ''),
                        "time": entry.get('time', ''),
                        "report_type": report_type,
                        "Org.": entry.get('org', ''),
                        "doctor": entry.get('doctor', ''),
                        "patient_name": entry.get('patient', ''),
                        "age": entry.get('age', ''),
                        "gender": entry.get('gender', ''),
                        "height": entry.get('height', ''),
                        "weight": entry.get('weight', ''),
                        "report_file": os.path.join(REPORTS_DIR, entry.get('filename', '')),
                        "username": entry.get('username', '')
                    }
                    history_entries.append(history_entry)

        # Fallback: basic patient list (older flow without report_type)
        if not history_entries:
            patients_file = os.path.join(BASE_DIR, "all_patients.json")
            if not os.path.exists(patients_file):
                return
            try:
                with open(patients_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                patients = data.get("patients", []) if isinstance(data, dict) else []
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load history from all_patients.json: {e}")
                return

            for p in patients:
                patient_name = p.get("patient_name") or (
                    (p.get("first_name", "") + " " + p.get("last_name", "")).strip()
                )
                org = p.get("Org.", "")
                doctor = p.get("doctor", "")
                age = str(p.get("age", ""))
                gender = p.get("gender", "")
                height = str(p.get("height", "")) if p.get("height", "") != "" else ""
                weight = str(p.get("weight", "")) if p.get("weight", "") != "" else ""

                date_time = p.get("date_time", "")
                date_str, time_str = "", ""
                if date_time and " " in date_time:
                    date_str, time_str = date_time.split(" ", 1)
                elif date_time:
                    date_str = date_time

                entry = {
                    "date": date_str,
                    "time": time_str,
                    "report_type": "ECG",
                    "Org.": org,
                    "doctor": doctor,
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "report_file": "",
                }
                history_entries.append(entry)

        # Store all entries and filter for ECG, 12 HRV, and Hyperkalemia reports
        self.all_history_entries = []
        for entry in history_entries:
            # Apply username filtering - if username is specified, only show reports for that user
            if self.username and entry.get("username") and entry.get("username") != self.username:
                continue

            report_file = entry.get("report_file", "") or ""
            report_type = entry.get("report_type", "")
            if not report_type:
                file_lower = report_file.lower()
                if "hyper" in file_lower:
                    report_type = "Hyperkalemia"
                elif "hrv" in file_lower:
                    report_type = "HRV"
                elif "ecg" in file_lower:
                    report_type = "ECG"
                else:
                    report_type = "ECG"
            
            # Filter for ECG, 12 HRV, and Hyperkalemia reports
            if report_type in ["ECG", "HRV", "Hyperkalemia"]:
                entry["report_type"] = report_type
                self.all_history_entries.append(entry)
        
        # Sort entries by date and time in reverse chronological order (latest first)
        def get_datetime_key(entry):
            """Extract datetime for sorting - returns tuple (date, time) for comparison."""
            try:
                date_str = entry.get("date", "")
                time_str = entry.get("time", "")
                
                # Parse date (format: YYYY-MM-DD)
                if date_str:
                    date_parts = date_str.split("-")
                    if len(date_parts) == 3:
                        date_tuple = tuple(map(int, date_parts))
                    else:
                        date_tuple = (0, 0, 0)
                else:
                    date_tuple = (0, 0, 0)
                
                # Parse time (format: HH:MM:SS)
                if time_str:
                    time_parts = time_str.split(":")
                    if len(time_parts) >= 2:
                        time_tuple = tuple(map(int, time_parts[:3] if len(time_parts) == 3 else time_parts + ["0"]))
                    else:
                        time_tuple = (0, 0, 0)
                else:
                    time_tuple = (0, 0, 0)
                
                return (date_tuple, time_tuple)
            except:
                return ((0, 0, 0), (0, 0, 0))
        
        # Sort in reverse order (latest first)
        self.all_history_entries.sort(key=get_datetime_key, reverse=True)
        
        # Add sorted entries to table
        for entry in self.all_history_entries:
            self.add_row(entry)

    def on_search_type_changed(self, search_type):
        """Handle search type change to show/hide appropriate search options."""
        if search_type == "Patient Name":
            # Show name search, hide date searches
            self.name_search_label.show()
            self.search_input.show()
            self.date_range_label.hide()
            self.start_date_edit.hide()
            self.to_label.hide()
            self.end_date_edit.hide()
            self.single_date_label.hide()
            self.single_date_edit.hide()
        elif search_type == "Date Range":
            # Show date range search, hide others
            self.name_search_label.hide()
            self.search_input.hide()
            self.date_range_label.show()
            self.start_date_edit.show()
            self.to_label.show()
            self.end_date_edit.show()
            self.single_date_label.hide()
            self.single_date_edit.hide()
        elif search_type == "Single Date":
            # Show single date search, hide others
            self.name_search_label.hide()
            self.search_input.hide()
            self.date_range_label.hide()
            self.start_date_edit.hide()
            self.to_label.hide()
            self.end_date_edit.hide()
            self.single_date_label.show()
            self.single_date_edit.show()
        
        # Trigger filter update
        self.filter_table()

    def filter_table(self):
        """Filter table rows based on search input (name or date)."""
        search_type = self.search_type_combo.currentText()
        
        # Clear current table
        self.table.setRowCount(0)
        
        # Re-add entries that match the search
        for entry in self.all_history_entries:
            should_show = False
            
            if search_type == "Patient Name":
                search_text = self.search_input.text().strip().lower()
                patient_name = entry.get("patient_name", "").lower()
                should_show = search_text == "" or search_text in patient_name
                
            elif search_type == "Date Range":
                start_date = self.start_date_edit.date().toPyDate()
                end_date = self.end_date_edit.date().toPyDate()
                entry_date_str = entry.get("date", "")
                if entry_date_str:
                    try:
                        entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
                        should_show = start_date <= entry_date <= end_date
                    except ValueError:
                        should_show = False
                else:
                    should_show = False
                    
            elif search_type == "Single Date":
                search_date = self.single_date_edit.date().toPyDate()
                entry_date_str = entry.get("date", "")
                if entry_date_str:
                    try:
                        entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
                        should_show = entry_date == search_date
                    except ValueError:
                        should_show = False
                else:
                    should_show = False
            
            if should_show:
                self.add_row(entry)

    def _get_report_datetime(self, patient_name, reports_index):
        """Get the actual date/time when report was generated for a patient."""
        # First, try to find in reports index
        if patient_name and patient_name in reports_index:
            patient_reports = reports_index[patient_name]
            if patient_reports:
                # Use the most recent report (first in list, as index.json is usually sorted by newest first)
                most_recent = patient_reports[0]
                date_str = most_recent.get("date", "")
                time_str = most_recent.get("time", "")
                if date_str and time_str:
                    return date_str, time_str
        
        # If not found in index, try to get from PDF file modification time
        if patient_name:
            pdf_file = self._find_report_file(patient_name, "")
            if pdf_file and os.path.exists(pdf_file):
                try:
                    mod_time = os.path.getmtime(pdf_file)
                    dt = datetime.datetime.fromtimestamp(mod_time)
                    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
                except:
                    pass
        
        # Fallback: use current time (should rarely happen)
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    def add_row(self, entry):
        """Append one row for a history entry."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Extract safe values
        date_str = entry.get("date", "")
        time_str = entry.get("time", "")
        org = entry.get("Org.", "")
        report_type = entry.get("report_type", "")
        doctor = entry.get("doctor", "")
        patient_name = entry.get("patient_name", "") or (
            (entry.get("first_name", "") + " " + entry.get("last_name", "")).strip()
        )
        age = str(entry.get("age", ""))
        gender = entry.get("gender", "")
        height = str(entry.get("height", ""))
        weight = str(entry.get("weight", ""))
        review_status = entry.get("review_status", "Pending")  # Default to Pending

        values = [
            date_str,
            time_str,
            org,
            doctor,
            patient_name,
            age,
            gender,
            height,
            weight,
            report_type,
            review_status,  # New column
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            
            # Apply color coding to Review Status column (column 10)
            if col == 10:  # Review Status column
                from PyQt5.QtGui import QColor
                if val == "Pending":
                    item.setBackground(QColor("#e9ecef"))
                    item.setForeground(QColor("#6c757d"))
                elif val == "Under Review":
                    item.setBackground(QColor("#fff3cd"))
                    item.setForeground(QColor("#856404"))
                elif val == "Reviewed":
                    item.setBackground(QColor("#d4edda"))
                    item.setForeground(QColor("#155724"))
            
            self.table.setItem(row, col, item)

        # Store report filename (if any) as row data for later open
        report_file = entry.get("report_file", "")
        self.table.setVerticalHeaderItem(row, QTableWidgetItem(""))
        self.table.setRowHeight(row, 24)
        # Use Qt.UserRole to store extra data on first column item
        if self.table.item(row, 0):
            self.table.item(row, 0).setData(Qt.UserRole, report_file)

    def on_row_double_clicked(self, row, column):
        """Handle double-click on a table row to open the report."""
        self.open_report_by_row(row)
    
    def on_status_cell_clicked(self, row, column):
        """Handle click on Review Status column to change status."""
        # Only handle clicks on Review Status column (column 10)
        if column != 10:
            return
        
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtGui import QCursor
        
        # Get current status
        status_item = self.table.item(row, 10)
        if not status_item:
            return
        
        current_status = status_item.text()
        
        # Create context menu with status options
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 2px solid #007bff;
                border-radius: 6px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        
        # Add status options
        pending_action = menu.addAction("⚪ Pending")
        review_action = menu.addAction("🟡 Under Review")
        reviewed_action = menu.addAction("🟢 Reviewed")
        
        # Mark current status
        if current_status == "Pending":
            pending_action.setEnabled(False)
        elif current_status == "Under Review":
            review_action.setEnabled(False)
        elif current_status == "Reviewed":
            reviewed_action.setEnabled(False)
        
        # Show menu and get selection
        action = menu.exec_(QCursor.pos())
        
        # Update status based on selection
        if action == pending_action:
            self.update_review_status(row, "Pending")
        elif action == review_action:
            self.update_review_status(row, "Under Review")
        elif action == reviewed_action:
            self.update_review_status(row, "Reviewed")

    def update_review_status(self, row, new_status):
        """Update review status for a report and save to file and backend."""
        from PyQt5.QtGui import QColor
        
        try:
            # Update table cell
            status_item = self.table.item(row, 10)
            if not status_item:
                return
            
            status_item.setText(new_status)
            
            # Update cell styling
            if new_status == "Pending":
                status_item.setBackground(QColor("#e9ecef"))
                status_item.setForeground(QColor("#6c757d"))
            elif new_status == "Under Review":
                status_item.setBackground(QColor("#fff3cd"))
                status_item.setForeground(QColor("#856404"))
            elif new_status == "Reviewed":
                status_item.setBackground(QColor("#d4edda"))
                status_item.setForeground(QColor("#155724"))
            
            # Get patient name to find entry in all_history_entries
            patient_item = self.table.item(row, 4)  # Patient Name column
            date_item = self.table.item(row, 0)  # Date column
            
            if not patient_item or not date_item:
                return
            
            patient_name = patient_item.text()
            date_str = date_item.text()
            
            # Update in all_history_entries
            for entry in self.all_history_entries:
                if (entry.get("patient_name") == patient_name and 
                    entry.get("date") == date_str):
                    entry["review_status"] = new_status
                    entry["review_updated_at"] = datetime.datetime.now().isoformat()
                    entry["review_updated_by"] = self.username or "unknown"
                    break
            
            # Save to ecg_history.json
            self._save_history_to_file()
            
            # Send update to backend API
            self._send_status_update_to_backend(row, new_status)
            
            print(f"✅ Review status updated to '{new_status}' for {patient_name}")
            
        except Exception as e:
            print(f"❌ Error updating review status: {e}")
            QMessageBox.warning(
                self,
                "Update Failed",
                f"Failed to update review status: {str(e)}"
            )
    
    def _save_history_to_file(self):
        """Save all_history_entries back to ecg_history.json."""
        try:
            # Load existing history file
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    all_entries = json.load(f)
            else:
                all_entries = []
            
            if not isinstance(all_entries, list):
                all_entries = []
            
            # Update entries with review status
            for entry in self.all_history_entries:
                patient_name = entry.get("patient_name", "")
                date_str = entry.get("date", "")
                
                # Find matching entry in all_entries
                found = False
                for saved_entry in all_entries:
                    if (saved_entry.get("patient_name") == patient_name and
                        saved_entry.get("date") == date_str):
                        # Update review status fields
                        saved_entry["review_status"] = entry.get("review_status", "Pending")
                        saved_entry["review_updated_at"] = entry.get("review_updated_at", "")
                        saved_entry["review_updated_by"] = entry.get("review_updated_by", "")
                        found = True
                        break
                
                # If not found, add new entry
                if not found:
                    all_entries.append(entry)
            
            # Save back to file
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(all_entries, f, indent=2)
            
            print(f"💾 Saved review status to {HISTORY_FILE}")
            
        except Exception as e:
            print(f"❌ Error saving history to file: {e}")
    
    def _send_status_update_to_backend(self, row, new_status):
        """Send review status update to backend API."""
        try:
            # Get report data
            report_data = self.get_report_data_from_row(row)
            if not report_data:
                return
            
            # Add review status
            report_data["review_status"] = new_status
            report_data["review_updated_at"] = datetime.datetime.now().isoformat()
            report_data["review_updated_by"] = self.username or "unknown"
            
            # Prepare payload
            payload = {
                "action": "update_review_status",
                "report": report_data,
                "metadata": {
                    "source": "ecg_monitor",
                    "version": "1.0",
                    "updated_by": self.username or "unknown"
                }
            }
            
            # Make API request (non-blocking)
            try:
                response = requests.post(
                    BACKEND_API_URL,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "ECG-Monitor/1.0"
                    },
                    timeout=5  # Short timeout for status updates
                )
                
                if response.status_code == 200:
                    print(f"📤 Review status sent to backend: {new_status}")
                else:
                    print(f"⚠️ Backend API returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print("⚠️ Backend API timeout (status update)")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Backend API error: {e}")
                
        except Exception as e:
            print(f"❌ Error sending status to backend: {e}")

    def open_selected_report(self):
        """Open the PDF report for the selected row, if available."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Open Report", "Please select a report row first.")
            return
        self.open_report_by_row(row)

    def open_report_by_row(self, row):
        """Open the PDF report for a specific row."""
        if row < 0 or row >= self.table.rowCount():
            return

        # First, try to get report file from stored data
        item = self.table.item(row, 0)
        if item:
            report_file = item.data(Qt.UserRole) or ""
            if report_file and os.path.exists(report_file):
                self._open_pdf_file(report_file)
                return

        # If no direct file path, try to find report based on patient details
        patient_name_item = self.table.item(row, 4)  # Patient Name column
        date_item = self.table.item(row, 0)  # Date column
        
        if not patient_name_item:
            QMessageBox.warning(
                self,
                "Open Report",
                "Could not find patient information for this entry."
            )
            return

        patient_name = patient_name_item.text().strip()
        date_str = date_item.text().strip() if date_item else ""

        # Try to find matching report file
        report_file = self._find_report_file(patient_name, date_str)
        
        if report_file and os.path.exists(report_file):
            self._open_pdf_file(report_file)
        else:
            # Attempt cloud preview via public reviewed reports API
            if self._open_reviewed_report_via_api(patient_name, date_str):
                return
            QMessageBox.information(
                self,
                "Report Not Found",
                f"Could not find a local PDF for '{patient_name}'.\n"
                f"Tried cloud preview from reviewed reports API as well."
            )

    def _find_report_file(self, patient_name, date_str=""):
        """Try to find a report file matching the patient name and optionally date."""
        reports_dir = os.path.join(BASE_DIR, "reports")
        if not os.path.exists(reports_dir):
            return None

        # Get all PDF files
        pdf_files = [f for f in os.listdir(reports_dir) if f.lower().endswith('.pdf')]
        
        # Try to find exact match by patient name in filename
        patient_name_clean = patient_name.replace(" ", "_").replace(",", "").upper()
        
        # First, try to find files with patient name
        for pdf_file in pdf_files:
            pdf_upper = pdf_file.upper()
            if patient_name_clean in pdf_upper:
                return os.path.join(reports_dir, pdf_file)
        
        # If date is provided, try to find by date pattern (YYYYMMDD)
        if date_str:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                date_pattern = date_obj.strftime("%Y%m%d")
                
                for pdf_file in pdf_files:
                    if date_pattern in pdf_file:
                        return os.path.join(reports_dir, pdf_file)
            except:
                pass

        # Return most recent ECG_Report file if no match found
        ecg_reports = [f for f in pdf_files if f.startswith("ECG_Report_")]
        if ecg_reports:
            # Sort by modification time, most recent first
            ecg_reports.sort(key=lambda f: os.path.getmtime(os.path.join(reports_dir, f)), reverse=True)
            return os.path.join(reports_dir, ecg_reports[0])

        return None

    def _open_pdf_file(self, report_file):
        """Open a PDF file using the system's default PDF viewer."""
        try:
            if os.name == "nt":
                os.startfile(report_file)
            elif sys.platform == "darwin":
                os.system(f'open "{report_file}"')
            else:
                os.system(f'xdg-open "{report_file}"')
        except Exception as e:
            QMessageBox.critical(self, "Open Report", f"Failed to open report: {e}")
    
    def _open_reviewed_report_via_api(self, patient_name, date_str=""):
        """Try opening a reviewed report preview via the public API."""
        try:
            # Build request; if backend supports query, pass patient/date
            params = {}
            if patient_name:
                params["patient"] = patient_name
            if date_str:
                params["date"] = date_str
            
            resp = requests.get(PUBLIC_REVIEWED_REPORTS_URL, params=params, timeout=10)
            if resp.status_code != 200:
                return False
            
            data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else None
            # If JSON list, try to find matching record and open its preview/file URL
            candidate_url = None
            if isinstance(data, list) and len(data) > 0:
                # Prefer records matching patient/date
                def matches(entry):
                    pn = str(entry.get("patient", entry.get("name", ""))).strip().lower()
                    dt = str(entry.get("date", "")).strip()
                    ok_p = not patient_name or patient_name.lower() in pn
                    ok_d = not date_str or date_str == dt
                    return ok_p and ok_d
                for entry in data:
                    if matches(entry):
                        candidate_url = entry.get("preview_url") or entry.get("file_url") or entry.get("url")
                        key = (patient_name or "").strip(), (date_str or "").strip()
                        if key[0] or key[1]:
                            if candidate_url:
                                self._cloud_preview_map[key] = candidate_url
                        if candidate_url:
                            break
                # Fallback to first record with any URL
                if not candidate_url:
                    for entry in data:
                        candidate_url = entry.get("preview_url") or entry.get("file_url") or entry.get("url")
                        if candidate_url:
                            break
            # If not JSON, try to extract a URL from text
            if not candidate_url and isinstance(resp.text, str):
                txt = resp.text
                # Naive URL find
                for token in txt.split():
                    if token.startswith("http://") or token.startswith("https://"):
                        candidate_url = token
                        break
            
            if candidate_url:
                webbrowser.open(candidate_url)
                return True
            return False
        except Exception:
            return False

    def refresh_reviewed_reports(self):
        """Fetch reviewed reports from public API and mark matching rows as Reviewed."""
        try:
            resp = requests.get(PUBLIC_REVIEWED_REPORTS_URL, timeout=10)
            if resp.status_code != 200:
                QMessageBox.information(self, "Reviewed Reports", "Could not fetch reviewed reports from cloud.")
                return
            data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else []
            if not isinstance(data, list):
                data = []
            # Build quick lookup by normalized patient/date
            def norm(s): return str(s or "").strip().lower()
            lookup = {}
            for entry in data:
                pn = norm(entry.get("patient", entry.get("name", "")))
                dt = str(entry.get("date", "")).strip()
                url = entry.get("preview_url") or entry.get("file_url") or entry.get("url")
                if pn or dt:
                    lookup[(pn, dt)] = url
            # Iterate rows and update status
            updated = 0
            for row in range(self.table.rowCount()):
                patient_item = self.table.item(row, 4)
                date_item = self.table.item(row, 0)
                if not patient_item or not date_item:
                    continue
                key = (norm(patient_item.text()), date_item.text().strip())
                url = lookup.get(key)
                if url:
                    self.update_review_status(row, "Reviewed")
                    self._cloud_preview_map[(patient_item.text().strip(), date_item.text().strip())] = url
                    updated += 1
            QMessageBox.information(self, "Reviewed Reports", f"Updated {updated} row(s) to Reviewed from cloud.")
        except Exception as e:
            QMessageBox.warning(self, "Reviewed Reports", f"Failed to refresh reviewed reports: {e}")

    def open_cloud_preview_selected(self):
        """Open cloud preview for the selected row, using cache or API lookup."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Cloud Preview", "Please select a report row first.")
            return
        patient_item = self.table.item(row, 4)
        date_item = self.table.item(row, 0)
        patient = patient_item.text().strip() if patient_item else ""
        date_str = date_item.text().strip() if date_item else ""
        key = (patient, date_str)
        url = self._cloud_preview_map.get(key)
        if url:
            try:
                webbrowser.open(url)
                return
            except Exception:
                pass
        # Fallback: query API now
        if not self._open_reviewed_report_via_api(patient, date_str):
            QMessageBox.information(self, "Cloud Preview", "No reviewed report found for this row.")

    def export_all_reports(self):
        """Export all saved reports from history to a user-selected directory."""
        # Ask user where to save the exported reports
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export All Reports",
            os.path.expanduser("~/Desktop"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not export_dir:
            return  # User cancelled
        
        try:
            reports_dir = os.path.join(BASE_DIR, "reports")
            if not os.path.exists(reports_dir):
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    f"Reports directory not found: {reports_dir}"
                )
                return
            
            # Get all PDF files from reports directory
            pdf_files = [f for f in os.listdir(reports_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                QMessageBox.information(
                    self,
                    "No Reports",
                    "No PDF reports found to export."
                )
                return
            
            # Create a subdirectory with timestamp for organized export
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = os.path.join(export_dir, f"ECG_Reports_Export_{timestamp}")
            os.makedirs(export_subdir, exist_ok=True)
            
            # Copy all PDF files
            copied_count = 0
            failed_count = 0
            
            for pdf_file in pdf_files:
                try:
                    src_path = os.path.join(reports_dir, pdf_file)
                    dst_path = os.path.join(export_subdir, pdf_file)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f" Failed to copy {pdf_file}: {e}")
                    failed_count += 1
            
            # Also export history data as CSV
            csv_path = os.path.join(export_subdir, "history_data.csv")
            try:
                with open(csv_path, "w", encoding="utf-8") as csv_file:
                    # Write header
                    csv_file.write("Date,Time,Report Type,Org.,Doctor,Patient Name,Age,Gender,Height (cm),Weight (kg)\n")
                    
                    # Write data from table
                    for row in range(self.table.rowCount()):
                        row_data = []
                        for col in range(self.table.columnCount()):
                            item = self.table.item(row, col)
                            value = item.text() if item else ""
                            # Escape commas and quotes in CSV
                            if "," in value or '"' in value:
                                value = '"' + value.replace('"', '""') + '"'
                            row_data.append(value)
                        csv_file.write(",".join(row_data) + "\n")
            except Exception as e:
                print(f" Failed to export CSV: {e}")
            
            # Show success message
            message = f"Export completed!\n\n"
            message += f"Location: {export_subdir}\n"
            message += f"PDFs copied: {copied_count}\n"
            if failed_count > 0:
                message += f"Failed: {failed_count}\n"
            message += f"\nHistory data exported as CSV."
            
            QMessageBox.information(
                self,
                "Export Successful",
                message
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export reports: {e}\n\nPlease check console for details."
            )
            import traceback
            traceback.print_exc()

    def send_report_for_review(self):
        """Send the selected report for review to backend API using CloudUploader."""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Send for Review", "Please select a report row first.")
            return

        try:
            # Get report details from selected row
            report_data = self.get_report_data_from_row(row)
            if not report_data:
                QMessageBox.warning(self, "Send for Review", "Could not get report details.")
                return

            # File path
            report_file = report_data.get('report_file_path')
            if not report_file or not os.path.exists(report_file):
                 QMessageBox.warning(self, "Send for Review", "Report file not found.")
                 return

            # Get available doctors
            uploader = get_cloud_uploader()
            available_doctors = uploader.get_available_doctors()
            
            if not available_doctors:
                QMessageBox.warning(self, "Error", "Could not fetch doctor list.")
                return

            # Pre-select current doctor if available
            current_doctor = report_data.get('doctor', '')
            default_index = 0
            if current_doctor in available_doctors:
                default_index = available_doctors.index(current_doctor)
            
            # Show selection dialog
            doctor_name = self.select_doctor_from_list(available_doctors, current_doctor)
            
            if not doctor_name:
                return
            
            # Confirmation
            reply = QMessageBox.question(
                self, 
                "Send for Review",
                f"Send {report_data['report_type']} report for {report_data['patient_name']} to {doctor_name}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

            # Show non-blocking progress dialog
            self.progress_dialog = QProgressDialog(f"Uploading report for {doctor_name}...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowTitle("Sending Report")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.show()

            # Create and start worker thread
            self.upload_worker = UploadWorker(uploader, report_file, doctor_name, metadata=report_data)
            self.upload_worker.finished.connect(lambda res: self.on_upload_finished(res, row, doctor_name))
            self.upload_worker.error.connect(self.on_upload_error)
            self.upload_worker.start()

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"An error occurred: {str(e)}"
            )

    def on_upload_finished(self, result, row, doctor_name):
        """Handle upload completion from worker thread."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            
        if result.get("status") == "success":
            QMessageBox.information(
                self, 
                "Success", 
                f"Report sent successfully to {doctor_name}!\n\n{result.get('message')}"
            )
            # Update status
            self.update_review_status(row, "Under Review")
            
        elif result.get("status") == "queued":
             QMessageBox.information(
                self, 
                "Queued", 
                f"Offline: {result.get('message')}"
            )
             self.update_review_status(row, "Queued")
        else:
            QMessageBox.warning(
                self, 
                "Failed", 
                f"Failed to send report.\n\n{result.get('message')}"
            )

    def on_upload_error(self, error_msg):
        """Handle upload error from worker thread."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            
        QMessageBox.critical(
            self, 
            "Error", 
            f"An error occurred in background upload: {error_msg}"
        )

    def get_report_data_from_row(self, row):
        """Extract report data from the selected table row."""
        try:
            # Get data from table cells
            date_item = self.table.item(row, 0)
            time_item = self.table.item(row, 1)
            org_item = self.table.item(row, 2)
            doctor_item = self.table.item(row, 3)
            patient_item = self.table.item(row, 4)
            age_item = self.table.item(row, 5)
            gender_item = self.table.item(row, 6)
            height_item = self.table.item(row, 7)
            weight_item = self.table.item(row, 8)
            report_type_item = self.table.item(row, 9)
            review_status_item = self.table.item(row, 10)  # New: Review Status column

            # Get report file path from stored data
            first_item = self.table.item(row, 0)
            report_file = first_item.data(Qt.UserRole) if first_item else ""

            # Find corresponding history entry for more details
            patient_name = patient_item.text() if patient_item else ""
            history_entry = None
            for entry in self.all_history_entries:
                if entry.get("patient_name", "") == patient_name:
                    history_entry = entry
                    break

            report_data = {
                "date": date_item.text() if date_item else "",
                "time": time_item.text() if time_item else "",
                "organization": org_item.text() if org_item else "",
                "doctor": doctor_item.text() if doctor_item else "",
                "patient_name": patient_name,
                "age": age_item.text() if age_item else "",
                "gender": gender_item.text() if gender_item else "",
                "height": height_item.text() if height_item else "",
                "weight": weight_item.text() if weight_item else "",
                "report_type": report_type_item.text() if report_type_item else "",
                "review_status": review_status_item.text() if review_status_item else "Pending",  # New field
                "report_file_path": report_file,
                "username": self.username,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Add additional data from history entry if available
            if history_entry:
                report_data.update({
                    "report_file": history_entry.get("report_file", ""),
                    "original_entry": history_entry,
                    "review_updated_at": history_entry.get("review_updated_at", ""),
                    "review_updated_by": history_entry.get("review_updated_by", "")
                })

            return report_data

        except Exception as e:
            print(f"Error getting report data from row: {e}")
            return None

    def send_to_backend_api(self, report_data):
        """Send report data to backend API."""
        try:
            # Prepare payload for backend
            payload = {
                "action": "send_for_review",
                "report": report_data,
                "metadata": {
                    "source": "ecg_monitor",
                    "version": "1.0",
                    "sent_by": self.username or "unknown"
                }
            }

            # Make API request
            response = requests.post(
                BACKEND_API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ECG-Monitor/1.0"
                },
                timeout=API_TIMEOUT
            )

            # Check response
            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                print(f"API Error: Status {response.status_code}, Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return False
        except Exception as e:
            print(f"API call error: {e}")
            return False


def append_history_entry(patient_details, report_file_path, report_type="ECG", username=None):
    """Append a new history entry when a report is generated."""
    print(f" append_history_entry called with patient_details={patient_details}, report_file_path={report_file_path}, username={username}")
    
    # Load existing dedicated history file (rich entries)
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                entries = json.load(f)
        else:
            entries = []
            print(f" Creating new history file: {HISTORY_FILE}")
    except Exception as e:
        print(f" Error loading history file: {e}")
        entries = []

    if not isinstance(entries, list):
        entries = []

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    base = {
        "date": date_str,
        "time": time_str,
        "report_type": report_type,
        "username": username,
        "report_file": os.path.abspath(report_file_path) if report_file_path else "",
        "review_status": "Pending",  # Default status for new reports
        "review_updated_at": "",
        "review_updated_by": "",
    }
    if isinstance(patient_details, dict):
        base.update(patient_details)
        print(f" Merged patient details into base entry")
    else:
        print(f" patient_details is not a dict: {type(patient_details)}")

    entries.append(base)
    print(f" Added entry to history. Total entries: {len(entries)}")

    # Save rich history file
    try:
        # Ensure directory exists (HISTORY_FILE is in project root, so dirname should exist)
        history_dir = os.path.dirname(HISTORY_FILE)
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(entries, f, indent=2)
        print(f" Successfully saved history to {HISTORY_FILE}")
        print(f" Entry content: {json.dumps(base, indent=2)}")
    except Exception as e:
        # History is non-critical; just print warning
        print(f" Failed to append ECG history entry: {e}")
        import traceback
        traceback.print_exc()

