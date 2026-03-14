"""
Hyperkalemia Detection Module - ECG-based hyperkalemia detection according to medical standards
This module provides a dedicated window for hyperkalemia testing with automatic ECG analysis.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox,
    QSizePolicy, QFrame, QGridLayout, QApplication
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - Hyperkalemia test hardware features disabled")
    SERIAL_AVAILABLE = False
    class Serial:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def readline(self): return b''
        def write(self, data): pass
        def reset_input_buffer(self): pass
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()

from utils.settings_manager import SettingsManager
from utils.crash_logger import get_crash_logger
from dashboard.history_window import append_history_entry

# Import ECGTestPage + helpers to reuse EXACT same calculation + smoothing as 12‑lead test
try:
    from ecg.twelve_lead_test import ECGTestPage, SamplingRateCalculator, SerialStreamReader
    from PyQt5.QtWidgets import QStackedWidget
    from ecg.hyperkalemia_ecg_report_generator import generate_hyperkalemia_report
    ECG_TEST_AVAILABLE = True
except ImportError:
    ECG_TEST_AVAILABLE = False
    print(" ECGTestPage not available - using fallback calculations")


class HyperkalemiaTestWindow(QWidget):
    """Hyperkalemia Detection Window - ECG analysis for hyperkalemia detection"""
    
    def __init__(self, parent=None, username=None):
        super().__init__(parent)

        # Full screen on open
        self.showFullScreen()

        # Only show close button, disable minimize/maximize/restore
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint | Qt.MSWindowsFixedSizeDialogHint)
        # Prevent window from being moved/draggable
        self.setWindowFlag(Qt.WindowTitleHint, True)

        self.dashboard_instance = parent  # Store reference to dashboard
        self.username = username
        self.setWindowTitle("Hyperkalemia Detection Test")
        try:
            screen = QApplication.primaryScreen().availableGeometry()
            width = int(screen.width() * 0.90)
            height = int(screen.height() * 0.90)
            self.resize(width, height)
            
            # Center the window
            x = (screen.width() - width) // 2
            y = (screen.height() - height) // 2
            self.move(x, y)
        except Exception:
            # Fallback if screen geometry fails
            self.resize(1400, 1000)
            
        self.setMinimumSize(1024, 768)
        # Set window flags to make it a separate window
        # self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Data storage - use circular buffer like 12-lead test
        HISTORY_LENGTH = 10000
        self.data = np.zeros(HISTORY_LENGTH, dtype=np.float32)  # Circular buffer for Lead II
        self.lead_ii_data = []  # Store all captured data with timestamps
        
        # Add data storage for V1-V6 leads (indices 6-11 in ecg_calculator.data)
        self.lead_data = {
            'II': [],  # Lead II
            'V1': [],  # V1
            'V2': [],  # V2
            'V3': [],  # V3
            'V4': [],  # V4
            'V5': [],  # V5
            'V6': []   # V6
        }
        
        # Lead mapping: name -> index in ecg_calculator.data
        self.lead_indices = {
            'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11
        }
        
        self.start_time = None
        self.capture_duration = 30  # 30 seconds for hyperkalemia detection
        self.is_capturing = False
        self.serial_reader = None
        self.crash_logger = get_crash_logger()
        
        # For adaptive scaling per lead
        self.y_centers = {lead: 0.0 for lead in self.lead_data.keys()}
        self.y_ranges = {lead: 200.0 for lead in self.lead_data.keys()}
        
        # Backward compatibility
        self.y_center = 0.0
        self.y_range = 200.0  # Initial range
        self.sampling_rate = 500.0  # Default sampling rate, will be estimated
        self.sample_index = 0
        
        # Settings
        self.settings_manager = SettingsManager()
        
        # Track active sample count to avoid skewing stats with leading zeros
        self.active_samples = 0

        # Store last displayed metrics so analysis dialog matches dashboard values
        self.last_metrics = {}
        
        # Create a minimal ECGTestPage instance to reuse its calculation methods
        # This ensures we use the EXACT same functions as the 12-lead test
        self.ecg_calculator = None
        if ECG_TEST_AVAILABLE:
            try:
                # Create a dummy stacked widget for ECGTestPage initialization
                dummy_stack = QStackedWidget()
                self.ecg_calculator = ECGTestPage("12 Lead ECG Test", dummy_stack)
                
                # IMPORTANT: Sync sampling rate from parent dashboard if available
                if parent and hasattr(parent, 'ecg_test_page'):
                    p_page = parent.ecg_test_page
                    if hasattr(p_page, 'sampler') and p_page.sampler.sampling_rate > 0:
                        self.sampling_rate = p_page.sampler.sampling_rate
                        print(f" Synced sampling rate from dashboard: {self.sampling_rate} Hz")
                
                if not hasattr(self.ecg_calculator, 'data') or len(self.ecg_calculator.data) < 12:
                    self.ecg_calculator.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
                
                if not hasattr(self.ecg_calculator, 'sampler'):
                    self.ecg_calculator.sampler = SamplingRateCalculator()
                self.ecg_calculator.sampler.sampling_rate = self.sampling_rate
                
                # CRITICAL: unique instance_id prevents sharing 12-lead smoothing buffers
                self.ecg_calculator._instance_id = 'hyperkalemia_test'

                print(" ECG calculator initialized for Hyperkalemia test")
            except Exception as e:
                print(f" Could not create ECG calculator: {e}")
                import traceback
                traceback.print_exc()
                self.ecg_calculator = None
        
        # ── HolterBPMController: stable BPM engine (background thread) ─────────
        try:
            from ecg.holter.holter_bpm_engine import HolterBPMController
            self._bpm_ctrl = HolterBPMController(
                parent_widget=self,
                fs=500,
                chunk_seconds=10,
            )
        except Exception as _e:
            print(f"[HyperkalemiaTestWindow] HolterBPMController init failed: {_e}")
            self._bpm_ctrl = None

        # Initialize UI
        self.init_ui()
        
        # Timers
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.update_plot)
        self.duration_timer = QTimer(self)
        self.duration_timer.timeout.connect(self.check_duration)

    def mousePressEvent(self, event):
        # Block all mouse press events (left/right click) for dragging
        event.ignore()

    def mouseMoveEvent(self, event):
        # Block all mouse move events for dragging
        event.ignore()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Hyperkalemia Detection Test")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setStyleSheet("color: #d2691e;")  # Orange suede color
        header.addWidget(title)
        header.addStretch()
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        header.addWidget(self.status_label)
        
        # Timer label
        self.timer_label = QLabel("Time: 00:00")
        self.timer_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.timer_label.setStyleSheet("color: #d2691e; padding: 5px;")
        header.addWidget(self.timer_label)
        
        layout.addLayout(header)
        
        # Control buttons
        controls = QHBoxLayout()
        
        # Start button
        self.start_btn = QPushButton("Start Capture")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #28a745;
                color: white;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #218838;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_capture)
        controls.addWidget(self.start_btn)
        
        # Stop button (initially disabled)
        self.stop_btn = QPushButton("Stop Capture")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #dc3545;
                color: white;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #c82333;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.stop_btn.clicked.connect(self.confirm_stop)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        
        controls.addStretch()
        
        # Analyze button (initially disabled)
        self.analyze_btn = QPushButton("Analyze for Hyperkalemia")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: #d2691e;
                color: white;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #cd853f;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.analyze_btn.clicked.connect(self.analyze_hyperkalemia)
        self.analyze_btn.setEnabled(False)
        controls.addWidget(self.analyze_btn)
        
        # Generate Report button (initially disabled)
        self.report_btn = QPushButton("Generate Report")
        self.report_btn.setStyleSheet("""
            QPushButton {
                background: #d2691e;
                color: white;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #cd853f;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setEnabled(False)
        controls.addWidget(self.report_btn)
        
        layout.addLayout(controls)
        
        # Metrics display section
        metrics_card = QFrame()
        metrics_card.setStyleSheet("background: white; border-radius: 10px; padding: 7px;")
        metrics_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setSpacing(20)
        
        # Store metric labels for live update
        self.metric_labels = {}
        metric_info = [
            ("HR", "00", "BPM", "heart_rate"),
            ("PR", "0", "ms", "pr_interval"),
            ("QRS", "0", "ms", "qrs_duration"),
            ("QT/QTc", "0", "ms", "qtc_interval"),
        ]
        
        for title, value, unit, key in metric_info:
            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 10, QFont.Bold))
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val = QLabel(f"{value} {unit}")
            val.setFont(QFont("Arial", 14, QFont.Bold))
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            box.addWidget(lbl)
            box.addWidget(val)
            metrics_layout.addLayout(box)
            self.metric_labels[key] = val
        
        layout.addWidget(metrics_card)
        
        # Plot area - Grid layout for 7 leads (Lead II + V1-V6)
        plot_frame = QFrame()
        plot_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_frame.setStyleSheet("background: #f8f9fa; border-radius: 16px; padding: 12px; border: 2px solid #e9ecef;")
        plot_layout = QGridLayout(plot_frame)
        plot_layout.setSpacing(10)
        
        # Create plot widgets and curves for each lead
        self.plot_widgets = {}
        self.plot_curves = {}
        self.data_lines = {} # For consistency with 12-lead naming
        
        lead_names = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Define colors for each lead type for consistent color coding (matching 12-lead test)
        lead_colors = {
            'II': '#4ecdc4',     # Teal  
            'V1': '#54a0ff',     # Light Blue
            'V2': '#5f27cd',     # Purple
            'V3': '#00d2d3',     # Cyan
            'V4': '#ff9f43',     # Orange
            'V5': '#10ac84',     # Dark Green
            'V6': '#ee5a24'      # Dark Orange
        }
        
        # Arrange in 2 columns (II on top full width, then V1-V6 in 3 rows)
        positions = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
        
        # Consistent colors from 12-lead dashboard
        lead_colors = {
            'II': '#ff0055', 
            'V1': '#ffcc00',
            'V2': '#00ffcc',
            'V3': '#ff6600',
            'V4': '#6600ff',
            'V5': '#00b894',
            'V6': '#ff0066'
        }
        
        for i, lead_name in enumerate(lead_names):
            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.setMenuEnabled(False)
            plot_widget.showGrid(x=False, y=False)
            
            # Hide Y-axis labels for cleaner display (clinical standard)
            plot_widget.getAxis('left').setTicks([])
            plot_widget.getAxis('left').setLabel('')
            plot_widget.getAxis('bottom').setTextPen('k')
            plot_widget.getAxis('bottom').setPen('k')
            
            lead_color = lead_colors.get(lead_name, '#000000')
            plot_widget.setTitle(f"Lead {lead_name}", color=lead_color, size='11pt')
            
            # Set initial Y-range (matching 12-lead dashboard style)
            # Set initial Y-range (Raw ADC 0-4096)
            plot_widget.setYRange(0, 4096, padding=0)
            
            # Add center line at 2048
            center_line = pg.InfiniteLine(pos=2048, angle=0, pen=pg.mkPen(color='gray', width=0.5, style=Qt.DashLine))
            plot_widget.addItem(center_line)

            vb = plot_widget.getViewBox()
            if vb is not None:
                vb.setLimits(yMin=0, yMax=4096)
                try:
                    vb.setRange(xRange=(0.0, 3.0))
                except Exception:
                    pass
            
            # Create plot curve with clinical line width
            plot_curve = plot_widget.plot(pen=pg.mkPen(color=lead_color, width=1.0))
            
            self.plot_widgets[lead_name] = plot_widget
            self.plot_curves[lead_name] = plot_curve
            self.data_lines[lead_name] = plot_curve
            
            row, col = positions[i]
            if lead_name == 'II':
                plot_layout.addWidget(plot_widget, 0, 0, 1, 2) # Lead II spanning 2 columns
            else:
                plot_layout.addWidget(plot_widget, row, col)
        
        # Keep backward compatibility - also store Lead II as primary
        self.plot_widget = self.plot_widgets['II']
        self.plot_curve = self.plot_curves['II']
        
        layout.addWidget(plot_frame, stretch=1)
        
        # Info label
        info_label = QLabel("Capture 30 seconds of Lead II and V1-V6 data for hyperkalemia detection. The system will analyze T-waves, PR interval, QRS duration, and P-wave morphology according to ECG standards.")
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #666; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Store analysis results
        self.analysis_results = None

    def refresh_com_ports(self):
        """Refresh available COM ports"""
        pass
    
    def start_capture(self):
        """Start capturing Lead II data"""
        # CHECK: Ensure no other test is running
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            # Check if dashboard has the can_start_test method
            if hasattr(self.dashboard_instance, 'can_start_test'):
                if not self.dashboard_instance.can_start_test("hyperkalemia_test"):
                    return
                # Set state to running
                self.dashboard_instance.update_test_state("hyperkalemia_test", True)

        # Port detection and serial connection
        if not SERIAL_AVAILABLE or not ECG_TEST_AVAILABLE:
            QMessageBox.warning(self, "Serial Not Available", 
                              "Serial/ECG modules are not available. Please install pyserial and restart.")
            return
        
        # FLUSH stale smoothing state from any previous capture (including 12-lead).
        try:
            from ecg.ecg_calculations import cleanup_instance
            cleanup_instance('hyperkalemia_test')
        except Exception:
            pass
        if self.ecg_calculator:
            for attr in ('_pr_smooth_buffer_tl', '_qrs_smooth_buffer',
                         '_qt_smooth_buffer', '_p_smooth_buffer',
                         '_last_displayed_qrs', '_last_displayed_qt',
                         '_last_displayed_qtc', '_last_displayed_p',
                         '_pending_qrs_value', '_pending_qt_value',
                         '_pending_p_value'):
                if hasattr(self.ecg_calculator, attr):
                    v = getattr(self.ecg_calculator, attr)
                    if isinstance(v, list):
                        v.clear()
                    else:
                        setattr(self.ecg_calculator, attr, 0)
            self.ecg_calculator.pr_interval = 0
            self.ecg_calculator.last_qrs_duration = 0
            self.ecg_calculator.last_qt_interval = 0
            self.ecg_calculator.last_qtc_interval = 0

        # Get port from settings or auto-detect
        port_to_use = self.settings_manager.get_serial_port()
        baudrate = int(self.settings_manager.get_setting("baud_rate", "115200"))

        # Check if Demo Mode is active on the dashboard's ECGTestPage
        is_demo = False
        try:
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                ecg_page = getattr(self.dashboard_instance, 'ecg_test_page', None)
                if ecg_page and getattr(ecg_page, 'demo_toggle', None):
                    is_demo = ecg_page.demo_toggle.isChecked()
        except Exception:
            pass

        if is_demo:
            print(" Starting Hyperkalemia Test in Demo Mode...")
            from ecg.demo_serial_reader import DemoSerialReader
            self.serial_reader = DemoSerialReader(self)
        else:
            # Check if we already have an active reader in GlobalHardwareManager
            from ecg.serial.serial_reader import GlobalHardwareManager
            existing_reader = GlobalHardwareManager().reader
            if existing_reader and existing_reader.ser and existing_reader.ser.is_open:
                if not port_to_use or port_to_use == "Select Port":
                    port_to_use = existing_reader.ser.port
                    print(f" Using existing active serial port: {port_to_use}")
            
            # Check if port needs scanning (not set or not in available ports)
            scan_needed = (not port_to_use or port_to_use == "Select Port")
            
            if not scan_needed:
                try:
                    available_ports = [p.device for p in serial.tools.list_ports.comports()]
                    if port_to_use not in available_ports:
                        print(f" Configured port {port_to_use} not found in available ports. forcing scan.")
                        scan_needed = True
                except Exception:
                    pass
            
            if scan_needed:
                print(" No COM port configured or port not found – will auto‑scan all ports.")
                try:
                    scan_result = SerialStreamReader.scan_and_detect_port(baudrate=baudrate, timeout=0.2)
                    if scan_result:
                        detected_port, detected_serial = scan_result
                        port_to_use = detected_port
                        print(f" Auto‑detected ECG device on port {detected_port}")
                        try:
                            if detected_serial and detected_serial.is_open:
                                detected_serial.close()
                        except Exception as e:
                            print(f" Warning: Failed to close detected serial port: {e}")
                        if hasattr(self, 'settings_manager'):
                            self.settings_manager.set_setting("serial_port", detected_port)
                            self.settings_manager.save_settings()
                    else:
                        QMessageBox.warning(self, "No Device Found",
                                          "Could not auto-detect ECG device. Please check connection.")
                        return
                except Exception as scan_err:
                    print(f" Port scan failed: {scan_err}")
                    QMessageBox.warning(self, "Scan Failed", f"Port scan failed: {scan_err}")
                    return
            
            try:
                from ecg.serial.serial_reader import GlobalHardwareManager
                self.serial_reader = GlobalHardwareManager().get_reader(port_to_use, baudrate)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open serial port: {e}")
                return

        try:
            # Start/Resume acquisition. The start() method now handles
            # skipping hardware commands if already running.
            self.serial_reader.start()
            
            # Reset data for all leads
            self.data = np.zeros(10000, dtype=np.float32)  # Reset circular buffer
            self.lead_ii_data = []
            for lead_name in self.lead_data.keys():
                self.lead_data[lead_name] = []
            # Reset adaptive scaling
            self.y_centers = {lead: 0.0 for lead in self.lead_data.keys()}
            self.y_ranges = {lead: 200.0 for lead in self.lead_data.keys()}
            self.y_center = 0.0
            self.y_range = 200.0
            self.sample_index = 0
            self.active_samples = 0
            self.analysis_results = None
            self.start_time = time.time()
            self.is_capturing = True
            
            # Reset smoothing buffers
            if self.ecg_calculator:
                if hasattr(self.ecg_calculator, 'smoothing_buffers'):
                    self.ecg_calculator.smoothing_buffers = {}
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Capturing from serial port...")
            self.status_label.setStyleSheet("color: #28a745; padding: 5px;")
            self._silent_data_warned = False
            
            # ── Start HolterBPMController ───────────────────────────────────────
            try:
                if self._bpm_ctrl is not None:
                    if self._bpm_ctrl.is_running:
                        self._bpm_ctrl.stop()
                    self._bpm_ctrl.start(target_hours=0)
                    main_layout = self.layout()
                    if main_layout and self._bpm_ctrl.display_bar is not None:
                        existing = [main_layout.itemAt(i).widget()
                                    for i in range(main_layout.count())
                                    if main_layout.itemAt(i).widget() is not None]
                        if self._bpm_ctrl.display_bar not in existing:
                            main_layout.insertWidget(0, self._bpm_ctrl.display_bar)
                        self._bpm_ctrl.display_bar.show()
                        
                        # Start 3-second BPM UI refresh timer
                        if not hasattr(self, '_bpm_refresh_timer'):
                            self._bpm_refresh_timer = QTimer()
                            self._bpm_refresh_timer.timeout.connect(self._refresh_holter_bpm_label)
                        if not self._bpm_refresh_timer.isActive():
                            self._bpm_refresh_timer.start(2000)
            except Exception as _bpm_err:
                print(f"[HyperkalemiaTestWindow] BPM controller start error: {_bpm_err}")
            
            # Start timers
            self.capture_timer.start(50)  # Update plot every 50ms
            self.duration_timer.start(1000)  # Check duration every second
            self.metrics_timer = QTimer(self)
            self.metrics_timer.timeout.connect(self.update_metrics)
            self.metrics_timer.start(200)
            
            # No success message needed - status label already shows the state
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to start capture: {str(e)}")
            self.crash_logger.log_error(
                message=f"Hyperkalemia test capture start error: {e}",
                exception=e,
                category="HYPERKALEMIA_TEST_ERROR"
            )
    
    def stop_capture(self, device_disconnected=False, device_not_sending=False):
        """Stop capturing data"""
        # UPDATE STATE: Test stopped
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if hasattr(self.dashboard_instance, 'update_test_state'):
                self.dashboard_instance.update_test_state("hyperkalemia_test", False)

        self.is_capturing = False
        
        # Serial reader cleanup
        if self.serial_reader:
            try:
                # IMPORTANT: We don't close the shared reader here because other tests 
                # might still be using it in the background.
                pass
            except:
                pass
            self.serial_reader = None
        
        # ── Stop HolterBPMController ────────────────────────────────────────
        try:
            if hasattr(self, '_bpm_refresh_timer') and self._bpm_refresh_timer.isActive():
                self._bpm_refresh_timer.stop()
            if self._bpm_ctrl is not None and self._bpm_ctrl.is_running:
                self._bpm_ctrl.stop()
                if self._bpm_ctrl.display_bar is not None:
                    self._bpm_ctrl.display_bar.hide()
        except Exception as _bpm_err:
            print(f"[HyperkalemiaTestWindow] BPM controller stop error: {_bpm_err}")
        
        # Stop timers
        self.capture_timer.stop()
        self.duration_timer.stop()
        if hasattr(self, 'metrics_timer'):
            self.metrics_timer.stop()
        
        # Update UI based on reason
        if device_disconnected:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Device disconnected")
        elif device_not_sending:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Device connected but not sending data")
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            if len(self.lead_ii_data) > 0:
                self.analyze_btn.setEnabled(True)
                self.report_btn.setEnabled(True)
                self.status_label.setText(f"Status: Capture Complete")
            else:
                self.status_label.setText("Status: Capture Stopped (No data)")
        
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        self.timer_label.setText("Time: 00:00")

    def _refresh_holter_bpm_label(self):
        """Called every 3 s by _bpm_refresh_timer.
        Reads the stable 30-second-window BPM from HolterBPMController and
        writes it to the heart_rate metric label.
        Also stores the value in self.last_heart_rate so reports match the display.
        """
        try:
            if self._bpm_ctrl is None or not self._bpm_ctrl.is_running:
                return
            bpm = self._bpm_ctrl.current_bpm()
            if bpm > 0:
                if hasattr(self, 'metric_labels') and 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText(f"{int(round(bpm))} BPM")
                # Always persist so report generation uses the stable Holter BPM
                self.last_heart_rate = int(round(bpm))
        except Exception as _e:
            print(f"[HyperkalemiaTestWindow] _refresh_holter_bpm_label error: {_e}")

    def confirm_stop(self):
        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            "Are you sure you want to stop?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                self.stop_capture()
            finally:
                if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                    try:
                        self.dashboard_instance.raise_()
                        self.dashboard_instance.activateWindow()
                    except Exception:
                        pass
                self.close()
    
    def check_duration(self):
        """Check if 30 seconds have elapsed"""
        if not self.is_capturing:
            return
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.capture_duration - elapsed)
        
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        self.timer_label.setText(f"Time: {minutes:02d}:{seconds:02d}")
        
        if elapsed >= self.capture_duration:
            self.stop_capture()
            QMessageBox.information(self, "Capture Complete", 
                                  "30-second capture completed successfully!")
    
    def update_plot(self):
        """Update all plots with new data matching 12-lead dashboard style"""
        if not self.is_capturing or not self.serial_reader:
            return

        # Check if device got disconnected suddenly
        if not self.serial_reader.running:
            print("⚠️ Device disconnected during Hyperkalemia test!")
            self.stop_capture(device_disconnected=True)
            return
            
        
        try:
            elapsed = time.time() - self.start_time
            
            # Read packets from serial reader
            packets = self.serial_reader.read_packets(max_packets=100)
            if not packets and hasattr(self.serial_reader, 'is_device_silent') and self.serial_reader.is_device_silent(3.0):
                if not getattr(self, '_silent_data_warned', False):
                    QMessageBox.warning(
                        self,
                        "Device Not Sending Data",
                        "Device is connected but not sending ECG packets.\n\n"
                        "Please check electrodes/cable and ensure device streaming is ON."
                    )
                    self._silent_data_warned = True
                self.stop_capture(device_not_sending=True)
                return

            if not packets:
                return
            
            # Process each packet
            for packet in packets:
                self.active_samples = min(len(self.data), self.active_samples + 1)

                # ── Feed packet to HolterBPMController ──────────────────────────
                try:
                    if self._bpm_ctrl is not None:
                        self._bpm_ctrl.push(packet)
                except Exception:
                    pass

                self.sample_index += 1
                packet_time = self.sample_index / 500.0
                
                # Update calculator's data buffer for all leads from serial data
                if self.ecg_calculator:
                    try:
                        for lead_name, idx in self.lead_indices.items():
                            if lead_name in packet:
                                raw_value = float(packet[lead_name])
                                
                                # 1. ANALYSIS DATA (RAW)
                                self.ecg_calculator.data[idx] = np.roll(self.ecg_calculator.data[idx], -1)
                                self.ecg_calculator.data[idx][-1] = raw_value
                                
                                # Store smoothed data for plotting
                                if lead_name in self.lead_data:
                                    self.lead_data[lead_name].append({
                                        'time': packet_time,
                                        'value': raw_value
                                    })
                                
                                # Primary Lead II backup
                                if lead_name == 'II':
                                    self.data = np.roll(self.data, -1)
                                    self.data[-1] = raw_value
                                    self.lead_ii_data.append({'time': packet_time, 'value': raw_value})
                        
                    except Exception as e:
                        print(f" Error updating buffers: {e}")
                        continue
            
            # Update sampling rate counter
            if self.ecg_calculator and hasattr(self.ecg_calculator, "sampler"):
                sr = 0.0
                try:
                    for _ in range(len(packets)):
                        sr = self.ecg_calculator.sampler.add_sample()
                except Exception:
                    sr = self.ecg_calculator.sampler.add_sample()
                if sr > 0:
                    safe_sr = float(sr)
                    if safe_sr < 100.0 or safe_sr > 1000.0:
                        safe_sr = 500.0
                    self.sampling_rate = safe_sr

            # Get filter settings from SettingsManager
            ac_val = self.settings_manager.get_setting("filter_ac", "50")
            emg_val = self.settings_manager.get_setting("filter_emg", "35")
            fs = self.sampling_rate if self.sampling_rate > 0 else 500.0
            if fs < 100.0 or fs > 1000.0:
                fs = 500.0
            
            # Update all plots with stable display window
            for lead_name in self.lead_data.keys():
                if len(self.lead_data[lead_name]) > 0:
                    # 25 mm/s → ~3 s window (wave speed logic disabled for Hyperkalemia test)
                    seconds_to_show = 3.0

                    # Lead II: use a larger window for wave movement visibility
                    if lead_name == 'II':
                        seconds_to_show = 6.0


                    # try:
                    #     wave_speed = float(self.settings_manager.get_wave_speed())
                    #     seconds_to_show = 3.0 * (25.0 / max(1e-6, wave_speed))
                    # except:
                    #     pass

                    times = [d['time'] for d in self.lead_data[lead_name]]
                    values = [d['value'] for d in self.lead_data[lead_name]]

                    # Apply Filters
                    try:
                        from ecg.ecg_filters import apply_ac_filter, apply_emg_filter
                        if len(values) > 100: # Ensure enough data
                            values_array = np.array(values)
                            if ac_val not in ["Off", "off"]:
                                values_array = apply_ac_filter(values_array, fs, ac_val)
                            if emg_val not in ["Off", "off"]:
                                values_array = apply_emg_filter(values_array, fs, emg_val)

                            # edge_trim = int(0.5 * fs)
                            # if len(values_array) > 2 * edge_trim:
                            #     values_array = values_array[edge_trim:-edge_trim]
                                
                            values = values_array.tolist()
                    except ImportError:
                        pass
                    
                    if len(times) > 0:
                        max_time = max(times)
                        min_time = max(0, max_time - seconds_to_show)
                        mask = [t >= min_time for t in times]
                        display_times = [t for i, t in enumerate(times) if mask[i]]
                        display_values = [v for i, v in enumerate(values) if mask[i]]

                        # --- Add Gaussian smoothing here ---
                        if len(display_values) > 5:
                            gaussian_sigma = 2  # You can adjust this value for more/less smoothing
                            display_values = gaussian_filter1d(display_values, sigma=gaussian_sigma)
                        
                        if len(display_times) > 0:
                            
                            # Update plot line
                            self.plot_curves[lead_name].setData(display_times, display_values)
                            self.plot_widgets[lead_name].setXRange(min(display_times), max(display_times), padding=0)
                            
                            # Fixed Y-axis scaling (0-4096)
                            self.plot_widgets[lead_name].setYRange(0, 4096, padding=0)
        
        except Exception as e:
            pass

    def update_metrics(self):
        """Calculate and update ECG metrics using same stable methods as 12-lead dashboard"""
        if not self.is_capturing:
            return
        if self.active_samples < max(200, int((self.sampling_rate or 500.0) * 0.5)):
            return
        
        try:
            # Sync sampling rate
            current_fs = self.sampling_rate if self.sampling_rate > 0 else 500.0
            if current_fs < 100.0 or current_fs > 1000.0:
                current_fs = 500.0
            if self.ecg_calculator and self.ecg_calculator.sampler:
                self.ecg_calculator.sampler.sampling_rate = current_fs
                self.ecg_calculator.sampling_rate = current_fs

                # TRIGGER STABLE MEDIAN-BEAT ANALYSIS
                # KEY FIX: Sync the headless ecg_calculator with the current Holter BPM
                # so calculate_ecg_metrics() uses the right rr_ms when computing QTc.
                _bpm_active = (self._bpm_ctrl is not None and self._bpm_ctrl.is_running)
                _current_bpm = 0
                if _bpm_active:
                    try:
                        _current_bpm = self._bpm_ctrl.current_bpm()
                        if _current_bpm > 0:
                            self.ecg_calculator.last_heart_rate = int(round(_current_bpm))
                    except Exception:
                        pass

                try:
                    original_buffers = {}
                    for idx in self.lead_indices.values():
                        original_buffers[idx] = self.ecg_calculator.data[idx]
                        if self.active_samples < len(original_buffers[idx]):
                            self.ecg_calculator.data[idx] = original_buffers[idx][-self.active_samples:]
                    
                    self.ecg_calculator.calculate_ecg_metrics()
                    
                    for idx, original in original_buffers.items():
                        self.ecg_calculator.data[idx] = original
                except Exception as e:
                    print(f" calculate_ecg_metrics error in Hyperkalemia test: {e}")

                # FETCH METRICS DIRECTLY from the calculator's stored attributes.
                def _attr_to_str(attr_name, fallback='0'):
                    v = getattr(self.ecg_calculator, attr_name, 0)
                    try:
                        iv = int(round(float(v))) if v else 0
                        return str(iv) if iv > 0 else fallback
                    except:
                        return fallback

                # If Holter BPM is active, recompute QTc from waveform QT + stable BPM.
                # This ensures QTc updates whenever BPM changes, even if QT waveform hasn't changed.
                if _bpm_active and _current_bpm > 0:
                    qt_raw = getattr(self.ecg_calculator, 'last_qt_interval', 0)
                    if qt_raw > 0:
                        import math
                        rr_s = 60.0 / _current_bpm
                        qtc_recomputed = int(round((qt_raw / 1000.0) / math.sqrt(rr_s) * 1000.0))
                        self.ecg_calculator.last_qtc_interval = qtc_recomputed

                # Also call get_current_metrics as a secondary/fallback source
                metrics = self.ecg_calculator.get_current_metrics()
                self.last_metrics = dict(metrics) if isinstance(metrics, dict) else {}

                # Update UI labels
                hr_val  = metrics.get('heart_rate', '0')
                pr_val  = _attr_to_str('pr_interval') or metrics.get('pr_interval', '0')
                qrs_val = _attr_to_str('last_qrs_duration') or metrics.get('qrs_duration', '0')
                qt_val  = _attr_to_str('last_qt_interval') or metrics.get('qt_interval', '0')
                qtc_val = _attr_to_str('last_qtc_interval') or metrics.get('qtc_interval', '0')

                # Also persist to last_metrics so report generation can use them
                if pr_val != '0':  self.last_metrics['pr_interval'] = pr_val
                if qrs_val != '0': self.last_metrics['qrs_duration'] = qrs_val
                if qt_val != '0':  self.last_metrics['qt_interval'] = qt_val
                if qtc_val != '0': self.last_metrics['qtc_interval'] = qtc_val

                print(f"Heart Rate: {hr_val} BPM, PR Interval: {pr_val} ms, QRS Duration: {qrs_val} ms, QTC Interval: {qtc_val} ms")

                if not _bpm_active and 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText(f"{hr_val} BPM" if hr_val != '0' else "00 BPM")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText(f"{pr_val} ms" if pr_val != '0' else "0 ms")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText(f"{qrs_val} ms" if qrs_val != '0' else "0 ms")
                if 'qtc_interval' in self.metric_labels:
                    if qt_val not in ('', '0') and qtc_val not in ('', '0'):
                        self.metric_labels['qtc_interval'].setText(f"{qt_val}/{qtc_val} ms")
                    else:
                        self.metric_labels['qtc_interval'].setText(f"{qtc_val} ms" if qtc_val not in ('', '0') else "0 ms")


        
        except Exception as e:
            pass

    def _compute_wave_amplitudes(self):
        p_amp = 0.0
        qrs_amp = 0.0
        t_amp = 0.0
        try:
            fs = float(self.sampling_rate) if getattr(self, "sampling_rate", 0) else 500.0
            if not self.ecg_calculator or not hasattr(self.ecg_calculator, "data") or len(self.ecg_calculator.data) <= 1:
                return p_amp, qrs_amp, t_amp
            lead_ii = self.ecg_calculator.data[1]
            if isinstance(lead_ii, (list, tuple)):
                lead_ii = np.asarray(lead_ii, dtype=np.float32)
            arr = lead_ii
            if self.active_samples > 0 and self.active_samples < len(arr):
                arr = arr[-self.active_samples:]
            max_len = int(10 * fs)
            if len(arr) > max_len:
                arr = arr[-max_len:]
            if arr is None or len(arr) < int(2 * fs) or np.std(arr) < 0.1:
                return p_amp, qrs_amp, t_amp
            nyq = fs / 2.0
            b, a = butter(2, [max(0.5 / nyq, 0.001), min(40.0 / nyq, 0.99)], btype="band")
            x = filtfilt(b, a, arr)
            squared = np.square(np.diff(x))
            win = max(1, int(0.15 * fs))
            env = np.convolve(squared, np.ones(win) / win, mode="same")
            thr = np.mean(env) + 0.5 * np.std(env)
            r_peaks, _ = find_peaks(env, height=thr, distance=int(0.6 * fs))
            if len(r_peaks) < 3:
                return p_amp, qrs_amp, t_amp
            p_vals = []
            qrs_vals = []
            t_vals = []
            for r in r_peaks[1:-1]:
                p_start = max(0, r - int(0.20 * fs))
                p_end = max(0, r - int(0.12 * fs))
                if p_end > p_start:
                    seg = x[p_start:p_end]
                    base = np.mean(x[max(0, p_start - int(0.05 * fs)):p_start])
                    p_vals.append(np.max(seg) - base)
                qrs_start = max(0, r - int(0.08 * fs))
                qrs_end = min(len(x), r + int(0.08 * fs))
                if qrs_end > qrs_start:
                    seg = x[qrs_start:qrs_end]
                    qrs_vals.append(np.max(seg) - np.min(seg))
                t_start = min(len(x), r + int(0.10 * fs))
                t_end = min(len(x), r + int(0.30 * fs))
                if t_end > t_start:
                    seg = x[t_start:t_end]
                    base = np.mean(x[r:t_start]) if t_start > r else 0.0
                    t_vals.append(np.max(seg) - base)
            def med(v):
                return float(np.median(v)) if len(v) > 0 else 0.0
            p_amp = med(p_vals)
            qrs_amp = med(qrs_vals)
            t_amp = med(t_vals)
        except Exception:
            pass
        return p_amp, qrs_amp, t_amp

    def analyze_hyperkalemia(self, enable=False):
        """Analyze captured ECG data for hyperkalemia indicators using clinical standards"""
        if self.active_samples < 500:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Please capture more data before analysis (at least 10 seconds).")
            return
            
        try:
            self.status_label.setText("Status: Analyzing ECG Morphology...")
            self.status_label.setStyleSheet("color: #007bff; font-weight: bold;")
            
            # 1. TRIGGER FULL CLINICAL ANALYSIS
            # Sync buffers to active portion
            original_buffers = {}
            for idx in self.lead_indices.values():
                original_buffers[idx] = self.ecg_calculator.data[idx]
                if self.active_samples < len(original_buffers[idx]):
                    self.ecg_calculator.data[idx] = original_buffers[idx][-self.active_samples:]
            
            # Force calculation
            self.ecg_calculator.calculate_ecg_metrics()
            
            # Restore buffers
            for idx, original in original_buffers.items():
                self.ecg_calculator.data[idx] = original
                
            # Get latest clinical metrics (last metrics stored)
            metrics = self.last_metrics if getattr(self, "last_metrics", None) else self.ecg_calculator.get_current_metrics()
            
            # 2. EXTRACT MEASUREMENTS
            def safe_float(val):
                try:
                    # Strip common units before conversion
                    if isinstance(val, str):
                        val = val.replace("ms", "").replace("MS", "").strip()
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            hr = safe_float(metrics.get('heart_rate', 0))
            pr = safe_float(metrics.get('pr_interval', 0))
            qrs = safe_float(metrics.get('qrs_duration', 0))
            qt = 0.0
            qtc = 0.0

            qtqtc_text = None
            # Prefer the Hyperkalemia dashboard label if present
            if hasattr(self, 'metric_labels') and 'qtc_interval' in self.metric_labels:
                try:
                    qtqtc_text = self.metric_labels['qtc_interval'].text().strip()
                except Exception:
                    qtqtc_text = None

            # Fallback to metrics dict if needed
            if (not qtqtc_text) and metrics.get('qtc_interval') is not None:
                qtqtc_text = str(metrics.get('qtc_interval')).strip()

            if qtqtc_text:
                clean = qtqtc_text.replace("ms", "").replace("MS", "").strip()
                if "/" in clean:
                    parts = [p.strip() for p in clean.split("/") if p.strip()]
                    if len(parts) >= 1:
                        qt = safe_float(parts[0])
                    if len(parts) >= 2:
                        qtc = safe_float(parts[1])
                else:
                    qtc = safe_float(clean)

            # Final fallback: use calculator's last clinical values if parsing failed
            if qt <= 0 and hasattr(self.ecg_calculator, 'last_qt_interval'):
                try:
                    qt = float(getattr(self.ecg_calculator, 'last_qt_interval') or 0)
                except Exception:
                    pass
            if qtc <= 0 and hasattr(self.ecg_calculator, 'last_qtc_interval'):
                try:
                    qtc = float(getattr(self.ecg_calculator, 'last_qtc_interval') or 0)
                except Exception:
                    pass

            p_amp, qrs_amp, t_amp = self._compute_wave_amplitudes()
            
            # 3. HYPERKALEMIA MORPHOLOGY LOGIC (GE/Philips standards)
            indicators = []
            risk_score = 0
            
            # Indicator 1: PR Interval Prolongation
            if pr > 200:
                indicators.append(f"Prolonged PR Interval ({pr}ms)")
                risk_score += 1
                if pr > 240:
                    risk_score += 1
                
            # Indicator 2: QRS Widening
            if qrs > 110:
                indicators.append(f"Widened QRS Complex ({qrs}ms)")
                risk_score += 1
                if qrs > 120:
                    risk_score += 2
                
            # Indicator 3: Peaked T-waves (Estimated from amplitude variation)
            # We check precordial leads V2-V4 for maximum amplitude
            max_t_amp = 0
            try:
                # Use Lead V2/V3 for peaked T-wave detection if available
                test_leads = [self.lead_indices.get('V2'), self.lead_indices.get('V3')]
                for l_idx in test_leads:
                    if l_idx is not None:
                        sig = self.ecg_calculator.data[l_idx]
                        if self.active_samples < len(sig):
                            sig = sig[-self.active_samples:]
                        amp = np.percentile(sig, 99) - np.percentile(sig, 1)
                        max_t_amp = max(max_t_amp, amp)
                
                if max_t_amp > 800:
                    indicators.append("Tall/Peaked T-waves detected (precordial leads)")
                    risk_score += 2
            except Exception:
                pass

            if qrs_amp > 0 and t_amp > 0 and (2.0 * t_amp) > qrs_amp:
                indicators.append("T-wave amplitude exceeds R-wave amplitude (Lead II)")
                risk_score += 2

            if qrs_amp > 0:
                if p_amp <= 0 or p_amp < 0.1 * qrs_amp:
                    if p_amp <= 0:
                        indicators.append("P-waves absent or extremely low amplitude (flattening)")
                    else:
                        indicators.append("P-wave flattening relative to QRS amplitude")
                    risk_score += 1

            sine_wave = False
            if qrs >= 160 and qrs_amp > 0 and t_amp > 0:
                if p_amp <= 0 or p_amp < 0.05 * qrs_amp:
                    sine_wave = True
            if sine_wave:
                indicators.append("Sine-wave morphology (very wide QRS with merged T-wave)")
                risk_score += 3

            # 4. DETERMINE RISK LEVEL
            risk_level = "Normal/Low"
            risk_color = "#28a745" # Green
            
            if risk_score >= 4:
                risk_level = "High"
                risk_color = "#dc3545" # Red
            elif risk_score >= 2:
                risk_level = "Moderate"
                risk_color = "#ffc107" # Yellow
            elif risk_score >= 1:
                risk_level = "Mild"
                risk_color = "#17a2b8" # Cyan
            
            # Store results for report generator
            self.analysis_results = {
                "heart_rate": hr,
                "pr_interval_ms": pr,
                "qrs_duration_ms": qrs,
                "qt_interval_ms": qt,
                "qtc_ms": qtc,
                "st_segment_ms": 0,
                "indicators": indicators,
                "risk_level": risk_level,
                "risk_score": risk_score
            }
                
            self.status_label.setText(f"Status: Analysis Complete ")
            self.status_label.setStyleSheet(f"color: {risk_color}; font-weight: bold;")
            self.report_btn.setEnabled(True)
            
            if not enable:
                msg = f"<b>Hyperkalemia Analysis Results</b><br><br>"
                msg += f"Risk Level: <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span><br><br>"
                msg += f"Heart Rate: {hr} BPM<br>"
                msg += f"PR Interval: {pr} ms<br>"
                msg += f"QRS Duration: {qrs} ms<br>"
                msg += f"QT Interval: {qt} ms<br>"
                msg += f"QTc Interval: {qtc} ms<br><br>"
                if indicators:
                    msg += "<b>Indicators:</b><br>" + "<br>".join(["- " + i for i in indicators])
                
                QMessageBox.information(self, "Hyperkalemia Analysis", msg)
            
        except Exception as e:
            self.crash_logger.log_crash("analyze_hyperkalemia", e)
            QMessageBox.critical(self, "Analysis Error", f"Failed to complete morphological analysis: {e}")
            self.status_label.setText("Status: Analysis Failed")

    def generate_report(self):
        """Generate hyperkalemia detection report PDF"""
        # If analysis has not been run yet but enough data is present, perform a silent analysis
        if self.analysis_results is None and len(self.lead_ii_data) > 0 and self.active_samples >= 500:
            try:
                self.analyze_hyperkalemia(enable=True)
            except Exception:
                pass

        # If analysis is still unavailable, fall back to the original warning
        if self.analysis_results is None:
            QMessageBox.warning(self, "No Analysis", 
                              "Please analyze the ECG data first.")
            return
        
        if len(self.lead_ii_data) == 0:
            QMessageBox.warning(self, "No Data", 
                              "No data available to generate report.")
            return
        
        # Get save location
        from PyQt5.QtWidgets import QFileDialog
        default_filename = f"Hyperkalemia_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Hyperkalemia Report", default_filename, "PDF Files (*.pdf)"
        )
        
        if not filepath:
            return
        
        
        try:
            # Load latest patient details (same strategy as HRV/ECG flows)
            patient = {}
            try:
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                base_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
                all_patients_file = os.path.join(base_dir, 'all_patients.json')
                if os.path.exists(all_patients_file):
                    with open(all_patients_file, 'r') as f:
                        all_patients_data = json.load(f)
                    if isinstance(all_patients_data, dict) and 'patients' in all_patients_data:
                        patients_list = all_patients_data['patients']
                        if patients_list:
                            patient = patients_list[-1].copy()
                if not patient:
                    last_patient_file = os.path.join(base_dir, 'last_patient_details.json')
                    if os.path.exists(last_patient_file):
                        with open(last_patient_file, 'r') as f:
                            patient = json.load(f)
                if not patient or "first_name" not in patient:
                    patient = {
                        "first_name": "Patient",
                        "last_name": "Hyperkalemia",
                        "age": "",
                        "gender": "",
                        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Org.": "",
                        "doctor_mobile": "",
                        "doctor": "",
                    }
                else:
                    patient["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if "doctor_mobile" not in patient:
                    patient["doctor_mobile"] = ""
            except Exception as e:
                print(f" Could not load patient details for hyperkalemia report: {e}")
                patient = {
                    "first_name": "Patient",
                    "last_name": "Hyperkalemia",
                    "age": "",
                    "gender": "",
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Org.": "",
                    "doctor_mobile": "",
                    "doctor": "",
                }

            # Attach patient into analysis_results so the generator can render it
            if isinstance(self.analysis_results, dict):
                self.analysis_results["patient"] = patient

            # Save ECG data (including V1-V6) to file so report generator can load it
            ecg_data_file = None
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                ecg_data_dir = os.path.join(base_dir, 'reports', 'ecg_data')
                os.makedirs(ecg_data_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                ecg_data_file = os.path.join(ecg_data_dir, f'ecg_data_{timestamp}.json')
                
                saved_data = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "sampling_rate": float(self.sampling_rate),
                    "leads": {}
                }
                
                # Save Lead II data
                if self.lead_ii_data and len(self.lead_ii_data) > 0:
                    lead_ii_values = [d['value'] for d in self.lead_ii_data]
                    saved_data["leads"]["II"] = lead_ii_values
                    print(f" Saving Lead II: {len(lead_ii_values)} samples")
                else:
                    print(f" Lead II data is empty!")
                
                # Save V1-V6 data from self.lead_data
                for lead_name in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
                    if lead_name in self.lead_data:
                        if len(self.lead_data[lead_name]) > 0:
                            lead_values = [d['value'] for d in self.lead_data[lead_name]]
                            saved_data["leads"][lead_name] = lead_values
                            print(f" Saving {lead_name}: {len(lead_values)} samples")
                
                # Also save from ecg_calculator.data if available (as fallback)
                if self.ecg_calculator and hasattr(self.ecg_calculator, 'data'):
                    for lead_name, idx in self.lead_indices.items():
                        if idx < len(self.ecg_calculator.data):
                            ecg_data = self.ecg_calculator.data[idx]
                            if isinstance(ecg_data, np.ndarray) and len(ecg_data) > 0:
                                # Get non-zero values
                                non_zero_data = ecg_data[ecg_data != 0]
                                if len(non_zero_data) > 0:
                                    if lead_name not in saved_data["leads"] or len(saved_data["leads"][lead_name]) == 0:
                                        saved_data["leads"][lead_name] = non_zero_data.tolist()
                
                # Save to file
                with open(ecg_data_file, 'w') as f:
                    json.dump(saved_data, f, indent=2)
                print(f"\n Saved ECG data (including V1-V6) to: {ecg_data_file}")
                
                # Store file path for passing to report generator
                self.last_saved_ecg_file = ecg_data_file
                
            except Exception as e:
                error_msg = f" Could not save ECG data file: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()

            # Pass ecg_data_file to report generator
            print(f"\n Starting report generation...")
            print(f"   PDF filepath: {filepath}")
            print(f"   ECG data file: {ecg_data_file}")
            
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                reports_dir = os.path.join(base_dir, 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                hyper_metrics_path = os.path.join(reports_dir, 'hyper_metric.json')

                metrics_source = self.analysis_results if isinstance(self.analysis_results, dict) else {}

                def _safe_int(val):
                    try:
                        return int(round(float(val)))
                    except Exception:
                        return 0

                # Prefer the stable Holter BPM (what the display shows) over the
                # short-window analysis_results value.
                if hasattr(self, 'last_heart_rate') and self.last_heart_rate > 0:
                    hr = int(self.last_heart_rate)
                else:
                    hr = _safe_int(metrics_source.get("heart_rate", 0))
                pr = _safe_int(metrics_source.get("pr_interval_ms", 0))
                qrs = _safe_int(metrics_source.get("qrs_duration_ms", 0))
                qt = _safe_int(metrics_source.get("qt_interval_ms", 0))
                qtc = _safe_int(metrics_source.get("qtc_ms", 0))

                hyper_entry = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "file": os.path.abspath(filepath),
                    "HR_bpm": hr,
                    "PR_ms": pr,
                    "QRS_ms": qrs,
                    "QT_ms": qt,
                    "QTc_ms": qtc,
                }

                hyper_list = []
                if os.path.exists(hyper_metrics_path):
                    try:
                        with open(hyper_metrics_path, 'r') as f:
                            existing = json.load(f)
                            if isinstance(existing, list):
                                hyper_list = existing
                    except Exception:
                        hyper_list = []

                hyper_list.append(hyper_entry)

                with open(hyper_metrics_path, 'w') as f:
                    json.dump(hyper_list, f, indent=2)
                print(f" Saved Hyperkalemia metrics to {hyper_metrics_path}")
            except Exception as e:
                print(f" Could not save Hyperkalemia metrics: {e}")
            
            generate_hyperkalemia_report(filepath, self.analysis_results, self.lead_ii_data, 
                                        ecg_data_file=ecg_data_file)
            
            print(f"\n Report generation completed!")
            
            QMessageBox.information(self, "Report Generated", 
                                  f"Hyperkalemia detection report saved successfully:\n{filepath}")
            try:
                append_history_entry(patient, filepath, report_type="Hyperkalemia", username=self.username)
            except Exception as hist_err:
                print(f" Failed to append Hyperkalemia history: {hist_err}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to generate report: {str(e)}")
            self.crash_logger.log_error(
                message=f"Hyperkalemia report generation error: {e}",
                exception=e,
                category="HYPERKALemia_REPORT_ERROR"
            )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.is_capturing:
            reply = QMessageBox.question(
                self, "Capture in Progress",
                "Capture is still in progress. Do you want to stop and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.stop_capture()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
