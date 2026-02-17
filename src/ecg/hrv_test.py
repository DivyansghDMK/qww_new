

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox,
    QSizePolicy, QFrame
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import find_peaks

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - HRV test hardware features disabled")
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
from ecg.ecg_filters import apply_ac_filter, apply_emg_filter
from dashboard.history_window import append_history_entry

# Import ECGTestPage + helpers to reuse EXACT same calculation + smoothing as 12‑lead test
try:
    from ecg.twelve_lead_test import ECGTestPage, SamplingRateCalculator, SerialStreamReader
    from PyQt5.QtWidgets import QStackedWidget
    ECG_TEST_AVAILABLE = True
except ImportError:
    ECG_TEST_AVAILABLE = False
    print(" ECGTestPage not available - using fallback calculations")


class HRVTestWindow(QWidget):
    """HRV Test Window - 5-minute Lead II capture and report generation"""
    
    def __init__(self, parent=None, username=None):
        super().__init__(parent)
        self.dashboard_instance = parent  # Store reference to dashboard
        self.username = username
        self.setWindowTitle("HRV Test - Lead II")
        self.setMinimumSize(1200, 700)
        self.setGeometry(100, 100, 1200, 700)
        # Set window flags to make it a separate window
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Data storage - use circular buffer like 12-lead test
        HISTORY_LENGTH = 10000
        self.data = np.zeros(HISTORY_LENGTH, dtype=np.float32)  # Circular buffer for selected lead
        self.captured_data = []  # Store all captured data with timestamps
        self.start_time = None
        self.capture_duration = 5 * 60  # 5 minutes in seconds
        self.is_capturing = False
        self.serial_reader = None
        self.crash_logger = get_crash_logger()
        
        # For adaptive scaling (simple, stable Y-axis for Lead II)
        self.y_center = 0.0
        self.y_range = 500.0  # Initial range
        self.sampling_rate = 500.0  # Default sampling rate, will be estimated
        self.sample_index = 0  # For synthetic time axis if needed
        
        # Settings
        self.settings_manager = SettingsManager()

        # Selected lead for display (Lead II)
        self.selected_lead = "II"
        
        # Track active sample count to avoid skewing stats with leading zeros
        self.active_samples = 0
        
        # Create a minimal ECGTestPage instance to reuse its calculation methods
        # This ensures we use the EXACT same functions as the 12-lead test
        self.ecg_calculator = None
        if ECG_TEST_AVAILABLE:
            try:
                # Create a dummy stacked widget for ECGTestPage initialization
                dummy_stack = QStackedWidget()
                self.ecg_calculator = ECGTestPage("12 Lead ECG Test", dummy_stack)
                
                # IMPORTANT: Sync sampling rate from parent dashboard if available
                # This ensures both windows use identical frequency assumptions
                if parent and hasattr(parent, 'ecg_test_page'):
                    p_page = parent.ecg_test_page
                    if hasattr(p_page, 'sampler') and p_page.sampler.sampling_rate > 0:
                        self.sampling_rate = p_page.sampler.sampling_rate
                        print(f" Synced sampling rate from dashboard: {self.sampling_rate} Hz")
                
                # Set up minimal data structure (we only need Lead II, index 1)
                # ECGTestPage already initializes data, but we ensure it's set up
                if not hasattr(self.ecg_calculator, 'data') or len(self.ecg_calculator.data) < 12:
                    self.ecg_calculator.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
                
                # Ensure sampler exists with proper sampling rate
                if not hasattr(self.ecg_calculator, 'sampler'):
                    self.ecg_calculator.sampler = SamplingRateCalculator()
                self.ecg_calculator.sampler.sampling_rate = self.sampling_rate
                
                print(" ECG calculator initialized for HRV test")
            except Exception as e:
                print(f" Could not create ECG calculator: {e}")
                import traceback
                traceback.print_exc()
                self.ecg_calculator = None
        
        # Initialize UI
        self.init_ui()
        
        # Timers
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.update_plot)
        self.duration_timer = QTimer(self)
        self.duration_timer.timeout.connect(self.check_duration)
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QHBoxLayout()
        self.title_label = QLabel("HRV Test - Lead II")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #ff6600;")
        header.addWidget(self.title_label)
        header.addStretch()
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        header.addWidget(self.status_label)
        
        # Timer label
        self.timer_label = QLabel("Time: 00:00")
        self.timer_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.timer_label.setStyleSheet("color: #ff6600; padding: 5px;")
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

        # Lead Selection
        lead_label = QLabel("Select Lead:")
        lead_label.setFont(QFont("Arial", 11))
        controls.addWidget(lead_label)
        
        self.lead_combo = QComboBox()
        self.lead_combo.setMinimumWidth(100)
        self.lead_combo.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
        self.lead_combo.addItems(["Lead I", "Lead II", "V1", "V2", "V3", "V4", "V5", "V6"])
        self.lead_combo.setCurrentText("Lead II")
        self.lead_combo.currentTextChanged.connect(self.on_lead_changed)
        controls.addWidget(self.lead_combo)
        
        controls.addSpacing(10)
        
        # Generate Report button (initially disabled)
        self.report_btn = QPushButton("Generate HRV Report")
        self.report_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff8533;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setEnabled(False)
        controls.addWidget(self.report_btn)
        
        layout.addLayout(controls)
        
        # Metrics display section (below buttons, without Time)
        metrics_card = QFrame()
        metrics_card.setStyleSheet("background: white; border-radius: 10px; padding: 10px;")
        metrics_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setSpacing(20)
        
        # Store metric labels for live update
        self.metric_labels = {}
        metric_info = [
            ("HR", "00", "BPM", "heart_rate"),
            ("PR", "0", "ms", "pr_interval"),
            ("QRS Complex", "0", "ms", "qrs_duration"),
            # ("P", "0", "ms", "st_interval"),
            ("QT/Qtc", "0", "ms", "qtc_interval"),
        ]
        
        for title, value, unit, key in metric_info:
            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val = QLabel(f"{value} {unit}")
            val.setFont(QFont("Arial", 18, QFont.Bold))
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            box.addWidget(lbl)
            box.addWidget(val)
            metrics_layout.addLayout(box)
            self.metric_labels[key] = val  # Store reference for live update
        
        layout.addWidget(metrics_card)
        
        # Plot area
        plot_frame = QFrame()
        plot_frame.setStyleSheet("background: #000; border-radius: 10px;")
        plot_layout = QVBoxLayout(plot_frame)
        
        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('black')

        # Disable manual zoom/pan (amplitude lock)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()  # Hide auto-scale button

        # self.plot_widget.setLabel('left', 'Amplitude (mV)', color='white', fontsize=12)
        # self.plot_widget.setLabel('bottom', 'Time (s)', color='white', fontsize=12)
        self.plot_widget.showGrid(x=False, y=False, alpha=0.3)
        # self.plot_widget.getAxis('left').setPen(pg.mkPen(color='white', width=0.7))
        # self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='white', width=0.7))
        # self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='white'))
        # self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='white'))
        self.plot_widget.showAxis('left', False)
        self.plot_widget.showAxis('bottom', False)
        
        # Plot curve
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen(color='#00FF00', width=1.2))
        
        plot_layout.addWidget(self.plot_widget)
        layout.addWidget(plot_frame, stretch=1)
        
        # Info label
        current_lead = self.lead_combo.currentText()
        self.info_label = QLabel(f"Capture 5 minutes of {current_lead} data for HRV analysis. The capture will stop automatically after 5 minutes.")
        self.info_label.setFont(QFont("Arial", 10))
        self.info_label.setStyleSheet("color: #666; padding: 10px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def on_lead_changed(self, text):
        """Handle lead selection change"""
        if text == "Lead I":
            self.selected_lead = "I"
        elif text == "Lead II":
            self.selected_lead = "II"
        else:
            self.selected_lead = text
            
        self.title_label.setText(f"HRV Test - {text}")

        self.info_label.setText(f"Capture 5 minutes of {text} data for HRV analysis. The capture will stop automatically after 5 minutes.")
        
    def refresh_com_ports(self):
        """Refresh available COM ports"""
        pass
    
    def start_capture(self):
        """Start capturing selected lead data"""
        # CHECK: Ensure no other test is running
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            # Check if dashboard has the can_start_test method
            if hasattr(self.dashboard_instance, 'can_start_test'):
                if not self.dashboard_instance.can_start_test("hrv_test"):
                    return
                # Set state to running
                self.dashboard_instance.update_test_state("hrv_test", True)

        if not SERIAL_AVAILABLE or not ECG_TEST_AVAILABLE:
            QMessageBox.warning(self, "Serial Not Available", 
                              "Serial/ECG modules are not available. Please install pyserial and restart.")
            return
        
        # Get port from settings or auto-detect
        port_to_use = self.settings_manager.get_serial_port()
        baudrate = int(self.settings_manager.get_setting("baud_rate", "115200"))

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
                    
                    # Close the detected serial object
                    try:
                        if detected_serial and detected_serial.is_open:
                            detected_serial.close()
                    except Exception as e:
                        print(f" Warning: Failed to close detected serial port: {e}")

                    # Save to settings
                    if hasattr(self, 'settings_manager'):
                        self.settings_manager.set_setting("serial_port", detected_port)
                        self.settings_manager.save_settings()
                else:
                    QMessageBox.warning(self, "No Device Found", 
                                      "Could not auto-detect ECG device. Please check connection.")
                    if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                        self.dashboard_instance.update_test_state("hrv_test", False)
                    return
            except Exception as scan_err:
                print(f" Port scan failed: {scan_err}")
                QMessageBox.warning(self, "Scan Failed", f"Port scan failed: {scan_err}")
                return
        
        try:
            # Use GlobalHardwareManager to get the shared SerialStreamReader
            from ecg.serial.serial_reader import GlobalHardwareManager
            self.serial_reader = GlobalHardwareManager().get_reader(port_to_use, baudrate)

            # Start/Resume acquisition. The start() method now handles 
            # skipping hardware commands if already running.
            self.serial_reader.start()
            
            # Reset data - use HISTORY_LENGTH for sufficient buffer size
            HISTORY_LENGTH = 10000  # Match initialization size
            self.data = np.zeros(HISTORY_LENGTH, dtype=np.float32)  # Reset circular buffer
            self.captured_data = []  # Store all captured data with timestamps
            self.sample_index = 0
            self.active_samples = 0
            self.start_time = time.time()
            self.is_capturing = True
            
            # Reset smoothing buffers
            if self.ecg_calculator:
                if hasattr(self.ecg_calculator, 'smoothing_buffers'):
                    self.ecg_calculator.smoothing_buffers = {}
                # Initialize calculator data buffers with sufficient size for calculations
                if not hasattr(self.ecg_calculator, 'data') or len(self.ecg_calculator.data) < 12:
                    self.ecg_calculator.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.report_btn.setEnabled(False)
            self.lead_combo.setEnabled(False)

            # Lock display interaction during capture
            self.plot_widget.setMouseEnabled(x=False, y=False)

            self.status_label.setText("Status: Capturing...")
            self.status_label.setStyleSheet("color: #28a745; padding: 5px;")
            
            # Start timers
            self.capture_timer.start(50)  # Update plot every 50ms
            self.duration_timer.start(1000)  # Check duration every second
            self.metrics_timer = QTimer(self)
            self.metrics_timer.timeout.connect(self.update_metrics)
            self.metrics_timer.start(200)  # Update metrics every 200ms
            
            QMessageBox.information(self, "Capture Started", 
                                  f"{self.selected_lead} capture started. It will automatically stop after 5 minutes.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to start capture: {str(e)}")
            self.crash_logger.log_error(
                message=f"HRV test capture start error: {e}",
                exception=e,
                category="HRV_TEST_ERROR"
            )
    
    def stop_capture(self):
        """Stop capturing data"""
        # UPDATE STATE: Test stopped
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if hasattr(self.dashboard_instance, 'update_test_state'):
                self.dashboard_instance.update_test_state("hrv_test", False)

        self.is_capturing = False
        
        if self.serial_reader:
            try:
                # IMPORTANT: We don't close the shared reader here because other tests 
                # (like 12-lead) might still be using it in the background.
                # We just stop our reference to it.
                pass
            except:
                pass
            self.serial_reader = None
        
        # Stop timers
        self.capture_timer.stop()
        self.duration_timer.stop()
        if hasattr(self, 'metrics_timer'):
            self.metrics_timer.stop()
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.lead_combo.setEnabled(True)

        # Re-enable display interaction after capture (optional, but allows inspection)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        if len(self.captured_data) > 0:
            self.report_btn.setEnabled(True)
            self.status_label.setText(f"Status: Capture Complete")
        else:
            self.status_label.setText("Status: Capture Stopped (No data)")
        
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        self.timer_label.setText("Time: 00:00")

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
        """Check if 5 minutes have elapsed"""
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
                                  "5-minute capture completed successfully!")
    
    def update_plot(self):
        """Update the plot with new data"""
        if not self.is_capturing or not self.serial_reader:
            return

        # Check if device got disconnected suddenly
        if not self.serial_reader.running:
            print("⚠️ Device disconnected during HRV test!")
            self.stop_capture()
            return
            
        
        try:
            # Read multiple packets per GUI tick (same idea as 12‑lead test)
            # so we don't under‑sample and miss beats when HR changes.
            max_packets = 100
            
            # Use new packet-based reading from SerialStreamReader
            packets = self.serial_reader.read_packets(max_packets=max_packets)
            
            for packet in packets:
                # Packet is a dictionary with lead names as keys (e.g., {"I": value, "II": value, ...})
                # Extract selected lead directly from the packet
                lead_value = packet.get(self.selected_lead, None)
                
                if lead_value is not None:
                    lead_value = float(lead_value)
                    self.active_samples = min(len(self.data), self.active_samples + 1)
                    
                    # 1. ANALYSIS DATA (Must be RAW for accurate interval calculation)
                    # We update the calculator's buffer with the UNTOUCHED lead value
                    if self.ecg_calculator:
                        try:
                            # Map lead name to index
                            lead_indices = {
                                "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                                "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                            }
                            lead_idx = lead_indices.get(self.selected_lead, 1)  # Default to II if not found
                            
                            self.ecg_calculator.data[lead_idx] = np.roll(self.ecg_calculator.data[lead_idx], -1)
                            self.ecg_calculator.data[lead_idx][-1] = lead_value

                            # FORCE UPDATE LEAD II (Index 1) if different
                            # This ensures the dashboard metrics (which often rely on Lead II) 
                            # reflect the SELECTED lead's data.
                            if lead_idx != 1:
                                self.ecg_calculator.data[1] = np.roll(self.ecg_calculator.data[1], -1)
                                self.ecg_calculator.data[1][-1] = lead_value

                        except Exception as e:
                            print(f" Error updating calculator buffer: {e}")
                    
                    # 2. DISPLAY DATA (Use RAW data for display as requested)
                    # User requested raw plot centered at 2048
                    # We skip smoothing for display to show raw signal
                    
                    # Define smoothed_value as raw value (since we are skipping smoothing)
                    # This ensures subsequent code (report storage) works and matches display
                    smoothed_value = lead_value
                    
                    # Update local circular buffer for plot
                    self.data = np.roll(self.data, -1)
                    self.data[-1] = lead_value
                    
                    # Fixed sampling rate for medical device (do not estimate from packet timing)
                    self.sampling_rate = 500.0
                    
                    # Store data point with timestamp for final report generation
                    elapsed = time.time() - self.start_time
                    self.captured_data.append({
                        'time': elapsed,
                        'value': smoothed_value  # Reports use smoothed values for clean graphs
                    })
                
            # Update plot (show last 10 seconds for real-time view)
            # Use circular buffer data for smoother display
            if len(self.captured_data) > 0:
                # Use the circular buffer for display (contains smoothed data)
                buffer_data = self.data[self.data != 0]  # Get non-zero values

                if len(buffer_data) > 100: # Ensure enough data for filtering
                    # Get filter settings from SettingsManager
                    ac_val = self.settings_manager.get_setting("filter_ac", "50")
                    emg_val = self.settings_manager.get_setting("filter_emg", "35")
                    
                    fs = self.sampling_rate if self.sampling_rate > 0 else 500.0

                    # Pad data to reduce transient response at start and end
                    # This fixes the "noise" at the start of acquisition
                    pad_len = 50
                    if len(buffer_data) > pad_len:
                        start_pad = np.full(pad_len, buffer_data[0])
                        end_pad = np.full(pad_len, buffer_data[-1])
                        padded_data = np.concatenate((start_pad, buffer_data, end_pad))
                    else:
                        padded_data = buffer_data
                        pad_len = 0
                    
                    # Apply AC Filter
                    if ac_val != "Off" and ac_val != "off":
                        padded_data = apply_ac_filter(padded_data, fs, ac_val)
                        
                    # Apply EMG Filter
                    if emg_val != "Off" and emg_val != "off":
                        padded_data = apply_emg_filter(padded_data, fs, emg_val)

                    # Trim padding
                    if pad_len > 0:
                        buffer_data = padded_data[pad_len:-pad_len]
                    else:
                        buffer_data = padded_data

                if len(buffer_data) > 0:
                    # Create time axis based on sampling rate
                    num_samples = len(buffer_data)
                    if self.sampling_rate > 0:
                        time_axis = np.arange(num_samples) / self.sampling_rate
                        # Show last 10 seconds
                        max_time = time_axis[-1] if len(time_axis) > 0 else 0
                        min_time = max(0, max_time - 10)
                        mask = time_axis >= min_time
                        display_times = time_axis[mask].tolist()
                        display_values = buffer_data[mask].tolist()
                    else:
                        # Fallback to timestamp-based display
                        times = [d['time'] for d in self.captured_data]
                        values = [d['value'] for d in self.captured_data]
                        if len(times) > 0:
                            max_time = max(times)
                            min_time = max(0, max_time - 10)
                            display_times = [t for t in times if t >= min_time]
                            display_values = [v for i, v in enumerate(values) if times[i] >= min_time]
                        else:
                            display_times = []
                            display_values = []
                else:
                    # Fallback to timestamp-based display if buffer is empty
                    times = [d['time'] for d in self.captured_data]
                    values = [d['value'] for d in self.captured_data]
                    if len(times) > 0:
                        max_time = max(times)
                        min_time = max(0, max_time - 10)
                        display_times = [t for t in times if t >= min_time]
                        display_values = [v for i, v in enumerate(values) if times[i] >= min_time]
                    else:
                        display_times = []
                        display_values = []
                
                if len(display_times) > 0:
                    # Plot RAW data directly without centering/scaling
                    # User requested: "raw plot... from 0 to 4096 along y-axis and center of lead II should be from 2048"
                    
                    scaled_values = [2048 + (v - 2048) * 0.5 for v in display_values]
                    self.plot_curve.setData(display_times, scaled_values)
                            
                    # Auto-scale X axis
                    self.plot_widget.setXRange(min(display_times), max(display_times), padding=0.1)
                            
                    # Fixed Y-axis scaling (0-4096)
                    self.plot_widget.setYRange(0, 4096, padding=0)
                    
                    # # Draw a center line at 2048 for reference (optional, but helpful)
                    # if not hasattr(self, 'center_line'):
                    #     self.center_line = pg.InfiniteLine(pos=2048, angle=0, pen=pg.mkPen(color='gray', width=0.7, style=Qt.DashLine))
                    #     self.plot_widget.addItem(self.center_line)
        
        except Exception as e:
            # Silently handle errors during capture
            pass
    
    def generate_report(self):
        """Generate HRV report PDF with 5 rows of 1-minute data each"""
        if len(self.captured_data) == 0:
            QMessageBox.warning(self, "No Data", 
                              "No data available to generate report.")
            return
        
        # Get save location
        from PyQt5.QtWidgets import QFileDialog
        default_filename = f"HRV_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save HRV Report", default_filename, "PDF Files (*.pdf)"
        )
        
        if not filepath:
            return
        
        try:
            # Import the HRV ECG report generator (separate file with EXACT same format as main report)
            from ecg.hrv_ecg_report_generator import generate_hrv_ecg_report
            
            # Prepare patient data - PICK LATEST FROM all_patients.json (COMPLETE DETAILS)
            # Priority 1: all_patients.json (HAS doctor_mobile and complete info)
            # Priority 2: last_patient_details.json (fallback)
            patient = {}
            try:
                # Get base directory (modularecg folder)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                base_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
                
                print(f" HRV Report: Looking for patient files in: {base_dir}")
                
                # Try all_patients.json FIRST (has complete details including doctor_mobile)
                all_patients_file = os.path.join(base_dir, 'all_patients.json')
                print(f" Checking all_patients.json at: {all_patients_file}")
                print(f" File exists: {os.path.exists(all_patients_file)}")
                
                if os.path.exists(all_patients_file):
                    with open(all_patients_file, 'r') as f:
                        all_patients_data = json.load(f)
                    
                    print(f" Loaded JSON data type: {type(all_patients_data)}")
                    print(f" Has 'patients' key: {'patients' in all_patients_data if isinstance(all_patients_data, dict) else False}")
                    
                    # Get LATEST patient (last entry in the list)
                    if isinstance(all_patients_data, dict) and 'patients' in all_patients_data:
                        patients_list = all_patients_data['patients']
                        print(f" Number of patients in list: {len(patients_list)}")
                        
                        if patients_list and len(patients_list) > 0:
                            patient = patients_list[-1].copy()  # LATEST = LAST ENTRY (make a copy)
                            print(f" Loaded LATEST patient from all_patients.json:")
                            print(f"   Name: {patient.get('first_name')} {patient.get('last_name')}")
                            print(f"   Age: {patient.get('age')}")
                            print(f"   Gender: {patient.get('gender')}")
                            print(f"   Doctor: {patient.get('doctor')}")
                            print(f"   Org: {patient.get('Org.')}")
                            print(f"   Mobile: {patient.get('doctor_mobile')}")
                        else:
                            print(f" Patients list is empty!")
                    else:
                        print(f" Invalid JSON structure in all_patients.json")
                else:
                    print(f" all_patients.json not found at: {all_patients_file}")
                
                # Fallback: Try last_patient_details.json
                if not patient:
                    patient_file = os.path.join(base_dir, 'last_patient_details.json')
                    print(f" Trying fallback: {patient_file}")
                    if os.path.exists(patient_file):
                        with open(patient_file, 'r') as f:
                            patient = json.load(f)
                        print(f" Loaded patient from last_patient_details.json (fallback): {patient.get('patient_name', 'Unknown')}")
                
                if not patient:
                    print(f" No patient files found, using defaults")
                    
            except Exception as e:
                print(f" ERROR loading patient details: {e}")
                import traceback
                traceback.print_exc()
            
            # Ensure all required fields exist (ONLY if patient not loaded)
            if not patient or "first_name" not in patient:
                print(f" Patient data empty or invalid, using defaults")
                patient = {
                    "first_name": "HRV",
                    "last_name": "Patient",
                    "age": "",
                    "gender": "",
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Org.": "",
                    "doctor_mobile": "",
                    "doctor": "",
                }
            else:
                print(f" Patient data loaded successfully: {patient.get('first_name')} {patient.get('last_name')}")
            
            # Ensure doctor_mobile exists (might be missing in old data)
            if "doctor_mobile" not in patient:
                patient["doctor_mobile"] = ""
                print(f" Added missing doctor_mobile field")
            
            # Update date_time to current (keep other fields from loaded data)
            patient["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f" Updated date_time to: {patient['date_time']}")
            
            # Calculate REAL metrics from captured Lead II data (SAME AS update_metrics() method)
            # Use the SAME calculation methods as 12-lead test
            hr_value = 0
            pr_value = 0
            qrs_value = 0
            st_value = 0
            qt_value = 0
            qtc_value = 0
            hr_max = 0
            hr_min = 0
            hr_avg = 0
            
            # Calculate from Lead II data using ECG calculator (if available)
            if self.ecg_calculator and len(self.captured_data) >= 200:
                try:
                    # Map lead name to index
                    lead_indices = {
                        "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                        "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                    }
                    lead_idx = lead_indices.get(self.selected_lead, 1)

                    # Use last 2000 samples or all available
                    num_samples = min(2000, len(self.captured_data))
                    recent_data = [d['value'] for d in self.captured_data[-num_samples:]]
                    signal = np.array(recent_data, dtype=np.float32)
                    
                    # Update calculator's data buffer
                    buffer_size = max(len(signal), 1000)
                    if len(self.ecg_calculator.data[lead_idx]) < buffer_size:
                        self.ecg_calculator.data[lead_idx] = np.zeros(buffer_size, dtype=np.float32)
                    
                    if len(signal) <= len(self.ecg_calculator.data[lead_idx]):
                        self.ecg_calculator.data[lead_idx][-len(signal):] = signal
                    else:
                        self.ecg_calculator.data[lead_idx] = signal[-len(self.ecg_calculator.data[lead_idx]):]
                    
                    # Set sampling rate
                    if hasattr(self.ecg_calculator, 'sampler') and self.ecg_calculator.sampler:
                        self.ecg_calculator.sampler.sampling_rate = self.sampling_rate
                    
                    # Calculate metrics using SAME methods as 12-lead test
                    hr_value = self.ecg_calculator.calculate_heart_rate(self.ecg_calculator.data[lead_idx])
                    pr_value = self.ecg_calculator.calculate_pr_interval(self.ecg_calculator.data[lead_idx])
                    qrs_value = self.ecg_calculator.calculate_qrs_duration(self.ecg_calculator.data[lead_idx])
                    st_value = self.ecg_calculator.calculate_st_interval(self.ecg_calculator.data[lead_idx])
                    qt_value = self.ecg_calculator.calculate_qt_interval(self.ecg_calculator.data[lead_idx])
                    
                    # Calculate HR stats from all captured data
                    all_hr_values = []
                    window_size = 200  # Calculate HR every 200 samples
                    for i in range(0, len(self.captured_data) - window_size, window_size // 2):
                        window_data = [d['value'] for d in self.captured_data[i:i+window_size]]
                        window_signal = np.array(window_data, dtype=np.float32)
                        if len(self.ecg_calculator.data[lead_idx]) >= len(window_signal):
                            self.ecg_calculator.data[lead_idx][-len(window_signal):] = window_signal
                            hr = self.ecg_calculator.calculate_heart_rate(self.ecg_calculator.data[lead_idx])
                            if hr > 0:
                                all_hr_values.append(hr)
                    
                    if all_hr_values:
                        hr_max = max(all_hr_values)
                        hr_min = min(all_hr_values)
                        hr_avg = int(np.mean(all_hr_values))
                    
                    print(f" Calculated metrics from HRV data: HR={hr_value}, PR={pr_value}, QRS={qrs_value}")
                except Exception as e:
                    print(f" Error calculating metrics: {e}")
            
            # Prepare metrics data - SAME STRUCTURE AS MAIN ECG REPORT
            data = {
                "HR": hr_value,
                "beat": hr_value,  # Same as HR for observation table
                "PR": pr_value,
                "QRS": qrs_value,
                "QT": qt_value,
                "QTc": qtc_value,
                "ST": st_value,
                "HR_max": hr_max,
                "HR_min": hr_min,
                "HR_avg": hr_avg,
                "Heart_Rate": hr_value,
            }

            # Ensure we feed exactly 1-minute of samples (30000 at 500 Hz) when available
            max_samples = int((self.sampling_rate or 500) * 60 * 5)
            raw_data_for_report = self.captured_data[-max_samples:] if len(self.captured_data) >= max_samples else self.captured_data
            if len(raw_data_for_report) > 0:
                t0 = raw_data_for_report[0]['time']
                raw_data_for_report = [
                    {'time': float(d['time']) - float(t0), 'value': float(d['value'])}
                    for d in raw_data_for_report
                ]
            
            # Generate report using the COMPLETE format (same as main ECG report)
            # This includes: Patient Details, Observation, Conclusions, and 5 one-minute Lead II graphs
            result = generate_hrv_ecg_report(
                filename=filepath,
                captured_data=raw_data_for_report,
                data=data,
                patient=patient,
                settings_manager=self.settings_manager,
                selected_lead = self.selected_lead
            )
            
            if result:
                QMessageBox.information(self, "Report Generated", 
                                      f"HRV ECG report saved successfully:\n{filepath}")
                try:
                    append_history_entry(patient, filepath, report_type="HRV", username=self.username)
                except Exception as hist_err:
                    print(f" Failed to append HRV history: {hist_err}")
            else:
                QMessageBox.warning(self, "Report Warning", 
                                  "Report generation completed with warnings.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to generate report: {str(e)}")
            self.crash_logger.log_error(
                message=f"HRV report generation error: {e}",
                exception=e,
                category="HRV_REPORT_ERROR"
            )

    def calculate_time_domain_hrv_metrics(self):
        """c
        Calculate time‑domain HRV metrics from the full selected lead capture.
        
        Returns a dict with:
            mean_rr_ms, sdnn_ms, rmssd_ms, nn50, pnn50, num_intervals
        or None if insufficient data.
        """
        try:
            if len(self.captured_data) < 500:
                return None

            # Build ECG signal array from captured selected lead values
            signal = np.array([d['value'] for d in self.captured_data], dtype=float)
            if signal.size < 500:
                return None

            # Use the same sampling rate we have been tracking during capture
            fs = float(self.sampling_rate or 0)
            if not np.isfinite(fs) or fs <= 0:
                fs = 500.0  # sensible default matching live capture

            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz) - same as 12-lead test
            from scipy.signal import butter, filtfilt
            try:
                nyquist = fs / 2
                low = max(0.001, 0.5 / nyquist)
                high = min(0.999, 40 / nyquist)
                if low < high:
                    b, a = butter(4, [low, high], btype='band')
                    signal = filtfilt(b, a, signal)
                    # Check for invalid values after filtering
                    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                        print(" Filter produced invalid values, using unfiltered signal")
                        signal = np.array([d['value'] for d in self.captured_data], dtype=float)
            except Exception as e:
                print(f" Error in signal filtering: {e}, using unfiltered signal")
                signal = np.array([d['value'] for d in self.captured_data], dtype=float)

            signal_std = np.std(signal)
            if signal_std == 0:
                return None
            peaks, _ = find_peaks(
                signal,
                distance=int(0.25 * fs),
                prominence=signal_std * 0.6
            )

            if len(peaks) < 3:
                return None

            # Proceed with detected peaks directly

            # R‑R intervals in milliseconds
            rr_intervals = np.diff(peaks) * (1000.0 / fs)

            rr = rr_intervals[(rr_intervals > 300.0) & (rr_intervals < 1500.0)]
            print("Pratyaksh rr[:50]:", rr[:50])
            print(np.abs(np.diff(rr))[:50])
            if rr.size < 2:
                return None

            print("Pratyaksh rr:", rr)

            median_rr = np.median(rr)
            mask = np.abs(rr - median_rr) < 0.2 * median_rr
            rr_clean = rr[mask]
            if rr_clean.size < 2:
                return None
            rr_diff = np.abs(np.diff(rr_clean))
            rr_final = rr_clean[1:][rr_diff < 100.0]
            if rr_final.size < 2:
                return None

            diff_rr = np.diff(rr_final)
            mean_rr_ms = float(np.mean(rr_final))
            sdnn_ms = float(np.std(rr_final, ddof=1))
            rmssd_ms = float(np.sqrt(np.mean(diff_rr ** 2)))
            nn50 = int(np.sum(np.abs(diff_rr) > 50.0))
            pnn50 = float((nn50 / len(diff_rr)) * 100.0) if len(diff_rr) > 0 else 0.0

            print(f" Pratyaksh mean_rr_ms: {mean_rr_ms}, sdnn_ms: {sdnn_ms}, rmssd_ms: {rmssd_ms}, nn50: {nn50}, pnn50: {pnn50}, num_intervals: {int(valid_rr.size)}")

            return {
                "mean_rr_ms": mean_rr_ms,
                "sdnn_ms": sdnn_ms,
                "rmssd_ms": rmssd_ms,
                "nn50": nn50,
                "pnn50": pnn50,
                "num_intervals": int(valid_rr.size),
            }
        except Exception as e:
            # Log but don't crash report generation
            try:
                self.crash_logger.log_error(
                    message=f"HRV metrics calculation error: {e}",
                    exception=e,
                    category="HRV_METRICS_ERROR"
                )
            except Exception:
                pass
            return None
    
    def update_metrics(self):
        """Calculate and update ECG metrics from selected lead data using same methods as 12-lead test"""
        if not self.is_capturing or len(self.captured_data) < 200:
            return
        
        try:
            current_fs = 500.0
            self.sampling_rate = 500.0
            
            if self.ecg_calculator:
                # Ensure the calculator's sampler is in sync
                if not hasattr(self.ecg_calculator, 'sampler') or self.ecg_calculator.sampler is None:
                    from ecg.twelve_lead_test import SamplingRateCalculator
                    self.ecg_calculator.sampler = SamplingRateCalculator()
                self.ecg_calculator.sampler.sampling_rate = current_fs

                # Update main sampling rate too
                self.ecg_calculator.sampling_rate = current_fs

                # TRIGGER STABLE MEDIAN-BEAT ANALYSIS (Same as 12-lead test)
                # This ensures we use the exact same logic as the dashboard for intervals
                try:
                    # ECGTestPage.calculate_ecg_metrics() updates its internal metric_labels
                    # using the median beat (GE/Philips standard)
                    self.ecg_calculator.calculate_ecg_metrics()
                except Exception as e:
                    print(f" calculate_ecg_metrics error in HRV test: {e}")

                # FETCH SMOOTHED METRICS (Same logic as dashboard)
                # get_current_metrics provides stable values, including HR smoothing
                metrics = self.ecg_calculator.get_current_metrics()
                
                # Extract values with fallbacks
                hr_val = metrics.get('heart_rate', '0')
                pr_val = metrics.get('pr_interval', '0')
                qrs_val = metrics.get('qrs_duration', '0')
                st_val = metrics.get('st_interval', '0')
                qtc_val = metrics.get('qtc_interval', '0')

                # Update UI labels with identical formatting to 12-lead test
                if 'heart_rate' in self.metric_labels:
                    # Dashboard uses "bpm", we use "BPM" for consistency with the rest of this UI
                    self.metric_labels['heart_rate'].setText(f"{hr_val} BPM" if hr_val != '0' else "00 BPM")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText(f"{pr_val} ms" if pr_val != '0' else "0 ms")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText(f"{qrs_val} ms" if qrs_val != '0' else "0 ms")
                if 'st_interval' in self.metric_labels:
                    try:
                        p_val = int(round(float(st_val)))
                        p_text = f"{p_val} ms"
                    except:
                        p_text = f"{st_val} ms" if st_val != '0' else "0 ms"
                    self.metric_labels['st_interval'].setText(p_text)
                if 'qtc_interval' in self.metric_labels:
                    # 12-lead test might return "QT/QTc", handle both
                    if '/' in str(qtc_val):
                        self.metric_labels['qtc_interval'].setText(f"{qtc_val} ms")
                    else:
                        self.metric_labels['qtc_interval'].setText(f"{qtc_val} ms" if qtc_val != '0' else "0 ms")
            else:
                # Fallback if calculator not available
                for key in self.metric_labels:
                    if key == 'heart_rate': self.metric_labels[key].setText("00 BPM")
                    elif key == 'st_interval': self.metric_labels[key].setText("0 ms")
                    else: self.metric_labels[key].setText("0 ms")
        
        except Exception as e:
            # Log but don't crash
            print(f" Error updating HRV metrics: {e}")
            pass
    
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
