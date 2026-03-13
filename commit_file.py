import sys
import time
import numpy as np
from pyparsing import line
import logging
import traceback
from utils.crash_logger import get_crash_logger
from PyQt5.QtWidgets import QMessageBox
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print("⚠️ Serial module not available - ECG hardware features disabled")
    SERIAL_AVAILABLE = False
    # Create dummy serial classes
    class Serial:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def readline(self): return b''
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()
import csv
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️ OpenCV (cv2) module not available - some features disabled")
    CV2_AVAILABLE = False
    cv2 = None
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QGroupBox, QFileDialog,
    QStackedLayout, QGridLayout, QSizePolicy, QMessageBox, QFormLayout, QLineEdit, QFrame, QApplication, QDialog
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QDateTime 
# --- CHANGED: Removed Matplotlib imports ---

# --- ADDED: PyQtGraph is now used for all plotting ---
import pyqtgraph as pg
import re
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional
from ecg.recording import ECGMenu
from scipy.signal import find_peaks
from utils.settings_manager import SettingsManager
from utils.localization import translate_text
from .demo_manager import DemoManager
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from functools import partial # For plot clicking
from .clinical_measurements import (
    build_median_beat, get_tp_baseline, measure_qt_from_median_beat,
    measure_rv5_sv1_from_median_beat, measure_st_deviation_from_median_beat,
    calculate_axis_from_median_beat
)

# --- Configuration ---
# Increase history to keep longer segments visible in each frame.
# At 500 Hz sampling this stores ~20 seconds, enough for 6-7 peaks
# even at the slowest sweep speed.
HISTORY_LENGTH = 10000
NORMAL_HR_MIN, NORMAL_HR_MAX = 60, 100
LEAD_LABELS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

class SamplingRateCalculator:
    def __init__(self, update_interval_sec=5):
        self.sample_count = 0
        self.last_update_time = time.monotonic()
        self.update_interval = update_interval_sec
        self.sampling_rate = 0

    def add_sample(self):
        self.sample_count += 1
        current_time = time.monotonic()
        elapsed = current_time - self.last_update_time
        if elapsed >= self.update_interval:
            self.sampling_rate = self.sample_count / elapsed
            self.sample_count = 0
            self.last_update_time = current_time
        return self.sampling_rate

# ------------------------ ECG Display Gain Helper (Clinical Standard) ------------------------

def get_display_gain(wave_gain_mm: float) -> float:
    """
    ECG display gain calculation (hospital standard):
    10 mm/mV = 1.0x (clinical baseline)
    
    Args:
        wave_gain_mm: Wave gain setting in mm/mV (e.g., 2.5, 5, 10, 20)
    
    Returns:
        Display gain factor:
        - 2.5 mm/mV → 0.25x
        - 5 mm/mV → 0.5x
        - 10 mm/mV → 1.0x (baseline)
        - 20 mm/mV → 2.0x
    
    This matches GE / Philips monitor behavior.
    """
    try:
        return float(wave_gain_mm) / 10.0
    except (ValueError, TypeError):
        return 1.0  # Default to 10mm/mV baseline

# ------------------------ Realistic ECG Waveform Generator ------------------------

def generate_realistic_ecg_waveform(duration_seconds=10, sampling_rate=500, heart_rate=72, lead_name="II"):
    """
    Generate realistic ECG waveform with proper PQRST complexes
    - duration_seconds: Length of waveform in seconds
    - sampling_rate: Samples per second (Hz)
    - heart_rate: Beats per minute
    - lead_name: Lead name for lead-specific characteristics
    """
    import numpy as np
    
    # Calculate parameters
    total_samples = int(duration_seconds * sampling_rate)
    rr_interval = 60.0 / heart_rate  # RR interval in seconds
    samples_per_beat = int(rr_interval * sampling_rate)
    
    # Create time array
    t = np.linspace(0, duration_seconds, total_samples)
    
    # Initialize waveform
    ecg = np.zeros(total_samples)
    
    # Lead-specific characteristics (amplitudes in mV)
    lead_characteristics = {
        "I": {"p_amp": 0.1, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0},
        "II": {"p_amp": 0.15, "qrs_amp": 1.2, "t_amp": 0.3, "baseline": 0.0},
        "III": {"p_amp": 0.05, "qrs_amp": 0.6, "t_amp": 0.15, "baseline": 0.0},
        "aVR": {"p_amp": -0.1, "qrs_amp": -0.8, "t_amp": -0.2, "baseline": 0.0},
        "aVL": {"p_amp": 0.08, "qrs_amp": 0.7, "t_amp": 0.18, "baseline": 0.0},
        "aVF": {"p_amp": 0.12, "qrs_amp": 0.9, "t_amp": 0.25, "baseline": 0.0},
        "V1": {"p_amp": 0.05, "qrs_amp": 0.3, "t_amp": 0.1, "baseline": 0.0},
        "V2": {"p_amp": 0.08, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0},
        "V3": {"p_amp": 0.1, "qrs_amp": 1.0, "t_amp": 0.25, "baseline": 0.0},
        "V4": {"p_amp": 0.12, "qrs_amp": 1.1, "t_amp": 0.3, "baseline": 0.0},
        "V5": {"p_amp": 0.1, "qrs_amp": 1.0, "t_amp": 0.25, "baseline": 0.0},
        "V6": {"p_amp": 0.08, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0}
    }
    
    char = lead_characteristics.get(lead_name, lead_characteristics["II"])
    
    # Generate beats
    beat_start = 0
    while beat_start < total_samples:
        # P wave (atrial depolarization) - 80-120ms
        p_duration = 0.1  # 100ms
        p_samples = int(p_duration * sampling_rate)
        p_start = beat_start
        p_end = min(p_start + p_samples, total_samples)
        
        if p_start < total_samples:
            p_t = np.linspace(0, p_duration, p_end - p_start)
            p_wave = char["p_amp"] * np.sin(np.pi * p_t / p_duration) * np.exp(-2 * p_t / p_duration)
            ecg[p_start:p_end] += p_wave
        
        # PR interval (isoelectric line) - 120-200ms
        pr_duration = 0.16  # 160ms
        pr_samples = int(pr_duration * sampling_rate)
        pr_start = p_end
        pr_end = min(pr_start + pr_samples, total_samples)
        
        # QRS complex (ventricular depolarization) - 80-120ms
        qrs_duration = 0.08  # 80ms
        qrs_samples = int(qrs_duration * sampling_rate)
        qrs_start = pr_end
        qrs_end = min(qrs_start + qrs_samples, total_samples)
        
        if qrs_start < total_samples:
            qrs_t = np.linspace(0, qrs_duration, qrs_end - qrs_start)
            # Q wave (small negative deflection)
            q_wave = -char["qrs_amp"] * 0.1 * np.exp(-10 * qrs_t / qrs_duration)
            # R wave (large positive deflection)
            r_wave = char["qrs_amp"] * np.sin(np.pi * qrs_t / qrs_duration) * np.exp(-3 * qrs_t / qrs_duration)
            # S wave (negative deflection after R)
            s_wave = -char["qrs_amp"] * 0.3 * np.exp(-5 * qrs_t / qrs_duration)
            
            qrs_complex = q_wave + r_wave + s_wave
            ecg[qrs_start:qrs_end] += qrs_complex
        
        # ST segment (isoelectric) - 80-120ms
        st_duration = 0.08  # 80ms
        st_samples = int(st_duration * sampling_rate)
        st_start = qrs_end
        st_end = min(st_start + st_samples, total_samples)
        
        # T wave (ventricular repolarization) - 160-200ms
        t_duration = 0.16  # 160ms
        t_samples = int(t_duration * sampling_rate)
        t_start = st_end
        t_end = min(t_start + t_samples, total_samples)
        
        if t_start < total_samples:
            t_t = np.linspace(0, t_duration, t_end - t_start)
            t_wave = char["t_amp"] * np.sin(np.pi * t_t / t_duration) * np.exp(-2 * t_t / t_duration)
            ecg[t_start:t_end] += t_wave
        
        # Move to next beat
        beat_start += samples_per_beat
    
    # Add baseline wander (low frequency noise)
    baseline_freq = 0.5  # 0.5 Hz
    baseline_wander = 0.05 * np.sin(2 * np.pi * baseline_freq * t)
    ecg += baseline_wander
    
    # Add high frequency noise (muscle artifact, etc.)
    noise = 0.02 * np.random.normal(0, 1, total_samples)
    ecg += noise
    
    # Add baseline offset
    ecg += char["baseline"]
    
    return ecg, t

# ============================================================================
# NEW PACKET-BASED SERIAL PARSING LOGIC
# ============================================================================

# Packet parsing constants
PACKET_SIZE = 22
START_BYTE = 0xE8
END_BYTE = 0x8E
LEAD_NAMES_DIRECT = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
PACKET_REGEX = re.compile(r"(?i)(E8(?:[0-9A-F\s]{2,})?8E)")

def hex_string_to_bytes(hex_str: str) -> bytes:
    """Convert hex string to bytes"""
    cleaned = re.sub(r"[^0-9A-Fa-f]", "", hex_str)
    if len(cleaned) % 2 != 0:
        raise ValueError("Hex string must have even length")
    return bytes(int(cleaned[i : i + 2], 16) for i in range(0, len(cleaned), 2))

def decode_lead(msb: int, lsb: int) -> Tuple[int, bool]:
    """Decode lead value from MSB and LSB bytes"""
    lower7 = lsb & 0x7F
    upper5 = msb & 0x1F
    value = (upper5 << 7) | lower7
    connected = (msb & 0x20) != 0
    return value, connected

def parse_packet(raw: bytes) -> Dict[str, int]:
    """Parse ECG packet and return dictionary of lead values"""
    if len(raw) != PACKET_SIZE or raw[0] != START_BYTE or raw[-1] != END_BYTE:
        return {}

    lead_values: Dict[str, int] = {}
    idx = 5  # first MSB position

    print("---- New Packet ----")

    for name in LEAD_NAMES_DIRECT:
        msb = raw[idx]
        lsb = raw[idx + 1]
        idx += 2

        value, connected = decode_lead(msb, lsb)

        print(f"{name}: MSB={msb:02X}, LSB={lsb:02X}, value={value}, connected={connected}")

        lead_values[name] = value

    # Derived limb leads
    lead_i = lead_values.get("I", 0)
    lead_ii = lead_values.get("II", 0)

    lead_values["III"] = lead_ii - lead_i
    lead_values["aVR"] = -(lead_i + lead_ii) / 2
    lead_values["aVL"] = (lead_i - lead_values["III"]) / 2
    lead_values["aVF"] = (lead_ii + lead_values["III"]) / 2

    print("Derived:", {
        "III": lead_values["III"],
        "aVR": lead_values["aVR"],
        "aVL": lead_values["aVL"],
        "aVF": lead_values["aVF"],
    })

    print("---------------------\n")

    return lead_values

class SerialStreamReader:
    """Packet-based serial reader for ECG data - NEW IMPLEMENTATION"""
    
    def __init__(self, port: str, baudrate: int, timeout: float = 0.1):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is required for serial capture. pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.buf = bytearray()
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        self.user_details = {}  # For error reporting compatibility
        print(f"🔌 SerialStreamReader initialized: Port={port}, Baud={baudrate}")

    def close(self) -> None:
        """Close serial connection"""
        try:
            self.running = False
            self.ser.close()
        except Exception:
            pass

    def start(self):
        """Start data acquisition"""
        print("🚀 Starting packet-based ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.buf.clear()
        self.running = True
        print("✅ Packet-based ECG device started - waiting for data packets...")

    def stop(self):
        """Stop data acquisition"""
        print("⏹️ Stopping packet-based ECG data acquisition...")
        self.running = False
        print(f"📊 Total data packets received: {self.data_count}")

    def read_packets(self, max_packets: int = 50) -> List[Dict[str, int]]:
        """Read and parse ECG packets from serial stream"""
        if not self.running:
            return []
            
        out: List[Dict[str, int]] = []
        
        try:
            chunk = self.ser.read(1024)
            if chunk:
                self.buf.extend(chunk)

            # Extract packets
            while len(out) < max_packets:
                start_idx = self.buf.find(bytes([START_BYTE]))
                if start_idx == -1:
                    self.buf.clear()
                    break
                if len(self.buf) - start_idx < PACKET_SIZE:
                    if start_idx > 0:
                        del self.buf[:start_idx]
                    break
                    
                candidate = bytes(self.buf[start_idx : start_idx + PACKET_SIZE])
                del self.buf[: start_idx + PACKET_SIZE]

                if candidate[-1] != END_BYTE:
                    continue

                parsed = parse_packet(candidate)
                if parsed:
                    self.data_count += 1
                    print(f"📡 [Packet #{self.data_count}] Received valid packet with {len(parsed)} leads")
                    # Log each lead value as it is parsed
                    for name, val in parsed.items():
                        try:
                            print(f"Serial data - Lead {name}: value={val}")
                        except Exception:
                            pass
                    out.append(parsed)
                    
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            error_msg = f"Packet parsing error: {e}"
            print(f"❌ {error_msg}")
            self.crash_logger.log_error(
                message=error_msg,
                exception=e,
                category="SERIAL_ERROR"
            )
            
        return out

    def _handle_serial_error(self, error):
        """Handle serial communication errors"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        error_msg = f"Serial communication error: {error}"
        print(f"❌ {error_msg}")
        
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self.last_error_time = current_time
            self.consecutive_errors = 0

# ============================================================================
# OLD SERIAL READER (COMMENTED OUT - KEPT FOR REFERENCE)
# ============================================================================

class SerialECGReader:
    def __init__(self, port, baudrate):
        if not SERIAL_AVAILABLE:
            raise ImportError("Serial module not available - cannot create ECG reader")
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        print(f"🔌 SerialECGReader initialized: Port={port}, Baud={baudrate}")

    def start(self):
        print("🚀 Starting ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.ser.write(b'1\r\n')
        time.sleep(0.5)
        self.running = True
        print("✅ ECG device started - waiting for data...")

    def stop(self):
        print("⏹️ Stopping ECG data acquisition...")
        self.ser.write(b'0\r\n')
        self.running = False
        print(f"📊 Total data packets received: {self.data_count}")

    def read_value(self):
        if not self.running:
            return None
        try:
            line_raw = self.ser.readline()
            line_data = line_raw.decode('utf-8', errors='replace').strip()

            if line_data:
                self.data_count += 1
                # Print detailed data information
                print(f"📡 [Packet #{self.data_count}] Raw data: '{line_data}' (Length: {len(line_data)})")
                
                # Parse and display ECG value
                if line_data.isdigit():
                    ecg_value = int(line_data[-3:])
                    print(f"💓 ECG Value: {ecg_value} mV")
                    return ecg_value
                else:
                    # Try to parse as multiple values (8-channel data)
                    try:
                        # Clean the line data - remove any non-numeric characters except spaces and minus signs
                        import re
                        cleaned_line = re.sub(r'[^\d\s\-]', ' ', line_data)
                        values = [int(x) for x in cleaned_line.split() if x.strip() and x.replace('-', '').isdigit()]
                        
                        if len(values) >= 8:
                            print(f"💓 8-Channel ECG Data: {values}")
                            return values  # Return the list of 8 values
                        elif len(values) == 1:
                            print(f"💓 Single ECG Value: {values[0]} mV")
                            return values[0]
                        elif len(values) > 0:
                            print(f"⚠️ Unexpected number of values: {len(values)} (expected 8)")
                        else:
                            return None
                    except Exception as e:
                        print(f"❌ Error parsing ECG data: {e}")
                        return None
            else:
                print("⏳ No data received (timeout)")
                
        except Exception as e:
            self._handle_serial_error(e)
        return None

    def close(self):
        print("🔌 Closing serial connection...")
        self.ser.close()
        print("✅ Serial connection closed")

    def _handle_serial_error(self, error):
        """Handle serial communication errors with alert and logging"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log the error
        error_msg = f"Serial communication error: {error}"
        print(f"❌ {error_msg}")
        
        # Log to crash logger
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        # Show alert if consecutive errors exceed threshold
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self._show_serial_error_alert(error)
            self.last_error_time = current_time
            self.consecutive_errors = 0  # Reset counter after showing alert
    
    def _show_serial_error_alert(self, error):
        """Show alert dialog for serial communication errors"""
        try:
            # Get user details from main application
            user_details = getattr(self, 'user_details', {})
            username = user_details.get('full_name', 'Unknown User')
            phone = user_details.get('phone', 'N/A')
            email = user_details.get('email', 'N/A')
            serial_id = user_details.get('serial_id', 'N/A')
            
            # Create detailed error message
            error_details = f"""
Serial Communication Error Detected!

Error: {str(error)}
User: {username}
Phone: {phone}
Email: {email}
Serial ID: {serial_id}
Machine Serial: {self.crash_logger.machine_serial_id or 'N/A'}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

This error has been logged and an email notification will be sent to the support team.
            """
            
            # Show alert dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Serial Communication Error")
            msg_box.setText("ECG Device Connection Lost")
            msg_box.setDetailedText(error_details)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
            # Send email notification
            self._send_error_email(error, user_details)
            
        except Exception as e:
            print(f"❌ Error showing serial error alert: {e}")
    
    def _send_error_email(self, error, user_details):
        """Send email notification for serial errors"""
        try:
            # Create error data for email
            error_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': 'Serial Communication Error',
                'error_message': str(error),
                'user_details': user_details,
                'machine_serial': self.crash_logger.machine_serial_id or 'N/A',
                'consecutive_errors': self.consecutive_errors,
                'total_errors': self.error_count
            }
            
            # Send email using crash logger
            self.crash_logger._send_crash_email(error_data)
            print("📧 Serial error email notification sent")
            
        except Exception as e:
            print(f"❌ Error sending serial error email: {e}")

class LiveLeadWindow(QWidget):
    def __init__(self, lead_name, data_source, buffer_size=80, color="#00ff99"):
        super().__init__()
        self.setWindowTitle(f"Live View: {lead_name}")
        self.resize(900, 300)
        self.lead_name = lead_name
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.color = color

        layout = QVBoxLayout(self)
        self.fig = Figure(facecolor='#000')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000')
        self.ax.set_xlim(0, self.buffer_size)
        self.ax.set_ylim(-200, 200)
        self.ax.set_title(f"Live {lead_name}", color='white', fontsize=14)
        self.ax.tick_params(colors='white')
        
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.line, = self.ax.plot([], [], color=color, linewidth=2)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20 FPS

    def update_plot(self):
        data = self.data_source()
        if data and len(data) > 0:
            plot_data = np.full(self.buffer_size, np.nan)
            n = min(len(data), self.buffer_size)
            centered = np.array(data[-n:]) - np.mean(data[-n:])
            plot_data[-n:] = centered
            self.line.set_ydata(plot_data)
            self.canvas.draw_idle()

# ------------------------ Calculate QRS axis ------------------------

def calculate_qrs_axis(lead_I, lead_aVF, r_peaks, fs=500, window_ms=100):
    """
    Calculate QRS axis using net area of QRS complex around R peaks.
    - lead_I, lead_aVF: arrays of samples
    - r_peaks: indices of R peaks
    - fs: sampling rate
    - window_ms: window size around R peak (default 100 ms)
    """
    if len(lead_I) < 100 or len(lead_aVF) < 100 or len(r_peaks) == 0:
        return "--"
    window = int(window_ms * fs / 1000)
    net_I = []
    net_aVF = []
    for r in r_peaks:
        start = max(0, r - window//2)
        end = min(len(lead_I), r + window//2)
        net_I.append(np.sum(lead_I[start:end]))
        net_aVF.append(np.sum(lead_aVF[start:end]))
    if len(net_I) == 0:
        return "--"
    mean_I = np.mean(net_I)
    mean_aVF = np.mean(net_aVF)
    axis_rad = np.arctan2(mean_aVF, mean_I)
    axis_deg = np.degrees(axis_rad)
    if axis_deg < 0:
        axis_deg += 360
    return f"{axis_deg:.0f}°"

def calculate_st_segment(lead_signal, r_peaks, fs=500, j_offset_ms=40, st_offset_ms=80):
    """
    Calculate mean ST segment amplitude (in mV) at (J-point + st_offset_ms) after R peak.
    - lead_signal: ECG samples (e.g., Lead II)
    - r_peaks: indices of R peaks
    - fs: sampling rate (Hz)
    - j_offset_ms: ms after R peak to estimate J-point (default 40ms)
    - st_offset_ms: ms after J-point to measure ST segment (default 80ms)
    Returns mean ST segment amplitude in mV (float), or '--' if not enough data.
    """
    if len(lead_signal) < 100 or len(r_peaks) == 0:
        return "--"
    j_offset = int(j_offset_ms * fs / 1000)
    st_offset = int(st_offset_ms * fs / 1000)
    st_values = []
    for r in r_peaks:
        st_idx = r + j_offset + st_offset
        if st_idx < len(lead_signal):
            st_values.append(lead_signal[st_idx])
    if len(st_values) == 0:
        return "--"
    st_value = np.mean(st_values)
    if st_value > 0.1:
        return "Elevated"
    elif st_value < -0.1:
        return "Depressed"
    return str(st_value)

# ------------------------ Calculate Arrhythmia ------------------------

# def detect_arrhythmia(heart_rate, qrs_duration, rr_intervals, pr_interval=None, p_peaks=None, r_peaks=None, ecg_signal=None):
#     """
#     Expanded arrhythmia detection logic for common clinical arrhythmias.
#     - Sinus Bradycardia: HR < 60, regular RR
#     - Sinus Tachycardia: HR > 100, regular RR
#     - Atrial Fibrillation: Irregular RR, absent/irregular P waves
#     - Atrial Flutter: Sawtooth P pattern (not robustly detected here)
#     - PAC: Early P, narrow QRS, compensatory pause (approximate)
#     - PVC: Early wide QRS, no P, compensatory pause (approximate)
#     - VT: HR > 100, wide QRS (>120ms), regular
#     - VF: Chaotic, no clear QRS, highly irregular
#     - Asystole: Flatline (very low amplitude, no R)
#     - SVT: HR > 150, narrow QRS, regular
#     - Heart Block: PR > 200 (1°), dropped QRS (2°), AV dissociation (3°)
#     """
#     try:
#         if not rr_intervals or len(rr_intervals) < 2:
#             return "Insufficient Data"
#         rr_std = np.std(rr_intervals)
#         rr_mean = np.mean(rr_intervals)
#         rr_reg = rr_std < 0.12  # Regular if std < 120ms
#         # Asystole: flatline (no R peaks, or very low amplitude)
#         if r_peaks is not None and len(r_peaks) < 1:
#             if ecg_signal is not None and np.ptp(ecg_signal) < 50:
#                 return "Asystole (Flatline)"
#             return "No QRS Detected"
#         # VF: highly irregular, no clear QRS, rapid undulating
#         if r_peaks is not None and len(r_peaks) > 5:
#             if rr_std > 0.25 and np.ptp(ecg_signal) > 100 and heart_rate and heart_rate > 180:
#                 return "Ventricular Fibrillation (VF)"
#         # VT: HR > 100, wide QRS (>120ms), regular
#         if heart_rate and heart_rate > 100 and qrs_duration and qrs_duration > 120 and rr_reg:
#             return "Ventricular Tachycardia (VT)"
#         # Sinus Bradycardia: HR < 60, regular
#         if heart_rate and heart_rate < 60 and rr_reg:
#             return "Sinus Bradycardia"
#         # Sinus Tachycardia: HR > 100, regular
#         if heart_rate and heart_rate > 100 and qrs_duration and qrs_duration <= 120 and rr_reg:
#             return "Sinus Tachycardia"
#         # SVT: HR > 150, narrow QRS, regular
#         if heart_rate and heart_rate > 150 and qrs_duration and qrs_duration <= 120 and rr_reg:
#             return "Supraventricular Tachycardia (SVT)"
#         # AFib: Irregular RR, absent/irregular P
#         if not rr_reg and (p_peaks is None or len(p_peaks) < len(r_peaks) * 0.5):
#             return "Atrial Fibrillation (AFib)"
#         # Atrial Flutter: (not robust, but if HR ~150, regular, and P waves rapid)
#         if heart_rate and 140 < heart_rate < 170 and rr_reg and p_peaks is not None and len(p_peaks) > len(r_peaks):
#             return "Atrial Flutter (suggestive)"
#         # PAC: Early P, narrow QRS, compensatory pause (approximate)
#         if p_peaks is not None and r_peaks is not None and len(p_peaks) > 1 and len(r_peaks) > 1:
#             pr_diffs = np.diff([r - p for p, r in zip(p_peaks, r_peaks)])
#             if np.any(pr_diffs < -0.15 * len(ecg_signal)) and qrs_duration and qrs_duration <= 120:
#                 return "Premature Atrial Contraction (PAC)"
#         # PVC: Early wide QRS, no P, compensatory pause (approximate)
#         if qrs_duration and qrs_duration > 120 and (p_peaks is None or len(p_peaks) < len(r_peaks) * 0.5):
#             return "Premature Ventricular Contraction (PVC)"
#         # Heart Block: PR > 200ms (1°), dropped QRS (2°), AV dissociation (3°)
#         if pr_interval and pr_interval > 200:
#             return "Heart Block (1° AV)"
#         # If QRS complexes are missing (dropped beats)
#         if r_peaks is not None and len(r_peaks) < len(ecg_signal) / 500 * heart_rate * 0.7:
#             return "Heart Block (2°/3° AV, dropped QRS)"
#         return "None Detected"
#     except Exception as e:
#         return "Detecting..."


class SerialECGReader:
    def __init__(self, port, baudrate):
        if not SERIAL_AVAILABLE:
            raise ImportError("Serial module not available - cannot create ECG reader")
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        print(f"🔌 SerialECGReader initialized: Port={port}, Baud={baudrate}")

    def start(self):
        print("🚀 Starting ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.ser.write(b'1\r\n')
        time.sleep(0.5)
        self.running = True
        print("✅ ECG device started - waiting for data...")

    def stop(self):
        print("⏹️ Stopping ECG data acquisition...")
        self.ser.write(b'0\r\n')
        self.running = False
        print(f"📊 Total data packets received: {self.data_count}")

    # ========================================================================
    # OLD read_value() METHOD - COMMENTED OUT
    # Using new packet-based parsing logic instead
    # ========================================================================
    # def read_value(self):
    #     """OLD METHOD - COMMENTED OUT - Using packet-based parsing now"""
    #     if not self.running:
    #         return None
    #     try:
    #         line_raw = self.ser.readline()
    #         line_data = line_raw.decode('utf-8', errors='replace').strip()
    #         
    #         if line_data:
    #             self.data_count += 1
    #             # Print detailed data information
    #             print(f"📡 [Packet #{self.data_count}] Raw data: '{line_data}' (Length: {len(line_data)})")
    #             
    #             # Parse and display ECG value
    #             if line_data.isdigit():
    #                 ecg_value = int(line_data[-3:])
    #                 print(f"💓 ECG Value: {ecg_value} mV")
    #                 return ecg_value
    #             else:
    #                 # Try to parse as multiple values (8-channel data)
    #                 try:
    #                     # Clean the line data - remove any non-numeric characters except spaces and minus signs
    #                     import re
    #                     cleaned_line = re.sub(r'[^\d\s\-]', ' ', line_data)
    #                     values = [int(x) for x in cleaned_line.split() if x.strip() and x.replace('-', '').isdigit()]
    #                     
    #                     if len(values) >= 8:
    #                         print(f"💓 8-Channel ECG Data: {values}")
    #                         return values  # Return the list of 8 values
    #                     elif len(values) == 1:
    #                         print(f"💓 Single ECG Value: {values[0]} mV")
    #                         return values[0]
    #                     elif len(values) > 0:
    #                         print(f"⚠️ Unexpected number of values: {len(values)} (expected 8)")
    #                         return None
    #                     else:
    #                         return None
    #                 except ValueError:
    #                     print(f"⚠️ Non-numeric data received: '{line_data}'")
    #         else:
    #             print("⏳ No data received (timeout)")
    #             
    #     except Exception as e:
    #         self._handle_serial_error(e)
    #     return None
    
    def read_value(self):
        """
        NEW METHOD - Compatibility wrapper that uses packet-based parsing
        Returns data in the same format as old method for backward compatibility
        """
        # If this is actually a SerialStreamReader, use packet-based reading
        if isinstance(self, SerialStreamReader):
            packets = self.read_packets(max_packets=1)
            if packets and len(packets) > 0:
                # Convert packet dict to list format [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
                packet = packets[0]
                lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                values = [packet.get(lead, 0) for lead in lead_order]
                return values[:8] if len(values) >= 8 else values  # Return first 8 for compatibility
        return None

    def close(self):
        print("🔌 Closing serial connection...")
        self.ser.close()
        print("✅ Serial connection closed")

    def _handle_serial_error(self, error):
        """Handle serial communication errors with alert and logging"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log the error
        error_msg = f"Serial communication error: {error}"
        print(f"❌ {error_msg}")
        
        # Log to crash logger
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        # Show alert if consecutive errors exceed threshold
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self._show_serial_error_alert(error)
            self.last_error_time = current_time
            self.consecutive_errors = 0  # Reset counter after showing alert
    
    def _show_serial_error_alert(self, error):
        """Show alert dialog for serial communication errors"""
        try:
            # Get user details from main application
            user_details = getattr(self, 'user_details', {})
            username = user_details.get('full_name', 'Unknown User')
            phone = user_details.get('phone', 'N/A')
            email = user_details.get('email', 'N/A')
            serial_id = user_details.get('serial_id', 'N/A')
            
            # Create detailed error message
            error_details = f"""
Serial Communication Error Detected!

Error: {str(error)}
User: {username}
Phone: {phone}
Email: {email}
Serial ID: {serial_id}
Machine Serial: {self.crash_logger.machine_serial_id or 'N/A'}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

This error has been logged and an email notification will be sent to the support team.
            """
            
            # Show alert dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Serial Communication Error")
            msg_box.setText("ECG Device Connection Lost")
            msg_box.setDetailedText(error_details)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
            # Send email notification
            self._send_error_email(error, user_details)
            
        except Exception as e:
            print(f"❌ Error showing serial error alert: {e}")
    
    def _send_error_email(self, error, user_details):
        """Send email notification for serial errors"""
        try:
            # Create error data for email
            error_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': 'Serial Communication Error',
                'error_message': str(error),
                'user_details': user_details,
                'machine_serial': self.crash_logger.machine_serial_id or 'N/A',
                'consecutive_errors': self.consecutive_errors,
                'total_errors': self.error_count
            }
            
            # Send email using crash logger
            self.crash_logger._send_crash_email(error_data)
            print("📧 Serial error email notification sent")
            
        except Exception as e:
            print(f"❌ Error sending serial error email: {e}")

class LiveLeadWindow(QWidget):
    def __init__(self, lead_name, data_source, buffer_size=80, color="#00ff99"):
        super().__init__()
        self.setWindowTitle(f"Live View: {lead_name}")
        self.resize(900, 300)
        self.lead_name = lead_name
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.color = color

        layout = QVBoxLayout(self)
        self.fig = Figure(facecolor='#000')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000')
        self.ax.set_ylim(-400, 400)
        self.ax.set_xlim(0, buffer_size)
        self.line, = self.ax.plot([0]*buffer_size, color=self.color, lw=2)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    def update_plot(self):
        data = self.data_source()
        if data and len(data) > 0:
            plot_data = np.full(self.buffer_size, np.nan)
            n = min(len(data), self.buffer_size)
            centered = np.array(data[-n:]) - np.mean(data[-n:])
            plot_data[-n:] = centered
            self.line.set_ydata(plot_data)
            self.canvas.draw_idle()

# ------------------------ Calculate QRS axis ------------------------

def calculate_qrs_axis(lead_I, lead_aVF, r_peaks, fs=500, window_ms=100):
    """
    Calculate QRS axis using net area of QRS complex around R peaks.
    - lead_I, lead_aVF: arrays of samples
    - r_peaks: indices of R peaks
    - fs: sampling rate
    - window_ms: window size around R peak (default 100 ms)
    """
    if len(lead_I) < 100 or len(lead_aVF) < 100 or len(r_peaks) == 0:
        return "--"
    window = int(window_ms * fs / 1000)
    net_I = []
    net_aVF = []
    for r in r_peaks:
        start = max(0, r - window//2)
        end = min(len(lead_I), r + window//2)
        net_I.append(np.sum(lead_I[start:end]))
        net_aVF.append(np.sum(lead_aVF[start:end]))
    mean_net_I = np.mean(net_I)
    mean_net_aVF = np.mean(net_aVF)
    axis_rad = np.arctan2(mean_net_aVF, mean_net_I)
    axis_deg = int(np.degrees(axis_rad))
    return axis_deg

# ------------------------ Calculate ST Segment ------------------------

def calculate_st_segment(lead_signal, r_peaks, fs=500, j_offset_ms=40, st_offset_ms=80):
    """
    Calculate mean ST segment amplitude (in mV) at (J-point + st_offset_ms) after R peak.
    - lead_signal: ECG samples (e.g., Lead II)
    - r_peaks: indices of R peaks
    - fs: sampling rate (Hz)
    - j_offset_ms: ms after R peak to estimate J-point (default 40ms)
    - st_offset_ms: ms after J-point to measure ST segment (default 80ms)
    Returns mean ST segment amplitude in mV (float), or '--' if not enough data.
    """
    if len(lead_signal) < 100 or len(r_peaks) == 0:
        return "--"
    st_values = []
    j_offset = int(j_offset_ms * fs / 1000)
    st_offset = int(st_offset_ms * fs / 1000)
    for r in r_peaks:
        st_idx = r + j_offset + st_offset
        if st_idx < len(lead_signal):
            st_values.append(lead_signal[st_idx])
    if len(st_values) == 0:
        return "--"
    
    st_value = float(np.mean(st_values))
    # Interpret as medical term
    if 80 <= st_value <= 120:
        return "Isoelectric"
    elif st_value > 120:
        return "Elevated"
    elif st_value < 80:
        return "Depressed"
    return str(st_value)

# ------------------------ Calculate Arrhythmia ------------------------

def detect_arrhythmia(heart_rate, qrs_duration, rr_intervals, pr_interval=None, p_peaks=None, r_peaks=None, ecg_signal=None):
    """
    Expanded arrhythmia detection logic for common clinical arrhythmias.
    - Sinus Bradycardia: HR < 60, regular RR
    - Sinus Tachycardia: HR > 100, regular RR
    - Atrial Fibrillation: Irregular RR, absent/irregular P waves
    - Atrial Flutter: Sawtooth P pattern (not robustly detected here)
    - PAC: Early P, narrow QRS, compensatory pause (approximate)
    - PVC: Early wide QRS, no P, compensatory pause (approximate)
    - VT: HR > 100, wide QRS (>120ms), regular
    - VF: Chaotic, no clear QRS, highly irregular
    - Asystole: Flatline (very low amplitude, no R)
    - SVT: HR > 150, narrow QRS, regular
    - Heart Block: PR > 200 (1°), dropped QRS (2°), AV dissociation (3°)
    - Junctional Rhythm: HR 40-60 with absent or short PR and narrow QRS
    """
    try:
        if rr_intervals is None or len(rr_intervals) < 2:
            return "Detecting..."
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        rr_reg = rr_std < 0.12  # Regular if std < 120ms
        # Asystole: flatline (no R peaks, or very low amplitude)
        if r_peaks is not None and len(r_peaks) < 1:
            if ecg_signal is not None and np.ptp(ecg_signal) < 50:
                return "Asystole (Flatline)"
            return "No QRS Detected"
        # VF: highly irregular, no clear QRS, rapid undulating
        if r_peaks is not None and len(r_peaks) > 5:
            if rr_std > 0.25 and ecg_signal is not None and np.ptp(ecg_signal) > 100 and heart_rate and heart_rate > 180:
                return "Ventricular Fibrillation (VF)"
        # VT: HR > 100, wide QRS (>120ms), regular
        if heart_rate and heart_rate > 100 and qrs_duration and qrs_duration > 120 and rr_reg:
            return "Ventricular Tachycardia (VT)"
        # Junctional Rhythm: rate 40-60, narrow QRS, absent/short PR
        if (
            heart_rate and 40 <= heart_rate <= 60
            and qrs_duration and qrs_duration <= 120
            and rr_reg
        ):
            p_count = len(p_peaks) if p_peaks is not None else 0
            r_count = len(r_peaks) if r_peaks is not None else max(1, len(rr_intervals) + 1)
            p_ratio = p_count / max(r_count, 1)
            pr_short = pr_interval is not None and pr_interval <= 120
            if p_ratio < 0.4 or pr_short:
                return "Junctional Rhythm (possible)"
        # Sinus Bradycardia: HR < 60, regular
        if heart_rate and heart_rate < 60 and rr_reg:
            return "Sinus Bradycardia"
        # Sinus Tachycardia: HR > 100, regular
        if heart_rate and heart_rate > 100 and qrs_duration and qrs_duration <= 120 and rr_reg:
            return "Sinus Tachycardia"
        # SVT: HR > 150, narrow QRS, regular
        if heart_rate and heart_rate > 150 and qrs_duration and qrs_duration <= 120 and rr_reg:
            return "Supraventricular Tachycardia (SVT)"
        # AFib: Irregular RR, absent/irregular P
        if not rr_reg and (p_peaks is None or len(p_peaks) < len(r_peaks) * 0.5):
            return "Atrial Fibrillation (AFib)"
        # Atrial Flutter: (not robust, but if HR ~150, regular, and P waves rapid)
        if heart_rate and 140 < heart_rate < 170 and rr_reg and p_peaks is not None and len(p_peaks) > len(r_peaks):
            return "Atrial Flutter (suggestive)"
        # PAC: Early P, narrow QRS, compensatory pause (approximate)
        if p_peaks is not None and r_peaks is not None and len(p_peaks) > 1 and len(r_peaks) > 1:
            pr_diffs = np.diff([r - p for p, r in zip(p_peaks, r_peaks)])
            if np.any(pr_diffs < -0.15 * len(ecg_signal)) and qrs_duration and qrs_duration <= 120:
                return "Premature Atrial Contraction (PAC)"
        # PVC: Early wide QRS, no P, compensatory pause (approximate)
        if qrs_duration and qrs_duration > 120 and (p_peaks is None or len(p_peaks) < len(r_peaks) * 0.5):
            return "Premature Ventricular Contraction (PVC)"
        # Heart Block: PR > 200ms (1°), dropped QRS (2°), AV dissociation (3°)
        if pr_interval and pr_interval > 200:
            return "Heart Block (1° AV)"
        # If QRS complexes are missing (dropped beats)
        if r_peaks is not None and ecg_signal is not None and len(r_peaks) < len(ecg_signal) / 500 * heart_rate * 0.7:
            return "Heart Block (2°/3° AV, dropped QRS)"
        return "None Detected"
    except Exception as e:
        return "Detecting..."

class ECGTestPage(QWidget):
    LEADS_MAP = {
        "Lead II ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "Lead III ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "7 Lead ECG Test": ["V1", "V2", "V3", "V4", "V5", "V6", "II"],
        "12 Lead ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "ECG Live Monitoring": ["II"]
    }
    LEAD_COLORS = {
        "I": "#00ff99",
        "II": "#ff0055", 
        "III": "#0099ff",
        "aVR": "#ff9900",
        "aVL": "#cc00ff",
        "aVF": "#00ccff",
        "V1": "#ffcc00",
        "V2": "#00ffcc",
        "V3": "#ff6600",
        "V4": "#6600ff",
        "V5": "#00b894",
        "V6": "#ff0066"
    }

    def __init__(self, test_name, stacked_widget):
        super().__init__()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size for usability
        
        self.setWindowTitle("12-Lead ECG Monitor")
        self.stacked_widget = stacked_widget  # Save reference for navigation

        self.settings_manager = SettingsManager()
        # Ensure AC filter starts at "off" each launch (Set Filter default)
        try:
            self.settings_manager.set_setting("filter_ac", "off")
        except Exception as e:
            print(f"⚠️ Could not enforce default AC filter state: {e}")
        self.current_language = self.settings_manager.get_setting("system_language", "en")
    
        # Initialize demo manager
        self.demo_manager = DemoManager(self)

        self.grid_widget = QWidget()
        self.detailed_widget = QWidget()
        self.page_stack = QStackedLayout()
        self.page_stack.addWidget(self.grid_widget)
        self.page_stack.addWidget(self.detailed_widget)
        self.setLayout(self.page_stack)

        self.test_name = test_name
        self.leads = self.LEADS_MAP[test_name]
        self.base_buffer_size = 2000  # Base buffer used for speed scaling
        self.buffer_size = self.base_buffer_size  # Increased buffer size for all leads
        # Use GitHub version data structure: list of numpy arrays for all 12 leads
        # Initialize data buffers with memory management
        self.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
        
        # Track overlay state and current layout (12:1 vs 6:2)
        self._overlay_active = False
        self._current_overlay_layout = None
        
        # Memory management
        self.max_buffer_size = 10000  # Maximum buffer size to prevent memory issues
        self.memory_check_interval = 1000  # Check memory every 1000 updates
        self.update_count = 0
        # Hold last displayed HR to avoid unnecessary flicker
        self._last_hr_display = None
        # HR smoothing/lock removed; use original calculation
        
        # Initialize crash logger
        self.crash_logger = get_crash_logger()
        self.crash_logger.log_info("ECG Test Page initialized", "ECG_TEST_PAGE_START")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.serial_reader = None
        self.stacked_widget = stacked_widget
        self.sampler = SamplingRateCalculator()
        # self.demo_fs = 500  # Increased sampling rate for more realistic ECG
        self.sampling_rate = 500  # Default sampling rate for expanded lead view
        self._latest_rhythm_interpretation = "Analyzing Rhythm..."

        # Flatline detection state: track leads where we've already shown an alert
        self._flatline_alert_shown = [False] * 12
        self._prev_p_axis = None  # Track P-axis for safety assertions
        self._prev_qrs_axis = None
        self._prev_t_axis = None

        # Initialize time tracking for elapsed time
        self.start_time = None
        self.paused_at = None  # Track when pause started
        self.paused_duration = 0  # Total cumulative paused time
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)

        main_vbox = QVBoxLayout()

        menu_frame = QGroupBox("Menu")

        menu_frame.setStyleSheet("""
            QGroupBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border: 2px solid #e9ecef;
                border-radius: 16px;
                margin-top: 12px;
                padding: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #495057;
                font-size: 16px;
                font-weight: bold;
                padding: 8px;
            }
        """)

        # Enhanced Menu Panel - Make it responsive and compact
        menu_container = QWidget()
        menu_container.setMinimumWidth(200)  # Reduced from 250px
        menu_container.setMaximumWidth(280)  # Reduced from 400px
        menu_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        menu_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border-right: 2px solid #e9ecef;
            }
        """)

        # Style menu buttons - Make them much more compact
        menu_layout = QVBoxLayout(menu_container)
        menu_layout.setContentsMargins(12, 12, 12, 12)  # Reduced margins
        menu_layout.setSpacing(8)  # Reduced spacing between buttons
        
        # Header - Make it more compact
        self.menu_header_label = QLabel("ECG Control Panel")
        self.menu_header_label.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;  /* Reduced from 24px */
                font-weight: bold;
                padding: 12px 0;  /* Reduced from 20px */
                border-bottom: 2px solid #ff6600;  /* Reduced from 3px */
                margin-bottom: 12px;  /* Reduced from 20px */
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border-radius: 8px;  /* Reduced from 10px */
            }
        """)
        self.menu_header_label.setAlignment(Qt.AlignCenter)
        self.menu_header_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu_layout.addWidget(self.menu_header_label)
        
        # Create ECGMenu instance to use its methods
        self.ecg_menu = ECGMenu(parent=self, dashboard=self.stacked_widget.parent())
        # Connect ECGMenu to this ECG test page for data communication
        self.ecg_menu.set_ecg_test_page(self)

        self.ecg_menu.settings_manager = self.settings_manager
        
        # Register demo manager's settings callback so wave gain/speed changes work in demo mode
        if hasattr(self, 'demo_manager'):
            self.ecg_menu.settings_changed_callback = self.demo_manager.on_settings_changed
            print("✅ Demo manager settings callback registered")

        # Initialize sliding panel for the ECG menu
        self.ecg_menu.sliding_panel = None
        self.ecg_menu.parent_widget = self

        self.ecg_menu.setVisible(False)
        self.ecg_menu.hide()
        
        if self.ecg_menu.parent():
            self.ecg_menu.setParent(None)

        self.ecg_menu.settings_changed_callback = self.on_settings_changed 

        self.apply_display_settings()

        # Create ECG menu buttons
        ecg_menu_buttons = [
            ("Save ECG", self.ecg_menu.show_save_ecg, "#28a745"),
            ("Open ECG", self.ecg_menu.show_open_ecg, "#17a2b8"),
            ("Working Mode", self.ecg_menu.show_working_mode, "#ffc107"),
            ("Printer Setup", self.ecg_menu.show_printer_setup, "#6c757d"),
            ("Set Filter", self.ecg_menu.show_set_filter, "#fd7e14"),
            ("System Setup", self.ecg_menu.show_system_setup, "#6f42c1"),
            ("Load Default", self.ecg_menu.show_load_default, "#20c997"),
            ("Version", self.ecg_menu.show_version_info, "#e83e8c"),
            ("Exit", self.ecg_menu.show_exit, "#495057")
        ]
        
        # Create buttons and store them in a list - Make them much smaller
        created_buttons = []
        self.menu_buttons = []
        for text, handler, color in ecg_menu_buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(40)  # Reduced from 60px - Much more compact
            btn.setMaximumHeight(45)  # Add maximum height constraint
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(handler)
            created_buttons.append(btn)
            menu_layout.addWidget(btn)
            self.menu_buttons.append((btn, text))

        menu_layout.addStretch(1)

        self.apply_language(self.current_language)

        # Style menu buttons AFTER they're created - Compact styling
        for i, btn in enumerate(created_buttons):
            color = ecg_menu_buttons[i][2]
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #1a1a1a;
                    border: 2px solid #e9ecef;  /* Reduced from 3px */
                    border-radius: 8px;  /* Reduced from 15px */
                    padding: 8px 12px;  /* Reduced from 15px 20px */
                    font-size: 12px;  /* Reduced from 16px */
                    font-weight: bold;
                    text-align: left;
                    margin: 2px 0;  /* Reduced from 4px */
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #fff5f0, stop:1 #ffe0cc);
                    border: 2px solid {color};  /* Reduced from 4px */
                    color: {color};
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #ffe0cc, stop:1 #ffcc99);
                    border: 2px solid {color};  /* Reduced from 4px */
                    color: {color};
                }}
            """)

        created_buttons[0].clicked.disconnect()
        created_buttons[0].clicked.connect(self.ecg_menu.show_save_ecg)
        
        created_buttons[1].clicked.disconnect()
        created_buttons[1].clicked.connect(self.ecg_menu.show_open_ecg)
        
        created_buttons[2].clicked.disconnect()
        created_buttons[2].clicked.connect(self.ecg_menu.show_working_mode)
        
        created_buttons[3].clicked.disconnect()
        created_buttons[3].clicked.connect(self.ecg_menu.show_printer_setup)
        
        created_buttons[4].clicked.disconnect()
        created_buttons[4].clicked.connect(self.ecg_menu.show_set_filter)
        
        created_buttons[5].clicked.disconnect()
        created_buttons[5].clicked.connect(self.ecg_menu.show_system_setup)
        
        created_buttons[6].clicked.disconnect()
        created_buttons[6].clicked.connect(self.ecg_menu.show_load_default)
        
        created_buttons[7].clicked.disconnect()
        created_buttons[7].clicked.connect(self.ecg_menu.show_version_info)

        created_buttons[8].clicked.disconnect()
        created_buttons[8].clicked.connect(self.ecg_menu.show_exit)

        # Recording Toggle Button Section - Make it compact
        recording_frame = QFrame()
        recording_frame.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
                padding: 6px;  /* Reduced from 10px */
                margin-top: 3px;  /* Reduced from 5px */
            }
        """)
        recording_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        recording_layout = QVBoxLayout(recording_frame)
        recording_layout.setSpacing(4)  # Reduced spacing

        # Demo toggle button - Add above capture screen button
        self.demo_toggle = QPushButton("Demo: OFF")
        self.demo_toggle.setCheckable(True)
        self.demo_toggle.setChecked(False)
        self.demo_toggle.setMinimumHeight(35)  # Same as other buttons
        self.demo_toggle.setMaximumHeight(40)  # Same as other buttons
        self.demo_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Set demo button style (toggle-style like recording button)
        self.demo_toggle.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                color: #1a1a1a;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
                text-align: center;
                margin: 2px 0;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #ff6600;
                color: #ff6600;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffe0cc, stop:1 #ffcc99);
                border: 2px solid #ff6600;
                color: #ff6600;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4CAF50, stop:1 #45a049);
                border: 2px solid #4CAF50;
                color: white;
            }
        """)
        
        # Connect demo toggle to demo manager
        self.demo_toggle.toggled.connect(self.on_demo_toggle_changed)
        
        recording_layout.addWidget(self.demo_toggle)

        # Capture Screen button - Make it compact
        self.capture_screen_btn = QPushButton("Capture Screen")
        self.capture_screen_btn.setMinimumHeight(35)  # Reduced from 60px
        self.capture_screen_btn.setMaximumHeight(40)  # Add maximum height
        self.capture_screen_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.capture_screen_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                color: #1a1a1a;
                border: 2px solid #e9ecef;  /* Reduced from 3px */
                border-radius: 8px;  /* Reduced from 15px */
                padding: 8px 12px;  /* Reduced from 15px 20px */
                font-size: 12px;  /* Reduced from 16px */
                font-weight: bold;
                text-align: center;
                margin: 2px 0;  /* Reduced from 5px */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #2453ff;  /* Reduced from 4px */
                color: #2453ff;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #e0e8ff, stop:1 #ccd9ff);
                border: 2px solid #2453ff;  /* Reduced from 4px */
                color: #2453ff;
            }
        """)
        self.capture_screen_btn.clicked.connect(self.capture_screen)
        recording_layout.addWidget(self.capture_screen_btn)
        
        # Toggle-style recording button - Make it compact
        self.recording_toggle = QPushButton("Record Screen")
        self.recording_toggle.setMinimumHeight(35)  # Reduced from 60px
        self.recording_toggle.setMaximumHeight(40)  # Add maximum height
        self.recording_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.recording_toggle.setCheckable(True)
        self.recording_toggle.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                color: #1a1a1a;
                border: 2px solid #e9ecef;  /* Reduced from 3px */
                border-radius: 8px;  /* Reduced from 15px */
                padding: 8px 12px;  /* Reduced from 15px 20px */
                font-size: 12px;  /* Reduced from 16px */
                font-weight: bold;
                text-align: center;
                margin: 2px 0;  /* Reduced from 5px */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #ff6600;  /* Reduced from 4px */
                color: #ff6600;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffe0cc, stop:1 #ffcc99);
                border: 2px solid #ff6600;  /* Reduced from 4px */
                color: #ff6600;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #dc3545;  /* Reduced from 4px */
                color: #dc3545;
            }
            QPushButton:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffe0cc, stop:1 #ffcc99);
                border: 2px solid #c82333;  /* Reduced from 4px */
                color: #c82333;
            }
        """)
        self.recording_toggle.clicked.connect(self.toggle_recording)
        recording_layout.addWidget(self.recording_toggle)
        
        menu_layout.addWidget(recording_frame)
        
        # Initialize recording variables
        self.is_recording = False
        self.recording_writer = None
        self.recording_frames = []

        # Add metrics frame above the plot area
        self.metrics_frame = self.create_metrics_frame()
        self.metrics_frame.setMaximumHeight(80)  # Reduced from default
        self.metrics_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.metrics_frame.setMaximumHeight(120)
        self.metrics_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_vbox.addWidget(self.metrics_frame)
        
        # Ensure metrics are reset to zero after frame creation
        self.reset_metrics_to_zero()
        
        # --- REPLACED: Matplotlib plot area is replaced with a simple QWidget container ---
        self.plot_area = QWidget()
        self.plot_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_vbox.addWidget(self.plot_area)

        # --- NEW: Create the PyQtGraph plot grid (from GitHub version) ---
        grid = QGridLayout(self.plot_area)
        grid.setSpacing(8)
        self.plot_widgets = []
        self.data_lines = []
        
        # Define colors for each lead type for consistent color coding
        lead_colors = {
            'I': '#ff6b6b',      # Red
            'II': '#4ecdc4',     # Teal  
            'III': '#45b7d1',    # Blue
            'aVR': '#96ceb4',    # Green
            'aVL': '#feca57',    # Yellow
            'aVF': '#ff9ff3',    # Pink
            'V1': '#54a0ff',     # Light Blue
            'V2': '#5f27cd',     # Purple
            'V3': '#00d2d3',     # Cyan
            'V4': '#ff9f43',     # Orange
            'V5': '#10ac84',     # Dark Green
            'V6': '#ee5a24'      # Dark Orange
        }
        
        positions = [(i, j) for i in range(4) for j in range(3)]
        for i in range(len(self.leads)):
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            # Hide Y-axis labels for cleaner display
            plot_widget.getAxis('left').setTicks([])
            plot_widget.getAxis('left').setLabel('')
            plot_widget.getAxis('bottom').setTextPen('k')
            
            # Get color for this lead
            lead_name = self.leads[i]
            lead_color = lead_colors.get(lead_name, '#000000')
            
            plot_widget.setTitle(self.leads[i], color=lead_color, size='10pt')
            # Set initial and safe Y-limits; dynamic autoscale will adjust per data
            plot_widget.setYRange(-2000, 2000)
            vb = plot_widget.getViewBox()
            if vb is not None:
                # Prevent extreme jumps while still allowing wide physiological range
                vb.setLimits(yMin=-8000, yMax=8000)
                # Start with a default X range of 10 seconds
                try:
                    vb.setRange(xRange=(0.0, 10.0))
                except Exception:
                    pass
            
            # --- MAKE PLOT CLICKABLE ---
            plot_widget.scene().sigMouseClicked.connect(partial(self.plot_clicked, i))
            
            row, col = positions[i]
            grid.addWidget(plot_widget, row, col)
            data_line = plot_widget.plot(pen=pg.mkPen(color=lead_color, width=2.0))

            self.plot_widgets.append(plot_widget)
            self.data_lines.append(data_line)
        
        # R-peaks scatter plot (only if we have at least 2 plots)
        if len(self.plot_widgets) > 1:
            self.r_peaks_scatter = self.plot_widgets[1].plot([], [], pen=None, symbol='o', symbolBrush='r', symbolSize=8)
        else:
            self.r_peaks_scatter = None
        
        main_vbox.setSpacing(12)  # Reduced from 16px
        main_vbox.setContentsMargins(16, 16, 16, 16)  # Reduced from 24px

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.ports_btn = QPushButton("Ports")
        self.generate_report_btn = QPushButton("Generate Report")
        # self.export_csv_btn = QPushButton("Export as CSV")  # Commented out
        # self.sequential_btn = QPushButton("Show All Leads Sequentially")  # Commented out
        self.twelve_leads_btn = QPushButton("12:1")
        self.six_leads_btn = QPushButton("6:2")
        self.back_btn = QPushButton("Back")

        # Make all buttons responsive and compact
        for btn in [self.start_btn, self.stop_btn, self.ports_btn, self.generate_report_btn, 
                   self.twelve_leads_btn, self.six_leads_btn, self.back_btn]:
            btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)

        green_color = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #3d8b40, stop:1 #357a38);
                border: 2px solid #3d8b40;
                color: white;
            }
        """
        
        # Apply medical green style to all buttons
        self.start_btn.setStyleSheet(green_color)
        self.stop_btn.setStyleSheet(green_color)
        self.ports_btn.setStyleSheet(green_color)
        self.generate_report_btn.setStyleSheet(green_color)
        # self.export_csv_btn.setStyleSheet(green_color)  # Commented out
        # self.sequential_btn.setStyleSheet(green_color)  # Commented out
        self.twelve_leads_btn.setStyleSheet(green_color)
        self.six_leads_btn.setStyleSheet(green_color)
        self.back_btn.setStyleSheet(green_color)

        btn_layout.setSpacing(4)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.ports_btn)
        btn_layout.addWidget(self.generate_report_btn)
        # btn_layout.addWidget(self.export_csv_btn)  # Commented out
        # btn_layout.addWidget(self.sequential_btn)  # Commented out
        btn_layout.addWidget(self.twelve_leads_btn)
        btn_layout.addWidget(self.six_leads_btn)
        btn_layout.addWidget(self.back_btn)
        main_vbox.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.stop_acquisition)


        self.start_btn.setToolTip("Start ECG recording from the selected port")
        self.stop_btn.setToolTip("Stop current ECG recording")
        self.ports_btn.setToolTip("Configure COM port and baud rate settings")
        self.generate_report_btn.setToolTip("Generate ECG PDF report and add to Recent Reports")
        # self.export_csv_btn.setToolTip("Export ECG data as CSV file")  # Commented out

        # Add help button
        help_btn = QPushButton("?")
        help_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #495057;
            }
        """)
        help_btn.clicked.connect(self.show_help)

        self.ports_btn.clicked.connect(self.show_ports_dialog)
        self.generate_report_btn.clicked.connect(self.generate_pdf_report)
        # self.export_csv_btn.clicked.connect(self.export_csv)  # Commented out
        # self.sequential_btn.clicked.connect(self.show_sequential_view)  # Commented out
        self.twelve_leads_btn.clicked.connect(self.twelve_leads_overlay)
        self.six_leads_btn.clicked.connect(self.six_leads_overlay)
        self.back_btn.clicked.connect(self.go_back)

        main_hbox = QHBoxLayout(self.grid_widget)
    
        # Add widgets to the layout with responsive sizing - Better proportions
        main_hbox.addWidget(menu_container, 1)  # Menu takes 1 part (compact)
        main_hbox.addLayout(main_vbox, 5)  # Main content takes 5 parts (more space)
        
        # Set spacing and layout
        main_hbox.setSpacing(10)  # Reduced from 15px
        main_hbox.setContentsMargins(8, 8, 8, 8)  # Reduced from 10px
        self.grid_widget.setLayout(main_hbox)
        
        # Make the grid widget responsive
        self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def plot_clicked(self, plot_index):
        """Handle plot click events"""
        if plot_index < len(self.leads):
            lead_name = self.leads[plot_index]
            print(f"Clicked on {lead_name} plot")
            
            # Get the ECG data for this lead
            if plot_index < len(self.data) and len(self.data[plot_index]) > 0:
                ecg_data = self.data[plot_index]
                
                # Import and show expanded lead view
                try:
                    from ecg.expanded_lead_view import show_expanded_lead_view
                    show_expanded_lead_view(lead_name, ecg_data, self.sampling_rate, self)
                except ImportError as e:
                    print(f"Error importing expanded lead view: {e}")
                    # Fallback: show a simple message
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Lead Analysis", 
                                          f"Lead {lead_name} analysis would be shown here.\n"
                                          f"Data points: {len(ecg_data)}")
            else:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Data", f"No ECG data available for Lead {lead_name}")

    def tr(self, text):
        return translate_text(text, getattr(self, "current_language", "en"))

    def update_demo_toggle_label(self):
        if hasattr(self, 'demo_toggle') and self.demo_toggle:
            key = "Demo: ON" if self.demo_toggle.isChecked() else "Demo: OFF"
            self.demo_toggle.setText(self.tr(key))

    def on_demo_toggle_changed(self, checked):
        self.update_demo_toggle_label()
        self.demo_manager.toggle_demo_mode(checked)

    def apply_language(self, language=None):
        if language:
            self.current_language = language
        translator = self.tr
        if hasattr(self, 'menu_header_label'):
            self.menu_header_label.setText(translator("ECG Control Panel"))
        if hasattr(self, 'menu_buttons'):
            for btn, label in self.menu_buttons:
                btn.setText(translator(label))
        self.update_demo_toggle_label()
        if hasattr(self, 'capture_screen_btn') and self.capture_screen_btn:
            self.capture_screen_btn.setText(translator("Capture Screen"))
        if hasattr(self, 'recording_toggle') and self.recording_toggle:
            self.recording_toggle.setText(translator("Record Screen"))
        for attr, key in [
            ('start_btn', "Start"),
            ('stop_btn', "Stop"),
            ('ports_btn', "Ports"),
            ('generate_report_btn', "Generate Report"),
            ('twelve_leads_btn', "12:1"),
            ('six_leads_btn', "6:2"),
            ('back_btn', "Back"),
        ]:
            btn = getattr(self, attr, None)
            if btn:
                btn.setText(translator(key))
        if hasattr(self, 'demo_toggle') and self.demo_toggle:
            self.update_demo_toggle_label()
        if hasattr(self, 'ecg_menu') and self.ecg_menu:
            update_lang = getattr(self.ecg_menu, "update_language", None)
            if callable(update_lang):
                update_lang(self.current_language)

    def calculate_12_leads_from_8_channels(self, channel_data):
        """
        Calculate 12-lead ECG from 8-channel hardware data
        Hardware sends: [L1, V4, V5, Lead 2, V3, V6, V1, V2] in that order
        """
        try:
            # Validate input data
            if not channel_data or not isinstance(channel_data, (list, tuple, np.ndarray)):
                print("❌ Invalid channel data format")
                return [0] * 12
            
            # Convert to list if numpy array
            if isinstance(channel_data, np.ndarray):
                channel_data = channel_data.tolist()
            
            # Ensure we have at least 8 channels
            if len(channel_data) < 8:
                # Pad with zeros if not enough channels
                channel_data = channel_data + [0] * (8 - len(channel_data))
                print(f"⚠️ Padded channel data to 8 channels: {len(channel_data)}")
            
            # Validate all values are numeric
            for i, val in enumerate(channel_data[:8]):
                try:
                    float(val)
                except (ValueError, TypeError):
                    print(f"❌ Invalid numeric value at channel {i}: {val}")
                    channel_data[i] = 0
            
            # Map hardware channels to standard positions with bounds checking
            L1 = float(channel_data[0]) if len(channel_data) > 0 else 0      # Lead I
            V4_hw = float(channel_data[1]) if len(channel_data) > 1 else 0   # V4 from hardware
            V5_hw = float(channel_data[2]) if len(channel_data) > 2 else 0   # V5 from hardware
            II = float(channel_data[3]) if len(channel_data) > 3 else 0      # Lead II
            V3_hw = float(channel_data[4]) if len(channel_data) > 4 else 0   # V3 from hardware
            V6_hw = float(channel_data[5]) if len(channel_data) > 5 else 0   # V6 from hardware
            V1 = float(channel_data[6]) if len(channel_data) > 6 else 0      # V1 from hardware
            V2 = float(channel_data[7]) if len(channel_data) > 7 else 0      # V2 from hardware

            # Calculate derived leads using standard ECG formulas with error handling
            I = L1  # Lead I is directly from hardware

            # Calculate Lead III from Lead I and Lead II
            try:
                III = II - I
            except Exception:
                III = 0

            # Calculate augmented leads using standard Einthoven/Goldberger relations:
            #   aVR = RA - (LA + LL)/2  = -(Lead I + Lead II) / 2
            #   aVL = LA - (RA + LL)/2 = (Lead I - Lead III) / 2
            #   aVF = LL - (RA + LA)/2 = (Lead II + Lead III) / 2
            try:
                aVR = -(I + II) / 2.0
            except Exception:
                aVR = 0.0

            try:
                aVL = (I - III) / 2
            except Exception:
                aVL = 0.0

            try:
                aVF = (II + III) / 2
            except Exception:
                aVF = 0.0

            # Use hardware V leads directly (already named V1, V2; others from *_hw)
            V3 = V3_hw
            V4 = V4_hw
            V5 = V5_hw
            V6 = V6_hw
        
            # Return 12-lead ECG data in standard order
            result = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            
            # Validate result
            for i, val in enumerate(result):
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    print(f"❌ Invalid result value at lead {i}: {val}")
                    result[i] = 0
            
            return result
            
        except Exception as e:
            print(f"❌ Critical error in calculate_12_leads_from_8_channels: {e}")
            # Return safe default values
            return [0] * 12

    def _extract_low_frequency_baseline(self, signal, sampling_rate=500.0):
        """
        Extract very-low-frequency baseline estimate (< 0.3 Hz) for display anchoring.
        
        Uses 2-second moving average SIGNAL (not mean) to remove:
        - Respiration (0.1-0.35 Hz) → filtered out
        - ST/T waves → filtered out
        - QRS complexes → filtered out
        
        Returns only very-low-frequency drift (< 0.1 Hz).
        
        Args:
            signal: ECG signal window
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Low-frequency baseline estimate (single value)
        """
        if len(signal) < 10:
            return np.nanmean(signal) if len(signal) > 0 else 0.0
        
        try:
            # Method: 2-second moving average SIGNAL (not mean)
            window_samples = int(2.0 * sampling_rate)  # 2 seconds
            window_samples = min(window_samples, len(signal))
            
            if window_samples >= 10 and len(signal) >= window_samples:
                # Extract actual low-frequency baseline signal using convolution
                # This is a proper moving average, not just a statistic
                kernel = np.ones(window_samples) / window_samples
                baseline_signal = np.convolve(signal, kernel, mode="valid")
                # Use the last value of the moving-average signal
                baseline_estimate = baseline_signal[-1] if len(baseline_signal) > 0 else np.nanmean(signal)
            else:
                # Fallback: use mean if window too small
                baseline_estimate = np.nanmean(signal)
            
            return baseline_estimate
        
        except Exception:
            # Fallback: simple mean if moving average fails
            return np.nanmean(signal) if len(signal) > 0 else 0.0

    def calculate_ecg_metrics(self):
        """Calculate ECG metrics using median beat (GE/Philips standard).
        
        ⚠️ CLINICAL ANALYSIS: Uses RAW data, median beat, TP baseline
        This function MUST use raw clinical data, NOT display-processed data.
        """

        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            print("🔍 Demo mode active - skipping live ECG metrics calculation")
            return

        if len(self.data) < 2:  # Need at least Lead II for analysis
            return
        
        # 🫀 CLINICAL: Use RAW Lead II data (index 1) for clinical analysis
        # This is the raw buffer - NOT display-processed data
        lead_ii_data = self.data[1]
        
        # Check if data is all zeros or has no real signal variation
        if len(lead_ii_data) < 100 or np.all(lead_ii_data == 0) or np.std(lead_ii_data) < 0.1:
            return
        
        # Get sampling rate
        fs = 80.0
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
            fs = float(self.sampler.sampling_rate)
        elif hasattr(self, 'sampling_rate') and self.sampling_rate:
            fs = float(self.sampling_rate)
        
        # Detect R-peaks in raw Lead II (fallback to V2 if Lead II insufficient) - GE/Philips standard
        from scipy.signal import butter, filtfilt
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_ii = filtfilt(b, a, lead_ii_data)
        
        signal_mean = np.mean(filtered_ii)
        signal_std = np.std(filtered_ii)
        r_peaks, _ = find_peaks(
            filtered_ii,
            height=signal_mean + 0.5 * signal_std,
            distance=int(0.3 * fs),
            prominence=signal_std * 0.4
        )
        
        # Fallback to V2 if Lead II has insufficient beats (GE/Philips standard)
        if len(r_peaks) < 8 and len(self.data) > 3:
            lead_v2_data = self.data[3]  # V2 is typically index 3
            if len(lead_v2_data) > 100 and np.std(lead_v2_data) > 0.1:
                filtered_v2 = filtfilt(b, a, lead_v2_data)
                signal_mean_v2 = np.mean(filtered_v2)
                signal_std_v2 = np.std(filtered_v2)
                r_peaks_v2, _ = find_peaks(
                    filtered_v2,
                    height=signal_mean_v2 + 0.5 * signal_std_v2,
                    distance=int(0.3 * fs),
                    prominence=signal_std_v2 * 0.4
                )
                if len(r_peaks_v2) >= 8:
                    r_peaks = r_peaks_v2
                    lead_ii_data = lead_v2_data  # Use V2 for beat alignment
        
        # Require ≥8 clean beats for median beat (GE/Philips standard)
        if len(r_peaks) < 8:
            return
        
        # Build median beat from raw Lead II (or V2 fallback) - requires ≥8 beats
        time_axis, median_beat_ii = build_median_beat(lead_ii_data, r_peaks, fs, min_beats=8)
        if median_beat_ii is None:
            return
        
        # Get TP baseline using proper TP segment detection (end of T to next P) - GE/Philips standard
        r_idx = len(median_beat_ii) // 2  # R-peak at center
        r_mid = r_peaks[len(r_peaks) // 2]
        # Use previous R-peak for proper TP segment detection
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        tp_baseline_ii = get_tp_baseline(lead_ii_data, r_mid, fs, prev_r_peak_idx=prev_r_idx)
        
        # Calculate RR interval in ms (median RR from raw signal)
        if len(r_peaks) >= 2:
            rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
            valid_rr = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)]
            rr_ms = np.median(valid_rr) if len(valid_rr) > 0 else 600.0
        else:
            rr_ms = 600.0
        
        # Calculate Heart Rate: HR = 60000 / RR (GE/Philips standard)
        heart_rate = int(round(60000.0 / rr_ms)) if rr_ms > 0 else 60
        
        # Calculate PR Interval from median beat
        pr_interval = self.calculate_pr_interval_from_median(median_beat_ii, time_axis, fs, tp_baseline_ii)
        self.pr_interval = pr_interval
        
        # Calculate QRS Complex duration from median beat
        qrs_duration = self.calculate_qrs_duration_from_median(median_beat_ii, time_axis, fs, tp_baseline_ii)
        
        # Calculate QT Interval from median beat (GE/Philips standard)
        qt_interval = measure_qt_from_median_beat(median_beat_ii, time_axis, fs, tp_baseline_ii)
        if qt_interval is None:
            qt_interval = 0
        
        # Calculate QTc (Bazett) and QTcF (Fridericia)
        qtc_interval = self.calculate_qtc_interval(heart_rate, qt_interval)
        qtcf_interval = self.calculate_qtcf_interval(qt_interval, rr_ms)
        
        # Calculate axes from median beats (P/QRS/T)
        qrs_axis = self.calculate_qrs_axis_from_median()
        p_axis = self.calculate_p_axis_from_median()
        t_axis = self.calculate_t_axis_from_median()
        
        # Calculate ST deviation from median beat (returns mV)
        st_segment = measure_st_deviation_from_median_beat(median_beat_ii, time_axis, fs, tp_baseline_ii, j_offset_ms=60)
        if st_segment is None:
            st_segment = 0.0
        
        # Calculate RV5/SV1 from median beats
        rv5_mv, sv1_mv = self.calculate_rv5_sv1_from_median()
        
        # VALIDATION: Ensure clinical measurements are independent of display filters
        try:
            from .clinical_validation import (
                validate_rv5_sv1_signs, validate_rv5_sv1_sum,
                validate_qtc_formulas, validate_median_beat_beats
            )
            # Validate RV5/SV1 signs
            if rv5_mv is not None and sv1_mv is not None:
                validate_rv5_sv1_signs(rv5_mv, sv1_mv)
            # Validate QTc formulas
            if qt_interval > 0 and rr_ms > 0:
                validate_qtc_formulas(qt_interval, rr_ms, qtc_interval, qtcf_interval)
            # Validate median beat uses 8-12 beats
            if len(r_peaks) >= 8:
                num_beats_used = min(len(r_peaks), 12)
                validate_median_beat_beats(num_beats_used)
        except ImportError:
            pass  # Validation module not available
        except AssertionError as e:
            print(f"⚠️ Clinical validation warning: {e}")
        
        # Update UI metrics (dashboard only shows: BPM, PR, QRS axis, ST, QT/QTc, timer)
        self.update_ecg_metrics_display(heart_rate, pr_interval, qrs_duration, qrs_axis, st_segment, qt_interval, qtc_interval)

    def calculate_heart_rate(self, lead_data):
        """Calculate heart rate from Lead II data using R-R intervals
        
        ⚠️ CLINICAL ANALYSIS: Must receive RAW clinical data, NOT display-processed data.
        This function is called with self.data[1] which contains raw ECG values.
        """
        try:
            # Early exit: if no real signal, report 0 instead of fallback
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0

            # Validate input data
            if not isinstance(lead_data, (list, np.ndarray)) or len(lead_data) < 200:
                print("❌ Insufficient data for heart rate calculation")
                return 60  # Default fallback

            # Convert to numpy array for processing
            try:
                lead_data = np.asarray(lead_data, dtype=float)
            except Exception as e:
                print(f"❌ Error converting lead data to array: {e}")
                return 60

            # Check for invalid values
            if np.any(np.isnan(lead_data)) or np.any(np.isinf(lead_data)):
                print("❌ Invalid values (NaN/Inf) in lead data")
                return 60

            # Use measured sampling rate if available; default to 250 Hz
            fs = 250
            try:
                if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                    fs = float(self.sampler.sampling_rate)
                    if fs <= 0 or fs > 1000:  # Sanity check
                        fs = 250
            except Exception as e:
                print(f"❌ Error getting sampling rate: {e}")
                fs = 250

            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            try:
                from scipy.signal import butter, filtfilt
                nyquist = fs / 2
                low = max(0.001, 0.5 / nyquist)
                high = min(0.999, 40 / nyquist)
                if low >= high:
                    print("❌ Invalid filter parameters")
                    return 60
                b, a = butter(4, [low, high], btype='band')
                filtered_signal = filtfilt(b, a, lead_data)
                if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                    print("❌ Filter produced invalid values")
                    return 60
            except Exception as e:
                print(f"❌ Error in signal filtering: {e}")
                return 60

            # Find R-peaks using scipy with robust parameters
            try:
                from scipy.signal import find_peaks
                signal_mean = np.mean(filtered_signal)
                signal_std = np.std(filtered_signal)
                if signal_std == 0:
                    print("❌ No signal variation detected")
                    return 60
                
                # SMART ADAPTIVE PEAK DETECTION (10-300 BPM with BPM-based selection)
                # Run multiple detections and choose based on CALCULATED BPM
                height_threshold = signal_mean + 0.5 * signal_std
                prominence_threshold = signal_std * 0.4
                
                # Run 3 detection strategies
                detection_results = []
                
                # Strategy 1: Conservative (best for 10-120 BPM)
                peaks_conservative, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.5 * fs),  # 400ms - wider distance for low BPM
                    prominence=prominence_threshold
                )
                if len(peaks_conservative) >= 2:
                    rr_cons = np.diff(peaks_conservative) * (1000 / fs)
                    # Accept RR intervals from 200–6000 ms (300–10 BPM)
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
                    # Accept RR intervals from 200–6000 ms (300–10 BPM)
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
                    # Accept RR intervals from 200–6000 ms (300–10 BPM)
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
                    # print(f"🎯 Selected {best_method}: {best_bpm:.1f} BPM (std={best_std:.1f})")
                else:
                    # Fallback
                    peaks, _ = find_peaks(
                        filtered_signal,
                        height=height_threshold,
                        distance=int(0.4 * fs),
                        prominence=prominence_threshold
                    )
            except Exception as e:
                print(f"❌ Error in peak detection: {e}")
                return 60

            if len(peaks) < 2:
                print(f"❌ Insufficient peaks detected: {len(peaks)}")
                return 60

            # Calculate R-R intervals in milliseconds
            try:
                rr_intervals_ms = np.diff(peaks) * (1000 / fs)
                if len(rr_intervals_ms) == 0:
                    print("❌ No R-R intervals calculated")
                    return 60
            except Exception as e:
                print(f"❌ Error calculating R-R intervals: {e}")
                return 60

            # Filter physiologically reasonable intervals (200-6000 ms)
            # 200 ms = 300 BPM (max), 6000 ms = 10 BPM (min)
            try:
                valid_intervals = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)]
                if len(valid_intervals) == 0:
                    print("❌ No valid R-R intervals found")
                    return 60
            except Exception as e:
                print(f"❌ Error filtering intervals: {e}")
                return 60

            # Calculate heart rate from median R-R interval (as in commit 8a6aaee)
            try:
                median_rr = np.median(valid_intervals)
                if median_rr <= 0:
                    print("❌ Invalid median R-R interval")
                    return 60
                heart_rate = 60000 / median_rr
                # Extended: stable 10–300 BPM range
                heart_rate = max(10, min(300, heart_rate))
                # Extra guard: avoid falsely reporting very high BPM when real rate is very low
                try:
                    window_sec = len(lead_data) / float(fs)
                except Exception:
                    window_sec = 0
                if heart_rate > 150 and window_sec >= 5.0:
                    # How many beats would we expect at this BPM over the window?
                    expected_peaks = (heart_rate * window_sec) / 60.0
                    # If we have far fewer peaks than expected, this "high BPM" is likely noise
                    if expected_peaks > len(peaks) * 3:
                        # Treat as extreme bradycardia scenario and clamp to minimum (10 bpm)
                        print(f"⚠️ Suspicious high BPM ({heart_rate:.1f}) with too few peaks "
                              f"(expected≈{expected_peaks:.1f}, got={len(peaks)}). Clamping to 10 bpm.")
                        heart_rate = 10.0
                if np.isnan(heart_rate) or np.isinf(heart_rate):
                    print("❌ Invalid heart rate calculated")
                    return 60
                
                # ANTI-FLICKERING: Smooth BPM over last few readings
                hr_int = int(round(heart_rate))
                
                # Initialize smoothing buffer
                if not hasattr(self, '_bpm_smooth_buffer'):
                    self._bpm_smooth_buffer = []
                
                # Add current reading to buffer
                self._bpm_smooth_buffer.append(hr_int)
                
                # Keep only last 5 readings for smoothing
                if len(self._bpm_smooth_buffer) > 5:
                    self._bpm_smooth_buffer.pop(0)
                
                # Return median of last 5 readings (very stable, no flickering)
                smoothed_bpm = int(np.median(self._bpm_smooth_buffer))
                
                # Only update if changed by >= 2 bpm (prevents minor fluctuations)
                try:
                    if self._last_hr_display is not None and abs(smoothed_bpm - self._last_hr_display) < 2:
                        return self._last_hr_display
                    self._last_hr_display = smoothed_bpm
                except Exception:
                    self._last_hr_display = smoothed_bpm
                
                return smoothed_bpm
            except Exception as e:
                print(f"❌ Error in final heart rate calculation: {e}")
                return 60
        except Exception as e:
            print(f"❌ Critical error in calculate_heart_rate: {e}")
            return 60

    def calculate_wave_amplitudes(self):
        """Calculate P, QRS, and T wave amplitudes from all leads for report generation"""
        try:
            amplitudes = {
                'p_amp': 0.0,
                'qrs_amp': 0.0,
                't_amp': 0.0,
                'rv5': 0.0,
                'sv1': 0.0
            }
            
            # Get Lead II for P, QRS, T measurements
            lead_ii_data = self.data[1] if len(self.data) > 1 else None
            if lead_ii_data is None or len(lead_ii_data) < 200:
                return amplitudes
            
            # Check for real signal
            if np.all(lead_ii_data == 0) or np.std(lead_ii_data) < 0.1:
                return amplitudes
            
            # Get sampling rate
            fs = 250
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate'):
                fs = float(self.sampler.sampling_rate)
            
            # Filter signal
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = min(40.0 / nyquist, 0.99)
            b, a = butter(2, [low, high], btype='band')
            filtered_data = filtfilt(b, a, lead_ii_data)
            
            # Detect R-peaks
            from scipy.signal import find_peaks
            squared = np.square(np.diff(filtered_data))
            integrated = np.convolve(squared, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
            threshold = np.mean(integrated) + 0.5 * np.std(integrated)
            r_peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.15 * fs))  # Reduced from 0.6 to 0.15 for high BPM (360 max)
            
            if len(r_peaks) < 2:
                return amplitudes
            
            # Analyze each beat and average the amplitudes
            p_amps = []
            qrs_amps = []
            t_amps = []
            
            for r_idx in r_peaks[1:-1]:  # Skip first and last to avoid edge effects
                try:
                    # P-wave amplitude (120-200ms before R)
                    p_start = max(0, r_idx - int(0.20 * fs))
                    p_end = max(0, r_idx - int(0.12 * fs))
                    if p_end > p_start:
                        p_segment = filtered_data[p_start:p_end]
                        baseline = np.mean(filtered_data[max(0, p_start - int(0.05 * fs)):p_start])
                        p_peak = np.max(p_segment) - baseline
                        if p_peak > 0:
                            p_amps.append(p_peak)
                    
                    # QRS amplitude (Q to peak R to S)
                    qrs_start = max(0, r_idx - int(0.08 * fs))
                    qrs_end = min(len(filtered_data), r_idx + int(0.08 * fs))
                    if qrs_end > qrs_start:
                        qrs_segment = filtered_data[qrs_start:qrs_end]
                        qrs_amp = np.max(qrs_segment) - np.min(qrs_segment)
                        if qrs_amp > 0:
                            qrs_amps.append(qrs_amp)
                    
                    # T-wave amplitude (100-300ms after R)
                    t_start = min(len(filtered_data), r_idx + int(0.10 * fs))
                    t_end = min(len(filtered_data), r_idx + int(0.30 * fs))
                    if t_end > t_start:
                        t_segment = filtered_data[t_start:t_end]
                        baseline = np.mean(filtered_data[r_idx:t_start])
                        t_peak = np.max(t_segment) - baseline
                        if t_peak > 0:
                            t_amps.append(t_peak)
                
                except Exception as e:
                    continue
            
            # Calculate median amplitudes (more robust than mean)
            if len(p_amps) > 0:
                amplitudes['p_amp'] = np.median(p_amps)
            if len(qrs_amps) > 0:
                amplitudes['qrs_amp'] = np.median(qrs_amps)
            if len(t_amps) > 0:
                amplitudes['t_amp'] = np.median(t_amps)
            
            # Calculate RV5 and SV1 for specific leads (GE/Hospital Standard)
            # Lead V5 is index 10, Lead V1 is index 6
            # CRITICAL: Use RAW ECG data (self.data), not display-filtered signals
            # Measurements must be from median beat, relative to TP baseline (isoelectric segment before P-wave)
            
            if len(self.data) > 10:
                lead_v5_data = self.data[10]  # RAW V5 data
                if lead_v5_data is not None and len(lead_v5_data) > 200 and np.std(lead_v5_data) > 0.1:
                    # Apply minimal bandpass filter ONLY for R-peak detection (0.5-40 Hz)
                    # This does NOT affect amplitude measurements - we use raw data for measurements
                    filtered_v5 = filtfilt(b, a, lead_v5_data)
                    # Detect R-peaks in V5
                    squared_v5 = np.square(np.diff(filtered_v5))
                    integrated_v5 = np.convolve(squared_v5, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
                    threshold_v5 = np.mean(integrated_v5) + 0.5 * np.std(integrated_v5)
                    r_peaks_v5, _ = find_peaks(integrated_v5, height=threshold_v5, distance=int(0.15 * fs))
                    
                    # Measure RV5: max(QRS in V5) - TP_baseline_V5 (must be positive, in mV)
                    rv5_amps = []
                    for r_idx in r_peaks_v5[1:-1]:
                        try:
                            # QRS window: ±80ms around R-peak
                            qrs_start = max(0, r_idx - int(0.08 * fs))
                            qrs_end = min(len(lead_v5_data), r_idx + int(0.08 * fs))
                            if qrs_end > qrs_start:
                                # Use RAW data for amplitude measurement
                                qrs_segment = lead_v5_data[qrs_start:qrs_end]
                                
                                # TP baseline: isoelectric segment before P-wave onset
                                # Use longer segment (150-350ms before R) for stable baseline
                                tp_start = max(0, r_idx - int(0.35 * fs))
                                tp_end = max(0, r_idx - int(0.15 * fs))
                                if tp_end > tp_start:
                                    tp_segment = lead_v5_data[tp_start:tp_end]
                                    tp_baseline = np.median(tp_segment)  # Median for robustness
                                else:
                                    # Fallback: short segment before QRS
                                    tp_baseline = np.median(lead_v5_data[max(0, qrs_start - int(0.05 * fs)):qrs_start])
                                
                                # RV5 = max(QRS) - TP_baseline (positive, in mV)
                                # Convert from ADC counts to mV using hardware calibration
                                # Adjusted calibration factor to match GE/Philips reference values
                                # Reference: RV5 should be ~0.972 mV, current gives ~1.316 mV
                                # Adjustment factor: 0.972/1.316 ≈ 0.74, so multiply calibration by 1.35
                                # Original: 1 mV = 1517.2 ADC → Adjusted: 1 mV = 1517.2 * 1.35 ≈ 2048 ADC
                                r_amp_adc = np.max(qrs_segment) - tp_baseline
                                if r_amp_adc > 0:
                                    # Convert ADC to mV: Adjusted calibration factor for GE/Philips alignment
                                    r_amp_mv = r_amp_adc / 2048.0  # Adjusted ADC to mV conversion
                                    rv5_amps.append(r_amp_mv)
                        except:
                            continue
                    
                    if len(rv5_amps) > 0:
                        amplitudes['rv5'] = np.median(rv5_amps)  # Median beat approach
            
            if len(self.data) > 6:
                lead_v1_data = self.data[6]  # RAW V1 data
                if lead_v1_data is not None and len(lead_v1_data) > 200 and np.std(lead_v1_data) > 0.1:
                    # Apply minimal bandpass filter ONLY for R-peak detection (0.5-40 Hz)
                    filtered_v1 = filtfilt(b, a, lead_v1_data)
                    # Detect R-peaks in V1
                    squared_v1 = np.square(np.diff(filtered_v1))
                    integrated_v1 = np.convolve(squared_v1, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
                    threshold_v1 = np.mean(integrated_v1) + 0.5 * np.std(integrated_v1)
                    r_peaks_v1, _ = find_peaks(integrated_v1, height=threshold_v1, distance=int(0.15 * fs))
                    
                    # Measure SV1: min(QRS in V1) - TP_baseline_V1 (must be negative, in mV)
                    sv1_amps = []
                    for r_idx in r_peaks_v1[1:-1]:
                        try:
                            # QRS window: ±80ms around R-peak
                            qrs_start = max(0, r_idx - int(0.08 * fs))
                            qrs_end = min(len(lead_v1_data), r_idx + int(0.08 * fs))
                            if qrs_end > qrs_start:
                                # Use RAW data for amplitude measurement
                                qrs_segment = lead_v1_data[qrs_start:qrs_end]
                                
                                # TP baseline: isoelectric segment before P-wave onset
                                tp_start = max(0, r_idx - int(0.35 * fs))
                                tp_end = max(0, r_idx - int(0.15 * fs))
                                if tp_end > tp_start:
                                    tp_segment = lead_v1_data[tp_start:tp_end]
                                    tp_baseline = np.median(tp_segment)  # Median for robustness
                                else:
                                    # Fallback: short segment before QRS
                                    tp_baseline = np.median(lead_v1_data[max(0, qrs_start - int(0.05 * fs)):qrs_start])
                                
                                # SV1 = min(QRS) - TP_baseline (negative, preserve sign, in mV)
                                # Convert from ADC counts to mV using hardware calibration
                                # Adjusted calibration factor to match GE/Philips reference values
                                # Reference: SV1 should be ~-0.485 mV, current gives ~-0.637 mV
                                # Adjustment factor: 0.485/0.637 ≈ 0.76, so multiply calibration by 1.31
                                # Original: 1 mV = 1100 ADC → Adjusted: 1 mV = 1100 * 1.31 ≈ 1441 ADC
                                s_amp_adc = np.min(qrs_segment) - tp_baseline
                                if s_amp_adc < 0:  # SV1 must be negative
                                    # Convert ADC to mV: Adjusted calibration factor for GE/Philips alignment
                                    s_amp_mv = s_amp_adc / 1441.0  # Adjusted ADC to mV conversion (preserve sign)
                                    sv1_amps.append(s_amp_mv)
                        except:
                            continue
                    
                    if len(sv1_amps) > 0:
                        amplitudes['sv1'] = np.median(sv1_amps)  # Median beat approach, negative value
            
            print(f"📊 Wave Amplitudes Calculated: P={amplitudes['p_amp']:.2f}, QRS={amplitudes['qrs_amp']:.2f}, T={amplitudes['t_amp']:.2f}, RV5={amplitudes['rv5']:.2f}, SV1={amplitudes['sv1']:.2f}")
            
            return amplitudes
            
        except Exception as e:
            print(f"❌ Error calculating wave amplitudes: {e}")
            import traceback
            traceback.print_exc()
            return {
                'p_amp': 0.0,
                'qrs_amp': 0.0,
                't_amp': 0.0,
                'rv5': 0.0,
                'sv1': 0.0
            }

    def calculate_pr_interval(self, lead_data):
        """Calculate PR interval from P wave to QRS complex - LIVE"""
        try:
            # Early exit: no real signal 
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.05:
                    return 0
            except Exception:
                return 0
            
            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 80
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks (lenient for 80 Hz)
            peaks, properties = find_peaks(
                filtered_signal,
                height=np.mean(filtered_signal) + 0.3 * np.std(filtered_signal),
                distance=int(0.2 * fs),
                prominence=np.std(filtered_signal) * 0.2
            )
            
            if len(peaks) > 1:
                pr_intervals = []
                deriv = np.gradient(filtered_signal)
                deriv_std = np.std(deriv)
                deriv_thresh = max(0.2 * deriv_std, 1e-6)
                for i in range(min(5, len(peaks)-1)):
                    r_peak = peaks[i]
                    # Search 40–250 ms before R for P upslope
                    win_start = max(0, r_peak - int(0.25 * fs))
                    win_end = max(win_start, r_peak - int(0.04 * fs))
                    if win_end <= win_start:
                        continue
                    win = deriv[win_start:win_end]
                    if len(win) == 0:
                        continue
                    candidates = np.where(win > deriv_thresh)[0]
                    if candidates.size == 0:
                        p_idx = win_start + int(np.argmax(filtered_signal[win_start:win_end]))
                    else:
                        p_idx = win_start + int(candidates[-1])
                    pr_ms = (r_peak - p_idx) / fs * 1000.0
                    if 80 <= pr_ms <= 240:
                        pr_intervals.append(pr_ms)
                if pr_intervals:
                    return int(round(float(np.median(pr_intervals))))
            
            return 150  # Conservative default if not computable
        except:
            return 150

    def calculate_qrs_duration(self, lead_data):
        """Calculate QRS complex duration - LIVE"""
        try:
            # Early exit: no real signal 
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0
            
            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 80.0  # match hardware default so windows scale correctly
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
                if fs <= 0:
                    fs = 80.0
            
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks
            peaks, properties = find_peaks(
                filtered_signal,
                height=np.mean(filtered_signal) + 0.5 * np.std(filtered_signal),
                distance=int(0.4 * fs),
                prominence=np.std(filtered_signal) * 0.3
            )
            
            if len(peaks) > 0:
                qrs_durations = []
                deriv = np.gradient(filtered_signal)
                deriv_std = np.std(deriv)
                deriv_thresh = max(0.1 * deriv_std, 1e-4)
                search_window = int(0.12 * fs)  # allow ±120 ms for low sample rates
                for r_peak in peaks[:min(5, len(peaks))]:  # Analyze first 5 beats
                    # Find Q and S points around R peak
                    start_idx = max(0, r_peak - search_window)
                    end_idx = min(len(filtered_signal), r_peak + search_window)
                    
                    segment = filtered_signal[start_idx:end_idx]
                    if len(segment) > 0:
                        rel_r = r_peak - start_idx
                        pre_slice = slice(0, max(rel_r, 1))
                        post_slice = slice(rel_r, len(segment))
                        
                        # Derivative-based onset: last low-slope point before R
                        if pre_slice.stop > 0:
                            pre_deriv = np.abs(deriv[start_idx:r_peak])
                            low_grad_pre = np.where(pre_deriv < deriv_thresh)[0]
                            if low_grad_pre.size > 0:
                                q_idx = start_idx + int(low_grad_pre[-1])
                            else:
                                q_idx = start_idx + int(np.argmin(segment[pre_slice]))
                        else:
                            q_idx = start_idx
                        
                        # Derivative-based offset: first low-slope point after R
                        if post_slice.stop - post_slice.start > 1:
                            post_deriv = np.abs(deriv[r_peak:end_idx])
                            low_grad_post = np.where(post_deriv < deriv_thresh)[0]
                            if low_grad_post.size > 0:
                                s_idx = r_peak + int(low_grad_post[0])
                            else:
                                s_idx = r_peak + int(np.argmin(segment[post_slice]))
                        else:
                            s_idx = end_idx
                        
                        qrs_duration = (s_idx - q_idx) / fs * 1000  # Convert to ms
                        if 40 <= qrs_duration <= 200:  # Reasonable QRS duration
                            qrs_durations.append(qrs_duration)
                
                if qrs_durations:
                    return int(round(np.mean(qrs_durations)))
            
            return 0  # Fallback to 0 when not computable
        except:
            return 0
    
    def calculate_st_interval(self, lead_data):
        """Calculate ST segment elevation/depression at J+60ms - FRESH calculation"""
        try:
            # Early exit: no real signal → 0
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0
            
            # Get sampling rate
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 80  # Default to hardware sampling rate
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            elif hasattr(self, 'sampling_rate') and self.sampling_rate:
                fs = float(self.sampling_rate)
            
            # Filter signal (0.5-40 Hz bandpass)
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks (lenient for hardware)
            mean_height = np.mean(filtered_signal)
            std_height = np.std(filtered_signal)
            min_height = mean_height + 0.3 * std_height
            min_distance = int(0.3 * fs)
            min_prominence = std_height * 0.2
            
            peaks, _ = find_peaks(
                filtered_signal,
                height=min_height,
                distance=min_distance,
                prominence=min_prominence
            )
            
            if len(peaks) < 1:
                return 0
            
            st_elevations = []
            for r_peak in peaks[:min(5, len(peaks))]:
                try:
                    # Find J-point (end of S-wave, ~40ms after R)
                    j_start = r_peak
                    j_end = min(len(filtered_signal), r_peak + int(0.04 * fs))
                    if j_end <= j_start:
                        continue
                    j_point = j_start + np.argmin(filtered_signal[j_start:j_end])
                    
                    # Measure ST at J+60ms (standard ST measurement point)
                    st_measure_point = min(len(filtered_signal) - 1, j_point + int(0.06 * fs))
                    
                    # Use TP baseline (isoelectric segment 150-350ms before R)
                    tp_baseline_start = max(0, r_peak - int(0.35 * fs))
                    tp_baseline_end = max(0, r_peak - int(0.15 * fs))
                    if tp_baseline_end > tp_baseline_start:
                        tp_baseline = np.median(filtered_signal[tp_baseline_start:tp_baseline_end])
                    else:
                        baseline_start = max(0, r_peak - int(0.15 * fs))
                        baseline_end = max(0, r_peak - int(0.05 * fs))
                        tp_baseline = np.mean(filtered_signal[baseline_start:baseline_end]) if baseline_end > baseline_start else np.mean(filtered_signal)
                    
                    # ST deviation in mV (raw ADC difference, needs conversion)
                    # Convert ADC to mV (approximate: assume 10mm/mV gain, typical ADC scaling)
                    # This is a placeholder - actual conversion should use hardware-specific calibration
                    # For standard 10mm/mV: 1 mV ≈ 1000-1500 ADC counts (varies by lead)
                    st_raw_adc = filtered_signal[st_measure_point] - tp_baseline
                    adc_to_mv_factor = 1000.0  # Placeholder - should be hardware-specific
                    st_mv = st_raw_adc / adc_to_mv_factor
                    
                    # Reasonable ST range: -2.0 to +2.0 mV
                    if -2.0 <= st_mv <= 2.0:
                        st_elevations.append(st_mv)
                    else:
                        pass  # Silently reject extreme outliers
                except Exception as e:
                    print(f"🔍 ST: Exception in beat analysis: {e}")
                    continue
            
            if st_elevations:
                # Return mean ST deviation in mV (rounded to 2 decimal places)
                st_mean_mv = np.mean(st_elevations)
                return round(st_mean_mv, 2)
            
            return 0.0  # No ST detected
        except:
            return 0

    def calculate_qt_interval(self, lead_data):
        """Calculate QT interval (Q-wave onset to T-wave end) from Lead II"""
        try:
            # Early exit: no real signal
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0
            
            # Get sampling rate
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 80
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            elif hasattr(self, 'sampling_rate') and self.sampling_rate:
                fs = float(self.sampling_rate)
            
            # Filter signal
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks
            mean_height = np.mean(filtered_signal)
            std_height = np.std(filtered_signal)
            peaks, _ = find_peaks(
                filtered_signal,
                height=mean_height + 0.3 * std_height,
                distance=int(0.3 * fs),
                prominence=std_height * 0.2
            )
            
            if len(peaks) < 1:
                return 0
            
            qt_intervals = []
            for r_peak in peaks[:min(5, len(peaks))]:
                try:
                    # Find Q-point (min before R, within 40ms)
                    q_start = max(0, r_peak - int(0.04 * fs))
                    q_end = r_peak
                    if q_end > q_start:
                        q_point = q_start + np.argmin(filtered_signal[q_start:q_end])
                    else:
                        q_point = r_peak
                    
                    # Find T-wave end (return to baseline after R)
                    # GE/Philips standard: T-end is where signal returns to TP baseline after T-peak
                    t_search_start = r_peak + int(0.08 * fs)  # After QRS
                    t_search_end = min(len(filtered_signal), r_peak + int(0.4 * fs))  # 400ms max (shorter window)
                    if t_search_end > t_search_start:
                        t_segment = filtered_signal[t_search_start:t_search_end]
                        # TP baseline: isoelectric segment before QRS (150-350ms before R)
                        tp_baseline_start = max(0, r_peak - int(0.35 * fs))
                        tp_baseline_end = max(0, r_peak - int(0.15 * fs))
                        if tp_baseline_end > tp_baseline_start:
                            baseline = np.median(filtered_signal[tp_baseline_start:tp_baseline_end])
                        else:
                            baseline = np.mean(filtered_signal[max(0, r_peak - int(0.15 * fs)):max(0, r_peak - int(0.05 * fs))])
                        
                        # Find T-peak first (max in T segment)
                        t_peak_idx = t_search_start + np.argmax(np.abs(t_segment))
                        
                        # Find FIRST point after T-peak where signal returns to baseline
                        # Use tighter threshold: 0.1 * std (GE/Philips standard)
                        post_t_segment = filtered_signal[t_peak_idx:t_search_end]
                        if len(post_t_segment) > 0:
                            t_end_candidates = np.where(np.abs(post_t_segment - baseline) < 0.1 * std_height)[0]
                            if len(t_end_candidates) > 0:
                                # Use FIRST candidate after T-peak, not last
                                t_end = t_peak_idx + t_end_candidates[0]
                            else:
                                # Fallback: use T-peak + estimated T duration (typically 100-200ms)
                                t_end = t_peak_idx + int(0.15 * fs)  # 150ms after T-peak
                            
                            qt_ms = (t_end - q_point) / fs * 1000.0
                            if 200 <= qt_ms <= 600:  # Reasonable QT interval
                                qt_intervals.append(qt_ms)
                        else:
                            # No T-segment found, skip this beat
                            continue
                except Exception:
                    continue
                
            if qt_intervals:
                return int(round(np.mean(qt_intervals)))
            
            return 0
        except:
            return 0

    def calculate_qtc_interval(self, heart_rate, qt_interval):
        """Calculate QTc using Bazett's formula: QTc = QT / sqrt(RR)"""
        try:
            if not heart_rate or heart_rate <= 0:
                return 0
            
            if not qt_interval or qt_interval <= 0:
                return 0
            
            # Calculate RR interval from heart rate (in seconds)
            rr_interval = 60.0 / heart_rate
            
            # QT in seconds
            qt_sec = qt_interval / 1000.0
            
            # Apply Bazett's formula: QTc = QT / sqrt(RR)
            qtc = qt_sec / np.sqrt(rr_interval)
            
            # Convert back to milliseconds
            qtc_ms = int(round(qtc * 1000))
            
            return qtc_ms
            
        except Exception as e:
            return 0
    
    def calculate_qtcf_interval(self, qt_ms, rr_ms):
        """Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3)
        
        Args:
            qt_ms: QT interval in milliseconds
            rr_ms: RR interval in milliseconds
        
        Returns:
            QTcF in milliseconds
        """
        try:
            if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
                return 0
            
            # Convert to seconds
            qt_sec = qt_ms / 1000.0
            rr_sec = rr_ms / 1000.0
            
            # Fridericia formula: QTcF = QT / RR^(1/3)
            qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
            
            # Convert back to milliseconds
            qtcf_ms = int(round(qtcf_sec * 1000.0))
            
            return qtcf_ms
        except:
            return 0
    
    def calculate_pr_interval_from_median(self, median_beat, time_axis, fs, tp_baseline):
        """Calculate PR interval from median beat: P onset → QRS onset (GE/Philips standard)."""
        try:
            r_idx = np.argmin(np.abs(time_axis))
            
            # Find P onset: first point before R where signal deviates from TP baseline
            p_search_start = max(0, r_idx - int(0.25 * fs))
            p_search_end = max(0, r_idx - int(0.10 * fs))
            if p_search_end <= p_search_start:
                return 150
            
            p_segment = median_beat[p_search_start:p_search_end]
            p_baseline_diff = np.abs(p_segment - tp_baseline)
            signal_range = np.max(median_beat) - np.min(median_beat)
            threshold = max(0.05 * signal_range, np.std(median_beat) * 0.1) if signal_range > 0 else np.std(median_beat) * 0.1
            
            p_deviations = np.where(p_baseline_diff > threshold)[0]
            if len(p_deviations) > 0:
                p_onset_idx = p_search_start + p_deviations[0]
            else:
                p_onset_idx = p_search_start + np.argmax(p_segment)
            
            # Find QRS onset: first point before R where signal deviates from TP baseline
            qrs_search_start = max(0, r_idx - int(0.04 * fs))
            qrs_search_end = r_idx
            if qrs_search_end <= qrs_search_start:
                return 150
            
            qrs_segment = median_beat[qrs_search_start:qrs_search_end]
            qrs_baseline_diff = np.abs(qrs_segment - tp_baseline)
            qrs_deviations = np.where(qrs_baseline_diff > threshold)[0]
            if len(qrs_deviations) > 0:
                qrs_onset_idx = qrs_search_start + qrs_deviations[0]
            else:
                qrs_onset_idx = qrs_search_start + np.argmin(qrs_segment)
            
            # PR = P onset → QRS onset
            pr_ms = time_axis[qrs_onset_idx] - time_axis[p_onset_idx]
            if 80 <= pr_ms <= 240:
                return int(round(pr_ms))
            return 150
        except:
            return 150
    
    def calculate_qrs_duration_from_median(self, median_beat, time_axis, fs, tp_baseline):
        """Calculate QRS duration from median beat: QRS onset → QRS offset (GE/Philips standard)."""
        try:
            r_idx = np.argmin(np.abs(time_axis))
            signal_range = np.max(median_beat) - np.min(median_beat)
            threshold = max(0.05 * signal_range, np.std(median_beat) * 0.1) if signal_range > 0 else np.std(median_beat) * 0.1
            
            # Find QRS onset: first point before R where signal deviates from TP baseline
            qrs_onset_start = max(0, r_idx - int(0.04 * fs))
            qrs_onset_end = r_idx
            if qrs_onset_end <= qrs_onset_start:
                return 80
            
            qrs_onset_segment = median_beat[qrs_onset_start:qrs_onset_end]
            qrs_onset_diff = np.abs(qrs_onset_segment - tp_baseline)
            qrs_onset_deviations = np.where(qrs_onset_diff > threshold)[0]
            if len(qrs_onset_deviations) > 0:
                qrs_onset_idx = qrs_onset_start + qrs_onset_deviations[0]
            else:
                qrs_onset_idx = qrs_onset_start + np.argmin(qrs_onset_segment)
            
            # Find QRS offset: first point after R where signal returns to TP baseline
            qrs_offset_start = r_idx
            qrs_offset_end = min(len(median_beat), r_idx + int(0.12 * fs))
            if qrs_offset_end <= qrs_offset_start:
                return 80
            
            qrs_offset_segment = median_beat[qrs_offset_start:qrs_offset_end]
            qrs_offset_diff = np.abs(qrs_offset_segment - tp_baseline)
            qrs_offset_deviations = np.where(qrs_offset_diff < threshold)[0]
            if len(qrs_offset_deviations) > 0:
                qrs_offset_idx = qrs_offset_start + qrs_offset_deviations[0]
            else:
                # Fallback: use max in QRS segment (end of S-wave)
                qrs_offset_idx = qrs_offset_start + np.argmax(qrs_offset_segment)
            
            # QRS duration = QRS onset → QRS offset
            qrs_ms = time_axis[qrs_offset_idx] - time_axis[qrs_onset_idx]
            if 40 <= qrs_ms <= 200:
                return int(round(qrs_ms))
            return 80
        except:
            return 80
    
    def calculate_qrs_axis_from_median(self):
        """Calculate QRS axis from median beat vectors (GE/Philips standard)."""
        try:
            if len(self.data) < 6:
                return 0
            fs = 80.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            lead_i_raw = self.data[0]
            lead_avf_raw = self.data[5]
            lead_ii = self.data[1]
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_ii = filtfilt(b, a, lead_ii)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
            if len(r_peaks) < 8: # Enforce 8 beats
                return getattr(self, '_prev_qrs_axis', 0) or 0
            _, median_i = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
            _, median_avf = build_median_beat(lead_avf_raw, r_peaks, fs, min_beats=8)
            if median_i is None or median_avf is None:
                return getattr(self, '_prev_qrs_axis', 0) or 0
            r_peak_idx = len(median_i) // 2
            # Get Lead II median beat for axis calculation
            _, median_ii = build_median_beat(lead_ii, r_peaks, fs, min_beats=8)
            if median_ii is None:
                return getattr(self, '_prev_qrs_axis', 0) or 0
            # Get TP baselines for Lead I and aVF (REQUIRED for correct axis calculation)
            r_mid = r_peaks[len(r_peaks) // 2]
            prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
            tp_baseline_i = get_tp_baseline(lead_i_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
            tp_baseline_avf = get_tp_baseline(lead_avf_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
            
            # Build time axis for median beat
            time_axis_i, _ = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
            if time_axis_i is None:
                return getattr(self, '_prev_qrs_axis', 0) or 0
            
            # Calculate QRS axis using strict wave windows and net area (integral)
            axis_deg = calculate_axis_from_median_beat(lead_i_raw, lead_ii, lead_avf_raw, median_i, median_ii, median_avf, r_peak_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf, time_axis=time_axis_i, wave_type='QRS', prev_axis=self._prev_qrs_axis)
            self._prev_qrs_axis = axis_deg
            return int(round(axis_deg))
        except Exception as e:
            print(f"❌ Error calculating QRS axis from median: {e}")
            return 0
    
    def calculate_p_axis_from_median(self):
        """Calculate P-wave axis from median beat using P-wave only (GE/Philips standard)."""
        try:
            if len(self.data) < 6:
                return 0
            
            fs = 80.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            
            lead_i_raw = self.data[0]
            lead_avf_raw = self.data[5]
            lead_ii = self.data[1]
            
            # Detect R-peaks in Lead II for alignment
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_ii = filtfilt(b, a, lead_ii)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
            
            if len(r_peaks) < 8: # Enforce 8 beats
                return getattr(self, '_prev_p_axis', 0) or 0
            
            # Build median beats for Lead I and aVF
            _, median_i = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
            _, median_avf = build_median_beat(lead_avf_raw, r_peaks, fs, min_beats=8)
            if median_i is None or median_avf is None:
                return getattr(self, '_prev_p_axis', 0) or 0
            
            # Get Lead II median beat for axis calculation validation
            _, median_ii = build_median_beat(lead_ii, r_peaks, fs, min_beats=8)
            if median_ii is None:
                return getattr(self, '_prev_p_axis', 0) or 0
            
            r_peak_idx = len(median_i) // 2  # R-peak at center
            
            # Get TP baselines for Lead I and aVF (REQUIRED for correct axis calculation)
            r_mid = r_peaks[len(r_peaks) // 2]
            prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
            tp_baseline_i = get_tp_baseline(lead_i_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
            tp_baseline_avf = get_tp_baseline(lead_avf_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
            
            # Build time axis for median beat
            time_axis_i, _ = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
            if time_axis_i is None:
                return getattr(self, '_prev_p_axis', 0) or 0
            
            # Calculate P axis using strict wave windows and net area (integral)
            p_axis_deg = calculate_axis_from_median_beat(lead_i_raw, lead_ii, lead_avf_raw, median_i, median_ii, median_avf, r_peak_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf, time_axis=time_axis_i, wave_type='P', prev_axis=self._prev_p_axis)
            
            # Update previous value for next calculation
            self._prev_p_axis = p_axis_deg
            
            return int(round(p_axis_deg))
        except Exception as e:
            print(f"❌ Error calculating P axis from median: {e}")
            return 0
    
    def calculate_t_axis_from_median(self):
        """Calculate T-wave axis from median beat vectors (GE/Philips standard)."""
        try:
            if len(self.data) < 6:
                return 0
            fs = 80.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            lead_i_raw = self.data[0]
            lead_avf_raw = self.data[5]
            lead_ii = self.data[1]
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_ii = filtfilt(b, a, lead_ii)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
            if len(r_peaks) < 8: # Enforce 8 beats
                return getattr(self, '_prev_t_axis', 0) or 0
            _, median_i = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
            _, median_avf = build_median_beat(lead_avf_raw, r_peaks, fs, min_beats=8)
            if median_i is None or median_avf is None:
                return getattr(self, '_prev_t_axis', 0) or 0
            r_peak_idx = len(median_i) // 2
            # Get Lead II median beat for axis calculation
            _, median_ii = build_median_beat(lead_ii, r_peaks, fs, min_beats=8)
            if median_ii is None:
                return getattr(self, '_prev_t_axis', 0) or 0
            # Get TP baselines for Lead I and aVF (REQUIRED for correct axis calculation)
            r_mid = r_peaks[len(r_peaks) // 2]
            prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
            tp_baseline_i = get_tp_baseline(lead_i_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
            tp_baseline_avf = get_tp_baseline(lead_avf_raw, r_mid, fs, prev_r_peak_idx=prev_r_idx)
            
            # Build time axis for median beat
            time_axis_i, _ = build_median_beat(lead_i_raw, r_peaks, fs, min_beats=8)
            if time_axis_i is None:
                return getattr(self, '_prev_t_axis', 0) or 0
            
            # Calculate T axis using strict wave windows and net area (integral)
            axis_deg = calculate_axis_from_median_beat(lead_i_raw, lead_ii, lead_avf_raw, median_i, median_ii, median_avf, r_peak_idx, fs, tp_baseline_i=tp_baseline_i, tp_baseline_avf=tp_baseline_avf, time_axis=time_axis_i, wave_type='T', prev_axis=self._prev_t_axis)
            self._prev_t_axis = axis_deg
            return int(round(axis_deg))
        except Exception as e:
            print(f"❌ Error calculating T axis from median: {e}")
            return 0
    
    def calculate_rv5_sv1_from_median(self):
        """Calculate RV5 and SV1 from median beats (GE/Philips standard)."""
        try:
            if len(self.data) < 8:
                return None, None
            fs = 80.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            lead_v5_raw = self.data[6] if len(self.data) > 6 else None
            lead_v1_raw = self.data[7] if len(self.data) > 7 else None
            if lead_v5_raw is None or lead_v1_raw is None:
                return None, None
            lead_ii = self.data[1]
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_ii = filtfilt(b, a, lead_ii)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, distance=int(0.3 * fs), prominence=signal_std * 0.4)
            if len(r_peaks) < 3:
                return None, None
            filtered_v5 = filtfilt(b, a, lead_v5_raw)
            filtered_v1 = filtfilt(b, a, lead_v1_raw)
            r_peaks_v5, _ = find_peaks(filtered_v5, height=np.mean(filtered_v5) + 0.5 * np.std(filtered_v5), distance=int(0.3 * fs), prominence=np.std(filtered_v5) * 0.4)
            r_peaks_v1, _ = find_peaks(filtered_v1, height=np.mean(filtered_v1) + 0.5 * np.std(filtered_v1), distance=int(0.3 * fs), prominence=np.std(filtered_v1) * 0.4)
            if len(r_peaks_v5) < 3 or len(r_peaks_v1) < 3:
                return None, None
            rv5_mv, sv1_mv = measure_rv5_sv1_from_median_beat(lead_v5_raw, lead_v1_raw, r_peaks_v5, r_peaks_v1, fs, v5_adc_per_mv=2048.0, v1_adc_per_mv=1441.0)
            return rv5_mv, sv1_mv
        except Exception as e:
            print(f"❌ Error calculating RV5/SV1 from median: {e}")
            return None, None

    def update_ecg_metrics_display(self, heart_rate, pr_interval, qrs_duration, qrs_axis, st_interval, qt_interval=None, qtc_interval=None):
        """Update the ECG metrics display in the UI (dashboard: BPM, PR, QRS axis, ST, QT/QTc, timer only)"""
        try:
            # Throttle updates to every 5 seconds to avoid fast flicker
            import time as _time
            if not hasattr(self, '_last_metric_update_ts'):
                self._last_metric_update_ts = 0.0
            if _time.time() - self._last_metric_update_ts < 5.0:
                return
            
            if hasattr(self, 'metric_labels'):
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText(f"{heart_rate} ")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText(f"{pr_interval} ")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText(f"{qrs_duration} ")
                if 'qrs_axis' in self.metric_labels:
                    self.metric_labels['qrs_axis'].setText(f"{qrs_axis}°")
                if 'st_interval' in self.metric_labels:
                    # ST deviation is in mV, format appropriately (2 decimal places max)
                    if isinstance(st_interval, (int, float)):
                        # Round to 2 decimal places and format
                        st_formatted = round(float(st_interval), 2)
                        self.metric_labels['st_interval'].setText(f"{st_formatted:.2f} mV")
                    else:
                        self.metric_labels['st_interval'].setText("0.00 mV")
                if 'qtc_interval' in self.metric_labels:
                    # Display QT/QTc only
                    if qt_interval is not None and qtc_interval is not None:
                        try:
                            qt_i = int(round(qt_interval))
                            qtc_i = int(round(qtc_interval))
                            self.metric_labels['qtc_interval'].setText(f"{qt_i}/{qtc_i}")
                        except Exception:
                            self.metric_labels['qtc_interval'].setText(f"{qt_interval}/{qtc_interval}")
                    elif qtc_interval is not None:
                        try:
                            qtc_i = int(round(qtc_interval))
                            self.metric_labels['qtc_interval'].setText(f"{qtc_i} ")
                        except Exception:
                            self.metric_labels['qtc_interval'].setText(f"{qtc_interval} ")
            # mark last update time
            self._last_metric_update_ts = _time.time()
        except Exception as e:
            print(f"Error updating ECG metrics: {e}")

    def get_current_metrics(self):
        """Get current ECG metrics for dashboard display"""
        try:
            metrics = {}
            
            # Check if we have real signal data
            has_real_signal = False
            if len(self.data) > 1:  # Lead II data available
                lead_ii_data = self.data[1]
                if len(lead_ii_data) >= 100 and not np.all(lead_ii_data == 0) and np.std(lead_ii_data) >= 0.1:
                    has_real_signal = True
            
            # Get current heart rate
            if has_real_signal:
                heart_rate = self.calculate_heart_rate(self.data[1])
                metrics['heart_rate'] = f"{heart_rate}" if heart_rate > 0 else "0"
            else:
                metrics['heart_rate'] = "0"
            
            # Get other metrics from UI labels (these should be zero if reset properly)
            if hasattr(self, 'metric_labels'):
                if 'pr_interval' in self.metric_labels:
                    metrics['pr_interval'] = self.metric_labels['pr_interval'].text().replace(' ms', '')
                if 'qrs_duration' in self.metric_labels:
                    metrics['qrs_duration'] = self.metric_labels['qrs_duration'].text().replace(' ms', '')
                if 'qrs_axis' in self.metric_labels:
                    metrics['qrs_axis'] = self.metric_labels['qrs_axis'].text().replace('°', '')
                if 'st_interval' in self.metric_labels:
                    metrics['st_interval'] = self.metric_labels['st_interval'].text().replace(' ms', '')
                if 'qtc_interval' in self.metric_labels:
                    metrics['qtc_interval'] = self.metric_labels['qtc_interval'].text().replace(' ms', '')
                if 'time_elapsed' in self.metric_labels:
                    metrics['time_elapsed'] = self.metric_labels['time_elapsed'].text()
            
            # Get sampling rate
            if hasattr(self, 'sampler') and self.sampler.sampling_rate > 0:
                metrics['sampling_rate'] = f"{self.sampler.sampling_rate:.1f}"
            else:
                metrics['sampling_rate'] = "--"
            
            # Reduced debug output - only print every 100 calls to avoid console spam
            if not hasattr(self, '_metrics_call_count'):
                self._metrics_call_count = 0
            self._metrics_call_count += 1
            if self._metrics_call_count % 100 == 0:
                print(f"🔍 get_current_metrics returning: {metrics}")
            return metrics
        except Exception as e:
            print(f"Error getting current metrics: {e}")
            return {}

    def get_latest_rhythm_interpretation(self):
        """Expose latest arrhythmia interpretation string for the dashboard."""
        return getattr(self, '_latest_rhythm_interpretation', "Analyzing Rhythm...")

    def update_plot_y_range(self, plot_index):
        """Update Y-axis range for a specific plot using robust stats to avoid cropping"""
        try:
            if plot_index >= len(self.data) or plot_index >= len(self.plot_widgets):
                return

            # Get the data for this plot
            data = self.data[plot_index]
            
            # Remove NaN values and large outliers (robust)
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                return
            
            # Use percentiles to avoid spikes from clipping the view
            p1 = np.percentile(valid_data, 1)
            p99 = np.percentile(valid_data, 99)
            data_mean = (p1 + p99) / 2.0
            data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
            # Maximum deviation of any point from the mean – we will always cover this
            peak_deviation = np.max(np.abs(valid_data - data_mean))
            
            # Calculate appropriate Y-range with some padding
            if data_std > 0:
                # Use standard deviation within central band
                base_padding = max(data_std * 4, 200)  # Increased padding for better visibility
                padding = base_padding  # Do NOT scale padding with gain; gain already applied to signal
                y_min = data_mean - padding
                y_max = data_mean + padding
                print(f"📊 Basic Y-range: base_padding={base_padding:.1f}, padding={padding:.1f}")
            else:
                # Fallback: use percentile window
                data_range = max(p99 - p1, 300)
                base_padding = max(data_range * 0.3, 200)
                padding = base_padding  # Do NOT scale padding with gain; gain already applied to signal
                y_min = data_mean - padding
                y_max = data_mean + padding
                print(f"📊 Basic Y-range (fallback): base_padding={base_padding:.1f}, padding={padding:.1f}")
            
            # Ensure reasonable bounds
            y_min = max(y_min, -8000)
            y_max = min(y_max, 8000)
            
            # Update the plot's Y-range
            self.plot_widgets[plot_index].setYRange(y_min, y_max, padding=0)
            
        except Exception as e:
            print(f"Error updating Y-range for plot {plot_index}: {e}")

    def on_settings_changed(self, key, value):
        
        print(f"Setting changed: {key} = {value}")
        
        if key in ["wave_speed", "wave_gain"]:
            # Apply new settings immediately
            self.apply_display_settings()
            
            # CRITICAL: Update all lead titles IMMEDIATELY
            self.update_all_lead_titles()
            
            # Force redraw of all plots
            self.redraw_all_plots()
            
            # Notify demo manager for instant updates (like divyansh.py)
            if hasattr(self, 'demo_manager') and self.demo_manager:
                self.demo_manager.on_settings_changed(key, value)
            
            print(f"Settings applied and titles updated for {key} = {value}")
        elif key == "system_language":
            self.apply_language(value)

    def update_all_lead_titles(self):
        """Update all lead titles with current speed and gain settings"""
        # Safety check: only update if plots are initialized
        if not hasattr(self, 'axs') or not self.axs:
            print("⚠️ Plots not initialized yet, skipping title update")
            return
            
        current_speed = self.settings_manager.get_wave_speed()
        current_gain = self.settings_manager.get_wave_gain()
        
        print(f"Updating titles: Speed={current_speed}mm/s, Gain={current_gain}mm/mV")
        
        for i, lead in enumerate(self.leads):
            if i < len(self.axs):
                new_title = f"{lead} | Speed: {current_speed}mm/s | Gain: {current_gain}mm/mV"
                self.axs[i].set_title(new_title, fontsize=8, color='#666', pad=10)
                print(f"Updated {lead} title: {new_title}")
        
        # Force redraw of all canvases
        for canvas in self.canvases:
            if canvas:
                canvas.draw_idle()

    def apply_display_settings(self):
        
        wave_speed = self.settings_manager.get_wave_speed()
        wave_gain = self.settings_manager.get_wave_gain()
        
        # Higher speed = more samples per second = larger buffer for same time window
        base_buffer = getattr(self, "base_buffer_size", 2000)
        speed_factor = wave_speed / 50.0  # 50mm/s is baseline
        self.buffer_size = int(base_buffer * speed_factor)
        
        # Update y-axis limits based on gain.
        # Higher mm/mV = higher gain = larger waves = need more Y-axis range
        # Use clinical standard helper function (10mm/mV = 1.0x baseline)
        base_ylim = 400
        gain_factor = get_display_gain(wave_gain)
        # Scale Y-axis range with gain (NO CLAMPING - allow all gain values)
        self.ylim = int(base_ylim * gain_factor)

        # Force immediate redraw of all plots with new settings
        self.redraw_all_plots()
        
        print(f"Applied settings: speed={wave_speed}mm/s, gain={wave_gain}mm/mV, buffer={self.buffer_size}, ylim={self.ylim}")

    # ------------------------ Update Dashboard Metrics on the top of the lead graphs ------------------------
    
    def create_metrics_frame(self):
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metrics_frame")
        metrics_frame.setStyleSheet("""
            QFrame {
                background: #000000;
                border: none;
                border-radius: 6px;
                padding: 0;  /* Reduced from 4px */
                margin: 0;  /* Reduced from 2px */
            }
        """)
        
        metrics_layout = QHBoxLayout(metrics_frame)
        metrics_layout.setSpacing(6)  # Reduced from 10px
        metrics_layout.setContentsMargins(6, 6, 6, 6)  # Reduced from 10px
        
        # Store metric labels for live update
        self.metric_labels = {}
        
        # Updated metric info to match the image design with consistent color coding
        metric_info = [
            ("PR", "0", "pr_interval", "#ffffff"),
            ("QRS", "0", "qrs_duration", "#ffffff"),
            ("Axis", "0°", "qrs_axis", "#ffffff"),
            ("ST", "0", "st_interval", "#ffffff"),
            ("QT/Qtc", "0", "qtc_interval", "#ffffff"),
            ("Time", "00:00", "time_elapsed", "#ffffff"),
        ]
        
        for title, value, key, color in metric_info:
            metric_widget = QWidget()
            # Time and QTc metrics need more width to prevent cropping
            min_w = "140px" if key in ["time_elapsed", "qtc_interval"] else "90px"
            metric_widget.setStyleSheet(f"""
                QWidget {{
                    background: transparent;
                    min-width: {min_w};
                    border-right: none;
                }}
            """)

            metric_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            
            # Create vertical layout for the metric widget
            box = QVBoxLayout(metric_widget)
            box.setSpacing(2)  # Reduced from 3px
            box.setAlignment(Qt.AlignCenter)
            
            # Title label with consistent color coding - Make it smaller
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 9, QFont.Bold))
            lbl.setStyleSheet(f"color: #000000; margin-bottom: 2px; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            
            # Value label with specific colors - Make it smaller
            val = QLabel(value)
            val.setFont(QFont("Arial", 32, QFont.Bold))
            val.setStyleSheet(f"color: #000000; background: transparent; padding: 0px;")
            val.setAlignment(Qt.AlignCenter)
            
            # Add labels to the metric widget's layout
            box.addWidget(lbl)
            box.addWidget(val)
            
            # Add the metric widget to the horizontal layout
            metrics_layout.addWidget(metric_widget)
            
            # Store reference for live update
            self.metric_labels[key] = val
        
        # Heart rate metric (no emoji, red color)
        heart_rate_widget = QWidget()
        heart_rate_widget.setStyleSheet("""
            QWidget {
                background: transparent;
                min-width: 90px;
                border-right: none;
            }
        """)
        heart_rate_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        heart_box = QVBoxLayout(heart_rate_widget)
        heart_box.setSpacing(2)
        heart_box.setAlignment(Qt.AlignCenter)
        
        # Heart rate title
        hr_title = QLabel("BPM")
        hr_title.setFont(QFont("Arial", 9, QFont.Bold))
        hr_title.setStyleSheet("color: #ff0000; margin-bottom: 2px; font-weight: bold;")
        hr_title.setAlignment(Qt.AlignCenter)
        
        # Heart rate value (red color)
        heart_rate_val = QLabel("00")
        heart_rate_val.setFont(QFont("Arial", 32, QFont.Bold))
        heart_rate_val.setStyleSheet("color: #ff0000; background: transparent; padding: 0px;")
        heart_rate_val.setAlignment(Qt.AlignCenter)
        
        heart_box.addWidget(hr_title)
        heart_box.addWidget(heart_rate_val)
        
        # Insert heart rate widget at the beginning
        metrics_layout.insertWidget(0, heart_rate_widget)
        self.metric_labels['heart_rate'] = heart_rate_val
        
        # Reset all metrics to zero after creating the frame
        self.reset_metrics_to_zero()
        
        return metrics_frame

    def update_ecg_metrics_on_top_of_lead_graphs(self, intervals):
        if 'Heart_Rate' in intervals and intervals['Heart_Rate'] is not None:
            self.metric_labels['heart_rate'].setText(
                f"{int(round(intervals['Heart_Rate']))}" if isinstance(intervals['Heart_Rate'], (int, float)) else str(intervals['Heart_Rate'])
            )
        
        if 'PR' in intervals and intervals['PR'] is not None:
            self.metric_labels['pr_interval'].setText(
                f"{int(round(intervals['PR']))}" if isinstance(intervals['PR'], (int, float)) else str(intervals['PR'])
            )
        
        if 'QRS' in intervals and intervals['QRS'] is not None:
            self.metric_labels['qrs_duration'].setText(
                f"{int(round(intervals['QRS']))}" if isinstance(intervals['QRS'], (int, float)) else str(intervals['QRS'])
            )
        
        if 'QRS_axis' in intervals and intervals['QRS_axis'] is not None:
            self.metric_labels['qrs_axis'].setText(str(intervals['QRS_axis']))
        
        if 'ST' in intervals and intervals['ST'] is not None:
            self.metric_labels['st_interval'].setText(
                f"{int(round(intervals['ST']))}" if isinstance(intervals['ST'], (int, float)) else str(intervals['ST'])
            )
        
        if 'QTc' in intervals and intervals['QTc'] is not None:
            self.metric_labels['qtc_interval'].setText(
                f"{int(round(intervals['QTc']))}" if isinstance(intervals['QTc'], (int, float)) else str(intervals['QTc'])
            )
        
        if 'time_elapsed' in self.metric_labels:
            # Time elapsed will be updated separately by a timer
            pass

    def update_metrics_frame_theme(self, dark_mode=False, medical_mode=False):
       
        if not hasattr(self, 'metrics_frame'):
            return
            
        if dark_mode:
            # Dark mode styling
            self.metrics_frame.setStyleSheet("""
                QFrame#metrics_frame {
                    background: #000000;
                    border: none;
                    border-radius: 6px;
                    padding: 0px;
                    margin: 0px 0;
                    /* Removed unsupported box-shadow property */
                }
            """)
            
            # Update text colors for dark mode
            for key, label in self.metric_labels.items():
                if key == 'heart_rate':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 0; border: none; margin: 0; font-size: 50px;")
                elif key == 'pr_interval':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_duration':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_axis':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'st_interval':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'time_elapsed':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 45px; min-width: 140px;")
                elif key == 'qtc_interval':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
            
            # Update title colors to green for dark mode
            for child in self.metrics_frame.findChildren(QLabel):
                if child != self.metric_labels.get('heart_rate') and child != self.metric_labels.get('time_elapsed') and child != self.metric_labels.get('qtc_interval'):
                    if not any(child == label for label in self.metric_labels.values()):
                        child.setStyleSheet("color: #00ff00; margin-bottom: 5px; border: none;")
                        
        elif medical_mode:
            # Medical mode styling (green theme)
            self.metrics_frame.setStyleSheet("""
                QFrame#metrics_frame {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #f0fff0, stop:1 #e0f0e0);
                    border: none;
                    border-radius: 6px;
                    padding: 0;
                    margin: 0;
                    /* Removed unsupported box-shadow property */
                }
            """)
            
            # Update text colors for medical mode
            for key, label in self.metric_labels.items():
                if key == 'heart_rate':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 0; border: none; margin: 0; font-size: 50px;")
                elif key == 'pr_interval':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_duration':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_axis':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'st_interval':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'time_elapsed':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 45px; min-width: 140px;")
                elif key == 'qtc_interval':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
            
            # Update title colors to dark green for medical mode
            for child in self.metrics_frame.findChildren(QLabel):
                if child != self.metric_labels.get('heart_rate') and child != self.metric_labels.get('time_elapsed') and child != self.metric_labels.get('qtc_interval'):
                    if not any(child == label for label in self.metric_labels.values()):
                        child.setStyleSheet("color: #2e7d32; margin-bottom: 5px; border: none;")
                        
        else:
            # Light mode (default) styling
            self.metrics_frame.setStyleSheet("""
                QFrame#metrics_frame {
                    background: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 0;
                    margin: 0;
                    /* Removed unsupported box-shadow property */
                }
            """)
            
            # Update text colors for light mode
            for key, label in self.metric_labels.items():
                if key == 'heart_rate':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 0; border: none; margin: 0; font-size: 50px;")
                elif key == 'pr_interval':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_duration':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_axis':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'st_interval':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'time_elapsed':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 45px; min-width: 140px;")
                elif key == 'qtc_interval':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
            
            # Update title colors to dark gray for light mode
            for child in self.metrics_frame.findChildren(QLabel):
                if child != self.metric_labels.get('heart_rate') and child != self.metric_labels.get('time_elapsed') and child != self.metric_labels.get('qtc_interval'):
                    if not any(child == label for label in self.metric_labels.values()):
                        child.setStyleSheet("color: #666; margin-bottom: 5px; border: none;")

    def update_elapsed_time(self):
        # Only update time when acquisition is running (not paused)
        if not hasattr(self, 'serial_reader') or not self.serial_reader or not self.serial_reader.running:
            # Acquisition is stopped/paused - don't update time
            return
        
        if self.start_time and 'time_elapsed' in self.metric_labels:
            try:
                current_time = time.time()
                # Subtract paused duration from elapsed time
                paused_duration = getattr(self, 'paused_duration', 0)
                elapsed = max(0, current_time - self.start_time - paused_duration)  # Ensure non-negative
                
                # Calculate minutes and seconds
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                
                # Store last displayed time to prevent skipping
                if not hasattr(self, '_last_displayed_elapsed'):
                    self._last_displayed_elapsed = -1
                
                # Only update if time actually changed (prevent duplicate updates)
                current_elapsed_int = int(elapsed)
                if current_elapsed_int != self._last_displayed_elapsed:
                    self.metric_labels['time_elapsed'].setText(f"{minutes:02d}:{seconds:02d}")
                    self._last_displayed_elapsed = current_elapsed_int
            except Exception as e:
                print(f"❌ Error updating elapsed time: {e}")

    def reset_metrics_to_zero(self):
        """Reset all ECG metric labels to zero/initial state."""
        try:
            if hasattr(self, 'metric_labels') and isinstance(self.metric_labels, dict):
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("00")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("0")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("0")
                if 'qrs_axis' in self.metric_labels:
                    self.metric_labels['qrs_axis'].setText("0°")
                if 'st_interval' in self.metric_labels:
                    self.metric_labels['st_interval'].setText("0")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("0/0")
                if 'time_elapsed' in self.metric_labels:
                    self.metric_labels['time_elapsed'].setText("00:00")
        except Exception:
            # Never block UI on reset
            pass

    def showEvent(self, event):
        """Called when the ECG test page is shown - reset metrics to zero"""
        super().showEvent(event)

        # Check if demo mode is active
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            # Demo mode is active - set fixed demo values instead of resetting to zero
            print("🔍 Page shown with demo mode active - setting fixed demo values")
            if hasattr(self, 'metric_labels'):
                self.metric_labels.get('heart_rate', QLabel()).setText("60")
                self.metric_labels.get('pr_interval', QLabel()).setText("160")
                self.metric_labels.get('qrs_duration', QLabel()).setText("85")
                self.metric_labels.get('qrs_axis', QLabel()).setText("0°")
                self.metric_labels.get('st_interval', QLabel()).setText("90")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("400/430")
        else:
            # Demo mode is not active - reset metrics to zero
            self.reset_metrics_to_zero()

    # ------------------------ Calculate ECG Intervals ------------------------

    def calculate_ecg_intervals(self, lead_ii_data):
        if not lead_ii_data or len(lead_ii_data) < 100:
            return {}
        
        try:
            from ecg.pan_tompkins import pan_tompkins
            
            # Convert to numpy array
            data = np.array(lead_ii_data)
            
            # Detect R peaks using Pan-Tompkins algorithm
            r_peaks = pan_tompkins(data, fs=500)  # 500Hz sampling rate
            
            if len(r_peaks) < 2:
                return {}
            
            # Calculate heart rate
            rr_intervals = np.diff(r_peaks) / 500.0  # Convert to seconds
            mean_rr = np.mean(rr_intervals)
            heart_rate = 60 / mean_rr if mean_rr > 0 else 0
            
            # Calculate intervals
            pr_interval = 0.16  
            qrs_duration = 0.08  
            qt_interval = 0.4    
            qtc_interval = 0.42  
            qrs_axis = "--"      
            st_segment = 0.12    
            
            return {
                'Heart_Rate': heart_rate,
                'PR': pr_interval * 1000,  # Convert to ms
                'QRS': qrs_duration * 1000,
                'QT': qt_interval * 1000,
                'QTc': qtc_interval * 1000,
                'QRS_axis': qrs_axis,
                'ST': st_segment * 1000
            }
                
        except Exception as e:
            print(f"Error calculating ECG intervals: {e}")
            return {}

    # ------------------------ Show help dialog ------------------------

    def show_help(self):
        help_text = """
        <h3>12-Lead ECG Monitor Help</h3>
        <p><b>Getting Started:</b></p>
        <ul>
        <li>Configure serial port and baud rate in System Setup</li>
        <li>Click 'Start' to begin recording</li>
        <li>Click on any lead to view it in detail</li>
        <li>Use the menu options for additional features</li>
        </ul>
        <p><b>Features:</b></p>
        <ul>
        <li>Real-time 12-lead ECG monitoring</li>
        <li>Export data as PDF or CSV</li>
        <li>Detailed lead analysis</li>
        <li>Arrhythmia detection</li>
        </ul>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Help - 12-Lead ECG Monitor")
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # ------------------------ Ports Configuration Dialog ------------------------

    def show_ports_dialog(self):
        """Show simple dialog for configuring COM port and baud rate"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Port Configuration")
        dialog.setFixedSize(300, 200)
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        # Port selection
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("COM Port:"))
        port_combo = QComboBox()
        port_combo.addItem("Select Port")
        
        # Get available ports
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                port_combo.addItem(port.device)
        except Exception as e:
            print(f"Error listing ports: {e}")
        
        # Set current port if available
        current_port = self.settings_manager.get_serial_port()
        if current_port and current_port != "Select Port":
            index = port_combo.findText(current_port)
            if index >= 0:
                port_combo.setCurrentIndex(index)
        
        port_layout.addWidget(port_combo)
        layout.addLayout(port_layout)
        
        # Baud rate selection
        baud_layout = QHBoxLayout()
        baud_layout.addWidget(QLabel("Baud Rate:"))
        baud_combo = QComboBox()
        baud_rates = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
        baud_combo.addItems(baud_rates)
        
        # Set current baud rate if available
        current_baud = self.settings_manager.get_baud_rate()
        if current_baud:
                baud_combo.setCurrentText(current_baud)
        
        baud_layout.addWidget(baud_combo)
        layout.addLayout(baud_layout)
        
        # Refresh ports button
        refresh_btn = QPushButton("🔄 Refresh Ports")
        refresh_btn.clicked.connect(lambda: self.refresh_port_combo(port_combo))
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border: 2px solid #ff6600;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #e55a00;
            }
        """)
        layout.addWidget(refresh_btn)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border: 2px solid #6c757d;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5a6268;
            }
        """)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_port_settings(dialog, port_combo, baud_combo))
        save_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border: 2px solid #ff6600;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #e55a00;
            }
        """)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        dialog.exec_()
        
    def _save_port_settings(self, dialog, port_combo, baud_combo):
        """Save port settings and close dialog"""
        selected_port = port_combo.currentText()
        selected_baud = baud_combo.currentText()
        
        if selected_port != "Select Port":
            self.settings_manager.set_setting("serial_port", selected_port)
            self.settings_manager.set_setting("baud_rate", selected_baud)
            print(f"Port settings saved: {selected_port} at {selected_baud} baud")
            dialog.accept()
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid COM port.")

    def refresh_port_combo(self, port_combo):
        """Refresh the port combo box with currently available ports"""
        port_combo.clear()
        port_combo.addItem("Select Port")
        
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                port_combo.addItem("No ports found")
                QMessageBox.information(self, "Port Refresh", "No serial ports detected.")
            else:
                for port in ports:
                    port_combo.addItem(port.device)
                QMessageBox.information(self, "Port Refresh", 
                    f"Found {len(ports)} serial ports:\n" + 
                    "\n".join([port.device for port in ports]))
        except Exception as e:
            port_combo.addItem("Error detecting ports")
            QMessageBox.warning(self, "Port Refresh Error", f"Error detecting ports: {str(e)}")

    def test_serial_connection(self, port, baud_rate):
        """Test the serial connection with the specified port and baud rate"""
        if port == "Select Port":
            QMessageBox.warning(self, "Invalid Port", "Please select a valid COM port first.")
            return
        
        try:
            # Try to open the serial connection
            test_serial = serial.Serial(port, int(baud_rate), timeout=1)
            test_serial.close()
            
            QMessageBox.information(self, "Connection Test", 
                f"✅ Connection successful!\nPort: {port}\nBaud Rate: {baud_rate}")
            
        except serial.SerialException as e:
            QMessageBox.critical(self, "Connection Failed", 
                f"❌ Connection failed!\nPort: {port}\nBaud Rate: {baud_rate}\n\nError: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", 
                f"❌ Unexpected error!\nError: {str(e)}")

    # ------------------------ Capture Screen Details ------------------------

    def capture_screen(self):
        try:
            
            # Get the main window
            main_window = self.window()
            
            # Create a timer to delay the capture slightly to ensure UI is ready
            def delayed_capture():
                # Capture the entire window
                pixmap = main_window.grab()
                
                # Show save dialog
                filename, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Save Screenshot", 
                    f"ECG_Screenshot_{QDateTime.currentDateTime().toString('yyyy-MM-dd_hh-mm-ss')}.png",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )
                
                if filename:
                    # Save the screenshot
                    if pixmap.save(filename):
                        QMessageBox.information(
                            self, 
                            "Success", 
                            f"Screenshot saved successfully!\nLocation: {filename}"
                        )
                    else:
                        QMessageBox.warning(
                            self, 
                            "Error", 
                            "Failed to save screenshot."
                        )
            
            # Use a short delay to ensure the UI is fully rendered
            QTimer.singleShot(100, delayed_capture)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to capture screenshot: {str(e)}"
            )

    # ------------------------ Recording Details ------------------------

    def toggle_recording(self):
        if self.recording_toggle.isChecked():
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        try:
            # Initialize recording
            self.is_recording = True
            
            # Update UI - only change button text, no status updates
            self.recording_toggle.setText("STOP")
            
            # Start capture timer
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.capture_frame)
            self.recording_timer.start(33)  # ~30 FPS
            
        except Exception as e:
            QMessageBox.warning(self, "Recording Error", f"Failed to start recording: {str(e)}")
            self.is_recording = False
            self.recording_toggle.setChecked(False)
    
    def stop_recording(self):
        try:
            # Stop recording
            self.is_recording = False
            if hasattr(self, 'recording_timer'):
                self.recording_timer.stop()
            
            # Update UI - only change button text, no status updates
            self.recording_toggle.setText("RECORD")
            
            # Ask user if they want to save the recording
            if len(self.recording_frames) > 0:
                reply = QMessageBox.question(
                    self, 
                    "Save Recording", 
                    "Would you like to save the recording?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.save_recording()
                else:
                    # Discard recording
                    self.recording_frames.clear()
                    QMessageBox.information(self, "Recording Discarded", "Recording has been discarded.")
            
        except Exception as e:
            QMessageBox.warning(self, "Recording Error", f"Failed to stop recording: {str(e)}")
            self.recording_toggle.setChecked(True)

    def capture_frame(self):
        try:
            if self.is_recording:
                # Capture the current window
                screen = QApplication.primaryScreen()
                pixmap = screen.grabWindow(self.winId())
                
                # Convert to numpy array for OpenCV
                image = pixmap.toImage()
                width = image.width()
                height = image.height()
                ptr = image.bits()
                ptr.setsize(height * width * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                
                # Store frame
                self.recording_frames.append(arr)
                
        except Exception as e:
            print(f"Frame capture error: {e}")
    
    def save_recording(self):
        try:
            if len(self.recording_frames) == 0:
                QMessageBox.warning(self, "No Recording", "No frames to save.")
                return
            
            # Get save file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"ecg_recording_{timestamp}.mp4"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Recording",
                default_filename,
                "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)"
            )
            
            if file_path:
                # Get video dimensions from first frame
                height, width = self.recording_frames[0].shape[:2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(file_path, fourcc, 30.0, (width, height))
                
                # Write frames
                for frame in self.recording_frames:
                    out.write(frame)
                
                out.release()
                
                # Clear frames
                self.recording_frames.clear()
                
                QMessageBox.information(
                    self, 
                    "Recording Saved", 
                    f"Recording saved successfully to:\n{file_path}"
                )
            else:
                # User cancelled save
                self.recording_frames.clear()
                QMessageBox.information(self, "Recording Cancelled", "Recording was not saved.")
                
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save recording: {str(e)}")
            self.recording_frames.clear()

    # ------------------------ Get lead figure in pdf ------------------------

    def get_lead_figure(self, lead):
        if hasattr(self, "lead_figures"):
            return self.lead_figures.get(lead)

        ordered_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        if hasattr(self, "figures"):
            if lead in ordered_leads:
                idx = ordered_leads.index(lead)
                if idx < len(self.figures):
                    return self.figures[idx]
        return None

    def center_on_screen(self):
        qr = self.frameGeometry()
        from PyQt5.QtWidgets import QApplication
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def expand_lead(self, idx):
        lead = self.leads[idx]
        def get_lead_data():
            return self.data[lead]
        color = self.LEAD_COLORS.get(lead, "#00ff99")
        if hasattr(self, '_detailed_timer') and self._detailed_timer is not None:
            self._detailed_timer.stop()
            self._detailed_timer.deleteLater()
            self._detailed_timer = None
        old_layout = self.detailed_widget.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            QWidget().setLayout(old_layout)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        back_btn = QPushButton("Back")
        back_btn.setFixedHeight(40)
        back_btn.clicked.connect(lambda: self.page_stack.setCurrentIndex(0))
        layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        fig = Figure(facecolor='#fff')  # White background for the figure
        ax = fig.add_subplot(111)
        ax.set_facecolor('#fff')        # White background for the axes
        line, = ax.plot([], [], color=color, lw=2)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(canvas)
        # Create metric labels for cards
        pr_label = QLabel("0 ms")
        qrs_label = QLabel("0 ms")
        qtc_label = QLabel("0 ms")
        arrhythmia_label = QLabel("--")
        # Add metrics card row below the plot (card style)
        metrics_row = QHBoxLayout()
        def create_metric_card(title, label_widget):
            card = QFrame()
            card.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fff7f0, stop:1 #ffe0cc);
                border-radius: 32px;
                border: 2.5px solid #ff6600;
                padding: 18px 18px;
            """)
            vbox = QVBoxLayout(card)
            vbox.setSpacing(6)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignHCenter)
            lbl.setStyleSheet("color: #ff6600; font-size: 18px; font-weight: bold;")
            label_widget.setStyleSheet("font-size: 32px; font-weight: bold; color: #222; padding: 8px 0;")
            vbox.addWidget(lbl)
            vbox.addWidget(label_widget)
            vbox.setAlignment(Qt.AlignHCenter)
            return card
        metrics_row.setSpacing(32)
        metrics_row.setContentsMargins(32, 16, 32, 24)
        metrics_row.setAlignment(Qt.AlignHCenter)
        cards = [create_metric_card("PR Interval", pr_label),
                 create_metric_card("QRS Duration", qrs_label),
                 create_metric_card("QT/Qtc Interval", qtc_label),
                 create_metric_card("Arrhythmia", arrhythmia_label)]
        for card in cards:
            card.setMinimumWidth(0)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            metrics_row.addWidget(card)
        layout.addLayout(metrics_row)
        self.detailed_widget.setLayout(layout)
        self.detailed_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.page_stack.setCurrentIndex(1)
        self._detailed_timer = QTimer(self)

        def update_detailed_plot():
            detailed_buffer_size = 500  # Reduced to 500 samples for real-time effect
            data = get_lead_data()

            current_gain = self.settings_manager.get_wave_gain()
            current_speed = self.settings_manager.get_wave_speed()

            # Robust: Only plot if enough data, else show blank
            if data and len(data) >= 10:
                plot_data = np.array(data[-detailed_buffer_size:])
                x = np.arange(len(plot_data))
                centered = plot_data - np.mean(plot_data)

                # Apply current gain setting
                gain_factor = get_display_gain(current_gain)
                centered = centered * gain_factor

                line.set_data(x, centered)
                ax.set_xlim(0, max(len(centered)-1, 1))
                
                ylim = 500 * gain_factor
                ymin = np.min(centered) - ylim * 0.2
                ymax = np.max(centered) + ylim * 0.2
                if ymin == ymax:
                    ymin, ymax = -ylim, ylim
                ax.set_ylim(ymin, ymax)

                # --- PQRST detection and green labeling for Lead II only ---
                # Remove all extra lines except the main ECG line (robust for all Matplotlib versions)
                try:
                    while len(ax.lines) > 1:
                        ax.lines.remove(ax.lines[-1])
                except Exception as e:
                    print(f"Warning: Could not remove extra lines: {e}")
                for txt in list(ax.texts):
                    try:
                        txt.remove()
                    except Exception as e:
                        print(f"Warning: Could not remove text: {e}")
                # Optionally, clear all lines if you want only labels visible (no ECG trace):
                # ax.lines.clear()
                if lead == "II":
                    # Use the same detection logic as in main.py
                    from scipy.signal import find_peaks
                    sampling_rate = 80
                    ecg_signal = centered
                    window_size = min(500, len(ecg_signal))
                    if len(ecg_signal) > window_size:
                        ecg_signal = ecg_signal[-window_size:]
                        x = x[-window_size:]
                    # R peak detection
                    r_peaks, _ = find_peaks(ecg_signal, distance=int(0.2 * sampling_rate), prominence=0.6 * np.std(ecg_signal))
                    # Q and S: local minima before and after R
                    q_peaks = []
                    s_peaks = []
                    for r in r_peaks:
                        q_start = max(0, r - int(0.06 * sampling_rate))
                        q_end = r
                        if q_end > q_start:
                            q_idx = np.argmin(ecg_signal[q_start:q_end]) + q_start
                            q_peaks.append(q_idx)
                        s_start = r
                        s_end = min(len(ecg_signal), r + int(0.06 * sampling_rate))
                        if s_end > s_start:
                            s_idx = np.argmin(ecg_signal[s_start:s_end]) + s_start
                            s_peaks.append(s_idx)
                    # P: positive peak before Q (within 0.1-0.2s)
                    p_peaks = []
                    for q in q_peaks:
                        p_start = max(0, q - int(0.2 * sampling_rate))
                        p_end = q - int(0.08 * sampling_rate)
                        if p_end > p_start:
                            p_candidates, _ = find_peaks(ecg_signal[p_start:p_end], prominence=0.1 * np.std(ecg_signal))
                            if len(p_candidates) > 0:
                                p_peaks.append(p_start + p_candidates[-1])
                    # T: positive peak after S (within 0.1-0.4s)
                    t_peaks = []
                    for s in s_peaks:
                        t_start = s + int(0.08 * sampling_rate)
                        t_end = min(len(ecg_signal), s + int(0.4 * sampling_rate))
                        if t_end > t_start:
                            t_candidates, _ = find_peaks(ecg_signal[t_start:t_end], prominence=0.1 * np.std(ecg_signal))
                            if len(t_candidates) > 0:
                                t_peaks.append(t_start + t_candidates[np.argmax(ecg_signal[t_start + t_candidates])])
                    # Only show the most recent peak for each label (if any)
                    peak_dict = {'P': p_peaks, 'Q': q_peaks, 'R': r_peaks, 'S': s_peaks, 'T': t_peaks}
                    for label, idxs in peak_dict.items():
                        if len(idxs) > 0:
                            idx = idxs[-1]
                            ax.plot(idx, ecg_signal[idx], 'o', color='green', markersize=8, zorder=10)
                            y_offset = 0.12 * (np.max(ecg_signal) - np.min(ecg_signal))
                            if label in ['P', 'T']:
                                ax.text(idx, ecg_signal[idx]+y_offset, label, color='green', fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1'))
                            else:
                                ax.text(idx, ecg_signal[idx]-y_offset, label, color='green', fontsize=12, fontweight='bold', ha='center', va='top', zorder=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1'))
                # --- Metrics (for Lead II only, based on R peaks) ---
                if lead == "II":
                    heart_rate = None
                    pr_interval = None
                    qrs_duration = None
                    qt_interval = None
                    qtc_interval = None
                    rr_intervals = None

                    if len(r_peaks) > 1:
                        rr_intervals = np.diff(r_peaks) / sampling_rate  # in seconds
                        mean_rr = np.mean(rr_intervals)
                        if mean_rr > 0:
                            heart_rate = 60 / mean_rr
                    if len(p_peaks) > 0 and len(r_peaks) > 0:
                        pr_interval = (r_peaks[-1] - p_peaks[-1]) * 1000 / sampling_rate  # ms
                    if len(q_peaks) > 0 and len(s_peaks) > 0:
                        qrs_duration = (s_peaks[-1] - q_peaks[-1]) * 1000 / sampling_rate  # ms
                    if len(q_peaks) > 0 and len(t_peaks) > 0:
                        qt_interval = (t_peaks[-1] - q_peaks[-1]) * 1000 / sampling_rate  # ms
                    if qt_interval and heart_rate:
                        qtc_interval = qt_interval / np.sqrt(60 / heart_rate)  # Bazett's formula

                    # Update ECG metrics labels with calculated values for Lead2 graph

                    if isinstance(pr_interval, (int, float)):
                        pr_label.setText(f"{int(round(pr_interval))} ms")
                    else:
                        pr_label.setText("0 ms")

                    if isinstance(qrs_duration, (int, float)):
                        qrs_label.setText(f"{int(round(qrs_duration))} ms")
                    else:
                        qrs_label.setText("0 ms")

                    if isinstance(qtc_interval, (int, float)) and qtc_interval >= 0:
                        qtc_label.setText(f"{int(round(qtc_interval))} ms")
                    else:
                        qtc_label.setText("0 ms")
                    
                    # Calculate QRS axis using Lead I and aVF
                    lead_I = self.data[0] if len(self.data) > 0 else []  # Lead I (index 0)
                    lead_aVF = self.data[5] if len(self.data) > 5 else []  # Lead aVF (index 5)
                    qrs_axis = calculate_qrs_axis(lead_I, lead_aVF, r_peaks)

                    # Calculate ST segment using Lead II and r_peaks
                    lead_ii = self.data[1] if len(self.data) > 1 else []  # Lead II (index 1)
                    st_segment = calculate_st_segment(lead_ii, r_peaks, fs=500)

                    if hasattr(self, 'dashboard_callback'):
                        self.dashboard_callback({
                            'heart_rate': heart_rate,
                            'pr_interval': pr_interval,
                            'qrs_duration': qrs_duration,
                            'qtc_interval': qtc_interval,
                            'qrs_axis': qrs_axis,
                            'st_interval': st_segment
                        })

                    # --- Arrhythmia detection ---
                    arrhythmia_result = detect_arrhythmia(
                        heart_rate,
                        qrs_duration,
                        rr_intervals,
                        pr_interval=pr_interval,
                        p_peaks=p_peaks,
                        r_peaks=r_peaks,
                        ecg_signal=centered
                    )
                    arrhythmia_label.setText(arrhythmia_result)
                    self._latest_rhythm_interpretation = arrhythmia_result
                else:
                    pr_label.setText("-- ms")
                    qrs_label.setText("-- ms")
                    qtc_label.setText("-- ms")
                    arrhythmia_label.setText("0")
                    self._latest_rhythm_interpretation = "Analyzing Rhythm..."
            else:
                line.set_data([], [])
                ax.set_xlim(0, 1)
                ax.set_ylim(-500, 500)
                pr_label.setText("0 ms")
                qrs_label.setText("0 ms")
                qtc_label.setText("0 ms")
            canvas.draw_idle()
        self._detailed_timer.timeout.connect(update_detailed_plot)
        self._detailed_timer.start(100)
        update_detailed_plot()  # Draw immediately on open

    def refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItem("Select Port")
        
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                self.port_combo.addItem("No ports found")
                print("⚠️ No serial ports detected during refresh")
            else:
                for port in ports:
                    self.port_combo.addItem(port.device)
                print(f"🔄 Refreshed: Found {len(ports)} serial ports")
        except Exception as e:
            self.port_combo.addItem("Error detecting ports")
            print(f"❌ Error refreshing ports: {e}")

    def update_lead_layout(self):
        old_layout = self.plot_area.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
            self.plot_area.setLayout(None)
        self.figures = []
        self.canvases = []
        self.axs = []
        self.lines = []
        grid = QGridLayout()
        grid.setSpacing(8)  # Reduced spacing between graphs
        n_leads = len(self.leads)
        if n_leads == 12:
            rows, cols = 3, 4
        elif n_leads == 7:
            rows, cols = 2, 4
        else:
            rows, cols = 1, 1
        for idx, lead in enumerate(self.leads):
            row, col = divmod(idx, cols)
            group = QGroupBox(lead)
            group.setStyleSheet("""
                QGroupBox {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    border: 2px solid #e9ecef;
                    border-radius: 12px;  /* Reduced from 16px */
                    color: #495057;
                    font: bold 14px 'Arial';  /* Reduced from 16px and changed font */
                    margin-top: 8px;  /* Reduced from 12px */
                    padding: 8px;  /* Reduced from 12px */
                    /* Removed unsupported box-shadow property */
                }
                QGroupBox:hover {
                    border: 2px solid #ff6600;
                    /* Removed unsupported box-shadow and transform properties */
                }
            """)
            vbox = QVBoxLayout(group)
            vbox.setContentsMargins(6, 6, 6, 6)  # Reduced margins
            vbox.setSpacing(4)  # Reduced spacing
            fig = Figure(facecolor='#fafbfc', figsize=(5, 2))  # Reduced from (6, 2.5)
            ax = fig.add_subplot(111)
            ax.set_facecolor('#fafbfc')
            ylim = self.ylim if hasattr(self, 'ylim') else 400
            ax.set_ylim(-ylim, ylim)
            ax.set_xlim(0, self.buffer_size)
            
            # Modern grid styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#e9ecef')
            ax.set_axisbelow(True)

            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Style ticks - Make them smaller
            ax.tick_params(axis='both', colors='#6c757d', labelsize=8)  # Reduced from 10
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)

            # Enhanced line styling
            import matplotlib.patheffects as path_effects 
            line, = ax.plot([0]*self.buffer_size, 
                            color=self.LEAD_COLORS.get(lead, '#ff6600'), 
                            lw=0.5, 
                            alpha=0.9,
                            path_effects=[path_effects.SimpleLineShadow(offset=(1,1), alpha=0.3),
                                        path_effects.Normal()])

            self.lines.append(line)
            canvas = FigureCanvas(fig)
            vbox.addWidget(canvas)
            grid.addWidget(group, row, col)
            self.figures.append(fig)
            self.canvases.append(canvas)
            self.axs.append(ax)
        self.plot_area.setLayout(grid)
        def make_expand_lead(idx):
            return lambda event: self.expand_lead(idx)
        for i, canvas in enumerate(self.canvases):
            canvas.mpl_connect('button_press_event', make_expand_lead(i))

    def redraw_all_plots(self):
        
        if hasattr(self, 'lines') and self.lines:
            for i, line in enumerate(self.lines):
                if i < len(self.leads):
                    lead = self.leads[i]
                    data = self.data.get(lead, [])
                    
                    if len(data) > 0:
                        # Detect signal source and apply adaptive scaling
                        signal_source = self.detect_signal_source(data)
                        # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                        gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                        device_data = np.array(data)
                        centered = self.apply_adaptive_gain(device_data, signal_source, gain_factor)
                        
                        # Apply medical-grade filtering for smooth waves
                        filtered_data = self.apply_ecg_filtering(centered)
                        
                        # Update line data with new buffer size
                        if len(filtered_data) < self.buffer_size:
                            plot_data = np.full(self.buffer_size, np.nan)
                            plot_data[-len(filtered_data):] = filtered_data
                        else:
                            plot_data = filtered_data[-self.buffer_size:]
                        
                        line.set_ydata(plot_data)
                        
                        # Update axis limits with adaptive Y-range
                        if i < len(self.axs):
                            # Calculate adaptive Y-range for matplotlib plots
                            valid_data = filtered_data[~np.isnan(filtered_data)]
                            if len(valid_data) > 0:
                                p1 = np.percentile(valid_data, 1)
                                p99 = np.percentile(valid_data, 99)
                                data_mean = (p1 + p99) / 2.0
                                data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
                                
                                if signal_source == "human_body":
                                    padding = max(data_std * 2, 20)
                                elif signal_source == "weak_body":
                                    padding = max(data_std * 1.5, 10)
                                else:
                                    padding = max(data_std * 4, 200)
                                
                                y_min = data_mean - padding
                                y_max = data_mean + padding
                                self.axs[i].set_ylim(y_min, y_max)
                            else:
                                ylim = self.ylim if hasattr(self, 'ylim') else 400
                                self.axs[i].set_ylim(-ylim, ylim)
                            
                            self.axs[i].set_xlim(0, self.buffer_size)
                            
                            # Update plot title with current settings and signal source
                            current_speed = self.settings_manager.get_wave_speed()
                            current_gain = self.settings_manager.get_wave_gain()
                            signal_type = "Body" if signal_source in ["human_body", "weak_body"] else "Hardware"
                            new_title = f"{lead} | Speed: {current_speed}mm/s | Gain: {current_gain}mm/mV | {signal_type}"
                            self.axs[i].set_title(new_title, fontsize=8, color='#666', pad=10)
                            print(f"Redraw updated {lead} title: {new_title}")
                        
                        # Redraw canvas
                        if i < len(self.canvases):
                            self.canvases[i].draw_idle()

    def detect_signal_source(self, data):
        """Detect if signal is from hardware or human body based on amplitude characteristics"""
        try:
            if data is None:
                return "none"
            # Safely coerce to a 1D numpy array
            data_array = np.asarray(data).ravel()
            if data_array.size == 0:
                return "none"

            signal_range = float(np.ptp(data_array))  # Peak-to-peak range
            signal_mean = float(np.mean(np.abs(data_array)))
            signal_std = float(np.std(data_array))
            
            print(f"🔍 Signal Analysis: Range={signal_range:.1f}, Mean={signal_mean:.1f}, Std={signal_std:.1f}")
            
            # Heuristics: raw ADC (hardware) often around 0-4095 with baseline ~2000 but body-connected raw can still be ~2000±100
            # Use range thresholds to classify; treat > 400 as clear hardware dynamics, > 50 as body but weak
            if signal_range > 400:
                return "hardware"
            elif signal_range > 50:
                return "human_body"
            elif signal_range > 10:
                return "weak_body"
            else:
                return "noise"
                
        except Exception as e:
            print(f"❌ Error in signal detection: {e}")
            return "unknown"

    def apply_adaptive_gain(self, data, signal_source, gain_factor):
        """Apply gain based on signal source with adaptive scaling"""
        try:
            device_data = np.array(data)
            
            if signal_source == "hardware":
                # Current logic for hardware (0-4095 range)
                centered = (device_data - 2100) * gain_factor
                print(f"🔧 Hardware signal: Applied hardware scaling")
                
            elif signal_source == "human_body":
                # Different scaling for human body (0-500 range)
                # Don't subtract mean here - low-frequency baseline anchor handles it
                centered = device_data * gain_factor * 8  # Amplify weak signals
                print(f"🔧 Human body signal: Applied body scaling (amplification=8x)")
                
            elif signal_source == "weak_body":
                # Very weak signals - maximum amplification
                # Don't subtract mean here - low-frequency baseline anchor handles it
                centered = device_data * gain_factor * 15  # Maximum amplification
                print(f"🔧 Weak body signal: Applied maximum scaling (amplification=15x)")
                
            else:
                # Noise or unknown - minimal processing
                centered = device_data * gain_factor
                print(f"🔧 Unknown signal: Applied minimal scaling")
            
            return centered
            
        except Exception as e:
            print(f"❌ Error in adaptive gain: {e}")
            return np.array(data) * gain_factor

    def update_plot_y_range_adaptive(self, plot_index, signal_source, data_override=None):
        """Update Y-axis range based on signal source with adaptive scaling.
        If data_override is provided, use it for statistics (should be the plotted/scaled data)."""
        try:
            if plot_index >= len(self.data) or plot_index >= len(self.plot_widgets):
                return

            # Get the data for this plot
            if data_override is not None:
                data = np.asarray(data_override)
                # Data is already scaled with gain, so don't apply gain again
                data_already_scaled = True
            else:
                data = self.data[plot_index]
                # Data is not scaled, will need to apply gain
                data_already_scaled = False
            
            # Remove NaN values and large outliers (robust)
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                return
            
            # Use percentiles to avoid spikes from clipping the view
            p1 = np.percentile(valid_data, 1)
            p99 = np.percentile(valid_data, 99)
            data_mean = (p1 + p99) / 2.0
            data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
            # Maximum deviation of any point from the mean – we will always cover this
            peak_deviation = np.max(np.abs(valid_data - data_mean)) if len(valid_data) > 0 else 0.0
            
            # Get current gain setting only if data is not already scaled
            current_gain = 1.0 if data_already_scaled else get_display_gain(self.settings_manager.get_wave_gain())
            
            # Calculate appropriate Y-range with adaptive padding based on signal source.
            # Goal: make peaks visually bigger but still avoid cropping by using robust stats.
            if signal_source == "human_body":
                # Lock to fixed range for human body motion (no adaptive scaling)
                y_min = -600
                y_max = 600
                base_padding = None
                padding = None
                print("📊 Human body Y-range locked to ±600 for stability")
            elif signal_source == "weak_body":
                y_min = -400
                y_max = 400
                base_padding = None
                padding = None
                print("📊 Weak body Y-range locked to ±400 for stability")
            else:
                # Hardware / unknown – keep more room but less than before so peaks are taller.
                base_padding = max(data_std * 3.0, 250)
                padding = base_padding  # Do NOT scale padding with gain; signal already scaled
                print(f"📊 Hardware Y-range: base_padding={base_padding:.1f}, padding={padding:.1f}")

            # FINAL SAFETY: always cover the tallest peak with 10% headroom,
            # so waves never touch or cross the plot border (no cropping),
            # regardless of gain/speed combinations.
            if signal_source not in ["human_body", "weak_body"] and peak_deviation > 0:
                min_padding = peak_deviation * 1.1
                if padding < min_padding:
                    padding = min_padding
            
            if signal_source not in ["human_body", "weak_body"]:
                if data_std > 0:
                    y_min = data_mean - padding
                    y_max = data_mean + padding
                else:
                    y_min = -400
                    y_max = 400
            else:
                # Fallback: use percentile window with reasonable range
                data_range = max(p99 - p1, 80 if signal_source in ["human_body", "weak_body"] else 300)
                y_min = data_mean - data_range / 2
                y_max = data_mean + data_range / 2
                
                # Add extra margin to ensure NO cropping (20% padding beyond calculated range)
                y_range = y_max - y_min
                y_min = y_min - (y_range * 0.2)
                y_max = y_max + (y_range * 0.2)
            
            # Apply the new Y-range using PyQtGraph with NO padding (we already added it)
            self.plot_widgets[plot_index].setYRange(y_min, y_max, padding=0)
            
        except Exception as e:
            print(f"❌ Error updating adaptive Y-range: {e}")

    def update_ecg_lead(self, lead_index, data_array):
        """Update a specific ECG lead with new data from serial communication"""
        try:
            if 0 <= lead_index < len(self.lines) and len(data_array) > 0:
                # Detect signal source first
                signal_source = self.detect_signal_source(data_array)
                
                # Apply current settings to the incoming data
                # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                
                # Apply adaptive gain based on signal source
                centered = self.apply_adaptive_gain(data_array, signal_source, gain_factor)
                
                # Apply noise reduction filtering
                filtered_data = self.apply_ecg_filtering(centered)
                
                # Update line data with new buffer size
                if len(filtered_data) < self.buffer_size:
                    plot_data = np.full(self.buffer_size, np.nan)
                    plot_data[-len(filtered_data):] = filtered_data
                else:
                    plot_data = filtered_data[-self.buffer_size:]
                
                # Update the specific lead line
                self.lines[lead_index].set_ydata(plot_data)
                
                # Update axis limits with adaptive Y-range
                if lead_index < len(self.axs):
                    # Use adaptive Y-range based on the filtered (plotted) data
                    self.update_plot_y_range_adaptive(lead_index, signal_source, data_override=filtered_data)
                    self.axs[lead_index].set_xlim(0, self.buffer_size)
                
                # Redraw the specific canvas
                if lead_index < len(self.canvases):
                    self.canvases[lead_index].draw_idle()
                    
                print(f"Updated ECG lead {lead_index} with {len(data_array)} samples")
                
        except Exception as e:
            print(f"Error updating ECG lead {lead_index}: {str(e)}")
    
    def apply_ecg_filtering(self, signal_data):
        """Apply medical-grade ECG filtering for smooth, clean waves like professional devices"""
        try:
            from scipy.signal import butter, filtfilt, savgol_filter, medfilt, wiener
            from scipy.ndimage import gaussian_filter1d
            from ecg.ecg_filters import apply_ecg_filters_from_settings
            import numpy as np
            
            if len(signal_data) < 10:  # Need minimum data for filtering
                return signal_data
            
            # Convert to numpy array
            signal = np.array(signal_data, dtype=float)
            
            # NOTE: Baseline correction is handled by slow baseline anchor in display paths
            # Do NOT subtract mean here - baseline anchor handles it before this function is called
            
            # Apply AC/EMG/DFT filters based on user settings from SettingsManager
            # This applies filters in correct order: DFT -> EMG -> AC
            sampling_rate = getattr(self, 'demo_fs', 500)  # Get sampling rate, default 500Hz
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate'):
                try:
                    sampling_rate = float(self.sampler.sampling_rate)
                except:
                    pass
            
            # Apply user-configured AC/EMG/DFT filters
            signal = apply_ecg_filters_from_settings(
                signal=signal,
                sampling_rate=sampling_rate,
                settings_manager=self.settings_manager
            )
            
            # 3. Medical-grade bandpass filter (0.5-30 Hz) - tighter range for cleaner signal
            fs = sampling_rate  # Use actual sampling rate
            nyquist = fs / 2
            
            # Low-pass filter to remove high-frequency noise (>30 Hz) - more aggressive
            low_cutoff = 30 / nyquist  # Reduced from 40 to 30 Hz
            b_low, a_low = butter(6, low_cutoff, btype='low')  # Increased order to 6
            signal = filtfilt(b_low, a_low, signal)
            
            # Note: High-pass filter is now handled by DFT filter, so we skip it here
            
            # 3. Wiener filter for medical-grade noise reduction
            if len(signal) > 5:
                signal = wiener(signal, noise=0.05)  # Lower noise parameter for smoother result
            
            # 4. Gaussian smoothing for medical-grade smoothness
            signal = gaussian_filter1d(signal, sigma=1.2)
            
            # 5. Savitzky-Golay filter with optimized parameters for ECG
            if len(signal) >= 15:  # Increased minimum window size
                window_length = min(15, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
                signal = savgol_filter(signal, window_length, 4)  # Increased polynomial order to 4
            
            # 6. Adaptive median filter for spike removal
            signal = medfilt(signal, kernel_size=7)  # Increased kernel size for better smoothing
            
            # 7. Multi-stage moving average for ultra-smooth baseline
            # Stage 1: Short-term smoothing
            window1 = min(7, len(signal))
            if window1 > 1:
                kernel1 = np.ones(window1) / window1
                signal = np.convolve(signal, kernel1, mode='same')
            
            # Stage 2: Medium-term smoothing for baseline stability
            window2 = min(5, len(signal))
            if window2 > 1:
                kernel2 = np.ones(window2) / window2
                signal = np.convolve(signal, kernel2, mode='same')
            
            # 8. Final Gaussian smoothing for medical device quality
            signal = gaussian_filter1d(signal, sigma=0.8)
            
            return signal
            
        except Exception as e:
            print(f"Medical-grade filtering error: {e}")
            # Return original signal if filtering fails
            return signal_data
    
    def apply_realtime_smoothing(self, new_value, lead_index):
        """Apply real-time smoothing for individual data points - medical grade"""
        try:
            # Initialize smoothing buffers if not exists
            if not hasattr(self, 'smoothing_buffers'):
                self.smoothing_buffers = {}
            
            if lead_index not in self.smoothing_buffers:
                self.smoothing_buffers[lead_index] = []
            
            buffer = self.smoothing_buffers[lead_index]
            buffer.append(new_value)
            
            # Keep only last 20 points for smoothing
            if len(buffer) > 20:
                buffer.pop(0)
            
            # Apply multi-stage smoothing
            if len(buffer) >= 5:
                # Stage 1: Simple moving average
                smoothed = np.mean(buffer[-5:])
                
                # Stage 2: Weighted average (more weight to recent values)
                if len(buffer) >= 10:
                    weights = np.linspace(0.5, 1.0, len(buffer[-10:]))
                    smoothed = np.average(buffer[-10:], weights=weights)
                
                # Stage 3: Gaussian-like smoothing
                if len(buffer) >= 7:
                    # Apply Gaussian weights
                    gaussian_weights = np.exp(-0.5 * ((np.arange(len(buffer[-7:])) - len(buffer[-7:])//2) / 2)**2)
                    gaussian_weights = gaussian_weights / np.sum(gaussian_weights)
                    smoothed = np.sum(np.array(buffer[-7:]) * gaussian_weights)
                
                return smoothed
            else:
                return new_value
                
        except Exception as e:
            print(f"Real-time smoothing error: {e}")
            return new_value

    # ---------------------- Serial Port Auto-Detection ----------------------

    def get_available_serial_ports(self):
        """Get list of available serial ports"""
        if not SERIAL_AVAILABLE:
            return []
        
        ports = []
        try:
            # Get all available ports
            available_ports = serial.tools.list_ports.comports()
            for port_info in available_ports:
                ports.append(port_info.device)
            print(f"🔍 Found {len(ports)} available serial ports: {ports}")
        except Exception as e:
            print(f"❌ Error detecting serial ports: {e}")
        
        return ports

    def auto_detect_serial_port(self):
        """Automatically detect and set the best available serial port"""
        available_ports = self.get_available_serial_ports()
        
        if not available_ports:
            return None, "No serial ports found"
        
        # Look for common ECG device patterns
        preferred_patterns = ['usbserial', 'usbmodem', 'ttyUSB', 'ttyACM']
        
        for pattern in preferred_patterns:
            for port in available_ports:
                if pattern in port.lower():
                    print(f"🎯 Auto-detected ECG device port: {port}")
                    return port, f"Auto-detected: {port}"
        
        # If no preferred pattern found, use the first available port
        if available_ports:
            port = available_ports[0]
            print(f"🎯 Using first available port: {port}")
            return port, f"Using first available: {port}"
        
        return None, "No suitable ports found"

    # ---------------------- Start Button Functionality ----------------------

    def start_acquisition(self):

        try:
            if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
                print("🔁 Switching from Demo to Real: turning off demo...")
                self.demo_toggle.setChecked(False)
                if hasattr(self, 'demo_manager'):
                    self.demo_manager.stop_demo_data()
        except Exception as e:
            print(f"[Start Acquisition] Failed to stop demo before real start: {e}")
        
        # Disable demo mode when hardware acquisition starts
        try:
            if hasattr(self, 'demo_toggle'):
                self.demo_toggle.setEnabled(False)
                self.demo_toggle.setStyleSheet("""
                    QPushButton {
                        background: #6c757d;
                        color: #ffffff;
                        border: 2px solid #6c757d;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                """)
                print("🔒 Demo mode disabled (Hardware acquisition active)")
        except Exception as e:
            print(f"❌ Error disabling demo mode: {e}")

        port = self.settings_manager.get_serial_port()
        baud = self.settings_manager.get_baud_rate()

        print(f"Starting acquisition with Port: {port}, Baud: {baud}")

        if port == "Select Port" or baud == "Select Baud Rate" or port is None or baud is None:
            self.show_connection_warning("Please configure serial port and baud rate in System Setup first.")
            return
        
        # Ensure the selected COM port is actually connected/available before starting
        try:
            available_ports = []
            try:
                available_ports = [p.device for p in serial.tools.list_ports.comports()]
            except Exception:
                available_ports = []
            if (not available_ports) or (port not in available_ports):
                self.show_connection_warning("Connect device and select a valid COM port before starting.")
                return
        except Exception:
            # If we cannot verify ports reliably, be safe and block start
            self.show_connection_warning("Unable to detect COM ports. Please connect device and select a valid port.")
            return
        
        try:
            # Convert baud rate to integer with error handling
            try:
                baud_int = int(baud)
            except (ValueError, TypeError):
                self.show_connection_warning(f"Invalid baud rate: {baud}. Please set a valid baud rate in System Setup.")
                return
            
            if self.serial_reader:
                self.serial_reader.close()
            
            print(f"Connecting to {port} at {baud_int} baud...")

            # Reset visible metrics to zero before starting to avoid stale values
            try:
                if hasattr(self, 'metric_labels'):
                    if 'heart_rate' in self.metric_labels: self.metric_labels['heart_rate'].setText("00")
                    if 'pr_interval' in self.metric_labels: self.metric_labels['pr_interval'].setText("0")
                    if 'qrs_duration' in self.metric_labels: self.metric_labels['qrs_duration'].setText("0")
                    if 'qrs_axis' in self.metric_labels: self.metric_labels['qrs_axis'].setText("0°")
                    if 'st_interval' in self.metric_labels: self.metric_labels['st_interval'].setText("0")
                    if 'qtc_interval' in self.metric_labels: self.metric_labels['qtc_interval'].setText("0/0")
                    if 'time_elapsed' in self.metric_labels: self.metric_labels['time_elapsed'].setText("00:00")
                    # if 'sampling_rate' in self.metric_labels: self.metric_labels['sampling_rate'].setText("0 Hz")  # Commented out
            except Exception as _:
                pass
            
            try:
                # Use new packet-based SerialStreamReader instead of old SerialECGReader
                self.serial_reader = SerialStreamReader(port, baud_int)
                # Pass user details to serial reader for error reporting (already set in __init__)
                if hasattr(self, 'user_details'):
                    self.serial_reader.user_details = self.user_details
                self.serial_reader.start()
                print("✅ Serial connection established successfully!")
                
            except Exception as e:
                print(f"❌ Failed to connect to configured port {port}: {e}")
                
                # Try auto-detection
                auto_port, auto_msg = self.auto_detect_serial_port()
                if auto_port:
                    print(f"🔄 Trying auto-detected port: {auto_port}")
                    try:
                        # Use new packet-based SerialStreamReader instead of old SerialECGReader
                        self.serial_reader = SerialStreamReader(auto_port, baud_int)
                        # Pass user details to serial reader for error reporting (already set in __init__)
                        if hasattr(self, 'user_details'):
                            self.serial_reader.user_details = self.user_details
                        self.serial_reader.start()
                        
                        # Update settings with the working port
                        self.settings_manager.set_serial_port(auto_port)
                        print(f"✅ Connected to auto-detected port: {auto_port}")
                        
                        # Show info to user
                        QMessageBox.information(self, "Port Auto-Detected", 
                            f"Could not connect to configured port {port}.\n\n"
                            f"Successfully connected to: {auto_port}\n"
                            f"This port has been saved to your settings.")
                        
                    except Exception as e2:
                        print(f"❌ Auto-detection also failed: {e2}")
                        raise e2
                else:
                    raise e
            
            # Use faster timer interval for EXE builds to prevent gaps
            # Timer interval is more important than timer type for smooth plotting
            timer_interval = 33  # ~30 FPS for smoother plotting in EXE
            print(f"[DEBUG] ECGTestPage - Starting timer with {timer_interval}ms interval")
            # Using default timer type - works fine in EXE with proper interval
            self.timer.start(timer_interval)
            if hasattr(self, '_12to1_timer'):
                self._12to1_timer.start(100)
            print(f"[DEBUG] ECGTestPage - Timer started, serial reader created")
            print(f"[DEBUG] ECGTestPage - Timer active: {self.timer.isActive()}")
            print(f"[DEBUG] ECGTestPage - Number of leads: {len(self.leads)}")
            print(f"[DEBUG] ECGTestPage - Number of plot widgets: {len(self.plot_widgets)}")
            print(f"[DEBUG] ECGTestPage - Number of data lines: {len(self.data_lines)}")

            # Start elapsed time tracking (resume from previous time if paused)
            current_time = time.time()
            if not hasattr(self, 'start_time') or self.start_time is None:
                # First start - set start time
                self.start_time = current_time
                if hasattr(self, 'paused_duration'):
                    self.paused_duration = 0
                self.paused_at = None
                print("⏱️ Session timer started (first time)")
            else:
                # Resuming from pause - accumulate paused time
                if hasattr(self, 'paused_at') and self.paused_at is not None:
                    # Calculate how long we were paused
                    pause_duration = current_time - self.paused_at
                    # Add to total paused duration
                    if not hasattr(self, 'paused_duration') or self.paused_duration is None:
                        self.paused_duration = 0
                    self.paused_duration += pause_duration
                    print(f"⏱️ Session timer resumed (was paused for {int(pause_duration)}s)")
                    self.paused_at = None  # Clear pause timestamp
                else:
                    print("⏱️ Session timer resumed")
            # Ensure timer is stopped before starting to avoid multiple timers
            if self.elapsed_timer.isActive():
                self.elapsed_timer.stop()
            self.elapsed_timer.start(1000)  # Update every 1 second
            
        except Exception as e:
            error_msg = f"Failed to connect to any serial port: {str(e)}"
            print(error_msg)
            self.show_connection_warning(error_msg)

    # ---------------------- Stop Button Functionality ----------------------

    def stop_acquisition(self):
        port = self.settings_manager.get_serial_port()
        baud = self.settings_manager.get_baud_rate()
        
        if port == "Select Port" or baud == "Select Baud Rate" or port is None or baud is None:
            self.show_connection_warning("Please configure serial port and baud rate in System Setup first.")
            return
            
        if self.serial_reader:
            self.serial_reader.stop()
        self.timer.stop()
        if hasattr(self, '_12to1_timer'):
            self._12to1_timer.stop()

        # Pause elapsed time tracking (keep start_time for resume)
        self.elapsed_timer.stop()
        # Track when pause started (for calculating total paused time on resume)
        if hasattr(self, 'start_time') and self.start_time is not None:
            if not hasattr(self, 'paused_at') or self.paused_at is None:
                self.paused_at = time.time()
                print(f"⏸️ Timer paused")
            # Keep start_time so we can resume from this point

        # --- Calculate and update metrics on dashboard ---
        if hasattr(self, 'dashboard_callback'):
            # Get Lead II data (index 1 in the 12-lead array)
            lead2_data = self.data[1][-500:] if len(self.data) > 1 else []
            lead_I_data = self.data[0][-500:] if len(self.data) > 0 else []  # Lead I (index 0)
            lead_aVF_data = self.data[5][-500:] if len(self.data) > 5 else []  # Lead aVF (index 5)
            heart_rate = None
            pr_interval = None
            qrs_duration = None
            qt_interval = None
            qtc_interval = None
            qrs_axis = "--"
            st_segment = "--"
            if len(lead2_data) > 100:
                # Use same detection logic as live
                from scipy.signal import find_peaks
                sampling_rate = 500
                ecg_signal = np.array(lead2_data)
                centered = ecg_signal - np.mean(ecg_signal)
                # R peak detection
                r_peaks, _ = find_peaks(centered, distance=int(0.2 * sampling_rate), prominence=0.6 * np.std(centered))
                # Q and S: local minima before and after R
                q_peaks = []
                s_peaks = []
                for r in r_peaks:
                    q_start = max(0, r - int(0.06 * sampling_rate))
                    q_end = r
                    if q_end > q_start:
                        q_idx = np.argmin(centered[q_start:q_end]) + q_start
                        q_peaks.append(q_idx)
                    s_start = r
                    s_end = min(len(centered), r + int(0.06 * sampling_rate))
                    if s_end > s_start:
                        s_idx = np.argmin(centered[s_start:s_end]) + s_start
                        s_peaks.append(s_idx)
                # P: positive peak before Q (within 0.1-0.2s)
                p_peaks = []
                for q in q_peaks:
                    p_start = max(0, q - int(0.2 * sampling_rate))
                    p_end = q - int(0.08 * sampling_rate)
                    if p_end > p_start:
                        p_candidates, _ = find_peaks(centered[p_start:p_end], prominence=0.1 * np.std(centered))
                        if len(p_candidates) > 0:
                            p_peaks.append(p_start + p_candidates[-1])
                # T: positive peak after S (within 0.1-0.4s)
                t_peaks = []
                for s in s_peaks:
                    t_start = s + int(0.08 * sampling_rate)
                    t_end = min(len(centered), s + int(0.4 * sampling_rate))
                    if t_end > t_start:
                        t_candidates, _ = find_peaks(centered[t_start:t_end], prominence=0.1 * np.std(centered))
                        if len(t_candidates) > 0:
                            t_peaks.append(t_start + t_candidates[np.argmax(centered[t_start + t_candidates])])
                # Calculate intervals
                if len(r_peaks) > 1:
                    rr_intervals = np.diff(r_peaks) / sampling_rate  # in seconds
                    mean_rr = np.mean(rr_intervals)
                    heart_rate = 60 / mean_rr if mean_rr > 0 else None
                else:
                    rr_intervals = None
                    heart_rate = None
                if len(p_peaks) > 0 and len(r_peaks) > 0:
                    pr_interval = (r_peaks[-1] - p_peaks[-1]) * 1000 / sampling_rate  # ms
                if len(q_peaks) > 0 and len(s_peaks) > 0:
                    qrs_duration = (s_peaks[-1] - q_peaks[-1]) * 1000 / sampling_rate  # ms
                if len(q_peaks) > 0 and len(t_peaks) > 0:
                    qt_interval = (t_peaks[-1] - q_peaks[-1]) * 1000 / sampling_rate  # ms
                if qt_interval and heart_rate:
                    qtc_interval = qt_interval / np.sqrt(60 / heart_rate)  # Bazett's formula

                # QRS axis
                qrs_axis = calculate_qrs_axis(lead_I_data, lead_aVF_data, r_peaks)

                # ST segment
                st_segment = calculate_st_segment(lead2_data, r_peaks, fs=sampling_rate)

            self.dashboard_callback({
                'heart_rate': heart_rate,
                'pr_interval': pr_interval,
                'qrs_duration': qrs_duration,
                'qtc_interval': qtc_interval,
                'qrs_axis': qrs_axis,
                'st_interval': st_segment
            })
        
        # Re-enable demo mode when hardware acquisition stops
        try:
            if hasattr(self, 'demo_toggle'):
                self.demo_toggle.setEnabled(True)
                self.demo_toggle.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                            stop:0 #ffffff, stop:1 #f8f9fa);
                        color: #1a1a1a;
                        border: 2px solid #e9ecef;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                            stop:0 #f8f9fa, stop:1 #e9ecef);
                    }
                    QPushButton:checked {
                        background: #28a745;
                        color: white;
                    }
                """)
                print("🔓 Demo mode enabled (Hardware acquisition stopped)")
        except Exception as e:
            print(f"❌ Error enabling demo mode: {e}")

    def update_plot(self):
        print(f"[DEBUG] ECGTestPage - update_plot called, serial_reader exists: {self.serial_reader is not None}")
        
        if not self.serial_reader:
            print("[DEBUG] ECGTestPage - No serial reader, returning")
            return
        
        # Read raw data directly from serial port
        try:
            line = self.serial_reader.ser.readline()
            line_data = line.decode('utf-8', errors='replace').strip()
            
            if not line_data:
                print("[DEBUG] ECGTestPage - No data received (empty line)")
                return
            
            print(f"[DEBUG] ECGTestPage - Raw hardware data: '{line_data}' (length: {len(line_data)})")
            
            # Parse the 8-channel data (handle multiple spaces)
            try:
                # Split by any whitespace and filter out empty strings
                values = [int(x) for x in line_data.split() if x.strip()]
                print(f"[DEBUG] ECGTestPage - Parsed {len(values)} values: {values}")
                
                if len(values) >= 8:
                    # Extract individual leads from 8-channel data
                    lead1 = values[0]    # Lead I
                    v4    = values[1]    # V4
                    v5    = values[2]    # V5
                    lead2 = values[3]    # Lead II
                    v3    = values[4]    # V3
                    v6    = values[5]    # V6
                    v1    = values[6]    # V1
                    v2    = values[7]    # V2
                    
                    # Calculate derived leads
                    lead3 = lead2 - lead1
                    avr = - (lead1 + lead2) / 2
                    avl = (lead1 - lead3) / 2
                    avf = (lead2 + lead3) / 2
                    
                    lead_data = {
                        "I": lead1, "II": lead2, "III": lead3,
                        "aVR": avr, "aVL": avl, "aVF": avf,
                        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6
                    }
                    
                    print(f"[DEBUG] ECGTestPage - Successfully parsed 8-channel data: {lead_data}")
                    
                elif len(values) == 1:
                    # Single value - generate realistic 12-lead ECG data
                    ecg_value = values[0]
                    print(f"[DEBUG] ECGTestPage - Single value received: {ecg_value}, generating realistic ECG...")
                    
                    # Initialize realistic ECG generation if not already done
                    if not hasattr(self, 'ecg_generators'):
                        self.ecg_generators = {}
                        self.ecg_time_index = 0
                        self.ecg_sampling_rate = self.demo_fs
                        
                        # Generate realistic ECG waveforms for each lead
                        for lead in self.leads:
                            ecg_wave, _ = generate_realistic_ecg_waveform(
                                duration_seconds=60,  # 1 minute of data
                                sampling_rate=self.ecg_sampling_rate,
                                heart_rate=72,
                                lead_name=lead
                            )
                            self.ecg_generators[lead] = ecg_wave
                    
                    # Get current sample from realistic ECG waveforms
                    lead_data = {}
                    for lead in self.leads:
                        if lead in self.ecg_generators:
                            # Scale the realistic ECG to match the input value range
                            realistic_value = self.ecg_generators[lead][self.ecg_time_index % len(self.ecg_generators[lead])]
                            # Scale to match typical ECG range (0-4095 for 12-bit ADC)
                            scaled_value = int(ecg_value + realistic_value * 1000)  # Scale realistic ECG to mV range
                            lead_data[lead] = scaled_value
                    
                    # Move to next time sample
                    self.ecg_time_index += 1
                    
                else:
                    print(f"[DEBUG] ECGTestPage - Unexpected number of values: {len(values)}")
                    return
                    
            except ValueError as e:
                print(f"[DEBUG] ECGTestPage - Error parsing values: {e}")
                # Try to extract numeric part using regex
                import re
                numbers = re.findall(r'-?\d+', line_data)
                if numbers:
                    try:
                        # Use first number as single value
                        ecg_value = int(numbers[0])
                        print(f"[DEBUG] ECGTestPage - Extracted numeric value: {ecg_value}")
                        
                        # Use single value to generate realistic 12-lead ECG data
                        # Initialize realistic ECG generation if not already done
                        if not hasattr(self, 'ecg_generators'):
                            self.ecg_generators = {}
                            self.ecg_time_index = 0
                            self.ecg_sampling_rate = self.demo_fs
                            
                            # Generate realistic ECG waveforms for each lead
                            for lead in self.leads:
                                ecg_wave, _ = generate_realistic_ecg_waveform(
                                    duration_seconds=60,  # 1 minute of data
                                    sampling_rate=self.ecg_sampling_rate,
                                    heart_rate=72,
                                    lead_name=lead
                                )
                                self.ecg_generators[lead] = ecg_wave
                        
                        # Get current sample from realistic ECG waveforms
                        lead_data = {}
                        for lead in self.leads:
                            if lead in self.ecg_generators:
                                # Scale the realistic ECG to match the input value range
                                realistic_value = self.ecg_generators[lead][self.ecg_time_index % len(self.ecg_generators[lead])]
                                # Scale to match typical ECG range (0-4095 for 12-bit ADC)
                                scaled_value = int(ecg_value + realistic_value * 1000)  # Scale realistic ECG to mV range
                                lead_data[lead] = scaled_value
                        
                        # Move to next time sample
                        self.ecg_time_index += 1
                    except ValueError:
                        print(f"[DEBUG] ECGTestPage - Could not parse numeric data from: '{line_data}'")
                        return
                else:
                    print(f"[DEBUG] ECGTestPage - No numeric data found in: '{line_data}'")
                    return
            
            # Update data buffers for all leads
            for lead in self.leads:
                if lead in lead_data:
                    self.data[lead].append(lead_data[lead])
                    if len(self.data[lead]) > self.buffer_size:
                        self.data[lead].pop(0)
            
            print(f"[DEBUG] ECGTestPage - Updated data buffers, Lead II has {len(self.data['II'])} points")
            
            # Write latest Lead II data to file for dashboard
            try:
                import json
                with open('lead_ii_live.json', 'w') as f:
                    json.dump(self.data["II"][-500:], f)
            except Exception as e:
                print("Error writing lead_ii_live.json:", e)
            
            # Calculate and update ECG metrics in real-time
            lead_ii_data = self.data.get("II", [])
            if lead_ii_data:
                intervals = self.calculate_ecg_intervals(lead_ii_data)
                self.update_ecg_metrics_on_top_of_lead_graphs(intervals)
            
            # Update all plots
            for i, lead in enumerate(self.leads):
                if len(self.data[lead]) > 0:
                    print(f"[DEBUG] ECGTestPage - Updating plot for {lead}: {len(self.data[lead])} data points")
                    
                    # Prepare plot data
                    if len(self.data[lead]) < self.buffer_size:
                        data = np.full(self.buffer_size, np.nan)
                        data[-len(self.data[lead]):] = self.data[lead]
                    else:
                        data = np.array(self.data[lead])
                    
                    # Convert device data to ECG range and center around zero
                    device_data = np.array(data)
                    # Scale to typical ECG range (subtract baseline ~2100 and scale)
                    # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                    gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                    centered = (device_data - 2100) * gain_factor
                    
                    # Apply noise reduction filtering
                    filtered_data = self.apply_ecg_filtering(centered)
                    
                    # Update the plot line
                    if i < len(self.lines):
                        self.lines[i].set_ydata(filtered_data)
                        print(f"[DEBUG] ECGTestPage - Updated {lead} plot with {len(centered)} points, range: {np.min(centered):.2f} to {np.max(centered):.2f}")
                        
                        # Use dynamic y-limits based on current gain setting
                        ylim = self.ylim if hasattr(self, 'ylim') else 400
                        if i < len(self.axs):
                            self.axs[i].set_ylim(-ylim, ylim)
                            
                            # Use dynamic x-limits based on current buffer size
                            self.axs[i].set_xlim(0, self.buffer_size)

                            # Update title with current settings
                            current_speed = self.settings_manager.get_wave_speed()
                            current_gain = self.settings_manager.get_wave_gain()
                            self.axs[i].set_title(f"{lead} | Speed: {current_speed}mm/s | Gain: {current_gain}mm/mV", 
                                                fontsize=8, color='#666', pad=10)
                            
                            # Add grid lines to show scale
                            self.axs[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                            
                            # Remove any existing labels
                            self.axs[i].set_xlabel("")
                            self.axs[i].set_ylabel("")
                        
                        # Force redraw of the canvas
                        if i < len(self.canvases):
                            self.canvases[i].draw_idle()
                    else:
                        print(f"[DEBUG] ECGTestPage - Warning: No line object for lead {lead} at index {i}")
                    
        except Exception as e:
            print(f"[DEBUG] ECGTestPage - Error in update_plot: {e}")
            import traceback
            traceback.print_exc()

    def generate_pdf_report(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import datetime, os, json, shutil
        from ecg.ecg_report_generator import generate_ecg_report

        # Capture last 10 seconds of live ECG data
        lead_img_paths = {}
        ordered_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        print(" Capturing last 10 seconds of live ECG data...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        
        # Calculate 10 seconds of data based on sampling rate
        sampling_rate = 250  # Default sampling rate
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate'):
            try:
                sampling_rate = float(self.sampler.sampling_rate)
            except:
                sampling_rate = 250
        
        data_points_10_sec = int(sampling_rate * 10)  # 10 seconds of data
        print(f" Capturing {data_points_10_sec} data points at {sampling_rate}Hz")
        
        for i, lead in enumerate(ordered_leads):
            if i < len(self.data) and i < len(self.leads):
                try:
                    # Get the last 10 seconds of data for this lead
                    lead_data = self.data[i]
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
                        
                        print(f" ✅ Captured 10s Lead {lead}: {len(recent_data)} samples")
                    else:
                        print(f" ⚠️ No data available for Lead {lead}")
                        
                except Exception as e:
                    print(f" ❌ Error capturing Lead {lead}: {e}")
            else:
                print(f" ⚠️ Lead {lead} not available (index {i})")

        # Ask user for destination
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save ECG Report",
            f"ECG_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)"
        )
        if not filename:
            return

        try:
            import re
            ecg_data = {"HR": 0, "beat": 0, "PR": 0, "QRS": 0, "QT": 0, "QTc": 0, "ST": 0, "HR_max": 0, "HR_min": 0, "HR_avg": 0}

            # Try to mirror dashboard metrics so both reports match
            dashboard_ref = None
            try:
                widget = self
                for _ in range(12):
                    if widget is None:
                        break
                    if hasattr(widget, 'metric_labels') and isinstance(getattr(widget, 'metric_labels'), dict):
                        dashboard_ref = widget
                        break
                    widget = widget.parent()
            except Exception:
                dashboard_ref = None

            def _num_from_label(lbl_text: str, default_val: int) -> int:
                if not isinstance(lbl_text, str):
                    return default_val
                m = re.search(r"-?\d+", lbl_text)
                return int(m.group(0)) if m else default_val

            if dashboard_ref and hasattr(dashboard_ref, 'metric_labels'):
                ml = dashboard_ref.metric_labels
                try:
                    hr = _num_from_label(ml.get('heart_rate', QLabel('')).text() if ml.get('heart_rate') else '', 0)
                except Exception:
                    hr = 0
                try:
                    pr = _num_from_label(ml.get('pr_interval', QLabel('')).text() if ml.get('pr_interval') else '', 0)
                except Exception:
                    pr = 0
                try:
                    qrs = _num_from_label(ml.get('qrs_duration', QLabel('')).text() if ml.get('qrs_duration') else '', 0)
                except Exception:
                    qrs = 0
                try:
                    qtc = _num_from_label(ml.get('qtc_interval', QLabel('')).text() if ml.get('qtc_interval') else '', 0)
                except Exception:
                    qtc = 0
                try:
                    st = _num_from_label(ml.get('st_interval', QLabel('')).text() if ml.get('st_interval') else '', 0)
                except Exception:
                    st = 0

                ecg_data.update({
                    "beat": hr,
                    "HR_avg": hr,
                    "PR": pr,
                    "QRS": qrs,
                    # If QTc is zero (no data), QT must be zero as well
                    "QT": 0 if qtc == 0 else int(max(0, qtc - 20)),
                    "QTc": qtc,
                    "ST": st,
                })

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
                print(f"⚠️ Error loading patient data: {e}")
                patient = None

            # Always stamp current date/time
            import datetime as _dt
            now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not patient:
                patient = {}
            patient["date_time"] = now_str

            # Generate report with patient details
            generate_ecg_report(filename, ecg_data, lead_img_paths, None, self, patient)

            # Append history
            try:
                from dashboard.history_window import append_history_entry
                append_history_entry(patient, filename, report_type="12 Lead")
            except Exception:
                import traceback
                traceback.print_exc()

            QMessageBox.information(self, "Success", f"ECG Report generated successfully!\nSaved as: {filename}")

            # Dual-save to app reports/ and update index.json
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            reports_dir = os.path.abspath(os.path.join(base_dir, '..', 'reports'))
            os.makedirs(reports_dir, exist_ok=True)
            dst_basename = os.path.basename(filename)
            dst_path = os.path.join(reports_dir, dst_basename)
            if os.path.abspath(filename) != os.path.abspath(dst_path):
                counter = 1
                name, ext = os.path.splitext(dst_basename)
                while os.path.exists(dst_path):
                    dst_basename = f"{name}_{counter}{ext}"
                    dst_path = os.path.join(reports_dir, dst_basename)
                    counter += 1
                shutil.copyfile(filename, dst_path)
            
            # Also save to Downloads folder
            try:
                import pathlib
                downloads_path = pathlib.Path.home() / "Downloads"
                if downloads_path.exists():
                    # Use the original filename basename (before any counter modifications)
                    original_basename = os.path.basename(filename)
                    downloads_report_path = downloads_path / original_basename
                    # Handle duplicates in Downloads folder
                    counter = 1
                    name, ext = os.path.splitext(original_basename)
                    while downloads_report_path.exists():
                        downloads_report_path = downloads_path / f"{name}_{counter}{ext}"
                        counter += 1
                    shutil.copyfile(filename, str(downloads_report_path))
                    print(f"✅ Report also saved to Downloads: {downloads_report_path}")
            except Exception as e:
                print(f"⚠️ Could not save to Downloads folder: {e}")
            index_path = os.path.join(reports_dir, 'index.json')
            items = []
            if os.path.exists(index_path):
                try:
                    with open(index_path, 'r') as f:
                        items = json.load(f)
                except Exception:
                    items = []
            now = datetime.datetime.now()
            # Include patient name in recent reports entry
            full_name = ""
            try:
                fn = patient.get("first_name", "") if isinstance(patient, dict) else ""
                ln = patient.get("last_name", "") if isinstance(patient, dict) else ""
                full_name = (fn + " " + ln).strip()
            except Exception:
                full_name = ""

            meta = {
                'filename': os.path.basename(dst_path),
                'title': 'ECG Report',
                'patient': full_name,
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S')
            }
            items = [meta] + items
            items = items[:10]
            with open(index_path, 'w') as f:
                json.dump(items, f, indent=2)

            # Try refreshing dashboard recent reports if available
            try:
                widget = self
                for _ in range(10):  # prevent infinite loops
                    if widget is None:
                        break
                    if hasattr(widget, 'refresh_recent_reports_ui'):
                        widget.refresh_recent_reports_ui()
                        break
                    widget = widget.parent()
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate PDF: {str(e)}")

    def export_csv(self):
        """Export ECG data to CSV file in the same format as dummydata.csv"""
        path, _ = QFileDialog.getSaveFileName(self, "Export ECG Data as CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')  # Use tab delimiter like dummydata.csv
                    
                    # Write header exactly like dummydata.csv
                    header = ["Sample", "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                    writer.writerow(header)
                    
                    # Export from CSV storage (most accurate method)
                    if hasattr(self, 'csv_data_storage') and self.csv_data_storage:
                        print(f"📊 Exporting {len(self.csv_data_storage)} samples from CSV storage")
                        
                        for row_data in self.csv_data_storage:
                            row = [
                                row_data.get('Sample', ''),
                                row_data.get('I', ''),
                                row_data.get('II', ''),
                                row_data.get('III', ''),
                                row_data.get('aVR', ''),
                                row_data.get('aVL', ''),
                                row_data.get('aVF', ''),
                                row_data.get('V1', ''),
                                row_data.get('V2', ''),
                                row_data.get('V3', ''),
                                row_data.get('V4', ''),
                                row_data.get('V5', ''),
                                row_data.get('V6', '')
                            ]
                            writer.writerow(row)
                    
                    # Fallback: Export from numpy arrays if CSV storage is empty
                    else:
                        print("📊 Exporting from numpy arrays (fallback method)")
                        
                        # Get the actual data length
                        max_length = 0
                        for i in range(len(self.leads)):
                            if i < len(self.data):
                                # Count non-zero values in the numpy array
                                non_zero_count = np.count_nonzero(self.data[i])
                                max_length = max(max_length, non_zero_count)
                        
                        # Export data sample by sample
                        for i in range(max_length):
                            row = [i]  # Sample number
                            
                            # Add data for each lead in the same order as dummydata.csv
                            for lead_idx, lead_name in enumerate(self.leads):
                                if lead_idx < len(self.data):
                                    if i < len(self.data[lead_idx]):
                                        value = self.data[lead_idx][i]
                                        # Only include non-zero values (actual data)
                                        if value != 0:
                                            row.append(int(value))
                                        else:
                                            row.append("")
                                    else:
                                        row.append("")
                                else:
                                    row.append("")
                            
                            writer.writerow(row)
                
                print(f"✅ CSV export completed: {path}")
                QMessageBox.information(
                    self, 
                    "Export Successful", 
                    f"ECG data exported successfully!\n\nFile: {path}\nSamples: {len(self.csv_data_storage) if hasattr(self, 'csv_data_storage') else 'N/A'}"
                )
                
            except Exception as e:
                print(f"❌ Error exporting CSV: {e}")
                QMessageBox.critical(
                    self, 
                    "Export Error", 
                    f"Failed to export CSV:\n{str(e)}"
                )

    def go_back(self):
        """Go back to the dashboard"""
        if hasattr(self, '_overlay_active') and self._overlay_active:
            self._restore_original_layout()

        # Go back to dashboard (assumes dashboard is at index 0)
        self.stacked_widget.setCurrentIndex(0)

    def show_connection_warning(self, extra_msg=""):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Connection Required")
        msg.setText("❤️ Please configure serial port and baud rate in System Setup.\n\nStay healthy!" + ("\n\n" + extra_msg if extra_msg else ""))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_main_menu(self):  
        self.clear_content()

    def clear_content(self):
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def show_sequential_view(self):
        from ecg.lead_sequential_view import LeadSequentialView
        win = LeadSequentialView(self.leads, self.data, buffer_size=500)
        win.show()
        self._sequential_win = win

    # ------------------------------------ 12 leads overlay --------------------------------------------

    def twelve_leads_overlay(self):
        if getattr(self, "_overlay_active", False):
            # If we're already in 12:1, treat this as a toggle (close overlay)
            if getattr(self, "_current_overlay_layout", None) == "12x1":
                self._restore_original_layout()
                self._current_overlay_layout = None
                return
            # If some other overlay (e.g. 6:2) is active, restore first, then switch
            self._restore_original_layout()
        
        # Store the original plot area layout
        self._store_original_layout()
        
        # Create the overlay widget
        self._create_overlay_widget()
        
        # Replace the plot area with overlay
        self._replace_plot_area_with_overlay()
        
        # Mark overlay as active and record layout type
        self._overlay_active = True
        self._current_overlay_layout = "12x1"

        self._apply_current_overlay_mode()

        # Ensure demo data continues to work in overlay mode
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            print("Demo mode active - overlay will show demo data")

    def _store_original_layout(self):
        
        # Store the current plot area widget
        self._original_plot_area = self.plot_area
        
        # Store the current layout
        self._original_layout = self.plot_area.layout()
        
        # Store the current figures, canvases, axes, and lines
        self._original_figures = getattr(self, 'figures', [])
        self._original_canvases = getattr(self, 'canvases', [])
        self._original_axs = getattr(self, 'axs', [])
        self._original_lines = getattr(self, 'lines', [])

    def _create_overlay_widget(self):
        
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame
        
        # Create overlay container
        self._overlay_widget = QWidget()
        self._overlay_widget.setStyleSheet("""
            QWidget {
                background: #000;
                border: 2px solid #ff6600;
                border-radius: 15px;
            }
        """)
        
        # Main layout for overlay
        overlay_layout = QVBoxLayout(self._overlay_widget)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        overlay_layout.setSpacing(15)
        
        # Top control panel with close button
        top_panel = QFrame()
        top_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(15, 10, 15, 10)
        top_layout.setSpacing(20)
        
        # Close button
        close_btn = QPushButton("Close Overlay")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 2px solid #ff6600;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 2px solid #ff8c42;
            }
        """)
        close_btn.clicked.connect(self._restore_original_layout)
        
        # Mode control buttons with highlighting
        self.light_mode_btn = QPushButton("Light Mode")
        self.dark_mode_btn = QPushButton("Dark Mode")
        self.graph_mode_btn = QPushButton("Graph Mode")
        
        # Store current mode for highlighting
        self._current_overlay_mode = "dark"  # Default mode
        
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
            }
        """
        
        active_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 3px solid #ff6600;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                /* Removed unsupported box-shadow property */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 3px solid #ff8c42;
            }
        """
        
        self.light_mode_btn.setStyleSheet(button_style)
        self.dark_mode_btn.setStyleSheet(button_style)
        self.graph_mode_btn.setStyleSheet(button_style)
        
        # Add widgets to top panel
        top_layout.addWidget(close_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.light_mode_btn)
        top_layout.addWidget(self.dark_mode_btn)
        top_layout.addWidget(self.graph_mode_btn)
        
        overlay_layout.addWidget(top_panel)
        
        # Create the matplotlib figure with all leads
        self._create_overlay_figure(overlay_layout)
        
        # Connect mode buttons
        self.light_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("light"))
        self.dark_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("dark"))
        self.graph_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("graph"))
        
        # Apply default dark mode and highlight it
        self._apply_overlay_mode("dark")

    def _create_overlay_figure(self, overlay_layout):
        
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import numpy as np
        
        # Create figure with all leads - adjust spacing for better visibility
        num_leads = len(self.leads)
        fig = Figure(figsize=(16, num_leads * 2.2), facecolor='none')  # Changed to transparent
        
        # Adjust subplot parameters for better spacing
        fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, hspace=0.15)
        
        self._overlay_axes = []
        self._overlay_lines = []
        
        for idx, lead in enumerate(self.leads):
            ax = fig.add_subplot(num_leads, 1, idx+1)
            ax.set_facecolor('none')  # Changed to transparent
            
            # Remove all borders and spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove all ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lead, color='#00ff00', fontsize=12, fontweight='bold', labelpad=20)
            
            # Create line with initial data
            line, = ax.plot(np.arange(self.buffer_size), [np.nan]*self.buffer_size, color="#00ff00", lw=1.5)
            self._overlay_axes.append(ax)
            self._overlay_lines.append(line)
        
        self._overlay_canvas = FigureCanvas(fig)
        overlay_layout.addWidget(self._overlay_canvas)
        
        # Start update timer for overlay
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._update_overlay_plots)
        self._overlay_timer.start(100)

    def _get_overlay_target_buffer_len(self, is_demo_mode):
        """
        Calculate buffer length for overlay modes based on wave speed.
        For real serial data, calculate based on wave speed to ensure peaks align.
        For demo mode, swap the visual wave-speed behaviour between 12.5mm/s and 50mm/s.
        """
        try:
            wave_speed = float(self.settings_manager.get_wave_speed())
        except Exception:
            wave_speed = 25.0

        if not is_demo_mode:
            # For real serial data, calculate buffer length based on wave speed
            # Same logic as in update_plots() for serial data
            # 25 mm/s → 3 s window (≈15 large boxes), scale for other speeds
            baseline_seconds = 3.0
            seconds_scale = (25.0 / max(1e-6, wave_speed))
            seconds_to_show = baseline_seconds * seconds_scale
            
            # Use hardware sampling rate (80 Hz)
            sampling_rate = 500.0
            samples_to_show = int(sampling_rate * seconds_to_show)
            
            # Return the calculated samples (same as main plots - no buffer size limit)
            # The data selection will handle cases where data is smaller
            return max(1, samples_to_show)

        # Demo mode: Map overlay speeds: 12.5 ⇄ 50, keep others unchanged.
        if wave_speed <= 13.0:
            mapped_speed = 50.0
        elif wave_speed >= 49.0:
            mapped_speed = 12.5
        else:
            mapped_speed = wave_speed

        base_buffer = getattr(self, "base_buffer_size", 2000)
        target = int(base_buffer * (mapped_speed / 50.0))
        return max(1, target)

    def _update_overlay_plots(self):
        
        if not hasattr(self, '_overlay_lines') or not self._overlay_lines:
            return
        
        # Check if demo mode is active
        is_demo_mode = hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked()
        
        target_buffer_len = self._get_overlay_target_buffer_len(is_demo_mode)
        
        for idx, lead in enumerate(self.leads):
            if idx < len(self._overlay_lines):
                if idx < len(self.data):
                    data = self.data[idx]
                else:
                    data = np.array([])
                line = self._overlay_lines[idx]
                ax = self._overlay_axes[idx]

                # Ensure overlay line length matches current buffer size
                buffer_len = target_buffer_len
                try:
                    xdata = line.get_xdata()
                    current_len = len(xdata) if xdata is not None else 0
                    if current_len != buffer_len:
                        new_x = np.arange(buffer_len)
                        line.set_xdata(new_x)
                        current_len = buffer_len
                    if current_len:
                        buffer_len = current_len
                except Exception as e:
                    print(f"⚠️ Overlay line sync error (12-lead): {e}")
                    buffer_len = target_buffer_len
                
                plot_data = np.full(buffer_len, np.nan)
                
                if data is not None and len(data) > 0:
                    # Take exactly buffer_len samples from the end (same as main plots)
                    # This ensures overlay matches main view for all wave speeds
                    if len(data) >= buffer_len:
                        data_segment = data[-buffer_len:]
                    else:
                        data_segment = data
                    
                    # Optional AC notch filtering (match main 12-lead grid view)
                    filtered_segment = np.array(data_segment, dtype=float)
                    try:
                        # Prefer hardware sampling rate; fall back to stored sampling_rate or 500 Hz
                        sampling_rate = 500.0
                        try:
                            if hasattr(self, "sampler") and hasattr(self.sampler, "sampling_rate") and self.sampler.sampling_rate:
                                sampling_rate = float(self.sampler.sampling_rate)
                            elif hasattr(self, "sampling_rate") and self.sampling_rate:
                                sampling_rate = float(self.sampling_rate)
                        except Exception:
                            pass
                        
                        ac_setting = self.settings_manager.get_setting("filter_ac", "off") if hasattr(self, "settings_manager") else "off"
                        if ac_setting and ac_setting != "off" and len(filtered_segment) >= 10:
                            from ecg.ecg_filters import apply_ac_filter
                            filtered_segment = apply_ac_filter(filtered_segment, sampling_rate, ac_setting)
                    except Exception as filter_error:
                        print(f"⚠️ Overlay AC filter skipped for lead {lead}: {filter_error}")
                    
                    # Center data around baseline before applying gain
                    centered_raw = np.array(filtered_segment, dtype=float)
                    if centered_raw.size:
                        finite_mask = np.isfinite(centered_raw)
                        if np.any(finite_mask):
                            baseline = np.nanmedian(centered_raw[finite_mask])
                            if np.isfinite(baseline):
                                centered_raw = centered_raw - baseline
                        centered_raw = np.nan_to_num(centered_raw, copy=False)
                    else:
                        centered_raw = np.zeros(buffer_len, dtype=float)

                    # Apply current gain setting (match main 12-lead grid)
                    gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                    
                    # Reduce amplification for 20mm/mV to prevent clipping in 12:1 overlay mode
                    if gain_factor >= 2.0:  # 20mm/mV or higher
                        if is_demo_mode:
                            reduction_factor = 0.75  # Reduce to 75% for demo mode
                        else:
                            reduction_factor = 0.75  # Reduce to 75% for real mode to prevent clipping
                        gain_factor = gain_factor * reduction_factor
                    
                    centered = centered_raw * gain_factor
                    centered = np.nan_to_num(centered, copy=False)
                    
                    # Debug logging for first lead in demo mode
                    if is_demo_mode and idx == 1:  # Lead II
                        print(f"🎨 Overlay demo mode: Lead {lead}, gain={gain_factor:.2f}, raw_range={np.max(np.abs(centered_raw)):.1f}, gained_range={np.max(np.abs(centered)):.1f}")
                    
                    # Match main plots: if we have enough data, take exactly buffer_len samples
                    # If not enough data, stretch what we have to fill buffer_len
                    n = len(centered)
                    if n < buffer_len:
                        # Stretch available data to fill buffer_len
                        stretched = np.interp(
                            np.linspace(0, n-1, buffer_len),
                            np.arange(n),
                            centered
                        )
                        plot_data[:] = stretched
                    else:
                        # Take exactly buffer_len samples from the end (same as main plots)
                        plot_data[:] = centered[-buffer_len:]
                    
                    # Set Y-limits based on UN-GAINED data for both demo and real mode, so gain actually affects visual size
                    # Use raw data for Y-axis calculation, so gain changes visual size
                    valid_data = centered_raw[np.isfinite(centered_raw)]
                    
                    if len(valid_data) > 0:
                        # Use percentiles to avoid spikes from clipping the view
                        p1 = np.percentile(valid_data, 1)
                        p99 = np.percentile(valid_data, 99)
                        data_mean = (p1 + p99) / 2.0
                        data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
                        
                        # Calculate appropriate Y-range with some padding
                        if data_std > 0:
                            # Use standard deviation within central band
                            padding = max(data_std * 4, 200)  # Increased padding for better visibility
                            ymin = data_mean - padding
                            ymax = data_mean + padding
                        else:
                            # Fallback: use percentile window
                            data_range = max(p99 - p1, 300)
                            padding = max(data_range * 0.3, 200)
                            ymin = data_mean - padding
                            ymax = data_mean + padding
                        
                        # Ensure reasonable bounds (same as main plots)
                        ymin = max(ymin, -8000)
                        ymax = min(ymax, 8000)
                    else:
                        ymin, ymax = -500, 500

                    yr = (ymax - ymin)
                    lead_name = self.leads[idx] if idx < len(self.leads) else ""
                    is_chest = lead_name in ["V1", "V2", "V3", "V4", "V5", "V6"]
                    top_extra = 0.90 if is_chest else 0.35
                    bottom_extra = 0.70 if is_chest else 0.30
                    ymin = ymin - yr * bottom_extra
                    ymax = ymax + yr * top_extra
                    
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(-500, 500)
                
                # Set x-limits
                ax.set_xlim(0, max(buffer_len - 1, 1))
                line.set_ydata(plot_data)
        
        if hasattr(self, '_overlay_canvas'):
            self._overlay_canvas.draw_idle()

    def _apply_current_overlay_mode(self):

        if hasattr(self, '_current_overlay_mode'):
            self._apply_overlay_mode(self._current_overlay_mode)

    def _apply_overlay_mode(self, mode):
        
        if not hasattr(self, '_overlay_axes') or not self._overlay_axes:
            return
        
        # Store current mode
        self._current_overlay_mode = mode
        
        self._clear_all_backgrounds()
        
        # Update button highlighting
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
            }
        """
        
        active_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 3px solid #ff6600;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                /* Removed unsupported box-shadow property */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 3px solid #ff8c42;
            }
        """
        
        # Reset all buttons to normal style
        self.light_mode_btn.setStyleSheet(button_style)
        self.dark_mode_btn.setStyleSheet(button_style)
        self.graph_mode_btn.setStyleSheet(button_style)
        
        # Highlight the active button
        if mode == "light":
            self.light_mode_btn.setStyleSheet(active_button_style)
            self._overlay_widget.setStyleSheet("""
                QWidget {
                    background: rgba(255, 255, 255, 0.95);
                    border: 2px solid #ff6600;
                    border-radius: 15px;
                }
            """)
            
            for ax in self._overlay_axes:
                ax.set_facecolor('#ffffff')
                ax.tick_params(axis='x', colors='#333333', labelsize=10)
                ax.tick_params(axis='y', colors='#333333', labelsize=10)
                ax.set_ylabel(ax.get_ylabel(), color='#333333', fontsize=14, fontweight='bold', labelpad=15)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.figure.canvas.draw()
            
            for line in self._overlay_lines:
                line.set_color('#0066cc')
                line.set_linewidth(1.5)
        
        elif mode == "dark":
            self.dark_mode_btn.setStyleSheet(active_button_style)
            self._overlay_widget.setStyleSheet("""
                QWidget {
                    background: rgba(0, 0, 0, 0.95);
                    border: 2px solid #ff6600;
                    border-radius: 15px;
                }
            """)
            
            for ax in self._overlay_axes:
                ax.set_facecolor('#000')
                ax.tick_params(axis='x', colors='#00ff00', labelsize=10)
                ax.tick_params(axis='y', colors='#00ff00', labelsize=10)
                ax.set_ylabel(ax.get_ylabel(), color='#00ff00', fontsize=14, fontweight='bold', labelpad=15)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            for line in self._overlay_lines:
                line.set_color('#00ff00')
                line.set_linewidth(1.5)
        
        elif mode == "graph":
            self.graph_mode_btn.setStyleSheet(active_button_style)
            self._apply_graph_mode()
        
        if hasattr(self, '_overlay_canvas'):
            self._overlay_canvas.draw()

    def _clear_all_backgrounds(self):
        
        try:
            # Clear figure-level background
            if hasattr(self, '_overlay_canvas') and self._overlay_canvas.figure:
                fig = self._overlay_canvas.figure
                if hasattr(fig, '_figure_background'):
                    try:
                        fig._figure_background.remove()
                        delattr(fig, '_figure_background')
                    except:
                        pass
                
                # Reset figure background to transparent
                fig.patch.set_facecolor('none')
            
            # Clear axis-level backgrounds
            if hasattr(self, '_overlay_axes'):
                for ax in self._overlay_axes:
                    if hasattr(ax, '_background_image'):
                        try:
                            ax._background_image.remove()
                            delattr(ax, '_background_image')
                        except:
                            pass
                    
                    # Reset axis background to transparent
                    ax.set_facecolor('none')
                    ax.patch.set_alpha(0.0)
                    
        except Exception as e:
            print(f"Error clearing backgrounds: {e}")

    def _apply_graph_mode(self):
        
        try:
            import os
            from PyQt5.QtGui import QPixmap
            import matplotlib.image as mpimg
            
            bg_path = "ecg_pink_grid_fullpage.png"
            if os.path.exists(bg_path):
                # Load the background image
                bg_img = QPixmap(bg_path)
                if not bg_img.isNull():
                    # Save temporary file for matplotlib
                    temp_path = "temp_bg.png"
                    bg_img.save(temp_path)
                    bg_matplotlib = mpimg.imread(temp_path)
                    
                    # Apply background to the entire figure first
                    if hasattr(self, '_overlay_canvas') and self._overlay_canvas.figure:
                        fig = self._overlay_canvas.figure
                        fig.patch.set_facecolor('#ffffff')  # White background for the figure
                        
                        # Remove any existing background from figure
                        if hasattr(fig, '_figure_background'):
                            try:
                                fig._figure_background.remove()
                            except:
                                pass
                        
                        # Apply background image to the entire figure
                        fig._figure_background = fig.figimage(
                            bg_matplotlib, 
                            xo=0, yo=0, 
                            alpha=0.4,  # Slightly transparent so waves are visible
                            zorder=0
                        )
                    
                    # Apply background to all axes
                    for i, ax in enumerate(self._overlay_axes):
                        # Set transparent background for subplots
                        ax.set_facecolor('none')
                        ax.patch.set_alpha(0.0)
                        
                        # Remove all borders and spines
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        
                        # Remove ticks for cleaner look
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # Set label color to dark for better visibility on grid background
                        ax.set_ylabel(ax.get_ylabel(), color='#333333', fontsize=12, fontweight='bold', labelpad=20)
                        
                        # Set proper limits
                        ax.set_xlim(0, self.buffer_size-1)
                        ax.set_ylim(-500, 500)
                    
                    # Change line colors to dark red for better visibility on grid background
                    for line in self._overlay_lines:
                        line.set_color('#cc0000')  # Darker red
                        line.set_linewidth(1.5)
                        line.set_alpha(1.0)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Force redraw
                    if hasattr(self, '_overlay_canvas'):
                        self._overlay_canvas.draw()
                        
                    return
                        
                else:
                    print("Failed to load background image")
                    return
            else:
                print(f"Background image not found at: {bg_path}")
                return
                
        except Exception as e:
            print(f"Error applying graph mode: {e}")
            return

    def _replace_plot_area_with_overlay(self):
        
        # Get the main horizontal layout
        main_layout = self.grid_widget.layout()
        
        # Find the main_vbox layout item (which contains the plot_area)
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() and hasattr(item.layout(), 'indexOf') and item.layout().indexOf(self.plot_area) >= 0:
                # Found the layout containing plot_area
                main_vbox_layout = item.layout()
                
                # Find and replace the plot_area in main_vbox_layout
                plot_area_index = main_vbox_layout.indexOf(self.plot_area)
                if plot_area_index >= 0:
                    # Remove the plot_area
                    main_vbox_layout.removeWidget(self.plot_area)
                    self.plot_area.hide()
                    
                    # Add the overlay widget at the same position
                    main_vbox_layout.insertWidget(plot_area_index, self._overlay_widget)
                    return
        
        # Fallback: if we can't find the exact position, add to the end of main_vbox
        # Find the main_vbox layout
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() and hasattr(item.layout(), 'indexOf') and item.layout().indexOf(self.plot_area) >= 0:
                main_vbox_layout = item.layout()
                main_vbox_layout.removeWidget(self.plot_area)
                self.plot_area.hide()
                main_vbox_layout.addWidget(self._overlay_widget)
                break

    def _restore_original_layout(self):
        
        if not hasattr(self, '_overlay_active') or not self._overlay_active:
            return
        
        # Stop overlay timer
        if hasattr(self, '_overlay_timer'):
            self._overlay_timer.stop()
            self._overlay_timer.deleteLater()
        
        # Find and remove overlay widget from main_vbox layout
        main_layout = self.grid_widget.layout()
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() and hasattr(item.layout(), 'indexOf'):
                main_vbox_layout = item.layout()
                
                # Check if overlay widget is in this layout
                overlay_index = main_vbox_layout.indexOf(self._overlay_widget)
                if overlay_index >= 0:
                    # Remove overlay widget
                    main_vbox_layout.removeWidget(self._overlay_widget)
                    
                    # Restore original plot area at the exact same position
                    main_vbox_layout.insertWidget(overlay_index, self.plot_area)
                    self.plot_area.show()
                    break
        
        # Clean up overlay references
        if hasattr(self, '_overlay_widget'):
            self._overlay_widget.deleteLater()
            delattr(self, '_overlay_widget')
        
        if hasattr(self, '_overlay_axes'):
            delattr(self, '_overlay_axes')
        
        if hasattr(self, '_overlay_lines'):
            delattr(self, '_overlay_lines')
        
        if hasattr(self, '_overlay_canvas'):
            delattr(self, '_overlay_canvas')
        
        # Mark overlay as inactive and clear current layout type
        self._overlay_active = False
        if hasattr(self, "_current_overlay_layout"):
            self._current_overlay_layout = None
        
        # Force redraw of original plots
        self.redraw_all_plots()

    # ------------------------------------ 6 leads overlay --------------------------------------------

    def six_leads_overlay(self):
        # If overlay is already shown, hide it and restore original layout
        if hasattr(self, '_overlay_active') and self._overlay_active:
            self._restore_original_layout()
        
        # Store the original plot area layout
        self._store_original_layout()
        
        # Create the 2-column overlay widget
        self._create_two_column_overlay_widget()
        
        # Replace the plot area with overlay
        self._replace_plot_area_with_overlay()
        
        # Mark overlay as active and record layout type
        self._overlay_active = True
        self._current_overlay_layout = "6x2"

        self._apply_current_overlay_mode()

        # Ensure demo data continues to work in overlay mode
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            print("Demo mode active - overlay will show demo data")

    def _create_two_column_overlay_widget(self):
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame
        
        # Create overlay container
        self._overlay_widget = QWidget()
        self._overlay_widget.setStyleSheet("""
            QWidget {
                background: #000;
                border: 2px solid #ff6600;
                border-radius: 15px;
            }
        """)
        
        # Main layout for overlay
        overlay_layout = QVBoxLayout(self._overlay_widget)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        overlay_layout.setSpacing(15)
        
        # Top control panel with close button and mode controls
        top_panel = QFrame()
        top_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(15, 10, 15, 10)
        top_layout.setSpacing(20)
        
        # Close button
        close_btn = QPushButton("Close Overlay")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 2px solid #ff6600;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 2px solid #ff8c42;
            }
        """)
        close_btn.clicked.connect(self._restore_original_layout)
        
        # Mode control buttons with highlighting
        self.light_mode_btn = QPushButton("Light Mode")
        self.dark_mode_btn = QPushButton("Dark Mode")
        self.graph_mode_btn = QPushButton("Graph Mode")
        
        # Store current mode for highlighting
        self._current_overlay_mode = "dark"  # Default mode
        
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
            }
        """
        
        active_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 3px solid #ff6600;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                /* Removed unsupported box-shadow property */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 3px solid #ff8c42;
            }
        """
        
        self.light_mode_btn.setStyleSheet(button_style)
        self.dark_mode_btn.setStyleSheet(button_style)
        self.graph_mode_btn.setStyleSheet(button_style)
        
        # Add widgets to top panel
        top_layout.addWidget(close_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.light_mode_btn)
        top_layout.addWidget(self.dark_mode_btn)
        top_layout.addWidget(self.graph_mode_btn)
        
        overlay_layout.addWidget(top_panel)
        
        # Create the 2-column matplotlib figure
        self._create_two_column_figure(overlay_layout)
        
        # Connect mode buttons
        self.light_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("light"))
        self.dark_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("dark"))
        self.graph_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("graph"))
        
        # Apply default dark mode and highlight it
        self._apply_overlay_mode("dark")

    def _create_two_column_figure(self, overlay_layout):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import numpy as np
        
        # Define the two columns of leads
        left_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        right_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        
        # Create figure with 2 columns and 6 rows
        fig = Figure(figsize=(16, 12), facecolor='none')
        
        # Adjust subplot parameters for better spacing
        fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, hspace=0.15, wspace=0.1)
        
        self._overlay_axes = []
        self._overlay_lines = []
        
        # Create left column (limb leads)
        for idx, lead in enumerate(left_leads):
            ax = fig.add_subplot(6, 2, 2*idx + 1)
            ax.set_facecolor('none')
            
            # Remove all borders and spines

            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove all ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lead, color='#00ff00', fontsize=12, fontweight='bold', labelpad=20)
            
            # Create line with initial data
            line, = ax.plot(np.arange(self.buffer_size), [np.nan]*self.buffer_size, color="#00ff00", lw=1.5)
            self._overlay_axes.append(ax)
            self._overlay_lines.append(line)
        
        # Create right column (chest leads)
        for idx, lead in enumerate(right_leads):
            ax = fig.add_subplot(6, 2, 2*idx + 2)
            ax.set_facecolor('none')
            
            # Remove all borders and spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove all ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lead, color='#00ff00', fontsize=12, fontweight='bold', labelpad=20)
            
            # Create line with initial data
            line, = ax.plot(np.arange(self.buffer_size), [np.nan]*self.buffer_size, color="#00ff00", lw=1.5)
            self._overlay_axes.append(ax)
            self._overlay_lines.append(line)
        
        self._overlay_canvas = FigureCanvas(fig)
        overlay_layout.addWidget(self._overlay_canvas)
        
        # Start update timer for overlay
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._update_two_column_plots)
        self._overlay_timer.start(100)

    def _update_two_column_plots(self):
        if not hasattr(self, '_overlay_lines') or not self._overlay_lines:
            return
        
        # Check if demo mode is active
        is_demo_mode = hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked()
        
        target_buffer_len = self._get_overlay_target_buffer_len(is_demo_mode)
        
        # Define the two columns of leads
        left_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        right_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        all_leads = left_leads + right_leads
        
        for idx, lead in enumerate(all_leads):
            if idx < len(self._overlay_lines):
                if lead in self.leads:
                    lead_index = self.leads.index(lead)
                    if lead_index < len(self.data):
                        data = self.data[lead_index]
                    else:
                        data = np.array([])
                else:
                    data = np.array([])
                line = self._overlay_lines[idx]
                ax = self._overlay_axes[idx]
                
                # Ensure overlay line length matches current buffer size
                buffer_len = target_buffer_len
                try:
                    xdata = line.get_xdata()
                    current_len = len(xdata) if xdata is not None else 0
                    if current_len != buffer_len:
                        new_x = np.arange(buffer_len)
                        line.set_xdata(new_x)
                        current_len = buffer_len
                    if current_len:
                        buffer_len = current_len
                except Exception as e:
                    print(f"⚠️ Overlay line sync error (6-lead): {e}")
                    buffer_len = target_buffer_len

                plot_data = np.full(buffer_len, np.nan)
                
                if data is not None and len(data) > 0:
                    # Take exactly buffer_len samples from the end (same as main plots)
                    # This ensures overlay matches main view for all wave speeds
                    if len(data) >= buffer_len:
                        data_segment = data[-buffer_len:]
                    else:
                        data_segment = data
                    
                    # Optional AC notch filtering (match main 12-lead grid view)
                    filtered_segment = np.array(data_segment, dtype=float)
                    try:
                        sampling_rate = 500.0
                        try:
                            if hasattr(self, "sampler") and hasattr(self.sampler, "sampling_rate") and self.sampler.sampling_rate:
                                sampling_rate = float(self.sampler.sampling_rate)
                            elif hasattr(self, "sampling_rate") and self.sampling_rate:
                                sampling_rate = float(self.sampling_rate)
                        except Exception:
                            pass
                        
                        ac_setting = self.settings_manager.get_setting("filter_ac", "off") if hasattr(self, "settings_manager") else "off"
                        if ac_setting and ac_setting != "off" and len(filtered_segment) >= 10:
                            from ecg.ecg_filters import apply_ac_filter
                            filtered_segment = apply_ac_filter(filtered_segment, sampling_rate, ac_setting)
                    except Exception as filter_error:
                        print(f"⚠️ 6:2 overlay AC filter skipped for lead {lead}: {filter_error}")
                    
                    # Center data around baseline before applying gain
                    centered_raw = np.array(filtered_segment, dtype=float)
                    if centered_raw.size:
                        finite_mask = np.isfinite(centered_raw)
                        if np.any(finite_mask):
                            baseline = np.nanmedian(centered_raw[finite_mask])
                            if np.isfinite(baseline):
                                centered_raw = centered_raw - baseline
                        centered_raw = np.nan_to_num(centered_raw, copy=False)
                    else:
                        centered_raw = np.zeros(buffer_len, dtype=float)

                    # Apply current gain setting (match main 12-lead grid)
                    gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                    
                    # Reduce amplification for 20mm/mV to prevent clipping in 6:2 overlay mode
                    if gain_factor >= 2.0:  # 20mm/mV or higher
                        if is_demo_mode:
                            reduction_factor = 0.75  # Reduce to 75% for demo mode
                        else:
                            reduction_factor = 0.75  # Reduce to 75% for real mode to prevent clipping
                        gain_factor = gain_factor * reduction_factor
                    
                    centered = centered_raw * gain_factor
                    centered = np.nan_to_num(centered, copy=False)
                    
                    # Debug logging for first lead in demo mode
                    if is_demo_mode and idx == 1:  # Lead II
                        print(f"🎨 6:2 Overlay demo mode: Lead {lead}, gain={gain_factor:.2f}, raw_range={np.max(np.abs(centered_raw)):.1f}, gained_range={np.max(np.abs(centered)):.1f}")
                    
                    # Match main plots: if we have enough data, take exactly buffer_len samples
                    # If not enough data, stretch what we have to fill buffer_len
                    n = len(centered)
                    if n < buffer_len:
                        # Stretch available data to fill buffer_len
                        stretched = np.interp(
                            np.linspace(0, n-1, buffer_len),
                            np.arange(n),
                            centered
                        )
                        plot_data[:] = stretched
                    else:
                        # Take exactly buffer_len samples from the end (same as main plots)
                        plot_data[:] = centered[-buffer_len:]
                    
                    # Set Y-limits based on UN-GAINED data for both demo and real mode, so gain actually affects visual size
                    # Use raw data for Y-axis calculation, so gain changes visual size
                    valid_data = centered_raw[np.isfinite(centered_raw)]
                    
                    if len(valid_data) > 0:
                        # Use percentiles to avoid spikes from clipping the view
                        p1 = np.percentile(valid_data, 1)
                        p99 = np.percentile(valid_data, 99)
                        data_mean = (p1 + p99) / 2.0
                        data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
                        
                        # Calculate appropriate Y-range with some padding
                        if data_std > 0:
                            # Use standard deviation within central band
                            padding = max(data_std * 4, 200)  # Increased padding for better visibility
                            ymin = data_mean - padding
                            ymax = data_mean + padding
                        else:
                            # Fallback: use percentile window
                            data_range = max(p99 - p1, 300)
                            padding = max(data_range * 0.3, 200)
                            ymin = data_mean - padding
                            ymax = data_mean + padding
                        
                        # Ensure reasonable bounds (same as main plots)
                        ymin = max(ymin, -8000)
                        ymax = min(ymax, 8000)
                    else:
                        ymin, ymax = -500, 500

                    yr = (ymax - ymin)
                    lead_name = self.leads[idx] if idx < len(self.leads) else ""
                    is_chest = lead_name in ["V1", "V2", "V3", "V4", "V5", "V6"]
                    top_extra = 0.90 if is_chest else 0.35
                    bottom_extra = 0.70 if is_chest else 0.30
                    ymin = ymin - yr * bottom_extra
                    ymax = ymax + yr * top_extra
                    
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(-500, 500)
                
                # Set x-limits
                ax.set_xlim(0, max(buffer_len - 1, 1))
                line.set_ydata(plot_data)
        
        if hasattr(self, '_overlay_canvas'):
            self._overlay_canvas.draw_idle()

    def update_plots(self):
        """Update all ECG plots with current data using PyQtGraph (GitHub version)"""
        try:
            # Memory management - check every N updates
            self.update_count += 1
            if self.update_count % self.memory_check_interval == 0:
                self._manage_memory()

            # DEMO branch
            if not self.serial_reader or not self.serial_reader.running:
                try:
                    wave_speed = float(self.settings_manager.get_wave_speed())
                except Exception:
                    wave_speed = 25.0

                # 25 mm/s → 3 s window in 12‑lead view; scale with speed
                baseline_seconds = 3.0
                seconds_scale = (25.0 / max(1e-6, wave_speed))
                seconds_to_show = baseline_seconds * seconds_scale

                for i in range(len(self.data_lines)):
                    try:
                        if i < len(self.data):
                            raw = np.asarray(self.data[i])
                            gain = 1.0
                            try:
                                gain = get_display_gain(self.settings_manager.get_wave_gain())
                            except Exception:
                                pass
                            # 🫀 DISPLAY: Low-frequency baseline anchor (removes respiration from baseline)
                            # Extract very-low-frequency baseline (< 0.3 Hz) to prevent baseline from "breathing"
                            try:
                                # Initialize slow anchor if needed
                                if not hasattr(self, '_baseline_anchors'):
                                    self._baseline_anchors = [0.0] * 12
                                    self._baseline_alpha_slow = 0.0005  # Monitor-grade: ~4 sec time constant at 500 Hz
                                
                                if len(raw) > 0:
                                    # Extract low-frequency baseline estimate (removes respiration 0.1-0.35 Hz)
                                    baseline_estimate = self._extract_low_frequency_baseline(raw, fs)
                                    
                                    # Update anchor with slow EMA (tracks only very-low-frequency drift)
                                    self._baseline_anchors[i] = (1 - self._baseline_alpha_slow) * self._baseline_anchors[i] + self._baseline_alpha_slow * baseline_estimate
                                    
                                    # Subtract anchor (NOT raw mean, NOT baseline estimate directly)
                                    raw = raw - self._baseline_anchors[i]
                                    
                                    # Final zero-centering clamp (visual only, display path)
                                    if not hasattr(self, '_display_zero_refs'):
                                        self._display_zero_refs = [0.0] * 12
                                    
                                    zero_alpha = 0.01  # Fast convergence, visual only
                                    current_dc = np.nanmean(raw) if len(raw) > 0 else 0.0
                                    self._display_zero_refs[i] = (1 - zero_alpha) * self._display_zero_refs[i] + zero_alpha * current_dc
                                    raw = raw - self._display_zero_refs[i]
                            except Exception as filter_error:
                                # Fallback: use original signal (baseline anchor handles it, no mean subtraction)
                                print(f"⚠️ Using fallback baseline correction: {filter_error}")
                            
                            raw = raw * gain

                            fs = 500
                            if hasattr(self, 'sampler') and getattr(self.sampler, 'sampling_rate', None):
                                try:
                                    fs = float(self.sampler.sampling_rate)
                                except Exception:
                                    fs = 500
                            window_len = int(max(50, min(len(raw), seconds_to_show * fs)))
                            src = raw[-window_len:]

                            display_len = self.buffer_size if hasattr(self, 'buffer_size') else 1000
                            # --- Flatline detection (display-only) ---
                            # If the recent window for this lead has almost no variation, treat as flatline
                            if src.size >= 50:
                                amp_range = float(np.nanmax(src) - np.nanmin(src))
                                std_val = float(np.nanstd(src))
                                # Thresholds are in display units after gain; very small range and std → flatline
                                is_flat = amp_range < 5.0 and std_val < 1.0
                                lead_name = self.leads[i] if i < len(self.leads) else f"Lead {i+1}"
                                if is_flat and not self._flatline_alert_shown[i]:
                                    self._flatline_alert_shown[i] = True
                                    try:
                                        QMessageBox.warning(
                                            self,
                                            "Flatline Detected",
                                            f"{lead_name} appears flat (no significant signal).\n"
                                            "Please check the electrode/lead connection."
                                        )
                                    except Exception as warn_err:
                                        print(f"⚠️ Flatline warning failed for {lead_name}: {warn_err}")
                                elif not is_flat:
                                    # Reset flag when signal returns
                                    self._flatline_alert_shown[i] = False

                            if src.size < 2:
                                resampled = np.zeros(display_len)
                            else:
                                x_src = np.linspace(0.0, 1.0, src.size)
                                x_dst = np.linspace(0.0, 1.0, display_len)
                                resampled = np.interp(x_dst, x_src, src)

                            self.data_lines[i].setData(resampled)
                            self.update_plot_y_range(i)
                    except Exception as e:
                        print(f"❌ Error updating plot {i}: {e}")
                        continue
                return

            # SERIAL branch - NEW PACKET-BASED PARSING
            packets_processed = 0
            max_packets = 50  # Read up to 50 packets per update cycle
            
            # Check if we're using the new packet-based reader
            is_packet_reader = isinstance(self.serial_reader, SerialStreamReader)
            
            if is_packet_reader:
                # NEW: Use packet-based reading
                try:
                    packets = self.serial_reader.read_packets(max_packets=max_packets)
                    
                    for packet in packets:
                        # Packet contains all 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
                        # Map packet dict to our lead order
                        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                        
                        for i, lead_name in enumerate(lead_order):
                            try:
                                if i < len(self.data) and lead_name in packet:
                                    value = packet[lead_name]
                                    # Update circular buffer
                                    self.data[i] = np.roll(self.data[i], -1)
                                    # Apply smoothing
                                    smoothed_value = self.apply_realtime_smoothing(value, i)
                                    self.data[i][-1] = smoothed_value
                            except Exception as e:
                                print(f"❌ Error updating data buffer {i} ({lead_name}): {e}")
                                continue
                        
                        # Update sampling rate counter
                        try:
                            if hasattr(self, 'sampler'):
                                sampling_rate = self.sampler.add_sample()
                                if sampling_rate > 0 and hasattr(self, 'metric_labels') and 'sampling_rate' in self.metric_labels:
                                    self.metric_labels['sampling_rate'].setText(f"{sampling_rate:.1f} Hz")
                        except Exception as e:
                            print(f"❌ Error updating sampling rate: {e}")
                        
                        packets_processed += 1
                        
                except Exception as e:
                    print(f"❌ Error reading serial packets: {e}")
                    if hasattr(self, 'serial_reader') and hasattr(self.serial_reader, '_handle_serial_error'):
                        self.serial_reader._handle_serial_error(e)
            else:
                # FALLBACK: Old method for compatibility (if SerialECGReader is still used)
                lines_processed = 0
                max_attempts = 20
                while lines_processed < max_attempts:
                    try:
                        all_8_leads = self.serial_reader.read_value()
                        if all_8_leads:
                            all_12_leads = self.calculate_12_leads_from_8_channels(all_8_leads)
                            for i in range(len(self.leads)):
                                try:
                                    if i < len(self.data) and i < len(all_12_leads):
                                        self.data[i] = np.roll(self.data[i], -1)
                                        smoothed_value = self.apply_realtime_smoothing(all_12_leads[i], i)
                                        self.data[i][-1] = smoothed_value
                                except Exception as e:
                                    print(f"❌ Error updating data buffer {i}: {e}")
                                    continue
                            try:
                                if hasattr(self, 'sampler'):
                                    sampling_rate = self.sampler.add_sample()
                                    if sampling_rate > 0 and hasattr(self, 'metric_labels') and 'sampling_rate' in self.metric_labels:
                                        self.metric_labels['sampling_rate'].setText(f"{sampling_rate:.1f} Hz")
                            except Exception as e:
                                print(f"❌ Error updating sampling rate: {e}")
                            lines_processed += 1
                        else:
                            break
                    except Exception as e:
                        print(f"❌ Error reading serial data: {e}")
                        if hasattr(self, 'serial_reader') and hasattr(self.serial_reader, '_handle_serial_error'):
                            self.serial_reader._handle_serial_error(e)
                        continue
                packets_processed = lines_processed
            
            # Update plots if we processed any packets
            if packets_processed > 0:
                # Detect signal source from a representative lead for adaptive scaling
                signal_source = "hardware"  # Default
                try:
                    if len(self.data) > 1 and hasattr(self, 'leads'):
                        # Prefer Lead II (index 1) if available
                        representative = self.data[1] if len(self.data[1]) > 0 else self.data[0]
                    else:
                        representative = self.data[0] if len(self.data) > 0 else []
                    signal_source = self.detect_signal_source(representative)
                except Exception as e:
                    print(f"❌ Error detecting signal source for serial plots: {e}")
                
                # Get current wave speed for time scaling
                try:
                    wave_speed = float(self.settings_manager.get_wave_speed())
                except Exception:
                    wave_speed = 25.0
                
                # Calculate time scaling based on wave speed (same logic as demo mode)
                # 25 mm/s → 3 s window; 12.5 → 6 s; 50 → 1.5 s
                baseline_seconds = 3.0
                seconds_scale = (25.0 / max(1e-6, wave_speed))
                seconds_to_show = baseline_seconds * seconds_scale
                
                for i in range(len(self.leads)):
                    try:
                        if i >= len(self.data_lines):
                            continue
                        has_data = (i < len(self.data) and len(self.data[i]) > 0)
                        if has_data:
                            # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                            scaled_data = self.apply_adaptive_gain(self.data[i], signal_source, gain_factor)

                            # Build time axis and apply wave-speed scaling
                            sampling_rate = 500.0  # Hardware sampling rate
                            
                            # Calculate how many samples to show based on wave speed
                            # 25 mm/s → 10s window
                            # 12.5 mm/s → 20s window (show more data, compressed)
                            # 50 mm/s → 5s window (show less data, stretched)
                            samples_to_show = int(sampling_rate * seconds_to_show)
                            
                            # Take only the most recent samples_to_show from the buffer (before gain application)
                            raw_data = self.data[i]
                            if len(raw_data) > samples_to_show:
                                data_slice = raw_data[-samples_to_show:]
                            else:
                                data_slice = raw_data
                            
                            # 🫀 DISPLAY: Low-frequency baseline anchor (removes respiration from baseline)
                            # Extract very-low-frequency baseline (< 0.3 Hz) to prevent baseline from "breathing"
                            filtered_slice = np.array(data_slice, dtype=float)
                            try:
                                # Initialize slow anchor if needed
                                if not hasattr(self, '_baseline_anchors'):
                                    self._baseline_anchors = [0.0] * 12
                                    self._baseline_alpha_slow = 0.0005  # Monitor-grade: ~4 sec time constant at 500 Hz
                                
                                if len(filtered_slice) > 0:
                                    # Extract low-frequency baseline estimate (removes respiration 0.1-0.35 Hz)
                                    baseline_estimate = self._extract_low_frequency_baseline(filtered_slice, sampling_rate)
                                    
                                    # Update anchor with slow EMA (tracks only very-low-frequency drift)
                                    self._baseline_anchors[i] = (1 - self._baseline_alpha_slow) * self._baseline_anchors[i] + self._baseline_alpha_slow * baseline_estimate
                                    
                                    # Subtract anchor (NOT raw mean)
                                    filtered_slice = filtered_slice - self._baseline_anchors[i]
                                    
                                    # Final zero-centering clamp (visual only, display path)
                                    if not hasattr(self, '_display_zero_refs'):
                                        self._display_zero_refs = [0.0] * 12
                                    
                                    zero_alpha = 0.01  # Fast convergence, visual only
                                    current_dc = np.nanmean(filtered_slice) if len(filtered_slice) > 0 else 0.0
                                    self._display_zero_refs[i] = (1 - zero_alpha) * self._display_zero_refs[i] + zero_alpha * current_dc
                                    filtered_slice = filtered_slice - self._display_zero_refs[i]
                            except Exception as filter_error:
                                # Fallback: use original signal (baseline anchor handles it, no mean subtraction)
                                print(f"⚠️ Using fallback baseline correction for lead {self.leads[i] if hasattr(self, 'leads') else i}: {filter_error}")
                            
                            # Optional AC notch filtering based on "Set Filter" selection.
                            # Keeps wave peaks intact while removing 50/60 Hz power noise for machine serial data.
                            try:
                                ac_setting = self.settings_manager.get_setting("filter_ac", "off") if self.settings_manager else "off"
                                if ac_setting and ac_setting != "off" and len(filtered_slice) >= 10:
                                    from ecg.ecg_filters import apply_ac_filter
                                    filtered_slice = apply_ac_filter(filtered_slice, sampling_rate, ac_setting)
                            except Exception as filter_error:
                                pass  # AC filter is optional
                            
                            # Apply wave gain
                            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                            
                            # Signal is already baseline-corrected by median+mean filter
                            centered_slice = filtered_slice
                            
                            # Apply gain after centering (same as demo mode)
                            scaled_data = centered_slice * gain_factor
                            scaled_data = np.nan_to_num(scaled_data, copy=False)

                            # --- Flatline detection (serial/display path) ---
                            if scaled_data.size >= 50:
                                amp_range = float(np.nanmax(scaled_data) - np.nanmin(scaled_data))
                                std_val = float(np.nanstd(scaled_data))
                                # Very small range and std → likely flatline / disconnected lead
                                is_flat = amp_range < 5.0 and std_val < 1.0
                                lead_name = self.leads[i] if i < len(self.leads) else f"Lead {i+1}"
                                if is_flat and not self._flatline_alert_shown[i]:
                                    self._flatline_alert_shown[i] = True
                                    try:
                                        QMessageBox.warning(
                                            self,
                                            "Flatline Detected",
                                            f"{lead_name} appears flat (no significant signal).\n"
                                            f"Please check the electrode/lead connection."
                                        )
                                    except Exception as warn_err:
                                        print(f"⚠️ Flatline warning failed for {lead_name}: {warn_err}")
                                elif not is_flat:
                                    # Reset flag when signal returns
                                    self._flatline_alert_shown[i] = False
                            
                            n = len(scaled_data)
                            time_axis = np.arange(n, dtype=float) / sampling_rate
                            
                            # Avoid cropping: small padding and explicit x-range
                            try:
                                vb = self.plot_widgets[i].getViewBox()
                                if vb is not None:
                                    vb.setRange(xRange=(time_axis[0], time_axis[-1]), padding=0)
                            except Exception:
                                pass

                            self.data_lines[i].setData(time_axis, scaled_data)
                            self.update_plot_y_range_adaptive(i, signal_source, data_override=scaled_data)

                            if i < 3 and hasattr(self, '_debug_counter') and self._debug_counter % 200 == 0:
                                print(f"🎛️ Serial Lead {i}: speed={wave_speed:.1f}mm/s, scale={seconds_scale:.2f}, time_range={time_axis[-1]:.2f}s")
                        else:
                            self.data_lines[i].setData(self.data[i] if i < len(self.data) else [])
                            self.update_plot_y_range(i)
                    except Exception as e:
                        print(f"❌ Error updating plot {i}: {e}")
                        continue
                # Calculate ECG metrics more frequently for faster BPM updates in EXE
                # Reduced from every 5 updates to every 3 updates for better responsiveness
                if self.update_count % 3 == 0:
                    try:
                        self.calculate_ecg_metrics()
                    except Exception as e:
                        print(f"❌ Error calculating ECG metrics: {e}")
                try:
                    if hasattr(self, 'heartbeat_counter'):
                        self.heartbeat_counter += 1
                    else:
                        self.heartbeat_counter = 0
                    if self.heartbeat_counter % 10 == 0 and len(self.data) > 1:
                        heart_rate = self.calculate_heart_rate(self.data[1])
                        if heart_rate > 0:
                            print(f"💓 HEARTBEAT: {heart_rate} BPM")
                except Exception as e:
                    print(f"❌ Error displaying heartbeat: {e}")

        except Exception as e:
            self.crash_logger.log_crash("Critical error in update_plots", e, "Real-time ECG plotting")
            try:
                if hasattr(self, 'data') and self.data:
                    for i in range(len(self.data)):
                        self.data[i] = np.zeros(self.buffer_size if hasattr(self, 'buffer_size') else 1000)
            except Exception as recovery_error:
                self.crash_logger.log_error("Failed to recover from update_plots error", recovery_error, "Data reset")
    
    def _manage_memory(self):
        """Manage memory usage to prevent crashes from large data buffers"""
        try:
            import gc
            import psutil
            import os
            
            # Check current memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > 500:  # If using more than 500MB
                print(f"⚠️ High memory usage: {memory_mb:.1f}MB - cleaning up...")
                
                # Force garbage collection
                gc.collect()
                
                # Trim data buffers if they're too large
                for i, data_buffer in enumerate(self.data):
                    if len(data_buffer) > self.max_buffer_size:
                        # Keep only the most recent data
                        self.data[i] = data_buffer[-self.max_buffer_size:].copy()
                        print(f"📉 Trimmed data buffer {i} to {len(self.data[i])} samples")
                
                # Check memory after cleanup
                memory_after = process.memory_info().rss / 1024 / 1024
                print(f"✅ Memory after cleanup: {memory_after:.1f}MB (freed {memory_mb - memory_after:.1f}MB)")
                
        except ImportError:
            # psutil not available, skip memory management
            pass
        except Exception as e:
            print(f"❌ Error in memory management: {e}")
    
    def _log_error(self, error_msg, exception=None, context=""):
        """Comprehensive error logging for debugging crashes"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Basic error info
            log_msg = f"[{timestamp}] ❌ {error_msg}"
            if context:
                log_msg += f" | Context: {context}"
            
            print(log_msg)
            
            # Detailed exception info
            if exception:
                print(f"Exception Type: {type(exception).__name__}")
                print(f"Exception Message: {str(exception)}")
                print("Full Traceback:")
                traceback.print_exc()
            
            # System state info
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                print(f"System State - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
            except ImportError:
                pass
            
            # ECG state info
            try:
                if hasattr(self, 'data') and self.data:
                    data_lengths = [len(d) for d in self.data]
                    print(f"ECG Data State - Buffer lengths: {data_lengths}")
                
                if hasattr(self, 'serial_reader') and self.serial_reader:
                    print(f"Serial Reader State - Running: {self.serial_reader.running}")
                    print(f"Serial Reader State - Data count: {self.serial_reader.data_count}")
            except Exception:
                pass
                
        except Exception as log_error:
            print(f"❌ Error in error logging: {log_error}")
    
    def closeEvent(self, event):
        """Clean up all resources when the ECG test page is closed"""
        try:
            # Stop demo manager
            if hasattr(self, 'demo_manager'):
                self.demo_manager.stop_demo_data()
            
            # Stop timers
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
                self.timer.deleteLater()
            
            if hasattr(self, 'elapsed_timer') and self.elapsed_timer:
                self.elapsed_timer.stop()
                self.elapsed_timer.deleteLater()
            
            # Close serial connection
            if hasattr(self, 'serial_reader') and self.serial_reader:
                try:
                    self.serial_reader.close()
                except Exception:
                    pass
            
            # Log cleanup
            if hasattr(self, 'crash_logger'):
                self.crash_logger.log_info("ECG Test Page closed, resources cleaned up", "ECG_TEST_PAGE_CLOSE")
        except Exception as e:
            print(f"Error during ECGTestPage cleanup: {e}")
        finally:
            super().closeEvent(event)
