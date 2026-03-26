import os
import sys
import time
import threading
import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QMessageBox, QLabel
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils.helpers import safe_print

# Use safe_print everywhere in this module to avoid Unicode issues on Windows consoles
print = safe_print


class DemoManager:
    
    def __init__(self, ecg_test_page):

        self.ecg_test_page = ecg_test_page
        debug_env = os.getenv("ECG_DEMO_DEBUG", "").strip().lower()
        self._debug_logging = debug_env in ("1", "true", "yes", "on")
        self._debug_counter = 0
        self.demo_fs = 200  # Demo sampling rate (200 Hz)
        self.demo_thread = None
        self.demo_timer = None
        self.demo_heart_rates = []  # For heart rate smoothing
        # Thread coordination
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # Track previous user-selected wave speed to restore after demo
        self._previous_wave_speed = None
        
        # Wave speed control variables (like divyansh.py)
        self.current_wave_speed = 25.0  # mm/s (default)
        self.samples_per_second = 150  # Base sampling rate for CSV demo
        self.time_window = 10.0  # Default time window (25mm/s)
        
        # Data pointer for plotting (like divyansh.py)
        self.data_ptr = 0
        
        # Provide dashboard with effective sampling rate when in demo
        self._set_demo_sampling_rate(self.samples_per_second)
        # Fixed metrics store for demo stability
        self._demo_fixed_metrics = None
        # Internal running flag to avoid touching Qt widgets from worker thread
        self._running_demo = False
        # Warmup control to avoid distorted first seconds
        self._warmup_until = 0.0
        self._baseline_means = {}
        # Track demo start time for live time metric
        self._demo_started_at = None
        self._demo_paused_time = None  # Track paused time for resume
        # Smooth Y-range targets per lead so axes react gently to gain changes
        self._demo_lead_ranges = {}
        # Delay initial plot display by 15 seconds to avoid jumping/stabilization period
        self._initial_display_delay = 15.0  # Wait 15 seconds before first display
        self._initial_display_start = None  # Set when demo starts
        self._first_display_done = False  # Track if first display completed
        # Plot update coordination
        self._plot_running = False
        self._skipped_plot_calls = 0
        # Performance optimization: throttle expensive operations
        self._last_percentile_update = {}  # Track last percentile calculation per lead
        self._last_dashboard_update = 0.0  # Track last dashboard callback
        self._percentile_update_interval = 0.1  # Update percentile every 100ms (10x per second)
        self._dashboard_update_interval = 0.2  # Update dashboard every 200ms (5x per second)
        self._cached_baseline_envelope = {}  # Cache baseline envelope values per lead
        # Stop threads if the page is destroyed
        try:
            self.ecg_test_page.destroyed.connect(self._on_page_destroyed)
        except Exception:
            pass
    
    def _get_calculation_functions(self):
       
        from .twelve_lead_test import calculate_st_segment
        return calculate_st_segment
    
    def _calculate_derived_leads(self, lead_I_value, lead_II_value):
        """
        Calculate derived leads (III, aVR, aVL, aVF) from Lead I and Lead II.
        Uses the same formulas as twelve_lead_test.py (lines 1833-1859).
        """
        try:
            # Lead III = Lead II - Lead I
            III = lead_II_value - lead_I_value
        except Exception:
            III = 0.0
        
        try:
            # aVR = -(Lead I + Lead II) / 2.0
            aVR = -(lead_I_value + lead_II_value) / 2.0
        except Exception:
            aVR = 0.0
        
        try:
            # aVL = (Lead I - Lead III) / 2
            aVL = (lead_I_value - III) / 2.0
        except Exception:
            aVL = 0.0
        
        try:
            # aVF = (Lead II + Lead III) / 2
            aVF = (lead_II_value + III) / 2.0
        except Exception:
            aVF = 0.0
        
        return III, aVR, aVL, aVF
    
    def _update_wave_speed_settings(self):
        """
        Update wave speed settings based on current settings manager.
        This affects the visual speed and spacing of ECG waves.
        EXACT SAME LOGIC AS DIVYANSH.PY
        """
        # Get current wave speed from settings
        self.current_wave_speed = self.ecg_test_page.settings_manager.get_wave_speed()
        
        # EXACT SAME LOGIC AS DIVYANSH.PY:
        # 12.5 mm/s → 10 s window (more peaks visible - compressed)
        # 25 mm/s → 5 s window (default)
        # 50 mm/s → 2.5 s window (fewer peaks visible - stretched)
        baseline_time_window = 5.0
        self.time_window = baseline_time_window * (25.0 / float(self.current_wave_speed))
        
        print(f" Wave speed updated: {self.current_wave_speed}mm/s (time window: {self.time_window:.1f}s)")

    
    
    def _set_demo_sampling_rate(self, sampling_rate):
        """Expose effective sampling rate to dashboard calculators via ecg_test_page.sampler."""
        try:
            if not hasattr(self.ecg_test_page, 'sampler') or self.ecg_test_page.sampler is None:
                self.ecg_test_page.sampler = type('Sampler', (), {})()
            # Keep dashboard filter stable: clamp to >=80 Hz (new default)
            safe_fs = max(80.0, float(sampling_rate))
            self.ecg_test_page.sampler.sampling_rate = safe_fs
        except Exception:
            pass

    def toggle_demo_mode(self, is_checked):
        
        if is_checked:
            # If real acquisition is running, prevent enabling demo
            try:
                if hasattr(self.ecg_test_page, 'timer') and self.ecg_test_page.timer.isActive():
                    self.ecg_test_page.demo_toggle.setChecked(False)
                    return
            except Exception as e:
                print(f"[Demo Toggle] Check real run state failed: {e}")

            # Demo is being turned ON - disable hardware controls
            self.ecg_test_page.demo_toggle.setText("Demo Mode: ON")
            self._disable_hardware_controls()

            # Get current wave speed and print it
            current_speed = self.ecg_test_page.settings_manager.get_wave_speed()
            # current_gain removed from recent changes

            print(" Demo mode ON - Starting demo data...")

            # Force demo start to 50mm/s regardless of previous selection (demo-only behavior)
            try:
                sm = self.ecg_test_page.settings_manager
                # Preserve previous string value to restore later
                self._previous_wave_speed = str(sm.get_setting("wave_speed", "25"))
                if self._previous_wave_speed != "25":
                    sm.set_setting("wave_speed", "25")
                # Ensure UI reacts immediately to the forced setting
                self.ecg_test_page.on_settings_changed("wave_speed", "25")
            except Exception as e:
                print(f" Could not enforce demo wave speed: {e}")
            
            # Update wave speed settings before starting (like divyansh.py)
            self._update_wave_speed_settings()
            
            # Reset fixed metrics for new demo session
            self._demo_fixed_metrics = None
            # Ensure fresh start - clear any paused time
            self._demo_paused_time = None
            self._demo_started_at = None
            self._running_demo = True
            self._demo_lead_ranges.clear()

            try:
                self.ecg_test_page._use_dashboard_metrics_only = False
                self.ecg_test_page._last_computed_metrics = {}
            except Exception:
                pass
            
            # Don't start the dashboard timer here - it will be handled by start_demo_data()
            # which resets _demo_started_at and the dashboard will sync via session_paused_time
            
                        # Immediately set hardcoded demo metrics on ECG test page and dashboard
            try:
                # Update ECG test page metrics
                if hasattr(self.ecg_test_page, 'metric_labels'):
                    self.ecg_test_page.metric_labels.get('heart_rate', QLabel()).setText("60")
                    self.ecg_test_page.metric_labels.get('pr_interval', QLabel()).setText("167")
                    self.ecg_test_page.metric_labels.get('qrs_duration', QLabel()).setText("86")
                    self.ecg_test_page.metric_labels.get('st_interval', QLabel()).setText("92")  # P duration
                    # QTc label may be named 'qtc_interval' depending on UI
                    if 'qtc_interval' in self.ecg_test_page.metric_labels:
                        self.ecg_test_page.metric_labels['qtc_interval'].setText("357/357")
                    print(" Demo metrics set on ECG test page")
                
                # Update dashboard metrics (same values)
                dashboard = getattr(self.ecg_test_page, 'dashboard_instance', None)
                if dashboard is None:
                    dashboard = getattr(self.ecg_test_page, 'parent', None)
                if dashboard is not None and hasattr(dashboard, 'metric_labels'):
                    dashboard.metric_labels.get('heart_rate', QLabel()).setText("60 BPM")
                    dashboard.metric_labels.get('pr_interval', QLabel()).setText("167 ms")
                    dashboard.metric_labels.get('qrs_duration', QLabel()).setText("86 ms")
                    dashboard.metric_labels.get('st_interval', QLabel()).setText("92 ms")  # P duration
                    if 'qtc_interval' in dashboard.metric_labels:
                        dashboard.metric_labels['qtc_interval'].setText("357/357 ms")
                    # Optional: reflect demo sampling rate if shown
                    dashboard.metric_labels.get('sampling_rate', QLabel()).setText("80 Hz")
                    print(" Demo metrics set on dashboard")
            except Exception as e:
                print(f" Could not set demo metrics: {e}")
            
            # Start demo data generation in the existing 12-lead grid
            self.start_demo_data()
            
            # Start elapsed timer for demo mode (sync with demo start time)
            # ALWAYS reset start_time to current time for fresh start
            try:
                if hasattr(self.ecg_test_page, 'elapsed_timer') and self.ecg_test_page.elapsed_timer:
                    # Force reset start_time to current time for fresh start
                    self.ecg_test_page.start_time = time.time()
                    self.ecg_test_page.paused_duration = 0
                    # Start elapsed timer - ensure it's stopped first to avoid duplicates
                    if self.ecg_test_page.elapsed_timer.isActive():
                        self.ecg_test_page.elapsed_timer.stop()
                    self.ecg_test_page.elapsed_timer.start(1000)  # Update every 1 second
                    print(" Elapsed timer started for demo mode (fresh start)")
            except Exception as e:
                print(f" Could not start elapsed timer for demo: {e}")
            
            # Start the dashboard timer when demo begins (after start_demo_data sets _demo_started_at)
            try:
                if hasattr(self.ecg_test_page, 'parent') and hasattr(self.ecg_test_page.parent, 'start_acquisition_timer'):
                    self.ecg_test_page.parent.start_acquisition_timer()
            except Exception:
                pass
            
        else:
            # Demo is being turned OFF - enable hardware controls
            self.ecg_test_page.demo_toggle.setText("Demo Mode: OFF")
            print(" Demo mode OFF - Stopping demo data...")

            try:
                if hasattr(self.ecg_test_page, 'hide_demo_wave_gain'):
                    self.ecg_test_page.hide_demo_wave_gain()
            except Exception as hide_err:
                print(f" Could not hide demo wave gain display: {hide_err}")

            # Restore user's previous wave speed selection after demo
            try:
                if self._previous_wave_speed is not None:
                    sm = self.ecg_test_page.settings_manager
                    sm.set_setting("wave_speed", self._previous_wave_speed)
                    self.ecg_test_page.on_settings_changed("wave_speed", self._previous_wave_speed)
            except Exception as e:
                print(f" Could not restore wave speed after demo: {e}")
            finally:
                self._previous_wave_speed = None
            
            try:
                dashboard = getattr(self.ecg_test_page, 'dashboard_instance', None)
                if dashboard is None:
                    dashboard = getattr(self.ecg_test_page, 'parent', None)
                if hasattr(self.ecg_test_page, 'metric_labels'):
                    ml = self.ecg_test_page.metric_labels
                    ml.get('heart_rate', QLabel()).setText("0")
                    ml.get('pr_interval', QLabel()).setText("0")
                    ml.get('qrs_duration', QLabel()).setText("0")
                    if 'st_interval' in ml:
                        ml['st_interval'].setText("0")
                    if 'qtc_interval' in ml:
                        ml['qtc_interval'].setText("0/0")

                if dashboard is not None and hasattr(dashboard, 'metric_labels'):
                    dml = dashboard.metric_labels
                    dml.get('heart_rate', QLabel()).setText("0 BPM")
                    dml.get('pr_interval', QLabel()).setText("0 ms")
                    dml.get('qrs_duration', QLabel()).setText("0 ms")
                    if 'st_interval' in dml:
                        dml['st_interval'].setText("0 ms")
                    if 'qtc_interval' in dml:
                        dml['qtc_interval'].setText("0/0 ms")
            except Exception as e:
                print(f" Could not reset demo metrics on dashboard: {e}")
            
            # Reset time tracking variables to zero
            self._demo_started_at = None
            self._demo_paused_time = None
            
            # Reset elapsed timer and time display
            # IMPORTANT: Set start_time to None FIRST, then stop timer, then reset display
            # This prevents timer callback from updating time after we reset it
            try:
                # First, set start_time to None so timer callback won't update time
                if hasattr(self.ecg_test_page, 'start_time'):
                    self.ecg_test_page.start_time = None
                if hasattr(self.ecg_test_page, 'paused_duration'):
                    self.ecg_test_page.paused_duration = 0
                
                # Now stop the timer
                if hasattr(self.ecg_test_page, 'elapsed_timer') and self.ecg_test_page.elapsed_timer:
                    if self.ecg_test_page.elapsed_timer.isActive():
                        self.ecg_test_page.elapsed_timer.stop()
                    print(" Elapsed timer stopped (demo mode OFF)")
                
                # Force reset time display AFTER stopping timer
                if hasattr(self.ecg_test_page, 'metric_labels') and 'time_elapsed' in self.ecg_test_page.metric_labels:
                    self.ecg_test_page.metric_labels['time_elapsed'].setText("00:00")
                    # Reset the last displayed elapsed time to prevent timer callback from updating
                    if hasattr(self.ecg_test_page, '_last_displayed_elapsed'):
                        self.ecg_test_page._last_displayed_elapsed = -1
                    print(" Time display reset to 00:00")
                
                # Reset dashboard time tracking and stop its timer
                try:
                    if hasattr(self.ecg_test_page, 'parent') and hasattr(self.ecg_test_page.parent, 'session_start_time'):
                        dashboard = self.ecg_test_page.parent
                        
                        # First, set session_start_time to None so timer callback won't update time
                        dashboard.session_start_time = None
                        dashboard.session_paused_time = None
                        # Reset other session tracking variables if they exist
                        if hasattr(dashboard, 'session_total_paused_time'):
                            dashboard.session_total_paused_time = 0
                        if hasattr(dashboard, 'session_last_elapsed'):
                            dashboard.session_last_elapsed = 0
                        
                        # Stop dashboard timer if it exists
                        if hasattr(dashboard, 'acquisition_timer') and dashboard.acquisition_timer:
                            if dashboard.acquisition_timer.isActive():
                                dashboard.acquisition_timer.stop()
                        
                        # Force reset dashboard time display AFTER stopping timer
                        if hasattr(dashboard, 'metric_labels') and 'time_elapsed' in dashboard.metric_labels:
                            dashboard.metric_labels['time_elapsed'].setText("00:00")
                        
                        print(" Dashboard time tracking reset and timer stopped")
                except Exception as e:
                    print(f" Could not reset dashboard time tracking: {e}")
            except Exception as e:
                print(f" Could not reset time tracking: {e}")
            
            # Reset fixed metrics for fresh start next time
            self._demo_fixed_metrics = None
            
            # Stop demo data generation
            self.stop_demo_data()
            
            self._enable_hardware_controls()
            self._demo_lead_ranges.clear()
    
    def _disable_hardware_controls(self):
        """Disable hardware control buttons when demo mode is ON"""
        try:
            if hasattr(self.ecg_test_page, 'start_btn'):
                self.ecg_test_page.start_btn.setEnabled(False)
                self.ecg_test_page.start_btn.setStyleSheet("""
                    QPushButton {
                        background: #6c757d;
                        color: #ffffff;
                        border: 2px solid #6c757d;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #5a6268;
                    }
                """)
            if hasattr(self.ecg_test_page, 'stop_btn'):
                self.ecg_test_page.stop_btn.setEnabled(False)
                self.ecg_test_page.stop_btn.setStyleSheet("""
                    QPushButton {
                        background: #6c757d;
                        color: #ffffff;
                        border: 2px solid #6c757d;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #5a6268;
                    }
                """)
            if hasattr(self.ecg_test_page, 'ports_btn'):
                self.ecg_test_page.ports_btn.setEnabled(False)
                self.ecg_test_page.ports_btn.setStyleSheet("""
                    QPushButton {
                        background: #6c757d;
                        color: #ffffff;
                        border: 2px solid #6c757d;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #5a6268;
                    }
                """)
            print(" Hardware controls disabled (Demo mode ON)")
        except Exception as e:
            print(f" Error disabling hardware controls: {e}")
    
    def _enable_hardware_controls(self):
        """Enable hardware control buttons when demo mode is OFF"""
        try:
            if hasattr(self.ecg_test_page, 'start_btn'):
                self.ecg_test_page.start_btn.setEnabled(True)
                self.ecg_test_page.start_btn.setStyleSheet("""
                    QPushButton {
                        background: #28a745;
                        color: white;
                        border: 2px solid #28a745;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #218838;
                    }
                """)
            if hasattr(self.ecg_test_page, 'stop_btn'):
                self.ecg_test_page.stop_btn.setEnabled(True)
                self.ecg_test_page.stop_btn.setStyleSheet("""
                    QPushButton {
                        background: #28a745;
                        color: white;
                        border: 2px solid #28a745;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #218838;
                    }
                """)
            if hasattr(self.ecg_test_page, 'ports_btn'):
                self.ecg_test_page.ports_btn.setEnabled(True)
                self.ecg_test_page.ports_btn.setStyleSheet("""
                    QPushButton {
                        background: #28a745;
                        color: white;
                        border: 2px solid #28a745;
                        border-radius: 8px;
                        padding: 4px 8px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #218838;
                    }
                """)
            print(" Hardware controls enabled (Demo mode OFF)")
        except Exception as e:
            print(f" Error enabling hardware controls: {e}")
    
    def start_demo_data(self):
        """Start reading real ECG data from dummycsv.csv file with wave speed control"""
        # Ensure any previous demo resources are stopped before starting new
        try:
            self._stop_event.set()
            if self.demo_thread and self.demo_thread.is_alive():
                self.demo_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self._stop_event.clear()
            self.demo_thread = None
            # Stop and delete any existing timer
            try:
                if self.demo_timer:
                    self.demo_timer.stop()
                    self.demo_timer.deleteLater()
            except Exception:
                pass
            self.demo_timer = None
        # Resolve dummycsv.csv from common locations including PyInstaller bundle
        ecg_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(ecg_dir, '..'))
        project_root = os.path.abspath(os.path.join(src_dir, '..'))
        
        # Check if running as PyInstaller bundle
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            bundle_dir = sys._MEIPASS
            candidates = [
                os.path.join(bundle_dir, 'dummycsv.csv'),
                os.path.join(bundle_dir, '_internal', 'dummycsv.csv'),
                os.path.join(os.path.dirname(sys.executable), 'dummycsv.csv'),
                os.path.join(ecg_dir, 'dummycsv.csv'),
                os.path.join(project_root, 'dummycsv.csv'),
                os.path.abspath('dummycsv.csv')
            ]
        else:
            # Running as script
            candidates = [
                os.path.join(ecg_dir, 'dummycsv.csv'),
                os.path.join(project_root, 'dummycsv.csv'),
                os.path.abspath('dummycsv.csv')
            ]
        csv_path = None
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break
        if not csv_path:
            msg = (
                "dummycsv.csv not found. Place it in one of these locations:\n"
                f"- {os.path.join(ecg_dir, 'dummycsv.csv')}\n"
                f"- {os.path.join(project_root, 'dummycsv.csv')}\n"
                f"- {os.path.abspath('dummycsv.csv')}"
            )
            QMessageBox.warning(self.ecg_test_page, "Demo file missing", msg)
            # Don't start demo if CSV is missing
            self.ecg_test_page.demo_toggle.setChecked(False)
            return
        
        try:
            df = pd.read_csv(csv_path)
            if len(df.columns) == 1:
                df = pd.read_csv(csv_path, sep='\t')
            print(f" Successfully loaded {len(df)} rows from dummycsv.csv")
            
            # Get all lead columns (excluding 'Sample' column)
            lead_columns = [col for col in df.columns if col != 'Sample']
            print(f" Found leads: {lead_columns}")
            
            # Clear existing data - data is a list of numpy arrays, not a dictionary
            for i in range(len(self.ecg_test_page.data)):
                self.ecg_test_page.data[i] = np.zeros(self.ecg_test_page.buffer_size)
            
            # Initialize data with enough rows to fill entire buffer for instant display
            # Prefill ENTIRE buffer to show complete graph immediately (not just 4 seconds)
            csv_base_fs = 80  # demo CSV base aligned to new default
            # Prefill full buffer_size OR 20 seconds worth, whichever is smaller
            # This ensures graph appears immediately with proper data
            buffer_size = self.ecg_test_page.buffer_size
            seconds_to_prefill = 20.0  # Prefill 20 seconds for instant display
            samples_needed = int(csv_base_fs * seconds_to_prefill)
            prefill_needed = min(buffer_size, samples_needed, len(df))
            
            print(f" Prefilling {prefill_needed} samples ({prefill_needed/csv_base_fs:.1f}s) for instant graph display")
            
            # First, read raw leads from CSV (I, II, V1-V6)
            raw_lead_data = {}
            for lead in lead_columns:
                if lead in self.ecg_test_page.leads:
                    lead_index = self.ecg_test_page.leads.index(lead)
                    # Add initial prefill samples to fill entire buffer
                    initial_samples = prefill_needed
                    series = df[lead].iloc[:initial_samples]
                    count = min(len(series), buffer_size)
                    arr = np.array(series[:count], dtype=float)
                    # Record per‑lead baseline from the first 200 samples (or all available)
                    baseline_window = max(1, min(200, arr.size))
                    baseline_mean = float(np.mean(arr[:baseline_window])) if arr.size > 0 else 0.0
                    self._baseline_means[lead_index] = baseline_mean
                    # Store raw data (will apply baseline centering later)
                    raw_lead_data[lead] = arr
            
            # Now calculate derived leads (III, aVR, aVL, aVF) from Lead I and Lead II
            # Lead indices: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5
            if 'I' in raw_lead_data and 'II' in raw_lead_data:
                lead_I_arr = raw_lead_data['I']
                lead_II_arr = raw_lead_data['II']
                count = min(len(lead_I_arr), len(lead_II_arr))
                
                # Calculate derived leads sample by sample
                III_arr = np.zeros(count, dtype=float)
                aVR_arr = np.zeros(count, dtype=float)
                aVL_arr = np.zeros(count, dtype=float)
                aVF_arr = np.zeros(count, dtype=float)
                
                for i in range(count):
                    III_val, aVR_val, aVL_val, aVF_val = self._calculate_derived_leads(
                        lead_I_arr[i], lead_II_arr[i]
                    )
                    III_arr[i] = III_val
                    aVR_arr[i] = aVR_val
                    aVL_arr[i] = aVL_val
                    aVF_arr[i] = aVF_val
                
                # Store calculated derived leads
                raw_lead_data['III'] = III_arr
                raw_lead_data['aVR'] = aVR_arr
                raw_lead_data['aVL'] = aVL_arr
                raw_lead_data['aVF'] = aVF_arr
                
                print(" Calculated derived leads (III, aVR, aVL, aVF) from Lead I and Lead II")
            
            # 🔧 FIX: Store RAW CSV values in buffer (no baseline centering during prefill)
            # Baseline centering will be skipped during first 15 seconds in plotting function
            # This ensures exact CSV values are plotted for first 15 seconds
            for lead in raw_lead_data:
                if lead in self.ecg_test_page.leads:
                    lead_index = self.ecg_test_page.leads.index(lead)
                    arr = raw_lead_data[lead]
                    count = min(len(arr), buffer_size)
                    # Store RAW values (no baseline centering) for exact CSV plotting
                    raw_arr = arr[:count]
                    
                    # Fill the ENTIRE buffer by repeating the prefill data pattern
                    # This ensures smooth display without gaps or flat lines
                    if count > 0:
                        # Tile the raw data to fill the entire buffer
                        repeats_needed = (buffer_size // count) + 1
                        tiled_data = np.tile(raw_arr, repeats_needed)
                        # Fill buffer with properly tiled data (no gaps, no flat padding)
                        self.ecg_test_page.data[lead_index][:buffer_size] = tiled_data[:buffer_size]
                    else:
                        # Fallback: zero-fill if no data
                        self.ecg_test_page.data[lead_index][:] = 0.0
                        
            # Don't pre-calculate Y-ranges here - wait for 15 seconds of data collection
            # Ranges will be calculated in _calculate_initial_stable_ranges() after stabilization
            
            # Set initial display delay - wait 15 seconds before showing plots to avoid jumping
            self._initial_display_start = time.time() + self._initial_display_delay
            self._warmup_until = self._initial_display_start  # Delay display until stable
            self._first_display_done = False  # Reset first display flag for new demo session
            
            # Set data pointer to start of buffer
            self.data_ptr = 0
            
            # ALWAYS start fresh - reset demo start time to current time
            # (Demo mode should always start from zero, not resume from paused time)
            self._demo_started_at = time.time()
            self._demo_paused_time = None  # Ensure paused time is cleared
            print(f" Demo starting - collecting data for {self._initial_display_delay} seconds before display (to avoid jumping)")
            
            # DO NOT update plots immediately - wait for stabilization period
            # Data will be collected in background, but plots won't update until stable

            # Start reading data row by row from CSV with wave speed control
            def read_csv_data():
                try:
                    # Start from where prefill ended (loop if needed)
                    row_index = prefill_needed % len(df) if prefill_needed >= len(df) else prefill_needed
                    consecutive_errors = 0
                    max_consecutive_errors = 10
                    
                    while (not self._stop_event.is_set()) and self._running_demo and row_index < len(df):
                        try:
                            # Validate row index
                            if row_index >= len(df) or row_index < 0:
                                print(f" Invalid row index: {row_index}")
                                row_index += 1
                                continue
                            
                            # First, read raw leads from CSV (I, II, V1-V6)
                            raw_lead_values = {}
                            for lead in lead_columns:
                                try:
                                    if lead in self.ecg_test_page.leads:
                                        # Get value with validation
                                        value = df[lead].iloc[row_index]
                                        
                                        # Validate value
                                        if pd.isna(value) or np.isnan(value) or np.isinf(value):
                                            value = 0.0
                                        
                                        # Convert to float safely
                                        try:
                                            value = float(value)
                                        except (ValueError, TypeError):
                                            value = 0.0
                                        
                                        raw_lead_values[lead] = value
                                except Exception as e:
                                    print(f" Error reading lead {lead} at row {row_index}: {e}")
                                    raw_lead_values[lead] = 0.0
                            
                            # Calculate derived leads (III, aVR, aVL, aVF) from Lead I and Lead II
                            # Lead indices: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5
                            if 'I' in raw_lead_values and 'II' in raw_lead_values:
                                lead_I_val = raw_lead_values['I']
                                lead_II_val = raw_lead_values['II']
                                
                                III_val, aVR_val, aVL_val, aVF_val = self._calculate_derived_leads(
                                    lead_I_val, lead_II_val
                                )
                                
                                # Store calculated derived leads
                                raw_lead_values['III'] = III_val
                                raw_lead_values['aVR'] = aVR_val
                                raw_lead_values['aVL'] = aVL_val
                                raw_lead_values['aVF'] = aVF_val
                            
                            # Now update all leads (raw + derived) in data buffers
                            with self._lock:
                                for lead, value in raw_lead_values.items():
                                    try:
                                        if lead in self.ecg_test_page.leads:
                                            lead_index = self.ecg_test_page.leads.index(lead)
                                            
                                            if (hasattr(self.ecg_test_page, 'data') and 
                                                lead_index < len(self.ecg_test_page.data) and
                                                len(self.ecg_test_page.data[lead_index]) > 0):
                                                # 🫀 CLINICAL: Store RAW value in data buffer (for clinical analysis)
                                                # Do NOT apply baseline centering here - that's display-only
                                                raw_value = float(value)
                                                self.ecg_test_page.data[lead_index] = np.roll(
                                                    self.ecg_test_page.data[lead_index], -1)
                                                self.ecg_test_page.data[lead_index][-1] = raw_value
                                    except Exception as e:
                                        print(f" Error updating lead {lead} at row {row_index}: {e}")
                                        continue
                                
                                # ── Feed packet to HolterStreamWriter if recording ─────────────
                                try:
                                    if (hasattr(self.ecg_test_page, '_holter_writer') and 
                                        self.ecg_test_page._holter_writer and 
                                        self.ecg_test_page._holter_writer.is_running):
                                        self.ecg_test_page._holter_writer.push(raw_lead_values)
                                except Exception:
                                    pass
                            
                            row_index += 1
                            consecutive_errors = 0  # Reset error counter on success
                            
                            # Loop back to beginning if we reach the end 
                            if row_index >= len(df):
                                row_index = 0
                                print(" Restarting ECG data from beginning...")
                            
                            # Dynamic delay based on wave speed with error handling
                            try:
                                base_delay = 1.0 / 80.0  # 80 samples per second base (matching new default)
                                # Use time window to calculate delay (like divyansh.py)
                                speed_factor = getattr(self, 'time_window', 10.0) / 10.0
                                actual_delay = max(0.001, base_delay * speed_factor)
                                time.sleep(actual_delay)
                            except Exception as e:
                                print(f" Error in sleep delay: {e}")
                                time.sleep(0.004)  # Fallback delay
                                
                        except Exception as e:
                            consecutive_errors += 1
                            print(f" Error in CSV data reading (attempt {consecutive_errors}): {e}")
                            
                            if consecutive_errors >= max_consecutive_errors:
                                print(f" Too many consecutive errors ({consecutive_errors}), stopping demo")
                                self._stop_event.set()
                                break
                            
                            # Try to recover by skipping problematic row
                            row_index += 1
                            if row_index >= len(df):
                                row_index = 0
                            
                            # Short delay before retry
                            time.sleep(0.01)
                            
                except Exception as e:
                    print(f" Critical error in read_csv_data: {e}")
                    self._stop_event.set()
            
            # Start CSV data reading in background thread
            self.demo_thread = threading.Thread(target=read_csv_data, name="ECGDemoCSVThread", daemon=True)
            self.demo_thread.start()
            
            # Update effective sampling rate for CSV demo (base 150 Hz)
            self.samples_per_second = 150
            self._set_demo_sampling_rate(self.samples_per_second)

            # Start timer to update plots with real CSV data
            # Timer interval also affected by wave speed
            self.demo_timer = QTimer(self.ecg_test_page)
            self.demo_timer.timeout.connect(self.update_demo_plots)
            
            # Adjust timer interval based on wave speed
            # Use faster timer interval for EXE builds to prevent gaps
            # Timer interval is more important than timer type for smooth plotting
            base_interval = 33  # ~30 FPS for smoother plotting in EXE
            speed_factor = max(0.5, getattr(self, 'time_window', 10.0) / 10.0)
            timer_interval = max(20, int(base_interval * speed_factor))
            # Using default timer type - works fine in EXE with proper interval
            self.demo_timer.start(timer_interval)
            
            print(f" Demo mode started with wave speed: {self.current_wave_speed}mm/s")
            
        except Exception as e:
            print(f" Error reading dummycsv.csv: {e}")
            QMessageBox.warning(self.ecg_test_page, "Error", f"Failed to load dummycsv.csv: {str(e)}")
            # Don't start demo if CSV reading fails
            self.ecg_test_page.demo_toggle.setChecked(False)

    def on_settings_changed(self, key, value):
        """Handle immediate settings changes for instant wave updates"""
        print(f" Demo Manager: on_settings_changed called with key={key}, value={value}")
        if key in ["wave_speed", "wave_gain"]:
            self._debug(f"{key} changed to {value} (running={self._running_demo})")
            # Always update internal settings, even if demo is not running
            self._update_wave_speed_settings()
            
            # If demo is running, apply changes immediately
            if self._running_demo:
                try:
                    self.update_demo_plots()
                except Exception as e:
                    print(f" Error in immediate demo update: {e}")
            else:
                self._debug(f"{key} change deferred until demo starts")
    
    def update_demo_plots(self):
        """Update plots using exact same logic as divyansh.py"""
        if self._plot_running:
            self._skipped_plot_calls = (self._skipped_plot_calls + 1) % 1000
            if self._debug_logging and self._skipped_plot_calls % 60 == 0:
                self._debug(f"Skipping plot update (skipped total={self._skipped_plot_calls})")
            return
        self._plot_running = True
        try:
            self._update_demo_plots_inner()
        finally:
            self._plot_running = False

    def _update_demo_plots_inner(self):
        # Check if this is the first display after stabilization period
        first_display = False
        if hasattr(self, '_initial_display_start'):
            current_time = time.time()
            if current_time >= self._initial_display_start and not hasattr(self, '_first_display_done'):
                first_display = True
                self._first_display_done = True
                # Calculate stable Y-ranges from collected data on first display
                print(f" Starting display after {self._initial_display_delay}s stabilization - calculating stable Y-ranges...")
                self._calculate_initial_stable_ranges()
        
        self._debug(f"update_demo_plots start, speed={self.current_wave_speed}")
        
        # Always get fresh values from settings manager (like divyansh.py does)
        current_speed = self.ecg_test_page.settings_manager.get_wave_speed()
        current_gain = self.ecg_test_page.settings_manager.get_wave_gain()
        
        self._debug(f"settings speed={current_speed}, gain={current_gain}")
        
        # Update internal values if they changed
        if current_speed != self.current_wave_speed:
            self._update_wave_speed_settings()
            # Keep dashboard calculator in sync with effective sampling rate
            self._set_demo_sampling_rate(self.samples_per_second)
        
        # ALWAYS apply display settings to update Y-axis limits for gain changes
        # (This ensures matplotlib plots reflect the current wave_gain setting)
        try:
            if hasattr(self.ecg_test_page, 'apply_display_settings'):
                self.ecg_test_page.apply_display_settings()
                if first_display:
                    print(f" First display: Applied display settings (gain={current_gain}mm/mV)")
        except Exception as e:
            print(f" Could not apply display settings: {e}")
        
        # --- EXACT SAME LOGIC AS DIVYANSH.PY ---
        time_window = getattr(self, 'time_window', 10.0)  # Fallback to 10s if not set
        num_samples_to_show = max(1, int(time_window * self.samples_per_second))
        
        self._debug(f"time_window={time_window}, samples={num_samples_to_show}")
        
        # Get current gain (match real-time serial behaviour: 10mm/mV baseline)
        try:
            wave_gain = float(self.ecg_test_page.settings_manager.get_wave_gain())
            current_gain = wave_gain / 10.0  # 10mm/mV = 1.0x, 20mm/mV = 2.0x, 2.5mm/mV = 0.25x
            self._debug(f"demo gain={current_gain:.2f} (settings {wave_gain}mm/mV)")
        except Exception:
            current_gain = 1.0
            self._debug("demo gain fallback to 1.0x")
        
        # Skip warmup ramp - data is pre-filled, so use full gain immediately
        # (Warmup was removed to enable instant graph display)
        effective_gain = current_gain
        
        self._debug(f"effective gain={effective_gain:.3f}")
        peak_amplitude = None
        
        # 2. For each lead, slice and update (exactly like divyansh.py)
        for i, lead in enumerate(self.ecg_test_page.leads):
            if i < len(self.ecg_test_page.data_lines) and i < len(self.ecg_test_page.data):
                lead_data = self.ecg_test_page.data[i]
                total_len = len(lead_data)
                if total_len == 0:
                    continue
                start = int(self.data_ptr % total_len)
                idx = (start + np.arange(num_samples_to_show)) % total_len
                data_slice = lead_data[idx]
                if data_slice.size == 0:
                    continue

                # Always center data to y-axis for demo mode (centered for all leads)
                centered_slice = np.array(data_slice, dtype=float)
                finite_mask = np.isfinite(centered_slice)
                if not np.any(finite_mask):
                    continue
                # Always apply baseline centering to center waves to y-axis in demo mode
                slice_mean = np.mean(centered_slice[finite_mask])
                centered_slice = centered_slice - slice_mean

                # --- DEMO MODE Y-AXIS FIX: Offset data to center at 2048 or -2048 ---
                if lead == 'aVR':
                    base_offset = -2048
                    self.ecg_test_page.plot_widgets[i].setYRange(-4096, 0)
                else:
                    base_offset = 2048
                    self.ecg_test_page.plot_widgets[i].setYRange(0, 4096)

                # Apply gain only to the wave, not the axis
                display_data = base_offset + (centered_slice * effective_gain)
                display_data = np.nan_to_num(display_data, copy=False)
                
                # Apply stronger smoothing for smooth wave appearance in 12-lead view
                try:
                    from scipy.ndimage import gaussian_filter1d
                    if len(display_data) > 5:
                        # Stronger Gaussian smoothing (sigma=1.0) for very smooth waves in 12-lead view
                        display_data = gaussian_filter1d(display_data, sigma=1.0)
                except (ImportError, Exception):
                    # Fallback: simple moving average if scipy not available
                    if len(display_data) > 5:
                        kernel_size = 5
                        kernel = np.ones(kernel_size) / kernel_size
                        display_data = np.convolve(display_data, kernel, mode='same')

                n = num_samples_to_show
                time_axis = np.arange(n, dtype=float) / float(self.samples_per_second)
                self.ecg_test_page.data_lines[i].setData(time_axis, display_data)

                self.ecg_test_page.plot_widgets[i].setXRange(0, time_window)
        
        # Smaller step for smoother scrolling animation
        # Calculate step based on samples per second and timer interval for smooth movement
        step = 2  # Reduced from 8 to 2 for much smoother scrolling
        if len(self.ecg_test_page.data) > 0:
            any_len = len(self.ecg_test_page.data[0])
            if any_len > 0:
                self.data_ptr = (self.data_ptr + step) % any_len
        
        try:
            gain_mm_per_mv = float(self.ecg_test_page.settings_manager.get_wave_gain())
        except Exception:
            gain_mm_per_mv = effective_gain * 10.0
        if hasattr(self.ecg_test_page, 'update_demo_wave_gain'):
            try:
                self.ecg_test_page.update_demo_wave_gain(effective_gain, gain_mm_per_mv, peak_amplitude)
            except Exception as gain_ui_err:
                print(f" Unable to update demo wave gain display: {gain_ui_err}")
        
        self._debug("update_demo_plots complete")
        
        # Calculate intervals for dashboard in demo mode (throttled to reduce CPU load)
        # Skip during warmup to avoid unstable early metrics
        current_time = time.time()
        if hasattr(self.ecg_test_page, 'dashboard_callback') and self.ecg_test_page.dashboard_callback:
            if (not hasattr(self, '_initial_display_start') or current_time >= self._initial_display_start):
                # Throttle dashboard updates to every 200ms (5x per second instead of 20x)
                if current_time - self._last_dashboard_update >= self._dashboard_update_interval:
                    self._calculate_demo_intervals()
                    self._last_dashboard_update = current_time
    
    def _calculate_initial_stable_ranges(self):
        """Calculate stable Y-axis ranges from collected data after 15-second stabilization"""
        try:
            # Get current gain setting (same as will be used in display)
            wave_gain = float(self.ecg_test_page.settings_manager.get_wave_gain())
            current_gain = wave_gain / 10.0  # 10mm/mV = 1.0x
        except Exception:
            current_gain = 1.0
        
        # Calculate time window and samples to show (same as display logic)
        time_window = getattr(self, 'time_window', 10.0)
        num_samples_to_show = max(1, int(time_window * self.samples_per_second))
        
        for i, lead in enumerate(self.ecg_test_page.leads):
            if i < len(self.ecg_test_page.data):
                lead_data = self.ecg_test_page.data[i]
                if len(lead_data) > 0:
                    # Use current data pointer position for slice
                    start = int(self.data_ptr % len(lead_data))
                    total_len = len(lead_data)
                    idx = (start + np.arange(min(num_samples_to_show, total_len))) % total_len
                    data_slice = lead_data[idx]
                    
                    if data_slice.size > 0:
                        # Apply SAME baseline centering as display
                        finite_mask = np.isfinite(data_slice)
                        if np.any(finite_mask):
                            slice_mean = np.mean(data_slice[finite_mask])
                            centered_slice = data_slice - slice_mean
                            
                            # Apply SAME gain as display
                            display_slice = centered_slice[finite_mask] * current_gain
                            
                            if len(display_slice) > 0:
                                # Calculate stable range from actual display data
                                abs_display = np.abs(display_slice)
                                try:
                                    # Use 97.5th percentile + 25% margin (same as display logic)
                                    baseline_envelope = float(np.percentile(abs_display, 97.5))
                                    base_span = max(200.0, baseline_envelope * 1.25)
                                    # Add extra margin for peaks (same as display logic)
                                    max_peak = float(np.max(abs_display))
                                    if max_peak > base_span:
                                        overshoot = max_peak - base_span
                                        base_span += overshoot * 0.35
                                        if max_peak > base_span:
                                            base_span = max_peak * 1.05
                                    stable_span = max(200.0, base_span)
                                except Exception:
                                    # Fallback: use max * 1.2
                                    stable_span = max(200.0, float(np.max(abs_display)) * 1.2)
                                
                                # Store stable range
                                self._demo_lead_ranges[i] = stable_span
                                
                                # Apply range immediately
                                try:
                                    self.ecg_test_page.plot_widgets[i].setYRange(-stable_span, stable_span)
                                except Exception:
                                    pass
                                continue
                
                # Fallback: default range if calculation failed
                self._demo_lead_ranges[i] = 300.0
                try:
                    self.ecg_test_page.plot_widgets[i].setYRange(-300.0, 300.0)
                except Exception:
                    pass
        
        print(f" Stable Y-ranges calculated and applied for all leads after {self._initial_display_delay}s stabilization")

    def start_synthetic_demo(self):
        """Stream synthetic ECG-like waves when CSV is unavailable."""
        # Ensure any previous demo resources are stopped before starting new
        try:
            self._stop_event.set()
            if self.demo_thread and self.demo_thread.is_alive():
                self.demo_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self._stop_event.clear()
            self.demo_thread = None
            try:
                if self.demo_timer:
                    self.demo_timer.stop()
                    self.demo_timer.deleteLater()
            except Exception:
                pass
            self.demo_timer = None
        # Initialize buffers
        for i in range(len(self.ecg_test_page.data)):
            self.ecg_test_page.data[i] = np.zeros(self.ecg_test_page.buffer_size)

        # Parameters
        try:
            fs = 250.0
            hr_bpm = 72.0
            rr = 60.0 / hr_bpm
            two_pi = 2.0 * np.pi
            gain = 1.0
            # Update speed settings
            self._update_wave_speed_settings()
        except Exception:
            fs = 250.0
            rr = 0.8
            gain = 1.0

        # Background thread to stream samples

        def stream():
            t = 0.0
            dt = 1.0 / fs
            while (not self._stop_event.is_set()) and self._running_demo:
                # Simple synthetic lead II heartbeat shape using summed gaussians
                # Base heartbeat
                phase = (t % rr) / rr
                # Construct a crude P-QRS-T morphology
                p = 0.1 * np.exp(-((phase - 0.2) ** 2) / 0.0008)
                q = -0.25 * np.exp(-((phase - 0.35) ** 2) / 0.0002)
                r = 1.0 * np.exp(-((phase - 0.375) ** 2) / 0.00005)
                s = -0.35 * np.exp(-((phase - 0.40) ** 2) / 0.0001)
                t_w = 0.3 * np.exp(-((phase - 0.6) ** 2) / 0.003)
                sample = (p + q + r + s + t_w) * 1000.0 * gain

                # Add a tiny noise
                sample += np.random.normal(0, 5)

                # Update all leads with simple variations
                for li in range(len(self.ecg_test_page.data)):
                    val = sample * (0.8 + 0.4 * np.sin(two_pi * (li + 1) * 0.03 * t))
                    with self._lock:
                        self.ecg_test_page.data[li] = np.roll(self.ecg_test_page.data[li], -1)
                        self.ecg_test_page.data[li][-1] = val

                # Respect wave speed for visual pacing (like divyansh.py)
                speed_factor = getattr(self, 'time_window', 10.0) / 10.0
                delay = (1.0 / fs) * speed_factor
                time.sleep(delay)
                t += dt

        self.demo_thread = threading.Thread(target=stream, name="ECGDemoSynthThread", daemon=True)
        self.demo_thread.start()

        # Set demo start time for timer consistency
        # Resume from paused time or start fresh
        if self._demo_paused_time is not None:
            # Resume from paused time
            current_time = time.time()
            self._demo_started_at = current_time - self._demo_paused_time
            self._demo_paused_time = None  # Reset paused time
            print(f" Demo (synthetic) resumed from paused time")
        else:
            # Start fresh
            self._demo_started_at = time.time()

        # Timer to draw plots
        self.demo_timer = QTimer(self.ecg_test_page)
        self.demo_timer.timeout.connect(self.update_demo_plots)
        base_interval = 33  # ~30 FPS base
        # Use time window to adjust timer interval (like divyansh.py)
        speed_factor = getattr(self, 'time_window', 10.0) / 10.0
        timer_interval = int(base_interval * speed_factor)
        self.demo_timer.start(max(10, timer_interval))

        # Effective sampling for synthetic: fs (like divyansh.py)
        self.samples_per_second = int(fs)
        self._set_demo_sampling_rate(self.samples_per_second)
        print(" Synthetic demo started (CSV missing)")
    
    def _calculate_demo_intervals(self):
        """Calculate ECG intervals for dashboard display in demo mode"""
        try:
            # Get Lead II data for interval calculations (Lead II is index 1)
            if len(self.ecg_test_page.data) > 1:
                # Copy under lock to avoid race with writer thread
                with self._lock:
                    lead2_data = np.copy(self.ecg_test_page.data[1])  # Lead II
                    lead_I_data = np.copy(self.ecg_test_page.data[0])  # Lead I
                    lead_aVF_data = np.copy(self.ecg_test_page.data[5])  # Lead aVF
                
                if len(lead2_data) > 100:  # Need enough data for calculations
                    # Use adjusted sampling rate based on wave speed
                    sampling_rate = self.samples_per_second
                    
                    # Use recent data for calculations
                    recent_data = np.array(lead2_data[-500:])
                    centered_data = recent_data - np.mean(recent_data)
                    
                    # Check for signal variation
                    signal_std = np.std(centered_data)
                    if signal_std < 1.0:  # Very low variation
                        print(f" Low signal variation detected (std: {signal_std:.2f})")
                        return
                    
                    # Demo-specific peak detection with adjusted parameters
                    min_prominence = max(0.5, signal_std * 0.5)  # Adaptive prominence
                    r_peaks, _ = find_peaks(centered_data, 
                                          distance=int(0.6 * sampling_rate),
                                          prominence=min_prominence)
                    
                    # Calculate intervals only if we have R peaks
                    if len(r_peaks) > 1:
                        # RR intervals and heart rate
                        rr_intervals = np.diff(r_peaks) / sampling_rate
                        mean_rr = np.mean(rr_intervals)
                        heart_rate = 60 / mean_rr if mean_rr > 0 else None
                        
                        # Apply demo heart rate correction
                        if heart_rate and heart_rate > 100:
                            # If demo heart rate is too high, apply correction factor
                            correction_factor = 0.6  # Reduce by 40%
                            heart_rate = heart_rate * correction_factor
                            print(f" Demo heart rate corrected: {heart_rate:.1f} BPM")
                        
                        # Apply demo heart rate smoothing
                        if heart_rate and 30 <= heart_rate <= 200:
                            self.demo_heart_rates.append(heart_rate)
                            if len(self.demo_heart_rates) > 3:
                                self.demo_heart_rates.pop(0)
                            
                            # Use smoothed heart rate for demo
                            heart_rate = np.mean(self.demo_heart_rates)
                            self._debug(f"smoothed heart rate {heart_rate:.1f} BPM")
                        
                        # Calculate P, Q, S, T peaks
                        q_peaks, s_peaks = self._calculate_qs_peaks(centered_data, r_peaks, sampling_rate)
                        p_peaks = self._calculate_p_peaks(centered_data, q_peaks, sampling_rate)
                        t_peaks = self._calculate_t_peaks(centered_data, s_peaks, sampling_rate)
                        
                        # Calculate intervals
                        pr_interval = self._calculate_pr_interval(p_peaks, r_peaks, sampling_rate)
                        qrs_duration = self._calculate_qrs_duration(q_peaks, s_peaks, sampling_rate)
                        qt_interval = self._calculate_qt_interval(q_peaks, t_peaks, sampling_rate)
                        qtc_interval = self._calculate_qtc_interval(qt_interval, heart_rate)
                        
                        # Store both QT and QTc for display
                        qt_value = int(round(qt_interval)) if (qt_interval is not None and qt_interval >= 0) else 380
                        qtc_value = int(round(qtc_interval)) if (qtc_interval is not None and qtc_interval >= 0) else 400
                        
                        # Get calculation functions at runtime to avoid circular import
                        calculate_st_segment = self._get_calculation_functions()
                        
                        # Calculate ST segment using imported functions
                        st_segment = calculate_st_segment(lead2_data, r_peaks, fs=sampling_rate)
                        
                        # Prepare safe ST value (numeric or descriptive text)
                        try:
                            if isinstance(st_segment, (int, float, np.floating)):
                                st_out = int(round(float(st_segment)))
                            elif isinstance(st_segment, str):
                                st_out = st_segment
                            else:
                                st_out = None
                        except Exception:
                            st_out = None

                        # Initialize fixed demo metrics once, then keep constant (hardcoded)
                        if self._demo_fixed_metrics is None:
                            try:
                                fixed_hr = 60
                                fixed_rr = 1000
                                fixed_p = 92
                                fixed_pr = 167
                                fixed_qrs = 86
                                fixed_qt = 357
                                fixed_qtc = 357
                                fixed_axis = "0°"
                                fixed_st = 92
                            except Exception:
                                fixed_hr, fixed_rr, fixed_p, fixed_pr, fixed_qrs, fixed_qt, fixed_qtc, fixed_axis, fixed_st = 60, 1000, 92, 167, 86, 357, 357, "0°", 92
                            self._demo_fixed_metrics = {
                                'Heart_Rate': fixed_hr,
                                'RR': fixed_rr,
                                'P': fixed_p,
                                'PR': fixed_pr,
                                'QRS': fixed_qrs,
                                'QT': fixed_qt,
                                'QTc': fixed_qtc,
                                'QTc_interval': f"{fixed_qt}/{fixed_qtc}",
                                'ST': fixed_st
                            }

                        # Always send fixed metrics in demo mode
                        payload = dict(self._demo_fixed_metrics)

                        try:
                            if self._demo_started_at:
                                elapsed = max(0, int(time.time() - self._demo_started_at))
                                mm = elapsed // 60
                                ss = elapsed % 60
                                payload['time_elapsed'] = f"{mm:02d}:{ss:02d}"
                        except Exception:
                            pass
                        try:
                            self.ecg_test_page.dashboard_callback(payload)
                        except Exception as cb_err:
                            print(f" Error updating dashboard from demo: {cb_err}")
                        
                        # Fixed print statement to handle None values
                        pr_str = f"{self._demo_fixed_metrics['PR']} ms" if self._demo_fixed_metrics else "N/A"
                        qrs_str = f"{self._demo_fixed_metrics['QRS']} ms" if self._demo_fixed_metrics else "N/A"
                        hr_str = f"{self._demo_fixed_metrics['Heart_Rate']} bpm" if self._demo_fixed_metrics else "N/A"
                        
                        self._debug(f"intervals: HR={hr_str}, PR={pr_str}, QRS={qrs_str}")
                        
        except Exception as e:
            print(f" Error calculating demo intervals: {e}")
    
    def _calculate_qs_peaks(self, centered_data, r_peaks, sampling_rate):
        """Calculate Q and S peaks"""
        q_peaks = []
        s_peaks = []
        for r in r_peaks:
            # Q peak before R
            q_start = max(0, r - int(0.06 * sampling_rate))
            q_end = r
            if q_end > q_start:
                q_idx = np.argmin(centered_data[q_start:q_end]) + q_start
                q_peaks.append(q_idx)
            
            # S peak after R
            s_start = r
            s_end = min(len(centered_data), r + int(0.06 * sampling_rate))
            if s_end > s_start:
                s_idx = np.argmin(centered_data[s_start:s_end]) + s_start
                s_peaks.append(s_idx)
        
        return q_peaks, s_peaks
    
    def _calculate_p_peaks(self, centered_data, q_peaks, sampling_rate):
        """Calculate P peaks"""
        p_peaks = []
        for q in q_peaks:
            p_start = max(0, q - int(0.2 * sampling_rate))
            p_end = q - int(0.08 * sampling_rate)
            if p_end > p_start:
                p_candidates, _ = find_peaks(centered_data[p_start:p_end], 
                                           prominence=0.1 * np.std(centered_data))
                if len(p_candidates) > 0:
                    p_peaks.append(p_start + p_candidates[-1])
        return p_peaks
    
    def _calculate_t_peaks(self, centered_data, s_peaks, sampling_rate):
        """Calculate T peaks"""
        t_peaks = []
        for s in s_peaks:
            t_start = s + int(0.08 * sampling_rate)
            t_end = min(len(centered_data), s + int(0.4 * sampling_rate))
            if t_end > t_start:
                t_candidates, _ = find_peaks(centered_data[t_start:t_end], 
                                           prominence=0.1 * np.std(centered_data))
                if len(t_candidates) > 0:
                    t_peaks.append(t_start + t_candidates[np.argmax(centered_data[t_start + t_candidates])])
        return t_peaks
    
    def _calculate_pr_interval(self, p_peaks, r_peaks, sampling_rate):
        """Calculate PR interval"""
        if len(p_peaks) > 0 and len(r_peaks) > 0:
            return (r_peaks[-1] - p_peaks[-1]) * 1000 / sampling_rate
        return None
    
    def _calculate_qrs_duration(self, q_peaks, s_peaks, sampling_rate):
        """Calculate QRS duration"""
        if len(q_peaks) > 0 and len(s_peaks) > 0:
            return (s_peaks[-1] - q_peaks[-1]) * 1000 / sampling_rate
        return None
    
    def _calculate_qt_interval(self, q_peaks, t_peaks, sampling_rate):
        """Calculate QT interval"""
        if len(q_peaks) > 0 and len(t_peaks) > 0:
            return (t_peaks[-1] - q_peaks[-1]) * 1000 / sampling_rate
        return None
    
    def _calculate_qtc_interval(self, qt_interval, heart_rate):
        """Calculate QTc interval using Bazett's formula"""
        if qt_interval and heart_rate:
            return qt_interval / np.sqrt(60 / heart_rate)
        return None

    def _debug(self, message):
        if self._debug_logging:
            safe_print(f"[DEMO DEBUG] {message}")
    
    def stop_demo_data(self):
        """Stop demo data generation"""
        self._running_demo = False
        # Signal thread to stop and join
        try:
            self._stop_event.set()
            if self.demo_thread and self.demo_thread.is_alive():
                self.demo_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self.demo_thread = None

        # Stop and delete timer
        try:
            if self.demo_timer:
                self.demo_timer.stop()
                self.demo_timer.deleteLater()
                print(" Demo timer stopped")
        except Exception:
            pass
        finally:
            self.demo_timer = None

        # Clear demo data and plots safely
        try:
            target_size = getattr(self.ecg_test_page, 'max_buffer_size', 10000)
            with self._lock:
                for i in range(len(self.ecg_test_page.data)):
                    self.ecg_test_page.data[i] = np.zeros(target_size)
                for line in self.ecg_test_page.data_lines:
                    line.setData(np.zeros(target_size))
            print(" Demo data cleared and plots reset")
        except Exception:
            pass

        # Clear heart rate smoothing data
        self.demo_heart_rates.clear()
        print(" Demo mode stopped successfully")

    def _on_page_destroyed(self):
        """Safely stop background activity when the owning page is destroyed."""
        try:
            self._running_demo = False
            self._stop_event.set()
            if self.demo_thread and self.demo_thread.is_alive():
                self.demo_thread.join(timeout=0.5)
            # Also stop timer to prevent memory leaks
            if self.demo_timer:
                self.demo_timer.stop()
                self.demo_timer.deleteLater()
                self.demo_timer = None
        except Exception:
            pass
