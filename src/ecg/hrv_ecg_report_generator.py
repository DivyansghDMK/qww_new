from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak,
    PageTemplate, Frame, NextPageTemplate, BaseDocTemplate
)
from reportlab.graphics.shapes import Drawing, Line, Rect, Path, String
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
import os
import sys
import json
import matplotlib.pyplot as plt  
import matplotlib
import numpy as np

# Set matplotlib to use non-interactive backend
matplotlib.use('Agg')

# ------------------------ Resource path helper for PyInstaller compatibility ------------------------

def _get_resource_path(relative_path):
    """
    Get resource path that works both in development and when packaged as exe.
    For PyInstaller: resources are in sys._MEIPASS
    For development: resources are relative to project root
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            # Development mode - get path relative to this file
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(base_path, relative_path)
    except Exception:
        # Fallback to relative path
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", relative_path)

# ==================== ECG DATA SAVE/LOAD FUNCTIONS ====================

def save_ecg_data_to_file(ecg_test_page, output_file=None):
    """
    Save ECG data from ecg_test_page.data to a JSON file
    Returns: path to saved file or None if failed
    
    Example:
        saved_file = save_ecg_data_to_file(ecg_test_page)
        # Saved to: reports/ecg_data/ecg_data_20241119_143022.json
    """
    from datetime import datetime
    
    if not ecg_test_page or not hasattr(ecg_test_page, 'data'):
        print(" No ECG test page data available to save")
        return None
    
    # Create output directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ecg_data_dir = os.path.join(base_dir, 'reports', 'ecg_data')
    os.makedirs(ecg_data_dir, exist_ok=True)
    
    # Generate filename with timestamp
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(ecg_data_dir, f'ecg_data_{timestamp}.json')
    
    # Prepare data for saving
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    saved_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "sampling_rate": 500.0,
        "leads": {}
    }
    
    # Get sampling rate if available
    if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            try:
                sampled_rate = float(ecg_test_page.sampler.sampling_rate)
                if sampled_rate < 50.0 or sampled_rate > 1000.0:
                    sampled_rate = 500.0
                saved_data["sampling_rate"] = sampled_rate
            except Exception:
                saved_data["sampling_rate"] = 500.0
    
    # Save each lead's data - use FULL buffer (ecg_buffers if available, otherwise data)
    # Priority: Use ecg_buffers (5000 samples) if available, otherwise use data (1000 samples)
    
    # Debug: Check what attributes ecg_test_page has
    print(f"🔍 DEBUG: ecg_test_page attributes check:")
    print(f"   has ecg_buffers: {hasattr(ecg_test_page, 'ecg_buffers')}")
    print(f"   has data: {hasattr(ecg_test_page, 'data')}")
    print(f"   has ptrs: {hasattr(ecg_test_page, 'ptrs')}")
    if hasattr(ecg_test_page, 'ecg_buffers'):
        print(f"   ecg_buffers length: {len(ecg_test_page.ecg_buffers) if ecg_test_page.ecg_buffers else 0}")
    if hasattr(ecg_test_page, 'data'):
        print(f"   data length: {len(ecg_test_page.data) if ecg_test_page.data else 0}")
        if ecg_test_page.data and len(ecg_test_page.data) > 0:
            print(f"   data[0] length: {len(ecg_test_page.data[0]) if isinstance(ecg_test_page.data[0], (list, np.ndarray)) else 'N/A'}")
    
    for i, lead_name in enumerate(lead_names):
        data_to_save = []
        
        # Priority 1: Try to use ecg_buffers (larger buffer, 5000 samples)
        if hasattr(ecg_test_page, 'ecg_buffers') and i < len(ecg_test_page.ecg_buffers):
            buffer = ecg_test_page.ecg_buffers[i]
            if isinstance(buffer, np.ndarray) and len(buffer) > 0:
                # Check if this is a rolling buffer with ptrs
                if hasattr(ecg_test_page, 'ptrs') and i < len(ecg_test_page.ptrs):
                    ptr = ecg_test_page.ptrs[i]
                    window_size = getattr(ecg_test_page, 'window_size', 1000)
                    
                    # For report generation: use FULL buffer (5000 samples), not just window_size (1000)
                    # Get all available data from buffer, starting from ptr
                    if ptr + len(buffer) <= len(buffer):
                        # No wrap needed: get from ptr to end, then from start to ptr
                        part1 = buffer[ptr:].tolist()
                        part2 = buffer[:ptr].tolist()
                        data_to_save = part1 + part2  # Full circular buffer
                    else:
                        # Simple case: use all buffer data
                        data_to_save = buffer.tolist()
                else:
                    # No ptrs: use ALL available data (full buffer)
                    data_to_save = buffer.tolist()
        
        # Priority 2: Fallback to ecg_test_page.data (smaller buffer, 1000 samples)
        if not data_to_save and i < len(ecg_test_page.data):
            lead_data = ecg_test_page.data[i]
            if isinstance(lead_data, np.ndarray):
                # Use ALL available data (not just window_size)
                data_to_save = lead_data.tolist()
            elif isinstance(lead_data, (list, tuple)):
                data_to_save = list(lead_data)
        
        saved_data["leads"][lead_name] = data_to_save if data_to_save else []
    
    # Check if we have sufficient data for report generation
    sample_counts = [len(saved_data["leads"][lead]) for lead in saved_data["leads"] if saved_data["leads"][lead]]
    if sample_counts:
        max_samples = max(sample_counts)
        min_samples = min(sample_counts)
        print(f"📊 Buffer analysis: Max samples={max_samples}, Min samples={min_samples}")
        
        # Calculate expected samples for 13.2s window at current sampling rate
        sampling_rate = saved_data.get("sampling_rate", 500.0)
        expected_samples_for_13_2s = int(13.2 * sampling_rate)
        
        if max_samples < expected_samples_for_13_2s:
            print(f" WARNING: Buffer has only {max_samples} samples, need {expected_samples_for_13_2s} for 13.2s window")
            print(f"   Current time window: {max_samples/sampling_rate:.2f}s")
            print(f"   Expected time window: 13.2s")
            print(f"    TIP: Run ECG for at least 15-20 seconds to accumulate sufficient data")
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(saved_data, f, indent=2)
        print(f"Saved ECG data to: {output_file}")
        print(f"   Leads saved: {list(saved_data['leads'].keys())}")
        print(f"   Sampling rate: {saved_data['sampling_rate']} Hz")
        print(f"   Total data points per lead: {[len(saved_data['leads'][lead]) for lead in saved_data['leads']]}")
        return output_file
    except Exception as e:
        print(f" Error saving ECG data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_ecg_data_from_file(file_path):
    """
    Load ECG data from JSON file
    Returns: dict with 'leads', 'sampling_rate', 'timestamp' or None if failed
    
    Example:
        data = load_ecg_data_from_file('reports/ecg_data/ecg_data_20241119_143022.json')
            # Returns: {'leads': {'I': [...], 'II': [...]}, 'sampling_rate': 500.0, ...}
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        if 'leads' in data:
            for lead_name in data['leads']:
                if isinstance(data['leads'][lead_name], list):
                    data['leads'][lead_name] = np.array(data['leads'][lead_name])
        
        print(f" Loaded ECG data from: {file_path}")
        print(f"   Leads loaded: {list(data.get('leads', {}).keys())}")
        print(f"   Sampling rate: {data.get('sampling_rate', 500.0)} Hz")
        return data
    except Exception as e:
        print(f" Error loading ECG data: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_time_window_from_bpm_and_wave_speed(hr_bpm, wave_speed_mm_s, desired_beats=6):
    """
    Calculate optimal time window based on BPM and wave_speed
    
    Important: Report  ECG graph  width = 33 boxes × 5mm = 165mm 
     wave_speed  time calculate    factor use :
        Time from wave_speed = (165mm / wave_speed_mm_s) seconds
    
    Formula:
        - Time window = (165mm / wave_speed_mm_s) seconds ONLY
          ( 33 boxes × 5mm = 165mm total width)
        - BPM window is NOT used - only wave speed window
        - Beats = (BPM / 60) × time_window
        - Final window clamped maximum 20 seconds (NO minimum clamp)
    
    
    Returns: (time_window_seconds, num_samples)
    """
    # Calculate time window from wave_speed ONLY (BPM window NOT used)
    # Report  ECG graph width = 33 boxes × 5mm = 165mm
    # Time = Distance / Speed
    ecg_graph_width_mm = 33 * 5  # 33 boxes × 5mm = 165mm
    calculated_time_window = ecg_graph_width_mm / max(1e-6, wave_speed_mm_s)
    
    # Only clamp maximum to 20 seconds (NO minimum clamp)
    calculated_time_window = min(calculated_time_window, 20.0)
    
    # Calculate number of samples (assuming 500 Hz default)
    num_samples = int(calculated_time_window * 500.0)
    
    # Calculate expected beats: beats = (BPM / 60) × time_window
    # Formula: beats per second = BPM / 60, then multiply by time window
    beats_per_second = hr_bpm / 60.0 if hr_bpm > 0 else 0
    expected_beats = beats_per_second * calculated_time_window
    
    print(f" Time Window Calculation (Wave Speed ONLY):")
    print(f"   Graph Width: 165mm (33 boxes × 5mm)")
    print(f"   Wave Speed: {wave_speed_mm_s}mm/s")
    print(f"   Time Window: 165 / {wave_speed_mm_s} = {calculated_time_window:.2f}s")
    print(f"   BPM: {hr_bpm} → Beats per second: {hr_bpm}/60 = {beats_per_second:.2f} beats/sec")
    print(f"   Expected Beats: {beats_per_second:.2f} × {calculated_time_window:.2f} = {expected_beats:.1f} beats")
    print(f"   Estimated Samples: {num_samples} (at 500Hz)")
    
    return calculated_time_window, num_samples

def create_ecg_grid_with_waveform(ecg_data, lead_name, width=6, height=2):
    """
    Create ECG graph with pink grid background and dark ECG waveform
    Returns: matplotlib figure with pink ECG grid background
    """
    # Create figure with pink background
    fig, ax = plt.subplots(figsize=(width, height), facecolor='#ffe6e6', frameon=True)
    
    # STEP 1: Create pink ECG grid background
    # ECG grid colors (even lighter pink/red like medical ECG paper)
    light_grid_color = '#ffd1d1'  # Darker minor grid
    major_grid_color = '#ffb3b3'  # Darker major grid
    bg_color = '#ffe6e6'  # Very light pink background
    
    # Set both figure and axes background to pink
    fig.patch.set_facecolor(bg_color)  # Figure background pink
    ax.set_facecolor(bg_color)         # Axes background pink
    
    # STEP 2: Draw pink ECG grid lines
    # Minor grid lines (1mm equivalent spacing) - LIGHT PINK
    minor_spacing_x = width / 60  # 60 minor divisions across width
    minor_spacing_y = height / 20  # 20 minor divisions across height
    
    # Draw vertical minor pink grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        ax.axvline(x=x_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Draw horizontal minor pink grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        ax.axhline(y=y_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Major grid lines (5mm equivalent spacing) - DARKER PINK
    major_spacing_x = width / 12  # 12 major divisions across width
    major_spacing_y = height / 4   # 4 major divisions across height
    
    # Draw vertical major pink grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        ax.axvline(x=x_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # Draw horizontal major pink grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        ax.axhline(y=y_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # STEP 3: Plot DARK ECG waveform on top of pink grid
    if ecg_data is not None and len(ecg_data) > 0:
        # Scale ECG data to fit in the grid
        t = np.linspace(0, width, len(ecg_data))
        # Normalize ECG data to fit in height with some margin
        if np.max(ecg_data) != np.min(ecg_data):
            ecg_normalized = ((ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))) * (height * 0.8) + (height * 0.1)
        else:
            ecg_normalized = np.full_like(ecg_data, height / 2)
        
        # DARK ECG LINE - clearly visible on pink grid
        ax.plot(t, ecg_normalized, color='#000000', linewidth=2.8, solid_capstyle='round', alpha=0.9)
    # REMOVE ENTIRE else BLOCK - just comment it out or delete lines 78-96
    
    # STEP 4: Set axis limits to match grid
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # STEP 5: Remove axis elements but keep the pink grid background
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    return fig

from reportlab.graphics.shapes import Drawing, Group, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.units import mm

def create_reportlab_ecg_drawing(lead_name, width=460, height=45):
    """
    Create ECG drawing using ReportLab (NO matplotlib - NO white background issues)
    Returns: ReportLab Drawing with guaranteed pink background
    """
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines (even lighter colors)
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid
    
    # Minor grid lines (1mm spacing equivalent)
    minor_spacing_x = width / 60  # 60 divisions across width
    minor_spacing_y = height / 20  # 20 divisions across height
    
    # Vertical minor grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Horizontal minor grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Major grid lines (5mm spacing equivalent)
    major_spacing_x = width / 12  # 12 divisions across width
    major_spacing_y = height / 4   # 4 divisions across height
    
    # Vertical major grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # Horizontal major grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # REMOVE ENTIRE "STEP 3: Draw ECG waveform as series of lines" section (lines ~166-214)
    
    return drawing

def capture_real_ecg_graphs_from_dashboard(dashboard_instance=None, ecg_test_page=None, samples_per_second=150, settings_manager=None):
    """
    Capture REAL ECG data from the live test page and create drawings
    Returns: dict with ReportLab Drawing objects containing REAL ECG data
    """
    lead_drawings = {}
    
    print(" Capturing REAL ECG data from live test page...")
    
    if settings_manager is None:
        from utils.settings_manager import SettingsManager
        settings_manager = SettingsManager()

    lead_sequence = settings_manager.get_setting("lead_sequence", "Standard")
    
    # Define lead orders based on sequence 
    LEAD_SEQUENCES = {
        "Standard": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "Cabrera": ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    
    # Use the appropriate sequence for REPORT ONLY
    ordered_leads = LEAD_SEQUENCES.get(lead_sequence, LEAD_SEQUENCES["Standard"])
    
    # Map lead names to indices
    lead_to_index = {
        "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
        "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
    }
    
    # Check if demo mode is active and get time window for filtering
    is_demo_mode = False
    time_window_seconds = None
    samples_per_second = samples_per_second or 150
    
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo_mode = ecg_test_page.demo_toggle.isChecked()
        if is_demo_mode:
            # Get time window from demo manager
            if hasattr(ecg_test_page, 'demo_manager') and ecg_test_page.demo_manager:
                time_window_seconds = getattr(ecg_test_page.demo_manager, 'time_window', None)
                samples_per_second = getattr(ecg_test_page.demo_manager, 'samples_per_second', 150)
                print(f"🔍 DEMO MODE ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    # NEW LOGIC: Time window = 165mm / wave_speed (33 boxes × 5mm = 165mm)
                    ecg_graph_width_mm = 33 * 5  # 165mm
                    time_window_seconds = ecg_graph_width_mm / wave_speed
                    print(f"🔍 DEMO MODE ON - Calculated window using NEW LOGIC: 165mm / {wave_speed}mm/s = {time_window_seconds}s")
                except Exception as e:
                    print(f"⚠️ Could not get demo time window: {e}")
                    time_window_seconds = None
    # Try to get REAL ECG data from the test page
    real_ecg_data = {}
    if ecg_test_page and hasattr(ecg_test_page, 'data'):
        
        # Calculate number of samples to capture based on demo mode
        if is_demo_mode and time_window_seconds is not None:
            # In demo mode: only capture data visible in one window frame
            num_samples_to_capture = int(time_window_seconds * samples_per_second)
            print(f" DEMO MODE: Capturing only {num_samples_to_capture} samples ({time_window_seconds}s window)")
        else:
            # Normal mode: capture maximum data (10 seconds or 10000 points, whichever is smaller)
            num_samples_to_capture = 10000
            print(f" NORMAL MODE: Capturing up to {num_samples_to_capture} samples")
        
        for lead in ordered_leads:
            if lead == "-aVR":
                # For -aVR, we need to invert aVR data
                if hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 3:
                    avr_data = np.array(ecg_test_page.data[3])  # aVR is at index 3
                    if is_demo_mode and time_window_seconds is not None:
                        # Demo mode: only capture window frame data
                        real_ecg_data[lead] = -avr_data[-num_samples_to_capture:]
                        print(f" Captured DEMO -aVR data: {len(real_ecg_data[lead])} points ({time_window_seconds}s window)")
                    else:
                        # Normal mode: capture maximum data
                        real_ecg_data[lead] = -avr_data[-num_samples_to_capture:]
                        print(f" Captured REAL -aVR data: {len(real_ecg_data[lead])} points")
            else:
                lead_index = lead_to_index.get(lead)
                if lead_index is not None and len(ecg_test_page.data) > lead_index:
                    lead_data = np.array(ecg_test_page.data[lead_index])
                    if len(lead_data) > 0:
                        if is_demo_mode and time_window_seconds is not None:
                            # Demo mode: only capture window frame data
                            real_ecg_data[lead] = lead_data[-num_samples_to_capture:]
                            print(f" Captured DEMO {lead} data: {len(real_ecg_data[lead])} points ({time_window_seconds}s window)")
                        else:
                            # Normal mode: capture maximum data
                            real_ecg_data[lead] = lead_data[-num_samples_to_capture:]
                            print(f" Captured REAL {lead} data: {len(real_ecg_data[lead])} points")
                    else:
                        print(f" No data found for {lead}")
                else:
                    print(f" Lead {lead} index not found")
    else:
        print(" No live ECG test page found - using grid only")
    
    # Get wave_gain from settings_manager for amplitude scaling
    wave_gain_mm_mv = 10.0  # Default
    if settings_manager:
        try:
            wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
            wave_gain_mm_mv = float(wave_gain_setting) if wave_gain_setting else 10.0
            print(f" Using wave_gain from ecg_settings.json: {wave_gain_mm_mv} mm/mV (for amplitude scaling)")
        except Exception:
            wave_gain_mm_mv = 10.0
            print(f" Could not get wave_gain from settings, using default: {wave_gain_mm_mv} mm/mV")
    
    # Apply report filters (AC/EMG/DFT) based on current settings
    filtered_ecg_data = real_ecg_data
    try:
        from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
        dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
        emg_setting = str(settings_manager.get_setting("filter_emg", "off")).strip()
        ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
        filtered_ecg_data = {}
        for lead, signal in real_ecg_data.items():
            if signal is None or len(signal) == 0:
                filtered_ecg_data[lead] = signal
                continue
            filtered = signal
            if dft_setting not in ("off", ""):
                filtered = apply_dft_filter(filtered, float(samples_per_second), dft_setting)
            if emg_setting not in ("off", ""):
                filtered = apply_emg_filter(filtered, float(samples_per_second), emg_setting)
            if ac_setting in ("50", "60"):
                filtered = apply_ac_filter(filtered, float(samples_per_second), ac_setting)
            filtered_ecg_data[lead] = filtered
        print(f" Applied report filters: DFT={dft_setting}, EMG={emg_setting}, AC={ac_setting}")
    except Exception as e:
        print(f" Could not apply report filters: {e}")
        filtered_ecg_data = real_ecg_data

    # Create ReportLab drawings with REAL (filtered) data
    for lead in ordered_leads:
        try:
            # Create ReportLab drawing with REAL ECG data (with wave_gain applied)
            drawing = create_reportlab_ecg_drawing_with_real_data(
                lead, 
                filtered_ecg_data.get(lead), 
                width=460, 
                height=45,
                wave_gain_mm_mv=wave_gain_mm_mv
            )
            lead_drawings[lead] = drawing
            
            if lead in filtered_ecg_data:
                print(f" Created drawing with MAXIMUM data for Lead {lead} - showing 7+ heartbeats")
            else:
                print(f"Created grid-only drawing for Lead {lead}")
            
        except Exception as e:
            print(f" Error creating drawing for Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    if is_demo_mode and time_window_seconds is not None:
        print(f" Successfully created {len(lead_drawings)}/12 ECG drawings with DEMO window filtering ({time_window_seconds}s window - visible peaks only)!")
    else:
        print(f" Successfully created {len(lead_drawings)}/12 ECG drawings with MAXIMUM heartbeats!")
    return lead_drawings

def create_reportlab_ecg_drawing_with_real_data(lead_name, ecg_data, width=460, height=45, wave_gain_mm_mv=10.0):
    """
    Create ECG drawing using ReportLab with REAL ECG data showing MAXIMUM heartbeats
    Returns: ReportLab Drawing with guaranteed pink background and REAL ECG waveform
    
    Parameters:
        wave_gain_mm_mv: Wave gain in mm/mV (default: 10.0 mm/mV)
                         Used for amplitude scaling: 10mm/mV = 1.0x, 20mm/mV = 2.0x, 5mm/mV = 0.5x
    """
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines (even lighter colors)
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid
    
    # Minor grid lines (1mm spacing equivalent)
    minor_spacing_x = width / 60  # 60 divisions across width
    minor_spacing_y = height / 20  # 20 divisions across height
    
    # Vertical minor grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Horizontal minor grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Major grid lines (5mm spacing equivalent)
    major_spacing_x = width / 12  # 12 divisions across width
    major_spacing_y = height / 4   # 4 divisions across height
    
    # Vertical major grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # Horizontal major grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # STEP 3: Draw ALL AVAILABLE ECG data - NO DOWNSAMPLING, NO LIMITS!
    if ecg_data is not None and len(ecg_data) > 0:
        print(f" Drawing ALL AVAILABLE ECG data for {lead_name}: {len(ecg_data)} points (NO LIMITS)")
        
        # SIMPLE APPROACH: Use ALL available data points - NO cutting, NO downsampling
        # This will show as many heartbeats as possible in the available data
        
        # Create time array for ALL the data
        t = np.linspace(0, width, len(ecg_data))
        
        
        # Use uniform ADC per box multiplier (HRV uses lead-specific mapping)
        adc_per_box_multiplier = 6400.0
        
        # Convert to numpy array
        adc_data = np.array(ecg_data, dtype=float)
        
        # Apply baseline 2000 (subtract baseline from ADC values)
        baseline_adc = 2000.0
        centered_adc = adc_data - baseline_adc
        
        # Calculate ADC per box based on wave_gain and lead-specific multiplier
        adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
        
        # Convert ADC offset to boxes (vertical units)
        # Direct calculation: boxes_offset = centered_adc / adc_per_box
        boxes_offset = centered_adc / adc_per_box
        
        # Convert boxes to Y position
        center_y = height / 2.0  # Center of the graph in points
        box_height_points = 5.0  # 1 box = 5mm = 5 points
        
        # Convert boxes offset to Y position
        ecg_normalized = center_y + (boxes_offset * box_height_points)
        
        # Draw ALL ECG data points - NO REDUCTION
        ecg_color = colors.HexColor("#000000")  # Black ECG line
        
        # OPTIMIZED: Draw every point for maximum detail
        for i in range(len(t) - 1):
            line = Line(t[i], ecg_normalized[i], 
                       t[i+1], ecg_normalized[i+1], 
                       strokeColor=ecg_color, strokeWidth=0.5)
            drawing.add(line)
        
        print(f" Drew ALL {len(ecg_data)} ECG data points for {lead_name} - showing MAXIMUM heartbeats!")
    else:
        print(f" No real data available for {lead_name} - showing grid only")
    
    return drawing

def create_clean_ecg_image(lead_name, width=6, height=2):
    """
    Create COMPLETELY CLEAN ECG image with GUARANTEED pink background
    NO labels, NO time markers, NO axes, NO white background
    """
    # FORCE matplotlib to use proper backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # STEP 1: Create figure with FORCED pink background
    fig = plt.figure(figsize=(width, height), facecolor='#ffe6e6', frameon=True)
    
    # FORCE figure background to pink
    fig.patch.set_facecolor('#ffe6e6')
    fig.patch.set_alpha(1.0)  # Full opacity
    
    # Create axes with FORCED pink background
    ax = fig.add_subplot(111)
    ax.set_facecolor('#ffe6e6')  # FORCE axes background pink
    ax.patch.set_facecolor('#ffe6e6')  # FORCE axes patch pink
    ax.patch.set_alpha(1.0)  # Full opacity
    
    # STEP 2: Draw pink ECG grid lines OVER pink background (darker for clarity)
    light_grid_color = '#ffd1d1'  # Darker minor grid
    major_grid_color = '#ffb3b3'  # Darker major grid
    
    # Minor grid lines (1mm equivalent spacing)
    minor_spacing_x = width / 60  # 60 minor divisions
    minor_spacing_y = height / 20  # 20 minor divisions
    
    # Draw vertical minor pink grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        ax.axvline(x=x_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Draw horizontal minor pink grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        ax.axhline(y=y_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Major grid lines (5mm equivalent spacing)
    major_spacing_x = width / 12  # 12 major divisions
    major_spacing_y = height / 4   # 4 major divisions
    
    # Draw vertical major pink grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        ax.axvline(x=x_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # Draw horizontal major pink grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        ax.axhline(y=y_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # REMOVE ENTIRE "STEP 3: Create realistic ECG waveform" section (lines ~315-356)
    # REMOVE ENTIRE "STEP 4: Plot DARK ECG line" section
    
    # STEP 5: Set limits and remove ALL visual elements except grid
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # COMPLETELY remove ALL spines, ticks, labels
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.axis('off')  # FORCE turn off all axis elements
    
    # Remove any text objects
    for text in ax.texts:
        text.set_visible(False)
    
    # FORCE tight layout with pink background
    fig.tight_layout(pad=0)
    
    return fig


def get_dashboard_conclusions_from_image(dashboard_instance):
    """
    Load dynamic conclusions from JSON file (saved by dashboard)
    Returns: List of clean conclusion headings (up to 12 conclusions)
    """
    conclusions = []
    
    # **NEW: Try to load from JSON file first (DYNAMIC)**
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        conclusions_file = os.path.join(base_dir, 'last_conclusions.json')
        
        print(f" Looking for conclusions at: {conclusions_file}")
        
        if os.path.exists(conclusions_file):
            with open(conclusions_file, 'r') as f:
                conclusions_data = json.load(f)
            
            print(f" Loaded JSON data: {conclusions_data}")
            
            # Extract findings from JSON
            findings = conclusions_data.get('findings', [])
            
            if findings:
                conclusions = findings[:12]  # Take up to 12 conclusions
                print(f" Loaded {len(conclusions)} DYNAMIC conclusions from JSON file")
                for i, conclusion in enumerate(conclusions, 1):
                    print(f"   {i}. {conclusion}")
            else:
                print(" No findings in JSON file")
        else:
            print(f" Conclusions JSON file not found: {conclusions_file}")
    
    except Exception as json_err:
        print(f" Error loading conclusions from JSON: {json_err}")
        import traceback
        traceback.print_exc()
    
    # **REMOVED: Old code that extracted from dashboard_instance.conclusion_box**
    # **REMOVED: Fallback default conclusions**
    
    # If still no conclusions found, use minimal fallback
    if not conclusions:
        conclusions = [
            "No ECG data available",
            "Please connect device or enable demo ",
           
            
        ]
        print("⚠️ Using zero-value fallback (no ECG data available)")
    
    # Ensure we have exactly 12 conclusions (pad with empty strings if needed)
    MAX_CONCLUSIONS = 12
    while len(conclusions) < MAX_CONCLUSIONS:
        conclusions.append("---")  # Use "---" for empty slots
    
    # Limit to maximum 12 conclusions
    conclusions = conclusions[:MAX_CONCLUSIONS]
    
    print(f" Final conclusions list (12 total): {len([c for c in conclusions if c and c != '---'])} filled, {len([c for c in conclusions if not c or c == '---'])} blank")
    
    return conclusions


def load_latest_metrics_entry(reports_dir):
    """
    Return the most recent metrics entry from reports/metrics.json, if available.
    """
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and data:
            return data[-1]

        if isinstance(data, dict):
            # support older shape where 'entries' may list the items
            entries = data.get('entries')
            if isinstance(entries, list) and entries:
                return entries[-1]

            # if dict already looks like one entry, return it
            if data.get('timestamp'):
                return data
    except Exception as e:
        print(f" Could not read metrics file for HR: {e}")

    return None

def _add_label_column(drawing, x_pos, y_text_pairs, font_size=10, font_name="Helvetica", text_color=colors.black):
    """
    Add multiple text labels to a drawing at a fixed X position.
    """
    from reportlab.graphics.shapes import String
    for y_pos, text in y_text_pairs:
        drawing.add(String(x_pos, y_pos, text,
                           fontSize=font_size, fontName=font_name, fillColor=text_color))

def generate_ecg_report(filename="ecg_report.pdf", data=None, lead_images=None, dashboard_instance=None, ecg_test_page=None, patient=None, ecg_data_file=None):
    """
    Generate ECG report PDF
    
    Parameters:
        ecg_data_file: Optional path to saved ECG data file. 
                       If provided, will load from file instead of live ecg_test_page.
                       If None and ecg_test_page provided, will save data first.
    
    Example:
        # Option 1: Save data first, then generate report
        saved_file = save_ecg_data_to_file(ecg_test_page)
        generate_ecg_report("report.pdf", data=metrics, ecg_test_page=ecg_test_page, ecg_data_file=saved_file)
        
        # Option 2: Generate report and auto-save data
        generate_ecg_report("report.pdf", data=metrics, ecg_test_page=ecg_test_page)
        # Data will be automatically saved before report generation
    """
   
   
    # Main function body starts here
    if data is None:
        # When no device connected or demo off - show ZERO values (not dummy values)
        data = {
            "HR": 0,
            "beat": 0,
            "PR": 0,
            "QRS": 0,
            "QT": 0,
            "QTc": 0,
            "ST": 0,
            "HR_max": 0,
            "HR_min": 0,
            "HR_avg": 0,
            "Heart_Rate": 0,  # Add for compatibility with dashboard
            "QRS_axis": "--",
        }

    # Define base_dir and reports_dir for file operations
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    from utils.settings_manager import SettingsManager
    settings_manager = SettingsManager()

    def _safe_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    # ==================== STEP 1: Get HR_bpm from metrics.json (PRIORITY) ====================
    # Priority: metrics.json  latest HR_bpm  (calculation-based beats  )
    latest_metrics = load_latest_metrics_entry(reports_dir)
    hr_bpm_value = 0
    
    # Priority 1: metrics.json  latest HR_bpm (CALCULATION-BASED BEATS   REQUIRED)
    if latest_metrics:
        hr_bpm_value = _safe_int(latest_metrics.get("HR_bpm"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from metrics.json: {hr_bpm_value} bpm (for calculation-based beats)")
    
    # Priority 2: Fallback to data parameter
    if hr_bpm_value == 0:
        hr_candidate = data.get("HR_bpm") or data.get("Heart_Rate") or data.get("HR")
        hr_bpm_value = _safe_int(hr_candidate)
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from data parameter: {hr_bpm_value} bpm")
    
    # Priority 3: Fallback to HR_avg
    if hr_bpm_value == 0 and data.get("HR_avg"):
        hr_bpm_value = _safe_int(data.get("HR_avg"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from HR_avg: {hr_bpm_value} bpm")

    data["HR_bpm"] = hr_bpm_value
    data["Heart_Rate"] = hr_bpm_value
    data["HR"] = hr_bpm_value
    if hr_bpm_value > 0:
        data["RR_ms"] = int(60000 / hr_bpm_value)
    else:
        data["RR_ms"] = data.get("RR_ms", 0)

    # ==================== STEP 2: Get wave_speed from ecg_settings.json (PRIORITY) ====================
    # Priority: ecg_settings.json  wave_speed  (calculation-based beats  )
    wave_speed_setting = settings_manager.get_setting("wave_speed", "25")
    wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
    wave_speed_mm_s = _safe_float(wave_speed_setting, 25.0)  # Default: 25.0 mm/s
    wave_gain_mm_mv = _safe_float(wave_gain_setting, 10.0)   # Default: 10.0 mm/mV
    print(f" Using wave_speed from ecg_settings.json: {wave_speed_mm_s} mm/s (for calculation-based beats)")
    computed_sampling_rate = 500

    data["wave_speed_mm_s"] = wave_speed_mm_s
    data["wave_gain_mm_mv"] = wave_gain_mm_mv

    print(f"🧮 Pre-plot checks: HR_bpm={hr_bpm_value}, RR_ms={data['RR_ms']}, wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV, sampling_rate={computed_sampling_rate}Hz")
    print(f"📐 Calculation-based beats formula:")
    print(f"   Graph width: 33 boxes × 5mm = 165mm")
    print(f"   BPM window: (desired_beats × 60) / {hr_bpm_value} = {(6 * 60.0 / hr_bpm_value) if hr_bpm_value > 0 else 0:.2f}s")
    print(f"   Wave speed window: 165mm / {wave_speed_mm_s}mm/s = {165.0 / wave_speed_mm_s:.2f}s")
    
    # ==================== STEP 3: SAVE ECG DATA TO FILE (ALWAYS) ====================
    # IMPORTANT:  data file  save ,    load  (calculation-based beats  )
    saved_ecg_data = None
    saved_data_file_path = None
    
    if ecg_data_file and os.path.exists(ecg_data_file):
        # Use provided file
        print(f" Using provided ECG data file: {ecg_data_file}")

        
        saved_data_file_path = ecg_data_file
        saved_ecg_data = load_ecg_data_from_file(ecg_data_file)
        if saved_ecg_data:
            # Override sampling rate from saved data
            computed_sampling_rate = 500
            print(f" Using sampling rate from provided file: {computed_sampling_rate} Hz (forced)")
    elif ecg_test_page and hasattr(ecg_test_page, 'data'):
        # ALWAYS save current data to file before generating report (REQUIRED for calculation-based beats)
        print(" Saving ECG data to file (required for calculation-based beats)...")
        saved_data_file_path = save_ecg_data_to_file(ecg_test_page)
        if saved_data_file_path:
            saved_ecg_data = load_ecg_data_from_file(saved_data_file_path)
            if saved_ecg_data:
                computed_sampling_rate = 500
                print(f" Using sampling rate from saved file: {computed_sampling_rate} Hz (forced)")
            else:
                print(" Warning: Could not load saved ECG data file")
        else:
            print(" Warning: Could not save ECG data to file")
    
    if not saved_ecg_data:
        print(" Warning: No saved ECG data available - beats will not be calculation-based")

    # Get conclusions from dashboard/JSON
    dashboard_conclusions = get_dashboard_conclusions_from_image(dashboard_instance)

    # SAFEGUARD: If there is no real data (all core metrics are zero), ignore any
    # persisted conclusions and use the explicit "no data" conclusions instead.
    try:
        core_keys = ["HR", "PR", "QRS", "QT", "QTc", "ST"]
        all_zero = True
        for k in core_keys:
            v = data.get(k, 0)
            try:
                all_zero = all_zero and (float(v) == 0.0)
            except Exception:
                all_zero = all_zero and (str(v).strip() in ["0", "--", "", "None"])
        if all_zero:
            dashboard_conclusions = [
                " No ECG data available",
                "Please connect device or enable demo ",
           
                
                
                
                

                

                


               

                
            ]
            print(" Overriding conclusions because all core metrics are zero (no data)")
    except Exception:
        pass

    # FILTER: Remove empty conclusions and "---" placeholders - ONLY SHOW REAL CONCLUSIONS
    # MAXIMUM 12 CONCLUSIONS (because only 12 boxes available)
    filtered_conclusions = []
    for conclusion in dashboard_conclusions:
        # Keep only non-empty conclusions that are not "---"
        if conclusion and conclusion.strip() and conclusion.strip() != "---":
            filtered_conclusions.append(conclusion.strip())
            # LIMIT: Maximum 12 conclusions (only 12 boxes available)
            if len(filtered_conclusions) >= 12:
                break
    
    print(f"\n Original conclusions: {len(dashboard_conclusions)}")
    print(f" Filtered conclusions (removed empty/---): {len(filtered_conclusions)}")
    print(f" Final conclusions to show (MAX 12): {filtered_conclusions}\n")

    #  FORCE DELETE ALL OLD WHITE BACKGROUND IMAGES
    if lead_images is None:
        print("  DELETING ALL OLD WHITE BACKGROUND IMAGES...") 
        
        # Get both possible locations
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.join(current_dir, '..', '..')
        project_root = os.path.abspath(project_root)
        src_dir = os.path.join(current_dir, '..')
        
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"] 
        
        # DELETE from both locations
        for lead in leads:
            # Location 1: project root
            img_path1 = os.path.join(project_root, f"lead_{lead}.png")
            if os.path.exists(img_path1):
                os.remove(img_path1)
                print(f"  Deleted OLD image: {img_path1}")
            
            # Location 2: src directory  
            img_path2 = os.path.join(src_dir, f"lead_{lead}.png")
            if os.path.exists(img_path2):
                os.remove(img_path2)
                print(f"  : {img_path2}")
        
        print(" CREATING NEW PINK GRID IMAGES...")
        
        # Create NEW pink grid images
        lead_images = {}
        for lead in leads:
            try:
                # Create pink grid ECG
                fig = create_ecg_grid_with_waveform(None, lead, width=6, height=2)
                
                # Save to project root with pink background
                img_path = os.path.join(project_root, f"lead_{lead}.png")
                fig.savefig(img_path, 
                           dpi=200, 
                           bbox_inches='tight', 
                           pad_inches=0.05,
                           facecolor='#ffe6e6',  # PINK background
                           edgecolor='none',
                           format='png')
                plt.close(fig)
                
                lead_images[lead] = img_path
                print(f" Created NEW PINK GRID image: {img_path}")
                
            except Exception as e:
                print(f" Error creating {lead}: {e}")
        
        if not lead_images:
            return "Error: Could not create PINK GRID ECG images"
    
    # Get REAL ECG drawings from live test page
    print(" Capturing REAL ECG data from live test page...")
    
    # Check if demo mode is active and data is available
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo = ecg_test_page.demo_toggle.isChecked()
        if is_demo:
            print("🔍 DEMO MODE DETECTED - Checking data availability...")
            if hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 0:
                # Check if data has actual variation (not just zeros)
                sample_data = ecg_test_page.data[0] if len(ecg_test_page.data) > 0 else []
                if len(sample_data) > 0:
                    std_val = np.std(sample_data)
                    print(f"    Data buffer size: {len(sample_data)}, Std deviation: {std_val:.4f}")
                    if std_val < 0.01:
                        print("    WARNING: Demo data appears to be flat/empty!")
                        print("    TIP: Make sure demo has been running for at least 5 seconds before generating report")
                    else:
                        print(f"    Demo data looks good (variation detected)")
                else:
                    print("    WARNING: Data buffer is empty!")
            else:
                print("    ERROR: No data structure found!")
    
    lead_drawings = capture_real_ecg_graphs_from_dashboard(
        dashboard_instance,
        ecg_test_page,
        samples_per_second=computed_sampling_rate,
        settings_manager=settings_manager
    )
    
    # Get lead sequence from settings (already initialized above)
    lead_sequence = settings_manager.get_setting("lead_sequence", "Standard")
    
    # Define lead orders based on sequence
    LEAD_SEQUENCES = {
        "Standard": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "Cabrera": ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    
    # Use the appropriate sequence for REPORT ONLY
    lead_order = LEAD_SEQUENCES.get(lead_sequence, LEAD_SEQUENCES["Standard"])
    
    print(f" Using lead sequence for REPORT: {lead_sequence}")
    print(f" Lead order for REPORT: {lead_order}")

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=30)

    story = []
    styles = getSampleStyleSheet()
    
    # HEADING STYLE FOR TITLE
    heading = ParagraphStyle(
        'Heading',
        fontSize=16,
        textColor=colors.HexColor("#000000"),
        spaceAfter=12,
        leading=20,
        alignment=1,  
        bold=True
    )

    # Title (switch based on demo mode)
    is_demo = False
    try:
        if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
            is_demo = bool(ecg_test_page.demo_toggle.isChecked())
    except Exception:
        pass

    title_text = "Demo ECG Report" if is_demo else "ECG Report"
    story.append(Paragraph(f"<b>{title_text}</b>", heading))
    story.append(Spacer(1, 12))


    # Patient Details
    if patient is None:
        patient = {}
    
    first_name = patient.get("first_name", "")
    last_name = patient.get("last_name", "")
    age = patient.get("age", "")
    gender = patient.get("gender", "")
   
    date_time = patient.get("date_time", "")
    
    story.append(Paragraph("<b>Patient Details</b>", styles['Heading3']))
    patient_table = Table([
        ["Name:", f"{first_name} {last_name}".strip(), "Age:", f"{age}", "Gender:", f"{gender}"],
        ["Date:", f"{date_time.split()[0] if date_time else ''}", "Time:", f"{date_time.split()[1] if len(date_time.split()) > 1 else ''}", "", ""],
        # ], colWidths=[80, 150, 50, 80, 60, 150])  # Increased all column widths
            ], colWidths=[70, 130, 40, 70, 50, 140])  # Total width = 500
    patient_table.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12)) 

    # Report Overview
    story.append(Paragraph("<b>Report Overview</b>", styles['Heading3']))
    overview_data = [
        # ["Total Number of Heartbeats (beats):", data["HR"]],
        ["Maximum Heart Rate:", f'{data["HR_max"]} bpm'],
        ["Minimum Heart Rate:", f'{data["HR_min"]} bpm'],
        ["Average Heart Rate:", f'{data["HR_avg"]} bpm'],
    ]
    table = Table(overview_data, colWidths=[300, 200])
    table.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 1, colors.black),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    story.append(table)
    story.append(Spacer(1, 15))  # Reduced from 35

    # Observation with 3 parts in ONE table (like in the image) - MADE SMALLER
    story.append(Paragraph("<b>OBSERVATION</b>", styles['Heading3']))
    story.append(Spacer(1, 6))  
    
    # Create table with 3 columns: Interval Names, Observed Values, Standard Range
    obs_headers = ["Interval Names", "Observed Values", "Standard Range"]
    
    def _fmt_ms(value):
        try:
            vf = float(value)
            if vf and vf > 0:
                return f"{vf:.0f} ms"
        except Exception:
            pass
        return "--"

    def _fmt_qtcf(value):
        try:
            vf = float(value)
            if vf and vf > 0:
                sec = vf / 1000.0
                return f"{sec:.3f} s"
        except Exception:
            pass
        return "--"

    def _fmt_st(value):
        try:
            vf = float(value)
            if vf is not None:
                return f"{int(round(vf))}"
        except Exception:
            pass
        return "--"

    obs_data = [
        ["Heart Rate", f"{data['beat']} bpm", "60-100"],                    
        ["PR Interval", _fmt_ms(data.get('PR')), "120 ms - 200 ms"],            
        ["QRS Complex", _fmt_ms(data.get('QRS')), "70 ms - 120 ms"],            
        ["QRS Axis", f"{data.get('QRS_axis', '--')}°", "Normal"],         
        ["QT Interval", _fmt_ms(data.get('QT')), "300 ms - 450 ms"],            
        ["QTCB (Bazett)", _fmt_ms(data.get('QTc')), "300 ms - 450 ms"],          
        ["QTCF (Fridericia)", _fmt_qtcf(data.get('QTc_Fridericia')), "300 ms - 450 ms"],          
        ["ST Interval", _fmt_st(data.get('ST')), "Normal"],            
    ]
    
    # Add headers to data
    obs_table_data = [obs_headers] + obs_data
    
    # Table dimensions - match total width (500) like other sections
    COLUMN_WIDTH_1 = 165  
    COLUMN_WIDTH_2 = 170 
    COLUMN_WIDTH_3 = 165
    ROW_HEIGHT = 12       
    HEADER_HEIGHT = 22    
    
    # Create table with 3 columns and custom dimensions
    obs_table = Table(obs_table_data, colWidths=[COLUMN_WIDTH_1, COLUMN_WIDTH_2, COLUMN_WIDTH_3])
    
    # Style the table with custom dimensions - SMALLER
    obs_table.setStyle(TableStyle([
        # Header row styling - REDUCED
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6f2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),  # Reduced from 11 to 9
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), HEADER_HEIGHT//2),
        ("TOPPADDING", (0, 0), (-1, 0), HEADER_HEIGHT//2),
        
        # Data rows styling - REDUCED
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),  # Reduced from 10 to 8
        ("ALIGN", (0, 1), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),  # Reduced from 5 to 3
        ("TOPPADDING", (0, 1), (-1, -1), 3),     # Reduced from 5 to 3
        
        # Grid and borders
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(obs_table)
    story.append(Spacer(1, 8))  # Reduced spacing

   

    # Conclusion in table format - NOW DYNAMIC FROM DASHBOARD - ONLY REAL CONCLUSIONS - MADE SMALLER
    story.append(Paragraph("<b>ECG Report Conclusion</b>", styles['Heading3']))
    story.append(Spacer(1, 6))   # Reduced spacing
    
    # Create dynamic conclusion table using ONLY filtered conclusions (no empty/---)
    conclusion_headers = ["S.No.", "Conclusion"]
    conclusion_data = []
    
    # ONLY show real conclusions with proper numbering (1, 2, 3...)
    for i, conclusion in enumerate(filtered_conclusions, 1):
        conclusion_data.append([str(i), conclusion])
    
    print(f" Creating conclusion table with {len(conclusion_data)} rows (only real conclusions):")
    for row in conclusion_data:
        print(f"   {row}")
    
    # Add headers to conclusion data
    conclusion_table_data = [conclusion_headers] + conclusion_data
    
    # Create conclusion table - match total width (500) like other sections
    conclusion_table = Table(conclusion_table_data, colWidths=[80, 420])
    
    # Style the conclusion table - SMALLER
    conclusion_table.setStyle(TableStyle([
        # Header row styling - REDUCED
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6f2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),  
        ("ALIGN", (0, 0), (-1, 0), "CENTER"), 
        ("TOPPADDING", (0, 0), (-1, 0), 6),  
        
        # Data rows styling - REDUCED
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),  
        ("ALIGN", (0, 1), (0, -1), "CENTER"),  
        ("ALIGN", (1, 1), (1, -1), "LEFT"),     
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),  
        ("TOPPADDING", (0, 1), (-1, -1), 4),    
        
        # Grid and borders
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(conclusion_table)
    story.append(Spacer(1, 8))  # Reduced spacing

    # REMOVE PageBreak HERE to send patient details to Page 2
    # story.append(PageBreak())

    # Now these patient details will be on Page 2 top
    # Patient header on Page 2 (Name, Age, Gender, Date/Time)
    if patient is None:
        patient = {}
    first_name = patient.get("first_name", "")
    last_name = patient.get("last_name", "")
    full_name = f"{first_name} {last_name}".strip()
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    date_time_str = patient.get("date_time", "")

    # REMOVED: Date/Time table from story - will be added in master drawing instead
    # Patient info and vital parameters are now in master drawing above ECG graph
    # No extra spacing needed as they're positioned in drawing coordinates

    
    #  CREATE SINGLE MASSIVE DRAWING with ALL ECG content (NO individual drawings)
    print("Creating SINGLE drawing with all ECG content...")
    
    # Single drawing dimensions - ADJUSTED HEIGHT to fit within page frame (max ~770)
    total_width = 520   # Full page width
    total_height = 720  # Reduced to 720 to fit within page frame (max ~770) with margin
    
    # Create ONE master drawing
    master_drawing = Drawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads based on selected sequence (SHIFTED UP by 80 points total: 40+25+15)
    y_positions = [580, 530, 480, 430, 380, 330, 280, 230, 180, 130, 80, 30]  
    6
    lead_positions = []
    
    for i, lead in enumerate(lead_order):
        lead_positions.append({
            "lead": lead, 
            "x": 60, 
            "y": y_positions[i]
        })
    
    print(f" Using lead positions in {lead_sequence} sequence: {[pos['lead'] for pos in lead_positions]}")
    
    # STEP 3: Draw ALL ECG content directly in master drawing
    successful_graphs = 0
    
    # Check if demo mode is active and get time window for filtering
    is_demo_mode = False
    time_window_seconds = None
    samples_per_second = computed_sampling_rate
    
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo_mode = ecg_test_page.demo_toggle.isChecked()
        if is_demo_mode:
            # Get time window from demo manager
            if hasattr(ecg_test_page, 'demo_manager') and ecg_test_page.demo_manager:
                time_window_seconds = getattr(ecg_test_page.demo_manager, 'time_window', None)
                samples_per_second = getattr(ecg_test_page.demo_manager, 'samples_per_second', samples_per_second)
                print(f"🔍 Report Generator: Demo mode ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    # NEW LOGIC: Time window = 165mm / wave_speed (33 boxes × 5mm = 165mm)
                    ecg_graph_width_mm = 33 * 5  # 165mm
                    time_window_seconds = ecg_graph_width_mm / wave_speed
                    print(f"🔍 Report Generator: Demo mode ON - Calculated window using NEW LOGIC: 165mm / {wave_speed}mm/s = {time_window_seconds}s")
                except Exception as e:
                    print(f"⚠️ Could not get demo time window: {e}")
                    time_window_seconds = None
        else:
            print(f"🔍 Report Generator: Demo mode is OFF")
    
    # Calculate number of samples to capture based on demo mode OR BPM + wave_speed
    calculated_time_window = None  # Initialize for use in data loading section
    if is_demo_mode and time_window_seconds is not None:
        # In demo mode: only capture data visible in one window frame
        calculated_time_window = time_window_seconds
        num_samples_to_capture = int(time_window_seconds * samples_per_second)
        print(f"📊 DEMO MODE: Master drawing will capture only {num_samples_to_capture} samples ({time_window_seconds}s window)")
    else:
        # Normal mode: Calculate time window based on wave_speed ONLY (NEW LOGIC)
        # This ensures proper number of beats are displayed based on graph width
        # Formula: 
        #   - Time window = 165mm / wave_speed ONLY (33 boxes × 5mm = 165mm)
        #   - BPM window is NOT used - only wave speed determines time window
        #   - Beats = (BPM / 60) × time_window
        #   - Maximum clamp: 20 seconds (NO minimum clamp)
        calculated_time_window, _ = calculate_time_window_from_bpm_and_wave_speed(
            hr_bpm_value,  # From metrics.json (priority) - for calculation-based beats
            wave_speed_mm_s,  # From ecg_settings.json - for calculation-based beats
            desired_beats=6  # Default: 6 beats desired
        )
        
        # Recalculate with actual sampling rate
        num_samples_to_capture = int(calculated_time_window * computed_sampling_rate)
        print(f"📊 NORMAL MODE: Calculated time window: {calculated_time_window:.2f}s")
        print(f"   Based on BPM={hr_bpm_value} and wave_speed={wave_speed_mm_s}mm/s")
        print(f"   Will capture {num_samples_to_capture} samples (at {computed_sampling_rate}Hz)")
        if hr_bpm_value > 0:
            expected_beats = int((calculated_time_window * hr_bpm_value) / 60)
            print(f"   Expected beats shown: ~{expected_beats} beats")
    
    for pos_info in lead_positions:
        lead = pos_info["lead"]
        x_pos = pos_info["x"]
        y_pos = pos_info["y"]
        
        try:
            # STEP 3A: Add lead label directly
            from reportlab.graphics.shapes import String
            lead_label = String(10, y_pos + 20, f"{lead}", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # STEP 3B: Get REAL ECG data for this lead (ONLY from saved file - calculation-based)
            # IMPORTANT:  saved file  data use , live dashboard   (calculation-based beats  )
            real_data_available = False
            real_ecg_data = None
            
            # Helper function to calculate derived leads from I and II
            def calculate_derived_lead(lead_name, lead_i_data, lead_ii_data):
                """Calculate derived leads: III, aVR, aVL, aVF from I and II"""
                lead_i = np.array(lead_i_data, dtype=float)
                lead_ii = np.array(lead_ii_data, dtype=float)
                
                if lead_name == "III":
                    return lead_ii - lead_i  # III = II - I
                elif lead_name == "aVR":
                    return -(lead_i + lead_ii) / 2.0  # aVR = -(I + II) / 2
                elif lead_name == "aVL":
                    # aVL = (Lead I - Lead III) / 2
                    lead_iii = lead_ii - lead_i  # Calculate Lead III first
                    return (lead_i - lead_iii) / 2.0  # aVL = (I - III) / 2
                elif lead_name == "aVF":
                    # aVF = (Lead II + Lead III) / 2
                    lead_iii = lead_ii - lead_i  # Calculate Lead III first
                    return (lead_ii + lead_iii) / 2.0  # aVF = (II + III) / 2
                elif lead_name == "-aVR":
                    return -(-(lead_i + lead_ii) / 2.0)  # -aVR = -aVR = (I + II) / 2
                else:
                    return None
            
            # Priority 1: Use saved_ecg_data (REQUIRED for calculation-based beats)
            saved_data_samples = 0  # Initialize for comparison with live data
            if saved_ecg_data and 'leads' in saved_ecg_data:
                # For calculated leads, calculate from I and II
                if lead in ["III", "aVR", "aVL", "aVF", "-aVR"]:
                    if "I" in saved_ecg_data['leads'] and "II" in saved_ecg_data['leads']:
                        lead_i_data = saved_ecg_data['leads']["I"]
                        lead_ii_data = saved_ecg_data['leads']["II"]
                        
                        # Ensure same length
                        min_len = min(len(lead_i_data), len(lead_ii_data))
                        lead_i_data = lead_i_data[:min_len]
                        lead_ii_data = lead_ii_data[:min_len]
                        
                        # IMPORTANT: Subtract baseline from Lead I and Lead II BEFORE calculating derived leads
                        # This ensures calculated leads are centered around 0, not around baseline
                        baseline_adc = 2000.0
                        lead_i_centered = np.array(lead_i_data, dtype=float) - baseline_adc
                        lead_ii_centered = np.array(lead_ii_data, dtype=float) - baseline_adc
                        
                        # Calculate derived lead from centered values
                        calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                        if calculated_data is not None:
                            raw_data = calculated_data.tolist() if isinstance(calculated_data, np.ndarray) else calculated_data
                            print(f"✅ Calculated {lead} from saved I and II data (baseline-subtracted): {len(raw_data)} points")
                        else:
                            # Fallback to saved data if calculation fails
                            lead_name_for_saved = lead.replace("-aVR", "aVR")
                            if lead_name_for_saved in saved_ecg_data['leads']:
                                raw_data = saved_ecg_data['leads'][lead_name_for_saved]
                                if lead == "-aVR":
                                    raw_data = [-x for x in raw_data]  # Invert for -aVR
                            else:
                                raw_data = []
                    else:
                        print(f"⚠️ Cannot calculate {lead}: I or II data missing in saved file")
                        raw_data = []
                else:
                    # For non-calculated leads, use saved data directly
                    lead_name_for_saved = lead.replace("-aVR", "aVR")  # Handle -aVR case
                    if lead_name_for_saved in saved_ecg_data['leads']:
                        raw_data = saved_ecg_data['leads'][lead_name_for_saved]
                        if lead == "-aVR":
                            raw_data = [-x for x in raw_data]  # Invert for -aVR
                    else:
                        raw_data = []
                
                if len(raw_data) > 0:
                    # Check if saved data has enough samples for calculated time window
                    saved_data_samples = len(raw_data)
                    if saved_data_samples < num_samples_to_capture:
                        print(f"⚠️ SAVED FILE {lead} has only {saved_data_samples} samples, need {num_samples_to_capture} for {calculated_time_window:.2f}s window")
                        print(f"   Will use ALL saved data ({saved_data_samples} samples) - may show fewer beats than calculated")
                        # Use all available saved data (don't filter)
                        raw_data_to_use = raw_data
                    else:
                        # Apply time window filtering based on calculated window
                        raw_data_to_use = raw_data[-num_samples_to_capture:]
                    
                    if len(raw_data_to_use) > 0 and np.std(raw_data_to_use) > 0.01:
                        real_ecg_data = np.array(raw_data_to_use)
                        real_data_available = True
                        time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                        actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                        print(f"✅ Using SAVED FILE {lead} data: {len(real_ecg_data)} points (requested: {time_window_str}, actual: {actual_time_window:.2f}s, std: {np.std(real_ecg_data):.2f})")
            
            # Priority 2: Fallback to live dashboard data (if saved data not available OR has insufficient samples)
            # Check if live data has MORE samples than saved data
            if ecg_test_page and hasattr(ecg_test_page, 'data'):
                lead_to_index = {
                    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                }
                
                live_data_available = False
                live_data_samples = 0
                
                # For calculated leads, calculate from live I and II
                if lead in ["III", "aVR", "aVL", "aVF", "-aVR"]:
                    if len(ecg_test_page.data) > 1:  # Need at least I and II
                        lead_i_data = ecg_test_page.data[0]  # I
                        lead_ii_data = ecg_test_page.data[1]  # II
                        
                        if len(lead_i_data) > 0 and len(lead_ii_data) > 0:
                            # Ensure same length
                            min_len = min(len(lead_i_data), len(lead_ii_data))
                            lead_i_slice = lead_i_data[-min_len:] if len(lead_i_data) >= min_len else lead_i_data
                            lead_ii_slice = lead_ii_data[-min_len:] if len(lead_ii_data) >= min_len else lead_ii_data
                            
                            # IMPORTANT: Subtract baseline from Lead I and Lead II BEFORE calculating derived leads
                            # This ensures calculated leads are centered around 0, not around baseline
                            baseline_adc = 2000.0
                            lead_i_centered = np.array(lead_i_slice, dtype=float) - baseline_adc
                            lead_ii_centered = np.array(lead_ii_slice, dtype=float) - baseline_adc
                            
                            # Calculate derived lead from centered values
                            calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                            if calculated_data is not None:
                                live_data_samples = len(calculated_data)
                                use_live_data = False
                                if not real_data_available:
                                    use_live_data = True
                                elif live_data_samples > saved_data_samples:
                                    use_live_data = True
                                
                                if use_live_data:
                                    raw_data = calculated_data
                                    if len(raw_data) >= num_samples_to_capture:
                                        raw_data = raw_data[-num_samples_to_capture:]
                                    if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                        real_ecg_data = np.array(raw_data)
                                        real_data_available = True
                                        actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                
                # For non-calculated leads, use existing logic
                if not real_data_available:
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        live_data_samples = len(ecg_test_page.data[3])
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        live_data_samples = len(ecg_test_page.data[lead_to_index[lead]])
                    
                    # Use live data if: (1) saved data not available OR (2) live data has MORE samples
                    use_live_data = False
                    if not real_data_available:
                        use_live_data = True
                    elif live_data_samples > saved_data_samples:
                        use_live_data = True
                    
                    if use_live_data:
                        if lead == "-aVR" and len(ecg_test_page.data) > 3:
                            # For -aVR, use filtered inverted aVR data
                            raw_data = ecg_test_page.data[3]
                            # Check if we have enough samples, otherwise use all available
                            if len(raw_data) >= num_samples_to_capture:
                                raw_data = raw_data[-num_samples_to_capture:]
                            # Check if data is not all zeros or flat
                            if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                                real_ecg_data = np.array(raw_data)
                                
                                real_data_available = True
                                actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                                if is_demo_mode and time_window_seconds is not None:
                                    pass
                                else:
                                    time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                            else:
                                pass
                        elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                            # Get filtered real data for this lead
                            lead_index = lead_to_index[lead]
                            if len(ecg_test_page.data[lead_index]) > 0:
                                raw_data = ecg_test_page.data[lead_index]
                                # Check if we have enough samples, otherwise use all available
                                if len(raw_data) >= num_samples_to_capture:
                                    raw_data = raw_data[-num_samples_to_capture:]
                                # Check if data has variation (not all zeros or flat line)
                                if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                    # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                                    real_ecg_data = np.array(raw_data)
                                    
                                    real_data_available = True
                                    actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                                    if is_demo_mode and time_window_seconds is not None:
                                        pass
                                    else:
                                        time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                                else:
                                    pass
                            else:
                                pass
            
            if real_data_available and len(real_ecg_data) > 0:
                # Draw ALL REAL ECG data - NO LIMITS
                ecg_width = 460
                ecg_height = 45
                
                # Create time array for ALL data
                t = np.linspace(x_pos, x_pos + ecg_width, len(real_ecg_data))
                
                
                # Step 1: Convert ADC data to numpy array
                adc_data = np.array(real_ecg_data, dtype=float)

                # Step 1.1: Apply report filters (DFT -> EMG -> AC) on raw ADC data
                try:
                    from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
                    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
                    emg_setting = str(settings_manager.get_setting("filter_emg", "off")).strip()
                    ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
                    if dft_setting not in ("off", ""):
                        adc_data = apply_dft_filter(adc_data, float(computed_sampling_rate), dft_setting)
                    if emg_setting not in ("off", ""):
                        adc_data = apply_emg_filter(adc_data, float(computed_sampling_rate), emg_setting)
                    if ac_setting in ("50", "60"):
                        adc_data = apply_ac_filter(adc_data, float(computed_sampling_rate), ac_setting)
                except Exception as filter_err:
                    print(f" Report filter apply failed for {lead}: {filter_err}")
                
                # Step 1: Apply baseline correction based on data type
                data_mean = np.mean(adc_data)
                baseline_adc = 2000.0
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                if abs(data_mean - 2000.0) < 500:  # Data is close to baseline 2000 (raw ADC)
                    baseline_corrected = adc_data - baseline_adc
                elif is_calculated_lead:
                    baseline_corrected = adc_data  # Calculated leads already centered
                else:
                    baseline_corrected = adc_data  # Already processed data
                
                # Step 2: FORCE CENTER for report - subtract mean to ensure perfect centering
                # IMPORTANT: Report me har lead apni grid line ke center me dikhni chahiye
                # Chahe baseline wander kitna bhi ho (respiration mode, Fluke data, etc.)
                # This ensures waveform is exactly centered on grid line regardless of baseline wander
                centered_adc = baseline_corrected - np.mean(baseline_corrected)
                
                # Step 3: Calculate ADC per box based on wave_gain and lead-specific multiplier
                # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
                # Each lead can have different ADC per box multiplier (will be divided by wave_gain)
                # Use uniform ADC per box multiplier (HRV uses lead-specific mapping)
                adc_per_box_multiplier = 6400.0
                # Formula: ADC_per_box = adc_per_box_multiplier / wave_gain_mm_mv
                # IMPORTANT: Each lead can have different ADC per box multiplier
                # For 10mm/mV with multiplier 6400: 6400 / 10 = 640 ADC per box
                # For 10mm/mV with multiplier 8209: 8209 / 10 = 821 ADC per box
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
                
                # DEBUG: Log actual ADC values for troubleshooting
                max_centered_adc_abs = np.max(np.abs(centered_adc))
                expected_boxes = max_centered_adc_abs / adc_per_box
                
                
                
                boxes_offset = centered_adc / adc_per_box
                
                # Step 5: Convert boxes to Y position (in mm, then to points)
                # Center of graph is at y_pos + (ecg_height / 2.0)
                # IMPORTANT: User changed to height/3 = 45/3 = 15.0 points per box
                # This matches the actual grid spacing the user wants
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                major_spacing_y = ecg_height / 3.0  # height/3 = 15.0 points per box (user's choice)
                box_height_points = major_spacing_y  # Use actual grid spacing (height/3)
                
                # Convert boxes offset to Y position
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                
                # Draw ALL REAL ECG data points
                from reportlab.graphics.shapes import Path
                ecg_path = Path(fillColor=None, 
                               strokeColor=colors.HexColor("#000000"), 
                               strokeWidth=0.4,
                               strokeLineCap=1,
                               strokeLineJoin=1)
                
                # DEBUG: Verify actual plotted values
                actual_min_y = np.min(ecg_normalized)
                actual_max_y = np.max(ecg_normalized)
                actual_span_points = actual_max_y - actual_min_y
                actual_span_boxes = actual_span_points / box_height_points
                
                # Start path
                ecg_path.moveTo(t[0], ecg_normalized[0])
                
                # Add ALL points
                for i in range(1, len(t)):
                    ecg_path.lineTo(t[i], ecg_normalized[i])
                
                # Add path to master drawing
                master_drawing.add(ecg_path)
                
                print(f"✅ Drew {len(real_ecg_data)} ECG data points for Lead {lead}")
            else:
                print(f"📋 No real data for Lead {lead} - showing grid only")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f"❌ Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 4: Add Patient Info, Date/Time and Vital Parameters to master drawing
    # POSITIONED ABOVE ECG GRAPH (not mixed inside graph)
    from reportlab.graphics.shapes import String

    # LEFT SIDE: Patient Info (ABOVE ECG GRAPH - shifted further up)
    patient_name_label = String(-30, 740, f"Name: {full_name}",  # Moved up from 700 to 710
                           fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_name_label)

    patient_age_label = String(-30, 720, f"Age: {age}",  # Moved up from 680 to 690
                          fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_age_label)

    patient_gender_label = String(-30, 700, f"Gender: {gender}",  # Moved up from 660 to 670
                             fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_gender_label)
    
    # RIGHT SIDE: Date/Time (ABOVE ECG GRAPH - shifted further up)
    if date_time_str:
        parts = date_time_str.split()
        date_part = parts[0] if parts else ""
        time_part = parts[1] if len(parts) > 1 else ""
    else:
        date_part, time_part = "____", "____"
    
    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    hr_val = data.get('HR') or data.get('HR_bpm') or data.get('Heart_Rate') or data.get('HR_avg', )
    HR = int(round(hr_val)) if hr_val else 0
    PR = data.get('PR', 192) 
    QRS = data.get('QRS', 93)
    QT = data.get('QT', 354)
    QTc = data.get('QTc', 260)
    ST = data.get('ST', 114)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   
    # Add vital parameters in TWO COLUMNS (ABOVE ECG GRAPH - shifted further up)
    # FIRST COLUMN (Left side - x=175)
    hr_label = String(250, 740, f"HR    : {HR} bpm",
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(hr_label)

    pr_label = String(250, 720, f"PR    : {PR} ms",
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(pr_label)

    qrs_label = String(250, 700, f"QRS : {QRS} ms",
                      fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qrs_label)
    
    rr_label = String(250, 682, f"RR    : {RR} ms",
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rr_label)

    qt_label = String(250, 664, f"QT    : {int(round(QT))} ms",
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qt_label)

    qtc_label = String(250, 646, f"QTc  : {int(round(QTc))} ms",
                      fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtc_label)

    # SECOND COLUMN (Right side - x=240)
    st_label = String(240, 664, f"ST            : {int(round(ST))} ms",  
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(st_label)

    # CALCULATED wave amplitudes and lead-specific measurements
    # Prefer values passed in data; if missing/zero, compute from live ecg_test_page data (last 10s)
    p_amp_mv = data.get('p_amp', 0.0)
    qrs_amp_mv = data.get('qrs_amp', 0.0)
    t_amp_mv = data.get('t_amp', 0.0)
    
    print(f"🔬 Report Generator - Received wave amplitudes from data:")
    print(f"   p_amp: {p_amp_mv}, qrs_amp: {qrs_amp_mv}, t_amp: {t_amp_mv}")
    print(f"   Available keys in data: {list(data.keys())}")
    
    # If not provided or zero, compute quickly from Lead II in ecg_test_page (robust fallback)
    def _compute_from_data_array(arr, fs):
        from scipy.signal import butter, filtfilt, find_peaks
        if arr is None or len(arr) < int(2*fs) or np.std(arr) < 0.1:
            return 0.0, 0.0, 0.0
        nyq = fs/2.0
        b,a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq,0.99)], btype='band')
        x = filtfilt(b,a,arr)
        # Simple R detection via Pan-Tompkins style envelope
        squared = np.square(np.diff(x))
        win = max(1, int(0.15*fs))
        env = np.convolve(squared, np.ones(win)/win, mode='same')
        thr = np.mean(env) + 0.5*np.std(env)
        r_peaks, _ = find_peaks(env, height=thr, distance=int(0.6*fs))
        if len(r_peaks) < 3:
            return 0.0, 0.0, 0.0
        p_vals, qrs_vals, t_vals = [], [], []
        for r in r_peaks[1:-1]:
            # P: 120-200ms before R
            p_start = max(0, r-int(0.20*fs)); p_end = max(0, r-int(0.12*fs))
            if p_end>p_start:
                seg = x[p_start:p_end]
                base = np.mean(x[max(0,p_start-int(0.05*fs)):p_start])
                p_vals.append(max(seg)-base)
            # QRS: +-80ms around R
            qrs_start = max(0, r-int(0.08*fs)); qrs_end = min(len(x), r+int(0.08*fs))
            if qrs_end>qrs_start:
                seg = x[qrs_start:qrs_end]
                qrs_vals.append(max(seg)-min(seg))
            # T: 100-300ms after R
            t_start = min(len(x), r+int(0.10*fs)); t_end = min(len(x), r+int(0.30*fs))
            if t_end>t_start:
                seg = x[t_start:t_end]
                base = np.mean(x[r:t_start]) if t_start>r else 0.0
                t_vals.append(max(seg)-base)
        def med(v):
            return float(np.median(v)) if len(v)>0 else 0.0
        return med(p_vals), med(qrs_vals), med(t_vals)

    if (p_amp_mv<=0 or qrs_amp_mv<=0 or t_amp_mv<=0) and ecg_test_page is not None and hasattr(ecg_test_page,'data'):
        try:
            fs = 500.0
            if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler,'sampling_rate') and ecg_test_page.sampler.sampling_rate:
                fs = float(ecg_test_page.sampler.sampling_rate)
            arr = None
            if len(ecg_test_page.data)>1:
                lead_ii = ecg_test_page.data[1]
                if isinstance(lead_ii, (list, tuple)):
                    lead_ii = np.asarray(lead_ii)
                arr = lead_ii[-int(10*fs):] if lead_ii is not None and len(lead_ii)>int(10*fs) else lead_ii
            cp, cqrs, ct = _compute_from_data_array(arr, fs)
            if p_amp_mv<=0: p_amp_mv = cp
            if qrs_amp_mv<=0: qrs_amp_mv = cqrs
            if t_amp_mv<=0: t_amp_mv = ct
            print(f"🔁 Fallback computed amplitudes from Lead II: P={p_amp_mv:.4f}, QRS={qrs_amp_mv:.4f}, T={t_amp_mv:.4f}")
        except Exception as e:
            print(f"⚠️ Fallback amplitude computation failed: {e}")

    # Calculate P/QRS/T Axis in degrees (using Lead I and Lead aVF)
    p_axis_deg = "--"
    qrs_axis_deg = "--"
    t_axis_deg = "--"
    
    if ecg_test_page is not None and hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 5:
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            
            # Get Lead I (index 0) and Lead aVF (index 5)
            lead_I = ecg_test_page.data[0] if len(ecg_test_page.data) > 0 else None
            lead_aVF = ecg_test_page.data[5] if len(ecg_test_page.data) > 5 else None
            
            # Get sampling rate
            fs = 500.0
            if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate') and ecg_test_page.sampler.sampling_rate:
                fs = float(ecg_test_page.sampler.sampling_rate)
            
            if lead_I is not None and lead_aVF is not None:
                # Convert to numpy arrays
                if isinstance(lead_I, (list, tuple)):
                    lead_I = np.asarray(lead_I)
                if isinstance(lead_aVF, (list, tuple)):
                    lead_aVF = np.asarray(lead_aVF)
                
                # Get last 10 seconds of data
                def _get_last(arr):
                    return arr[-int(10*fs):] if arr is not None and len(arr) > int(10*fs) else arr
                
                lead_I_data = _get_last(lead_I)
                lead_aVF_data = _get_last(lead_aVF)
                
                if len(lead_I_data) > int(2*fs) and len(lead_aVF_data) > int(2*fs):
                    # Filter signals
                    nyq = fs/2.0
                    b, a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq, 0.99)], btype='band')
                    lead_I_filt = filtfilt(b, a, lead_I_data)
                    lead_aVF_filt = filtfilt(b, a, lead_aVF_data)
                    
                    # Detect R peaks using Pan-Tompkins style
                    squared = np.square(np.diff(lead_aVF_filt))
                    win = max(1, int(0.15*fs))
                    env = np.convolve(squared, np.ones(win)/win, mode='same')
                    thr = np.mean(env) + 0.5*np.std(env)
                    r_peaks, _ = find_peaks(env, height=thr, distance=int(0.6*fs))
                    
                    if len(r_peaks) >= 3:
                        # Calculate QRS Axis
                        from .twelve_lead_test import calculate_qrs_axis
                        qrs_axis_result = calculate_qrs_axis(lead_I_filt, lead_aVF_filt, r_peaks, fs=fs, window_ms=100)
                        if qrs_axis_result != "--":
                            qrs_axis_deg = qrs_axis_result
                        
                        # Helper function to calculate axis for any wave
                        def calculate_wave_axis(lead_I_sig, lead_aVF_sig, wave_peaks, fs, window_before_ms, window_after_ms):
                            """Calculate axis for P or T wave"""
                            if len(lead_I_sig) < 100 or len(lead_aVF_sig) < 100 or len(wave_peaks) == 0:
                                return "--"
                            window_before = int(window_before_ms * fs / 1000)
                            window_after = int(window_after_ms * fs / 1000)
                            net_I = []
                            net_aVF = []
                            for peak in wave_peaks:
                                start = max(0, peak - window_before)
                                end = min(len(lead_I_sig), peak + window_after)
                                if end > start:
                                    net_I.append(np.sum(lead_I_sig[start:end]))
                                    net_aVF.append(np.sum(lead_aVF_sig[start:end]))
                            if len(net_I) == 0:
                                return "--"
                            mean_I = np.mean(net_I)
                            mean_aVF = np.mean(net_aVF)
                            if abs(mean_I) < 1e-6 and abs(mean_aVF) < 1e-6:
                                return "--"
                            axis_rad = np.arctan2(mean_aVF, mean_I)
                            axis_deg = np.degrees(axis_rad)
                            
                            # Normalize to -180 to +180 (clinical standard, matches standardized function)
                            # This ensures consistency with calculate_axis_from_median_beat()
                            if axis_deg > 180:
                                axis_deg -= 360
                            if axis_deg < -180:
                                axis_deg += 360
                            
                            return f"{int(round(axis_deg))}°"
                        
                        # Detect P peaks (adaptive window based on HR)
                        # Calculate HR from R-peaks for adaptive detection
                        if len(r_peaks) >= 2:
                            rr_intervals = np.diff(r_peaks) / fs  # in seconds
                            mean_rr = np.mean(rr_intervals)
                            estimated_hr = 60.0 / mean_rr if mean_rr > 0 else 100
                        else:
                            estimated_hr = 100
                        
                        # Adaptive P wave detection window based on HR
                        # At very high HR (>140), P waves are hard to detect due to T-P overlap
                        # At high HR (>100), use narrower window to avoid T wave overlap
                        if estimated_hr > 140:
                            # Very high HR: use very narrow window or skip P detection
                            p_window_before_ms = 0.12  # 120ms - very narrow
                            p_window_after_ms = 0.08   # 80ms - very narrow
                            use_lead_I_for_p = True  # Prefer Lead I at very high HR
                        elif estimated_hr > 100:
                            p_window_before_ms = 0.15  # 150ms instead of 200ms
                            p_window_after_ms = 0.10   # 100ms instead of 120ms
                            use_lead_I_for_p = False
                        else:
                            p_window_before_ms = 0.20  # Standard 200ms
                            p_window_after_ms = 0.12   # Standard 120ms
                            use_lead_I_for_p = False
                        
                        # For very high HR, try Lead I first (usually clearer P waves)
                        if use_lead_I_for_p:
                            p_peaks = []
                            for r in r_peaks[1:-1]:  # Skip first and last
                                p_start = max(0, r - int(p_window_before_ms*fs))
                                p_end = max(0, r - int(p_window_after_ms*fs))
                                if p_end > p_start:
                                    # Try Lead I first at very high HR
                                    segment = lead_I_filt[p_start:p_end]
                                    if len(segment) > 0:
                                        # Look for positive deflection (P wave is usually positive)
                                        # Use argmax but validate it's actually a peak
                                        p_idx = p_start + np.argmax(segment)
                                        # Validate: peak should be above baseline
                                        if segment[np.argmax(segment)] > np.mean(segment) + 0.1 * np.std(segment):
                                            p_peaks.append(p_idx)
                        else:
                            # Standard detection using Lead aVF
                            p_peaks = []
                            for r in r_peaks[1:-1]:  # Skip first and last
                                p_start = max(0, r - int(p_window_before_ms*fs))
                                p_end = max(0, r - int(p_window_after_ms*fs))
                                if p_end > p_start:
                                    segment = lead_aVF_filt[p_start:p_end]
                                    if len(segment) > 0:
                                        p_idx = p_start + np.argmax(segment)
                                        p_peaks.append(p_idx)
                        
                        # Try to calculate P axis even with fewer peaks if possible
                        if len(p_peaks) >= 2:
                            p_axis_result = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks, fs, 20, 60)
                            if p_axis_result != "--":
                                # Validate P axis is in normal range (0-75°)
                                p_axis_num = int(str(p_axis_result).replace("°", ""))
                                # Normalize to -180 to +180 range for comparison
                                if p_axis_num > 180:
                                    p_axis_num_normalized = p_axis_num - 360
                                else:
                                    p_axis_num_normalized = p_axis_num
                                
                                # Debug: Print HR and P axis for troubleshooting
                                print(f"🔍 P axis validation: HR={estimated_hr:.1f} BPM, P_axis={p_axis_num}°, normalized={p_axis_num_normalized}°")
                                
                                # Check if P axis is in normal range (0 to 75°)
                                # P axis normal range: 0° to +75°
                                # For values > 180°, normalize to negative (e.g., 174° stays 174°, but 200° becomes -160°)
                                # But 174° is still abnormal (> 75°)
                                is_normal = False
                                if p_axis_num_normalized >= 0 and p_axis_num_normalized <= 75:
                                    is_normal = True
                                elif p_axis_num >= 0 and p_axis_num <= 75:
                                    is_normal = True
                                
                                if is_normal:
                                    p_axis_deg = p_axis_result
                                else:
                                    # P axis abnormal - try multiple fallback methods to get best possible value
                                    # Always try to return a value instead of "--"
                                    hr_from_data = data.get('HR', 0) if data else 0
                                    hr_from_data = hr_from_data if isinstance(hr_from_data, (int, float)) else 0
                                    
                                    # Try multiple fallback methods
                                    p_axis_candidates = []
                                    
                                    # Method 1: Try Lead I detection (if not already used)
                                    if not use_lead_I_for_p:
                                        p_peaks_alt1 = []
                                        for r in r_peaks[1:-1]:
                                            p_start = max(0, r - int(p_window_before_ms*fs))
                                            p_end = max(0, r - int(p_window_after_ms*fs))
                                            if p_end > p_start:
                                                segment = lead_I_filt[p_start:p_end]
                                                if len(segment) > 0:
                                                    p_idx = p_start + np.argmax(segment)
                                                    p_peaks_alt1.append(p_idx)
                                        
                                        if len(p_peaks_alt1) >= 2:
                                            p_axis_result_alt1 = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks_alt1, fs, 20, 60)
                                            if p_axis_result_alt1 != "--":
                                                p_axis_candidates.append(p_axis_result_alt1)
                                    
                                    # Method 2: Try Lead aVF detection (if not already used)
                                    if use_lead_I_for_p:
                                        p_peaks_alt2 = []
                                        for r in r_peaks[1:-1]:
                                            p_start = max(0, r - int(p_window_before_ms*fs))
                                            p_end = max(0, r - int(p_window_after_ms*fs))
                                            if p_end > p_start:
                                                segment = lead_aVF_filt[p_start:p_end]
                                                if len(segment) > 0:
                                                    p_idx = p_start + np.argmax(segment)
                                                    p_peaks_alt2.append(p_idx)
                                        
                                        if len(p_peaks_alt2) >= 2:
                                            p_axis_result_alt2 = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks_alt2, fs, 20, 60)
                                            if p_axis_result_alt2 != "--":
                                                p_axis_candidates.append(p_axis_result_alt2)
                                    
                                    # Method 3: Try wider window for high HR
                                    if estimated_hr > 100:
                                        p_peaks_alt3 = []
                                        wider_window_before = 0.18 if estimated_hr > 140 else 0.16
                                        wider_window_after = 0.11 if estimated_hr > 140 else 0.10
                                        for r in r_peaks[1:-1]:
                                            p_start = max(0, r - int(wider_window_before*fs))
                                            p_end = max(0, r - int(wider_window_after*fs))
                                            if p_end > p_start:
                                                segment = lead_I_filt[p_start:p_end]
                                                if len(segment) > 0:
                                                    p_idx = p_start + np.argmax(segment)
                                                    p_peaks_alt3.append(p_idx)
                                        
                                        if len(p_peaks_alt3) >= 2:
                                            p_axis_result_alt3 = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks_alt3, fs, 15, 50)
                                            if p_axis_result_alt3 != "--":
                                                p_axis_candidates.append(p_axis_result_alt3)
                                    
                                    # Add original result as candidate
                                    p_axis_candidates.append(p_axis_result)
                                    
                                    # Select best candidate: prefer values in normal range, otherwise use closest to normal
                                    best_p_axis = None
                                    best_score = -1
                                    
                                    for candidate in p_axis_candidates:
                                        if candidate == "--":
                                            continue
                                        cand_num = int(str(candidate).replace("°", ""))
                                        if cand_num > 180:
                                            cand_normalized = cand_num - 360
                                        else:
                                            cand_normalized = cand_num
                                        
                                        # Score: prefer values in normal range (0-75°)
                                        if 0 <= cand_normalized <= 75:
                                            score = 100 - abs(cand_normalized - 37.5)  
                                        else:
                                            # For abnormal values, prefer closer to normal range
                                            if cand_normalized > 75:
                                                score = max(0, 50 - (cand_normalized - 75))
                                            else:
                                                score = max(0, 50 - abs(cand_normalized))
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_p_axis = candidate
                                    
                                    # Use best candidate or original if no better option
                                    if best_p_axis:
                                        p_axis_deg = best_p_axis
                                        if best_p_axis != p_axis_result:
                                            print(f"⚠️ P axis adjusted using fallback method: {p_axis_deg} (original: {p_axis_result}, HR: {estimated_hr:.0f} BPM)")
                                        else:
                                            print(f"⚠️ P axis value: {p_axis_deg} (may be less accurate at HR {estimated_hr:.0f} BPM)")
                                    else:
                                        # Last resort: use original value even if abnormal
                                        p_axis_deg = p_axis_result
                                        print(f"⚠️ P axis value: {p_axis_deg} (calculated at HR {estimated_hr:.0f} BPM, may be less accurate)")
                        else:
                            # If less than 2 P peaks detected, try to calculate with available peaks
                            if len(p_peaks) >= 1:
                                # Try with single peak (less accurate but better than "--")
                                p_axis_result_single = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks, fs, 20, 60)
                                if p_axis_result_single != "--":
                                    p_axis_deg = p_axis_result_single
                                    print(f"⚠️ P axis calculated with limited peaks: {p_axis_deg} (HR: {estimated_hr:.0f} BPM, may be less accurate)")
                            else:
                                # Last resort: try to estimate from R-peaks timing
                                # Use average PR interval assumption (150ms) to estimate P wave position
                                if len(r_peaks) >= 3:
                                    estimated_p_peaks = []
                                    for r in r_peaks[1:-1]:
                                        estimated_p_idx = max(0, r - int(0.15*fs))  # Assume 150ms PR interval
                                        if estimated_p_idx < len(lead_I_filt):
                                            estimated_p_peaks.append(estimated_p_idx)
                                    
                                    if len(estimated_p_peaks) >= 2:
                                        p_axis_result_est = calculate_wave_axis(lead_I_filt, lead_aVF_filt, estimated_p_peaks, fs, 20, 60)
                                        if p_axis_result_est != "--":
                                            p_axis_deg = p_axis_result_est
                                            print(f"⚠️ P axis estimated from R-peaks timing: {p_axis_deg} (HR: {estimated_hr:.0f} BPM, estimated)")
                        
                        # Detect T peaks (100-300ms after R peaks)
                        t_peaks = []
                        for r in r_peaks[1:-1]:  # Skip first and last
                            t_start = min(len(lead_aVF_filt), r + int(0.10*fs))
                            t_end = min(len(lead_aVF_filt), r + int(0.30*fs))
                            if t_end > t_start:
                                segment = lead_aVF_filt[t_start:t_end]
                                if len(segment) > 0:
                                    t_idx = t_start + np.argmax(segment)
                                    t_peaks.append(t_idx)
                        
                        if len(t_peaks) >= 2:
                            t_axis_result = calculate_wave_axis(lead_I_filt, lead_aVF_filt, t_peaks, fs, 40, 80)
                            if t_axis_result != "--":
                                t_axis_deg = t_axis_result
                        
                        print(f"🔬 Calculated P/QRS/T Axis: P={p_axis_deg}, QRS={qrs_axis_deg}, T={t_axis_deg}")
        except Exception as e:
            print(f"⚠️ Axis calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
    # Dummy values (remove real axis calculations for labels)
    p_axis_display = "--"
    qrs_axis_display = "--"
    t_axis_display = "--"
    p_mm = 0
    qrs_mm = 0
    t_mm = 0
    
    #P/QRS/T SECOND COLUMN - P/QRS/T Axis (ABOVE ECG GRAPH - same position)
    p_qrs_label = String(240, 740, f"P/QRS/T  : {p_axis_display}/{qrs_axis_display}/{t_axis_display}°",  # Changed to axis values
                         fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(p_qrs_label)

    # Dummy values (remove real RV5/SV1 calculations for labels)
    rv5_mv = 0.0
    sv1_mv = 0.0
    
    # SECOND COLUMN - RV5/SV1 (ABOVE ECG GRAPH - shifted further up)
    # Display SV1 as negative mV (GE/Hospital standard)
    # Use 3 decimal places for precision (not rounded to integers)
    rv5_sv_label = String(240, 720, f"RV5/SV1  : {rv5_mv:.3f} mV/{sv1_mv:.3f} mV",  # SV1 will show as negative
                          fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv_label)

    # Dummy value for RV5+SV1 label
    rv5_sv1_sum = 0.0
    
    # SECOND COLUMN - RV5+SV1 (ABOVE ECG GRAPH - shifted further up)
    # Use 3 decimal places for precision
    rv5_sv1_sum_label = String(240, 700, f"RV5+SV1 : {rv5_sv1_sum:.3f} mV",  # Moved up from 660 to 670
                               fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv1_sum_label)

    # Dummy value for QTCF label (remove real calculation)
    qtcf_val = None
    qtcf_text = "QTCF       : --"
    qtcf_label = String(240, 682, qtcf_text,  # Moved up from 642 to 652
                        fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_label)

    # SECOND COLUMN - Speed/Gain (merged in one line) (ABOVE ECG GRAPH - shifted further up)
    emg_setting = str(settings_manager.get_setting("filter_emg", "off")).strip()
    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
    ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
    ac_frequency = f"{ac_setting}Hz" if ac_setting in ("50", "60") else "Off"
    if dft_setting not in ("off", "") and emg_setting not in ("off", ""):
        filter_band = f"{dft_setting}-{emg_setting}Hz"
    elif dft_setting not in ("off", ""):
        filter_band = f"HP: {dft_setting}Hz"
    elif emg_setting not in ("off", ""):
        filter_band = f"LP: {emg_setting}Hz"
    else:
        filter_band = "Filter: Off"
    master_drawing.add(String(
        240,
        646,  # Moved up from 606 to 616
        f"{wave_speed_mm_s} mm/s   {filter_band}   AC : {ac_frequency}   {wave_gain_mm_mv} mm/mV",
        fontSize=10,
        fontName="Helvetica",
        fillColor=colors.black,
    ))

    


    
    from reportlab.pdfbase.pdfmetrics import stringWidth
    label_text = "Doctor Name: "
    
    # Value from Save ECG -> passed in 'patient'
    doctor = ""
    try:
        if patient:
            doctor = str(patient.get("doctor", "")).strip()
    except Exception:
        doctor = ""
  
    # Doctor Name (below V6 lead)
    doctor_name_label = String(-30, -10, "Doctor Name: ", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(doctor_name_label)
    
    if doctor:
        value_x = -30 + stringWidth("Doctor Name: ", "Helvetica-Bold", 10) + 5
        doctor_name_value = String(value_x, -10, doctor,
                                fontSize=10, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(doctor_name_value)

    # Doctor Signature (below Doctor Name)
    doctor_sign_label = String(-30, -25, "Doctor Sign: ", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(doctor_sign_label)

    # Add RIGHT-SIDE Conclusion Box (moved to the right) - NOW DYNAMIC FROM DASHBOARD (12 conclusions max) - MADE SMALLER
    # SHIFTED DOWN further (additional 5 points)
    conclusion_y_start = -9.  # Shifted down from 0 to -5 (5 more points down to shift container lower)
    
    # Create a rectangular box for conclusions (shifted right) - INCREASED HEIGHT (same position)
    # Height increased: bottom extended down (top position same). Length increased by 20 (x position fixed)
    # Rect already imported at top
    conclusion_box = Rect(200, conclusion_y_start - 55, 355, 75,  # Width 325→345 (+20); height 65→75 (+10)
                         fillColor=None, strokeColor=colors.black, strokeWidth=1.5)
    master_drawing.add(conclusion_box)
    
    # CENTERED and STYLISH "Conclusion" header - DYNAMIC - SMALLER (AT TOP OF CONTAINER - CLOSE TO TOP LINE)
    # Box center: 200 + (325/2) = 362.5, so text should be centered around 362.5
    # Box top is at conclusion_y_start - 55, so header should be very close to top line
    conclusion_header = String(362.5, conclusion_y_start + 8, "✦ CONCLUSION ✦",  # Moved very close to top line: y=0→-53 (just below top edge at -55)
                              fontSize=9, fontName="Helvetica-Bold",  # Reduced from 11 to 9
                              fillColor=colors.HexColor("#2c3e50"),
                              textAnchor="middle")  # This centers the text
    master_drawing.add(conclusion_header)
    
    # DYNAMIC conclusions from dashboard in the box - ONLY REAL CONCLUSIONS (no empty/---)
    # Split filtered conclusions into rows (2 conclusions per row) - COMPACT SPACING
    print(f"🎨 Drawing conclusions in graph from filtered list: {filtered_conclusions}")
    
    # Calculate how many rows we need based on actual conclusions
    num_conclusions = len(filtered_conclusions)
    num_rows = (num_conclusions + 1) // 2  # Round up division for rows
    
    # Split into rows (2 conclusions per row)
    conclusion_rows = []
    for i in range(0, num_conclusions, 2):
        row_conclusions = filtered_conclusions[i:i+2]
        conclusion_rows.append(row_conclusions)
    
    print(f"   Total conclusions: {num_conclusions}, Rows needed: {num_rows}")
    for idx, row in enumerate(conclusion_rows):
        print(f"   Row {idx+1}: {row}")
    
    # Draw conclusions row by row - ONLY REAL ONES with proper numbering
    row_spacing = 8  # Vertical spacing between rows
    start_y = conclusion_y_start - 10  # Starting Y position
    
    conclusion_num = 1  # Start numbering from 1
    for row_idx, row_conclusions in enumerate(conclusion_rows):
        row_y = start_y - (row_idx * row_spacing)
        
        for col_idx, conclusion in enumerate(row_conclusions):
            # Truncate long conclusions
            display_conclusion = conclusion[:30] + "..." if len(conclusion) > 30 else conclusion
            conc_text = f"{conclusion_num}. {display_conclusion}"
            
            # Position horizontally across the box (2 conclusions per row)
            x_pos = 210 + (col_idx * 160)  # 160 points spacing for 2 conclusions per row
            
            conc = String(x_pos, row_y, conc_text, 
                         fontSize=9, fontName="Helvetica", fillColor=colors.black)
            master_drawing.add(conc)
            
            conclusion_num += 1  # Increment for next conclusion

    print(f"✅ Added Patient Info, Vital Parameters, {len(filtered_conclusions)} REAL Conclusions (no empty/---), and Doctor Name/Signature to ECG grid")
    
    # STEP 5: Add SINGLE master drawing to story (NO containers)
    story.append(master_drawing)
    story.append(Spacer(1, 15))
    
    print(f" Added SINGLE master drawing with {successful_graphs}/12 ECG leads (ZERO containers)!")
    
    # Final summary
    if is_demo_mode:
        print(f"\n{'='*60}")
        print(f"📊 DEMO MODE REPORT SUMMARY:")
        print(f"   • Total leads processed: {successful_graphs}/12")
        print(f"   • Demo mode: {'ON' if is_demo_mode else 'OFF'}")
        if successful_graphs == 0:
            print(f"   ⚠️ WARNING: No ECG graphs were added to the report!")
            print(f"   💡 SOLUTION: Ensure demo is running for 5-10 seconds before generating report")
        elif successful_graphs < 12:
            print(f"   ⚠️ WARNING: Only {successful_graphs} graphs added (expected 12)")
        else:
            print(f"   ✅ SUCCESS: All 12 ECG graphs added successfully!")
        print(f"{'='*60}\n")

    # Measurement info (NO background)
    measurement_style = ParagraphStyle(
        'MeasurementStyle',
        fontSize=8,
        textColor=colors.HexColor("#000000"),
        alignment=1  # center
        # backColor removed
    )


    # Summary (NO background)
    summary_style = ParagraphStyle( 
        'SummaryStyle',
        fontSize=10,
        textColor=colors.HexColor("#000000"),
        alignment=1  # center
        # backColor removed
    )
    # summary_para = Paragraph(f"ECG Report: {successful_graphs}/12 leads displayed", summary_style)
    # story.append(summary_para)

    # Extract patient data for use in canvas drawing
    patient_org = patient.get("Org.", "") if patient else ""
    patient_doctor_mobile = patient.get("doctor_mobile", "") if patient else ""
    
    # Helper: draw logo on every page AND ALIGNED pink grid background on Page 2
    def _draw_logo_and_footer(canvas, doc):
        import os
        from reportlab.lib.units import mm
        
        # STEP 1: Draw FULL PAGE pink ECG grid background on Page 1 (ECG graphs page)
        if canvas.getPageNumber() == 1:  # Changed from 2 to 1 (Page 2 is now Page 1)
            page_width, page_height = canvas._pagesize
            
            # Fill entire page with pink background
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            
            # ECG grid colors - darker for better visibility
            light_grid_color = colors.HexColor("#ffd1d1")  
            
            major_grid_color = colors.HexColor("#ffb3b3")   
            
            # Draw minor grid lines (1mm spacing) - 59 boxes complete (0 to 295mm)
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.6)
            
            minor_spacing = 1 * mm
            
            # Vertical minor lines - Draw up to 295mm (includes 295mm line)
            max_x_limit = 59 * 5 * mm  # 295mm = right edge of 59th box
            x = 0
            while x <= max_x_limit:  # Draw lines 0 to 295mm (complete 59 boxes)
                canvas.line(x, 0, x, page_height)
                x += minor_spacing
                if x > max_x_limit:  # Stop immediately after 295mm
                    break
            
            # Horizontal minor lines - full page
            y = 0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += minor_spacing
                
            
            # Draw major grid lines - FULL PAGE
            # IMPORTANT: Match waveform calculation: height/3 = 15.0 points per box
            # For individual lead graphs: ecg_height = 45 points, so 15 points = 1 box
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(1.2)
            
            # Use standard ECG paper spacing: 5mm per box
            # 5mm = 5 * 2.834645669 points = 14.17 points per box
            from reportlab.lib.units import mm
            major_spacing = 5 * mm  # Standard ECG: 5mm = 14.17 points per box
            
            # Vertical major lines - Draw 60 lines (0, 5, 10...295mm) for 59 complete boxes
            max_x_limit = 59 * 5 * mm  # 295mm = right edge of 59th box
            x = 0
            while x <= max_x_limit:  # Include 295mm line (completes 59 boxes)
                canvas.line(x, 0, x, page_height)
                x += major_spacing
                if x > max_x_limit:  # Stop after 295mm
                    break
            
            # Horizontal major lines - STRICT: Only up to 295mm width (not full page_width)
            y = 0
            while y <= page_height:
                canvas.line(0, y, max_x_limit, y)  # End at 295mm (not page_width) ✅
                y += major_spacing
            

        
        # STEP 1.5: Draw Org. and Phone No. labels on Page 1 (TOP LEFT)
        if canvas.getPageNumber() == 1:
            canvas.saveState()
            
            # Position in top-left corner (below margin)
            x_pos = doc.leftMargin  # 30 points from left
            y_pos = doc.height + doc.bottomMargin - 5  # 20 points from top
            
            # Always draw "Org." label with value
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.black)
            org_label = "Org:"
            canvas.drawString(x_pos, y_pos, org_label)
            
            # Calculate width of label and add small gap
            org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica", 10)
            canvas.drawString(x_pos + org_label_width + 5, y_pos, patient_org if patient_org else "")
            
            y_pos -= 15  # Move down for next line
            
            # Always draw "Phone No." label with value
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.black)
            phone_label = "Phone No:"
            canvas.drawString(x_pos, y_pos, phone_label)
            
            # Calculate width of label and add small gap
            phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica", 10)
            canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient_doctor_mobile if patient_doctor_mobile else "")
            
            canvas.restoreState()
        
        # STEP 2: Draw logo on all pages (existing code)
        # Prefer PNG (ReportLab-friendly); fallback to WebP if PNG missing
        # Use resource_path helper for PyInstaller compatibility
        png_path = _get_resource_path("assets/Deckmountimg.png")
        webp_path = _get_resource_path("assets/Deckmount.webp")
        logo_path = png_path if os.path.exists(png_path) else webp_path

        if os.path.exists(logo_path):
            canvas.saveState()
            # Different positioning for different pages
            if canvas.getPageNumber() == 1:  # Changed from 2 to 1 (Page 2 is now Page 1)
                logo_w, logo_h = 120, 40  # bigger size for ECG page
                # SHIFTED LEFT FROM RIGHT TOP CORNER
                page_width, page_height = canvas._pagesize
                x = page_width - logo_w - 35  # Shifted 50 pixels left from right edge
                y = page_height - logo_h  # Top edge touch
            else:
                logo_w, logo_h = 120, 40  # normal size for other pages
                # For landscape mode, use page dimensions directly
                page_width, page_height = canvas._pagesize
                x = page_width - logo_w - 35  # Same positioning as Page 1
                y = page_height - logo_h  # Top edge touch
            try:
                canvas.drawImage(logo_path, x, y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')
            except Exception:
                # If WebP unsupported, silently skip
                pass
            canvas.restoreState()
        
        # STEP 3: Add footer with company address on all pages
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.black)  # Ensure text is black on pink background
        footer_text = "Deckmount Electronic , Plot No. 260, Phase IV, Udyog Vihar, Sector 18, Gurugram, Haryana 122015"
        # Center the footer text at bottom of page
        text_width = canvas.stringWidth(footer_text, "Helvetica", 8)
        x = (doc.width + doc.leftMargin + doc.rightMargin - text_width) / 2
        y = 10  # 20 points from bottom
        canvas.drawString(x, y, footer_text)
        canvas.restoreState()

    # Save parameters to a JSON index for later reuse
    try:
        from datetime import datetime
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        reports_dir = os.path.join(base_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        index_path = os.path.join(reports_dir, 'index.json')
        metrics_path = os.path.join(reports_dir, 'metrics.json')

        params_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "file": os.path.abspath(filename),
            "patient": {
                "name": full_name,
                "age": str(age),
                "gender": gender,
                "date_time": date_time_str,
            },
            "metrics": {
                "HR_bpm": HR,
                "PR_ms": PR,
                "QRS_ms": QRS,
                "QT_ms": QT,
                "QTc_ms": QTc,
                "ST_ms": ST,
                "RR_ms": RR,
                "RV5_plus_SV1_mV": round(rv5_sv1_sum, 3),
                "P_QRS_T_mm": [p_mm, qrs_mm, t_mm],
                "RV5_SV1_mV": [round(rv5_mv, 3), round(sv1_mv, 3)],
                "QTCF": round(qtcf_val, 1) if 'qtcf_val' in locals() and qtcf_val and qtcf_val > 0 else None,
            }
        }

        existing_list = []
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    existing_json = json.load(f)
                    if isinstance(existing_json, list):
                        existing_list = existing_json
                    elif isinstance(existing_json, dict) and isinstance(existing_json.get('entries'), list):
                        existing_list = existing_json['entries']
            except Exception:
                existing_list = []

        existing_list.append(params_entry)

        # Persist as a flat list for simplicity
        with open(index_path, 'w') as f:
            json.dump(existing_list, f, indent=2)
        print(f"✓ Saved parameters to {index_path}")

        # Save ONLY the 11 metrics in a lightweight separate JSON file (append to list)
        metrics_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "file": os.path.abspath(filename),
            "HR_bpm": HR,
            "PR_ms": PR,
            "QRS_ms": QRS,
            "QT_ms": QT,
            "QTc_ms": QTc,
            "ST_ms": ST,
            "RR_ms": RR,
            "RV5_plus_SV1_mV": round(rv5_sv1_sum, 3),
            "P_QRS_T_mm": [p_mm, qrs_mm, t_mm],
            "QTCF": None,  # Removed hardcoded constant - must be calculated from QT and RR
            "RV5_SV1_mV": [round(rv5_mv, 3), round(sv1_mv, 3)]
        }

        metrics_list = []
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    mj = json.load(f)
                    if isinstance(mj, list):
                        metrics_list = mj
            except Exception:
                metrics_list = []

        metrics_list.append(metrics_entry)

        with open(metrics_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        print(f"✓ Saved 11 metrics to {metrics_path}")
    except Exception as e:
        print(f"⚠️ Could not save parameters JSON: {e}")

    # Build PDF
    doc.build(story, onFirstPage=_draw_logo_and_footer, onLaterPages=_draw_logo_and_footer)
    print(f"✓ ECG Report generated: {filename}")
    
    # Upload to cloud if configured
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.cloud_uploader import get_cloud_uploader
        
        cloud_uploader = get_cloud_uploader()
        if cloud_uploader.is_configured():
            print(f"☁️  Uploading report to cloud ({cloud_uploader.cloud_service})...")
            
            # Prepare metadata
            upload_metadata = {
                "patient_name": data.get('patient', {}).get('name', 'Unknown'),
                "patient_age": str(data.get('patient', {}).get('age', '')),
                "report_date": data.get('date', ''),
                "machine_serial": data.get('machine_serial', ''),
                "heart_rate": str(data.get('Heart_Rate', '')),
            }
            
            # Upload the report
            result = cloud_uploader.upload_report(filename, metadata=upload_metadata)
            
            if result.get('status') == 'success':
                print(f"✓ Report uploaded successfully to {cloud_uploader.cloud_service}")
                if 'url' in result:
                    print(f"  URL: {result['url']}")
            else:
                print(f"⚠️  Cloud upload failed: {result.get('message', 'Unknown error')}")
        else:
            print("ℹ️  Cloud upload not configured (see cloud_config_template.txt)")
            
    except ImportError:
        print("ℹ️  Cloud uploader not available")
    except Exception as e:
        print(f"⚠️  Cloud upload error: {e}")


# ==================== HRV ECG REPORT GENERATION ====================
# COMPLETE ECG REPORT FORMAT - Same as generate_ecg_report() but with 5 one-minute Lead II graphs

def generate_hrv_ecg_report(filename="hrv_ecg_report.pdf", captured_data=None, data=None, patient=None, settings_manager=None, selected_lead="II"):
    """
    Generate HRV ECG report PDF with EXACT SAME format as main 12-lead ECG report
    Only difference: Page 2 shows 5 one-minute Lead II graphs in LANDSCAPE mode instead of 12 leads
    All Page 1 content, styling, formulas - EXACTLY SAME as generate_ecg_report()
    
    Parameters:
        filename: Output PDF filename
        captured_data: List of {'time': seconds, 'value': adc_value} dictionaries (5 minutes of selected lead)
        data: Metrics dictionary (HR, PR, QRS, etc.) - same format as main report
        patient: Patient details dictionary
        settings_manager: Settings manager for wave_speed, wave_gain, etc.
        selected_lead: The lead selected by the user (e.g., "I", "II", "V1")
    """
    
    if captured_data is None or len(captured_data) == 0:
        print(f"⚠️ No data provided for HRV ECG report (Lead {selected_lead})")
        return None
    
    # ==================== INITIALIZE (EXACT SAME AS MAIN REPORT) ====================
    
    if data is None:
        data = {
            "HR": 0, "beat": 0, "PR": 0, "QRS": 0, "QT": 0, "QTc": 0, "ST": 0,
            "HR_max": 0, "HR_min": 0, "HR_avg": 0, "Heart_Rate": 0, "QRS_axis": "--"
        }
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    if settings_manager is None:
        from utils.settings_manager import SettingsManager
        settings_manager = SettingsManager()
    
    def _safe_float(value, default):
        try:
            return float(value)
        except Exception:
            return default
    
    def _safe_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default
    
    # ==================== STEP 1: Get HR_bpm from metrics.json (PRIORITY) - SAME AS MAIN REPORT ====================
    latest_metrics = load_latest_metrics_entry(reports_dir)
    
    # If last entry has zero values, find last valid (non-zero) entry
    if latest_metrics and latest_metrics.get("HR_bpm", 0) == 0:
        try:
            metrics_path = os.path.join(reports_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    all_metrics = json.load(f)
                if isinstance(all_metrics, list) and len(all_metrics) > 0:
                    # Find last non-zero HR_bpm entry
                    for i in range(len(all_metrics)-1, -1, -1):
                        if all_metrics[i].get("HR_bpm", 0) > 0:
                            latest_metrics = all_metrics[i]
                            print(f"📊 HRV Report: Found last valid metric entry (index {i}) with HR_bpm > 0")
                            break
        except Exception as e:
            print(f"⚠️ Could not find last valid metric: {e}")
    
    hr_bpm_value = 0
    
    # Priority 1: metrics.json  latest HR_bpm
    if latest_metrics:
        hr_bpm_value = _safe_int(latest_metrics.get("HR_bpm"))
        if hr_bpm_value > 0:
            print(f"📊 HRV Report: Using HR_bpm from metrics.json: {hr_bpm_value} bpm (timestamp: {latest_metrics.get('timestamp', 'N/A')})")
    
    # Priority 2: Fallback to data parameter
    if hr_bpm_value == 0:
        hr_candidate = data.get("HR_bpm") or data.get("Heart_Rate") or data.get("HR")
        hr_bpm_value = _safe_int(hr_candidate)
        if hr_bpm_value > 0:
            print(f"📊 HRV Report: Using HR_bpm from data parameter: {hr_bpm_value} bpm")
    
    # Priority 3: Fallback to HR_avg
    if hr_bpm_value == 0 and data.get("HR_avg"):
        hr_bpm_value = _safe_int(data.get("HR_avg"))
        if hr_bpm_value > 0:
            print(f"📊 HRV Report: Using HR_bpm from HR_avg: {hr_bpm_value} bpm")
    
    # Save original HR_bpm from metrics.json (for reference - will NOT be changed)
    original_hr_bpm_from_metrics = hr_bpm_value  # This is from metrics.json (12-lead ECG)
    
    # Update data dictionary (SAME AS MAIN REPORT)
    data["HR_bpm"] = hr_bpm_value
    data["Heart_Rate"] = hr_bpm_value
    data["HR"] = hr_bpm_value
    if hr_bpm_value > 0:
        data["RR_ms"] = int(60000 / hr_bpm_value)
    else:
        data["RR_ms"] = data.get("RR_ms", 0)
    
    # ==================== STEP 2: Get ALL metrics from metrics.json (PRIORITY) - SAME AS MAIN REPORT ====================
    # Save original metrics.json values BEFORE they might get overwritten by HRV-calculated values
    # These will be used on Page 2 (ECG waves) to show metrics.json values
    original_metrics_from_json = {
        "HR": 0, "PR": 0, "QRS": 0, "QT": 0, "QTc": 0, "ST": 0, "RR_ms": 0
    }
    if latest_metrics:
        # ALWAYS save metrics.json values (for Page 2 - ECG waves)
        original_metrics_from_json["HR"] = _safe_int(latest_metrics.get("HR_bpm", 0))
        original_metrics_from_json["PR"] = _safe_int(latest_metrics.get("PR_ms", 0))
        original_metrics_from_json["QRS"] = _safe_int(latest_metrics.get("QRS_ms", 0))
        original_metrics_from_json["QT"] = _safe_int(latest_metrics.get("QT_ms", 0))
        original_metrics_from_json["QTc"] = _safe_int(latest_metrics.get("QTc_ms", 0))
        original_metrics_from_json["ST"] = _safe_int(latest_metrics.get("ST_ms", 0))
        original_metrics_from_json["RR_ms"] = _safe_int(latest_metrics.get("RR_ms", 0))
        if original_metrics_from_json["HR"] > 0 and original_metrics_from_json["RR_ms"] == 0:
            original_metrics_from_json["RR_ms"] = int(60000 / original_metrics_from_json["HR"])
        
        print(f"📊 HRV Report: Saved metrics.json values for Page 2 (ECG waves):")
        print(f"   HR={original_metrics_from_json['HR']}, PR={original_metrics_from_json['PR']}, QRS={original_metrics_from_json['QRS']}")
        print(f"   QT={original_metrics_from_json['QT']}, QTc={original_metrics_from_json['QTc']}, ST={original_metrics_from_json['ST']}, RR={original_metrics_from_json['RR_ms']}")
        
        # Get all available metrics from metrics.json (SAME AS MAIN REPORT)
        if data.get("PR", 0) == 0:
            data["PR"] = original_metrics_from_json["PR"]
        if data.get("QRS", 0) == 0:
            data["QRS"] = original_metrics_from_json["QRS"]
        if data.get("QT", 0) == 0:
            data["QT"] = original_metrics_from_json["QT"]
        if data.get("QTc", 0) == 0:
            data["QTc"] = original_metrics_from_json["QTc"]
        if data.get("ST", 0) == 0:
            data["ST"] = original_metrics_from_json["ST"]
        
        print(f"📊 HRV Report: Loaded metrics from metrics.json: HR={data['HR']}, PR={data['PR']}, QRS={data['QRS']}, QT={data['QT']}, QTc={data['QTc']}, ST={data['ST']}")
    
    # Update beat value for observation table (SAME AS MAIN REPORT)
    if data.get("beat", 0) == 0:
        data["beat"] = data["HR"]
    
    # Get settings (SAME AS MAIN REPORT)
    wave_speed_setting = settings_manager.get_setting("wave_speed", "25")
    wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
    wave_speed_mm_s = _safe_float(wave_speed_setting, 25.0)
    wave_gain_mm_mv = _safe_float(wave_gain_setting, 10.0)
    emg_setting = str(settings_manager.get_setting("filter_emg", "off")).strip()
    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
    ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
    ac_frequency = f"{ac_setting}Hz" if ac_setting in ("50", "60") else "Off"
    if dft_setting not in ("off", "") and emg_setting not in ("off", ""):
        filter_band = f"{dft_setting}-{emg_setting}Hz"
    elif dft_setting not in ("off", ""):
        filter_band = f"HP: {dft_setting}Hz"
    elif emg_setting not in ("off", ""):
        filter_band = f"LP: {emg_setting}Hz"
    else:
        filter_band = "Filter: Off"
    
    print(f"📊 HRV Report Settings: wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV")
    print(f"📊 HRV Report Final Metrics: HR={data['beat']}, PR={data['PR']}, QRS={data['QRS']}, QT={data['QT']}, QTc={data['QTc']}, ST={data['ST']}")
    
    # ==================== CALCULATE HEART RATE FROM 5 MINUTES (BEFORE REPORT GENERATION) ====================
    # Calculate average Heart Rate from 5 minutes of data (to use in report)
    hr_per_minute_for_report = []
    segment_duration = 11.0  # Same as ECG graphs: 11 seconds per strip
    import math
    total_duration_pre = max(d['time'] for d in captured_data) if captured_data else 0
    num_segments = max(1, min(5, int(math.ceil(total_duration_pre / 60.0))))
    
    # Helper function to calculate RR intervals from segment data
    def calculate_rr_from_segment_early(segment_data, sampling_rate=500.0):
        """Calculate ALL RR intervals from segment data by detecting R-peaks"""
        if len(segment_data) < 100:
            return None, None, []  # Return empty list for RR intervals
        
        try:
            from scipy.signal import find_peaks
            values = np.array([d['value'] for d in segment_data], dtype=float)
            if np.std(values) < 1e-6:
                return None, None, []
            
            values_norm = (values - np.mean(values)) / (np.std(values) + 1e-6)
            seg_duration = (segment_data[-1]['time'] - segment_data[0]['time']) if len(segment_data) > 1 else 60.0
            sr_dynamic = sampling_rate
            min_distance = int(0.25 * sr_dynamic)
            peaks, _ = find_peaks(values_norm, distance=min_distance, height=0.3)
            
            if len(peaks) < 2:
                return None, None, []
            
            rr_intervals = np.diff(peaks) * (1000.0 / sr_dynamic)
            rr_intervals = rr_intervals[(rr_intervals > 250) & (rr_intervals < 2000)]
            
            if len(rr_intervals) == 0:
                return None, None, []
            
            avg_rr = float(np.mean(rr_intervals))
            hr = 60000 / avg_rr if avg_rr > 0 else None
            return avg_rr, hr, rr_intervals.tolist()  # Return all RR intervals as list
            
        except Exception as e:
            return None, None, []
    
    # Get sampling rate from settings
    sampling_rate = 500.0
    
    # Calculate HR for each minute AND collect average RR per minute
    avg_rr_per_minute = []  # Collect average RR for each of the 5 minutes
    if captured_data and len(captured_data) > 100:
        for seg_idx in range(num_segments):
            minute_start = seg_idx * 60.0
            seg_start = minute_start
            seg_end = minute_start + 60.0
            seg_data = [d for d in captured_data if seg_start <= d['time'] < seg_end]
            
            if len(seg_data) > 100:
                avg_rr, hr_val, rr_intervals_list = calculate_rr_from_segment_early(seg_data, sampling_rate)
                if avg_rr is not None and hr_val is not None:
                    hr_per_minute_for_report.append(hr_val)
                    # Collect average RR for this minute (not all individual intervals)
                    avg_rr_per_minute.append(avg_rr)
                else:
                    hr_per_minute_for_report.append(data.get('HR_avg', ))
                    # Use fallback RR value
                    fallback_rr = 60000 / data.get('HR_avg', ) if data.get('HR_avg', ) > 0 else 750
                    avg_rr_per_minute.append(fallback_rr)
            else:
                hr_per_minute_for_report.append(data.get('HR_avg', ))
                # Use fallback RR value
                fallback_rr = 60000 / data.get('HR_avg', ) if data.get('HR_avg', ) > 0 else 750
                avg_rr_per_minute.append(fallback_rr)
    else:
        # Use default if no data
        hr_per_minute_for_report = [data.get('HR_avg', 80)] * 5
        fallback_rr = 60000 / data.get('HR_avg', 80) if data.get('HR_avg', ) > 0 else 750
        avg_rr_per_minute = [fallback_rr] * 5
    
    # Use available minutes only (no padding to 5)
    
    # NEW CALCULATION: HR = (N * 60000) / sum_of_avg_rr_per_minute
    # N = number of available minutes
    if len(avg_rr_per_minute) >= 1:
        sum_of_avg_rr = sum(avg_rr_per_minute)  # Sum of available average RR values
        minutes_count = len(avg_rr_per_minute)
        avg_hr_from_5_minutes = (minutes_count * 60000) / sum_of_avg_rr if sum_of_avg_rr > 0 else data.get('HR_avg')
        
        print(f"📊 HRV-Specific Heart Rate Calculation (NEW METHOD - 300000 / sum of avg RR per minute):")
        print(f"   Original HR_bpm from metrics.json (12-lead ECG): {original_hr_bpm_from_metrics} bpm (NOT CHANGED)")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   Average RR per minute: {[round(r, 1) for r in avg_rr_per_minute]} ms")
        print(f"   Sum of {minutes_count} average RR values: {sum_of_avg_rr:.2f} ms")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   NEW Formula: HR = ({minutes_count} * 60000) / sum_of_avg_rr_per_minute")
        print(f"   Calculation: ({minutes_count}*60000) / {sum_of_avg_rr:.2f} = {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   Per-minute HR values (for reference):")
        for i, hr_val in enumerate(hr_per_minute_for_report):
            print(f"   Min {i+1}: {hr_val:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   ✅ HRV-Specific BPM: {round(avg_hr_from_5_minutes)} bpm (WILL BE SAVED to metrics.json as HR_bpm)")
        print(f"   ✅ Original 12-lead ECG HR_bpm: {original_hr_bpm_from_metrics} bpm (saved as Original_HR_bpm for reference)\n")
    else:
        # Fallback to old method if no average RR values found
        if len(avg_rr_per_minute) >= 1:
            sum_of_avg_rr = sum(avg_rr_per_minute)
            minutes_count = len(avg_rr_per_minute)
            avg_hr_from_5_minutes = (minutes_count * 60000) / sum_of_avg_rr if sum_of_avg_rr > 0 else data.get('HR_avg', 80)
        else:
            avg_hr_from_5_minutes = np.mean(hr_per_minute_for_report) if len(hr_per_minute_for_report) > 0 else data.get('HR_avg', 80)
        print(f"⚠️ Using fallback method: {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   Original HR_bpm from metrics.json (12-lead ECG): {original_hr_bpm_from_metrics} bpm (NOT CHANGED)")
        print(f"   ─────────────────────────────────────────────────────────────")
        for i, hr_val in enumerate(hr_per_minute_for_report):
            print(f"   Min {i+1}: {hr_val:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   HRV Average HR (fallback): {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   Calculation: mean(per-minute HR values) = {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   ✅ HRV-Specific BPM: {round(avg_hr_from_5_minutes)} bpm (WILL BE SAVED to metrics.json as HR_bpm)")
        print(f"   ✅ Original 12-lead ECG HR_bpm: {original_hr_bpm_from_metrics} bpm (saved as Original_HR_bpm for reference)\n")
    
    # Save HRV-specific BPM separately (for Page 3 only)
    hrv_specific_bpm = round(avg_hr_from_5_minutes)
    
    # Use original HR_bpm from metrics.json (saved earlier, NOT to change it)
    # original_hr_bpm_from_metrics was already saved above from metrics.json
    
    # Keep data dictionary with metrics.json values for Page 1 and Page 2
    # Only Page 3 will use HRV-specific values
    # DO NOT overwrite data["HR_avg"], data["beat"] here - keep original metrics.json values
    
    # ==================== PAGE SETUP (MIXED: Page 1 Portrait, Page 2 Landscape) ====================
    
    # Patient org and phone for logo/footer callback
    patient_org = patient.get("Org.", "") if patient else ""
    patient_doctor_mobile = patient.get("doctor_mobile", "") if patient else ""
    
    # Define callback function for headers/footers BEFORE creating templates
    def _draw_logo_and_footer_callback(canvas, doc_obj):
        from reportlab.lib.units import mm
        
        # STEP 1: Draw pink ECG grid background ONLY on Page 1 (57 BOXES IN FULL 297MM WIDTH)
        if canvas.getPageNumber() == 1:  # Changed from 2 to 1 (Page 2 is now Page 1)
            page_width, page_height = canvas._pagesize
            
            # ========== 57 BOXES IN FULL 297MM PAGE WIDTH ==========
            # Page width: 297mm (full A4 landscape)
            # Number of boxes: 57
            # Box size: 297mm / 57 = 5.2105mm per box
            num_boxes_width = 57
            page_width_mm = 297.0
            box_width_mm = page_width_mm / num_boxes_width  # 297/57 = 5.2105mm per box
            box_width_pts = box_width_mm * mm
            
            # Pink background - FULL PAGE (297mm width, no white space)
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            
            # Grid colors
            light_grid_color = colors.HexColor("#ffd1d1")
            major_grid_color = colors.HexColor("#ffb3b3")
            
            # Minor grid lines - 5 minor boxes per major box (scaled proportionally)
            # Width: 57 boxes across 297mm → 5.2105mm per box → minor = 1.042mm
            # Height: 40 boxes across 210mm → 5.25mm per box → minor = 1.05mm
            minor_spacing_mm = box_width_mm / 5.0  # 1.042mm per minor division
            minor_spacing_pts = minor_spacing_mm * mm
            
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.6)  # Minor grid lines (1mm spacing) - keep original thickness
            
            # Vertical minor lines - full page width (297mm)
            x = 0
            while x <= page_width:
                canvas.line(x, 0, x, page_height)
                x += minor_spacing_pts
                if x > page_width:
                    break
            
            # Horizontal minor lines - 5 minor boxes per major box
            # Use proportional spacing to match 40 major boxes across 210mm height.
            num_boxes_height = 40
            page_height_mm = 210.0
            box_height_mm = page_height_mm / num_boxes_height  # 210/40 = 5.25mm per box
            minor_spacing_y = (box_height_mm / 5.0) * mm
            y = 0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += minor_spacing_y
            
            # Major grid lines - exactly 57 boxes across full 297mm width
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(0.6)  # Thinner major grid lines (5mm spacing) - was 1.2
            
            # Vertical major lines - 57 boxes (297mm width, 5.2105mm per box)
            x = 0
            for i in range(num_boxes_width + 1):  # 58 lines for 57 boxes
                canvas.line(x, 0, x, page_height)
                x += box_width_pts
            
            # Horizontal major lines - 40 boxes (210mm height, 5.25mm per box)
            box_height_pts = box_height_mm * mm
            y = 0
            for i in range(num_boxes_height + 1):  # 41 lines for 40 boxes
                canvas.line(0, y, page_width, y)
                y += box_height_pts
        
        # STEP 1.5: Draw Org. and Phone No. on Page 1 (POSITIONED BELOW DATE/TIME, NEAR LOGO)
        if canvas.getPageNumber() == 1:
            canvas.saveState()
            # Position on right side, below Date/Time labels, near logo position
            # Logo is at (page_width - 155, 565), Date/Time at (685, 540/525)
            x_pos = 685  # Same X as Date/Time labels (right side)
            
            # Org label: 15 points below Time label (525 - 15 = 510) + 15 points up = 525
            y_pos = 525
            
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.black)
            org_label = "Org:"
            canvas.drawString(x_pos, y_pos, org_label)
            
            org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica", 10)
            canvas.drawString(x_pos + org_label_width + 5, y_pos, patient_org if patient_org else "")
            
            # Phone No label: 15 points below Org label (525 + 15 = 540)
            y_pos = 540
            
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.black)
            phone_label = "Phone No:"
            canvas.drawString(x_pos, y_pos, phone_label)
            
            phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica", 10)
            canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient_doctor_mobile if patient_doctor_mobile else "")
            
            canvas.restoreState()
        
        # STEP 2: Draw logo (REPOSITIONED - lower from top)
        # Use resource_path helper for PyInstaller compatibility
        png_path = _get_resource_path("assets/Deckmountimg.png")
        webp_path = _get_resource_path("assets/Deckmount.webp")
        logo_path = png_path if os.path.exists(png_path) else webp_path
        
        if os.path.exists(logo_path):
            canvas.saveState()
            if canvas.getPageNumber() in [1, 2]:  # Changed from [2, 3] to [1, 2] (Page 2 is now Page 1)
                # Page 1 & 2 are LANDSCAPE - logo at top right
                logo_w, logo_h = 120, 40
                page_width, page_height = canvas._pagesize
                x = page_width - logo_w - 35
                y = page_height - logo_h + 10  # Shifted 10 points up from original position
            else:
                # Page 1 is PORTRAIT - position at top right (very close to top)
                logo_w, logo_h = 120, 40
                page_height = 842  # A4 portrait height
                x = 595 - logo_w - 30  # 595 = A4 width, 30 = right margin
                y = page_height - 35  # 35 points from top 
            try:
                canvas.drawImage(logo_path, x, y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
            canvas.restoreState()
        
        # STEP 3: Footer
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.black)
        footer_text = "Deckmount Electronic , Plot No. 260, Phase IV, Udyog Vihar, Sector 18, Gurugram, Haryana 122015"
        text_width = canvas.stringWidth(footer_text, "Helvetica", 8)
        
        if canvas.getPageNumber() in [1, 2]:  # Changed from [2, 3] to [1, 2] (Page 2 is now Page 1)
            # Page 1 & 2 are LANDSCAPE
            page_width, page_height = canvas._pagesize
            x = (page_width - text_width) / 2
        else:
            # Page 1 is PORTRAIT
            x = (595 - text_width) / 2
        
        y = 10
        canvas.drawString(x, y, footer_text)
        canvas.restoreState()
    
    # Create BaseDocTemplate for landscape pages only
    doc = BaseDocTemplate(filename, pagesize=landscape(A4),
                         rightMargin=20, leftMargin=20,
                         topMargin=20, bottomMargin=20)
    
    # Define Landscape template only (for all pages) with onPage callback
    landscape_width, landscape_height = landscape(A4)
    landscape_frame = Frame(20, 20,  # reduced margins for landscape to fit taller drawing
                           landscape_width - 40, landscape_height - 40,
                           id='landscape_frame')
    landscape_template = PageTemplate(id='landscape', frames=[landscape_frame], 
                                     pagesize=landscape(A4), onPage=_draw_logo_and_footer_callback)
    
    # Add only landscape template to document
    doc.addPageTemplates([landscape_template])
    story = []
    styles = getSampleStyleSheet()
    
    # HEADING STYLE (EXACT SAME AS MAIN REPORT)
    heading = ParagraphStyle(
        'Heading',
        fontSize=16,
        textColor=colors.HexColor("#000000"),
        spaceAfter=12,
        leading=20,
        alignment=1,
        bold=True
    )
    
    # ==================== SKIP PAGE 1 CONTENT - START DIRECTLY WITH LANDSCAPE PAGE 1 ====================
    # Original Page 1 content (Patient Details, Report Overview, Observation, Conclusion) has been removed
    # PDF will now start directly with ECG graphs (original Page 2) as Page 1
    
    # ==================== CALCULATE PATIENT DETAILS (NEEDED FOR ECG GRAPHS PAGE) ====================
    # Patient details are still needed for the ECG graphs page (now Page 1)
    if patient is None:
        patient = {}
    
    first_name = patient.get("first_name", "")
    last_name = patient.get("last_name", "")
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    date_time = patient.get("date_time", "")
    full_name = f"{first_name} {last_name}".strip()
    date_time_str = date_time
    
    # ==================== CALCULATE CONCLUSIONS (NEEDED FOR ECG GRAPHS PAGE) ====================
    # Conclusions are still needed for the conclusion box on the ECG graphs page (now Page 1)
    # Get conclusions from JSON (SAME LOGIC AS MAIN REPORT)
    dashboard_conclusions = get_dashboard_conclusions_from_image(None)
    
    # Filter conclusions (SAME AS MAIN REPORT)
    filtered_conclusions = []
    for conclusion in dashboard_conclusions:
        if conclusion and conclusion.strip() and conclusion.strip() != "---":
            filtered_conclusions.append(conclusion.strip())
            if len(filtered_conclusions) >= 12:
                break
    
    # If no real conclusions, use HRV-specific defaults
    if not filtered_conclusions:
        filtered_conclusions = [
            f"5-minute Lead {selected_lead} HRV analysis completed",
            "Heart rate variability recorded successfully"
        ]
    
    # ==================== PAGE 1: 5 ONE-MINUTE LEAD II GRAPHS (LANDSCAPE MODE) ====================
    
    print("📊 Creating SINGLE master drawing with 5 one-minute selected graphs in LANDSCAPE...")
    
    # Landscape A4 frame dimensions (landscape_width - 40, landscape_height - 40)
    # A4 landscape: 842 x 595, minus reduced margins (20 each side) = 802 x 555
    # This allows taller drawing to fit patient info at Y=528
    total_width = 780  # Fits in landscape frame (802 - margin)
    total_height = 540  # Increased to fit patient info at Y=528 comfortably (555 - margin)
    master_drawing = Drawing(total_width, total_height)
    
    # Define positions for 5 one-minute segments (LANDSCAPE MODE - POSITIONED FOR 540 HEIGHT)
    # Patient info/params are at 535-490, graphs should start well below that (< 490)
    # Graph height is 75. Drawing height = 540
    # First graph y=400 means top at 400+75=475, safely below settings at 490 (15pt gap)
    y_positions = [350, 270, 190, 110, 30]  # 5 graphs (80pt spacing) with proper spacing from params - SHIFTED DOWN 50 points total (25+25)
    
    # 🎯 Configuration: Fixed 6400 ADC samples per strip (device standard)
    samples_per_strip = 6400
    segment_duration = (samples_per_strip / float(sampling_rate)) if 'sampling_rate' in locals() else 11.0
    num_segments = 5  # Always 5 strips
    
    print(f"📊 HRV Report Configuration:")
    print(f"   Samples per strip: {samples_per_strip} ADC samples (sampling_rate={sampling_rate} Hz)")
    print(f"   Duration per strip: {segment_duration}s (first 11 seconds)")
    print(f"   Total strips: {num_segments}")
     
    # HRV report supports ONLY these leads (same as HRV UI lead combo)
    allowed_hrv_leads = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    # Lead-specific ADC per box multipliers (HRV report only)
    ADC_PER_BOX_CONFIG = {
        "V6": 6400.0,
        "V5": 6400.0,
        "V4": 6400.0,
        "V3": 6400.0,
        "V2": 6400.0,
        "V1": 6400.0,
        "II": 6400.0,
        "I": 6400.0,
    }
    def _get_adc_per_box_multiplier(lead_name, default=6400.0):
        return ADC_PER_BOX_CONFIG.get(lead_name, default)
    if selected_lead not in allowed_hrv_leads:
        print(f"⚠️ HRV lead '{selected_lead}' not supported, defaulting to Lead II")
        selected_lead = "II"
    adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(selected_lead, 6400.0)
    baseline_adc = 2000.0  #
    
    successful_graphs = 0
    
    for segment_idx in range(num_segments):
        # Calculate actual minute start times (0s, 60s, 120s, 180s, 240s)
        minute_start = segment_idx * 60.0  # Start of each minute
        segment_start = minute_start
        segment_end = minute_start + segment_duration  # First 10 seconds of this minute
        
        # Filter data for this segment (first 10 seconds of each minute)
        full_segment_data = [d for d in captured_data if segment_start <= d['time'] < segment_end]
        
        # LIMIT: Use only first N samples per strip based on sampling_rate
        if len(full_segment_data) > samples_per_strip:
            segment_data = full_segment_data[:samples_per_strip]
            print(f"📊 Strip {segment_idx + 1} ({int(segment_start)}s-{int(segment_end)}s): Plotting first {samples_per_strip} samples (trimmed from {len(full_segment_data)})")
        else:
            segment_data = full_segment_data
            print(f"📊 Strip {segment_idx + 1} ({int(segment_start)}s-{int(segment_end)}s): Plotting {len(segment_data)} samples (all available)")
        
        if len(segment_data) == 0:
            print(f"⚠️ No data for Strip {segment_idx + 1}")
            continue
        
        y_pos = y_positions[segment_idx]
        # 🎯 SHIFTED 2 BOXES LEFT: Plot now starts from 1mm (to prevent notch in box 2)
        # Grid calculation: 1 box = 5mm, 1mm = minor box
        from reportlab.lib.units import mm as mm_unit
        box_width = 5.0 * mm_unit  # 1 major box = 5mm = 14.173228 points
        minor_box_width = 1.0 * mm_unit  # 1 minor box = 1mm = 2.834646 points
        
        # NEW POSITION: Just 1mm from left edge (shifted 2 boxes left from previous 11mm)
        # Previous: 11mm (2 boxes + 1mm), Now: 1mm (no boxes, just 1mm offset)
        x_pos = minor_box_width  # Just 1mm from left = 2.83 points
        
        right_gap_boxes = 2  # 2 boxes gap from right
        left_gap_boxes = 0  # No full box gap
        
        # Width: Use maximum available - 1mm (left offset) - 2 boxes (right gap)
        # Total landscape width ≈ 59 boxes, so: 59 - 0.2 (1mm) - 2 (right) = 56.8 boxes
        plotting_boxes = 56.8  # Maximum plotting area
        ecg_width = plotting_boxes * box_width  # 56.8 boxes ≈ 804.6 points
        ecg_height = 75  # Fits 5 graphs in landscape height with 80pt spacing
        
        print(f"📐 Strip {segment_idx + 1}: 1mm from left (x={x_pos:.2f}), Width={ecg_width:.2f} (56.8 boxes), End={x_pos+ecg_width:.2f}")
        
        # ==================== ADD CALIBRATION NOTCH (FOR ALL STRIPS - ALWAYS) ====================
        # Notch appears at the beginning of each strip, regardless of data availability
        from reportlab.graphics.shapes import Path as PathShape
        
        # Calibration notch specifications:
        # Width: Fixed at 5mm (standard calibration pulse width)
        notch_width_mm = 5.0
        notch_width = notch_width_mm * mm_unit  # Convert to points
        
        # Height: Based on wave gain (1mV calibration signal)
        # Standard: 1mV should produce 10mm deflection at 10mm/mV gain
        # Formula: Height (mm) = 1mV × wave_gain_mm_mv
        notch_height_mm = 1.0 * wave_gain_mm_mv  # 1mV × gain
        notch_height = notch_height_mm * mm_unit  # Convert to points
        
        # Position: Shifted 1.5 boxes from left edge
        # X position: 1mm from left edge - 1.5 boxes shift
        notch_x = minor_box_width - (1.5 * box_width)  # 1mm - 1.5 boxes (7.5mm)
        
        # Y position: Center of the strip baseline - SHIFTED DOWN 25 points
        center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
        notch_y_base = center_y + 25  # Center baseline of the strip shifted down 25 points
        
        # Draw the calibration notch as a rectangular pulse
        notch_path = PathShape(
            fillColor=None,
            strokeColor=colors.HexColor("#000000"),
            strokeWidth=0.8,  # Thicker line for visibility
            strokeLineCap=1,
            strokeLineJoin=0  # Sharp corners for calibration box
        )
        
        # Draw calibration notch:
        # Start at baseline
        notch_path.moveTo(notch_x, notch_y_base)
        # Go up (calibration height)
        notch_path.lineTo(notch_x, notch_y_base + notch_height)
        # Go right (calibration width)
        notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
        # Go down back to baseline
        notch_path.lineTo(notch_x + notch_width, notch_y_base)
        # Connect notch end to ECG starting point (baseline continuation) - SHIFTED DOWN 25 points
        notch_path.lineTo(x_pos, center_y + 25)  # Draw line from notch end to ECG start shifted down
        
        # Add notch to drawing (ALWAYS, regardless of data)
        master_drawing.add(notch_path)
        
        # Calculate connection details for logging
        notch_end_x = notch_x + notch_width
        connection_distance_mm = (x_pos - notch_end_x) / mm_unit
        print(f"✅ Strip {segment_idx + 1}: Calibration notch added (ALWAYS)")
        print(f"   📏 Notch: Width={notch_width_mm}mm, Height={notch_height_mm}mm (1mV @ {wave_gain_mm_mv}mm/mV)")
        print(f"   🔗 Connection line: {connection_distance_mm:.2f}mm from notch end to ECG start")
        print(f"   📍 Positions: Notch@{notch_x:.2f}pt → ECG@{x_pos:.2f}pt")
        
        if len(segment_data) == 0:
            print(f"⚠️ No data for Strip {segment_idx + 1} - Calibration notch added anyway")
            continue
        
        try:
            # REMOVED: No segment labels
            # Plot starts from 1mm offset (just inside box 1) - prevents notch in box 2
            
            # Get ECG data for this segment
            if len(segment_data) > 0:
                values = np.array([d['value'] for d in segment_data], dtype=float)
                
                # Create time array for drawing
                t = np.linspace(x_pos, x_pos + ecg_width, len(values))
                
                # ========== EXACT SAME SCALING LOGIC AS MAIN REPORT (Lines 2665-2740) ==========
                
                # Step 1: Convert ADC data to numpy array (SAME AS MAIN REPORT)
                adc_data = np.array(values, dtype=float)
                
                # Step 1: Apply baseline correction for Lead II
                data_mean = np.mean(adc_data)
                baseline_adc = 2000.0
                
                if abs(data_mean - 2000.0) < 500:  # Data is close to baseline 2000 (raw ADC)
                    baseline_corrected = adc_data - baseline_adc
                else:
                    baseline_corrected = adc_data  # Already processed data
                
                # Step 2: FORCE CENTER for report - subtract mean to ensure perfect centering
                # IMPORTANT: Report me selected lead apni grid line ke center me dikhni chahiye
                # Chahe baseline wander kitna bhi ho (respiration mode, Fluke data, etc.)
                # This ensures waveform is exactly centered on grid line regardless of baseline wander
                centered_adc = baseline_corrected - np.mean(baseline_corrected)
                
                # Step 3: Calculate ADC per box (EXACT SAME AS MAIN REPORT line 2689-2714)
                # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
                # Selected Lead: {selected_lead} (from config)
                # Formula: ADC_per_box = adc_per_box_multiplier / wave_gain_mm_mv
                # For 10mm/mV with multiplier {adc_per_box_multiplier}: {adc_per_box_multiplier} / 10 = {adc_per_box_multiplier/10} ADC per box
                # This means: {adc_per_box_multiplier/10} ADC offset = 1 box (5mm) vertical movement
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
                
                # Step 4: Convert ADC offset to boxes (EXACT SAME AS MAIN REPORT line 2722)
                # Direct calculation: boxes_offset = centered_adc / adc_per_box
                boxes_offset = centered_adc / adc_per_box
                
                 
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                # IMPORTANT: Standard ECG paper uses 5mm per box
                # 5mm = 5 * 2.834645669 points = 14.17 points per box
                from reportlab.lib.units import mm as mm_unit
                box_height_points = 5.0 * mm_unit  # Standard ECG: 5mm = 14.17 points per box
                
                # Convert boxes offset to Y position (EXACT FORMULA FROM MAIN REPORT) - SHIFTED DOWN 25 points
                ecg_normalized = center_y + 25 + (boxes_offset * box_height_points)
                
                # Draw ECG waveform (EXACT SAME AS MAIN REPORT line 2240-2263)
                ecg_path = Path(fillColor=None, 
                               strokeColor=colors.HexColor("#000000"), 
                               strokeWidth=0.4,  # Same as main report
                               strokeLineCap=1,
                               strokeLineJoin=1)
                
                # Start path
                ecg_path.moveTo(t[0], ecg_normalized[0])
                
                # Add ALL points
                for i in range(1, len(t)):
                    ecg_path.lineTo(t[i], ecg_normalized[i])
                
                # Add path to master drawing
                master_drawing.add(ecg_path)
                
                successful_graphs += 1
                
                print(f" Drew {len(segment_data)} ECG data points for Strip {segment_idx + 1} ({int(segment_start)}s-{int(segment_end)}s, 6400 max samples)")
        
        except Exception as e:
            print(f" Error adding Strip {segment_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== ADD PATIENT INFO TO PAGE 2 (LANDSCAPE MODE - POSITIONED PROPERLY) ====================
    
    # LEFT SIDE: Patient Info (SHIFTED LEFT + UP)
    patient_name_label = String(-15, 545, f"Name: {full_name}",  # Shifted UP: 535 → 545
                                fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_name_label)
    
    patient_age_label = String(-15, 525, f"Age: {age}",  # Shifted UP: 515 → 525
                               fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_age_label)
    
    patient_gender_label = String(-15, 505, f"Gender: {gender}",  # Shifted UP: 495 → 505
                                  fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_gender_label)
    
    # RIGHT SIDE: Date/Time (shifted UP by 20 points - aligned with patient info)
    if date_time_str:
        parts = date_time_str.split()
        date_part = parts[0] if parts else ""
        time_part = parts[1] if len(parts) > 1 else ""
    else:
        date_part, time_part = "____", "____"
    
    date_label = String(655, 540, f"Date: {date_part}",  # Additional shift: RIGHT by 15 (670→685) and UP by 5 (535→540)
                       fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(date_label)
    
    time_label = String(655, 525, f"Time: {time_part}",  # Additional shift: RIGHT by 15 (670→685) and UP by 5 (500→525)
                       fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(time_label)
    
    # ==================== VITAL PARAMETERS (LANDSCAPE MODE - 2 COLUMNS SIDE BY SIDE) ====================
    # Page 1 vitals: Use arithmetic averages from per-minute values (chart-aligned)
    # RR_top = average of minute RR values
    # HR_top = average of minute HR values derived from RR (60000/RR_i)
    if 'original_metrics_from_json' in locals() and original_metrics_from_json:
        PR = original_metrics_from_json.get('PR', 0)
        QRS = original_metrics_from_json.get('QRS', 0)
        QT = original_metrics_from_json.get('QT', 0)
        QTc = original_metrics_from_json.get('QTc', 0)
        ST = original_metrics_from_json.get('ST', 0)
        
        if 'avg_rr_per_minute' in locals() and isinstance(avg_rr_per_minute, list) and len(avg_rr_per_minute) > 0:
            rr_count = len(avg_rr_per_minute)
            rr_avg_float = (sum(avg_rr_per_minute) / rr_count)
            rr_decimal = rr_avg_float - int(rr_avg_float)
            RR = int(rr_avg_float) + 1 if rr_decimal >= 0.50 else int(rr_avg_float)
            hr_values = [60000 / r for r in avg_rr_per_minute if r and r > 0]
            HR = int(round(sum(hr_values) / len(hr_values))) if len(hr_values) > 0 else original_metrics_from_json.get('HR', 0)
        else:
            HR = original_metrics_from_json.get('HR', 0)
            RR = original_metrics_from_json.get('RR_ms', int(60000 / HR) if HR else 0)
        
        print(f"📊 Page 1 (ECG waves): HR={HR} bpm (arithmetic avg of per-minute HR)")
        print(f"   RR={RR} ms (arithmetic avg of per-minute RR)")
        print(f"   Other metrics from metrics.json: PR={PR}, QRS={QRS}, QT={QT}, QTc={QTc}, ST={ST}")
    else:
        PR = data.get('PR', 0)
        QRS = data.get('QRS', 0)
        QT = data.get('QT', 0)
        QTc = data.get('QTc', 0)
        ST = data.get('ST', 0)
        if 'avg_rr_per_minute' in locals() and isinstance(avg_rr_per_minute, list) and len(avg_rr_per_minute) > 0:
            rr_count = len(avg_rr_per_minute)
            rr_avg_float = (sum(avg_rr_per_minute) / rr_count)
            rr_decimal = rr_avg_float - int(rr_avg_float)
            RR = int(rr_avg_float) + 1 if rr_decimal >= 0.50 else int(rr_avg_float)
            hr_values = [60000 / r for r in avg_rr_per_minute if r and r > 0]
            HR = int(round(sum(hr_values) / len(hr_values))) if len(hr_values) > 0 else data.get('HR_avg', 0)
        else:
            HR = data.get('HR_avg', 0)
            RR = int(60000 / HR) if HR and HR > 0 else 0
        print(f"⚠️ Page 1 (ECG waves): Using arithmetic averages - HR={HR} bpm, RR={RR} ms")
    
    # LEFT COLUMN (130, y) - HR, PR, QRS, RR (SHIFTED UP BY 20 POINTS)
    _add_label_column(
        master_drawing,
        205,
        [
            (555, f"HR    : {HR} bpm"),
            (540, f"PR    : {PR} ms"),
            (525, f"QRS : {QRS} ms"),
            (510, f"RR    : {RR} ms"),
            (495, f"QT     : {QT} ms"),
            (480, f"QTc   : {QTc} ms"),
        ],
    )
    
    st_label = String(350, 495, f"ST     : {ST} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(st_label)
    
    # ==================== ADDITIONAL LABELS (RIGHT COLUMN) ====================
    # P/QRS/T Axis
    p_axis = data.get('p_axis', '--').replace('°', '')
    qrs_axis = data.get('qrs_axis', '--').replace('°', '')
    t_axis = data.get('t_axis', '--').replace('°', '')
    p_qrs_label = String(350, 555, f"P/QRS/T  : {p_axis}/{qrs_axis}/{t_axis}°", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(p_qrs_label)
    
    # RV5/SV1
    rv5 = data.get('rv5', 0.0)
    sv1 = data.get('sv1', 0.0)
    rv5_sv_label = String(350, 540, f"RV5/SV1  : {rv5:.3f}/{sv1:.3f} mV", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv_label)
    
    # RV5+SV1
    rv5_sv1_sum = rv5 + abs(sv1)
    rv5_sv1_sum_label = String(350, 525, f"RV5+SV1 : {rv5_sv1_sum:.3f} mV", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv1_sum_label)
    
    # QTCF
    qtcf = data.get('QTc_Fridericia', 0)
    if qtcf and qtcf > 0:
        qtcf_text = f"QTCF       : {qtcf:.0f} ms"
    else:
        qtcf_text = "QTCF       : -- ms"
    qtcf_label = String(350, 510, qtcf_text, fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_label)

    
    master_drawing.add(String(
        350, 480,
        f"{wave_speed_mm_s} mm/s   {filter_band}   AC : {ac_frequency}   {wave_gain_mm_mv} mm/mV",
        fontSize=10, fontName="Helvetica", fillColor=colors.black
    ))
    
    # ==================== DOCTOR INFO (LANDSCAPE MODE - POSITIONED INSIDE DRAWING) ====================
    
    doctor = patient.get("doctor", "") if patient else ""
    label_text = "Doctor Name: "
    
    doctor_name_label = String(10, 50, "Doctor Name: ",  # X=10 (visible, inside drawing)
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(doctor_name_label)
    
    if doctor:
        value_x = 10 + stringWidth(label_text, "Helvetica-Bold", 10) + 6
        doctor_name_value = String(value_x, 50, doctor,  # Starts after "Doctor Name: " label
                                fontSize=10, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(doctor_name_value)
    
    doctor_sign_label = String(10, 35, "Doctor Sign: ",  # X=10 (visible, inside drawing)
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(doctor_sign_label)
    
    # ==================== CONCLUSION BOX ON PAGE 1 (LANDSCAPE MODE - ADJUSTED FOR 780 WIDTH) ====================
    
    conclusion_y_start = 45.0  # Positioned to avoid overlap with graphs
    conclusion_x_start = 280  # Shifted right for better positioning
    
    # Conclusion box (WIDER for landscape - adjusted for 780 total width)
    conclusion_box = Rect(conclusion_x_start, conclusion_y_start - 55, 490, 75,  # Bottom at -10, top at 65
                         fillColor=None, strokeColor=colors.black, strokeWidth=1.5)
    master_drawing.add(conclusion_box)
    
    # Conclusion header (CENTER adjusted for new X position and width)
    conclusion_header = String(conclusion_x_start + 245, conclusion_y_start + 8, "✦ CONCLUSION ✦",  # Center: 280 + 490/2 = 525
                              fontSize=9, fontName="Helvetica-Bold",
                              fillColor=colors.HexColor("#2c3e50"),
                              textAnchor="middle")
    master_drawing.add(conclusion_header)
    
    # Draw conclusions in the box (LANDSCAPE - adjusted for 520 width box)
    conclusion_rows = []
    for i in range(0, len(filtered_conclusions), 2):
        row_conclusions = filtered_conclusions[i:i+2]
        conclusion_rows.append(row_conclusions)
    
    row_spacing = 8

    
    start_y = conclusion_y_start - 10
    conclusion_num = 1
    
    for row_idx, row_conclusions in enumerate(conclusion_rows):
        row_y = start_y - (row_idx * row_spacing)
        for col_idx, conclusion in enumerate(row_conclusions):
            # More space in landscape, so can show longer conclusions
            display_conclusion = conclusion[:42] + "..." if len(conclusion) > 42 else conclusion
            conc_text = f"{conclusion_num}. {display_conclusion}"
            x_pos = conclusion_x_start + 10 + (col_idx * 230)  # Adjusted: start from box x + margin
            conc = String(x_pos, row_y, conc_text, 
                         fontSize=9, fontName="Helvetica", fillColor=colors.black)
            master_drawing.add(conc)
            conclusion_num += 1
    
    print(f"✅ Added Patient Info, Vital Parameters, {len(filtered_conclusions)} Conclusions to HRV report")
    
    # Add master drawing to story (NO spacer to avoid creating 3rd page)
    story.append(master_drawing)
    # story.append(Spacer(1, 15))  # REMOVED - was creating unwanted 3rd page
    
    print(f"📊 Added master drawing with {successful_graphs}/5 Lead II strips (each: 5,500 samples, first 10s of each minute)!")
    
    # ==================== PAGE 2: TIME DOMAIN & FREQUENCY DOMAIN ANALYSIS (LANDSCAPE MODE) ====================
    
    # Add page break for Page 2 (still landscape mode)
    story.append(PageBreak())
    
    print("📊 Creating Page 2: HRV Analysis (Time Domain + Frequency Domain)...")
    
    # NO PATIENT INFO - As per user request
    # NO DATE/TIME - As per user request
    # ONLY ANALYSIS GRAPHS IN SEPARATE CONTAINERS
    
    # Reduced spacing at top to shift Time Domain box upward (so Frequency fits on Page 2)
    story.append(Spacer(1, 15))  # Reduced from 40 to 15 to shift upward
    
    # ==================== TIME DOMAIN ANALYSIS CONTAINER ====================
    
    # Calculate per-minute RR intervals and HR for bar charts
    # 🎯 Minutes available based on captured duration
    import math
    segment_duration = 11.0  # Strip display remains 11s per strip
    total_duration = max(d['time'] for d in captured_data) if captured_data else 0
    num_segments = max(1, min(5, int(total_duration // 60.0) + (1 if total_duration % 60.0 >= 1.0 else 0)))
    
    rr_per_minute = []
    hr_per_minute = []
    
    # Helper function to calculate RR intervals from segment data
    def calculate_rr_from_segment(segment_data, sampling_rate=500.0):
        """Calculate ALL RR intervals from segment data by detecting R-peaks"""
        if len(segment_data) < 100:
            return None, None, []  # Return empty list for RR intervals
        
        try:
            from scipy.signal import find_peaks
            
            # Extract values from segment
            values = np.array([d['value'] for d in segment_data], dtype=float)
            
            # Normalize data for peak detection
            if np.std(values) < 1e-6:
                return None, None, []
            
            values_norm = (values - np.mean(values)) / (np.std(values) + 1e-6)
            
            # Improved R-peak detection with adaptive parameters
            # Minimum distance = 0.35 seconds (for HR up to 170 bpm, more sensitive)
            # This allows detection of higher heart rates
            segment_duration_sec = (segment_data[-1]['time'] - segment_data[0]['time']) if len(segment_data) > 1 else 60.0
            local_sr = sampling_rate
            min_distance = int(0.25 * local_sr)
            
            # Adaptive height threshold - use percentile-based approach for better detection
            height_threshold = np.percentile(values_norm, 75)  # Use 75th percentile as threshold
            if height_threshold < 0.2:
                height_threshold = 0.2  # Minimum threshold
            elif height_threshold > 0.5:
                height_threshold = 0.5  # Maximum threshold
            
            # Detect R-peaks with improved parameters
            peaks, properties = find_peaks(values_norm, 
                                         distance=min_distance, 
                                         height=height_threshold,
                                         prominence=0.15)  # Add prominence for better peak quality
            
            if len(peaks) < 2:
                # Try with lower threshold if no peaks found
                peaks, properties = find_peaks(values_norm, 
                                             distance=min_distance, 
                                             height=0.15,
                                             prominence=0.1)
            
            if len(peaks) < 2:
                return None, None, []
            
            # Calculate RR intervals from peaks (in milliseconds)
            rr_intervals = np.diff(peaks) * (1000.0 / local_sr)
            
            # Filter valid RR intervals (300ms to 2000ms) - wider range for accuracy
            # 300ms = 200 bpm max, 2000ms = 30 bpm min
            rr_intervals = rr_intervals[(rr_intervals > 250) & (rr_intervals < 2000)]
            
            # Additional filtering: Remove outliers (intervals that are > 2x or < 0.5x the median)
            if len(rr_intervals) > 2:
                median_rr = np.median(rr_intervals)
                rr_intervals = rr_intervals[(rr_intervals > 0.5 * median_rr) & (rr_intervals < 2.0 * median_rr)]
            
            if len(rr_intervals) == 0:
                return None, None, []
            
            # Calculate average RR interval (use median for robustness against outliers)
            avg_rr = float(np.median(rr_intervals))  # Median is more robust than mean
            mean_rr = float(np.mean(rr_intervals))  # Also calculate mean for comparison
            
            # Calculate HR from average RR (DYNAMIC: HR = 60000 / RR_interval_ms)
            hr = 60000 / avg_rr if avg_rr > 0 else None
            hr_from_mean = 60000 / mean_rr if mean_rr > 0 else None
            
            # Expected HR based on segment duration and number of peaks
            expected_hr_from_peaks = (len(peaks) / segment_duration_sec) * 60 if segment_duration_sec > 0 else None
            
            # Debug output to verify dynamic calculation
            print(f"      ✅ RR Interval Calculation (Improved Peak Detection):")
            print(f"         Segment duration: {segment_duration_sec:.2f} seconds")
            print(f"         Dynamic sampling rate: {local_sr:.1f} Hz")
            print(f"         Min peak distance: {min_distance/local_sr:.2f} s")
            print(f"         R-peaks detected: {len(peaks)}")
            print(f"         Expected HR from peak count: {expected_hr_from_peaks:.1f} bpm (if 11s segment)")
            print(f"         Valid RR intervals: {len(rr_intervals)}")
            print(f"         RR interval range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms")
            print(f"         Median RR: {avg_rr:.2f} ms → HR: {hr:.2f} bpm")
            print(f"         Mean RR: {mean_rr:.2f} ms → HR: {hr_from_mean:.2f} bpm")
            print(f"         Formula: HR = 60000 / RR_interval = 60000 / {avg_rr:.2f} = {hr:.2f} bpm")
            
            return avg_rr, hr, rr_intervals.tolist()  # Return all RR intervals as list
            
        except Exception as e:
            print(f"⚠️ Error calculating RR from segment: {e}")
            return None, None, []
    
    # Get sampling rate from settings or use default
    sampling_rate = 500.0
    
    # Collect average RR per minute for Page 3 (rr_per_minute already has these values)
    for seg_idx in range(num_segments):
        # Use minute-based starts: 0s, 60s, 120s, 180s, 240s (first 11s of each minute)
        minute_start = seg_idx * 60.0
        seg_start = minute_start
        seg_end = minute_start + 60.0
        seg_data = [d for d in captured_data if seg_start <= d['time'] < seg_end]
        
        if len(seg_data) > 100:
            # DYNAMIC: Calculate actual RR intervals from R-peaks in this segment
            avg_rr, hr_val, rr_intervals_list = calculate_rr_from_segment(seg_data, sampling_rate)
            
            if avg_rr is not None and hr_val is not None:
                # Verify Heart Rate calculation: HR = 60000 / RR_interval_ms
                calculated_hr = 60000 / avg_rr
                verification = "✅ CORRECT" if abs(calculated_hr - hr_val) < 0.1 else "⚠️ MISMATCH"
                
                rr_per_minute.append(avg_rr)
                hr_per_minute.append(hr_val)
                
                print(f"📊 Minute {seg_idx + 1}: ✅ DYNAMIC VALUES CALCULATED")
                print(f"   → Avg RR Interval: {avg_rr:.2f} ms (calculated from R-peaks)")
                print(f"   → Avg Heart Rate: {hr_val:.2f} bpm (calculated from RR interval)")
                print(f"   → Verification: HR = 60000 / {avg_rr:.2f} = {calculated_hr:.2f} bpm {verification}")
                print(f"   → Formula check: 60000 ÷ {avg_rr:.2f} ms = {calculated_hr:.2f} bpm (Displayed: {hr_val:.2f} bpm)")
                print(f"   → Data samples: {len(seg_data)}")
                print(f"   → RR intervals in this segment: {len(rr_intervals_list)}")
                print(f"   → NOTE: If BPM changes in this minute, it WILL be detected!")
            else:
                # Fallback: Use HR_avg if available
                hr_val = data.get('HR_avg', 80)
                rr_val = 60000 / hr_val if hr_val > 0 else 750
                rr_per_minute.append(rr_val)
                hr_per_minute.append(hr_val)
                print(f"⚠️ Minute {seg_idx + 1}: Using fallback HR={hr_val} bpm (insufficient peaks detected)")
        else:
            # Use default values if no data
            hr_val = data.get('HR_avg', 80)
            rr_val = 60000 / hr_val if hr_val > 0 else 750
            rr_per_minute.append(rr_val)
            hr_per_minute.append(hr_val)
            print(f"⚠️ Minute {seg_idx + 1}: Using default values (insufficient data)")
    
    # Do not pad to 5; use available minutes only
    
    # NEW CALCULATION for Page 2: HR = (N * 60000) / sum_of_avg_rr_per_minute
    # N = number of available minutes
    if len(rr_per_minute) >= 1:
        sum_of_avg_rr_page2 = sum(rr_per_minute)
        minutes_count_page2 = len(rr_per_minute)
        avg_hr_from_sum = (minutes_count_page2 * 60000) / sum_of_avg_rr_page2 if sum_of_avg_rr_page2 > 0 else data.get('HR_avg', 80)
        print(f"\n{'='*70}")
        print(f"📊 PAGE 2: NEW Average HR Calculation ((N*60000) / sum of avg RR per minute)")
        print(f"{'='*70}")
        print(f"   Average RR per minute: {[round(r, 1) for r in rr_per_minute]} ms")
        print(f"   Sum of {minutes_count_page2} average RR values: {sum_of_avg_rr_page2:.2f} ms")
        print(f"   NEW Formula: HR = ({minutes_count_page2}*60000) / sum_of_avg_rr_per_minute")
        print(f"   Calculation: ({minutes_count_page2}*60000) / {sum_of_avg_rr_page2:.2f} = {avg_hr_from_sum:.2f} bpm")
        print(f"{'='*70}\n")
    
    # Print summary of all calculated values with verification
    print(f"\n{'='*70}")
    print(f"📊 DYNAMIC RR INTERVAL & HEART RATE SUMMARY WITH VERIFICATION")
    print(f"{'='*70}")
    print(f"{'Min':<6} {'RR (ms)':<12} {'HR (bpm)':<12} {'Formula Check':<25} {'Status'}")
    print(f"{'-'*70}")
    for i in range(len(rr_per_minute)):
        rr_val = rr_per_minute[i]
        hr_val = hr_per_minute[i]
        # Verify: HR should equal 60000 / RR_interval
        calculated_hr = 60000 / rr_val if rr_val > 0 else 0
        diff = abs(calculated_hr - hr_val)
        verification = "✅ CORRECT" if diff < 0.5 else f"⚠️ DIFF: {diff:.2f}"
        status = "✅ Dynamic" if rr_val != 750 or hr_val != 80 else "⚠️ Default"
        
        formula_text = f"60000/{rr_val:.0f}={calculated_hr:.1f}"
        print(f"Min {i+1:<4} {rr_val:<12.2f} {hr_val:<12.2f} {formula_text:<25} {verification} ({status})")
    print(f"{'='*70}")
    print(f"✅ Formula: Heart Rate (bpm) = 60000 / RR_Interval (ms)")
    print(f"✅ If BPM changes between minutes, values WILL be different!")
    if len(rr_per_minute) >= 5:
        sum_of_avg_rr_page2 = sum(rr_per_minute[:5])
        if sum_of_avg_rr_page2 > 0:
            avg_hr_from_sum = 300000 / sum_of_avg_rr_page2
            print(f"✅ NEW: Overall Average HR = 300000 / sum_of_avg_rr = 300000 / {sum_of_avg_rr_page2:.2f} = {avg_hr_from_sum:.2f} bpm")
    print(f"✅ Note: Report Heart Rate already updated from 5-minute average (calculated earlier)\n")
    
    # Create TIME DOMAIN heading style (will be placed inside grey container)
    time_domain_style = ParagraphStyle(
        'TimeDomainStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
        alignment=0  # Left align
    )
    
    # Create matplotlib bar charts for Time Domain
    from datetime import datetime
    
    # Create reports/hrv_charts directory for saving charts
    hrv_charts_dir = os.path.join(reports_dir, 'hrv_charts')
    os.makedirs(hrv_charts_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Chart 1: Avg. RR Interval per minute (Bar Chart) - Standard professional colors
    fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))
    minutes = [f"Min {i+1}" for i in range(5)]
    rr_source = avg_rr_per_minute if ('avg_rr_per_minute' in locals() and isinstance(avg_rr_per_minute, list)) else []
    rr_values_plot = [(rr_source[i] if i < len(rr_source) else 0) for i in range(5)]
    rr_colors = [('#6497b1' if i < len(rr_source) else (0,0,0,0)) for i in range(5)]
    rr_edges = [('#6497b1' if i < len(rr_source) else (0,0,0,0)) for i in range(5)]
    x_pos = np.arange(5)
    bars1 = ax1.bar(x_pos, rr_values_plot, width=0.6, color=rr_colors, edgecolor=rr_edges, linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(minutes)
    ax1.set_xlabel('Minutes', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Milliseconds', fontsize=10, fontweight='bold')
    ax1.set_title('Avg. RR Interval per minute', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    rr_max = max(rr_values_plot) if len(rr_values_plot) > 0 else 0
    rr_upper = int(np.ceil(rr_max * 1.10)) if rr_max > 0 else 1000
    rr_upper = max(800, min(1200, rr_upper))
    ax1.set_ylim(0, rr_upper)
    
    # Add value labels above each bar (like in reference image)
    for i, bar in enumerate(bars1):
        if i < len(rr_source):
            value = rr_source[i]
            height = bar.get_height()
            rr_dec = value - int(value)
            rr_disp = int(value) + 1 if rr_dec >= 0.50 else int(value)
            label_text = f"{rr_disp}"
            label_y = min(height + (rr_upper * 0.02), rr_upper - 6)
            ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                    label_text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=True)
    
    print(f"📊 RR Interval Chart - Values above bars (custom rounding 0.50 threshold):")
    for i in range(len(rr_per_minute)):
        val = rr_per_minute[i]
        rr_dec = val - int(val)
        rr_disp = int(val) + 1 if rr_dec >= 0.50 else int(val)
        print(f"   Min {i+1}: {rr_disp} ms (original: {val:.2f} ms)")
    
    # Save chart 1 to reports directory
    temp_rr_chart_path = os.path.join(hrv_charts_dir, f'rr_interval_{timestamp}.png')
    fig1.savefig(temp_rr_chart_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Chart 2: Avg. Heart Rate per minute (Bar Chart) - Standard professional colors
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
    hr_values_plot = [(60000 / rr_source[i] if i < len(rr_source) and rr_source[i] > 0 else 0) for i in range(5)]
    hr_colors = [('#6497b1' if i < len(rr_source) else (0,0,0,0)) for i in range(5)]
    hr_edges = [('#6497b1' if i < len(rr_source) else (0,0,0,0)) for i in range(5)]
    x_pos2 = np.arange(5)
    bars2 = ax2.bar(x_pos2, hr_values_plot, width=0.6, color=hr_colors, edgecolor=hr_edges, linewidth=1.5)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(minutes)
    ax2.set_xlabel('Minutes', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Beats per minute', fontsize=10, fontweight='bold')
    ax2.set_title('Avg. Heart Rate per minute', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    hr_max = max(hr_values_plot) if len(hr_values_plot) > 0 else 0
    hr_upper = int(np.ceil(hr_max * 1.10)) if hr_max > 0 else 120
    hr_upper = max(80, min(220, hr_upper))
    ax2.set_ylim(0, hr_upper)
    
    # Add value labels above each bar (like in reference image)
    # IMPORTANT: Recalculate HR from rounded RR to ensure consistency
    # If RR displays as 569 ms, HR should be calculated from 569, not from 568.5
    for i, bar in enumerate(bars2):
        if i < len(rr_source):
            value = hr_values_plot[i]
            height = bar.get_height()
            rounded_rr = round(rr_source[i])
            recalculated_hr = 60000 / rounded_rr if rounded_rr > 0 else value
            decimal_part = recalculated_hr - int(recalculated_hr)
            if decimal_part < 0.50:
                hr_display = int(recalculated_hr)
            else:
                hr_display = int(recalculated_hr) + 1
            label_text = f"{hr_display}"
            label_y = min(height + (hr_upper * 0.02), hr_upper - 6)
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    label_text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=True)
    
    print(f"📊 Heart Rate Chart - Values above bars (recalculated from rounded RR with custom rounding):")
    for i in range(len(rr_source)):
        rounded_rr = round(rr_source[i])
        recalculated_hr = 60000 / rounded_rr if rounded_rr > 0 else 0
        decimal_part = recalculated_hr - int(recalculated_hr)
        if decimal_part < 0.50:
            hr_display = int(recalculated_hr)
        else:
            hr_display = int(recalculated_hr) + 1
        rounding_note = "rounded down" if decimal_part < 0.50 else "rounded up"
        print(f"   Min {i+1}: RR={rounded_rr} ms → HR={recalculated_hr:.4f} bpm → {hr_display} bpm ({rounding_note}, decimal={decimal_part:.2f})")
    
    # Save chart 2 to reports directory
    temp_hr_chart_path = os.path.join(hrv_charts_dir, f'heart_rate_{timestamp}.png')
    fig2.savefig(temp_hr_chart_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # Chart 3: HRV Metrics Radar Chart (like Spandan report)
    # Calculate HRV metrics for radar chart
    from scipy import signal
    
    # Extract Lead II values for HRV calculation
    lead_ii_values = np.array([d['value'] for d in captured_data]) if captured_data else np.array([])
    
    # Initialize rr_intervals_calc and average_nn_intervals for later use in saving metrics
    rr_intervals_calc = None
    average_nn_intervals = None
    sdann = None  # SDANN: Standard Deviation of Average NN intervals
    
    if len(lead_ii_values) > 100:
        # Detect R-peaks using simple peak detection
        from scipy.signal import find_peaks
        # Normalize data
        lead_ii_norm = (lead_ii_values - np.mean(lead_ii_values)) / (np.std(lead_ii_values) + 1e-6)
        # Find peaks (R-peaks)
        peaks, _ = find_peaks(lead_ii_norm, distance=50, height=0.5)
        
        # Calculate RR intervals from peaks
        if len(peaks) > 1:
            rr_intervals_calc = np.diff(peaks) * (1000.0 / 500.0)  # Convert to ms (assuming 500 Hz sampling)
            rr_intervals_calc = rr_intervals_calc[(rr_intervals_calc > 300) & (rr_intervals_calc < 2000)]
            
            if len(rr_intervals_calc) > 2:
                # Calculate average_nn_intervals (mean of all RR intervals)
                average_nn_intervals = np.mean(rr_intervals_calc)
                
                # Calculate SDNN: Standard deviation of NN intervals
                # Method 1: Direct std of RR intervals (standard method)
                sdnn = float(np.std(rr_intervals_calc))
                
                # Calculate SDANN: Standard Deviation of Average NN intervals
                # SDANN is calculated by dividing RR intervals into segments,
                # calculating average for each segment, then std of those averages
                # For 5-minute recording: use 1-minute segments (5 segments)
                segment_length = max(1, len(rr_intervals_calc) // 5)  # Divide into 5 segments
                segment_averages = []
                for i in range(0, len(rr_intervals_calc), segment_length):
                    segment = rr_intervals_calc[i:i + segment_length]
                    if len(segment) > 0:
                        segment_avg = np.mean(segment)
                        segment_averages.append(segment_avg)
                
                if len(segment_averages) > 1:
                    sdann = float(np.std(segment_averages))
                else:
                    # If not enough segments, use alternative: SDANN = SDNN / sqrt(number_of_intervals)
                    # Or use a simpler approach
                    sdann = sdnn / np.sqrt(len(rr_intervals_calc)) if len(rr_intervals_calc) > 0 else 0.0
                
                rmssd = float(np.sqrt(np.mean(np.diff(rr_intervals_calc)**2)))
                nn50_count = np.sum(np.abs(np.diff(rr_intervals_calc)) > 50)
                pnn50 = (nn50_count / len(rr_intervals_calc)) * 100 if len(rr_intervals_calc) > 0 else 0
                mean_hr_calc = 60000 / average_nn_intervals if average_nn_intervals > 0 else 80
            else:
                sdnn, rmssd, nn50_count, pnn50, mean_hr_calc = 0.01, 0.22, 0, 0.0, 80
                rr_intervals_calc = None
                average_nn_intervals = None
                sdann = None
        else:
            sdnn, rmssd, nn50_count, pnn50, mean_hr_calc = 0.01, 0.22, 0, 0.0, 80
            rr_intervals_calc = None
            average_nn_intervals = None
            sdann = None
    else:
        # Default values if no data
        sdnn, rmssd, nn50_count, pnn50, mean_hr_calc = 0.01, 0.22, 0, 0.0, 80
        rr_intervals_calc = None
        average_nn_intervals = None
        sdann = None
    
    # ==================== SAVE HRV METRICS TO TEXT FILE (APPEND TO SINGLE FILE) ====================
    try:
        from datetime import datetime
        
        # Delete all old HRV metrics files first
        import glob
        old_metrics_files = glob.glob(os.path.join(reports_dir, 'hrv_metrics_*.txt'))
        for old_file in old_metrics_files:
            try:
                os.remove(old_file)
                print(f"🗑️  Deleted old metrics file: {os.path.basename(old_file)}")
            except Exception as e:
                print(f"⚠️  Could not delete {old_file}: {e}")
        
        # Use single file name (no timestamp - will append to same file)
        hrv_metrics_file = os.path.join(reports_dir, 'hrv_metrics.txt')
        
        # Append mode - add new entry to existing file
        # Add separator if file already exists
        file_exists = os.path.exists(hrv_metrics_file)
        with open(hrv_metrics_file, 'a') as f:  # Changed from 'w' to 'a' for append
            if file_exists:
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("=" * 50 + "\n")  # Double separator for new entry
            f.write("=" * 50 + "\n")
            f.write("HRV Metrics Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")
            # Calculate and save average_nn_intervals first
            if average_nn_intervals is not None:
                f.write(f"Average NN Intervals (mean RR interval): {average_nn_intervals:.2f} ms\n")
                f.write(f"   Formula: average_nn_intervals = mean(RR_intervals)\n")
            f.write(f"SDNN (Standard Deviation of NN intervals): {sdnn:.2f} ms\n")
            f.write(f"   Formula: SDNN = std(RR_intervals)\n")
            if sdann is not None:
                f.write(f"SDANN (Standard Deviation of Average NN intervals): {sdann:.2f} ms\n")
                f.write(f"   Formula: SDANN = std(mean_segment_intervals)\n")
            f.write(f"RMSSD (Root Mean Square of Successive Differences): {rmssd:.2f} ms\n")
            f.write(f"NN50 (Number of NN intervals > 50ms different): {int(nn50_count)}\n")
            f.write(f"pNN50 (Percentage of NN50): {pnn50:.2f}%\n")
            f.write(f"Mean Heart Rate: {mean_hr_calc:.2f} bpm\n")
            f.write("-" * 50 + "\n")
            
            # Add calculation details
            if len(lead_ii_values) > 100:
                f.write(f"\nCalculation Details:\n")
                f.write(f"Total samples: {len(lead_ii_values)}\n")
                # Check if rr_intervals_calc is available and has valid data
                if rr_intervals_calc is not None and len(rr_intervals_calc) > 2:
                    f.write(f"RR intervals used: {len(rr_intervals_calc)}\n")
                    if average_nn_intervals is not None: 
                        f.write(f"Average NN Intervals: {average_nn_intervals:.2f} ms\n")
                    f.write(f"Min RR interval: {np.min(rr_intervals_calc):.2f} ms\n")
                    f.write(f"Max RR interval: {np.max(rr_intervals_calc):.2f} ms\n")
                    f.write(f"RR interval range: {np.max(rr_intervals_calc) - np.min(rr_intervals_calc):.2f} ms\n")
                    # SDANN verification
                    if sdann is not None:
                        f.write(f"\nSDANN Calculation Details:\n")
                        segment_length = max(1, len(rr_intervals_calc) // 5)
                        num_segments = len(rr_intervals_calc) // segment_length
                        f.write(f"   Segments used: {num_segments}\n")
                        f.write(f"   Segment length: {segment_length} intervals per segment\n")
                        f.write(f"   SDANN = std(segment_averages) = {sdann:.2f} ms ✓\n")
                else:
                    f.write("RR intervals: Not available (insufficient data)\n")
            else:
                f.write(f"\nTotal samples: {len(lead_ii_values)} (insufficient for calculation)\n")
        
        print(f" HRV metrics saved to: {hrv_metrics_file}")
        if average_nn_intervals is not None:
            print(f"   Average NN Intervals: {average_nn_intervals:.2f} ms")
        print(f"   SDNN: {sdnn:.2f} ms")
        if sdann is not None:
            print(f"   SDANN: {sdann:.2f} ms")
        print(f"   RMSSD: {rmssd:.2f} ms")
        print(f"   NN50: {int(nn50_count)}")
        print(f"   pNN50: {pnn50:.2f}%")
        
    except Exception as e:
        print(f"⚠️ Error saving HRV metrics to file: {e}")
        import traceback
        traceback.print_exc()
    
    # Create Radar Chart - Standard professional colors
    fig_radar, ax_radar = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))  # Increased size
    
    # Radar chart parameters (normalize to 0-100 scale for display)
    
    categories = ['SDANN', 'NN50', 'RMSSD', 'SDNN', 'BPM']
    
    # Calculate SDANN normalized value (if available)
    sdann_normalized = min((sdann / 2.0) if sdann is not None and sdann > 0 else 0, 100)
    
    values = [
        sdann_normalized,  # SDANN (where SDNN was) - Normalize (0-200ms → 0-100)
        min(nn50_count, 100),  # NN50 (where RMSSD was) - count
        min(rmssd / 2.0, 100),  # RMSSD (where NN50 was) - Normalize (0-200ms → 0-100)
        min(sdnn / 2.0, 100),  # SDNN (where pNN50 was) - Normalize (0-200ms → 0-100)
        min(mean_hr_calc, 100)  # BPM (stays same) - cap at 100
    ]
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # Close the plot
    angles += angles[:1]
    
    # Plot - Standard report colors (professional green)
    ax_radar.plot(angles, values, 'o-', linewidth=2.5, color='#228B22', markersize=6)
    ax_radar.fill(angles, values, alpha=0.3, color='#32CD32')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=9)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title('Radar chart ', fontsize=12, fontweight='bold', pad=20)
    ax_radar.grid(True)
    
    # Save radar chart
    temp_radar_chart_path = os.path.join(hrv_charts_dir, f'radar_chart_{timestamp}.png')
    fig_radar.savefig(temp_radar_chart_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig_radar)
    
    # Create metrics table for SDANN, SDDN (SDNN), RMSSD, NN50 (displayed inside container)
    # Format values for display - USING DYNAMIC CALCULATED VALUES
    # Ensure we use actual calculated values, not defaults
    sdann_display = f"{sdann:.2f}" if sdann is not None and sdann > 0 else "0.00"
    sdnn_display = f"{sdnn:.2f}" if sdnn is not None and sdnn > 0 else "0.00"
    rmssd_display = f"{rmssd:.2f}" if rmssd is not None and rmssd > 0 else "0.00"
    nn50_display = f"{int(nn50_count)}" if nn50_count is not None and nn50_count >= 0 else "0"
    
    # Debug: Print calculated values to verify they're dynamic
    print(f"📊 HRV Metrics (Dynamic Values):")
    print(f"   SDANN: {sdann_display}, SDNN: {sdnn_display}, RMSSD: {rmssd_display}, NN50: {nn50_display}")
    
    # Create metrics in single row (like reference image) - More gap between metrics
    metrics_table_data = [
        [Paragraph(f"<b>SDANN:</b> {sdann_display}", styles['Normal']),
         Paragraph(f"<b>SDDN:</b> {sdnn_display}", styles['Normal']),
         Paragraph(f"<b>RMSDDN:</b> {rmssd_display}", styles['Normal']),
         Paragraph(f"<b>NN50:</b> {nn50_display}", styles['Normal'])]
    ]
    
    # Further increased column widths for MORE gap between metrics (to fill entire container width)
    metrics_table = Table(metrics_table_data, colWidths=[220, 220, 220, 220]) 
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.transparent),  # Transparent background
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0, colors.transparent),  # No grid
        ("BOX", (0,0), (-1,-1), 0, colors.transparent),  # No box
        ("LEFTPADDING", (0,0), (-1,-1), 15),  # Further increased left padding
        ("RIGHTPADDING", (0,0), (-1,-1), 15),  # Further increased right padding for MORE gap
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    
    # Add charts to story in a table (3 charts: RR, HR, Radar) with shadow container effect
    # Charts placed on top/in center of shadow containers - FURTHER INCREASED SIZE
    # Increased column widths for bigger charts to fill container properly
    time_domain_table = Table([
        [Image(temp_rr_chart_path, width=240, height=155),  # Further increased: 220x140 to 240x155
         Image(temp_hr_chart_path, width=240, height=155),  # Further increased: 220x140 to 240x155
         Image(temp_radar_chart_path, width=240, height=155)]  # Further increased: 220x140 to 240x155
    ], colWidths=[255, 255, 255])  # Further increased: 240 to 255 for bigger charts
    
    time_domain_table.setStyle(TableStyle([
        # Shadow container effect: Light background with border (cards/containers)
        # Each chart cell has its own shadow container
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#ffffff")),  # White background for chart 1 container
        ("BACKGROUND", (1,0), (1,-1), colors.HexColor("#ffffff")),  # White background for chart 2 container
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#ffffff")),  # White background for chart 3 container
        
        # Border effect (shadow-like): Subtle gray border creating shadow container
        ("BOX", (0,0), (0,-1), 1.5, colors.HexColor("#d0d0d0")),  # Shadow border for chart 1
        ("BOX", (1,0), (1,-1), 1.5, colors.HexColor("#d0d0d0")),  # Shadow border for chart 2
        ("BOX", (2,0), (2,-1), 1.5, colors.HexColor("#d0d0d0")),  # Shadow border for chart 3
        
        # Additional shadow effect: Inner border for depth
        ("INNERGRID", (0,0), (0,-1), 0.5, colors.HexColor("#e8e8e8")),  # Subtle inner border
        ("INNERGRID", (1,0), (1,-1), 0.5, colors.HexColor("#e8e8e8")),  # Subtle inner border
        ("INNERGRID", (2,0), (2,-1), 0.5, colors.HexColor("#e8e8e8")),  # Subtle inner border
        
        # Center alignment for charts within containers
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        
        # Proper padding to place charts on top/center of shadow containers (reduced for bigger charts)
        ("LEFTPADDING", (0,0), (-1,-1), 10),  # Further reduced padding for bigger charts
        ("RIGHTPADDING", (0,0), (-1,-1), 10),  # Further reduced padding for bigger charts
        ("TOPPADDING", (0,0), (-1,-1), 10),   # Further reduced top padding for bigger charts
        ("BOTTOMPADDING", (0,0), (-1,-1), 10), # Further reduced bottom padding for bigger charts
    ]))
    
    # Create grey container for entire Time Domain Analysis section
    # Heading OUTSIDE container, metrics and charts INSIDE container
    time_domain_heading = Paragraph("<b>Time Domain Analysis</b>", time_domain_style)
    
    # Create container table with metrics and charts (heading is outside)
    time_domain_container = Table([
        [metrics_table],        # Row 0: Metrics (SDANN, SDDN, RMSDDN, NN50)
        [time_domain_table]     # Row 1: Charts
    ], colWidths=[780])  # Full width for landscape page
    
    time_domain_container.setStyle(TableStyle([
        # Grey background container (darker grey like reference image)
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#e8e8e8")),  # Darker grey background
        
        # Border around entire container (thin black border like reference)
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#000000")),  # Black border
        
        # Padding inside container (reduced to make it compact so Frequency Domain fits on Page 2)
        ("LEFTPADDING", (0,0), (-1,-1), 12),  # Reduced from 15 to 12
        ("RIGHTPADDING", (0,0), (-1,-1), 12),  # Reduced from 15 to 12
        ("TOPPADDING", (0,0), (0,0), 6),   # Reduced from 8 to 6 for metrics row
        ("TOPPADDING", (0,1), (-1,-1), 6), # Reduced from 8 to 6 for charts row
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),  # Reduced from 15 to 12
        
        # Align metrics table to left (inside container)
        ("ALIGN", (0,0), (-1,0), "LEFT"),
        ("VALIGN", (0,0), (-1,0), "TOP"),
        
        # Align charts table to center (inside container)
        ("ALIGN", (0,1), (-1,-1), "CENTER"),
        ("VALIGN", (0,1), (-1,-1), "MIDDLE"),
    ]))
    
    # Add heading OUTSIDE container (before container)
    story.append(time_domain_heading)
    story.append(Spacer(1, 6))  # Reduced spacing (from 8 to 6) to make it compact
    
    story.append(time_domain_container)
    story.append(Spacer(1, 10))  # Further reduced spacing between sections (from 12 to 10) so Frequency Domain fits on Page 2
    
    # ==================== FREQUENCY DOMAIN ANALYSIS CONTAINER ====================
    
    # Create FREQUENCY DOMAIN heading style (will be placed outside grey container)
    freq_domain_style = ParagraphStyle(
        'FreqDomainStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
        alignment=0  # Left align
    )
    
    # Create Frequency Domain chart (further reduced size to fit on Page 2)
    # Frequency Domain Analysis: leave blank space (remove fake/placeholder graph)
    # Reserve the same space as the previous chart to keep layout stable.
    freq_domain_table = Table([
        [Spacer(1, 130)]
    ], colWidths=[440])  # Matches previous chart width
    
    freq_domain_table.setStyle(TableStyle([
        # NO BOX border - removed for cleaner look
        # NO BACKGROUND - removed for cleaner look
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    
    # Create grey container for entire Frequency Domain Analysis section
    # Heading OUTSIDE container, chart INSIDE container
    freq_domain_heading = Paragraph("<b>Frequency Domain Analysis</b>", freq_domain_style)
    
    # Create container table with chart only (heading is outside)
    freq_domain_container = Table([
        [freq_domain_table]  # Only chart inside container
    ], colWidths=[780])  # Full width for landscape page
    
    freq_domain_container.setStyle(TableStyle([
        # Grey background container (darker grey like reference image)
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#e8e8e8")),  # Darker grey background
        
        # Border around entire container (thin black border like reference)
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#000000")),  # Black border
        
        # Padding inside container (reduced to make it compact so it fits on Page 2)
        ("LEFTPADDING", (0,0), (-1,-1), 12),  # Reduced from 15 to 12
        ("RIGHTPADDING", (0,0), (-1,-1), 12),  # Reduced from 15 to 12
        ("TOPPADDING", (0,0), (-1,-1), 6),  # Reduced from 8 to 6 for chart row
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),  # Reduced from 15 to 12
        
        # Align chart to center (inside container)
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    
    # Add heading OUTSIDE container (before container)
    story.append(freq_domain_heading)
    story.append(Spacer(1, 6))  # Reduced spacing (from 8 to 6) to make it compact
    story.append(freq_domain_container)
    story.append(Spacer(1, 10))
    
    # Note: Chart files saved in reports/hrv_charts/ directory for reference
    # They can be cleaned up later if needed, but kept for now for debugging
    print(f"📊 HRV charts saved to: {hrv_charts_dir}")
    print(f"   - RR Interval chart: {temp_rr_chart_path}")
    print(f"   - Heart Rate chart: {temp_hr_chart_path}")
    print(f"   - Radar chart: {temp_radar_chart_path}")
    print("📊 Added Page 2: HRV Analysis with 3 charts (2 bar + 1 radar) and blank frequency space")
    
    # ==================== SAVE METRICS TO metrics.json (SAME AS MAIN ECG REPORT) ====================
    try:
        from datetime import datetime
        metrics_path = os.path.join(reports_dir, 'metrics.json')
        
        # Ensure HRV variables are defined (in case calculation failed)
        if 'sdnn' not in locals():
            sdnn = 0
        if 'rmssd' not in locals():
            rmssd = 0
        if 'nn50_count' not in locals():
            nn50_count = 0
        if 'pnn50' not in locals():
            pnn50 = 0
        if 'sdann' not in locals():
            sdann = None
        
        # Save metrics.json with HRV-specific HR_bpm (5 minutes average)
        # HR_bpm will be the HRV-specific average, which will be picked up for future reports
        hrv_hr_bpm = hrv_specific_bpm if 'hrv_specific_bpm' in locals() else 0
        hrv_rr_ms = int(60000 / hrv_hr_bpm) if hrv_hr_bpm > 0 else 0
        
        metrics_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "file": os.path.abspath(filename),
            "HR_bpm": hrv_hr_bpm,  # HRV-specific 5 minutes average (main value)
            "PR_ms": original_metrics_from_json.get("PR", 0),
            "QRS_ms": original_metrics_from_json.get("QRS", 0),
            "QT_ms": original_metrics_from_json.get("QT", 0),
            "QTc_ms": original_metrics_from_json.get("QTc", 0),
            "ST_ms": original_metrics_from_json.get("ST", 0),
            "RR_ms": hrv_rr_ms,  # Calculated from HRV-specific HR
            # Additional HRV-specific metrics
            "HRV_SDNN_ms": float(sdnn) if sdnn else 0,
            "HRV_RMSSD_ms": float(rmssd) if rmssd else 0,
            "HRV_NN50": int(nn50_count) if nn50_count else 0,
            "HRV_pNN50": float(pnn50) if pnn50 else 0,
            "HRV_SDANN_ms": float(sdann) if sdann else 0,
            # Original 12-lead ECG HR for reference
            "Original_HR_bpm": original_metrics_from_json.get("HR", 0),  # 12-lead ECG HR (for reference)
        }
        
        metrics_list = []
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    mj = json.load(f)
                    if isinstance(mj, list):
                        metrics_list = mj
            except Exception:
                metrics_list = []
        
        metrics_list.append(metrics_entry)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        print(f"✅ Saved metrics to {metrics_path}")
        print(f"   HR_bpm: {metrics_entry['HR_bpm']} bpm (HRV-specific, 5 minutes average) ← Main value")
        print(f"   RR_ms: {metrics_entry['RR_ms']} ms (calculated from HRV HR)")
        print(f"   Original_HR_bpm: {metrics_entry.get('Original_HR_bpm', 0)} bpm (12-lead ECG, for reference)")
        print(f"   ✅ This HR_bpm will be picked up for future reports")
    except Exception as e:
        print(f"⚠️ Could not save metrics to JSON: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== BUILD PDF (2 PAGES) ====================
    
    # Build PDF - callbacks are already attached to PageTemplates
    doc.build(story)
    print(f"✅ HRV ECG Report generated: {filename}")
    print(f"   📄 Page 1: 5 One-Minute Lead II ECG Graphs (Landscape)")
    print(f"   📄 Page 2: HRV Analysis - Time & Frequency Domain (Landscape)")
    
    return filename


# ==================== END OF HRV ECG REPORT GENERATION ====================
