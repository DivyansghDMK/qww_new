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

ECG_HEIGHT_MM = 210.0
ECG_WIDTH_MM = 297.0
ECG_LARGE_BOX_MM_HEIGHT = ECG_HEIGHT_MM / 40.0
ECG_LARGE_BOX_MM_WIDTH = ECG_WIDTH_MM / 56.0
ECG_SMALL_BOX_MM_HEIGHT = ECG_LARGE_BOX_MM_HEIGHT / 5.0
ECG_SMALL_BOX_MM_WIDTH = ECG_LARGE_BOX_MM_WIDTH / 5.0
ECG_BASE_BOX_MM = 5.0
ECG_SPEED_SCALE = ECG_LARGE_BOX_MM_HEIGHT / ECG_BASE_BOX_MM
ECG_BASELINE_ADC = 2000.0

def beats_in_boxes(bpm, boxes, mm_per_box=ECG_LARGE_BOX_MM_WIDTH, speed_mm_per_s=25.0):
    if bpm <= 0 or boxes <= 0:
        return 0.0
    t_box = mm_per_box / speed_mm_per_s
    return (bpm / 60.0) * (boxes * t_box)


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
        "sampling_rate": 500.0,  # Default, will be updated if available
        "leads": {}
    }
    
    # Get sampling rate if available
    if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            sampled_rate = float(ecg_test_page.sampler.sampling_rate)
            # Guard against invalid/low sampling rates (report expects 500 Hz)
            if sampled_rate < 50.0 or sampled_rate > 1000.0:
                sampled_rate = 500.0
            saved_data["sampling_rate"] = sampled_rate
    
    # Save each lead's data - use FULL buffer (ecg_buffers if available, otherwise data)
    # Priority: Use ecg_buffers (5000 samples) if available, otherwise use data (1000 samples)
    
    # Debug: Check what attributes ecg_test_page has
    print(f" DEBUG: ecg_test_page attributes check:")
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
        print(f" Buffer analysis: Max samples={max_samples}, Min samples={min_samples}")
        
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
    
    Important: Report  ECG graph  width = 27 boxes × 5.25mm = 141.75mm 
     wave_speed  time calculate    factor use :
        Time from wave_speed = (140.7mm / wave_speed_mm_s) seconds
    
    Formula:
        - Time window = (141.75mm / wave_speed_mm_s) seconds ONLY
          ( 27 boxes × 5.25mm = 141.75mm total width)
        - BPM window is NOT used - only wave speed window
        - Beats = (BPM / 60) × time_window
        - Final window clamped maximum 20 seconds (NO minimum clamp)
    
    Examples:
        # Example 1: BPM 80, wave_speed 12.5 mm/s
        #   Time window: 165 / 12.5 = 13.2 seconds
        #   Beats: (80/60) × 13.2 = 1.33 × 13.2 = 17.6 ≈ 18 beats
        
        # Example 2: BPM 80, wave_speed 25 mm/s
        #   Time window: 165 / 25 = 6.6 seconds
        #   Beats: (80/60) × 6.6 = 1.33 × 6.6 = 8.8 ≈ 9 beats
        
        # Example 3: BPM 80, wave_speed 50 mm/s
        #   Time window: 165 / 50 = 3.3 seconds
        #   Beats: (80/60) × 3.3 = 1.33 × 3.3 = 4.4 ≈ 4 beats
    
    Returns: (time_window_seconds, num_samples)
    """
    # Calculate time window from wave_speed ONLY (BPM window NOT used)
    # Report  ECG graph width = 27 boxes × 5.25mm = 141.75mm
    # Time = Distance / Speed
    ecg_graph_width_mm = 27 * ECG_LARGE_BOX_MM_WIDTH
    calculated_time_window = ecg_graph_width_mm / max(1e-6, wave_speed_mm_s)
    
    # Only clamp maximum to 20 seconds (NO minimum clamp)
    calculated_time_window = min(calculated_time_window, 20.0)
    
    # Calculate number of samples (assuming 500 Hz default, will be adjusted by actual sampling rate)
    num_samples = int(calculated_time_window * 500.0)  # Will be recalculated with actual sampling rate
    
    # Calculate expected beats: beats = (BPM / 60) × time_window
    # Formula: beats per second = BPM / 60, then multiply by time window
    beats_per_second = hr_bpm / 60.0 if hr_bpm > 0 else 0
    expected_beats = beats_per_second * calculated_time_window
    
    print(f" Time Window Calculation (Wave Speed ONLY):")
    print(f"   Graph Width: {27 * ECG_LARGE_BOX_MM_WIDTH:.2f}mm (27 boxes × {ECG_LARGE_BOX_MM_WIDTH:.2f}mm)")
    print(f"   Wave Speed: {wave_speed_mm_s}mm/s")
    print(f"   Time Window: {ecg_graph_width_mm:.1f} / {wave_speed_mm_s} = {calculated_time_window:.2f}s")
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
                print(f" DEMO MODE ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    ecg_graph_width_mm = 27 * ECG_LARGE_BOX_MM_WIDTH
                    time_window_seconds = ecg_graph_width_mm / wave_speed
                    print(f" DEMO MODE ON - Calculated window using NEW LOGIC: {ecg_graph_width_mm:.2f}mm / {wave_speed}mm/s = {time_window_seconds}s")
                except Exception as e:
                    print(f" Could not get demo time window: {e}")
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

# ADC-per-box config for the leads plotted in this report (kept near baseline/plot logic).
ADC_PER_BOX_CONFIG = {
    'V1': 6400.0,
    'V2': 6400.0,
    'V3': 6400.0,
    'V4': 6400.0,
    'V5': 6400.0,
    'V6': 6400.0,
    'II': 6400.0,
}

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
        
        
        # Get lead-specific ADC per box multiplier (default: 6400)
        adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead_name, 6400.0)
        
        # Convert to numpy array
        adc_data = np.array(ecg_data, dtype=float)
        
        # Apply baseline (subtract baseline from ADC values)
        baseline_adc = ECG_BASELINE_ADC
        centered_adc = adc_data - baseline_adc
        
        # Calculate ADC per box based on wave_gain and lead-specific multiplier
        adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
        
        # Convert ADC offset to boxes (vertical units)
        # Direct calculation: boxes_offset = centered_adc / adc_per_box
        boxes_offset = centered_adc / adc_per_box
        
        # Convert boxes to Y position
        center_y = height / 2.0  # Center of the graph in points
        box_height_points = ECG_LARGE_BOX_MM_HEIGHT * mm
        
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
        print(" Using zero-value fallback (no ECG data available)")
    
    # Ensure we have exactly 12 conclusions (pad with empty strings if needed)
    MAX_CONCLUSIONS = 12
    while len(conclusions) < MAX_CONCLUSIONS:
        conclusions.append("---")  # Use "---" for empty slots
    
    # Limit to maximum 12 conclusions
    conclusions = conclusions[:MAX_CONCLUSIONS]
    
    print(f" Final conclusions list (12 total): {len([c for c in conclusions if c and c != '---'])} filled, {len([c for c in conclusions if not c or c == '---'])} blank")
    
    return conclusions


def load_latest_metrics_entry(reports_dir):
    metrics_path = os.path.join(reports_dir, 'hyper_metric.json')
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and data:
            return data[-1]

        if isinstance(data, dict):
            entries = data.get('entries')
            if isinstance(entries, list) and entries:
                return entries[-1]

            if data.get('timestamp'):
                return data
    except Exception as e:
        print(f" Could not read hyper metrics file for HR: {e}")

    return None


def load_latest_hyper_metrics_entry(reports_dir):
    return load_latest_metrics_entry(reports_dir)
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

    # ==================== STEP 1: Get HR_bpm from Hyperkalemia metrics (PRIORITY) ====================
    latest_metrics = load_latest_metrics_entry(reports_dir)
    hr_bpm_value = 0
    if latest_metrics:
        hr_bpm_value = _safe_int(latest_metrics.get("HR_bpm"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from hyper_metric.json: {hr_bpm_value} bpm (for calculation-based beats)")

    data["HR_bpm"] = hr_bpm_value
    data["Heart_Rate"] = hr_bpm_value
    data["HR"] = hr_bpm_value
    data["HR_avg"] = hr_bpm_value
    if hr_bpm_value > 0:
        data["RR_ms"] = int(60000 / hr_bpm_value)
    else:
        data["RR_ms"] = 0

    # ==================== STEP 2: Get wave_speed from ecg_settings.json (PRIORITY) ====================
    # Priority: ecg_settings.json  wave_speed  (calculation-based beats  )
    wave_speed_setting = settings_manager.get_setting("wave_speed", "25")
    wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
    wave_speed_mm_s = _safe_float(wave_speed_setting, 25.0)  # Default: 25.0 mm/s
    wave_gain_mm_mv = _safe_float(wave_gain_setting, 10.0)   # Default: 10.0 mm/mV
    print(f" Using wave_speed from ecg_settings.json: {wave_speed_mm_s} mm/s (for calculation-based beats)")
    # Keep sampling rate aligned with main ECG report logic
    computed_sampling_rate = 500

    data["wave_speed_mm_s"] = wave_speed_mm_s
    data["wave_gain_mm_mv"] = wave_gain_mm_mv

    print(f" Pre-plot checks: HR_bpm={hr_bpm_value}, RR_ms={data['RR_ms']}, wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV, sampling_rate={computed_sampling_rate}Hz")
    print(f" Calculation-based beats formula:")
    print(f"   Graph width: 27 boxes × 5.25mm = 141.75mm")
    print(f"   BPM window: (desired_beats × 60) / {hr_bpm_value} = {(6 * 60.0 / hr_bpm_value) if hr_bpm_value > 0 else 0:.2f}s")
    print(f"   Wave speed window: 141.75mm / {wave_speed_mm_s}mm/s = {141.75 / wave_speed_mm_s:.2f}s")
    
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
            computed_sampling_rate = 500
            print(f" Using sampling rate from provided file: {computed_sampling_rate} Hz")
    elif ecg_test_page and hasattr(ecg_test_page, 'data'):
        # ALWAYS save current data to file before generating report (REQUIRED for calculation-based beats)
        print(" Saving ECG data to file (required for calculation-based beats)...")
        saved_data_file_path = save_ecg_data_to_file(ecg_test_page)
        if saved_data_file_path:
            saved_ecg_data = load_ecg_data_from_file(saved_data_file_path)
            if saved_ecg_data:
                computed_sampling_rate = 500
                print(f" Using sampling rate from saved file: {computed_sampling_rate} Hz")
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
            print(" DEMO MODE DETECTED - Checking data availability...")
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

    
    # Vital Parameters Header (completely transparent)
    vital_style = ParagraphStyle(
        'VitalStyle',
        fontSize=12,  # Increased from 11
        fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceAfter=15,
        alignment=1,  # center
        # Add white background 
        # for better visibility on pink grid
        backColor=colors.white,
    )

    # Vital Parameters Header (on top of background)
    vital_style = ParagraphStyle(


            
        'VitalStyle',
        fontSize=11,
        fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceAfter=15,
        alignment=1,  # center
        
    )

    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT', )
    QTc = data.get('QTc',)
    ST = data.get('ST',)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   

    # Create table data: 4 rows × 2 columns
    vital_table_data = [
        [f"HR : {int(round(HR))} bpm", f"QT: {int(round(QT))} ms"],
        [f"PR : {int(round(PR))} ms", f"QTc: {int(round(QTc))} ms"],
        [f"QRS: {int(round(QRS))} ms", f"ST: {int(round(ST))} ms"],
        [f"RR : {int(round(RR))} ms", ""]  
    ]

    # Create vital parameters table with MORE LEFT and TOP positioning
    vital_params_table = Table(vital_table_data, colWidths=[100, 100])  # Even smaller widths for more left

    vital_params_table.setStyle(TableStyle([
        # Transparent background to show pink grid
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0, 0, 0, alpha=0)),
        ("GRID", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("BOX", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),  # Left align
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),  # Normal font
        ("FONTSIZE", (0, 0), (-1, -1), 10),  # Same size as Name, Age, Gender
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),  # Zero left padding for extreme left
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),  # Zero right padding too
        ("TOPPADDING", (0, 0), (-1, -1), 0),   # Zero top padding for top shift
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

   

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
                print(f" Report Generator: Demo mode ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    ecg_graph_width_mm = 27 * ECG_LARGE_BOX_MM_WIDTH
                    time_window_seconds = ecg_graph_width_mm / wave_speed
                    print(f" Report Generator: Demo mode ON - Calculated window using NEW LOGIC: {ecg_graph_width_mm:.2f}mm / {wave_speed}mm/s = {time_window_seconds}s")
                except Exception as e:
                    print(f" Could not get demo time window: {e}")
                    time_window_seconds = None
        else:
            print(f" Report Generator: Demo mode is OFF")
    
    # Calculate number of samples to capture based on demo mode OR BPM + wave_speed
    calculated_time_window = None  # Initialize for use in data loading section
    if is_demo_mode and time_window_seconds is not None:
        # In demo mode: only capture data visible in one window frame
        calculated_time_window = time_window_seconds
        num_samples_to_capture = int(time_window_seconds * samples_per_second)
        print(f" DEMO MODE: Master drawing will capture only {num_samples_to_capture} samples ({time_window_seconds}s window)")
    else:
        # Normal mode: Calculate time window based on wave_speed ONLY (NEW LOGIC)
        # This ensures proper number of beats are displayed based on graph width
        # Formula: 
        #   - Time window = 141.75mm / wave_speed ONLY (27 boxes × 5.25mm = 141.75mm)
        #   - BPM window is NOT used - only wave speed determines time window
        #   - Beats = (BPM / 60) × time_window
        #   - Maximum clamp: 20 seconds (NO minimum clamp)
        calculated_time_window, _ = calculate_time_window_from_bpm_and_wave_speed(
            hr_bpm_value,      # 90 bpm
            wave_speed_mm_s,   # 25 mm/s (current)
            desired_beats=15   
        )
        
        # Recalculate with actual sampling rate
        num_samples_to_capture = int(calculated_time_window * computed_sampling_rate)
        print(f" NORMAL MODE: Calculated time window: {calculated_time_window:.2f}s")
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
                        baseline_adc = ECG_BASELINE_ADC
                        lead_i_centered = np.array(lead_i_data, dtype=float) - baseline_adc
                        lead_ii_centered = np.array(lead_ii_data, dtype=float) - baseline_adc
                        
                        # Calculate derived lead from centered values
                        calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                        if calculated_data is not None:
                            raw_data = calculated_data.tolist() if isinstance(calculated_data, np.ndarray) else calculated_data
                            print(f" Calculated {lead} from saved I and II data (baseline-subtracted): {len(raw_data)} points")
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
                        print(f" Cannot calculate {lead}: I or II data missing in saved file")
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
                    desired_samples = int(num_samples_to_capture * (2 if lead == "II" else 1))
                    if saved_data_samples < desired_samples:
                        print(f" SAVED FILE {lead} has only {saved_data_samples} samples, need {desired_samples} for {calculated_time_window:.2f}s window")
                        print(f"   Will use ALL saved data ({saved_data_samples} samples) - may show fewer beats than calculated")
                        # Use all available saved data (don't filter)
                        raw_data_to_use = raw_data
                    else:
                        # Apply time window filtering based on calculated window
                        raw_data_to_use = raw_data[-desired_samples:]
                    
                    if len(raw_data_to_use) > 0 and np.std(raw_data_to_use) > 0.01:
                        real_ecg_data = np.array(raw_data_to_use)
                        real_data_available = True
                        time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                        actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                        print(f" Using SAVED FILE {lead} data: {len(real_ecg_data)} points (requested: {time_window_str}, actual: {actual_time_window:.2f}s, std: {np.std(real_ecg_data):.2f})")
            
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
                            baseline_adc = ECG_BASELINE_ADC
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
                                    desired_samples = int(num_samples_to_capture * (2 if lead == "II" else 1))
                                    if len(raw_data) >= desired_samples:
                                        raw_data = raw_data[-desired_samples:]
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
                            desired_samples = int(num_samples_to_capture * (2 if lead == "II" else 1))
                            if len(raw_data) >= desired_samples:
                                raw_data = raw_data[-desired_samples:]
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
                                desired_samples = int(num_samples_to_capture * (2 if lead == "II" else 1))
                                if len(raw_data) >= desired_samples:
                                    raw_data = raw_data[-desired_samples:]
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
                box_width_points = ECG_LARGE_BOX_MM_WIDTH * mm_unit
                width_boxes = 54.0 if lead == "II" else 27.0
                ecg_width = width_boxes * box_width_points
                ecg_height = 45
                
                # Create time array for ALL data with 1mm gap after calibration notch for specific leads
                gap = mm_unit if lead in ["V1", "V2", "V3", "II"] else 0
                t = np.linspace(x_pos + gap, x_pos + ecg_width, len(real_ecg_data))
                
                
                
                # Step 1: Convert ADC data to numpy array
                adc_data = np.array(real_ecg_data, dtype=float)

                # Step 1.1: Apply report filters (DFT -> EMG -> AC) on raw ADC data
                try:
                    from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
                    dft_setting = str(settings_manager.get_setting("filter_dft", "0.5")).strip()
                    emg_setting = str(settings_manager.get_setting("filter_emg", "150")).strip()
                    ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip()
                    if dft_setting not in ("off", ""):
                        adc_data = apply_dft_filter(adc_data, float(computed_sampling_rate), dft_setting)
                    if emg_setting not in ("off", ""):
                        adc_data = apply_emg_filter(adc_data, float(computed_sampling_rate), emg_setting)
                    if ac_setting in ("50", "60"):
                        adc_data = apply_ac_filter(adc_data, float(computed_sampling_rate), ac_setting)
                except Exception as filter_err:
                    print(f" Report filter apply failed for {lead}: {filter_err}")
                
                # DEBUG: Check if data is already processed (baseline-subtracted)
                # If data range is far from 2000 baseline, it might already be processed
                data_mean = np.mean(adc_data)
                data_std = np.std(adc_data)
                
                # Step 2: Apply baseline (subtract baseline from ADC values)
                # IMPORTANT: For calculated leads (III, aVR, aVL, aVF), data is already calculated from processed I and II
                # So it's already centered (mean ~0), but we still need to scale it properly
                baseline_adc = ECG_BASELINE_ADC
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                if abs(data_mean - ECG_BASELINE_ADC) < 500:  # Data is close to baseline 2000 (raw ADC)
                    centered_adc = adc_data - baseline_adc
                elif is_calculated_lead:
                    # For calculated leads, data is already centered from calculation (II - I, etc.)
                    # The calculated value is already the difference, so it's centered around 0
                    # We use it directly without baseline subtraction
                    centered_adc = adc_data  # Use data as-is (already centered from calculation)
                else:  # Data is already processed (baseline-subtracted or filtered)
                    centered_adc = adc_data  # Use data as-is (already centered)
                
                # Step 3: Calculate ADC per box based on wave_gain and lead-specific multiplier
                # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
                # Each lead can have different ADC per box multiplier (will be divided by wave_gain)
                # Get lead-specific ADC per box multiplier (default: 6400)
                adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead, 6400.0)
                # Formula: ADC_per_box = adc_per_box_multiplier / wave_gain_mm_mv
                # IMPORTANT: Each lead can have different ADC per box multiplier
                # For 10mm/mV with multiplier 6400: 6400 / 10 = 640 ADC per box
                # This means: 640 ADC offset = 1 box (5mm) vertical movement
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
                
                # DEBUG: Log actual ADC values for troubleshooting
                max_centered_adc = np.max(np.abs(centered_adc))
                min_centered_adc = np.min(centered_adc)
                max_centered_adc_abs = np.max(np.abs(centered_adc))
                expected_boxes = max_centered_adc_abs / adc_per_box
                
                # Step 4: Convert ADC offset to boxes (vertical units)
                # Direct calculation: boxes_offset = centered_adc / adc_per_box
                # Example: 2000 ADC offset / 750 ADC per box = 2.6666 boxes
                # BUT: If actual ADC values are smaller (e.g., 375 ADC), then:
                # 375 ADC / 750 ADC per box = 0.5 boxes (which matches what user sees!)
                boxes_offset = centered_adc / adc_per_box
                
                # Log boxes offset for verification
                
                # Step 5: Convert boxes to Y position
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                # IMPORTANT: Standard ECG paper uses 5mm per box
                # 5mm = 5 * 2.834645669 points = 14.17 points per box
                from reportlab.lib.units import mm
                box_height_points = 5.0 * mm  # Standard ECG: 5mm = 14.17 points per box
                major_spacing_y = box_height_points  # Use standard ECG spacing (5mm)
                
                # Convert boxes offset to Y position
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                # DEBUG: Verify Y position calculation
                
                
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
                
                print(f" Drew {len(real_ecg_data)} ECG data points for Lead {lead}")
            else:
                print(f" No real data for Lead {lead} - showing grid only")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
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
                print(f" Report Generator: Demo mode ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    ecg_graph_width_mm = 27 * ECG_LARGE_BOX_MM_WIDTH
                    time_window_seconds = ecg_graph_width_mm / wave_speed
                    print(f" Report Generator: Demo mode ON - Calculated window using NEW LOGIC: {ecg_graph_width_mm:.2f}mm / {wave_speed}mm/s = {time_window_seconds}s")
                except Exception as e:
                    print(f" Could not get demo time window: {e}")
                    time_window_seconds = None
        else:
            print(f" Report Generator: Demo mode is OFF")
    
    # Calculate number of samples to capture based on demo mode OR BPM + wave_speed
    calculated_time_window = None  # Initialize for use in data loading section
    if is_demo_mode and time_window_seconds is not None:
        # In demo mode: only capture data visible in one window frame
        calculated_time_window = time_window_seconds
        num_samples_to_capture = int(time_window_seconds * samples_per_second)
        print(f" DEMO MODE: Master drawing will capture only {num_samples_to_capture} samples ({time_window_seconds}s window)")
    else:
        # Normal mode: Calculate time window based on wave_speed ONLY (NEW LOGIC)
        # This ensures proper number of beats are displayed based on graph width
        # Formula: 
        #   - Time window = 141.75mm / wave_speed ONLY (27 boxes × 5.25mm = 141.75mm)
        #   - BPM window is NOT used - only wave speed determines time window
        #   - Beats = (BPM / 60) × time_window
        #   - Maximum clamp: 20 seconds (NO minimum clamp)
        calculated_time_window, _ = calculate_time_window_from_bpm_and_wave_speed(
            hr_bpm_value,  # From hyper_metric.json (priority) - for calculation-based beats
            wave_speed_mm_s,  # From ecg_settings.json - for calculation-based beats
            desired_beats=6  # Default: 6 beats desired
        )
        
        # Recalculate with actual sampling rate
        num_samples_to_capture = int(calculated_time_window * computed_sampling_rate)
        print(f" NORMAL MODE: Calculated time window: {calculated_time_window:.2f}s")
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
                        baseline_adc = ECG_BASELINE_ADC
                        lead_i_centered = np.array(lead_i_data, dtype=float) - baseline_adc
                        lead_ii_centered = np.array(lead_ii_data, dtype=float) - baseline_adc
                        
                        # Calculate derived lead from centered values
                        calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                        if calculated_data is not None:
                            raw_data = calculated_data.tolist() if isinstance(calculated_data, np.ndarray) else calculated_data
                            print(f" Calculated {lead} from saved I and II data (baseline-subtracted): {len(raw_data)} points")
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
                        print(f" Cannot calculate {lead}: I or II data missing in saved file")
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
                        print(f" SAVED FILE {lead} has only {saved_data_samples} samples, need {num_samples_to_capture} for {calculated_time_window:.2f}s window")
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
                        print(f" Using SAVED FILE {lead} data: {len(real_ecg_data)} points (requested: {time_window_str}, actual: {actual_time_window:.2f}s, std: {np.std(real_ecg_data):.2f})")
            
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
                            baseline_adc = ECG_BASELINE_ADC
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
                
                # Create time array for ALL data with 1mm gap after calibration notch for specific leads
                gap = mm_unit if lead in ["V1", "V2", "V3", "II"] else 0
                t = np.linspace(x_pos + gap, x_pos + ecg_width, len(real_ecg_data))
                
                
                # Step 1: Convert ADC data to numpy array
                adc_data = np.array(real_ecg_data, dtype=float)

                # Step 1.1: Apply report filters (DFT -> EMG -> AC) on raw ADC data
                try:
                    from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
                    dft_setting = str(settings_manager.get_setting("filter_dft", "0.5")).strip()
                    emg_setting = str(settings_manager.get_setting("filter_emg", "150")).strip()
                    ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip()
                    if dft_setting not in ("off", ""):
                        adc_data = apply_dft_filter(adc_data, float(computed_sampling_rate), dft_setting)
                    if emg_setting not in ("off", ""):
                        adc_data = apply_emg_filter(adc_data, float(computed_sampling_rate), emg_setting)
                    if ac_setting in ("50", "60"):
                        adc_data = apply_ac_filter(adc_data, float(computed_sampling_rate), ac_setting)
                except Exception as filter_err:
                    print(f" Report filter apply failed for {lead}: {filter_err}")
                
                # DEBUG: Check if data is already processed (baseline-subtracted)
                data_mean = np.mean(adc_data)
                data_std = np.std(adc_data)
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                # Step 2: Apply baseline (subtract baseline from ADC values)
                # IMPORTANT: For calculated leads, data is already calculated from processed I and II
                # So it's already centered (mean ~0), but we still need to scale it properly
                baseline_adc = ECG_BASELINE_ADC
                
                if abs(data_mean - ECG_BASELINE_ADC) < 500:  # Data is close to baseline 2000 (raw ADC)
                    centered_adc = adc_data - baseline_adc
                elif is_calculated_lead:
                    # For calculated leads, data is already centered from calculation (II - I, etc.)
                    # The calculated value is already the difference, so it's centered around 0
                    # We use it directly without baseline subtraction
                    centered_adc = adc_data  # Use data as-is (already centered from calculation)
                else:  # Data is already processed (baseline-subtracted or filtered)
                    centered_adc = adc_data  # Use data as-is (already centered)
                
                # Step 3: Calculate ADC per box based on wave_gain and lead-specific multiplier
                # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
                # Each lead can have different ADC per box multiplier (will be divided by wave_gain)
                # Get lead-specific ADC per box multiplier (default: 6400)
                adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead, 6400.0)
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
                
                print(f" Drew {len(real_ecg_data)} ECG data points for Lead {lead}")
            else:
                print(f" No real data for Lead {lead} - showing grid only")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
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
    
    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    # Values come strictly from hyper_metric.json entry (or zero if not available)
    HR = data.get('HR', 0)
    PR = data.get('PR', 0)
    QRS = data.get('QRS', 0)
    QT = data.get('QT', 0)
    QTc = data.get('QTc', 0)
    ST = data.get('ST', 0)
    RR = data.get('RR_ms', 0)
   
    # Add vital parameters in TWO COLUMNS (ABOVE ECG GRAPH - shifted further up)
    # FIRST COLUMN (Left side - x=130)

    # CALCULATED wave amplitudes and lead-specific measurements
    # Prefer values passed in data; if missing/zero, compute from live ecg_test_page data (last 10s)
    p_amp_mv = data.get('p_amp', 0.0)
    qrs_amp_mv = data.get('qrs_amp', 0.0)
    t_amp_mv = data.get('t_amp', 0.0)
    
    print(f" Report Generator - Received wave amplitudes from data:")
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
            print(f" Fallback computed amplitudes from Lead II: P={p_amp_mv:.4f}, QRS={qrs_amp_mv:.4f}, T={t_amp_mv:.4f}")
        except Exception as e:
            print(f" Fallback amplitude computation failed: {e}")

    # Format axis values for display (remove ° symbol for compact display)
    # Convert to string first in case they're integers
    p_axis_display = "--"
    qrs_axis_display = "--"
    t_axis_display = "--"
    
    # Extract numeric values for JSON storage (convert from string format like "45°" to int)
    def extract_axis_value(axis_str):
        """Extract numeric value from axis string like '45°' or '--'"""
        if axis_str == "--":
            return 0  # Default value if not calculated
        try:
            # Remove ° symbol and convert to int
            return int(str(axis_str).replace("°", "").strip())
        except (ValueError, AttributeError):
            return 0
    
    p_mm = extract_axis_value(p_axis_deg)
    qrs_mm = extract_axis_value(qrs_axis_deg)
    t_mm = extract_axis_value(t_axis_deg)

    # SECOND COLUMN - P/QRS/T Axis (SAME LINE AS HR - right column) - DUMMY LABEL
    p_qrs_label = String(240, 740, "P/QRS/T  : --/--/--°",  # Same y as HR (740)
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(p_qrs_label)

    # SECOND COLUMN - RV5/SV1 (SAME LINE AS PR - right column) - DUMMY LABEL
    rv5_sv_label = String(240, 720, "RV5/SV1  : -- mV/-- mV",  # Same y as PR (720)
                      fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv_label)

    # SECOND COLUMN - RV5+SV1 (SAME LINE AS QRS - right column) - DUMMY LABEL
    rv5_sv1_sum_label = String(240, 700, "RV5+SV1 : -- mV",  # Same y as QRS (700)
                               fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv1_sum_label)

    # SECOND COLUMN - QTCF (SAME LINE AS RR - right column) - DUMMY LABEL
    qtcf_label = String(240, 682, "QTCF       : --",  # Same y as RR (682)
                        fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_label)

    # SECOND COLUMN - ST (SAME LINE AS QT - right column)
    st_label = String(240, 664, f"ST            : {int(round(ST))} ms",  # Same y as QT (664)
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(st_label)

    # SECOND COLUMN - Filter Band (SAME LINE AS QTc - right column)
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
    filter_label = String(240, 646, f"{filter_band}   AC : {ac_frequency}",  # Same y as QTc (646)
                        fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(filter_label)

    # SECOND COLUMN - Speed/Gain (below all parameters)
    master_drawing.add(String(
        240,
        626,  # Below all other parameters
        f"{wave_speed_mm_s} mm/s   {wave_gain_mm_mv} mm/mV",
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
    print(f" Drawing conclusions in graph from filtered list: {filtered_conclusions}")
    
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

    print(f" Added Patient Info, Vital Parameters, {len(filtered_conclusions)} REAL Conclusions (no empty/---), and Doctor Name/Signature to ECG grid")
    
    # STEP 5: Add SINGLE master drawing to story (NO containers)
    story.append(master_drawing)
    story.append(Spacer(1, 15))
    
    print(f" Added SINGLE master drawing with {successful_graphs}/12 ECG leads (ZERO containers)!")
    
    # Final summary
    if is_demo_mode:
        print(f"\n{'='*60}")
        print(f" DEMO MODE REPORT SUMMARY:")
        print(f"   • Total leads processed: {successful_graphs}/12")
        print(f"   • Demo mode: {'ON' if is_demo_mode else 'OFF'}")
        if successful_graphs == 0:
            print(f"    WARNING: No ECG graphs were added to the report!")
            print(f"    SOLUTION: Ensure demo is running for 5-10 seconds before generating report")
        elif successful_graphs < 12:
            print(f"    WARNING: Only {successful_graphs} graphs added (expected 12)")
        else:
            print(f"    SUCCESS: All 12 ECG graphs added successfully!")
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

    # Helper: draw logo on every page AND ALIGNED pink grid background on Page 1
    def _draw_logo_and_footer(canvas, doc):
        import os
        from reportlab.lib.units import mm
        
        # STEP 1: Draw FULL PAGE pink ECG grid background on Page 1 (ECG graphs page)
        if canvas.getPageNumber() == 1:  # Changed from 2 to 1
            page_width, page_height = canvas._pagesize
            full_boxes = 56
            major_spacing = ECG_LARGE_BOX_MM_WIDTH * mm
            minor_spacing = ECG_SMALL_BOX_MM_WIDTH * mm
            
            # Fill entire page with pink background
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            
            # ECG grid colors - darker for better visibility
            light_grid_color = colors.HexColor("#ffd1d1")  
            
            major_grid_color = colors.HexColor("#ffb3b3")   
            
            # Minor grid lines (spacing derived from 56 boxes across width)
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.6)
            
            # Vertical minor lines - full page width (ensure last line aligns to page edge)
            for i in range(full_boxes * 5 + 1):
                x = i * minor_spacing
                canvas.line(x, 0, x, page_height)
            
            # Horizontal minor lines - use same spacing for square boxes
            y = 0.0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += minor_spacing
                
            
            # Major grid lines - FULL PAGE
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(1.2)
            
            # Vertical major lines - 56 boxes across width
            for i in range(full_boxes + 1):  # include right edge
                x = i * major_spacing
                canvas.line(x, 0, x, page_height)
            
            # Horizontal major lines - same spacing for square boxes
            y = 0.0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
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
            if canvas.getPageNumber() == 1:
                logo_w, logo_h = 120, 40  # bigger size for ECG page
                # SHIFTED LEFT FROM RIGHT TOP CORNER
                page_width, page_height = canvas._pagesize
                x = page_width - logo_w - 35  # Shifted 50 pixels left from right edge
                y = page_height - logo_h  # Top edge touch
            else:
                logo_w, logo_h = 120, 40  # normal size for other pages
                x = doc.width + doc.leftMargin - logo_w
                y = doc.height + doc.bottomMargin - logo_h  # top positioning
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
        metrics_path = os.path.join(reports_dir, 'hyper_metric.json')

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
        print(f" Saved parameters to {index_path}")

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
        print(f" Saved 11 metrics to {metrics_path}")
    except Exception as e:
        print(f" Could not save parameters JSON: {e}")

    # Build PDF
    doc.build(story, onFirstPage=_draw_logo_and_footer, onLaterPages=_draw_logo_and_footer)
    print(f" ECG Report generated: {filename}")
    
    # Upload to cloud if configured
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.cloud_uploader import get_cloud_uploader
        
        cloud_uploader = get_cloud_uploader()
        if cloud_uploader.is_configured():
            print(f"  Uploading report to cloud ({cloud_uploader.cloud_service})...")
            
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
                print(f"  Cloud upload failed: {result.get('message', 'Unknown error')}")
        else:
            print("  Cloud upload not configured (see cloud_config_template.txt)")
            
    except ImportError:
        print("  Cloud uploader not available")
    except Exception as e:
        print(f"  Cloud upload error: {e}")


# ==================== HYPERKALEMIA REPORT WRAPPER ====================
def generate_hyperkalemia_report(filename, analysis_results, lead_ii_data, sampling_rate=500.0, ecg_data_file=None):
    """
    Reuse the main ECG report layout for the Hyperkalemia flow.
    Expects:
      - analysis_results: dict with keys like heart_rate, pr_interval_ms, qrs_duration_ms,
        qt_interval_ms, qtc_ms, st_segment_ms, qrs_axis, patient (optional)
      - lead_ii_data: sequence of ADC samples for Lead II or list of dicts with {"value": ..., "time": ...}
      - ecg_data_file: Optional path to saved ECG data file with V1-V6 leads
    """
    if not lead_ii_data:
        raise ValueError("No Lead II data provided for hyperkalemia report")

    # Normalize Lead II samples to a numpy array
    first_item = lead_ii_data[0]
    if isinstance(first_item, dict) and "value" in first_item:
        lead_ii_array = np.array([d.get("value", 0) for d in lead_ii_data], dtype=float)
    else:
        lead_ii_array = np.array(lead_ii_data, dtype=float)

    # Minimal stub that mimics the ecg_test_page shape used by generate_ecg_report
    class _Sampler:
        def __init__(self, fs):
            self.sampling_rate = fs

    class _DummyECGPage:
        def __init__(self, lead_ii, fs):
            # 12 leads, only Lead II (index 1) populated
            self.data = [np.zeros_like(lead_ii) for _ in range(12)]
            self.data[1] = lead_ii
            self.sampler = _Sampler(fs)

    ecg_page = _DummyECGPage(lead_ii_array, sampling_rate)

    # Map analysis results into the fields expected by the main report
    hr = int(analysis_results.get("heart_rate", 0) or 0)
    data = {
        "HR": hr,
        "beat": hr,
        "PR": int(analysis_results.get("pr_interval_ms", 0) or 0),
        "QRS": int(analysis_results.get("qrs_duration_ms", 0) or 0),
        "QT": int(analysis_results.get("qt_interval_ms", 0) or 0),
        "QTc": int(analysis_results.get("qtc_ms", 0) or 0),
        "ST": int(analysis_results.get("st_segment_ms", 0) or 0),
        "HR_max": hr,
        "HR_min": hr,
        "HR_avg": hr,
        "Heart_Rate": hr,
        "QRS_axis": analysis_results.get("qrs_axis", "--"),
    }

    # Optional patient info passthrough
    patient = analysis_results.get("patient", {}) if isinstance(analysis_results, dict) else {}

    # Convert lead_ii_data to the format expected by generate_hyperkalemia_ecg_report
    # If it's already a list of dicts with 'time' and 'value', use as is
    # Otherwise, create time-based dicts
    if lead_ii_data and isinstance(lead_ii_data[0], dict) and "value" in lead_ii_data[0]:
        # Already in correct format
        formatted_lead_ii_data = lead_ii_data
    else:
        # Convert array to list of dicts with time and value
        formatted_lead_ii_data = []
        for i, value in enumerate(lead_ii_array):
            formatted_lead_ii_data.append({
                'time': i / sampling_rate,  # Time in seconds
                'value': float(value)
            })
    
    # Generate using the Hyperkalemia-specific report generator (with logging and landscape Page 2)
    from utils.settings_manager import SettingsManager
    settings_manager = SettingsManager()
    
    print(f" Calling generate_hyperkalemia_ecg_report with ecg_data_file: {ecg_data_file}")
    
    return generate_hyperkalemia_ecg_report(
        filename=filename,
        lead_ii_data=formatted_lead_ii_data,
        data=data,
        patient=patient,
        settings_manager=settings_manager,
        ecg_data_file=ecg_data_file
    )


# ==================== Hyperkalemia ECG REPORT GENERATION ====================
# COMPLETE ECG REPORT FORMAT - Same as generate_ecg_report() but with 5 one-minute Lead II graphs

def generate_hyperkalemia_ecg_report(filename="hyperkalemia_ecg_report.pdf", lead_ii_data=None, data=None, patient=None, settings_manager=None, ecg_data_file=None):
    """
    Generate Hyperkalemia ECG report PDF with EXACT SAME format as main 12-lead ECG report
    Only difference: Page 2 shows 5 one-minute Lead II graphs in LANDSCAPE mode instead of 12 leads
    All Page 1 content, styling, formulas - EXACTLY SAME as generate_ecg_report()
    
    Parameters:
        filename: Output PDF filename
        lead_ii_data: List of {'time': seconds, 'value': adc_value} dictionaries (5 minutes of Lead II)
        data: Metrics dictionary (HR, PR, QRS, etc.) - same format as main report
        patient: Patient details dictionary
        settings_manager: Settings manager for wave_speed, wave_gain, etc.
        ecg_data_file: Optional path to saved ECG data file with V1-V6 leads
    """
    # ==================== SETUP REPORT PATHS ====================
    from datetime import datetime
    # Normalize output path and ensure directory exists
    filename = os.path.abspath(filename)
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    if lead_ii_data is None or len(lead_ii_data) == 0:
        print(" No Lead II data provided for Hyperkalemia ECG report")
        return None
    
    # ==================== INITIALIZE (EXACT SAME AS MAIN REPORT) ====================
    
    if data is None:
        data = {
            "HR": 0, "beat": 0, "PR": 0, "QRS": 0, "QT": 0, "QTc": 0, "ST": 0,
            "HR_max": 0, "HR_min": 0, "HR_avg": 0, "Heart_Rate": 0, "QRS_axis": "--"
        }
    
    if settings_manager is None:
        from utils.settings_manager import SettingsManager
        settings_manager = SettingsManager()
    
    def _safe_float(value, default):
        try:
            return float(value)
        except Exception:
            return default
    
    beats_27_at_60 = beats_in_boxes(60.0, 27)
    complete_rr = int(beats_27_at_60)
    r_peaks = complete_rr + 1
    print(f" 27 boxes @60 bpm: beats={beats_27_at_60:.2f}, complete RR={complete_rr}, R peaks≈{r_peaks}")
    
    def _safe_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default
    
    # ==================== STEP 1: Get HR_bpm from Hyperkalemia metrics (PRIORITY) - SAME AS MAIN REPORT ====================
    latest_metrics = load_latest_hyper_metrics_entry(reports_dir)
    
    # If last entry has zero values, find last valid (non-zero) entry
    if latest_metrics and latest_metrics.get("HR_bpm", 0) == 0:
        try:
            metrics_path = os.path.join(reports_dir, 'hyper_metric.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    all_metrics = json.load(f)
                if isinstance(all_metrics, list) and len(all_metrics) > 0:
                    # Find last non-zero HR_bpm entry
                    for i in range(len(all_metrics)-1, -1, -1):
                        if all_metrics[i].get("HR_bpm", 0) > 0:
                            latest_metrics = all_metrics[i]
                            print(f" Hyperkalemia Report: Found last valid metric entry (index {i}) with HR_bpm > 0")
                            break
        except Exception as e:
            print(f" Could not find last valid metric: {e}")
    
    hr_bpm_value = 0
    if latest_metrics:
        hr_bpm_value = _safe_int(latest_metrics.get("HR_bpm"))
        if hr_bpm_value > 0:
            print(f" Hyperkalemia Report: Using HR_bpm from hyper_metric.json: {hr_bpm_value} bpm (timestamp: {latest_metrics.get('timestamp', 'N/A')})")

    original_hr_bpm_from_metrics = hr_bpm_value

    data["HR_bpm"] = hr_bpm_value
    data["Heart_Rate"] = hr_bpm_value
    data["HR"] = hr_bpm_value
    data["HR_avg"] = hr_bpm_value
    if hr_bpm_value > 0:
        data["RR_ms"] = int(60000 / hr_bpm_value)
    else:
        data["RR_ms"] = 0
    
    # ==================== STEP 2: Get ALL metrics from hyper_metric.json (PRIORITY) ====================
    original_metrics_from_json = {
        "HR": 0, "PR": 0, "QRS": 0, "QT": 0, "QTc": 0, "ST": 0, "RR_ms": 0
    }
    if latest_metrics:
        # ALWAYS save hyper_metric.json values (for Page 2 - ECG waves)
        original_metrics_from_json["HR"] = _safe_int(latest_metrics.get("HR_bpm", 0))
        original_metrics_from_json["PR"] = _safe_int(latest_metrics.get("PR_ms", 0))
        original_metrics_from_json["QRS"] = _safe_int(latest_metrics.get("QRS_ms", 0))
        original_metrics_from_json["QT"] = _safe_int(latest_metrics.get("QT_ms", 0))
        original_metrics_from_json["QTc"] = _safe_int(latest_metrics.get("QTc_ms", 0))
        original_metrics_from_json["ST"] = _safe_int(latest_metrics.get("ST_ms", 0))
        original_metrics_from_json["RR_ms"] = _safe_int(latest_metrics.get("RR_ms", 0))
        if original_metrics_from_json["HR"] > 0 and original_metrics_from_json["RR_ms"] == 0:
            original_metrics_from_json["RR_ms"] = int(60000 / original_metrics_from_json["HR"])
        
        print(f" Hyperkalemia Report: Saved hyper_metric.json values for Page 2 (ECG waves):")
        print(f"   HR={original_metrics_from_json['HR']}, PR={original_metrics_from_json['PR']}, QRS={original_metrics_from_json['QRS']}")
        print(f"   QT={original_metrics_from_json['QT']}, QTc={original_metrics_from_json['QTc']}, ST={original_metrics_from_json['ST']}, RR={original_metrics_from_json['RR_ms']}")
        
        data["PR"] = original_metrics_from_json["PR"]
        data["QRS"] = original_metrics_from_json["QRS"]
        data["QT"] = original_metrics_from_json["QT"]
        data["QTc"] = original_metrics_from_json["QTc"]
        data["ST"] = original_metrics_from_json["ST"]
        
        print(f" Hyperkalemia Report: Loaded metrics from hyper_metric.json: HR={data['HR']}, PR={data['PR']}, QRS={data['QRS']}, QT={data['QT']}, QTc={data['QTc']}, ST={data['ST']}")
    
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
    
    print(f" Hyperkalemia Report Settings: wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV")
    print(f" Hyperkalemia Report Final Metrics: HR={data['beat']}, PR={data['PR']}, QRS={data['QRS']}, QT={data['QT']}, QTc={data['QTc']}, ST={data['ST']}")
    
    # ==================== CALCULATE HEART RATE FROM 5 MINUTES (BEFORE REPORT GENERATION) ====================
    # Calculate average Heart Rate from 5 minutes of data (to use in report)
    hr_per_minute_for_report = []
    segment_duration = 11.0  # Same as ECG graphs: 11 seconds per strip
    num_segments = 5  # Always 5 strips
    
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
            min_distance = int(0.4 * sampling_rate)
            peaks, _ = find_peaks(values_norm, distance=min_distance, height=0.3)
            
            if len(peaks) < 2:
                return None, None, []
            
            rr_intervals = np.diff(peaks) * (1000.0 / sampling_rate)
            rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
            
            if len(rr_intervals) == 0:
                return None, None, []
            
            avg_rr = float(np.mean(rr_intervals))
            hr = 60000 / avg_rr if avg_rr > 0 else None
            return avg_rr, hr, rr_intervals.tolist()  # Return all RR intervals as list
            
        except Exception as e:
            return None, None, []
    
    # Get sampling rate from settings
    sampling_rate = 500.0  # Default
    if settings_manager:
        try:
            sampling_rate = 500.0
        except Exception:
            pass
    
    # Calculate HR for each minute AND collect average RR per minute
    avg_rr_per_minute = []  # Collect average RR for each of the 5 minutes
    if lead_ii_data and len(lead_ii_data) > 100:
        for seg_idx in range(num_segments):
            minute_start = seg_idx * 60.0
            seg_start = minute_start
            seg_end = minute_start + segment_duration
            seg_data = [d for d in lead_ii_data if seg_start <= d['time'] < seg_end]
            
            if len(seg_data) > 6400:
                seg_data = seg_data[:6400]
            
            if len(seg_data) > 100:
                avg_rr, hr_val, rr_intervals_list = calculate_rr_from_segment_early(seg_data, sampling_rate)
                if avg_rr is not None and hr_val is not None:
                    hr_per_minute_for_report.append(hr_val)
                    # Collect average RR for this minute (not all individual intervals)
                    avg_rr_per_minute.append(avg_rr)
                else:
                    hr_per_minute_for_report.append(0)
                    avg_rr_per_minute.append(0)
            else:
                hr_per_minute_for_report.append(0)
                avg_rr_per_minute.append(0)
    else:
        hr_per_minute_for_report = [0] * 5
        avg_rr_per_minute = [0] * 5
    
    while len(hr_per_minute_for_report) < 5:
        hr_per_minute_for_report.append(0)
        avg_rr_per_minute.append(0)
    
    # NEW CALCULATION: HR = 300000 / sum_of_avg_rr_per_minute
    # Sum of 5 average RR values (one per minute)
    if len(avg_rr_per_minute) >= 5:
        sum_of_avg_rr = sum(avg_rr_per_minute[:5])
        avg_hr_from_5_minutes = 300000 / sum_of_avg_rr if sum_of_avg_rr > 0 else 0
        
        print(f" Hyperkalemia-Specific Heart Rate Calculation (NEW METHOD - 300000 / sum of avg RR per minute):")
        print(f"   Original HR_bpm from hyper_metric.json: {original_hr_bpm_from_metrics} bpm (NOT CHANGED)")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   Average RR per minute: {[round(r, 1) for r in avg_rr_per_minute[:5]]} ms")
        print(f"   Sum of 5 average RR values: {sum_of_avg_rr:.2f} ms")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   NEW Formula: HR = 300000 / sum_of_avg_rr_per_minute")
        print(f"   Calculation: 300000 / {sum_of_avg_rr:.2f} = {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   Per-minute HR values (for reference):")
        for i, hr_val in enumerate(hr_per_minute_for_report):
            print(f"   Min {i+1}: {hr_val:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"    Hyperkalemia-Specific BPM: {round(avg_hr_from_5_minutes)} bpm")
        print(f"    Original HR_bpm from hyper_metric.json: {original_hr_bpm_from_metrics} bpm (saved as Original_HR_bpm for reference)\n")
    else:
        # Fallback to old method if no average RR values found
        if len(avg_rr_per_minute) >= 5:
            sum_of_avg_rr = sum(avg_rr_per_minute[:5])
            avg_hr_from_5_minutes = 300000 / sum_of_avg_rr if sum_of_avg_rr > 0 else 0
        else:
            avg_hr_from_5_minutes = np.mean(hr_per_minute_for_report) if len(hr_per_minute_for_report) > 0 else 0
        print(f" Using fallback method: {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   Original HR_bpm from hyper_metric.json: {original_hr_bpm_from_metrics} bpm (NOT CHANGED)")
        print(f"   ─────────────────────────────────────────────────────────────")
        for i, hr_val in enumerate(hr_per_minute_for_report):
            print(f"   Min {i+1}: {hr_val:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"   Hyperkalemia Average HR (fallback): {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   Calculation: ({hr_per_minute_for_report[0]:.1f} + {hr_per_minute_for_report[1]:.1f} + {hr_per_minute_for_report[2]:.1f} + {hr_per_minute_for_report[3]:.1f} + {hr_per_minute_for_report[4]:.1f}) / 5 = {avg_hr_from_5_minutes:.2f} bpm")
        print(f"   ─────────────────────────────────────────────────────────────")
        print(f"    Hyperkalemia-Specific BPM: {round(avg_hr_from_5_minutes)} bpm")
        print(f"    Original HR_bpm from hyper_metric.json: {original_hr_bpm_from_metrics} bpm (saved as Original_HR_bpm for reference)\n")
    
    # Save Hyperkalemia-specific BPM separately (for Page 3 only)
    hyperkalemia_specific_bpm = round(avg_hr_from_5_minutes)
    
    # Use original HR_bpm from hyper_metric.json (saved earlier, NOT to change it)
    # original_hr_bpm_from_metrics was already saved above from hyper_metric.json
    
    # Keep data dictionary with hyper_metric.json values for Page 1 and Page 2
    # Only Page 3 will use Hyperkalemia-specific values
    # DO NOT overwrite data["HR_avg"], data["beat"] here - keep original hyper_metric.json values
    
    # ==================== PAGE SETUP (MIXED: Page 1 Portrait, Page 2 Landscape) ====================
    
    # Patient org and phone for logo/footer callback
    patient_org = patient.get("Org.", "") if patient else ""
    patient_doctor_mobile = patient.get("doctor_mobile", "") if patient else ""
    
    # Define callback function for headers/footers BEFORE creating templates
    def _draw_logo_and_footer_callback(canvas, doc_obj):
        from reportlab.lib.units import mm
        
        # STEP 1: Draw pink ECG grid background ONLY on Page 1 (57 BOXES IN FULL 297MM WIDTH)
        if canvas.getPageNumber() == 1:
            page_width, page_height = canvas._pagesize
            
            num_boxes_width = 57
            page_width_mm = 297.0
            box_width_mm = page_width_mm / num_boxes_width
            box_width_pts = box_width_mm * mm
            
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            
            light_grid_color = colors.HexColor("#ffd1d1")
            major_grid_color = colors.HexColor("#ffb3b3")
            
            minor_spacing_mm = box_width_mm / 5.0
            minor_spacing_pts = minor_spacing_mm * mm
            
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.6)
            
            x = 0
            while x <= page_width:
                canvas.line(x, 0, x, page_height)
                x += minor_spacing_pts
                if x > page_width:
                    break
            
            num_boxes_height = 40
            page_height_mm = 210.0
            box_height_mm = page_height_mm / num_boxes_height
            minor_spacing_y = (box_height_mm / 5.0) * mm
            y = 0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += minor_spacing_y
            
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(0.6)
            
            x = 0
            for i in range(num_boxes_width + 1):
                canvas.line(x, 0, x, page_height)
                x += box_width_pts
            
            box_height_pts = box_height_mm * mm
            y = 0
            for i in range(num_boxes_height + 1):
                canvas.line(0, y, page_width, y)
                y += box_height_pts
        
        # STEP 1.5: Draw Org. and Phone No. on Page 1 (REPOSITIONED - slightly higher, more left)
        if canvas.getPageNumber() == 1:
            canvas.saveState()
            # Portrait A4 height = 842 points, position very close to top
            page_height = 842  # A4 portrait height
            x_pos = 15  # More to the left (was 30, now 15)
            y_pos = page_height - 30  \
            
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.black)
            org_label = "Org:"
            canvas.drawString(x_pos, y_pos, org_label)
            
            org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica", 10)
            canvas.drawString(x_pos + org_label_width + 5, y_pos, patient_org if patient_org else "")
            
            y_pos -= 15
            
            canvas.setFont("Helvetica-Bold", 10)
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
            if canvas.getPageNumber() in [1]:
                # Page 1 is LANDSCAPE - logo at top right
                logo_w, logo_h = 120, 40
                page_width, page_height = canvas._pagesize
                x = page_width - logo_w - 35
                y = page_height - logo_h
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
        
        if canvas.getPageNumber() in [1]:
            # Page 1 is LANDSCAPE
            page_width, page_height = canvas._pagesize
            x = (page_width - text_width) / 2
        else:
            # Page 1 is PORTRAIT
            x = (595 - text_width) / 2
        
        y = 10
        canvas.drawString(x, y, footer_text)
        canvas.restoreState()
    
    # Create BaseDocTemplate for mixed page orientations
    doc = BaseDocTemplate(filename, pagesize=A4,
                         rightMargin=30, leftMargin=30,
                         topMargin=30, bottomMargin=30)
    
    # Define Portrait template (for Page 1) with onPage callback
    portrait_frame = Frame(doc.leftMargin, doc.bottomMargin,
                          doc.width, doc.height,
                          id='portrait_frame')
    portrait_template = PageTemplate(id='portrait', frames=[portrait_frame], 
                                    pagesize=A4, onPage=_draw_logo_and_footer_callback)
    
    # Define Landscape template (for Page 2) with onPage callback
    landscape_width, landscape_height = landscape(A4)
    # Reduce margins to increase frame size (from 30 to 20 on each side)
    landscape_frame = Frame(20, 20,  # reduced margins for landscape to fit taller drawing
                           landscape_width - 40, landscape_height - 40,
                           id='landscape_frame')
    landscape_template = PageTemplate(id='landscape', frames=[landscape_frame], 
                                     pagesize=landscape(A4), onPage=_draw_logo_and_footer_callback)
    
    # Add templates to document - landscape first (default for Page 1)
    # IMPORTANT: First template becomes default, so landscape_template must be first
    doc.addPageTemplates([landscape_template, portrait_template])
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
    
    # ==================== SKIP PAGE 1 - GO DIRECTLY TO PAGE 2 ====================
    
    # Extract minimal patient info needed for drawing
    def _load_latest_patient_from_file():
        try:
            patients_file = os.path.abspath(os.path.join(base_dir, "all_patients.json"))
            if not os.path.exists(patients_file):
                return {}
            with open(patients_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            patients_list = data.get("patients", []) if isinstance(data, dict) else []
            if not patients_list:
                return {}
            return patients_list[-1]  # latest entry
        except Exception as e:
            print(f" Could not load all_patients.json: {e}")
            return {}
    
    if patient is None:
        patient = {}
    latest_patient = _load_latest_patient_from_file()
    
    # Prefer non-empty from patient; fallback to latest_patient
    def pick(field_name):
        val = patient.get(field_name, "") if isinstance(patient, dict) else ""
        if val not in [None, "", " "]:
            return val
        return latest_patient.get(field_name, "") if isinstance(latest_patient, dict) else ""
    
    first_name = pick("first_name")
    last_name = pick("last_name")
    age = pick("age")
    gender = pick("gender")
    # Always stamp current date/time for the report (override stored value)
    from datetime import datetime
    date_time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time = date_time_now
    doctor = pick("doctor") or "—"
    doctor_mobile = pick("doctor_mobile") or "—"
    org_name = pick("Org.") or pick("org") or "—"
    
    # For canvas labels (top-left)
    patient_org = org_name
    patient_doctor_mobile = doctor_mobile
    full_name = f"{first_name} {last_name}".strip()
    date_time_str = date_time
    
    # Get conclusions for landscape page
    dashboard_conclusions = get_dashboard_conclusions_from_image(None)
    
    # Filter conclusions (SAME AS MAIN REPORT)
    filtered_conclusions = []
    for conclusion in dashboard_conclusions:
        if conclusion and conclusion.strip() and conclusion.strip() != "---":
            filtered_conclusions.append(conclusion.strip())
            if len(filtered_conclusions) >= 12:
                break
    
    # If no real conclusions, use Hyperkalemia-specific defaults
    if not filtered_conclusions:
        filtered_conclusions = [
            "5-minute Lead II Hyperkalemia analysis completed",
            "Heart rate variability recorded successfully"
        ]
    
    # Switch to Landscape template directly (no Page 1)
    # No need for NextPageTemplate since landscape is now default
    
    # ==================== PAGE 1: V1-V6 LEADS + LEAD II (LANDSCAPE MODE) ====================
    
    print(" Creating master drawing with V1-V6 leads (2 columns) + Lead II (bottom) in LANDSCAPE...")
    
    # Landscape A4 frame dimensions (landscape_width - 40, landscape_height - 40)
    # A4 landscape: 842 x 595, minus reduced margins (20 each side) = 802 x 555
    total_width = 780  # Fits in landscape frame (802 - margin)
    total_height = 540  # Height for patient info and graphs
    master_drawing = Drawing(total_width, total_height)
    
    # Load saved ECG data to get V1-V6 leads (and Lead II to match main ECG report)
    saved_ecg_data = None
    
    # Priority 1: Use provided ecg_data_file if available
    if ecg_data_file and os.path.exists(ecg_data_file):
        try:
            with open(ecg_data_file, 'r') as f:
                saved_ecg_data = json.load(f)
        except Exception as e:
            print(f" Could not load provided ECG data file: {e}")
    
    # Priority 2: Find latest ECG data file if no file was provided
    if not saved_ecg_data:
        ecg_data_dir = os.path.join(reports_dir, 'ecg_data')
        if os.path.exists(ecg_data_dir):
            ecg_files = [f for f in os.listdir(ecg_data_dir) if f.startswith('ecg_data_') and f.endswith('.json')]
            if ecg_files:
                ecg_files.sort(reverse=True)  # Latest first
                latest_file = os.path.join(ecg_data_dir, ecg_files[0])
                try:
                    with open(latest_file, 'r') as f:
                        saved_ecg_data = json.load(f)
                except Exception as e:
                    print(f" Could not load ECG data: {e}")

    # Match main ECG report windowing for lead data (fixed 25 mm/s, 150 mm => 6.0 s)
    report_sampling_rate = 500.0
    num_samples_to_capture = int(6.0 * report_sampling_rate)

    # If saved ECG data has Lead II, prefer it to keep data identical to main ECG report
    if saved_ecg_data and isinstance(saved_ecg_data, dict):
        try:
            saved_leads = saved_ecg_data.get('leads', {})
            saved_lead_ii = saved_leads.get('II')
            if saved_lead_ii is not None and len(saved_lead_ii) > 0:
                lead_ii_array = np.array(saved_lead_ii, dtype=float)
                desired_samples_ii = int(num_samples_to_capture * 2) if num_samples_to_capture else None
                if desired_samples_ii and len(lead_ii_array) > desired_samples_ii:
                    lead_ii_array = lead_ii_array[-desired_samples_ii:]
                saved_fs = report_sampling_rate
                lead_ii_data = [
                    {'time': i / saved_fs, 'value': float(v)}
                    for i, v in enumerate(lead_ii_array)
                ]
                print(f" Using Lead II from saved ECG data: {len(lead_ii_array)} samples at {saved_fs} Hz")
        except Exception as e:
            print(f" Could not override Lead II from saved ECG data: {e}")
    
    # Lead-specific ADC per box multipliers aligned with main ECG report
    
    # Reuse reportlab mm unit locally for clarity
    mm_unit = mm
    
    # Function to add grid lines to a drawing area
    def add_grid_lines(drawing, x_pos, y_pos, width, height):
        """Add ECG grid lines (major and minor)"""
        box_width_points = ECG_LARGE_BOX_MM_WIDTH * mm_unit
        minor_box_width = ECG_SMALL_BOX_MM_WIDTH * mm_unit
        box_height_points = ECG_LARGE_BOX_MM_HEIGHT * mm_unit
        minor_box_height = ECG_SMALL_BOX_MM_HEIGHT * mm_unit
        
        # Major grid lines (every 5mm)
        major_color = colors.HexColor("#FF0000")  # Red for major lines
        minor_color = colors.HexColor("#FF9999")  # Light red for minor lines
        
        # Vertical major lines (every 5mm)
        num_major_x = int(width / box_width_points) + 1
        for i in range(num_major_x):
            x = x_pos + (i * box_width_points)
            if x <= x_pos + width:
                line = Line(x, y_pos, x, y_pos + height, 
                          strokeColor=major_color, strokeWidth=0.5)
                drawing.add(line)
        
        # Vertical minor lines (every 1mm)
        num_minor_x = int(width / minor_box_width) + 1
        for i in range(num_minor_x):
            x = x_pos + (i * minor_box_width)
            if x <= x_pos + width and i % 5 != 0:  # Skip major lines
                line = Line(x, y_pos, x, y_pos + height,
                          strokeColor=minor_color, strokeWidth=0.3)
                drawing.add(line)
        
        # Horizontal major lines (every 5mm)
        num_major_y = int(height / box_height_points) + 1
        for i in range(num_major_y):
            y = y_pos + (i * box_height_points)
            if y <= y_pos + height:
                line = Line(x_pos, y, x_pos + width, y,
                          strokeColor=major_color, strokeWidth=0.5)
                drawing.add(line)
        
        # Horizontal minor lines (every 1mm)
        num_minor_y = int(height / minor_box_height) + 1
        for i in range(num_minor_y):
            y = y_pos + (i * minor_box_height)
            if y <= y_pos + height and i % 5 != 0:  # Skip major lines
                line = Line(x_pos, y, x_pos + width, y,
                          strokeColor=minor_color, strokeWidth=0.3)
                drawing.add(line)
    
    # Function to create ECG drawing for a lead
    def create_lead_drawing(lead_name, lead_data, x_pos, y_pos, width, height):
        """Create ECG drawing for a single lead; returns (ecg_path, notch_path, dotted_path)"""
        if lead_data is None or len(lead_data) == 0 or width <= 0:
            return None
        
        # Get lead-specific ADC per box multiplier
        adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead_name, 6400.0)
        
        # Convert to numpy array
        adc_data = np.array(lead_data, dtype=float)

        # Apply report filters (DFT -> EMG -> AC) on lead data
        try:
            from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
            dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
            emg_setting = str(settings_manager.get_setting("filter_emg", "off")).strip()
            ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
            if dft_setting not in ("off", ""):
                adc_data = apply_dft_filter(adc_data, float(report_sampling_rate), dft_setting)
            if emg_setting not in ("off", ""):
                adc_data = apply_emg_filter(adc_data, float(report_sampling_rate), emg_setting)
            if ac_setting in ("50", "60"):
                adc_data = apply_ac_filter(adc_data, float(report_sampling_rate), ac_setting)
        except Exception as filter_err:
            print(f" Report filter apply failed for {lead_name}: {filter_err}")
                
        # Apply baseline
        baseline_adc = ECG_BASELINE_ADC
        data_mean = np.mean(adc_data)
        if abs(data_mean - ECG_BASELINE_ADC) < 500:
            centered_adc = adc_data - baseline_adc
        else:
            centered_adc = adc_data
        # Match main ECG report: force centering on baseline
        centered_adc = centered_adc - np.mean(centered_adc)
        
        # Calculate ADC per box
        adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)
                
        # Convert to boxes offset
        boxes_offset = centered_adc / adc_per_box
        
        # Calculate Y positions
        center_y = y_pos + (height / 2.0)
        box_height_points = ECG_LARGE_BOX_MM_HEIGHT * mm_unit
        ecg_normalized = center_y + (boxes_offset * box_height_points)
                
        box_width_points = ECG_LARGE_BOX_MM_WIDTH * mm_unit
        ecg_width = (54.0 if lead_name == "II" else 27.0) * box_width_points
        if ecg_width <= 0:
            return None
        gap = mm_unit if lead_name in ["V1", "V2", "V3", "II"] else 0
        t = np.linspace(x_pos + gap, x_pos + ecg_width, len(adc_data))
        
        # Draw ECG waveform
        ecg_path = Path(fillColor=None, 
                       strokeColor=colors.HexColor("#000000"), 
                       strokeWidth=0.4,
                       strokeLineCap=1,
                       strokeLineJoin=1)
        
        ecg_path.moveTo(t[0], ecg_normalized[0])
        for i in range(1, len(t)):
            ecg_path.lineTo(t[i], ecg_normalized[i])
                
        # Calibration notch (only for V1, V2, V3, II)
        notch_path = None
        if lead_name in ["V1", "V2", "V3", "II"]:
            notch_width_mm = 5.0   # width 5mm
            notch_height_mm = 10.0 # height 10mm
            notch_width = notch_width_mm * mm_unit
            notch_height = notch_height_mm * mm_unit
            
            # Place notch inside first box, shifted left (near first-half)
            first_box_width = 5.0 * mm_unit
            notch_x = x_pos + (first_box_width * -1.5)
            notch_y_base = center_y
            
            notch_path = Path(
                fillColor=None,
                strokeColor=colors.HexColor("#000000"),
                strokeWidth=0.8,
                strokeLineCap=1,
                strokeLineJoin=0
            )
            notch_path.moveTo(notch_x, notch_y_base)
            notch_path.lineTo(notch_x, notch_y_base + notch_height)
            notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
            notch_path.lineTo(notch_x + notch_width, notch_y_base)
            # Small forward tick to the right (extra 2mm) for clearer notch end
            notch_path.lineTo(notch_x + notch_width + (2.0 * mm_unit), notch_y_base)
        
        # Dotted continuation / markers
        dotted_path = None
        if lead_name in ["V1", "V2", "V3"]:
            # Vertical dotted marker at end of graph (after 27 boxes ≈ 140.7mm)
            dotted_path = Path(
                fillColor=None,
                strokeColor=colors.HexColor("#000000"),
                strokeWidth=0.4,
                strokeLineCap=1,
                strokeLineJoin=1,
                strokeDashArray=[2, 3]
            )
            marker_x = x_pos + (27.0 * box_width_points)  # Mark at end of 27-box width
            dotted_path.moveTo(marker_x, y_pos)
            dotted_path.lineTo(marker_x, y_pos + height)
        elif width > ecg_width:
            dotted_path = Path(
                fillColor=None,
                strokeColor=colors.HexColor("#000000"),
                strokeWidth=0.4,
                strokeLineCap=1,
                strokeLineJoin=1,
                strokeDashArray=[2, 3]
            )
            dotted_path.moveTo(x_pos + ecg_width, center_y)
            dotted_path.lineTo(x_pos + width, center_y)
        
        return ecg_path, notch_path, dotted_path
    
    # Layout configuration
    # Left column: V1, V2, V3 (stacked vertically)
    # Right column: V4, V5, V6 (stacked vertically)
    # Bottom: Lead II (full width)
    
    lead_height = 60  # Height per lead
    lead_spacing = 10  # Spacing between leads
    column_width = 360  # Width for each column
    left_col_x = 0    # Further left
    right_col_x = 360  # Further left
    
    # Y positions for V1-V6 (starting from top, below patient info) - SHIFTED DOWN by 20 points
    v_leads_y_start = 380  # Shifted down by 20 points (400 -> 380)
    v_leads_y_positions = [
        v_leads_y_start,  # V1/V4
        v_leads_y_start - (lead_height + lead_spacing),  # V2/V5
        v_leads_y_start - 2 * (lead_height + lead_spacing),  # V3/V6
    ]
    
    # Lead II position (bottom, full width) - SHIFTED DOWN by 20 points
    lead_ii_y = 130  # Shifted down by 20 points (150 -> 130)
    lead_ii_height = 80
    lead_ii_x = 0
    lead_ii_width = total_width - 10  # Wider span after shifting left
    
    # Get V1-V6 data from saved ECG data
    v_leads_data = {}
    if saved_ecg_data and 'leads' in saved_ecg_data:
        for lead_name in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            if lead_name in saved_ecg_data['leads']:
                lead_data = saved_ecg_data['leads'][lead_name]
                if isinstance(lead_data, list) and len(lead_data) > 0:
                    lead_array = np.array(lead_data, dtype=float)
                    if num_samples_to_capture and len(lead_array) > num_samples_to_capture:
                        lead_array = lead_array[-num_samples_to_capture:]
                    v_leads_data[lead_name] = lead_array
                elif isinstance(lead_data, np.ndarray) and len(lead_data) > 0:
                    lead_array = lead_data
                    if num_samples_to_capture and len(lead_array) > num_samples_to_capture:
                        lead_array = lead_array[-num_samples_to_capture:]
                    v_leads_data[lead_name] = lead_array
    
    # Draw V1-V6 leads
    left_leads = ['V1', 'V2', 'V3']
    right_leads = ['V4', 'V5', 'V6']
    
    for idx, lead_name in enumerate(left_leads):
        y_pos = v_leads_y_positions[idx]
        if lead_name in v_leads_data:
            graph_x = left_col_x + 10 - (4 * mm_unit)  # shift 4mm further left
            graph_y = y_pos
            graph_width = column_width - 40
            
            # Add lead label
            lead_label = String(left_col_x + 5, y_pos + lead_height - 10, lead_name,
                               fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # Create and add ECG drawing
            result = create_lead_drawing(lead_name, v_leads_data[lead_name],
                                          graph_x, graph_y, graph_width, lead_height)
            if result:
                ecg_path, notch_path, dotted_path = result
                if ecg_path:
                    master_drawing.add(ecg_path)
                if notch_path:
                    master_drawing.add(notch_path)
                if dotted_path:
                    master_drawing.add(dotted_path)
    
    # Shift V4, V5, V6 graphs and labels 20mm to the right
    v4_v5_v6_shift = 20.0 * mm_unit  # 20mm shift to the right
    
    for idx, lead_name in enumerate(right_leads):
        y_pos = v_leads_y_positions[idx]
        if lead_name in v_leads_data:
            # V4, V5, V6 should be in right column, shifted 20mm to the right
            graph_x = right_col_x + 10 - (4 * mm_unit) + v4_v5_v6_shift  # Shift 20mm to the right
            graph_y = y_pos
            graph_width = column_width - 40 + (27.0 * mm_unit)  
            
            # Add lead label, also shifted 20mm to the right
            lead_label_x = right_col_x + 5 + v4_v5_v6_shift  # Shift label 20mm to the right
            lead_label = String(lead_label_x, y_pos + lead_height - 10, lead_name,
                               fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # Create and add ECG drawing
            result = create_lead_drawing(lead_name, v_leads_data[lead_name],
                                          graph_x, graph_y, graph_width, lead_height)
            if result:
                ecg_path, notch_path, dotted_path = result
                if ecg_path:
                    master_drawing.add(ecg_path)
                if notch_path:
                    master_drawing.add(notch_path)
                if dotted_path:
                    master_drawing.add(dotted_path)
                
    # Draw Lead II at bottom (full width)
    if lead_ii_data and len(lead_ii_data) > 0:
        # Get first segment of Lead II data (first 10 seconds or available)
        lead_ii_values = np.array([d['value'] for d in lead_ii_data[:5500]], dtype=float)
        
        graph_x = lead_ii_x + 10 - (4 * mm_unit)  # shift 4mm further left
        graph_y = lead_ii_y
        graph_width = lead_ii_width - 40
        
        # Add Lead II label
        lead_ii_label = String(lead_ii_x + 5, lead_ii_y + lead_ii_height - 10, "Lead II",
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
        master_drawing.add(lead_ii_label)
        
        # Create and add Lead II ECG drawing
        result = create_lead_drawing('II', lead_ii_values,
                                      graph_x, graph_y, graph_width, lead_ii_height)
        if result:
            ecg_path, notch_path, dotted_path = result
            if ecg_path:
                master_drawing.add(ecg_path)
            if notch_path:
                master_drawing.add(notch_path)
            if dotted_path:
                master_drawing.add(dotted_path)
        print(f" Added Lead II graph at bottom ({len(lead_ii_values)} samples)")
                
    successful_graphs = len([l for l in left_leads + right_leads if l in v_leads_data]) + (1 if lead_ii_data else 0)
    print(f" Created {successful_graphs} ECG graphs: V1-V6 ({len([l for l in left_leads + right_leads if l in v_leads_data])}) + Lead II")
    
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
    
    date_label = String(670, 515, f"Date: {date_part}",  # Shifted RIGHT by 30: 640→670
                       fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(date_label)
    
    time_label = String(670,  500, f"Time: {time_part}",  # Shifted RIGHT by 30: 640→670
                       fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(time_label)

    # Org. and Phone No. labels below Date/Time (15 points below Time)
    org_label = String(670, 485, f"Org: {patient_org}",  # Shifted RIGHT by 30: 640→670
                    fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(org_label)
    
    phone_label = String(670, 470, f"Phone No: {patient_doctor_mobile}",  # Shifted RIGHT by 30: 640→670
                      fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(phone_label)
    
    # ==================== VITAL PARAMETERS (LANDSCAPE MODE - 2 COLUMNS SIDE BY SIDE) ====================
    # Page 2: Use metrics from hyper_metric.json
    if 'original_metrics_from_json' in locals() and original_metrics_from_json:
        HR = original_metrics_from_json.get('HR', 0)
        PR = original_metrics_from_json.get('PR', 0)
        QRS = original_metrics_from_json.get('QRS', 0)
        QT = original_metrics_from_json.get('QT', 0)
        QTc = original_metrics_from_json.get('QTc', 0)
        ST = original_metrics_from_json.get('ST', 0)
        
        # Calculate RR from HR
        if HR > 0:
            RR = int(60000 / HR)
        else:
            RR = original_metrics_from_json.get('RR_ms', 0)
        
        print(f" Page 2 (ECG waves): HR={HR} bpm")
        print(f"   Other metrics from hyper_metric.json: PR={PR}, QRS={QRS}, QT={QT}, QTc={QTc}, ST={ST}, RR={RR}")
    else:
        # Fallback: if hyper_metric.json not available, use zeros
        HR = 0
        PR = 0
        QRS = 0
        QT = 0
        QTc = 0
        ST = 0
        RR = 0
        print(" Page 2 (ECG waves): No metrics available in hyper_metric.json, using zeros")
    
    # LEFT COLUMN (130, y) - HR, PR, QRS, RR (SHIFTED UP BY 20 POINTS)
    hr_label = String(210, 548, f"HR    : {HR} bpm", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(hr_label)
    
    pr_label = String(210, 528, f"PR    : {PR} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(pr_label)
    
    qrs_label = String(210, 508, f"QRS : {QRS} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qrs_label)
    
    rr_label = String(210, 490, f"RR    : {RR} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rr_label)
    
    # QT (LEFT COLUMN - below RR)
    qt_label = String(210, 470, f"QT    : {QT} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qt_label)
    
    # QTc (LEFT COLUMN - below QT)
    qtc_label = String(210, 450, f"QTc   : {QTc} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtc_label)
    
    # RIGHT COLUMN (350, y) - Add dummy labels to match table layout
    # P/QRS/T Axis (SAME LINE AS HR)
    p_qrs_label = String(350, 548, "P/QRS/T  : --/--/--°", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(p_qrs_label)
    
    # RV5/SV1 (SAME LINE AS PR)
    rv5_sv_label = String(350, 528, "RV5/SV1  : -- mV/-- mV", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv_label)
    
    # RV5+SV1 (SAME LINE AS QRS)
    rv5_sv1_sum_label = String(350, 508, "RV5+SV1 : -- mV", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv1_sum_label)
    
    # QTcF (SAME LINE AS RR)
    qtcf_label = String(350, 490, "QTCF       : --", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_label)
    
    # ST (SAME LINE AS QT - shifted up by 15)
    st_label = String(350, 470, f"ST            : {ST} ms", fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(st_label)
    
    # Filter Band and Speed/Gain (merged in one line)
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
        350,
        450,  # Same position as Filter Band was
        f"{wave_speed_mm_s} mm/s   {filter_band}   AC : {ac_frequency}   {wave_gain_mm_mv} mm/mV",
        fontSize=10,
        fontName="Helvetica",
        fillColor=colors.black,
    ))
    
    # ==================== DOCTOR INFO (LANDSCAPE MODE - POSITIONED INSIDE DRAWING) ====================
    
    doctor = doctor or ""
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
    
    # ==================== CONCLUSION BOX ON PAGE 2 (LANDSCAPE MODE - ADJUSTED FOR 780 WIDTH) ====================
    
    conclusion_y_start = 45.0  # Positioned to avoid overlap with graphs
    conclusion_x_start = 280  # Shifted right for better positioning
    
    # Conclusion box (WIDER for landscape - adjusted for 780 total width)
    conclusion_box = Rect(conclusion_x_start, conclusion_y_start - 55, 490, 75,  # Bottom at -10, top at 65
                         fillColor=None, strokeColor=colors.black, strokeWidth=1.5)
    master_drawing.add(conclusion_box)
    
    # Conclusion header (CENTER adjusted for new X position and width)
    conclusion_header = String(conclusion_x_start + 245, conclusion_y_start + 8, "✦ CONCLUSION ✦", 
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
    
    print(f" Added Patient Info, Vital Parameters, {len(filtered_conclusions)} Conclusions to Hyperkalemia report")
    
    # Add master drawing to story (NO spacer to avoid creating 3rd page)
    story.append(master_drawing)
    # story.append(Spacer(1, 15))  # REMOVED - was creating unwanted 3rd page
    
    print(f" Added master drawing with V1-V6 leads + Lead II graph")
    print(f" Story contains {len(story)} elements before PDF build")
    
    # Build PDF (1 page only: Landscape with V1-V6 Leads + Lead II)
    doc.build(story)
    if not os.path.exists(filename):
        raise RuntimeError(f"Report not saved: {filename}")
    print(f" Hyperkalemia ECG Report generated: {filename}")
    print(f"    Page 1: V1-V6 Leads (2 columns) + Lead II (bottom) (Landscape)")
    
    return filename


# ==================== END OF Hyperkalemia ECG REPORT GENERATION ====================
