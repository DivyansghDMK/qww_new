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
import os
import logging
from datetime import datetime

# Setup logging for edge trimming debugging
log_dir = "/Users/indresh/Desktop/2_mar/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"4_3_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console_handler)

import numpy as np

# ------------------------ ECG grid scale constants ------------------------
# 40 large boxes across A4 height (210mm) => 1 large box = 5.25mm
ECG_BASE_BOX_MM = 5.0
ECG_LARGE_BOX_MM = 210.0 / 40.0
ECG_SMALL_BOX_MM = ECG_LARGE_BOX_MM / 5.0
# Column width for 4x3 layout: 3 columns of 17.5 boxes each (total 18 boxes per column budget)
COLUMN_BOXES = 17.5
# Scale wave speed so 1 second equals 5 large boxes at 25 mm/s on 40-box grid
ECG_SPEED_SCALE = ECG_LARGE_BOX_MM / ECG_BASE_BOX_MM
FOUR_THREE_SAMPLES_COLUMN = 1750  # 17.5 boxes * 100 samples per box (at 500Hz/5boxes_per_sec)
FOUR_THREE_SAMPLES_EXTRA_II = 5100 # 51 boxes

# matplotlib.use('Agg') # Removed to prevent main thread Qt canvas corruption

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
            saved_data["sampling_rate"] = float(ecg_test_page.sampler.sampling_rate)
    
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
    
    Important: Report ECG graph width = 17.5 boxes × ECG_LARGE_BOX_MM
     wave_speed time calculate factor use :
        Time from wave_speed = (graph_width_mm / effective_wave_speed_mm_s) seconds
    
    Formula:
        - Time window = (graph_width_mm / effective_wave_speed_mm_s) seconds ONLY
          ( 17.5 boxes × ECG_LARGE_BOX_MM total width)
        - BPM window is NOT used - only wave speed window
        - Beats = (BPM / 60) × time_window
        - Final window clamped maximum 20 seconds (NO minimum clamp)
    
    Returns: (time_window_seconds, num_samples)
    """
    # Calculate time window from wave_speed ONLY (BPM window NOT used)
    # Report ECG graph width = 17.5 boxes × ECG_LARGE_BOX_MM
    # Time = Distance / Speed (scaled for 40-box grid)
    ecg_graph_width_mm = COLUMN_BOXES * ECG_LARGE_BOX_MM
    effective_wave_speed_mm_s = wave_speed_mm_s * ECG_SPEED_SCALE
    calculated_time_window = ecg_graph_width_mm / max(1e-6, effective_wave_speed_mm_s)
    
    # Only clamp maximum to 20 seconds (NO minimum clamp)
    calculated_time_window = min(calculated_time_window, 20.0)
    
    # Calculate number of samples (assuming 500 Hz default)
    num_samples = int(calculated_time_window * 500.0)
    
    # Calculate expected beats: beats = (BPM / 60) × time_window
    # Formula: beats per second = BPM / 60, then multiply by time window
    beats_per_second = hr_bpm / 60.0 if hr_bpm > 0 else 0
    expected_beats = beats_per_second * calculated_time_window
    
    print(f" Time Window Calculation (Wave Speed ONLY):")
    print(f"   Graph Width: {ecg_graph_width_mm:.2f}mm (17.5 boxes × {ECG_LARGE_BOX_MM:.2f}mm)")
    print(f"   Wave Speed: {wave_speed_mm_s}mm/s (effective {effective_wave_speed_mm_s:.2f}mm/s)")
    print(f"   Time Window: {ecg_graph_width_mm:.2f} / {effective_wave_speed_mm_s:.2f} = {calculated_time_window:.2f}s")
    print(f"   BPM: {hr_bpm} → Beats per second: {hr_bpm}/60 = {beats_per_second:.2f} beats/sec")
    print(f"   Expected Beats: {beats_per_second:.2f} × {calculated_time_window:.2f} = {expected_beats:.1f} beats")
    print(f"   Estimated Samples: {num_samples} (at 500Hz)")
    
    return calculated_time_window, num_samples

def calculate_time_window_from_width_points(wave_speed_mm_s, width_points):
    """
    Calculate time window based on actual strip width in points.
    Width (mm) = width_points / mm, time = width_mm / wave_speed_mm_s
    """
    width_mm = width_points / mm
    effective_wave_speed_mm_s = wave_speed_mm_s * ECG_SPEED_SCALE
    return width_mm / max(1e-6, effective_wave_speed_mm_s)

def apply_report_ecg_filters(signal, sampling_rate, settings_manager):
    from ecg.ecg_filters import apply_ecg_filters, apply_baseline_wander_median_mean
    arr = np.asarray(signal, dtype=float)
    if arr.size < 10:
        return arr
    try:
        fs = float(sampling_rate)
    except Exception:
        fs = 500.0
    try:
        win_sec = float(str(settings_manager.get_setting("report_window_seconds", "10")).strip() or "10")
    except Exception:
        win_sec = 10.0
    window_samples = int(max(1.0, win_sec) * fs)
    if arr.size > window_samples:
        arr = arr[-window_samples:]
    pad_seconds = 4.0
    pad_samples = int(pad_seconds * fs)
    pad = min(pad_samples, max(0, arr.size - 1))
    if pad > 0:
        work = np.pad(arr, pad_width=pad, mode="reflect")
    else:
        work = arr
    dft_setting = str(settings_manager.get_setting("filter_dft", "0.5")).strip()
    emg_setting = str(settings_manager.get_setting("filter_emg", "150")).strip()
    ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip()
    dft_param = dft_setting if dft_setting not in ("off", "") else None
    emg_param = emg_setting if emg_setting not in ("off", "") else None
    ac_param = ac_setting if ac_setting not in ("off", "") else None
    filtered = apply_ecg_filters(
        work,
        sampling_rate=fs,
        ac_filter=ac_param,
        emg_filter=emg_param,
        dft_filter=dft_param,
    )
    filtered = apply_baseline_wander_median_mean(filtered, fs)
    try:
        if filtered.size > 5:
            from scipy.ndimage import gaussian_filter1d
            filtered = gaussian_filter1d(filtered, sigma=0.3)
    except Exception:
        pass
    try:
        n = filtered.size
        if n > 50:
            edge_len = max(50, int(0.10 * n))
            mid_start = n // 3
            mid_end = (2 * n) // 3
            start_std = np.std(np.diff(filtered[:edge_len]))
            end_std = np.std(np.diff(filtered[-edge_len:]))
            mid_std = np.std(np.diff(filtered[mid_start:mid_end])) + 1e-6
            start_trim = int(0.05 * n) if start_std > 1.5 * mid_std else 0
            end_trim = int(0.05 * n) if end_std > 1.5 * mid_std else 0
            if start_trim + end_trim < n - 20:
                filtered = filtered[start_trim:n - end_trim]
    except Exception:
        pass
    try:
        if filtered.size > 5:
            dif = np.diff(filtered, prepend=filtered[0])
            th = 5.0 * (np.std(dif) + 1e-6)
            bad = np.where(np.abs(dif) > th)[0]
            for i in bad:
                if 1 <= i < filtered.size - 1:
                    filtered[i] = 0.5 * (filtered[i - 1] + filtered[i + 1])
    except Exception:
        pass
    if pad > 0 and filtered.size > 2 * pad:
        filtered = filtered[pad:-pad]
    try:
        n2 = filtered.size
        if n2 > 50:
            edge_len2 = max(50, int(0.10 * n2))
            mid_start2 = n2 // 3
            mid_end2 = (2 * n2) // 3
            start_std2 = np.std(np.diff(filtered[:edge_len2]))
            end_std2 = np.std(np.diff(filtered[-edge_len2:]))
            mid_std2 = np.std(np.diff(filtered[mid_start2:mid_end2])) + 1e-6
            start_trim2 = int(0.05 * n2) if start_std2 > 1.3 * mid_std2 else 0
            end_trim2 = int(0.05 * n2) if end_std2 > 1.3 * mid_std2 else 0
            if start_trim2 + end_trim2 < n2 - 20:
                filtered = filtered[start_trim2:n2 - end_trim2]
    except Exception:
        pass
    try:
        n3 = filtered.size
        if n3 > 100:
            hard_trim = max(10, int(0.03 * n3))
            filtered = filtered[hard_trim:n3 - hard_trim]
    except Exception:
        pass
    try:
        n4 = filtered.size
        if n4 > 50:
            alpha = 0.5
            m = max(10, int((alpha * n4) / 2.0))
            if m * 2 < n4:
                ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
                w = np.ones(n4)
                w[:m] = ramp
                w[-m:] = ramp[::-1]
                mu = float(np.mean(filtered))
                filtered = mu + (filtered - mu) * w
    except Exception:
        pass
    return filtered

def create_ecg_grid_with_waveform(ecg_data, lead_name, width=6, height=2, sampling_rate=500.0):
    """
    Create ECG graph with pink grid background and dark ECG waveform
    Returns: matplotlib figure with pink ECG grid background
    """
    # Create figure with pink background
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
    fig = _Figure(figsize=(width, height), facecolor='#ffe6e6')
    _FCA(fig)
    ax = fig.add_subplot(111)
    
    # STEP 1: Create pink ECG grid background
    # ECG grid colors (even lighter pink/red like medical ECG paper)
    light_grid_color = '#ffd1d1'  # Darker minor grid
    major_grid_color = '#ffb3b3'  # Darker major grid
    bg_color = '#ffe6e6'  # Very light pink background
    
    # Set both figure and axes background to pink
    fig.patch.set_facecolor(bg_color)  # Figure background pink
    ax.set_facecolor(bg_color)         # Axes background pink
    
    # STEP 2: Draw pink ECG grid lines
    # ECG grid colors (even lighter pink/red like medical ECG paper)
    light_grid_color = '#ffd1d1'  # Darker minor grid
    major_grid_color = '#ffb3b3'  # Darker major grid
    bg_color = '#ffe6e6'  # Very light pink background
    
    # Set both figure and axes background to pink
    fig.patch.set_facecolor(bg_color)  # Figure background pink
    ax.set_facecolor(bg_color)         # Axes background pink
    
    # Minor grid lines (scaled for 40-box grid)
    minor_spacing = ECG_SMALL_BOX_MM
    
    # Draw vertical minor pink grid lines
    num_minor_x = int(width / minor_spacing) + 1
    for i in range(num_minor_x):
        x_pos = i * minor_spacing
        ax.axvline(x=x_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Draw horizontal minor pink grid lines
    num_minor_y = int(height / minor_spacing) + 1
    for i in range(num_minor_y):
        y_pos = i * minor_spacing
        ax.axhline(y=y_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Major grid lines (ECG_LARGE_BOX_MM spacing)
    major_spacing = ECG_LARGE_BOX_MM
    
    # Draw vertical major pink grid lines
    num_major_x = int(width / major_spacing) + 1
    for i in range(num_major_x):
        x_pos = i * major_spacing
        ax.axvline(x=x_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # Draw horizontal major pink grid lines
    num_major_y = int(height / major_spacing) + 1
    for i in range(num_major_y):
        y_pos = i * major_spacing
        ax.axhline(y=y_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # STEP 3: Plot DARK ECG waveform on top of pink grid
    if ecg_data is not None and len(ecg_data) > 0:
        # Scale ECG data to fit in the grid (25mm/s * SCALE)
        fs = float(sampling_rate)
        t_sec = np.arange(len(ecg_data)) / fs
        x_mm = t_sec * 25.0 * ECG_SPEED_SCALE
        
        # Limit to width
        mask = x_mm <= width
        x_plot = x_mm[mask]
        data_plot = np.array(ecg_data)[mask]
        
        # Normalize amplitude CONSISTENTLY (10mm/mV standard)
        med_abs = np.nanmedian(np.abs(data_plot)) if len(data_plot) else 0.0
        # Consistent amplitude normalization for all leads
        if med_abs > 500:
            ecg_mv = (data_plot - 2000.0) / 6400.0  # Raw ADC to mV conversion
        else:
            ecg_mv = data_plot  # Already in mV
            
        y_mm = (height / 2.0) + (ecg_mv * 10.0)
        
        # DARK ECG LINE - clearly visible on pink grid
        ax.plot(x_plot, y_mm, color='#000000', linewidth=2.8, solid_capstyle='round', alpha=0.9)
    
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
    
    # STEP 2: Draw pink ECG grid lines (GE/Philips fixed diagnostic scale)
    # Fixed diagnostic grid requirements:
    # Minor: 0.04s / 0.1mV (1mm at 25mm/s, 10mm/mV)
    # Major: 0.20s / 1.0mV (5mm at 25mm/s, 10mm/mV)
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid
    
    from reportlab.lib.units import mm
    
    # Minor grid: 1mm spacing
    minor_spacing_mm = 1.0 * mm
    
    # Vertical minor lines
    num_minor_x = int(width / minor_spacing_mm) + 1
    for i in range(num_minor_x):
        x_pos = i * minor_spacing_mm
        if x_pos <= width:
            line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
            drawing.add(line)
    
    # Horizontal minor lines
    num_minor_y = int(height / minor_spacing_mm) + 1
    for i in range(num_minor_y):
        y_pos = i * minor_spacing_mm
        if y_pos <= height:
            line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
            drawing.add(line)
    
    # Major grid: 5mm horizontal, 10mm vertical
    major_spacing_x_mm = 5.0 * mm
    major_spacing_y_mm = 10.0 * mm
    
    # Vertical major lines
    num_major_x = int(width / major_spacing_x_mm) + 1
    for i in range(num_major_x):
        x_pos = i * major_spacing_x_mm
        if x_pos <= width:
            line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
            drawing.add(line)
    
    # Horizontal major lines (every 1.0mV = 10mm)
    num_major_y = int(height / major_spacing_y_mm) + 1
    for i in range(num_major_y):
        y_pos = i * major_spacing_y_mm
        if y_pos <= height:
            line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
            drawing.add(line)
    
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
    
    # FORCE REAL DATA ONLY - DISABLE DEMO MODE FOR REPORTS
    is_demo_mode = False  # Always use real data for reports
    time_window_seconds = None
    print(" 4_3 REPORT GENERATION: Using REAL data only (Demo mode disabled for consistency)")
    
    # Try to get ACTUAL sampling rate from the test page
    if ecg_test_page and hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            samples_per_second = float(ecg_test_page.sampler.sampling_rate)
            print(f" Found sampler sampling rate: {samples_per_second}Hz")
    elif ecg_test_page and hasattr(ecg_test_page, 'sampling_rate'):
        if ecg_test_page.sampling_rate:
            samples_per_second = float(ecg_test_page.sampling_rate)
            print(f" Found page sampling rate: {samples_per_second}Hz")
    
    samples_per_second = samples_per_second or 500.0 # Default to 500Hz for diagnostic accuracy
    # Try to get REAL ECG data from the test page
    real_ecg_data = {}
    if ecg_test_page and hasattr(ecg_test_page, 'data'):
        
        # ALWAYS USE REAL DATA - No demo mode windowing
        num_samples_to_capture = 10000
        print(f" REAL DATA MODE: Capturing up to {num_samples_to_capture} samples from live buffer")
        
        for lead in ordered_leads:
            if lead == "-aVR":
                # For -aVR, we need to invert aVR data
                if hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 3:
                    avr_data = np.array(ecg_test_page.data[3])  # aVR is at index 3
                    real_ecg_data[lead] = -avr_data[-num_samples_to_capture:]
                    print(f" Captured REAL -aVR data: {len(real_ecg_data[lead])} points")
            else:
                lead_index = lead_to_index.get(lead)
                if lead_index is not None and len(ecg_test_page.data) > lead_index:
                    lead_data = np.array(ecg_test_page.data[lead_index])
                    if len(lead_data) > 0:
                        real_ecg_data[lead] = lead_data[-num_samples_to_capture:]
                        print(f" Captured REAL {lead} data: {len(real_ecg_data[lead])} points")
                    else:
                        print(f" No data found for {lead}")
                else:
                    print(f" Lead {lead} index not found")
    else:
        print(" No live ECG test page found - using grid only")
    
    # Get wave_gain from settings_manager for amplitude scaling (FULLY DYNAMIC)
    wave_gain_mm_mv = None
    if settings_manager:
        try:
            wave_gain_setting = settings_manager.get_setting("wave_gain")
            wave_gain_mm_mv = float(wave_gain_setting) if wave_gain_setting else None
            print(f" Using wave_gain from ecg_settings.json: {wave_gain_mm_mv} mm/mV (for amplitude scaling)")
        except Exception:
            wave_gain_mm_mv = None
            print(f" Could not get wave_gain from settings")
    
    if wave_gain_mm_mv is None:
        print(" ERROR: wave_gain not found in settings - cannot generate report")
        return {}
    
    # Apply report filters (AC/EMG/DFT) based on current settings
    filtered_ecg_data = {}
    for lead, signal in real_ecg_data.items():
        try:
            filtered_ecg_data[lead] = apply_report_ecg_filters(signal, samples_per_second, settings_manager)
        except Exception:
            filtered_ecg_data[lead] = signal

    # Create ReportLab drawings with REAL (filtered) data
    for lead in ordered_leads:
        try:
            # Create ReportLab drawing with REAL ECG data (with wave_gain applied)
            drawing = create_reportlab_ecg_drawing_with_real_data(
                lead, 
                filtered_ecg_data.get(lead), 
                width=460, 
                height=45,
                wave_gain_mm_mv=wave_gain_mm_mv,
                sampling_rate=samples_per_second,
                settings_manager=settings_manager
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

# ADC-per-box configuration (shared across report sections; kept near baseline/plot logic).
ADC_PER_BOX_CONFIG = {
    'I': 6400.0,
    'II': 6400.0,
    'III': 6400.0,
    'aVR': 6400.0,
    'aVL': 6400.0,
    'aVF': 6400.0,
    'V1': 6400.0,
    'V2': 6400.0,
    'V3': 6400.0,
    'V4': 6400.0,
    'V5': 6400.0,
    'V6': 6400.0,
    '-aVR': 6400.0,  # For Cabrera sequence
}

def create_reportlab_ecg_drawing_with_real_data(lead_name, ecg_data, width=460, height=45, wave_gain_mm_mv=None, sampling_rate=500.0, settings_manager=None):
    """
    Create ECG drawing using ReportLab with REAL ECG data (GE/Philips fixed diagnostic scale).
    
    Fixed diagnostic scale requirements:
    - Speed = 25 mm/s (no autoscale)
    - Gain = 10 mm/mV (no autoscale)
    - Grid: Minor 0.04s/0.1mV, Major 0.20s/1.0mV
    - Doctor must be able to count squares
    
    Parameters:
        wave_gain_mm_mv: Wave gain in mm/mV (MUST be 10.0 for diagnostic reports)
        sampling_rate: The sampling rate of the ECG data (e.g., 500.0 or 150.0)
    """
    # VALIDATION: Ensure fixed diagnostic scale
    # (Allowing some flexibility but warning if not 10.0)
    if abs(wave_gain_mm_mv - 10.0) > 0.1:
        print(f"⚠️ Warning: Report using non-standard gain {wave_gain_mm_mv} mm/mV")
    
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines (GE/Philips fixed diagnostic scale)
    # Fixed diagnostic grid requirements:
    # Minor: 0.04s / 0.1mV
    # Major: 0.20s / 1.0mV
    # At 25 mm/s: 0.04s = 1mm, 0.20s = 5mm
    # NOTE: On this 40-box grid, we scale the boxes to ECG_LARGE_BOX_MM
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid
    
    from reportlab.lib.units import mm
    
    # Minor grid: 1/5 of a large box
    minor_spacing_mm = ECG_SMALL_BOX_MM
    minor_spacing_x_points = minor_spacing_mm * mm
    minor_spacing_y_points = minor_spacing_mm * mm
    
    # Vertical minor lines
    num_minor_x = int(width / minor_spacing_x_points) + 1
    for i in range(num_minor_x):
        x_pos = i * minor_spacing_x_points
        if x_pos <= width:
            line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
            drawing.add(line)
    
    # Horizontal minor lines
    num_minor_y = int(height / minor_spacing_y_points) + 1
    for i in range(num_minor_y):
        y_pos = i * minor_spacing_y_points
        if y_pos <= height:
            line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
            drawing.add(line)
    
    # Major grid: ECG_LARGE_BOX_MM spacing
    major_spacing_x_mm = ECG_LARGE_BOX_MM * mm
    major_spacing_y_mm = ECG_LARGE_BOX_MM * mm
    
    # Vertical major lines
    num_major_x = int(width / major_spacing_x_mm) + 1
    for i in range(num_major_x):
        x_pos = i * major_spacing_x_mm
        if x_pos <= width:
            line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
            drawing.add(line)
    
    # Horizontal major lines
    num_major_y = int(height / major_spacing_y_mm) + 1
    for i in range(num_major_y):
        y_pos = i * major_spacing_y_mm
        if y_pos <= height:
            line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
            drawing.add(line)
    
    # STEP 3: Plot ECG in fixed diagnostic scale (25 mm/s, 10 mm/mV) with scaling for 40-box grid
    if ecg_data is None or len(ecg_data) == 0:
        print(f" No real data available for {lead_name} - showing grid only")
        return drawing

    # Fixed scale parameters - DYNAMIC wave_speed from settings
    if settings_manager:
        try:
            wave_speed_setting = settings_manager.get_setting("wave_speed")
            speed_mm_s = float(wave_speed_setting) if wave_speed_setting else None
        except Exception:
            speed_mm_s = None
    
    if speed_mm_s is None:
        print(" ERROR: wave_speed not found in settings - cannot generate drawing")
        return drawing
        
    width_mm = width / mm  # Convert width points to mm
    total_seconds = width_mm / (speed_mm_s * ECG_SPEED_SCALE)
    height_mm_physical = height / mm  # Convert height points to mm

    ecg_array = np.asarray(ecg_data, dtype=float)
    
    # ORIGINAL WORKING APPROACH: Simple amplitude normalization like ecg_report_generator.py
    med_abs = np.nanmedian(np.abs(ecg_array)) if len(ecg_array) else 0.0
    if med_abs > 20.0:  # Use 20.0 threshold like ecg_report_generator.py
        ecg_mv = ecg_array / 1000.0  # Simple division by 1000
    else:
        ecg_mv = ecg_array  # Already in mV
    
    # FORCEFUL STRAIGHTENING: Force straight baseline - no exceptions
    if len(ecg_mv) > 0:
        # Remove DC offset first
        dc_offset = np.nanmean(ecg_mv)  # Use mean for gentler removal
        ecg_mv = ecg_mv - dc_offset
        
        # FORCEFUL STRAIGHTENING: Always remove any slope
        if len(ecg_mv) > 20:
            x = np.arange(len(ecg_mv))
            # Fit linear trend - ALWAYS remove it regardless of slope
            coeffs = np.polyfit(x, ecg_mv, 1)  # Linear fit
            slope = coeffs[0]
            trend = np.polyval(coeffs, x)
            ecg_mv = ecg_mv - trend  # ALWAYS remove trend - no threshold
            print(f" {lead_name}: FORCEFUL: Removed slope={slope:.6f} (no threshold)")
            
            # Edge conditioning for all BPM: smooth approach + hard-flat terminal segment.
            edge_samples = min(80, max(20, len(ecg_mv) // 16))
            if len(ecg_mv) > edge_samples * 3:
                t = np.linspace(0.0, 1.0, edge_samples)
                taper = 0.5 * (1.0 + np.cos(np.pi * t))
                ecg_mv[:edge_samples] = ecg_mv[:edge_samples] * (1.0 - taper)
                ecg_mv[-edge_samples:] = ecg_mv[-edge_samples:] * taper

                flat_tail = max(10, edge_samples // 4)
                blend = max(8, edge_samples // 5)
                if len(ecg_mv) > flat_tail + blend:
                    blend_start = len(ecg_mv) - (flat_tail + blend)
                    blend_end = len(ecg_mv) - flat_tail
                    ramp = np.linspace(1.0, 0.0, blend)
                    ecg_mv[blend_start:blend_end] = ecg_mv[blend_start:blend_end] * ramp
                    ecg_mv[-flat_tail:] = 0.0
                print(f" {lead_name}: Applied edge taper + flat tail ({flat_tail} samples)")
    
    # ORIGINAL WORKING GAIN: Calculate y_mm AFTER slope removal
    y_mm = ecg_mv * wave_gain_mm_mv
    baseline_mm = height_mm_physical / 2.0
    y_mm = baseline_mm + y_mm
    
    # PROPER ECG TIME WINDOWING: Use fixed time window instead of scaling
    # This ensures medically accurate beat representation
    effective_speed_mm_s = speed_mm_s * ECG_SPEED_SCALE
    max_display_seconds = width_mm / effective_speed_mm_s
    max_samples = int(max_display_seconds * fs)
    
    # Take only the data that fits in the time window (no scaling distortion)
    if len(t_sec) > max_samples:
        display_t_sec = t_sec[:max_samples]
        display_ecg_mv = ecg_mv[:max_samples]
        print(f" Time window: showing {max_samples} samples ({max_display_seconds:.2f}s) of {len(ecg_mv)} available")
    else:
        display_t_sec = t_sec
        display_ecg_mv = ecg_mv
        print(f" Full data fits: {len(display_ecg_mv)} samples ({len(display_ecg_mv)/fs:.2f}s)")
    
    # Convert to physical mm positions (no scaling - medically accurate)
    x_mm = display_t_sec * effective_speed_mm_s
    
    # Update variables for the rest of the processing
    t_sec = display_t_sec
    ecg_mv = display_ecg_mv

    # Clip to panel
    y_mm = np.clip(y_mm, 0.0, height_mm_physical)
    x_mm = np.clip(x_mm, 0.0, width_mm)

    # Convert to points for plotting
    ecg_color = colors.HexColor("#000000")
    for i in range(len(x_mm) - 1):
        x1 = x_mm[i] * mm
        y1 = y_mm[i] * mm
        x2 = x_mm[i+1] * mm
        y2 = y_mm[i+1] * mm
        drawing.add(Line(x1, y1, x2, y2, strokeColor=ecg_color, strokeWidth=0.6))
    
    print(f" Drew {len(x_mm)} ECG data points for {lead_name} in fixed diagnostic scale at {fs}Hz")
    return drawing

def create_clean_ecg_image(lead_name, width=6, height=2):
    """
    Create COMPLETELY CLEAN ECG image with GUARANTEED pink background
    NO labels, NO time markers, NO axes, NO white background
    """
    # FORCE matplotlib to use proper backend
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # STEP 1: Create figure with FORCED pink background
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
    fig = _Figure(figsize=(width, height), facecolor='#ffe6e6')
    _FCA(fig)
    
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
    
    # Minor grid lines (scaled for 40-box grid)
    minor_spacing = ECG_SMALL_BOX_MM
    
    # Draw vertical minor pink grid lines
    num_minor_x = int(width / minor_spacing) + 1
    for i in range(num_minor_x):
        x_pos = i * minor_spacing
        ax.axvline(x=x_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Draw horizontal minor pink grid lines
    num_minor_y = int(height / minor_spacing) + 1
    for i in range(num_minor_y):
        y_pos = i * minor_spacing
        ax.axhline(y=y_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Major grid lines (ECG_LARGE_BOX_MM spacing)
    major_spacing = ECG_LARGE_BOX_MM
    
    # Draw vertical major pink grid lines
    num_major_x = int(width / major_spacing) + 1
    for i in range(num_major_x):
        x_pos = i * major_spacing
        ax.axvline(x=x_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # Draw horizontal major pink grid lines
    num_major_y = int(height / major_spacing) + 1
    for i in range(num_major_y):
        y_pos = i * major_spacing
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
def _draw_logo_and_footer_callback(canvas, doc_obj, patient=None):
    from reportlab.lib.units import mm
    
    # STEP 1: Draw pink ECG grid background on Page 1 (now the only landscape page)
    if canvas.getPageNumber() == 1:
        page_width, page_height = canvas._pagesize
        
        # ========== CONSISTENT 5.25MM BOXES (40 BOXES IN 210MM) ==========
        # Using ECG_LARGE_BOX_MM (5.25mm) for both width and height grid
        box_size_mm = ECG_LARGE_BOX_MM  # 5.25mm
        box_size_pts = box_size_mm * mm
        
        # Pink background - FULL PAGE
        canvas.setFillColor(colors.HexColor("#ffe6e6"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        
        # Grid colors
        light_grid_color = colors.HexColor("#ffd1d1")
        major_grid_color = colors.HexColor("#ffb3b3")
        
        # Minor grid lines - 5 minor boxes per major box
        minor_spacing_mm = box_size_mm / 5.0  # 1.05mm
        minor_spacing_pts = minor_spacing_mm * mm
        
        canvas.setStrokeColor(light_grid_color)
        canvas.setLineWidth(0.6)
        # Vertical minor lines
        x = 0
        while x <= page_width:
            canvas.line(x, 0, x, page_height)
            x += minor_spacing_pts
        
        # Horizontal minor lines
        y = 0
        while y <= page_height:
            canvas.line(0, y, page_width, y)
            y += minor_spacing_pts
        
        # Major grid lines
        canvas.setStrokeColor(major_grid_color)
        canvas.setLineWidth(0.6)
        # Vertical major lines
        x = 0
        while x <= page_width:
            canvas.line(x, 0, x, page_height)
            x += box_size_pts
        
        # Horizontal major lines
        y = 0
        while y <= page_height:
            canvas.line(0, y, page_width, y)
            y += box_size_pts
    
    # STEP 1.5: Draw Org. and Phone No. on Page 1 (REPOSITIONED - slightly higher, more left)
    if canvas.getPageNumber() == 1:
        canvas.saveState()
        # Portrait A4 height = 842 points, position very close to top
        page_height = 842  # A4 portrait height
        x_pos = 15  # More to the left (was 30, now 15)
        y_pos = page_height - 30
        
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(colors.black)
        org_label = "Org:"
        canvas.drawString(x_pos, y_pos, org_label)
        
        org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 10)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x_pos + org_label_width + 5, y_pos, patient.get("Org.", "") if patient else "")
        
        y_pos -= 15
        
        canvas.setFont("Helvetica-Bold", 10)
        phone_label = "Phone No:"
        canvas.drawString(x_pos, y_pos, phone_label)
        
        phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 10)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient.get("doctor_mobile", "") if patient else "")
        canvas.restoreState()
    
    # STEP 2: Draw logo (REPOSITIONED - lower from top)
    # Use resource_path helper for PyInstaller compatibility
    png_path = _get_resource_path("assets/Deckmountimg.png")
    webp_path = _get_resource_path("assets/Deckmount.webp")
    logo_path = png_path if os.path.exists(png_path) else webp_path
    
    if os.path.exists(logo_path):
        canvas.saveState()
        if canvas.getPageNumber() == 1:
            # Page 1 is now LANDSCAPE - logo at top right
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
            canvas.drawImage(logo_path, x, y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
        canvas.restoreState()
    
    # STEP 3: Footer
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.black)
    footer_text = "Deckmount Electronic , Plot No. 260, Phase IV, Udyog Vihar, Sector 18, Gurugram, Haryana 122015"
    text_width = canvas.stringWidth(footer_text, "Helvetica", 8)
    
    if canvas.getPageNumber() in [2, 3]:
        # Page 2 & 3 are LANDSCAPE
        page_width, page_height = canvas._pagesize
        x = (page_width - text_width) / 2
    else:
        # Page 1 is PORTRAIT
        x = (595 - text_width) / 2
    
    y = 10
    canvas.drawString(x, y, footer_text)
    canvas.restoreState()

    # STEP 2.5: Add Date and Time labels below logo on Page 1
    if canvas.getPageNumber() == 1:
        from datetime import datetime
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.black)
        
        # Get current date and time
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        time_str = now.strftime("%H:%M:%S")
        
        # Position below logo (logo ends at y, so date at y-15, time at y-30)
        # Page 1 is now LANDSCAPE - use same x position as logo (right aligned)
        page_width, page_height = canvas._pagesize
        logo_w = 120
        date_x = page_width - logo_w - 35  # Same x as logo
        time_x = page_width - logo_w - 35  # Same x as logo
        date_y = page_height - 40  # Right below logo (shifted up 15 points from -55)
        time_y = page_height - 55  # 15 points below logo (shifted up 15 points from -70)
        
        # Draw date label
        canvas.drawString(date_x, date_y, f"Date: {date_str}")
        # Draw time label  
        canvas.drawString(time_x, time_y, f"Time: {time_str}")
        
        # STEP 2.6: Add Org and Phone No. labels below date/time on Page 2
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.black)
        
        # Org label (15 points below time)
        org_y = page_height - 70  # 15 points below time (time_y = -55, so -55 - 15 = -70)
        org_label = "Org:"
        canvas.drawString(date_x, org_y, org_label)
        
        org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 9)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(date_x + org_label_width + 5, org_y, patient.get("Org.", "") if patient else "")
        
        # Phone No. label (15 points below org)
        phone_y = page_height - 85  # 15 points below org (org_y = -70, so -70 - 15 = -85)
        canvas.setFont("Helvetica-Bold", 9)
        phone_label = "Phone No:"
        canvas.drawString(date_x, phone_y, phone_label)
        
        phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 9)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(date_x + phone_label_width + 5, phone_y, patient.get("doctor_mobile", "") if patient else "")
        
        canvas.restoreState()

def generate_4_3_ecg_report(filename="ecg_report.pdf", data=None, lead_images=None, dashboard_instance=None, ecg_test_page=None, patient=None, ecg_data_file=None):
    """
    Generate ECG report PDF
    """
    from reportlab.lib.units import mm
    
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
    # Priority: ecg_settings.json wave_speed (FULLY DYNAMIC)
    wave_speed_setting = settings_manager.get_setting("wave_speed")
    wave_gain_setting = settings_manager.get_setting("wave_gain")
    
    if wave_speed_setting is None or wave_gain_setting is None:
        print(" ERROR: wave_speed or wave_gain not found in settings - cannot generate report")
        return None
        
    wave_speed_mm_s = _safe_float(wave_speed_setting, None)   # No default - must be in settings
    wave_gain_mm_mv = _safe_float(wave_gain_setting, None)    # No default - must be in settings
    
    if wave_speed_mm_s is None or wave_gain_mm_mv is None:
        print(" ERROR: Invalid wave_speed or wave_gain values - cannot generate report")
        return None
        
    print(f" Using wave_speed from ecg_settings.json: {wave_speed_mm_s} mm/s (for calculation-based beats)")
    
    # Try to get sampling rate from the test page first
    computed_sampling_rate = 500.0
    if ecg_test_page and hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            computed_sampling_rate = float(ecg_test_page.sampler.sampling_rate)
    elif ecg_test_page and hasattr(ecg_test_page, 'sampling_rate'):
        if ecg_test_page.sampling_rate:
            computed_sampling_rate = float(ecg_test_page.sampling_rate)

    data["wave_speed_mm_s"] = wave_speed_mm_s
    data["wave_gain_mm_mv"] = wave_gain_mm_mv

    print(f" Pre-plot checks: HR_bpm={hr_bpm_value}, RR_ms={data['RR_ms']}, wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV, sampling_rate={computed_sampling_rate}Hz")
    print(f" Calculation-based beats formula:")
    ecg_graph_width_mm = COLUMN_BOXES * ECG_LARGE_BOX_MM
    effective_wave_speed_mm_s = wave_speed_mm_s * ECG_SPEED_SCALE
    print(f"   Graph width: 17.5 boxes × {ECG_LARGE_BOX_MM:.2f}mm = {ecg_graph_width_mm:.2f}mm")
    print(f"   BPM window: (desired_beats × 60) / {hr_bpm_value} = {(6 * 60.0 / hr_bpm_value) if hr_bpm_value > 0 else 0:.2f}s")
    print(f"   Wave speed window: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {ecg_graph_width_mm / max(1e-6, effective_wave_speed_mm_s):.2f}s")
    
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
            computed_sampling_rate = saved_ecg_data.get('sampling_rate', computed_sampling_rate)
            print(f" Using sampling rate from provided file: {computed_sampling_rate} Hz")
    elif ecg_test_page and hasattr(ecg_test_page, 'data'):
        # ALWAYS save current data to file before generating report (REQUIRED for calculation-based beats)
        print(" Saving ECG data to file (required for calculation-based beats)...")
        saved_data_file_path = save_ecg_data_to_file(ecg_test_page)
        if saved_data_file_path:
            saved_ecg_data = load_ecg_data_from_file(saved_data_file_path)
            if saved_ecg_data:
                computed_sampling_rate = saved_ecg_data.get('sampling_rate', computed_sampling_rate)
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
                del fig
                
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

    # Create BaseDocTemplate for mixed page orientations (from hyperkalemia)
    doc = BaseDocTemplate(filename, pagesize=A4,
                         rightMargin=30, leftMargin=30,
                         topMargin=30, bottomMargin=30)
    # Create wrapper function for callback with patient parameter
    def callback_wrapper(canvas, doc):
        return _draw_logo_and_footer_callback(canvas, doc, patient)


    # Define Landscape template as DEFAULT (only template) with onPage callback
    landscape_width, landscape_height = landscape(A4)
    # Reduce margins to increase frame size (from 30 to 20 on each side)
    landscape_frame = Frame(20, 20,  # reduced margins for landscape to fit taller drawing
                           landscape_width - 40, landscape_height - 40,
                           id="landscape_frame")
    landscape_template = PageTemplate(id="landscape", frames=[landscape_frame],
                                     pagesize=landscape(A4), onPage=callback_wrapper)

    # Add ONLY landscape template as default
    doc.addPageTemplates([landscape_template])

    story = []
    styles = getSampleStyleSheet()
    
    # Skip title and other Page 1 content - go directly to landscape ECG content

    # Skip all Page 1 content - go directly to landscape ECG graphs
    # Patient details will be shown in the landscape page master drawing

    # Patient details for landscape page (will be used in master drawing)
    if patient is None:
        patient = {}
    first_name = patient.get("first_name", "")
    last_name = patient.get("last_name", "")
    full_name = f"{first_name} {last_name}".strip()
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    date_time_str = patient.get("date_time", "")

    # REMOVED: All Page 1 tables and content from story
    # Patient info, vital parameters and conclusions are now in master drawing above ECG graph
    print("Creating SINGLE drawing with all ECG content...")
    
    # Column width based on 17.5 boxes (data) + 0.5 boxes (gap) = 18 boxes total budget
    COLUMN_WIDTH_PTS = 18 * ECG_LARGE_BOX_MM * mm
    
    # Single drawing dimensions - ADJUSTED HEIGHT to fit within page frame (max ~770)
    total_width = 3 * COLUMN_WIDTH_PTS + 20   # Full width for 3 columns + small margin
    total_height = 540  # Reduced to 720 to fit within page frame (max ~770) with margin
    
    # Create ONE master drawing
    master_drawing = Drawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads based on selected sequence (START 90 points below QTc at 345)
    # Special arrangement for all rows: 3x4 grid with 259 points horizontal gaps, shifted 30 points down
    y_positions = [345, 295, 245, 195, 145, 95, 45, -5, -55, -105, -155, -205]  # Start 90 points below QTc (435-90=345)  
    6
    lead_positions = []
    
    for i, lead in enumerate(lead_order):
        if lead == "I":
            # Lead I at position 0, y=345
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 345
            })
        elif lead == "II":
            # Lead II at COLUMN_WIDTH_PTS, same row as Lead I
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  # 18 boxes gap from Lead I
                "y": 345   # Same y as Lead I
            })
        elif lead == "III":
            # Lead III at 2 * COLUMN_WIDTH_PTS, same row as Lead I
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  # 36 boxes gap from Lead I
                "y": 345   # Same y as Lead I
            })
        elif lead == "aVR":
            # aVR at position 0, next row (y=280) - shifted 15 points down
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 280
            })
        elif lead == "aVL":
            # aVL at COLUMN_WIDTH_PTS, same row as aVR
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  
                "y": 280   # Same y as aVR
            })
        elif lead == "aVF":
            # aVF at 2 * COLUMN_WIDTH_PTS, same row as aVR
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  
                "y": 280   # Same y as aVR
            })
        elif lead == "V1":
            # V1 at position 0, third row (y=215) - shifted 15 points down
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 215
            })
        elif lead == "V2":
            # V2 at COLUMN_WIDTH_PTS, same row as V1
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  
                "y": 215   # Same y as V1
            })
        elif lead == "V3":
            # V3 at 2 * COLUMN_WIDTH_PTS, same row as V1
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  
                "y": 215   # Same y as V1
            })
        elif lead == "V4":
            # V4 at position 0, fourth row (y=150) - shifted 30 points down
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 150
            })
        elif lead == "V5":
            # V5 at COLUMN_WIDTH_PTS, same row as V4
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  
                "y": 150   # Same y as V4
            })
        elif lead == "V6":
            # V6 at 2 * COLUMN_WIDTH_PTS, same row as V4
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  
                "y": 150   # Same y as V4
            })
    
    print(f" Using lead positions in {lead_sequence} sequence: {[pos['lead'] for pos in lead_positions]}")
    
    # Add additional Lead II strip at V4 position + 60 points (y=90) - EXTENDED to 420 points
    lead_positions.append({
        "lead": "II",
        "x": 0,  # Same x as V4
        "y": 90   # 60 points below V4 (150 - 60 = 90)
    })
    
    # Draw vertical dotted lines to separate columns
    # First column line (COLUMN_WIDTH_PTS) for Lead I, aVR, V1, V4
    from reportlab.graphics.shapes import Line
    from reportlab.lib.colors import black
    first_column_line = Line(COLUMN_WIDTH_PTS, 158, COLUMN_WIDTH_PTS, 377, strokeColor=black, strokeWidth=1, strokeDashArray=[2, 2])
    master_drawing.add(first_column_line)
    
    # Second column line (2 * COLUMN_WIDTH_PTS) for Lead II, aVL, V2, V5
    second_column_line = Line(2 * COLUMN_WIDTH_PTS, 158, 2 * COLUMN_WIDTH_PTS, 377, strokeColor=black, strokeWidth=1, strokeDashArray=[2, 2])
    master_drawing.add(second_column_line)
    
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
                    # NEW LOGIC: Time window = graph_width / effective_wave_speed (17.5 boxes)
                    ecg_graph_width_mm = COLUMN_BOXES * ECG_LARGE_BOX_MM
                    effective_wave_speed_mm_s = wave_speed * ECG_SPEED_SCALE
                    time_window_seconds = ecg_graph_width_mm / effective_wave_speed_mm_s
                    print(f" Report Generator: Demo mode ON - Calculated window using NEW LOGIC: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {time_window_seconds:.2f}s")
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
        #   - Time window = graph_width / effective_wave_speed ONLY (17.5 boxes × ECG_LARGE_BOX_MM)
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
            lead_label = String(x_pos + 10, y_pos + 35, f"{lead}", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)

            # Determine per-lead sample window based on wave speed
            # NEW LOGIC: 
            #   If wave_speed == 50.0: Use HALF samples (875/2550) to fit in 17.5/51 boxes
            #   If wave_speed == 25.0: Use FULL samples (1750/5100) to fit in 17.5/51 boxes
            #   If wave_speed == 12.5: Use FULL samples (1750/5100) which will take HALF width (8.75/25.5 boxes)
            if abs(wave_speed_mm_s - 50.0) < 0.1:
                # 50 mm/s mode -> Half number of samples to stay within 17.5/51 boxes
                if lead == "II" and y_pos == 90:
                    target_samples = FOUR_THREE_SAMPLES_EXTRA_II // 2  # 2550 samples -> 17.5 boxes at 50mm/s
                else:
                    target_samples = FOUR_THREE_SAMPLES_COLUMN // 2    # 875 samples -> 17.5 boxes at 50mm/s
                print(f" Wave speed 50.0 mm/s detected (LOOP 1): Using half-samples ({target_samples}) to fit in 17.5 boxes")
            else:
                # 25.0 or 12.5 mm/s mode -> Full samples (1750/5100)
                # At 25.0: 1750 samples = 3.5s = 17.5 boxes
                # At 12.5: 1750 samples = 3.5s = 8.75 boxes
                if lead == "II" and y_pos == 90:
                    target_samples = FOUR_THREE_SAMPLES_EXTRA_II
                else:
                    target_samples = FOUR_THREE_SAMPLES_COLUMN
                print(f" Wave speed {wave_speed_mm_s} mm/s detected (LOOP 1): Using full-samples ({target_samples})")

            if is_demo_mode and time_window_seconds is not None:
                lead_time_window_seconds = time_window_seconds
                lead_samples_to_capture = int(time_window_seconds * samples_per_second)
            else:
                lead_samples_to_capture = target_samples
                lead_time_window_seconds = lead_samples_to_capture / max(1e-6, computed_sampling_rate)
            
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
            
            # Priority 1: Use ONLY live dashboard data (ignore saved data completely)
            real_data_available = False
            real_ecg_data = None
            
            # Use live dashboard data only
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
                                    if len(raw_data) >= lead_samples_to_capture:
                                        raw_data = raw_data[-lead_samples_to_capture:]
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
                    
                    # Always use live dashboard data (ignore any saved data)
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        # For -aVR, use filtered inverted aVR data
                        raw_data = ecg_test_page.data[3]
                        # Check if we have enough samples, otherwise use all available
                        if len(raw_data) >= lead_samples_to_capture:
                            raw_data = raw_data[-lead_samples_to_capture:]
                        # Check if data is not all zeros or flat
                        if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                            # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                            real_ecg_data = np.array(raw_data)
                            
                            real_data_available = True
                            actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                            if is_demo_mode and time_window_seconds is not None:
                                pass
                            else:
                                time_window_str = f"{lead_time_window_seconds:.2f}s" if lead_time_window_seconds else "auto"
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        # Get filtered real data for this lead
                        lead_index = lead_to_index[lead]
                        if len(ecg_test_page.data[lead_index]) > 0:
                            raw_data = ecg_test_page.data[lead_index]
                            # Check if we have enough samples, otherwise use all available
                            if len(raw_data) >= lead_samples_to_capture:
                                raw_data = raw_data[-lead_samples_to_capture:]
                            # Check if data has variation (not all zeros or flat line)
                            if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                                real_ecg_data = np.array(raw_data)
                                
                                real_data_available = True
                                actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                                if is_demo_mode and time_window_seconds is not None:
                                    pass
                                else:
                                    time_window_str = f"{lead_time_window_seconds:.2f}s" if lead_time_window_seconds else "auto"
            
            print(f" DEBUG: real_data_available={real_data_available}, len(real_ecg_data)={len(real_ecg_data) if real_ecg_data is not None else 0} for Lead {lead}")

            
            if real_data_available and len(real_ecg_data) > 0:
                # Calculate ECG width based on column position to stop at dotted lines
                # Special case: Additional Lead II at V4 position (y=90) gets extended width using exact 52-box calculation
                # Also Lead I at x_pos=0, y_pos=345 gets same extended width for consistency
                if (lead == "II" and y_pos == 90) or (lead == "I" and x_pos == 0 and y_pos == 345):
                    # Calculate exact width for 52 boxes to ensure proper R-R intervals
                    # 52 boxes × ECG_LARGE_BOX_MM × mm = exact width in points
                    exact_52_box_width = 52 * ECG_LARGE_BOX_MM * mm
                    ecg_width = exact_52_box_width
                    if lead == "I":
                        print(f" Lead I: Extended to 52-box width for consistency - 52 × {ECG_LARGE_BOX_MM:.2f}mm × mm = {ecg_width:.1f} points")
                    else:
                        print(f" Extra Lead II: Exact 52-box width calculation - 52 × {ECG_LARGE_BOX_MM:.2f}mm × mm = {ecg_width:.1f} points")
                elif x_pos == 0:
                    # First column: stop at COLUMN_WIDTH_PTS
                    ecg_width = COLUMN_WIDTH_PTS
                elif x_pos == COLUMN_WIDTH_PTS:
                    # Second column: stop at 2 * COLUMN_WIDTH_PTS
                    ecg_width = COLUMN_WIDTH_PTS
                else:
                    # Third column: use COLUMN_WIDTH_PTS
                    ecg_width = COLUMN_WIDTH_PTS
                
                ecg_height = 45
                
                # Create time array for ALL data
                # NEW LOGIC: Use time * speed * scale to align with 5.25mm boxes
                # 25mm/s speed, scaled by ECG_SPEED_SCALE (1.05) to match 5.25mm boxes
                # result: 1 second = 5 boxes = 26.25mm (exactly aligned with grid)
                # Create time array for ALL data (same approach as ecg_report_generator.py)
                ecg_width = COLUMN_WIDTH_PTS
                t = np.linspace(x_pos, x_pos + ecg_width, len(real_ecg_data))
                
                adc_data = np.array(real_ecg_data, dtype=float)

                # Apply SAME filters as dashboard (AC notch → EMG → DFT → Gaussian)
                try:
                    from ecg.ecg_filters import apply_ecg_filters
                    from scipy.ndimage import gaussian_filter1d as _gf1d
                    
                    ac_setting  = str(settings_manager.get_setting("filter_ac",  "50")).strip()
                    emg_setting = str(settings_manager.get_setting("filter_emg", "150")).strip()
                    dft_setting = str(settings_manager.get_setting("filter_dft", "0.5")).strip()
                    
                    # Nyquist guard: AC notch at F Hz requires sampling rate > 2*F Hz
                    if ac_setting in ("50", "60"):
                        required_fs = float(ac_setting) * 2.0 + 1.0
                        if float(computed_sampling_rate) <= required_fs:
                            print(f" AC filter disabled (rate {computed_sampling_rate} Hz too low for {ac_setting} Hz notch)")
                            ac_setting = "off"
                    
                    adc_data = apply_ecg_filters(
                        signal=adc_data,
                        sampling_rate=float(computed_sampling_rate),
                        ac_filter=ac_setting  if ac_setting  not in ("off", "") else None,
                        emg_filter=emg_setting if emg_setting not in ("off", "") else None,
                        dft_filter=dft_setting if dft_setting not in ("off", "") else None,
                    )
                    
                    # Light Gaussian smoothing (sigma=0.8 — same as dashboard)
                    if len(adc_data) > 5:
                        adc_data = _gf1d(adc_data, sigma=0.8)
                    
                    print(f" Applied dashboard filters: AC={ac_setting}, EMG={emg_setting}, DFT={dft_setting} for {lead}")
                except Exception as filter_err:
                    print(f" Dashboard filter apply failed for {lead}: {filter_err}")

                # DEBUG: Check if data is already processed (baseline-subtracted)
                # If data range is far from 2000 baseline, it might already be processed
                data_mean = np.mean(adc_data)
                data_std = np.std(adc_data)
                
                # Step 2: Apply baseline 2000 (subtract baseline from ADC values)
                # IMPORTANT: For calculated leads (III, aVR, aVL, aVF), data is already calculated from processed I and II
                # So it's already centered (mean ~0), but we still need to scale it properly
                baseline_adc = 2000.0
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                if abs(data_mean - 2000.0) < 500:  # Data is close to baseline 2000 (raw ADC)
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
            
                centered_adc = centered_adc - (float(np.mean(centered_adc)) if centered_adc.size > 0 else 0.0)
                if centered_adc.size > 20:
                    x_idx = np.arange(centered_adc.size)
                    trend = np.polyval(np.polyfit(x_idx, centered_adc, 1), x_idx)
                    centered_adc = centered_adc - trend
                boxes_offset = centered_adc / adc_per_box
                
                # Log boxes offset for verification
                
                # Step 5: Convert boxes to Y position
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                # IMPORTANT: Grid is 40 boxes over 210mm -> 5.25mm per box
                # 5.25mm = 5.25 * 2.834645669 points = 14.88 points per box
                box_height_points = ECG_LARGE_BOX_MM * mm  # Match 40-box height grid (5.25mm)
                major_spacing_y = box_height_points  # Use 5.25mm spacing
                
                # Convert boxes offset to Y position
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                try:
                    from ecg.ecg_filters import apply_baseline_wander_median_mean
                    local = ecg_normalized - center_y
                    local = apply_baseline_wander_median_mean(local, 500.0)
                    ecg_normalized = center_y + local
                except Exception:
                    pass
                
                try:
                    idx = np.arange(len(ecg_normalized))
                    local = ecg_normalized - center_y
                    trend = np.polyval(np.polyfit(idx, local, 2), idx)
                    ecg_normalized = center_y + (local - trend)
                except Exception:
                    pass
                try:
                    local = ecg_normalized - center_y
                    wl = max(25, int(0.12 * len(local)))
                    kernel = np.ones(wl) / float(wl)
                    baseline = np.convolve(local, kernel, mode="same")
                    ecg_normalized = center_y + (local - baseline)
                except Exception:
                    pass
                edge = max(30, int(0.25 * len(ecg_normalized)))
                if len(ecg_normalized) > edge * 2:
                    r = np.sin(np.linspace(0.0, np.pi / 2.0, edge)) ** 2
                    ecg_normalized[:edge] = center_y + (ecg_normalized[:edge] - center_y) * r
                    ecg_normalized[-edge:] = center_y + (ecg_normalized[-edge:] - center_y) * r[::-1]
                clamp = max(10, int(0.05 * len(ecg_normalized)))
                if len(ecg_normalized) > clamp * 2:
                    ecg_normalized[:clamp] = center_y
                    ecg_normalized[-clamp:] = center_y
                try:
                    idx = np.arange(len(ecg_normalized))
                    local = ecg_normalized - center_y
                    trend = np.polyval(np.polyfit(idx, local, 2), idx)
                    ecg_normalized = center_y + (local - trend)
                except Exception:
                    pass
                edge = max(20, int(0.12 * len(ecg_normalized)))
                if len(ecg_normalized) > edge * 2:
                    r = np.sin(np.linspace(0.0, np.pi / 2.0, edge)) ** 2
                    ecg_normalized[:edge] = center_y + (ecg_normalized[:edge] - center_y) * r
                    ecg_normalized[-edge:] = center_y + (ecg_normalized[-edge:] - center_y) * r[::-1]
                clamp = max(5, int(0.01 * len(ecg_normalized)))
                if len(ecg_normalized) > clamp * 2:
                    ecg_normalized[:clamp] = center_y
                    ecg_normalized[-clamp:] = center_y
                
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
                
                ecg_path.moveTo(t[0], ecg_normalized[0])
                for i in range(1, len(t)):
                    ecg_path.lineTo(t[i], ecg_normalized[i])
                
                # Add path to master drawing
                master_drawing.add(ecg_path)
                
                print(f" DEBUG: About to check calibration notch for Lead {lead}")
                # Calibration notch (only for first column leads with x=0: I, aVR, V1, V4, and additional II)
                notch_path = None
                print(f" DEBUG: Checking notch for Lead {lead} - x_pos={x_pos}, is in first column: {x_pos == 0}")
                if x_pos == 0:  # Only first column leads (x=0)
                    print(f" DEBUG: Creating calibration notch for Lead {lead}")
                    notch_width_mm = 5.0   # width 5mm
                    notch_height_mm = 10.0 # height 10mm
                    notch_width = notch_width_mm * mm
                    notch_height = notch_height_mm * mm
                    
                    # Place notch 10 points after where ECG strip starts, then shift 30 points left
                    notch_x = x_pos + 10.0 - 30.0  # 10 points after strip start, then 30 points left (using x_pos instead of adjusted_x_pos)
                    notch_y_base = center_y
                    print(f" DEBUG: Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                    
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
                    notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                    
                    # Add notch to master drawing
                    master_drawing.add(notch_path)
                    print(f" DEBUG: Notch added to master drawing for Lead {lead}")
                else:
                    print(f" DEBUG: Skipping notch for Lead {lead} (not in first column leads)")
                
                print(f" Drew {len(real_ecg_data)} ECG data points for Lead {lead}")
                if x_pos == 0 and notch_path:
                    print(f" Added calibration notch for Lead {lead}")
            else:
                print(f" *** NO DATA CONDITION REACHED for Lead {lead} - showing grid only ***")
                
                # Draw flat line when no real data available (like dashboard)
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Line, Path
                
                # Calculate center_y same as real data section
                ecg_height = 45  # Same as real data section
                center_y = y_pos + (ecg_height / 2.0)  # Center of graph in points
                
                # Draw flat line at center (baseline)
                flat_line_start_x = x_pos + 15.0  # Same start as real data
                flat_line_end_x = x_pos + 100  # Default end position
                flat_line_y = center_y  # Center/baseline position
                
                flat_line = Line(flat_line_start_x, flat_line_y, flat_line_end_x, flat_line_y,
                              strokeColor=colors.HexColor("#000000"), strokeWidth=1.2)
                master_drawing.add(flat_line)
                
                # Create calibration notch for first column leads even when no data is available
                print(f" DEBUG: About to check calibration notch for Lead {lead} (no data case)")
                notch_path = None
                print(f" DEBUG: Checking notch for Lead {lead} - x_pos={x_pos}, is in first column: {x_pos == 0}")
                if x_pos == 0:  # Only first column leads (x=0)
                    print(f" DEBUG: Creating calibration notch for Lead {lead} (no data case)")
                    notch_width_mm = 5.0   # width 5mm
                    notch_height_mm = 10.0 # height 10mm
                    notch_width = notch_width_mm * mm
                    notch_height = notch_height_mm * mm
                    
                    # Calculate center_y same as real data section
                    ecg_height = 45  # Same as real data section
                    center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                    
                    # Place notch 10 points after where ECG strip starts, then shift 30 points left
                    notch_x = x_pos + 10.0 - 30.0  # 10 points after strip start, then 30 points left
                    notch_y_base = center_y  # Use same center_y calculation as real data section
                    print(f" DEBUG: Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                    
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
                    notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                    
                    # Add notch to master drawing
                    master_drawing.add(notch_path)
                    print(f" DEBUG: Notch added to master drawing for Lead {lead} (no data case)")
                    print(f" Added calibration notch for Lead {lead} (no data case)")
                else:
                    print(f" DEBUG: Skipping notch for Lead {lead} (not in first column leads, no data case)")
            
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
    
    # TODO: Add new Date/Time labels below DECKMOUNT logo

    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT',)
    QTc = data.get('QTc',)
    QTcF = data.get('QTc_Fridericia') or data.get('QTcF') or 0
    ST = data.get('ST', 114)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   

    # Create table data: 4 rows × 2 columns
    vital_table_data = [
        [f"HR : {int(round(HR))} bpm", f"QT: {int(round(QT))} ms"],
        [f"PR : {int(round(PR))} ms", f"QTc: {int(round(QTc))} ms"],
        [f"QRS: {int(round(QRS))} ms", f"ST: {int(round(ST))} ms"],
        [f"RR : {int(round(RR))} ms", f"QTcF: {QTcF/1000.0:.3f} s"]  
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
    
    # Column width based on 17.5 boxes (data) + 0.5 boxes (gap) = 18 boxes total budget
    COLUMN_WIDTH_PTS = 18 * ECG_LARGE_BOX_MM * mm
    
    # Single drawing dimensions - ADJUSTED HEIGHT to fit within page frame (max ~770)
    total_width = 3 * COLUMN_WIDTH_PTS + 20   # Full width for 3 columns + small margin
    total_height = 520  # Adjusted to fit within landscape frame (555.3 available)
    
    # Create ONE master drawing
    master_drawing = Drawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads based on selected sequence (START 90 points below QTc at 345)
    # Special arrangement for all rows: 3x4 grid with 259 points horizontal gaps, shifted 30 points down
    y_positions = [345, 295, 245, 195, 145, 95, 45, -5, -55, -105, -155, -205]  # Start 90 points below QTc (435-90=345)  
    6
    lead_positions = []
    
    for i, lead in enumerate(lead_order):
        if lead == "I":
            # Lead I at position 0, y=345
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 345
            })
        elif lead == "II":
            # Lead II at COLUMN_WIDTH_PTS, same row as Lead I
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  # 18 boxes gap from Lead I
                "y": 345   # Same y as Lead I
            })
        elif lead == "III":
            # Lead III at 2 * COLUMN_WIDTH_PTS, same row as Lead I
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  # 36 boxes gap from Lead I
                "y": 345   # Same y as Lead I
            })
        elif lead == "aVR":
            # aVR at position 0, next row (y=280) - shifted 15 points down
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 280
            })
        elif lead == "aVL":
            # aVL at COLUMN_WIDTH_PTS, same row as aVR
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  
                "y": 280   # Same y as aVR
            })
        elif lead == "aVF":
            # aVF at 2 * COLUMN_WIDTH_PTS, same row as aVR
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  
                "y": 280   # Same y as aVR
            })
        elif lead == "V1":
            # V1 at position 0, third row (y=215) - shifted 15 points down
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 215
            })
        elif lead == "V2":
            # V2 at COLUMN_WIDTH_PTS, same row as V1
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  
                "y": 215   # Same y as V1
            })
        elif lead == "V3":
            # V3 at 2 * COLUMN_WIDTH_PTS, same row as V1
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  
                "y": 215   # Same y as V1
            })
        elif lead == "V4":
            # V4 at position 0, fourth row (y=150) - shifted 30 points down
            lead_positions.append({
                "lead": lead, 
                "x": 0,  
                "y": 150
            })
        elif lead == "V5":
            # V5 at COLUMN_WIDTH_PTS, same row as V4
            lead_positions.append({
                "lead": lead, 
                "x": COLUMN_WIDTH_PTS,  
                "y": 150   # Same y as V4
            })
        elif lead == "V6":
            # V6 at 2 * COLUMN_WIDTH_PTS, same row as V4
            lead_positions.append({
                "lead": lead, 
                "x": 2 * COLUMN_WIDTH_PTS,  
                "y": 150   # Same y as V4
            })
    
    print(f" Using lead positions in {lead_sequence} sequence: {[pos['lead'] for pos in lead_positions]}")
    
    # Add additional Lead II strip at V4 position + 60 points (y=90) - EXTENDED to 420 points
    lead_positions.append({
        "lead": "II",
        "x": 0,  # Same x as V4
        "y": 90   # 60 points below V4 (150 - 60 = 90)
    })
    
    # Draw vertical dotted lines to separate columns
    # First column line (COLUMN_WIDTH_PTS) for Lead I, aVR, V1, V4
    from reportlab.graphics.shapes import Line
    from reportlab.lib.colors import black
    first_column_line = Line(COLUMN_WIDTH_PTS, 158, COLUMN_WIDTH_PTS, 377, strokeColor=black, strokeWidth=1, strokeDashArray=[2, 2])
    master_drawing.add(first_column_line)
    
    # Second column line (2 * COLUMN_WIDTH_PTS) for Lead II, aVL, V2, V5
    second_column_line = Line(2 * COLUMN_WIDTH_PTS, 158, 2 * COLUMN_WIDTH_PTS, 377, strokeColor=black, strokeWidth=1, strokeDashArray=[2, 2])
    master_drawing.add(second_column_line)
    
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
                    # NEW LOGIC: Time window = graph_width / effective_wave_speed (17.5 boxes)
                    ecg_graph_width_mm = COLUMN_BOXES * ECG_LARGE_BOX_MM
                    effective_wave_speed_mm_s = wave_speed * ECG_SPEED_SCALE
                    time_window_seconds = ecg_graph_width_mm / effective_wave_speed_mm_s
                    print(f" Report Generator: Demo mode ON - Calculated window using NEW LOGIC: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {time_window_seconds:.2f}s")
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
        #   - Time window = graph_width / effective_wave_speed ONLY (17.5 boxes × ECG_LARGE_BOX_MM)
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
            lead_label = String(x_pos + 10, y_pos + 35, f"{lead}", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)

            # Determine per-lead sample window based on wave speed
            # NEW LOGIC: 
            #   If wave_speed == 50.0: Use HALF samples (875/2550) to fit in 17.5/51 boxes
            #   If wave_speed == 25.0: Use FULL samples (1750/5100) to fit in 17.5/51 boxes
            #   If wave_speed == 12.5: Use FULL samples (1750/5100) which will take HALF width (8.75/25.5 boxes)
            if abs(wave_speed_mm_s - 50.0) < 0.1:
                # 50 mm/s mode -> Half number of samples to stay within 17.5/51 boxes
                if lead == "II" and y_pos == 90:
                    target_samples = FOUR_THREE_SAMPLES_EXTRA_II // 2  # 2550 samples -> 17.5 boxes at 50mm/s
                else:
                    target_samples = FOUR_THREE_SAMPLES_COLUMN // 2    # 875 samples -> 17.5 boxes at 50mm/s
                print(f" Wave speed 50.0 mm/s detected (LOOP 2): Using half-samples ({target_samples}) to fit in 17.5 boxes")
            else:
                # 25.0 or 12.5 mm/s mode -> Full samples (1750/5100)
                # At 25.0: 1750 samples = 3.5s = 17.5 boxes
                # At 12.5: 1750 samples = 3.5s = 8.75 boxes
                if lead == "II" and y_pos == 90:
                    target_samples = FOUR_THREE_SAMPLES_EXTRA_II
                else:
                    target_samples = FOUR_THREE_SAMPLES_COLUMN
                print(f" Wave speed {wave_speed_mm_s} mm/s detected (LOOP 2): Using full-samples ({target_samples})")

            if is_demo_mode and time_window_seconds is not None:
                lead_time_window_seconds = time_window_seconds
                lead_samples_to_capture = int(time_window_seconds * samples_per_second)
            else:
                lead_samples_to_capture = target_samples
                lead_time_window_seconds = lead_samples_to_capture / max(1e-6, computed_sampling_rate)
            
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
            
            # Priority 1: Use ONLY live dashboard data (ignore saved data completely)
            real_data_available = False
            real_ecg_data = None
            
            # Use live dashboard data only
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
                                    if len(raw_data) >= lead_samples_to_capture:
                                        raw_data = raw_data[-lead_samples_to_capture:]
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
                    
                    # Always use live dashboard data (ignore any saved data)
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        # For -aVR, use filtered inverted aVR data
                        raw_data = ecg_test_page.data[3]
                        # Check if we have enough samples, otherwise use all available
                        if len(raw_data) >= lead_samples_to_capture:
                            raw_data = raw_data[-lead_samples_to_capture:]
                        # Check if data is not all zeros or flat
                        if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                            # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                            real_ecg_data = np.array(raw_data)
                            
                            real_data_available = True
                            actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                            if is_demo_mode and time_window_seconds is not None:
                                pass
                            else:
                                time_window_str = f"{lead_time_window_seconds:.2f}s" if lead_time_window_seconds else "auto"
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        # Get filtered real data for this lead
                        lead_index = lead_to_index[lead]
                        if len(ecg_test_page.data[lead_index]) > 0:
                            raw_data = ecg_test_page.data[lead_index]
                            # Check if we have enough samples, otherwise use all available
                            if len(raw_data) >= lead_samples_to_capture:
                                raw_data = raw_data[-lead_samples_to_capture:]
                            # Check if data has variation (not all zeros or flat line)
                            if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                                real_ecg_data = np.array(raw_data)
                                
                                real_data_available = True
                                actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                                if is_demo_mode and time_window_seconds is not None:
                                    pass
                                else:
                                    time_window_str = f"{lead_time_window_seconds:.2f}s" if lead_time_window_seconds else "auto"
            
            print(f" DEBUG: real_data_available={real_data_available}, len(real_ecg_data)={len(real_ecg_data) if real_ecg_data is not None else 0} for Lead {lead}")

            
            if real_data_available and len(real_ecg_data) > 0:
                # Calculate ECG width based on column position to stop at dotted lines
                # Special case: Additional Lead II at V4 position (y=90) gets extended width using exact 52-box calculation
                # Also Lead I at x_pos=0, y_pos=345 gets same extended width for consistency
                if (lead == "II" and y_pos == 90) or (lead == "I" and x_pos == 0 and y_pos == 345):
                    # Calculate exact width for 52 boxes to ensure proper R-R intervals
                    # 52 boxes × ECG_LARGE_BOX_MM × mm = exact width in points
                    exact_52_box_width = 52 * ECG_LARGE_BOX_MM * mm
                    ecg_width = exact_52_box_width
                    if lead == "I":
                        print(f" Lead I: Extended to 52-box width for consistency - 52 × {ECG_LARGE_BOX_MM:.2f}mm × mm = {ecg_width:.1f} points")
                    else:
                        print(f" Extra Lead II: Exact 52-box width calculation - 52 × {ECG_LARGE_BOX_MM:.2f}mm × mm = {ecg_width:.1f} points")
                elif x_pos == 0:
                    # First column: stop at COLUMN_WIDTH_PTS
                    ecg_width = COLUMN_WIDTH_PTS
                elif x_pos == COLUMN_WIDTH_PTS:
                    # Second column: stop at 2 * COLUMN_WIDTH_PTS
                    ecg_width = COLUMN_WIDTH_PTS
                else:
                    # Third column: use COLUMN_WIDTH_PTS
                    ecg_width = COLUMN_WIDTH_PTS
                
                ecg_height = 45
                
                # Create time array for ALL data
                # NEW LOGIC: Use time * speed * scale to align with 5.25mm boxes
                # 25mm/s speed, scaled by ECG_SPEED_SCALE (1.05) to match 5.25mm boxes
                # result: 1 second = 5 boxes = 26.25mm (exactly aligned with grid)
                fs = float(computed_sampling_rate)
                t_sec = np.arange(len(real_ecg_data)) / fs
                
                # Calculate X points in points (1/72 inch)
                # x_pos is the starting point in the master drawing
                # ADD 1mm (1 minor box) gap at start
                gap_points = ECG_SMALL_BOX_MM * mm
                # Use actual wave_speed_mm_s for physical horizontal scaling
                t = x_pos + gap_points + (t_sec * wave_speed_mm_s * ECG_SPEED_SCALE * mm)
                
                print(f" Lead {lead}: Using diagnostic time scaling - {wave_speed_mm_s} mm/s * {ECG_SPEED_SCALE:.2f} scale")
                print(f"   X range: {t[0]:.1f} to {t[-1]:.1f} (points)")
                
                
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
                
                # DEBUG: Check if data is already processed (baseline-subtracted)
                data_mean = np.mean(adc_data)
                data_std = np.std(adc_data)
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                # Step 2: Apply baseline 2000 (subtract baseline from ADC values)
                # IMPORTANT: For calculated leads, data is already calculated from processed I and II
                # So it's already centered (mean ~0), but we still need to scale it properly
                baseline_adc = 2000.0
                
                if abs(data_mean - 2000.0) < 500:  # Data is close to baseline 2000 (raw ADC)
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
                # For 10mm/mV with multiplier 6400: 6400 / 10 = 640 ADC per box
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
                
                # DEBUG: Log actual ADC values for troubleshooting
                max_centered_adc_abs = np.max(np.abs(centered_adc))
                expected_boxes = max_centered_adc_abs / adc_per_box
                
                
                
                centered_adc = centered_adc - (float(np.mean(centered_adc)) if centered_adc.size > 0 else 0.0)
                if centered_adc.size > 20:
                    x_idx = np.arange(centered_adc.size)
                    trend = np.polyval(np.polyfit(x_idx, centered_adc, 1), x_idx)
                    centered_adc = centered_adc - trend
                boxes_offset = centered_adc / adc_per_box
                
                # Step 5: Convert boxes to Y position (in mm, then to points)
                # Center of graph is at y_pos + (ecg_height / 2.0)
                # IMPORTANT: Use 5.25mm per box to match 40-box height grid
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                box_height_points = ECG_LARGE_BOX_MM * mm  # 5.25mm per box
                
                # Convert boxes offset to Y position
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                try:
                    from ecg.ecg_filters import apply_baseline_wander_median_mean
                    local = ecg_normalized - center_y
                    local = apply_baseline_wander_median_mean(local, 500.0)
                    ecg_normalized = center_y + local
                except Exception:
                    pass
                
                try:
                    from ecg.ecg_filters import apply_baseline_wander_median_mean
                    local = ecg_normalized - center_y
                    local = apply_baseline_wander_median_mean(local, 500.0)
                    ecg_normalized = center_y + local
                except Exception:
                    pass
                
                
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
                
                ecg_path.moveTo(t[0], ecg_normalized[0])
                for i in range(1, len(t)):
                    ecg_path.lineTo(t[i], ecg_normalized[i])
                
                # Add path to master drawing
                master_drawing.add(ecg_path)
                
                print(f" DEBUG: About to check calibration notch for Lead {lead}")
                # Calibration notch (only for first column leads with x=0: I, aVR, V1, V4, and additional II)
                notch_path = None
                print(f" DEBUG: Checking notch for Lead {lead} - x_pos={x_pos}, is in first column: {x_pos == 0}")
                if x_pos == 0:  # Only first column leads (x=0)
                    print(f" DEBUG: Creating calibration notch for Lead {lead}")
                    notch_width_mm = 5.0   # width 5mm
                    notch_height_mm = 10.0 # height 10mm
                    notch_width = notch_width_mm * mm
                    notch_height = notch_height_mm * mm
                    
                    # Place notch 10 points after where ECG strip starts, then shift 30 points left
                    notch_x = x_pos + 10.0 - 30.0  # 10 points after strip start, then 30 points left (using x_pos instead of adjusted_x_pos)
                    notch_y_base = center_y
                    print(f" DEBUG: Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                    
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
                    notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                    
                    # Add notch to master drawing
                    master_drawing.add(notch_path)
                    print(f" DEBUG: Notch added to master drawing for Lead {lead}")
                else:
                    print(f" DEBUG: Skipping notch for Lead {lead} (not in first column leads)")
                
                print(f" Drew {len(real_ecg_data)} ECG data points for Lead {lead}")
                if x_pos == 0 and notch_path:
                    print(f" Added calibration notch for Lead {lead}")
            else:
                print(f" *** NO DATA CONDITION REACHED for Lead {lead} - showing grid only ***")
                
                # Draw flat line when no real data available (like dashboard)
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Line, Path
                
                # Calculate center_y same as real data section
                ecg_height = 45  # Same as real data section
                center_y = y_pos + (ecg_height / 2.0)  # Center of graph in points
                
                # Draw flat line at center (baseline)
                flat_line_start_x = x_pos + 15.0  # Same start as real data
                flat_line_end_x = x_pos + 100  # Default end position
                flat_line_y = center_y  # Center/baseline position
                
                flat_line = Line(flat_line_start_x, flat_line_y, flat_line_end_x, flat_line_y,
                              strokeColor=colors.HexColor("#000000"), strokeWidth=1.2)
                master_drawing.add(flat_line)
                
                # Create calibration notch for first column leads even when no data is available
                print(f" DEBUG: About to check calibration notch for Lead {lead} (no data case)")
                notch_path = None
                print(f" DEBUG: Checking notch for Lead {lead} - x_pos={x_pos}, is in first column: {x_pos == 0}")
                if x_pos == 0:  # Only first column leads (x=0)
                    print(f" DEBUG: Creating calibration notch for Lead {lead} (no data case)")
                    notch_width_mm = 5.0   # width 5mm
                    notch_height_mm = 10.0 # height 10mm
                    notch_width = notch_width_mm * mm
                    notch_height = notch_height_mm * mm
                    
                    # Calculate center_y same as real data section
                    ecg_height = 45  # Same as real data section
                    center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                    
                    # Place notch 10 points after where ECG strip starts, then shift 30 points left
                    notch_x = x_pos + 10.0 - 30.0  # 10 points after strip start, then 30 points left
                    notch_y_base = center_y  # Use same center_y calculation as real data section
                    print(f" DEBUG: Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                    
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
                    notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                    
                    # Add notch to master drawing
                    master_drawing.add(notch_path)
                    print(f" DEBUG: Notch added to master drawing for Lead {lead} (no data case)")
                    print(f" Added calibration notch for Lead {lead} (no data case)")
                else:
                    print(f" DEBUG: Skipping notch for Lead {lead} (not in first column leads, no data case)")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 4: Add Patient Info, Date/Time and Vital Parameters to master drawing
    # POSITIONED ABOVE ECG GRAPH (not mixed inside graph)
    from reportlab.graphics.shapes import String

    # LEFT SIDE: Patient Info (ABOVE ECG GRAPH - shifted further up)
    patient_name_label = String(-5, 535, f"Name: {full_name}",  # Shifted down by 15 points
                           fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_name_label)

    patient_age_label = String(-5, 515, f"Age: {age}",  # Shifted down by 15 points
                          fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_age_label)

    patient_gender_label = String(-5, 495, f"Gender: {gender}",  # Shifted down by 15 points
                             fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_gender_label)
    
    # TODO: Add new Date/Time labels below DECKMOUNT logo

    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT',)
    QTc = data.get('QTc',)
    ST = data.get('ST',)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   
    # Add vital parameters in TWO COLUMNS (ABOVE ECG GRAPH - shifted further up)
    # FIRST COLUMN (Left side - x=130)
    hr_label = String(130, 535, f"HR    : {HR} bpm",  # Shifted down by 20 points total  
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(hr_label)

    pr_label = String(130, 515, f"PR    : {PR} ms",  # Shifted down by 20 points total  
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(pr_label)

    qrs_label = String(130, 495, f"QRS : {QRS} ms",  # Shifted down by 20 points total 
                      fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qrs_label)
    
    rr_label = String(130, 475, f"RR    : {RR} ms",  # Shifted down by 20 points total 
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rr_label)

    qt_label = String(130, 455, f"QT    : {int(round(QT))} ms",  # Shifted down by 20 points total  
                     fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qt_label)

    qtc_label = String(130, 435, f"QTc  : {int(round(QTc))} ms",  # Shifted down by 20 points total  
                      fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtc_label)

    # SECOND COLUMN (Right side) - QTcF replaces ST
    _qtcf_val = data.get('QTc_Fridericia') or data.get('QTcF_ms') or data.get('QTcF') or data.get('QTcF_interval')
    _qtcf_display = f"{int(round(float(_qtcf_val)))} ms" if _qtcf_val and float(_qtcf_val) > 0 else "-- ms"
    qtcf_header_label = String(240, 455, f"QTcF : {_qtcf_display}",
                               fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_header_label)

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
            fs = float(computed_sampling_rate)
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
            fs = float(computed_sampling_rate)
            
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
                                print(f" P axis validation: HR={estimated_hr:.1f} BPM, P_axis={p_axis_num}°, normalized={p_axis_num_normalized}°")
                                
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
                                            print(f" P axis adjusted using fallback method: {p_axis_deg} (original: {p_axis_result}, HR: {estimated_hr:.0f} BPM)")
                                        else:
                                            print(f" P axis value: {p_axis_deg} (may be less accurate at HR {estimated_hr:.0f} BPM)")
                                    else:
                                        # Last resort: use original value even if abnormal
                                        p_axis_deg = p_axis_result
                                        print(f" P axis value: {p_axis_deg} (calculated at HR {estimated_hr:.0f} BPM, may be less accurate)")
                        else:
                            # If less than 2 P peaks detected, try to calculate with available peaks
                            if len(p_peaks) >= 1:
                                # Try with single peak (less accurate but better than "--")
                                p_axis_result_single = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks, fs, 20, 60)
                                if p_axis_result_single != "--":
                                    p_axis_deg = p_axis_result_single
                                    print(f" P axis calculated with limited peaks: {p_axis_deg} (HR: {estimated_hr:.0f} BPM, may be less accurate)")
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
                                            print(f" P axis estimated from R-peaks timing: {p_axis_deg} (HR: {estimated_hr:.0f} BPM, estimated)")
                        
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
                        
                        print(f" Calculated P/QRS/T Axis: P={p_axis_deg}, QRS={qrs_axis_deg}, T={t_axis_deg}")
        except Exception as e:
            print(f" Axis calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
    # Format axis values for display (remove ° symbol for compact display)
    # Convert to string first in case they're integers
    p_axis_display = str(p_axis_deg).replace("°", "") if p_axis_deg != "--" else "--"
    qrs_axis_display = str(qrs_axis_deg).replace("°", "") if qrs_axis_deg != "--" else "--"
    t_axis_display = str(t_axis_deg).replace("°", "") if t_axis_deg != "--" else "--"
    
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
    
    # SECOND COLUMN - P/QRS/T Axis (ABOVE ECG GRAPH - same position)
    p_qrs_label = String(240, 535, f"P/QRS/T  : {p_axis_display}/{qrs_axis_display}/{t_axis_display}°",  # Shifted down by 20 points total
                         fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(p_qrs_label)

    # Get RV5 and SV1 amplitudes
    rv5_amp = data.get('rv5', 0.0)
    sv1_amp = data.get('sv1', 0.0)
    
    print(f" Report Generator - Received RV5/SV1 from data:")
    print(f"   rv5: {rv5_amp}, sv1: {sv1_amp}")
    
    # If missing/zero, compute from V5 and V1 of last 10 seconds (GE/Hospital Standard)
    # CRITICAL: Use RAW ECG data, not display-filtered signals
    # Measurements must be from median beat, relative to TP baseline (isoelectric segment before P-wave)
    # NOTE: sv1_amp can be negative (SV1 is negative by definition), so check for == 0.0, not <= 0
    if (rv5_amp<=0 or sv1_amp==0.0) and ecg_test_page is not None and hasattr(ecg_test_page,'data'):
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            fs = float(computed_sampling_rate)
            def _get_last(arr):
                return arr[-int(10*fs):] if arr is not None and len(arr)>int(10*fs) else arr
            # V5 index 10, V1 index 6
            v5 = _get_last(ecg_test_page.data[10]) if len(ecg_test_page.data)>10 else None
            v1 = _get_last(ecg_test_page.data[6]) if len(ecg_test_page.data)>6 else None
            # V5 index 10, V1 index 6 - Get RAW data
            v5_raw = _get_last(ecg_test_page.data[10]) if len(ecg_test_page.data)>10 else None
            v1_raw = _get_last(ecg_test_page.data[6]) if len(ecg_test_page.data)>6 else None
            
            if v5_raw is not None and len(v5_raw)>int(2*fs):
                # Apply filter ONLY for R-peak detection (0.5-40 Hz)
                # Use RAW data for amplitude measurements
                nyq = fs/2.0
                b,a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq,0.99)], btype='band')
                v5f = filtfilt(b,a, np.asarray(v5_raw))
                env = np.convolve(np.square(np.diff(v5f)), np.ones(int(0.15*fs))/(0.15*fs), mode='same')
                r,_ = find_peaks(env, height=np.mean(env)+0.5*np.std(env), distance=int(0.6*fs))
                vals=[]
                for rr in r[1:-1]:
                    # QRS window: ±80ms around R-peak
                    qs = max(0, rr-int(0.08*fs))
                    qe = min(len(v5_raw), rr+int(0.08*fs))
                    if qe > qs:
                        # Use RAW data for amplitude measurement
                        qrs_segment = np.asarray(v5_raw[qs:qe])
                        
                        # TP baseline: isoelectric segment before P-wave onset (150-350ms before R)
                        tp_start = max(0, rr-int(0.35*fs))
                        tp_end = max(0, rr-int(0.15*fs))
                        if tp_end > tp_start:
                            tp_segment = np.asarray(v5_raw[tp_start:tp_end])
                            tp_baseline = np.median(tp_segment)  # Median for robustness
                        else:
                            # Fallback: short segment before QRS
                            tp_baseline = np.median(np.asarray(v5_raw[max(0,qs-int(0.05*fs)):qs]))
                        
                        # RV5 = max(QRS) - TP_baseline (positive, in mV)
                        # Convert from ADC counts to mV: V5 uses 6400 multiplier, 10mm/mV → 1 mV = 1280 ADC
                        r_amp_adc = np.max(qrs_segment) - tp_baseline
                        if r_amp_adc > 0:
                            r_amp_mv = r_amp_adc / 1280.0  # Convert ADC to mV
                            vals.append(r_amp_mv)
                if len(vals)>0 and rv5_amp<=0: 
                    rv5_amp = float(np.median(vals))  # Median beat approach
                    
            if v1_raw is not None and len(v1_raw)>int(2*fs):
                # Apply filter ONLY for R-peak detection (0.5-40 Hz)
                # Use RAW data for amplitude measurements
                nyq = fs/2.0
                b,a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq,0.99)], btype='band')
                v1f = filtfilt(b,a, np.asarray(v1_raw))
                env = np.convolve(np.square(np.diff(v1f)), np.ones(int(0.15*fs))/(0.15*fs), mode='same')
                r,_ = find_peaks(env, height=np.mean(env)+0.5*np.std(env), distance=int(0.6*fs))
                vals=[]
                for rr in r[1:-1]:
                    # QRS window: ±80ms around R-peak
                    ss = rr
                    se = min(len(v1_raw), rr+int(0.08*fs))
                    if se > ss:
                        # Use RAW data for amplitude measurement
                        qrs_segment = np.asarray(v1_raw[ss:se])
                        
                        # TP baseline: isoelectric segment before P-wave onset (150-350ms before R)
                        tp_start = max(0, rr-int(0.35*fs))
                        tp_end = max(0, rr-int(0.15*fs))
                        if tp_end > tp_start:
                            tp_segment = np.asarray(v1_raw[tp_start:tp_end])
                            tp_baseline = np.median(tp_segment)  # Median for robustness
                        else:
                            # Fallback: short segment before QRS
                            tp_baseline = np.median(np.asarray(v1_raw[max(0, ss-int(0.05*fs)):ss]))
                        
                        # SV1 = min(QRS) - TP_baseline (negative, preserve sign, in mV)
                        # Convert from ADC counts to mV: V1 uses 6400 multiplier, 10mm/mV → 1 mV = 1280 ADC
                        s_amp_adc = np.min(qrs_segment) - tp_baseline
                        if s_amp_adc < 0:  # SV1 must be negative
                            s_amp_mv = s_amp_adc / 1280.0  # Convert ADC to mV (preserve sign)
                            vals.append(s_amp_mv)
                if len(vals)>0 and sv1_amp==0.0:
                    sv1_amp = float(np.median(vals))  # Median beat approach, negative value
            print(f" Fallback computed RV5/SV1: RV5={rv5_amp:.4f}, SV1={sv1_amp:.4f}")
        except Exception as e:
            print(f" Fallback RV5/SV1 computation failed: {e}")

    # Unit conversion: GE/Hospital Standard - Values must be in mV
    # CRITICAL: calculate_wave_amplitudes() now returns values in mV (converted from ADC counts)
    # No additional conversion needed - use values directly
    # GE Standard ranges: RV5 typically 0.5-3.0 mV, SV1 typically -0.5 to -2.0 mV
    rv5_mv = rv5_amp if rv5_amp > 0 else 0.0
    sv1_mv = sv1_amp if sv1_amp != 0.0 else 0.0  # SV1 is negative (preserved from calculation)
    
    print(f"   Converted to mV: RV5={rv5_mv:.3f}, SV1={sv1_mv:.3f}")
    
    # SECOND COLUMN - RV5/SV1 (ABOVE ECG GRAPH - shifted further up)
    # Display SV1 as negative mV (GE/Hospital standard)
    # Use 3 decimal places for precision (not rounded to integers)
    rv5_sv_label = String(240, 515, f"RV5/SV1  : {rv5_mv:.3f} mV/{sv1_mv:.3f} mV",  # Shifted down by 20 points total
                          fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv_label)

    # Calculate RV5+SV1 = RV5 + abs(SV1) (GE/Philips standard)
    # CRITICAL: Calculate from unrounded values to avoid rounding errors
    # SV1 is negative, so RV5+SV1 = RV5 + abs(SV1) for Sokolow-Lyon index
    rv5_sv1_sum = rv5_mv + abs(sv1_mv)  # RV5 + abs(SV1) as per GE/Philips standard
    
    # SECOND COLUMN - RV5+SV1 (ABOVE ECG GRAPH - shifted further up)
    # Use 3 decimal places for precision
    rv5_sv1_sum_label = String(240, 495, f"RV5+SV1 : {rv5_sv1_sum:.3f} mV",  # Shifted down by 20 points total
                               fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv1_sum_label)

    # SECOND COLUMN - QTCF (ABOVE ECG GRAPH - shifted further up)
    # Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3)
    qtcf_val = _safe_float(data.get("QTc_Fridericia") or data.get("QTcF_ms") or data.get("QTcF"), None)
    if qtcf_val and qtcf_val > 0:
        qtcf_sec = qtcf_val / 1000.0
        qtcf_text = f"QTCF       : {qtcf_val:.0f} ms ({qtcf_sec:.3f} s)"
    else:
        qtcf_text = "QTCF       : --"
    qtcf_label = String(240, 475, qtcf_text,  # Moved up from 642 to 652
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
        435,  # Moved up from 606 to 616
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
    doctor_name_label = String(-15, 15, "Doctor Name: ",  # Shifted down by 20 points (35 - 20 = 15) 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(doctor_name_label)
    
    if doctor:
        value_x = -15 + stringWidth("Doctor Name: ", "Helvetica-Bold", 10) + 5
        doctor_name_value = String(value_x, 15, doctor,  # Same y-coordinate as label
                                fontSize=10, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(doctor_name_value)

    # Doctor Signature (below Doctor Name)
    doctor_sign_label = String(-15, -10, "Doctor Sign: ",  # Moved up to fit within height 530 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(doctor_sign_label)

    # Add RIGHT-SIDE Conclusion Box (moved to the right) - NOW DYNAMIC FROM DASHBOARD (12 conclusions max) - MADE SMALLER
    # SHIFTED DOWN further (additional 5 points)
    conclusion_y_start = 32.5  # Shifted down by 45 points (81 - 45 = 36, then + 3.5 = 39.5, then - 7 = 32.5)
    
    # Create a rectangular box for conclusions (shifted right) - INCREASED HEIGHT (same position)
    # Height increased: bottom extended down (top position same). Length increased by 20 (x position fixed)
    # Rect already imported at top
    conclusion_box = Rect(455, conclusion_y_start - 55, 355, 75,  # Shifted right by 255 points total (200 + 225 + 30 = 455)
                         fillColor=None, strokeColor=colors.black, strokeWidth=1.5)
    master_drawing.add(conclusion_box)
    
    # CENTERED and STYLISH "Conclusion" header - DYNAMIC - SMALLER (AT TOP OF CONTAINER - CLOSE TO TOP LINE)
    # Box center: 200 + (325/2) = 362.5, so text should be centered around 362.5
    # Box top is at conclusion_y_start - 55, so header should be very close to top line
    conclusion_header = String(617.5, conclusion_y_start + 8, "✦ CONCLUSION ✦",  # Shifted right by 255 points total (362.5 + 225 + 30 = 617.5)
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
    start_y = conclusion_y_start - 5  # Starting Y position (shifted down by 15 points)
    
    conclusion_num = 1  # Start numbering from 1
    for row_idx, row_conclusions in enumerate(conclusion_rows):
        row_y = start_y - (row_idx * row_spacing)
        
        for col_idx, conclusion in enumerate(row_conclusions):
            # Truncate long conclusions
            display_conclusion = conclusion[:30] + "..." if len(conclusion) > 30 else conclusion
            conc_text = f"{conclusion_num}. {display_conclusion}"
            
            # Position horizontally across the box (2 conclusions per row)
            x_pos = 465 + (col_idx * 160)  # Shifted right by 255 points total (210 + 225 + 30 = 465)
            
            conc = String(x_pos, row_y, conc_text, 
                         fontSize=9, fontName="Helvetica", fillColor=colors.black)
            master_drawing.add(conc)
            
            conclusion_num += 1  # Increment for next conclusion

    print(f" Added Patient Info, Vital Parameters, {len(filtered_conclusions)} REAL Conclusions (no empty/---), and Doctor Name/Signature to ECG grid")
    
    # STEP 5: Add SINGLE master drawing to story on Page 1 (Landscape)
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

    # Helper: draw logo on every page AND ALIGNED pink grid background on Page 2
    def _draw_logo_and_footer(canvas, doc):
        import os
        
        # STEP 1: Draw FULL PAGE pink ECG grid background on Page 2 (ECG graphs page)
        if canvas.getPageNumber() == 1:  # Changed from 3 to 2
            page_width, page_height = canvas._pagesize
            
            # Fill entire page with pink background
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            
            # ECG grid colors - darker for better visibility
            light_grid_color = colors.HexColor("#ffd1d1")  
            
            major_grid_color = colors.HexColor("#ffb3b3")   
            
            # Use 57 x 40 boxes across 297mm x 210mm
            num_boxes_width = 57
            num_boxes_height = 40
            page_width_mm = 297.0
            page_height_mm = 210.0
            box_width_mm = page_width_mm / num_boxes_width   # 297/57 = 5.2105mm
            box_height_mm = page_height_mm / num_boxes_height  # 210/40 = 5.25mm

            # Draw minor grid lines (1/5th of a box)
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.6)
            minor_spacing_x = (box_width_mm / 5.0) * mm
            minor_spacing_y = (box_height_mm / 5.0) * mm

            max_x_limit = page_width_mm * mm
            x = 0
            while x <= max_x_limit:
                canvas.line(x, 0, x, page_height)
                x += minor_spacing_x
                if x > max_x_limit:
                    break
            
            # Horizontal minor lines - full page
            y = 0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += minor_spacing_y
                
            
            # Draw major grid lines - FULL PAGE
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(1.2)
            major_spacing_x = box_width_mm * mm
            major_spacing_y = box_height_mm * mm

            # Vertical major lines - 57 boxes across 297mm
            max_x_limit = page_width_mm * mm
            x = 0
            while x <= max_x_limit:
                canvas.line(x, 0, x, page_height)
                x += major_spacing_x
                if x > max_x_limit:
                    break
            
            # Horizontal major lines - full width
            y = 0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += major_spacing_y
            

        
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
            canvas.drawString(x_pos + org_label_width + 5, y_pos, patient.get("Org.", "") if patient else "")
            
            y_pos -= 15  # Move down for next line
            
            # Always draw "Phone No." label with value
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(colors.black)
            phone_label = "Phone No:"
            canvas.drawString(x_pos, y_pos, phone_label)
            
            # Calculate width of label and add small gap
            phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 10)
            canvas.setFont("Helvetica", 10)
            canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient.get("doctor_mobile", "") if patient else "")
            
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
            "RV5_plus_SV1_mV": round(rv5_sv1_sum, 3),
            "P_QRS_T_mm": [p_mm, qrs_mm, t_mm],
            "QTCF": round(qtcf_val, 1) if 'qtcf_val' in locals() and qtcf_val and qtcf_val > 0 else None,
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
        print(f" Saved 11 metrics to {metrics_path}")
    except Exception as e:
        print(f" Could not save parameters JSON: {e}")

    # Build PDF
    doc.build(story)
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


# ====================  REPORT WRAPPER ====================
