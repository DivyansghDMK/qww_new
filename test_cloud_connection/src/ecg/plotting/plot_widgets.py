"""PyQtGraph plot widget creation and configuration"""
import pyqtgraph as pg
from functools import partial
from PyQt5.QtWidgets import QGridLayout, QWidget
from typing import List, Tuple, Callable


# Lead colors for consistent visualization
LEAD_COLORS_PLOT = {
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


def create_plot_grid(plot_area: QWidget, leads: List[str], click_handler: Callable) -> Tuple[List[pg.PlotWidget], List]:
    """Create a grid of PyQtGraph plot widgets for ECG leads
    
    Args:
        plot_area: QWidget container for the plots
        leads: List of lead names (e.g., ['I', 'II', 'III', ...])
        click_handler: Function to handle plot clicks (plot_index)
    
    Returns:
        Tuple of (plot_widgets list, data_lines list)
    """
    grid = QGridLayout(plot_area)
    grid.setSpacing(8)
    plot_widgets = []
    data_lines = []
    
    # 4x3 grid positions
    positions = [(i, j) for i in range(4) for j in range(3)]
    
    for i in range(len(leads)):
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Hide Y-axis labels for cleaner display
        plot_widget.getAxis('left').setTicks([])
        plot_widget.getAxis('left').setLabel('')
        plot_widget.getAxis('bottom').setTextPen('k')
        
        # Get color for this lead
        lead_name = leads[i]
        lead_color = LEAD_COLORS_PLOT.get(lead_name, '#000000')
        
        plot_widget.setTitle(leads[i], color=lead_color, size='10pt')
        
        # Set initial and safe Y-limits; LOCKED for stability
        plot_widget.setYRange(-2000, 2000)
        vb = plot_widget.getViewBox()
        if vb is not None:
            # Prevent extreme jumps while still allowing wide physiological range
            vb.setLimits(yMin=-8000, yMax=8000)
            # LOCK X-axis to fixed 10-second range - NEVER change it
            try:
                vb.setRange(xRange=(0.0, 10.0), padding=0)
                # Lock X-axis limits to prevent any changes
                vb.setLimits(xMin=0.0, xMax=10.0)
                # Enable better rendering for smooth curves
                vb.setAspectLocked(False)
                vb.enableAutoRange(axis='y', enable=False)
            except Exception:
                pass
        
        # Make plot clickable
        plot_widget.scene().sigMouseClicked.connect(partial(click_handler, i))
        
        row, col = positions[i]
        grid.addWidget(plot_widget, row, col)
        data_line = plot_widget.plot(pen=pg.mkPen(color=lead_color, width=0.7))
        
        plot_widgets.append(plot_widget)
        data_lines.append(data_line)
    
    # R-peaks scatter plot (only if we have at least 2 plots)
    r_peaks_scatter = None
    if len(plot_widgets) > 1:
        r_peaks_scatter = plot_widgets[1].plot([], [], pen=None, symbol='o', symbolBrush='r', symbolSize=8)
    
    return plot_widgets, data_lines, r_peaks_scatter


def update_plot_data(plot_widget: pg.PlotWidget, data_line, data: List[float], time_axis: List[float] = None):
    """Update plot data for a single lead
    
    Args:
        plot_widget: PyQtGraph PlotWidget instance
        data_line: Plot data line item
        data: ECG signal data
        time_axis: Optional time axis (if None, uses indices)
    """
    try:
        if time_axis is None:
            time_axis = list(range(len(data)))
        
        # Update the plot line
        data_line.setData(time_axis, data)
        
    except Exception as e:
        print(f"Error updating plot data: {e}")


def update_plot_y_range(plot_widget: pg.PlotWidget, y_min: float, y_max: float):
    """Update Y-axis range for a plot
    
    Args:
        plot_widget: PyQtGraph PlotWidget instance
        y_min: Minimum Y value
        y_max: Maximum Y value
    """
    try:
        plot_widget.setYRange(y_min, y_max, padding=0)
    except Exception as e:
        print(f"Error updating plot Y range: {e}")
