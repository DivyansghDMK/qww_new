from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, 
    QLineEdit, QComboBox, QSlider, QGroupBox, QListWidget, QDialog,
    QGridLayout, QFormLayout, QSizePolicy, QMessageBox, QApplication, QRadioButton, QScrollArea
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QRect, QPoint, QEvent
from PyQt5.QtGui import QIntValidator, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from utils.settings_manager import SettingsManager
from utils.localization import translate_text
import os
import matplotlib.pyplot as plt
import pandas as pd
import json 
import sys

# NOTE: ECGRecording class removed (was never used in codebase)
# Recording functionality is handled by SessionRecorder in utils/session_recorder.py
        
class Lead12BlackPage(QWidget):
    def __init__(self, parent=None, dashboard=None):
        super().__init__(parent)
        self.dashboard = dashboard
        self.setStyleSheet("background: black;")
        layout = QVBoxLayout(self)
        self.canvases = []
        self.lines = []
        self.ecg_buffers = [np.zeros(5000) for _ in range(12)]
        self.ptrs = [0 for _ in range(12)]
        self.window_size = 1000
        self.lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            label = QLabel(self.lead_names[i])
            label.setStyleSheet("color: white; font-size: 14px; font-weight: bold; margin-bottom: 2px;")
            label.setFixedWidth(70)
            layout.addWidget(label, alignment=Qt.AlignLeft)
            fig = Figure(figsize=(2, 2), facecolor='black')
            ax = fig.add_subplot(111)
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-3, 3)
            ax.axvline(x=0, color='white', linestyle='--', linewidth=1)
            ax.set_title("", color='white', fontsize=12, loc='left')
            line, = ax.plot(np.zeros(self.window_size), color='lime', lw=1)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.canvases.append(canvas)
            self.lines.append(line)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)  # 20 FPS (reduced from 33 FPS for performance)
    
    def closeEvent(self, event):
        """Clean up resources when widget is closed"""
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
            # self.timer.deleteLater() # Removed: can cause issues during close
        super().closeEvent(event)

    def update_data(self):
        for i in range(12):
            # Slide a window over the simulated ECG for animation
            self.ptrs[i] = (self.ptrs[i] + 1) % (len(self.ecg_buffers[i]) - self.window_size)
            window = self.ecg_buffers[i][self.ptrs[i]:self.ptrs[i]+self.window_size]
            self.lines[i].set_ydata(window)
            # --- P peak detection and labeling for each lead ---
            if len(window) >= 1000:
                try:
                    # Placeholder for PQRST detection logic
                    p_peaks = np.array([100, 200, 300])  # Dummy values for illustration
                    ax = self.canvases[i].figure.axes[0]
                    main_line = ax.lines[0]
                    ax.lines = [main_line]
                    # Remove old text labels
                    for txt in ax.texts:
                        txt.remove()
                    # Plot green markers and labels for P peaks only
                    if len(p_peaks) > 0:
                        ax.plot(p_peaks, window[p_peaks], 'o', color='green', label='P', markersize=8, zorder=10)
                        for idx in p_peaks:
                            ax.text(idx, window[idx]+0.3, 'P', color='green', fontsize=10, ha='center', va='bottom', zorder=11)
                    # Optional: update legend
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
                except Exception as e:
                    print(f"ECG analysis error in lead {self.lead_names[i]}:", e)
            self.canvases[i].draw()
        # --- Lead II metrics and dashboard update (as before) ---
        lead_ii_signal = self.ecg_buffers[1][self.ptrs[1]:self.ptrs[1]+self.window_size]
        if len(lead_ii_signal) >= 1000:
            try:
                # TODO: Replace with real metric calculations from twelve_lead_test.py
                # These are placeholder/dummy values for testing visualization only
                pr_interval = 0.2  # TODO: Calculate from real P-R wave detection
                qrs_duration = 0.08  # TODO: Calculate from real QRS complex
                qt_interval = 0.4  # TODO: Calculate from real Q-T interval
                qtc_interval = 0.42  # TODO: Calculate corrected QT (Bazett's formula)
                st_segment = "--"  # TODO: Calculate ST elevation/depression
                with open("ecg_metrics_output.txt", "w") as f:
                    f.write("# ECG Metrics Output (TEST/DEBUG ONLY)\n")
                    f.write("# WARNING: These are placeholder values, not real calculations!\n")
                    f.write("# Format: PR_interval(ms), QRS_duration(ms), QTc_interval(ms), ST_segment\n")
                    f.write(f"{pr_interval*1000}, {qrs_duration*1000}, {qtc_interval*1000}, {st_segment}\n")
                    # TODO: Replace with real peak detection from Pan-Tompkins algorithm
                    f.write(f"P_peaks: {list(np.array([100, 200, 300]))}  # PLACEHOLDER\n")
                    f.write(f"Q_peaks: {list(np.array([150, 250, 350]))}  # PLACEHOLDER\n")
                    f.write(f"R_peaks: {list(np.array([180, 280, 380]))}  # PLACEHOLDER\n")
                    f.write(f"S_peaks: {list(np.array([210, 310, 410]))}  # PLACEHOLDER\n")
                    f.write(f"T_peaks: {list(np.array([240, 340, 440]))}  # PLACEHOLDER\n")
                if self.dashboard and hasattr(self.dashboard, "update_ecg_metrics"):
                    self.dashboard.update_ecg_metrics(pr_interval, qrs_duration, qtc_interval, st_segment)
                    QTimer.singleShot(0, self.dashboard.repaint)
            except Exception as e:
                print("ECG analysis error:", e)

class SlidingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Responsive sizing based on parent size (captured once at creation)
        if parent:
            parent_width = parent.width()
            parent_height = parent.height()
            
            # Calculate responsive panel size (25-35% of parent width, max 900px)
            # On smaller screens, allow the panel to take more percentage but set a lower minimum
            if parent_width < 1000:
                panel_width = min(max(int(parent_width * 0.40), 320), 500)
            else:
                panel_width = min(max(int(parent_width * 0.30), 400), 900)
            
            panel_height = min(max(int(parent_height * 0.85), 450), 1000)
        else:
            panel_width, panel_height = 600, 800
        
        # Store base size so panel stays visually identical even when window is maximized
        self.base_panel_width = panel_width
        self.base_panel_height = panel_height
        self.panel_width = panel_width
        self.panel_height = panel_height
        
        # Set size policy for responsiveness
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(panel_width, panel_height)
        
        # Responsive styling with dynamic sizing
        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border: 3px solid #e0e0e0;
                border-radius: 15px;
            }}
        """)
        
        # Initialize position off-screen to the right
        if parent:
            self.setGeometry(parent.width(), (parent.height() - self.height()) // 2, 
                           panel_width, panel_height)
        else:
            self.setGeometry(1200, 200, panel_width, panel_height)
        
        # Create responsive layout with dynamic margins
        self.layout = QVBoxLayout(self)
        margin_size = max(12, min(30, int(panel_width * 0.04)))  # Initial margins
        spacing_size = max(10, min(25, int(panel_height * 0.03)))  # Initial spacing
        
        self.layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        self.layout.setSpacing(spacing_size)
        
        # Content area with scroll support
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.layout.addWidget(self.content_widget)
        
        # Animation setup
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        self.is_visible = False
        self.is_animating = False
        
        # Store responsive parameters (frozen to initial values)
        self.margin_size = margin_size
        self.spacing_size = spacing_size
        self.base_margin_size = margin_size
        self.base_spacing_size = spacing_size
        
        # Add resize event handler for responsiveness
        # Keep panel visually constant; only reposition when parent resizes
        if parent:
            parent.resizeEvent = self.parent_resize_handler
        
    def parent_resize_handler(self, event):
        """Handle parent resize events for responsive behavior"""
        if hasattr(event, 'size'):
            self.update_responsive_sizing()
        if hasattr(event, 'oldSize'):
            event.oldSize = event.size()
        event.accept()
        
    def update_responsive_sizing(self):
        """Keep panel size and internal spacing fixed; only move it with parent."""
        if self.parent:
            # Ensure we keep the original visual size
            self.panel_width = getattr(self, "base_panel_width", self.panel_width)
            self.panel_height = getattr(self, "base_panel_height", self.panel_height)
            self.setFixedSize(self.panel_width, self.panel_height)

            # Restore original margins/spacing
            self.margin_size = getattr(self, "base_margin_size", self.margin_size)
            self.spacing_size = getattr(self, "base_spacing_size", self.spacing_size)
            self.layout.setContentsMargins(self.margin_size, self.margin_size, 
                                        self.margin_size, self.margin_size)
            self.layout.setSpacing(self.spacing_size)
            
            # Reposition panel on the right side if visible
            if self.is_visible:
                self.reposition_panel()
        
    def reposition_panel(self):
        if self.parent and self.is_visible:
            target_x = self.parent.width() - self.width() - 15  # Reduced margin
            target_y = (self.parent.height() - self.height()) // 2
            self.move(target_x, target_y)
        
    def set_title(self, title):
        pass
        
    def clear_content(self):
        # Clear existing content
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Reset processed widgets tracking for new content
        if hasattr(self, '_processed_widgets'):
            self._processed_widgets.clear()
                
    def slide_in(self, content_widget=None, title="Settings Panel"):
        
        if self.parent and not self.is_animating:
            self.is_animating = True
            self.clear_content()
            
            # Update responsive sizing before showing
            self.update_responsive_sizing()
            
            if content_widget:
                # Make content widget responsive
                self.make_content_responsive(content_widget)
                self.content_layout.addWidget(content_widget)

            # Force Layout so first-open renders correctly
            try:
                self.content_widget.adjustSize()
                self.layout.activate()
                QApplication.processEvents()
                self.update()
            except:
                pass
            
            # Calculate target position (centered on the right side with proper margins)
            # Ensure panel doesn't go off-screen on small devices
            target_x = max(10, self.parent.width() - self.width() - 10)  # At least 10px from right edge
            target_y = max(10, (self.parent.height() - self.height()) // 2)  # At least 10px from top/bottom
            
            # Ensure panel doesn't exceed parent bounds
            if target_y + self.height() > self.parent.height() - 10:
                target_y = self.parent.height() - self.height() - 10

            start_rect = QRect(self.parent.width(), target_y, self.width(), self.height())
            end_rect = QRect(target_x, target_y, self.width(), self.height())

            # Set up animation using Local QRect
            self.setGeometry(start_rect)

            self.animation.setStartValue(start_rect)
            self.animation.setEndValue(end_rect)

            # Disconnect any existing connections
            try:
                self.animation.finished.disconnect()
            except:
                pass
            
            # Connect animation finished signal
            self.animation.finished.connect(self.on_slide_in_finished)
            
            self.show()
            self.raise_()
            self.animation.start()

    def make_content_responsive(self, content_widget):
        if hasattr(content_widget, 'layout'):
            layout = content_widget.layout()
            if layout:
                # Adjust margins and spacing based on panel size
                content_margin = max(15, min(35, int(self.panel_width * 0.04)))
                content_spacing = max(10, min(20, int(self.panel_height * 0.025)))
                
                layout.setContentsMargins(content_margin, content_margin, 
                                       content_margin, content_margin)
                layout.setSpacing(content_spacing)
                
                # Make child widgets responsive
                self.make_children_responsive(content_widget)
    
    def make_children_responsive(self, parent_widget):
        """Make child widgets responsive, with proper error handling for touch screens"""
        try:
            # Track processed widgets to avoid infinite recursion
            if not hasattr(self, '_processed_widgets'):
                self._processed_widgets = set()
            
            for child in parent_widget.findChildren(QWidget):
                # Skip if already processed
                if id(child) in self._processed_widgets:
                    continue
                
                # Mark as processed
                self._processed_widgets.add(id(child))
                
                try:
                    # Adjust font sizes for better readability on small panels
                    if self.panel_width < 500:
                        if hasattr(child, 'font'):
                            font = child.font()
                            if font.pointSize() > 12:
                                font.setPointSize(font.pointSize() - 2)
                                child.setFont(font)
                    
                    # Recursively process children (only if not already processed)
                    if child != parent_widget:
                        self.make_children_responsive(child)
                        
                except Exception:
                    # Skip widgets that cause errors (common on touch screens during initialization)
                    continue
                    
        except Exception:
            # Fail silently to prevent UI crashes on touch screens
            pass

    def on_slide_in_finished(self):
        self.is_visible = True
        self.is_animating = False
            
    def slide_out(self):
        if self.parent and self.is_visible and not self.is_animating:
            self.is_animating = True

            # Calculate end position (off-screen to the right)
            end_x = self.parent.width()
            end_y = (self.parent.height() - self.height()) // 2

            # Set up animation using Local QRect
            start_rect = QRect(self.x(), self.y(), self.width(), self.height())
            end_rect = QRect(end_x, end_y, self.width(), self.height())
            self.animation.setStartValue(start_rect)
            self.animation.setEndValue(end_rect)
            
            # Disconnect any existing connections
            try:
                self.animation.finished.disconnect()
            except:
                pass
            
            # Connect animation finished signal
            self.animation.finished.connect(self.on_slide_out_finished)
            self.animation.start()

    def on_slide_out_finished(self):
        self.hide()
        self.is_visible = False
        self.is_animating = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent and self.is_visible:
            self.reposition_panel()

class ECGMenu(QGroupBox):
    def __init__(self, parent=None, dashboard=None):
        super().__init__("", parent)
        self.dashboard = dashboard
        # Reference to ECG test page for cross-component communication
        self.ecg_test_page = None
        self.settings_manager = None
        self.sliding_panel = None
        self.settings_changed_callback = None
        self.current_language = 'en'

        self.setStyleSheet("QGroupBox { font: bold 14pt Arial; background-color: #fff; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        self.buttons = {}
        self.menu_button_defs = [
            ("Save ECG", self.on_save_ecg),
            ("Open ECG", self.on_open_ecg),
            ("Working Mode", self.on_working_mode),
            ("Report Setup", self.on_report_setup),
            ("Set Filter", self.on_set_filter),
            ("System Setup", self.on_system_setup),
            ("Load Default", self.on_load_default),
            ("Exit", self.on_exit)
        ]
        for text, handler in self.menu_button_defs:
            btn = QPushButton(self.tr(text))
            btn.setFixedHeight(36)
            btn.clicked.connect(handler)
            layout.addWidget(btn)
            self.buttons[text] = btn
        layout.addStretch(1)
    
        # Initialize sliding panel
        self.sliding_panel = None
        self.current_panel_content = None
        self.current_open_panel = None
        self.panel_buttons = {}
        
        # Store parent reference for responsive updates
        self.parent_widget = None
        
        # Connect to parent resize events
        if parent:
            self.setup_parent_monitoring(parent)
        
        # Setup global resize monitoring
        QTimer.singleShot(100, self.setup_global_resize_monitoring)

    def set_ecg_test_page(self, ecg_test_page):
        """Attach the active ECG test page so menu actions can interact with it."""
        self.ecg_test_page = ecg_test_page

    def update_language(self, language=None):
        if language:
            self.current_language = language
        elif self.settings_manager:
            lang = self.settings_manager.get_setting("system_language", "en")
            self.current_language = lang
        self.refresh_button_texts()
        if self.sliding_panel and self.sliding_panel.is_visible and self.current_open_panel:
            # Reopen current panel content to update texts
            panel = self.current_open_panel
            self.hide_sliding_panel()
            reopen_map = {
                "Save ECG": self.show_save_ecg,
                "Open ECG": self.show_open_ecg,
                "Working Mode": self.show_working_mode,
                "Report Setup": self.show_report_setup,
                "Set Filter": self.show_set_filter,
                "System Setup": self.show_system_setup,
                "Load Default": self.show_load_default,
                "Exit": self.show_exit,
            }
            if panel in reopen_map:
                reopen_map[panel]()

    def tr(self, text):
        return translate_text(text, getattr(self, "current_language", "en"))

    def refresh_button_texts(self):
        if hasattr(self, "menu_button_defs"):
            for label, _ in self.menu_button_defs:
                btn = self.buttons.get(label)
                if btn:
                    btn.setText(self.tr(label))

    def setup_parent_monitoring(self, parent):
        """Setup monitoring for parent widget changes"""
        if parent and hasattr(parent, 'resizeEvent'):
            # Store original resize event
            original_resize = parent.resizeEvent
            
            def enhanced_resize_event(event):
                # Call original resize event
                if hasattr(original_resize, '__call__'):
                    original_resize(event)
                
                # Update sliding panel if it exists
                if hasattr(self, 'sliding_panel') and self.sliding_panel:
                    self.sliding_panel.update_responsive_sizing()
                    if self.sliding_panel.is_visible:
                        self.sliding_panel.reposition_panel()
                
                event.accept()
            
            # Replace the resize event handler
            parent.resizeEvent = enhanced_resize_event

    def create_resize_handler(self, original_resize_event):
        def resize_handler(event):
            # Call original resize event
            if original_resize_event:
                original_resize_event(event)
            
            # Update sliding panel if it exists
            if self.sliding_panel and hasattr(self.sliding_panel, 'update_responsive_sizing'):
                self.sliding_panel.update_responsive_sizing()
                
        return resize_handler

    def setup_global_resize_monitoring(self):
        app = QApplication.instance()
        if app:
            # Monitor all top-level windows
            for widget in app.topLevelWidgets():
                if hasattr(widget, 'resizeEvent'):
                    original_resize = widget.resizeEvent
                    widget.resizeEvent = self.create_global_resize_handler(original_resize)
    
    def create_global_resize_handler(self, original_resize_event):
        def global_resize_handler(event):
            # Call original resize event
            if original_resize_event:
                original_resize_event(event)
            
            # Update sliding panel if it exists
            if self.sliding_panel and hasattr(self.sliding_panel, 'update_responsive_sizing'):
                self.sliding_panel.update_responsive_sizing()
                
        return global_resize_handler

    # Placeholder methods to be connected externally
    def on_save_ecg(self):
        self.show_save_ecg()
    def on_open_ecg(self):
        self.show_open_ecg()
    def on_working_mode(self):
        self.show_working_mode()
    def on_report_setup(self):
        self.show_report_setup()
    def on_set_filter(self):
        self.show_set_filter()
    def on_system_setup(self):
        self.show_system_setup()
    def on_load_default(self):
        self.show_load_default()
    def on_exit(self):
        self.show_exit()


    def create_scrollable_content(self, content_widget):
        from PyQt5.QtWidgets import QScrollArea
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Responsive scroll area styling
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar:horizontal {
                background: #f0f0f0;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #c0c0c0;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #a0a0a0;
            }
        """)
        
        return scroll_area

    def ensure_sliding_panel_ready(self):
        """Ensure the sliding panel exists with up-to-date responsive metrics."""
        # Ensure we have a reference to the ECG test page parent
        if not hasattr(self, 'parent_widget') or not self.parent_widget:
            parent = self.parent()
            while parent and not hasattr(parent, 'grid_widget'):
                parent = parent.parent()
            if parent:
                self.parent_widget = parent
        
        parent = getattr(self, 'parent_widget', None)
        if not parent:
            parent = self.parent()
            while parent and not hasattr(parent, 'grid_widget'):
                parent = parent.parent()
            if parent:
                self.parent_widget = parent
        
        # Lazily create sliding panel if missing
        if not self.sliding_panel and parent:
            self.sliding_panel = SlidingPanel(parent)
            self.setup_parent_monitoring(parent)
            print("Sliding panel created as overlay (not in layout)")
        elif not self.sliding_panel:
            print("Could not find parent widget for sliding panel")
        
        if self.sliding_panel:
            self.sliding_panel.update_responsive_sizing()

    def show_sliding_panel(self, content_widget, title, button_name):
        
        # If same button clicked while visible → toggle close
        if self.sliding_panel and self.sliding_panel.is_visible and self.current_open_panel == button_name:
            self.hide_sliding_panel()
            self.current_open_panel = None
            return

        # If panel exists and is visible but a different button was clicked → IMMEDIATE SWAP (no slide-out/in)
        if self.sliding_panel and self.sliding_panel.is_visible and self.current_open_panel != button_name:
            try:
                self.sliding_panel.animation.stop()
            except:
                pass
            self.sliding_panel.is_animating = False
            self.cancel_draft_settings()

            # Prepare content (wrap in scroll area for short panels)
            target = self.create_scrollable_content(content_widget) if (content_widget and self.sliding_panel.panel_height < 700) else content_widget

            # Replace content immediately
            self.sliding_panel.clear_content()
            if target:
                # Make responsive and add
                try:
                    self.sliding_panel.make_content_responsive(target)
                except:
                    pass
                self.sliding_panel.content_layout.addWidget(target)

            # Ensure sizing/position and keep it visible
            self.sliding_panel.update_responsive_sizing()
            self.sliding_panel.reposition_panel()
            self.sliding_panel.show()
            self.sliding_panel.raise_()

            self.current_open_panel = button_name
            return

        # Create sliding panel if it doesn't exist
        if not self.sliding_panel:
            parent = self.parent_widget
            if not parent:
                parent = self.parent()
                while parent and not hasattr(parent, 'grid_widget'):
                    parent = parent.parent()
            if parent:
                self.sliding_panel = SlidingPanel(parent)
                self.setup_parent_monitoring(parent)
                # DO NOT add sliding panel to layout
                print("Sliding panel created as overlay")
            else:
                print("Could not find parent widget")

        # First open → slide in
        if self.sliding_panel:
            target = self.create_scrollable_content(content_widget) if (content_widget and self.sliding_panel.panel_height < 700) else content_widget
            self.sliding_panel.slide_in(target, title)
            self.current_open_panel = button_name
        else:
            print("Sliding panel is None") 

    def hide_sliding_panel(self):
        if self.sliding_panel and self.sliding_panel.is_visible:
            self.cancel_draft_settings()
            self.sliding_panel.slide_out()
            self.current_open_panel = None

    def set_draft_setting(self, key, value):
        if not hasattr(self, 'panel_draft') or self.panel_draft is None:
            self.panel_draft = {}
        self.panel_draft[key] = value

    def commit_draft_settings(self):
        if hasattr(self, 'panel_draft') and isinstance(self.panel_draft, dict):
            for key, value in self.panel_draft.items():
                try:
                    self.on_setting_changed(key, value)
                except Exception as e:
                    print(f"Commit draft error for {key}: {e}")
        self.panel_original = dict(self.panel_draft) if hasattr(self, 'panel_draft') else {}
        self.panel_draft = {}

    def cancel_draft_settings(self):
        self.panel_draft = {}

    # ----------------------------- Save ECG Details-----------------------------

    # Modified methods to use sliding panel
    def show_save_ecg(self):
        # Ensure we have access to the parent widget with ECG data
        if not hasattr(self, 'parent_widget') or not self.parent_widget:
            # Try to find the parent widget
            parent = self.parent()
            while parent and not hasattr(parent, 'data'):
                parent = parent.parent()
            if parent:
                self.parent_widget = parent
        
        self.ensure_sliding_panel_ready()
        
        content_widget = self.create_save_ecg_content()
        # Always wrap in scroll area so all fields stay visible on smaller panels
        try:
            content_widget = self.create_scrollable_content(content_widget)
        except Exception:
            pass
        self.show_sliding_panel(content_widget, self.tr("Save ECG Details"), "Save ECG Details")

    def get_user_specific_patient_file(self):
        """Get user-specific patient details file path"""
        username = "default"
        
        # Try to get username from dashboard
        if self.dashboard and hasattr(self.dashboard, 'username') and self.dashboard.username:
            username = self.dashboard.username
        
        # Sanitize username for file name (remove special characters)
        safe_username = "".join(c for c in username if c.isalnum() or c in ('_', '-')).lower()
        return f"patient_details_{safe_username}.json"

    def create_save_ecg_content(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Responsive margins and spacing
        margin_size = getattr(self.sliding_panel, 'margin_size', 30) if self.sliding_panel else 30
        spacing_size = getattr(self.sliding_panel, 'spacing_size', 20) if self.sliding_panel else 20
        
        layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        layout.setSpacing(spacing_size)

        # Responsive title with dynamic font size
        title = QLabel(self.tr("Save ECG Details"))
        title_font_size = max(16, min(24, int(margin_size * 0.8)))
        title.setStyleSheet(f"""
            QLabel {{
                font: bold {max(14, title_font_size-2)}pt 'Arial';
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                border: 2px solid #343434;
                border-radius: 12px;
                padding: {max(12, margin_size-12)}px;
                margin: {max(4, margin_size-18)}px;
            }}
        """)
        title.setAlignment(Qt.AlignCenter)
        # Prevent the header from stretching vertically when the parent window is maximized
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(title)

        # Main form container with responsive styling
        form_frame = QFrame()
        # Responsive padding and margins for small screens
        form_padding = max(10, min(20, int(margin_size * 0.5)))
        form_margin = max(5, min(10, int(margin_size * 0.25)))
        form_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                padding: {max(8, form_padding-2)}px;
                margin: {max(4, form_margin-2)}px;
            }}
        """)
        form_layout = QVBoxLayout(form_frame)
        form_layout.setSpacing(max(10, min(15, spacing_size-5)))
        form_layout.setContentsMargins(max(8, min(15, int(margin_size * 0.3))), 
                                      max(8, min(15, int(margin_size * 0.3))), 
                                      max(8, min(15, int(margin_size * 0.3))), 
                                      max(8, min(15, int(margin_size * 0.3))))
        
        labels = ["Org.", "Doctor", "Phone No.", "Patient Name"]
        entries = {}

        # Responsive form fields
# Style for labels and inputs
        label_style = f"""
            QLabel {{
                font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                color: #2c3e50;
                background: #f8f9fa;
                padding: {max(8, int(margin_size * 0.4))}px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                min-width: 140px;
                max-width: 140px;
            }}
        """
        input_style = f"""
            QLineEdit {{
                font: {max(10, int(margin_size * 0.5))}pt Arial;
                color: #ff6600;
                background: white;
                padding: {max(8, int(margin_size * 0.4))}px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                min-height: {max(25, min(35, int(margin_size * 0.9)))}px;
            }}
            QLineEdit:focus {{
                border: 2px solid #ff6600;
                background: #fff8f0;
            }}
            QLineEdit:hover {{
                border: 1px solid #ffb347;
                background: #fafafa;
            }}
        """

        # Responsive form fields
        for label in labels:
            row = QHBoxLayout()
            row.setSpacing(10)

            lbl = QLabel(self.tr(label))
            lbl.setStyleSheet(label_style)
            
            entry = QLineEdit()
            entry.setPlaceholderText(self.tr(f"Enter {label}"))
            entry.setStyleSheet(input_style)
            entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(lbl)
            row.addWidget(entry)
            form_layout.addLayout(row)
            entries[label] = entry
        
        entries["Doctor"].setMaxLength(20)
        entries["Phone No."].setMaxLength(10)
        entries["Patient Name"].setMaxLength(20) 

        # Age field with responsive sizing

        age_row = QHBoxLayout()
        age_row.setSpacing(10)
        
        age_lbl = QLabel(self.tr("Age"))
        age_lbl.setStyleSheet(label_style)

        age_entry = QLineEdit()
        age_entry.setPlaceholderText(self.tr("Enter Age"))
        age_entry.setValidator(QIntValidator(0, 120, age_entry))
        age_entry.setStyleSheet(input_style)
        age_entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        age_row.addWidget(age_lbl)
        age_row.addWidget(age_entry)
        form_layout.addLayout(age_row)
        entries["Age"] = age_entry

        # Gender field with responsive sizing

        gender_row = QHBoxLayout()
        gender_row.setSpacing(10)
        
        gender_lbl = QLabel(self.tr("Gender"))
        gender_lbl.setStyleSheet(label_style)

        gender_menu = QComboBox()
        gender_menu.addItems([self.tr("Select Gender"), self.tr("Male"), self.tr("Female"), self.tr("Other")])
        gender_menu.setStyleSheet(f"""
            QComboBox {{
                font: {max(10, int(margin_size * 0.5))}pt Arial;
                padding: {max(8, int(margin_size * 0.4))}px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: white;
                color: #ff6600;
                min-height: {max(25, min(35, int(margin_size * 0.9)))}px;
            }}
            QComboBox:focus {{
                border: 2px solid #343434;
                background: white;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ff6600;
                margin-right: 10px;
            }}
            QComboBox QAbstractItemView {{
                background: white;
                border: 2px solid #ff6600;
                border-radius: 8px;
                selection-background-color: #ff6600;
                selection-color: white;
                outline: none;
            }}
        """)
        
        gender_menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gender_row.addWidget(gender_lbl)
        gender_row.addWidget(gender_menu)
        form_layout.addLayout(gender_row)

        # Prefill previously saved values if available (from centralized database)
        try:
            prefill = None
            user_file = self.get_user_specific_patient_file()
            
            # 1) Use in-memory cached details if present
            if hasattr(self, "patient_details") and isinstance(self.patient_details, dict):
                prefill = self.patient_details
            
            # 2) If not in memory, load the LAST patient from centralized database
            if prefill is None:
                try:
                    # Get path to centralized database (in modularecg folder)
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    patients_db_file = os.path.join(base_dir, "all_patients.json")
                    
                    if os.path.exists(patients_db_file):
                        with open(patients_db_file, "r") as jf:
                            all_patients = json.load(jf)
                            if all_patients.get("patients") and len(all_patients["patients"]) > 0:
                                # Get the last patient (most recent)
                                prefill = all_patients["patients"][-1]
                                # Cache in memory for subsequent opens during this session
                                setattr(self, "patient_details", prefill)
                                print(f" Loaded last patient from centralized DB: {prefill.get('patient_name', 'Unknown')}")
                except Exception as e:
                    print(f" Could not load from centralized DB: {e}")
                    prefill = None

            if prefill:
                pd = prefill

                
                # Org. (optional in cached data)
                if "Org." in pd and pd["Org."]:
                    entries["Org."].setText(pd["Org."]) 
                # Doctor
                if "doctor" in pd and pd["doctor"]:
                    entries["Doctor"].setText(pd["doctor"])
                # Phone No.
                if "doctor_mobile" in pd and pd["doctor_mobile"]:
                    entries["Phone No."].setText(pd["doctor_mobile"])
                # Patient Name
                first = pd.get("first_name", "") or ""
                last = pd.get("last_name", "") or ""
                full_name = (first + (" " + last if last else "")).strip()
                if full_name:
                    entries["Patient Name"].setText(full_name)
                # Age
                if "age" in pd and pd["age"] is not None:
                    entries["Age"].setText(str(pd["age"]))
                # Gender
                if "gender" in pd and pd["gender"]:
                    
                    idx = gender_menu.findText(str(pd["gender"]))
                    if idx != -1:
                        gender_menu.setCurrentIndex(idx)
        except Exception:
            pass

        layout.addWidget(form_frame)

        # Submit button
        submit_btn = QPushButton(self.tr("Save ECG"))
        submit_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #28a745, stop:1 #20c997);
                color: white;
                border: 2px solid #28a745;
                border-radius: 10px;
                padding: {max(8, int(margin_size * 0.35))}px;
                font: bold {max(11, int(margin_size * 0.38))}pt Arial;
                min-height: {max(32, int(margin_size * 1.0))}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #20c997, stop:1 #28a745);
                border: 2px solid #20c997;
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #1e7e34, stop:1 #1c7430);
                border: 2px solid #1e7e34;
            }}
        """)
        submit_btn.clicked.connect(lambda: self.submit_ecg_details(entries, gender_menu))
        submit_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(submit_btn)

        layout.addStretch(1)

        return widget

    def submit_ecg_details(self, entries, gender_menu):
        values = {label: entries[label].text().strip() for label in ["Org.", "Doctor", "Phone No.", "Patient Name", "Age"]}
        values["Gender"] = gender_menu.currentText()

        if any(v == "" for v in values.values()) or values["Gender"] == "Select":
            QMessageBox.warning(self.parent(), self.tr("Missing Data"), self.tr("Please fill all the fields and select gender."))
            return

        # Store patient details on the menu and dashboard for PDF generation
        try:
            from datetime import datetime
            name = values["Patient Name"]
            first, *rest = name.split()
            patient_details = {
                "first_name": first,
                "last_name": " ".join(rest),
                "age": values["Age"],
                "gender": values["Gender"],
                "doctor": values["Doctor"],
                "doctor_mobile": values["Phone No."],
                "Org.": values.get("Org.", ""),
                "patient_name": values.get("Patient Name", ""),
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            setattr(self, "patient_details", patient_details)
            if self.dashboard:
                setattr(self.dashboard, "patient_details", patient_details)
            
            # Persist to centralized all_patients.json database
            try:
                # Get path to centralized database (in modularecg folder)
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                patients_db_file = os.path.join(base_dir, "all_patients.json")
                
                # Load existing patients database
                all_patients = {"patients": []}
                if os.path.exists(patients_db_file):
                    try:
                        with open(patients_db_file, "r") as jf:
                            all_patients = json.load(jf)
                        print(f" Loaded existing patients database: {len(all_patients.get('patients', []))} patients")
                    except Exception as load_err:
                        print(f" Could not load patients database, creating new: {load_err}")
                        all_patients = {"patients": []}
                
                # Add unique ID to new patient
                patient_details["id"] = len(all_patients.get("patients", [])) + 1
                
                # Append new patient to the list
                if "patients" not in all_patients:
                    all_patients["patients"] = []
                all_patients["patients"].append(patient_details)
                
                # Save back to centralized file
                with open(patients_db_file, "w") as jf:
                    json.dump(all_patients, jf, indent=2)
                
                print(f" Patient saved to centralized database! Total patients: {len(all_patients['patients'])}")
                
            except Exception as disk_err:
                print(f" Could not persist patient details to centralized DB: {disk_err}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f" Could not cache patient details: {e}")

        # Success message and close panel
        QMessageBox.information(self.parent(), "Saved", f" Patient details saved successfully!\n\nTotal patients in database: {len(all_patients.get('patients', []))}")
        # Close the panel after successful save (values remain persisted for next open)
        self.hide_sliding_panel()

    # ----------------------------- Open ECG Details-----------------------------

    def show_open_ecg(self):
        """Show open ECG file dialog"""
        self.ensure_sliding_panel_ready()
        content_widget = self.create_open_ecg_content()
        self.show_sliding_panel(content_widget, self.tr("Open ECG File"), "Open ECG Details")

    def create_open_ecg_content(self):
        # Create a simple open ECG Details interface
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Responsive margins and spacing
        margin_size = getattr(self.sliding_panel, 'margin_size', 20) if self.sliding_panel else 20
        spacing_size = getattr(self.sliding_panel, 'spacing_size', 15) if self.sliding_panel else 15
        
        layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        layout.setSpacing(spacing_size)

        # Professional title
        title = QLabel(self.tr("Open ECG File"))
        title_font_size = max(16, min(22, int(margin_size * 0.8)))
        title.setStyleSheet(f"""
            QLabel {{
                font: bold {title_font_size}pt Arial;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                border: 2px solid #ff6600;
                border-radius: 12px;
                padding: {max(12, int(margin_size * 0.6))}px;
                margin: {max(8, int(margin_size * 0.4))}px;
                min-height: {max(30, int(margin_size * 1.5))}px;
            }}
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # File selection container with responsive styling
        file_frame = QFrame()
        # Responsive padding and margins for small screens
        file_padding = max(10, min(20, int(margin_size * 0.5)))
        file_margin = max(5, min(10, int(margin_size * 0.25)))
        file_frame.setStyleSheet(f"""
            QFrame {{
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: {file_padding}px;
                margin: {file_margin}px;
            }}
        """)
        file_layout = QVBoxLayout(file_frame)
        file_layout.setSpacing(max(8, min(15, spacing_size)))
        file_layout.setContentsMargins(max(8, min(15, int(margin_size * 0.3))), 
                                      max(8, min(15, int(margin_size * 0.3))), 
                                      max(8, min(15, int(margin_size * 0.3))), 
                                      max(8, min(15, int(margin_size * 0.3))))

        # File path display
        path_label = QLabel(self.tr("Selected File:"))
        path_label.setStyleSheet(f"""
            QLabel {{
                font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                color: #2c3e50;
                background: transparent;
                padding: 5px;
            }}
        """)
        file_layout.addWidget(path_label)
        
        self.file_path_display = QLabel(self.tr("No file selected"))
        self.file_path_display.setStyleSheet(f"""
            QLabel {{
                font: {max(10, int(margin_size * 0.5))}pt Arial;
                color: #666;
                background: #f8f9fa;
                padding: {max(6, min(12, int(margin_size * 0.3)))}px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                min-height: {max(25, min(40, int(margin_size * 1.2)))}px;
            }}
        """)
        self.file_path_display.setWordWrap(True)  # Allow text wrapping on small screens
        self.file_path_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_layout.addWidget(self.file_path_display)

        # File format selection
        format_label = QLabel("File Format:")
        format_label.setStyleSheet(f"""
            QLabel {{
                font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                color: #2c3e50;
                background: transparent;
                padding: 5px;
            }}
        """)
        file_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Auto-detect", "CSV", "TXT", "JSON", "XML", "DICOM"])
        combo_padding = max(6, min(10, int(margin_size * 0.3)))
        combo_height = max(25, min(40, int(margin_size * 1.2)))
        self.format_combo.setStyleSheet(f"""
            QComboBox {{
                font: {max(10, int(margin_size * 0.5))}pt Arial;
                color: #2c3e50;
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                padding: {combo_padding}px;
                min-height: {combo_height}px;
            }}
            QComboBox:hover {{
                border: 2px solid #ffb347;
            }}
            QComboBox:focus {{
                border: 2px solid #ff6600;
            }}
            QComboBox::drop-down {{
                border: none;
                width: {max(18, min(25, int(margin_size * 0.5)))}px;
            }}
            QComboBox QAbstractItemView {{
                background: white;
                border: 2px solid #ff6600;
                border-radius: 6px;
                selection-background-color: #ff6600;
                selection-color: white;
                font: {max(10, int(margin_size * 0.5))}pt Arial;
                padding: {max(4, min(8, int(margin_size * 0.2)))}px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: {max(4, min(8, int(margin_size * 0.2)))}px;
                min-height: {max(20, min(28, int(margin_size * 0.7)))}px;
            }}
        """)
        self.format_combo.setMinimumWidth(max(100, int(margin_size * 3.0)))  # Ensure touch-friendly width
        self.format_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_layout.addWidget(self.format_combo)

        layout.addWidget(file_frame)

        # Buttons with responsive sizing
        btn_frame = QFrame()
        btn_margin = max(5, min(10, int(margin_size * 0.25)))
        btn_frame.setStyleSheet(f"""
            QFrame {{
                background: transparent;
                margin: {btn_margin}px;
            }}
        """)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setSpacing(max(6, min(12, int(margin_size * 0.4))))

        # Responsive button sizing for small screens
        button_width = max(70, min(120, int(margin_size * 4.0)))
        button_height = max(30, min(40, int(margin_size * 1.5)))

        # Browse button with responsive sizing
        browse_btn = QPushButton("Browse")
        btn_padding = max(6, min(10, int(margin_size * 0.3)))
        btn_font_size = max(10, min(13, int(margin_size * 0.5)))
        browse_btn.setMinimumSize(button_width, button_height)
        browse_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                font: bold {btn_font_size}pt Arial;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #17a2b8, stop:0.5 #138496, stop:1 #17a2b8);
                color: white;
                border: 2px solid #17a2b8;
                border-radius: 8px;
                padding: {btn_padding}px;
                min-height: {button_height}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #138496, stop:0.5 #17a2b8, stop:1 #138496);
                border: 2px solid #138496;
            }}
        """)
        browse_btn.clicked.connect(self.browse_ecg_file)
        btn_layout.addWidget(browse_btn)

        # Open button with responsive sizing
        open_btn = QPushButton("Open")
        open_btn.setMinimumSize(button_width, button_height)
        open_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        open_btn.setStyleSheet(f"""
            QPushButton {{
                font: bold {btn_font_size}pt Arial;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:0.5 #45a049, stop:1 #4CAF50);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: {btn_padding}px;
                min-height: {button_height}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:0.5 #4CAF50, stop:1 #45a049);
                border: 2px solid #45a049;
            }}
        """)
        open_btn.clicked.connect(self.open_ecg_file)
        btn_layout.addWidget(open_btn)

        # Cancel button with responsive sizing
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumSize(button_width, button_height)
        cancel_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                font: bold {btn_font_size}pt Arial;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f44336, stop:0.5 #d32f2f, stop:1 #f44336);
                color: white;
                border: 2px solid #f44336;
                border-radius: 8px;
                padding: {btn_padding}px;
                min-height: {button_height}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #d32f2f, stop:0.5 #f44336, stop:1 #d32f2f);
                border: 2px solid #d32f2f;
            }}
        """)
        cancel_btn.clicked.connect(self.hide_sliding_panel)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(btn_frame)

        layout.addStretch(1)
        
        return widget

    def browse_ecg_file(self):
        """Browse for ECG file"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent(),
            "Select ECG File",
            "",
            "ECG Files (*.csv *.txt *.json *.xml *.dcm);;All Files (*.*)"
        )
        if file_path:
            self.file_path_display.setText(file_path)
            self.file_path_display.setStyleSheet(f"""
                QLabel {{
                    font: {max(10, int(getattr(self.sliding_panel, 'margin_size', 20) * 0.5))}pt Arial;
                    color: #2c3e50;
                    background: #e8f5e8;
                    padding: 10px;
                    border: 1px solid #4CAF50;
                    border-radius: 6px;
                    min-height: {max(25, int(getattr(self.sliding_panel, 'margin_size', 20) * 1.2))}px;
                }}
            """)

    def open_ecg_file(self):
        """Open the selected ECG file"""
        file_path = self.file_path_display.text()
        if file_path == self.tr("No file selected") or file_path == "No file selected":
            QMessageBox.warning(self.parent(), self.tr("No File"), self.tr("Please select a file first!"))
            return
        
        try:
            # Here you would implement the actual file opening logic
            QMessageBox.information(self.parent(), "Success", f"ECG file opened successfully!\nFile: {file_path}")
            self.hide_sliding_panel()
        except Exception as e:
            QMessageBox.critical(self.parent(), "Error", f"Failed to open file: {str(e)}")

    # ----------------------------- Working Mode -----------------------------

    def show_working_mode(self):
        self.ensure_sliding_panel_ready()
        content_widget = self.create_working_mode_content()
        self.show_sliding_panel(content_widget, "Working Mode Settings", "Working Mode")

    def create_working_mode_content(self):
        # Get current settings from settings manager
        if not self.settings_manager:
            self.settings_manager = SettingsManager()
        
        # Define sections for working mode
        sections = [
            {
                'title': 'Wave Speed',
                'options': [("12.5mm/s", "12.5"), ("25.0mm/s", "25"), ("50.0mm/s", "50")],
                'setting_key': 'wave_speed'
            },
            {
                'title': 'Wave Gain',
                'options': [("2.5mm/mV", "2.5"), ("5mm/mV", "5"), ("10mm/mV", "10"), ("20mm/mV", "20")],
                'setting_key': 'wave_gain'
            }
        ]
        
        # Define buttons
        buttons = [
            {
                'text': 'OK',
                'action': self.save_working_mode_settings,
                'style': 'primary'
            },
            {
                'text': 'Cancel',
                'action': self.hide_sliding_panel,
                'style': 'danger'
            }
        ]
        
        return self.create_unified_control_panel("Working Mode Settings", sections, buttons)

    def on_setting_changed(self, key, value):
    
        print(f"ECG Menu: Setting {key} changed to {value}")
    
        # Save to settings manager
        self.settings_manager.set_setting(key, value)
        
        if hasattr(self, 'settings_changed_callback') and self.settings_changed_callback:
            print(f"Calling settings callback for {key}={value}")
            self.settings_changed_callback(key, value)
        else:
            print("No settings callback found!")
        
        # Also notify parent ECG test page if available
        if hasattr(self.parent(), 'on_settings_changed'):
            print(f"Calling parent on_settings_changed for {key}={value}")
            self.parent().on_settings_changed(key, value)
        else:
            print("No parent on_settings_changed found!")
        
        if self.dashboard and hasattr(self.dashboard, 'on_settings_changed'):
            try:
                self.dashboard.on_settings_changed(key, value)
            except Exception as e:
                print(f"Dashboard settings callback error: {e}")
        
        # For wave speed and gain, apply immediate visual feedback
        if key in ["wave_speed", "wave_gain"]:
            print(f"Applied {key}: {value}")
            try:
                if hasattr(self.parent(), 'ecg_test_page') and self.parent().ecg_test_page:
                    self.parent().ecg_test_page.update_plots()
            except Exception as e:
                print(f"Immediate refresh error: {e}")
        elif key == "system_language":
            self.update_language(value)

    def save_working_mode_settings(self):
        
        QMessageBox.information(self.parent(), "Saved", "Working mode settings saved and applied to ECG display")
        self.hide_sliding_panel()

    # ----------------------------- Report Setup -----------------------------

    def show_report_setup(self):
        self.ensure_sliding_panel_ready()
        content_widget = self.create_report_setup_content()
        self.show_sliding_panel(content_widget, "Report Setup", "Report Setup")

    def create_report_setup_content(self):

        if not self.settings_manager:
            self.settings_manager = SettingsManager()

        # Define sections — Report Format added at top
        sections = [
            {
                'title': 'Report Format',
                'options': [
                    ("12:1  (Portrait)",   "12_1"),
                    ("6:2  (Landscape)",   "6_2"),
                    ("4:3  (Landscape)",   "4_3"),
                ],
                'setting_key': 'report_format'
            },
            {
                'title': 'Average Wave',
                'options': [("On", "on"), ("Off", "off")],
                'setting_key': 'printer_average_wave'
            },
            {
                'title': 'Lead Sequence',
                'options': [("Standard", "Standard"), ("Cabrera", "Cabrera")],
                'setting_key': 'lead_sequence'
            },
        ]

        buttons = [
            {
                'text': 'Save',
                'action': self.save_report_settings,
                'style': 'primary'
            },
            {
                'text': 'Cancel',
                'action': self.hide_sliding_panel,
                'style': 'danger'
            }
        ]

        return self.create_unified_control_panel("Report Setup", sections, buttons)

    def on_report_setting_changed(self, value):
        print(f"Report setting changed to: {value}")

    def save_report_settings(self):
        from PyQt5.QtWidgets import QMessageBox
        fmt_labels = {"12_1": "12:1 Portrait", "6_2": "6:2 Landscape", "4_3": "4:3 Landscape"}
        try:
            fmt = self.settings_manager.get_setting('report_format', '12_1') if self.settings_manager else '12_1'
            label = fmt_labels.get(fmt, fmt)
            QMessageBox.information(self.parent() or self,
                "Saved", f"Report settings saved!\nFormat: {label}")
        except Exception:
            pass
        # Refresh Generate Report button label on the ECG page
        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                if hasattr(self.ecg_test_page, '_refresh_report_btn_label'):
                    self.ecg_test_page._refresh_report_btn_label()
        except Exception:
            pass
        self.hide_sliding_panel()

    # ----------------------------- Set Filter -----------------------------

    def show_set_filter(self):
        """Show filter settings panel"""
        self.ensure_sliding_panel_ready()
        content_widget = self.create_filter_content()
        self.show_sliding_panel(content_widget, "Filter Settings", "Set Filter")

    def create_filter_content(self):

        # Get current settings from settings manager
        if not self.settings_manager:
            self.settings_manager = SettingsManager()

        # Define sections for filter settings
        sections = [
            {
                'title': 'AC Filter',
                'options': [("off", "off"), ("50Hz", "50"), ("60Hz", "60")],
                'setting_key': 'filter_ac'
            },
            {
                'title': 'EMG Filter',
                'options': [("25Hz", "25"), ("35Hz", "35"), ("40Hz", "40"), ("75Hz", "75"), ("100Hz", "100"), ("150Hz", "150")],
                'setting_key': 'filter_emg'
            },
            {
                'title': 'DFT Filter',
                'options': [("off", "off"), ("0.05Hz", "0.05"), ("0.5Hz", "0.5")],
                'setting_key': 'filter_dft'
            },
        ]
        
        # Define buttons
        buttons = [
            {
                'text': 'OK',
                'action': self.apply_filter_settings,
                'style': 'primary'
            },
            {
                'text': 'Cancel',
                'action': self.hide_sliding_panel,
                'style': 'danger'
            }
        ]
        
        return self.create_unified_control_panel("Filter Settings", sections, buttons)

    def apply_filter_settings(self):
        QMessageBox.information(self.parent(), "Applied", "Filter settings applied successfully!")
        self.hide_sliding_panel()

    def show_system_setup(self):
        self.ensure_sliding_panel_ready()
        content_widget = self.create_system_setup_content()
        self.show_sliding_panel(content_widget, "System Setup", "System Setup")

    def show_load_default(self):
        self.ensure_sliding_panel_ready()
        content_widget = self.create_load_default_content()
        self.show_sliding_panel(content_widget, "Load Default Settings", "Load Default")

    # ----------------------------- System Setup -----------------------------

    def create_system_setup_content(self):

        # Get current settings from settings manager
        if not self.settings_manager:
            self.settings_manager = SettingsManager()

        # Define sections for system setup
        sections = [
            {
                'title': 'BEAT VOL',
                'options': [("On", "on"), ("Off", "off")],
                'setting_key': 'system_beat_vol'
            },
            {
                'title': 'LANGUAGE',
                'options': [("English", "en"), ("Hindi", "hi"), ("Spanish", "es"), ("French", "fr")],
                'setting_key': 'system_language'
            }
        ]
        
        # Define buttons
        buttons = [
            {
                'text': 'Save',
                'action': self.save_system_settings,
                'style': 'primary'
            },
            {
                'text': 'Cancel',
                'action': self.hide_sliding_panel,
                'style': 'danger'
            }
        ]
        
        return self.create_unified_control_panel("System Setup", sections, buttons)

    def save_system_settings(self):
        QMessageBox.information(self.parent(), "Saved", "System settings saved successfully!")
        self.hide_sliding_panel()

    # ----------------------------- Load Default -----------------------------

    def create_load_default_content(self):

        # Get current settings from settings manager
        if not self.settings_manager:
            self.settings_manager = SettingsManager()

        widget = QWidget()
        layout = QVBoxLayout(widget)

        margin_size = getattr(self.sliding_panel, 'margin_size', 20) if self.sliding_panel else 20
        spacing_size = getattr(self.sliding_panel, 'spacing_size', 15) if self.sliding_panel else 15
        layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        layout.setSpacing(spacing_size)

        title_font_size = max(16, min(22, int(margin_size * 0.8)))

        # Top load_default header bar
        load_default_header = QLabel(self.tr("Load Default Settings") if hasattr(self, 'tr') else "Load Default Settings")
        load_default_header.setAlignment(Qt.AlignCenter)
        load_default_header.setStyleSheet(f"""
            QLabel {{
                font: bold {max(14, title_font_size-2)}pt 'Arial';
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                border: 2px solid #343434;
                border-radius: 12px;
                padding: {max(12, margin_size-12)}px;
                margin: {max(4, margin_size-18)}px;
            }}
        """)
        layout.addWidget(load_default_header)

        # Message area replicating hardware UI copy
        message_frame = QFrame()
        message_frame.setStyleSheet("""
            QFrame {
                background: #f5f5f5;
                border: 2px solid #d0d0d0;
                border-radius: 12px;
                padding: 15px 20px;
            }
        """)
        message_layout = QVBoxLayout(message_frame)
        message_layout.setSpacing(8)

        msg_line1 = QLabel(self.tr("Adopt Factory Default Config?"))
        msg_line1.setAlignment(Qt.AlignCenter)
        msg_line1.setStyleSheet("""
            QLabel {
                font: 12pt 'Arial';
                color: #222;
            }
        """)
        message_layout.addWidget(msg_line1)

        msg_line2 = QLabel(self.tr("The previous configure will be lost!"))
        msg_line2.setAlignment(Qt.AlignCenter)
        msg_line2.setStyleSheet("""
            QLabel {
                font: italic 10pt 'Arial';
                color: #444;
            }
        """)
        message_layout.addWidget(msg_line2)

        layout.addWidget(message_frame)

        # Buttons row (No / Yes) styled like screenshot
        btn_row = QHBoxLayout()
        btn_row.setSpacing(max(15, int(margin_size * 0.6)))
        btn_row.addStretch(1)

        no_btn = QPushButton(self.tr("No"))
        no_btn.setFixedSize(100, 36)
        no_btn.setStyleSheet("""
            QPushButton {
                background: #e74c3c;
                color: white;
                border: 2px solid #c0392b;
                border-radius: 6px;
                font: bold 11pt 'Arial';
            }
            QPushButton:hover {
                background: #ff5c4b;
            }
        """)
        no_btn.clicked.connect(self.hide_sliding_panel)
        btn_row.addWidget(no_btn)

        yes_btn = QPushButton(self.tr("Yes"))
        yes_btn.setFixedSize(100, 36)
        yes_btn.setStyleSheet("""
            QPushButton {
                background: #2ecc71;
                color: #fff;
                border: 2px solid #27ae60;
                border-radius: 6px;
                font: bold 11pt 'Arial';
            }
            QPushButton:hover {
                background: #3ddc80;
            }
        """)
        yes_btn.clicked.connect(self.apply_factory_defaults)
        btn_row.addWidget(yes_btn)
        btn_row.addStretch(1)

        layout.addLayout(btn_row)
        return widget

    def apply_factory_defaults(self):
        """Reset every configurable option to startup defaults."""
        if not self.settings_manager:
            self.settings_manager = SettingsManager()

        restored_settings = self.settings_manager.reset_to_defaults()

        # Notify downstream listeners so active widgets update immediately
        if restored_settings:
            plotting_keys = ["wave_speed", "wave_gain", "filter_ac", "filter_emg", "filter_dft"]
            for key, value in restored_settings.items():
                if key in plotting_keys:
                    # Just update the manager silently without broadcasting to UI
                    self.settings_manager.set_setting(key, value)
                    continue
                try:
                    self.on_setting_changed(key, value)
                except Exception as err:
                    print(f" Unable to broadcast default for {key}: {err}")

        QMessageBox.information(
            self.parent(),
            self.tr("Factory Defaults"),
            self.tr("All settings restored to factory defaults.")
        )
        self.hide_sliding_panel()

    # ----------------------------- Version Info (Function call changed and added to dashboard.py) -----------------------------

    def create_version_info_content(self):
        # Create a simple version info display
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Responsive margins and spacing
        margin_size = getattr(self.sliding_panel, 'margin_size', 20) if self.sliding_panel else 20
        spacing_size = getattr(self.sliding_panel, 'spacing_size', 15) if self.sliding_panel else 15
        
        layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        layout.setSpacing(spacing_size)

        # Professional title
        title = QLabel(self.tr("Version Information"))
        title_font_size = max(16, min(22, int(margin_size * 0.8)))
        title.setStyleSheet(f"""
            QLabel {{
                font: bold {title_font_size}pt Arial;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                border: 2px solid #ff6600;
                border-radius: 12px;
                padding: {max(12, int(margin_size * 0.6))}px;
                margin: {max(8, int(margin_size * 0.4))}px;
                min-height: {max(30, int(margin_size * 1.5))}px;
            }}
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Version info container with scroll area for better handling
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #ff6600;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #ff8800;
            }
        """)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(15)

        if not self.settings_manager:
            self.settings_manager = SettingsManager()
        else:
            # Reload settings to ensure we have the latest version from disk
            self.settings_manager.settings = self.settings_manager.load_settings()
        hardware_version = ""
        
        # Determine if physically disconnected at the dashboard level
        is_connected = True
        if hasattr(self, 'dashboard') and self.dashboard:
            is_connected = getattr(self.dashboard, 'device_connected', True)
            
        if is_connected:
            # 1. Try to get active version from current serial connection if streaming
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                if hasattr(self.ecg_test_page, 'serial_reader') and self.ecg_test_page.serial_reader:
                    # Try getting the cached device version
                    if hasattr(self.ecg_test_page.serial_reader, 'device_version') and self.ecg_test_page.serial_reader.device_version:
                        hardware_version = self.ecg_test_page.serial_reader.device_version
    
            # 2. Fallback to settings
            if not hardware_version and self.settings_manager:
                hardware_version = self.settings_manager.get_setting("hardware_version", "")
                
            # 3. Cache it back to settings if we received a valid one
            if hardware_version and hardware_version != "Not Detected" and self.settings_manager:
                current_saved = self.settings_manager.get_setting("hardware_version", "")
                if current_saved != hardware_version:
                    self.settings_manager.set_setting("hardware_version", hardware_version)

        if not hardware_version:
            hardware_version = "Not Detected"

        # Version details
        version_info = [
            (self.tr("Software Version"), "V 1.1.1"),
            (self.tr("Hardware Version"), hardware_version),
            (self.tr("Firmware Version"), "V.3.0.1"),
            (self.tr("Build Date"), "2024-08-26"),
            (self.tr("Manufacturer"), "Modular ECG Systems"),
            (self.tr("Model"), "ECG-121 Pro"),
            (self.tr("Serial Number"), "MF-2024-001"),
            (self.tr("License"), "Professional Edition")
        ]

        # Calculate better minimum widths based on content
        max_label_width = max(len(label) for label, _ in version_info) * 8 + 20  # 8 pixels per char + padding
        max_value_width = max(len(value) for _, value in version_info) * 8 + 20

        for label, value in version_info:
            row = QHBoxLayout()
            row.setSpacing(10)  # Consistent spacing between label and value
            
            # Label with better sizing
            lbl = QLabel(label)
            lbl.setWordWrap(True)
            lbl.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
            lbl.setStyleSheet(f"""
                QLabel {{
                    font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                    color: #2c3e50;
                    background: #f8f9fa;
                    padding: {max(8, int(margin_size * 0.4))}px;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    min-width: {max(max_label_width, 140)}px;
                    max-width: 200px;
                }}
            """)
            
            # Value with better sizing
            val = QLabel(value)
            val.setWordWrap(True)
            val.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
            val.setStyleSheet(f"""
                QLabel {{
                    font: {max(10, int(margin_size * 0.5))}pt Arial;
                    color: #ff6600;
                    background: white;
                    padding: {max(8, int(margin_size * 0.4))}px;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    min-width: {max(max_value_width, 120)}px;
                    max-width: 250px;
                }}
            """)
            
            row.addWidget(lbl, 0, Qt.AlignLeft)
            row.addWidget(val, 0, Qt.AlignLeft)
            row.addStretch()
            info_layout.addLayout(row)

        scroll_area.setWidget(info_frame)
        layout.addWidget(scroll_area)
        
        return widget

    # ----------------------------- Exit -----------------------------

    def show_exit(self):
        """Show exit confirmation dialog"""
        self.ensure_sliding_panel_ready()
        content_widget = self.create_exit_content()
        self.show_sliding_panel(content_widget, self.tr("Exit Application"), "Exit")

    def create_exit_content(self):
        # Create a simple exit confirmation
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Responsive margins and spacing
        margin_size = getattr(self.sliding_panel, 'margin_size', 20) if self.sliding_panel else 20
        spacing_size = getattr(self.sliding_panel, 'spacing_size', 15) if self.sliding_panel else 15
        
        layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        layout.setSpacing(spacing_size)

        # Professional title
        title = QLabel(self.tr("Exit Application"))
        title_font_size = max(16, min(22, int(margin_size * 0.8)))
        title.setStyleSheet(f"""
            QLabel {{
                font: bold {title_font_size}pt Arial;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                border: 2px solid #ff6600;
                border-radius: 12px;
                padding: {max(12, int(margin_size * 0.6))}px;
                margin: {max(8, int(margin_size * 0.4))}px;
                min-height: {max(30, int(margin_size * 1.5))}px;
            }}
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Confirmation message
        msg_frame = QFrame()
        msg_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
            }
        """)
        msg_layout = QVBoxLayout(msg_frame)
        
        confirm_msg = QLabel(self.tr("Do you want to quit?"))
        confirm_msg.setStyleSheet(f"""
            QLabel {{
                font: bold {max(12, int(margin_size * 0.6))}pt Arial;
                color: #2c3e50;
                background: transparent;
                padding: 10px;
                text-align: center;
            }}
        """)
        confirm_msg.setAlignment(Qt.AlignCenter)
        msg_layout.addWidget(confirm_msg)
        
        warning_msg = QLabel(self.tr("Any unsaved data will be lost!"))
        warning_msg.setStyleSheet(f"""
            QLabel {{
                font: {max(10, int(margin_size * 0.5))}pt Arial;
                color: #e74c3c;
                background: #fdf2f2;
                padding: 8px;
                border: 1px solid #f5c6cb;
                border-radius: 6px;
                text-align: center;
            }}
        """)
        warning_msg.setAlignment(Qt.AlignCenter)
        msg_layout.addWidget(warning_msg)
        
        layout.addWidget(msg_frame)

        # Buttons
        btn_frame = QFrame()
        btn_frame.setStyleSheet("""
            QFrame {
                background: transparent;
                margin: 10px;
            }
        """)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setSpacing(max(10, int(margin_size * 0.5)))

        # Responsive button sizing
        button_width = max(100, min(140, int(margin_size * 5)))
        button_height = max(35, min(45, int(margin_size * 1.8)))

        # Cancel button
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setFixedSize(button_width, button_height)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #6c757d, stop:0.5 #495057, stop:1 #6c757d);
                color: white;
                border: 2px solid #6c757d;
                border-radius: 8px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #495057, stop:0.5 #6c757d, stop:1 #495057);
                border: 2px solid #495057;
            }}
        """)
        cancel_btn.clicked.connect(self.hide_sliding_panel)
        btn_layout.addWidget(cancel_btn)

        # Exit button
        exit_btn = QPushButton(self.tr("Exit"))
        exit_btn.setFixedSize(button_width, button_height)
        exit_btn.setStyleSheet(f"""
            QPushButton {{
                font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #f44336, stop:0.5 #d32f2f, stop:1 #f44336);
                color: white;
                border: 2px solid #f44336;
                border-radius: 8px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #d32f2f, stop:0.5 #f44336, stop:1 #d32f2f);
                border: 2px solid #d32f2f;
            }}
        """)
        exit_btn.clicked.connect(self.confirm_exit)
        btn_layout.addWidget(exit_btn)

        layout.addWidget(btn_frame)
        
        return widget

    def confirm_exit(self):
        """Confirm and exit the application"""
        reply = QMessageBox.question(
            self.parent(), 
            self.tr('Exit Application'), 
            self.tr('Are you absolutely sure you want to exit?'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Set flag to indicate application exit
            if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), '_exited_application'):
                self.parent().parent()._exited_application = True
            
            # Close main windows and exit application
            try:
                # Close the dashboard and all its child windows
                if hasattr(self.parent(), 'parent'):
                    dashboard = self.parent().parent()
                    if hasattr(dashboard, 'ecg_test_page') and dashboard.ecg_test_page:
                        dashboard.ecg_test_page.close()
                    dashboard.close()
                
                # Force exit the application
                QApplication.quit()
                sys.exit(0)
            except Exception as e:
                print(f"Error during exit: {e}")
                QApplication.quit()
                sys.exit(0)
        else:
            self.hide_sliding_panel()

    def refresh_ecg_graphs(self):
        """Refresh ECG graphs with latest data"""
        if not hasattr(self, 'ecg_graphs'):
            return
            
        # Get latest ECG data
        ecg_data = {}
        if hasattr(self, 'parent_widget') and self.parent_widget:
            if hasattr(self.parent_widget, 'data'):
                ecg_data = self.parent_widget.data
        else:
            # Try to find ECG data from the current parent
            current_parent = self.parent()
            while current_parent:
                if hasattr(current_parent, 'data') and current_parent.data:
                    ecg_data = current_parent.data
                    break
                current_parent = current_parent.parent()
        
        # Update each graph
        for lead, graph_info in self.ecg_graphs.items():
            if lead in ecg_data and len(ecg_data[lead]) > 0:
                # Get latest data
                data = ecg_data[lead][-500:] if len(ecg_data[lead]) > 500 else ecg_data[lead]
                x = np.arange(len(data))
                
                # Update the line data
                graph_info['line'].set_xdata(x)
                graph_info['line'].set_ydata(data)
                
                # Update canvas if available
                if graph_info['canvas']:
                    graph_info['canvas'].draw_idle()
        
        QMessageBox.information(self.parent(), "Updated", "ECG graphs refreshed with latest data!")

    def create_unified_control_panel(self, title, sections, buttons=None):
        """
        Create a unified, responsive control panel with consistent design
        sections: list of dicts with 'title', 'options', 'variable', 'setting_key'
        buttons: list of dicts with 'text', 'action', 'style' (optional)
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Responsive margins and spacing
        margin_size = getattr(self.sliding_panel, 'margin_size', 20) if self.sliding_panel else 20
        spacing_size = getattr(self.sliding_panel, 'spacing_size', 15) if self.sliding_panel else 15
        
        layout.setContentsMargins(margin_size, margin_size, margin_size, margin_size)
        layout.setSpacing(spacing_size)

        translated_title = self.tr(title)
        # Professional title
        title_label = QLabel(translated_title)
        title_font_size = max(16, min(22, int(margin_size * 0.8)))
        title_label.setStyleSheet(f"""
            QLabel {{
                font: bold {title_font_size}pt Arial;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                border: 2px solid #ff6600;
                border-radius: 12px;
                padding: {max(12, int(margin_size * 0.6))}px;
                margin: {max(8, int(margin_size * 0.4))}px;
                min-height: {max(30, int(margin_size * 1.5))}px;
            }}
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Create scrollable area for content
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Responsive height based on panel size
        panel_width = getattr(self.sliding_panel, 'panel_width', 600) if self.sliding_panel else 600
        panel_height = getattr(self.sliding_panel, 'panel_height', 800) if self.sliding_panel else 800
        is_small_screen = panel_width < 500
        if is_small_screen:
            scroll_area.setMaximumHeight(max(300, min(400, int(panel_height * 0.5))))
        else:
            scroll_area.setMaximumHeight(500)  # Limit height for smaller screens
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        # Responsive spacing for small screens
        if is_small_screen:
            content_layout.setSpacing(max(5, min(10, int(spacing_size * 0.6))))
            content_layout.setContentsMargins(max(3, min(5, int(margin_size * 0.15))), 
                                            max(3, min(5, int(margin_size * 0.15))), 
                                            max(3, min(5, int(margin_size * 0.15))), 
                                            max(3, min(5, int(margin_size * 0.15))))
        else:
            content_layout.setSpacing(spacing_size)
            content_layout.setContentsMargins(5, 5, 5, 5)
        try:
            self.current_panel_keys = [s['setting_key'] for s in sections if 'setting_key' in s]
            if self.settings_manager:
                self.panel_original = {k: self.settings_manager.get_setting(k) for k in self.current_panel_keys}
            else:
                self.panel_original = {}
            self.panel_draft = dict(self.panel_original)
        except Exception as e:
            print(f"Draft init error: {e}")
            self.panel_original = {}
            self.panel_draft = {}

        def add_section(section_data):
            # Check if small screen for responsive styling
            panel_width = getattr(self.sliding_panel, 'panel_width', 600) if self.sliding_panel else 600
            is_small_screen = panel_width < 500
            
            group_box = QGroupBox(self.tr(section_data['title']))
            if is_small_screen:
                group_padding = max(5, min(8, int(margin_size * 0.3)))
                group_margin = max(3, min(5, int(margin_size * 0.15)))
                group_margin_top = max(12, min(15, int(margin_size * 0.7)))
                group_font_size = max(10, min(12, int(margin_size * 0.5)))
                title_font_size = max(9, min(11, int(margin_size * 0.45)))
                title_padding = max(4, min(6, int(margin_size * 0.2)))
                title_left = max(8, min(10, int(margin_size * 0.4)))
            else:
                group_padding = max(8, int(margin_size * 0.4))
                group_margin = max(5, int(margin_size * 0.25))
                group_margin_top = max(18, int(margin_size * 1.0))
                group_font_size = max(12, int(margin_size * 0.6))
                title_font_size = max(11, int(margin_size * 0.55))
                title_padding = 8
                title_left = 12
            
            group_box.setStyleSheet(f"""
                QGroupBox {{
                    font: bold {group_font_size}pt Arial;
                    color: #2c3e50;
                    background: white;
                    border: 2px solid #ff6600;
                    border-radius: {max(6, min(10, int(margin_size * 0.5)))}px;
                    padding: {group_padding}px;
                    margin: {group_margin}px;
                    margin-top: {group_margin_top}px;
                }}
                QGroupBox:title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: {title_left}px;
                    top: 0px;
                    padding: 0 {title_padding}px 0 {title_padding}px;
                    color: #ff6600;
                    font-weight: bold;
                    background: white;
                    font-size: {title_font_size}pt;
                }}
            """)
            
            # Use grid layout for better organization
            grid_layout = QGridLayout(group_box)
            # Responsive spacing and margins based on screen size (is_small_screen already defined above)
            grid_spacing = max(4, min(8, int(margin_size * 0.3))) if is_small_screen else 8
            grid_margin = max(4, min(8, int(margin_size * 0.3))) if is_small_screen else 8
            grid_layout.setSpacing(grid_spacing)
            grid_layout.setContentsMargins(grid_margin, grid_margin, grid_margin, grid_margin)
            
            # Calculate optimal button size - smaller for small screens
            if is_small_screen:
                button_width = max(80, min(120, int(margin_size * 4.0)))
                button_height = max(20, min(28, int(margin_size * 1.0)))
                radio_font_size = max(8, min(10, int(margin_size * 0.4)))
                radio_padding = max(3, min(5, int(margin_size * 0.15)))
                indicator_size = max(8, min(10, int(margin_size * 0.4)))
                indicator_margin_left = max(4, min(5, int(margin_size * 0.2)))
                indicator_margin_right = max(5, min(6, int(margin_size * 0.25)))
            else:
                button_width = max(120, min(180, int(margin_size * 6.0)))
                button_height = max(25, min(35, int(margin_size * 1.3)))
                radio_font_size = max(9, int(margin_size * 0.45))
                radio_padding = max(4, int(margin_size * 0.2))
                indicator_size = max(10, int(margin_size * 0.5))
                indicator_margin_left = 6
                indicator_margin_right = 8
            
            for i, (text, val) in enumerate(section_data['options']):
                # Keep language option labels in English (don't translate them)
                # Check if this is the language section by setting_key or title
                is_language_section = (
                    section_data.get('setting_key') == 'system_language' or 
                    section_data.get('title') == 'LANGUAGE'
                )
                display_text = text if is_language_section else self.tr(text)
                btn = QRadioButton(display_text)
                btn.setStyleSheet(f"""
                    QRadioButton {{
                        font: bold {radio_font_size}pt Arial;
                        color: #2c3e50;
                        background: white;
                        padding: {radio_padding}px;
                        border: 1px solid #e0e0e0;
                        border-radius: {max(4, min(6, int(margin_size * 0.3)))}px;
                        min-width: {button_width}px;
                        max-width: {button_width + 5}px;
                        min-height: {button_height}px;
                    }}
                    QRadioButton:hover {{
                        border: 2px solid #ffb347;
                        background: #fff8f0;
                    }}
                    QRadioButton:checked {{
                        border: 2px solid #ff6600;
                        background: #fff0e0;
                        color: #ff6600;
                        font-weight: bold;
                    }}
                    QRadioButton::indicator {{
                        width: {indicator_size}px;
                        height: {indicator_size}px;
                        border: 2px solid #e0e0e0;
                        border-radius: {max(4, min(5, int(margin_size * 0.25)))}px;
                        background: white;
                        margin-left: {indicator_margin_left}px;
                        margin-right: {indicator_margin_right}px;
                    }}
                    QRadioButton::indicator:checked {{
                        border: 1px solid #ff6600;
                        background: #ff6600;
                    }}
                """)
                # Set size policy for better responsive behavior
                btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                
                # Set checked state if variable exists
                if 'variable' in section_data and section_data['variable']:
                    btn.setChecked(section_data['variable'].get('value') == val)
                elif 'setting_key' in section_data and self.settings_manager:
                    # Load saved setting value
                    saved_value = self.settings_manager.get_setting(section_data['setting_key'])
                    btn.setChecked(saved_value == val)
                
                # Connect to appropriate handler
                if 'setting_key' in section_data:
                    btn.toggled.connect(lambda checked, v=val, key=section_data['setting_key']: 
                                     self.set_draft_setting(key, v) if checked else None)
                elif 'variable' in section_data:
                    btn.toggled.connect(lambda checked, v=val, var=section_data['variable']: 
                                     var.update({'value': v}) if checked else None)
                
                # Arrange in grid (2 columns for better space usage)
                row, col = divmod(i, 2)
                grid_layout.addWidget(btn, row, col)
            
            content_layout.addWidget(group_box)

        # Add all sections
        for section in sections:
            add_section(section)

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # Add buttons if provided
        if buttons:
            btn_frame = QFrame()
            btn_frame.setStyleSheet("""
                QFrame {
                    background: transparent;
                    margin: 10px;
                }
            """)
            btn_layout = QHBoxLayout(btn_frame)
            btn_layout.setSpacing(max(10, int(margin_size * 0.5)))

            # Responsive button sizing
            button_width = max(100, min(140, int(margin_size * 5)))
            button_height = max(35, min(45, int(margin_size * 1.8)))

            for btn_data in buttons:
                btn = QPushButton(self.tr(btn_data['text']))
                btn.setFixedSize(button_width, button_height)
                
                # Default style if not specified
                style = btn_data.get('style', 'primary')
                if style == 'primary':
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #4CAF50, stop:0.5 #45a049, stop:1 #4CAF50);
                            color: white;
                            border: 2px solid #4CAF50;
                            border-radius: 8px;
                            padding: 8px;
                        }}
                        QPushButton:hover {{
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #45a049, stop:0.5 #4CAF50, stop:1 #45a049);
                            border: 2px solid #45a049;
                        }}
                    """)
                elif style == 'danger':
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #f44336, stop:0.5 #d32f2f, stop:1 #f44336);
                            color: white;
                            border: 2px solid #f44336;
                            border-radius: 8px;
                            padding: 8px;
                        }}
                        QPushButton:hover {{
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #d32f2f, stop:0.5 #f44336, stop:1 #d32f2f);
                            border: 2px solid #d32f2f;
                        }}
                    """)
                elif style == 'info':
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            font: bold {max(11, int(margin_size * 0.55))}pt Arial;
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #17a2b8, stop:0.5 #138496, stop:1 #17a2b8);
                            color: white;
                            border: 2px solid #17a2b8;
                            border-radius: 8px;
                            padding: 8px;
                        }}
                        QPushButton:hover {{
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                stop:0 #138496, stop:0.5 #17a2b8, stop:1 #138496);
                            border: 2px solid #138496;
                        }}
                    """)
                
                def _make_handler(action=btn_data['action'], style=style):
                    def _handler():
                        if style == 'primary':
                            try:
                                self.commit_draft_settings()
                            except Exception:
                                pass
                        action()
                    return _handler
                btn.clicked.connect(_make_handler())
                btn_layout.addWidget(btn)

            layout.addWidget(btn_frame)
        
        return widget
    
    def show_dummy(self):
        import threading
        import time
        
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, "dummydata.csv")
        
        leads = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]

        # --- Start dummy data writer in background ---
        
        # --- Create dialog with matplotlib canvas ---
        dialog = QDialog(self.dashboard if self.dashboard else self)
        dialog.setWindowTitle("Live Dummy ECG")
        layout = QVBoxLayout(dialog)
        
        # Create matplotlib figure
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
        axes = axes.flatten()
        
        # Create canvas and add to layout
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Load the existing CSV data once
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep='\t')  # Tab-separated file ke liye
            print(f"Loaded {len(df)} rows of data")
            print(f"Columns: {list(df.columns)}")
        else:
            print(f"CSV file {csv_path} not found!")
            return
        
        # Get the number of samples
        num_samples = len(df)
        window_size = 120  
        current_index = 0
        
        # Set up plots with compact styling
        for idx, lead in enumerate(leads):
            axes[idx].set_ylabel(lead, fontsize=9, fontweight='bold') 
            axes[idx].grid(True, alpha=0.2) 
            axes[idx].set_ylim(df[lead].min() - 50, df[lead].max() + 50)  
            
            # Remove unnecessary elements for cleaner look
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])  
        
        # Compact title
        fig.suptitle('12-Lead ECG Monitor', fontsize=14, fontweight='bold', y=0.95)
        
        def update_plot():
            nonlocal current_index
            
            # Calculate start and end indices for sliding window
            start_idx = max(0, current_index - window_size)
            end_idx = current_index + 1
            
            # Update each lead plot
            for idx, lead in enumerate(leads):
                axes[idx].cla()
                
                # Plot the data in sliding window
                x_data = range(start_idx, end_idx)
                y_data = df[lead].iloc[start_idx:end_idx]
                axes[idx].plot(x_data, y_data, linewidth=0.7, color='#1f77b4') 
                
                # Add current position indicator
                if current_index < len(df):
                    current_value = df[lead].iloc[current_index]
                    axes[idx].plot(current_index, current_value, 'ro', markersize=4, alpha=0.8) 
                    axes[idx].axvline(x=current_index, color='red', linestyle='--', alpha=0.6, linewidth=0.8)  
                
                # Compact styling
                axes[idx].set_ylabel(lead, fontsize=9, fontweight='bold')
                axes[idx].grid(True, alpha=0.2)
                axes[idx].set_ylim(df[lead].min() - 50, df[lead].max() + 50)
                
                # Remove all unnecessary elements
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
                axes[idx].spines['top'].set_visible(False)
                axes[idx].spines['right'].set_visible(False)
            
            # Compact title with sample info
            fig.suptitle(f'ECG Monitor - Sample: {current_index}/{num_samples-1}', 
                         fontsize=12, fontweight='bold', y=0.95)
            
            # Tighter layout - less wasted space
            fig.tight_layout(pad=0.5)
            canvas.draw()
            
            # Move to next sample (loop back to start)
            current_index = (current_index + 1) % num_samples

        # --- Timer for live update ---
        timer = QTimer(dialog)
        timer.timeout.connect(update_plot)
        timer.start(100)  # 10 FPS for smooth animation

        dialog.setMinimumSize(1600, 1000)
        dialog.showMaximized() 
        dialog.exec_()