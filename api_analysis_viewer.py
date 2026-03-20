import sys
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import json
import numpy as np
import requests
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QGridLayout, QLineEdit, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
from scipy.ndimage import gaussian_filter1d

class OnlineAnalysisViewer(QMainWindow):
    def __init__(self, api_data):
        super().__init__()
        self.api_data = api_data
        
        # Parse API metadata
        self.data_dict = self.api_data.get("data", {})
        
        # GUI Settings
        patient_name = self.data_dict.get("name", "Unknown")
        self.setWindowTitle(f"API ECG Analysis - {patient_name}")
        self.resize(1600, 900)
        self.setStyleSheet("background-color: #f4f6f9;")
        
        self.leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.sampling_rate = float(self.data_dict.get("sampling_rate", 500))
        
        # Parse JSON blocks within strings
        self.results = {}
        try:
            self.results = json.loads(self.data_dict.get("result_reading", "{}"))
        except Exception:
            pass
            
        self.conclusion = []
        try:
            self.conclusion = json.loads(self.data_dict.get("conclusion", "[]"))
        except Exception:
            pass
            
        self.arrhythmia = []
        try:
            self.arrhythmia = json.loads(self.data_dict.get("arrhythmia", "[]"))
        except Exception:
            pass
        
        # Build Lead Data structure robustly (capitalization insensitive mapping)
        self.lead_data = np.zeros((12, int(self.data_dict.get("samples", 5000))))
        
        possible_keys = [
            ["lead1_reading", "lead_1_reading", "lead1"],
            ["lead2_reading", "lead_2_reading", "lead2"],
            ["lead3_reading", "lead_3_reading", "lead3"],
            ["leadavr_reading", "lead_avr_reading", "leadavr"],
            ["leadavl_reading", "lead_avl_reading", "leadavl"],
            ["leadavf_reading", "lead_avf_reading", "leadavf"],
            ["leadv1_reading", "lead_v1_reading", "leadV1_reading", "leadv1"],
            ["leadv2_reading", "lead_v2_reading", "leadV2_reading", "leadv2"],
            ["leadv3_reading", "lead_v3_reading", "leadV3_reading", "leadv3"],
            ["leadv4_reading", "lead_v4_reading", "leadV4_reading", "leadv4"],
            ["leadv5_reading", "lead_v5_reading", "leadV5_reading", "leadv5"],
            ["leadv6_reading", "lead_v6_reading", "leadV6_reading", "leadv6"]
        ]
        
        # Helper to find correct key in data_dict (case-insensitive)
        lower_keys = {k.lower(): k for k in self.data_dict.keys()}
        
        for i, key_variants in enumerate(possible_keys):
            found_data = False
            for variant in key_variants:
                actual_key = lower_keys.get(variant.lower())
                if actual_key and actual_key in self.data_dict:
                    val_str = str(self.data_dict[actual_key]).strip()
                    if val_str:
                        # Process string "2003,2001,1998,..."
                        try:
                            # Handle cases where there might be a trailing comma
                            if val_str.endswith(','):
                                val_str = val_str[:-1]
                            arr = np.array([float(x.strip()) for x in val_str.split(',') if x.strip()])
                            
                            # Trim or pad to match the matrix
                            if len(arr) > self.lead_data.shape[1]:
                                arr = arr[:self.lead_data.shape[1]]
                            elif len(arr) < self.lead_data.shape[1]:
                                arr = np.pad(arr, (0, self.lead_data.shape[1] - len(arr)), mode='edge')
                                
                            self.lead_data[i] = arr
                            found_data = True
                            break
                        except Exception as parse_e:
                            print(f"Error parsing array for {actual_key}: {parse_e}")
            if not found_data:
                print(f"Warning: Could not find valid data for Lead {self.leads[i]}!")

        self.init_ui()
        self.plot_all_leads()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Top Header (Patient & Results)
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e6ed;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        # Patient Details
        p_label = QLabel(f"<b>Patient:</b> {self.data_dict.get('name', '')} | <b>Age:</b> {self.data_dict.get('age', '')} | <b>Gender:</b> {self.data_dict.get('gender', '')}")
        p_label.setStyleSheet("font-size: 16px; color: #2c3e50;")
        header_layout.addWidget(p_label)
        
        # Report Details
        r_label = QLabel(f"<b>HR:</b> {self.results.get('HR', '--')} bpm | <b>PR:</b> {self.results.get('PR', '--')} ms | <b>QRS:</b> {self.results.get('QRS', '--')} ms | <b>QT/QTc:</b> {self.results.get('QT', '--')}/{self.results.get('QTc', '--')} ms")
        r_label.setStyleSheet("font-size: 16px; color: #2980b9;")
        header_layout.addWidget(r_label)
        
        # Conclusions
        conc_str = ", ".join(self.conclusion) if self.conclusion else "None"
        arr_str = ", ".join(self.arrhythmia) if self.arrhythmia else "None"
        c_label = QLabel(f"<b>Diagnosis:</b> {conc_str} <br> <b>Arrhythmia:</b> {arr_str}")
        c_label.setStyleSheet("font-size: 14px; color: #c0392b; font-weight: bold;")
        c_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(c_label)

        layout.addWidget(header_frame)

        # 12-Lead Plot Grid
        self.plot_grid = QWidget()
        grid_layout = QGridLayout(self.plot_grid)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)
        
        self.plot_widgets = []
        
        for i in range(12):
            row = i % 6
            col = i // 6
            
            p_widget = pg.PlotWidget()
            p_widget.setBackground("white")
            p_widget.showGrid(x=True, y=True, alpha=0.3)
            p_widget.setMouseEnabled(x=True, y=False)
            
            # Grid styling to match physical ECG paper
            p_widget.getAxis('bottom').setPen(pg.mkPen(color='#e74c3c', width=1))
            p_widget.getAxis('left').setPen(pg.mkPen(color='#e74c3c', width=1))
            
            # No axis labels for cleanliness
            p_widget.getAxis('bottom').setStyle(showValues=False)
            p_widget.getAxis('left').setStyle(showValues=False)

            self.plot_widgets.append(p_widget)
            grid_layout.addWidget(p_widget, row, col)

        layout.addWidget(self.plot_grid, stretch=1)
        self.setCentralWidget(main_widget)

    def plot_all_leads(self):
        time_axis = np.arange(self.lead_data.shape[1]) / self.sampling_rate

        for i in range(12):
            raw = self.lead_data[i]
            
            # Basic smoothing (Gaussian) to clean JSON data if needed
            filtered = gaussian_filter1d(raw, sigma=1.5)
            
            # Center the baseline to zero
            current_mean = np.mean(filtered)
            if np.isnan(current_mean): current_mean = 0
            filtered = filtered - current_mean
            
            # Re-offset to GUI display center
            if self.leads[i] == 'aVR':
                filtered = filtered - 2048
            else:
                filtered = filtered + 2048

            self.plot_widgets[i].clear()
            
            # Custom ECG path pen
            pen = pg.mkPen(color='#2c3e50', width=1.5)
            self.plot_widgets[i].plot(time_axis, filtered, pen=pen, connect='finite')

            # Ensure proper view scale
            self.plot_widgets[i].setYRange(-4096, 4096, padding=0.1) if self.leads[i] == 'aVR' else self.plot_widgets[i].setYRange(0, 4096, padding=0.1)
            if len(time_axis) > 0:
                self.plot_widgets[i].setXRange(time_axis[0], time_axis[-1], padding=0)
            
            # Add Title back due to clear()
            title = pg.TextItem(self.leads[i], color="#2980b9", anchor=(0, 0))
            title.setFont(pg.QtGui.QFont("Arial", 12, pg.QtGui.QFont.Bold))
            self.plot_widgets[i].addItem(title)
            # Position title at the top-left based on scale
            title_y = -1000 if self.leads[i] == 'aVR' else 3800
            title.setPos(0.1, title_y)

class APILoader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG API Loader")
        self.resize(500, 150)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout(self)
        
        title = QLabel("Fetch ECG Report from Deckmount API")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        row = QHBoxLayout()
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Enter Report ID (e.g. 91)")
        self.id_input.setText("91")
        self.id_input.setStyleSheet("padding: 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px;")
        
        btn = QPushButton("Fetch and Plot")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        btn.clicked.connect(self.fetch_data)
        
        row.addWidget(QLabel("Report ID:"))
        row.addWidget(self.id_input)
        row.addWidget(btn)
        layout.addLayout(row)
        
    def fetch_data(self):
        report_id = self.id_input.text().strip()
        if not report_id:
            return
            
        url = f"https://deckmount.in/ankur_bhaiya.php?id={report_id}"
        print(f"Fetching from: {url}")
        
        try:
            # Set a timeout so UI doesn't hang indefinitely
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get("status"):
                    self.viewer = OnlineAnalysisViewer(result)
                    self.viewer.show()
                else:
                    QMessageBox.warning(self, "API Error", "API returned status: false\nMaybe report ID not found.")
            else:
                QMessageBox.warning(self, "HTTP Error", f"Server returned status code: {response.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to fetch data:\n{str(e)}")

def main():
    app = QApplication(sys.argv)
    loader = APILoader()
    loader.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
