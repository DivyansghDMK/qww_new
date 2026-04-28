import sys
import os
import json
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QGridLayout)
from PyQt5.QtCore import Qt
from scipy.ndimage import gaussian_filter1d

class OnlineAnalysisViewer(QMainWindow):
    def __init__(self, json_data):
        super().__init__()
        self.json_data = json_data
        self.setWindowTitle(f"Online ECG Analysis - {json_data['patient_details']['name']}")
        self.resize(1600, 900)
        self.setStyleSheet("background-color: #f4f6f9;")
        
        self.leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.sampling_rate = self.json_data["data_details"].get("sampling_rate", 500)
        
        # Parse Device Data into 12 numpy arrays
        raw_data = np.array(self.json_data["ecg_data"]["device_data"], dtype=float)
        if raw_data.shape[1] == 12:
            self.lead_data = raw_data.T  # Transpose to shape (12, N_samples)
        else:
            self.lead_data = raw_data
            
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
        patient = self.json_data["patient_details"]
        p_label = QLabel(f"<b>Patient:</b> {patient['name']} | <b>Age:</b> {patient['age']} | <b>Gender:</b> {patient['gender']}")
        p_label.setStyleSheet("font-size: 16px; color: #2c3e50;")
        header_layout.addWidget(p_label)
        
        # Report Details
        results = self.json_data["result_reading"]
        r_label = QLabel(f"<b>HR:</b> {results['HR']} bpm | <b>PR:</b> {results['PR']} ms | <b>QRS:</b> {results['QRS']} ms | <b>QT/QTc:</b> {results['QT']}/{results['QTc']} ms")
        r_label.setStyleSheet("font-size: 16px; color: #2980b9;")
        header_layout.addWidget(r_label)
        
        # Conclusions
        conclusions = ", ".join(self.json_data["conclusion"])
        c_label = QLabel(f"<b>Diagnosis:</b> {conclusions} <br> <b>Arrhythmia:</b> {', '.join(self.json_data['arrhythmia'])}")
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

            # Lead title
            title = pg.TextItem(self.leads[i], color="#2980b9", anchor=(0, 0))
            title.setFont(pg.QtGui.QFont("Arial", 12, pg.QtGui.QFont.Bold))
            p_widget.addItem(title)
            title.setPos(0, 3000) # approximate Y position, adjusted later

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
            filtered = filtered - np.mean(filtered)
            
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
            self.plot_widgets[i].setXRange(time_axis[0], time_axis[-1], padding=0)
            
            # Add Title back due to clear()
            title = pg.TextItem(self.leads[i], color="#2980b9", anchor=(0, 0))
            title.setFont(pg.QtGui.QFont("Arial", 12, pg.QtGui.QFont.Bold))
            self.plot_widgets[i].addItem(title)
            # Position title at the top-left based on scale
            title_y = -1000 if self.leads[i] == 'aVR' else 3800
            title.setPos(0.1, title_y)

def load_data_and_run(txt_filepath):
    # Parse the text file to form the JSON
    parsed_json = None
    try:
        with open(txt_filepath, 'r') as f:
            lines = f.readlines()
            
        data_line = None
        for line in lines:
            if "|[" in line or "]|" in line:
                data_line = line.strip()
                break
                
        if data_line:
            # Convert custom pipe format to standard JSON 2D Array
            json_array_str = "[" + data_line.replace("|", ",") + "]"
            device_data = json.loads(json_array_str)
            
            # Scaffold the full JSON response exactly as User requested
            parsed_json = {
              "patient_details": {
                "user_id": "1",
                "name": "Ramesh",
                "age": 21,
                "gender": "Male",
                "mobile_no": "6388399399",
                "report_id": "771589418404941",
                "report_date": "2026-02-20 05:40 PM"
              },
              "device_details": {
                "device_name": "RhythmPro ECG",
                "manufacturer": "Deckmount Electronics Pvt Ltd",
                "app_version": "4.10.1",
                "report_version": "1.0.18"
              },
              "data_details": {
                "format": "int16_le",
                "leads": 12,
                "sampling_rate": 500,
                "samples": len(device_data)
              },
              "ecg_settings": {
                "paper_speed": "12.5mm/s",
                "gain": "10mm/mV",
                "filter_band": "0.5-25Hz",
                "ac_filter": "50Hz"
              },
              "ecg_data": {
                "device_data": device_data
              },
              "result_reading": {
                "QT": 486,
                "QTc": 637,
                "QRS": 186,
                "HR": 300,
                "RR": 580,
                "T": 486,
                "PR": 120,
                "ST": 300,
                "P_QRS_T_axis": "41°/46°/49°",
                "RV5_SV1": "3.647 mV/-3.572 mV",
                "RV5_plus_SV1": "7.219 mV"
              },
              "conclusion": [
                "IRREGULAR_RHYTHM",
                "ST_Elevation",
                "Tachycardia"
              ],
              "arrhythmia": [
                "AtrialFibrillation"
              ]
            }
            
    except Exception as e:
        print(f"Error parsing data: {e}")
        return
        
def main():
    app = QApplication(sys.argv)
    target_file = r"C:\Users\DELL\Documents\QW\qww\RhythmUltraMax (1).txt"
    load_data_and_run(target_file)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
