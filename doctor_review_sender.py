"""
Doctor Review Report Sender
A simple utility to send ECG reports for doctor review
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from utils.cloud_uploader import get_cloud_uploader
import json


class DoctorReviewSender(QWidget):
    def __init__(self):
        super().__init__()
        self.uploader = get_cloud_uploader()
        self.selected_pdf = None
        self.selected_ecg_data = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Send Report for Doctor Review")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QWidget {
                background: #f8f9fa;
                font-family: Arial;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QPushButton:pressed {
                background: #e65c00;
            }
            QPushButton:disabled {
                background: #cccccc;
                color: #666666;
            }
            QLabel {
                color: #333;
                font-size: 13px;
            }
            QTextEdit {
                background: white;
                border: 2px solid #ddd;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QGroupBox {
                border: 2px solid #ff6600;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #ff6600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("📤 Send ECG Report for Doctor Review")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #ff6600; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Configuration status
        config_group = QGroupBox("Configuration Status")
        config_layout = QVBoxLayout()
        
        if self.uploader.doctor_review_enabled and self.uploader.doctor_review_api_url:
            status_label = QLabel(f"✅ Doctor Review API: Enabled\n🌐 Endpoint: {self.uploader.doctor_review_api_url}")
            status_label.setStyleSheet("color: green; font-weight: normal;")
        else:
            status_label = QLabel("❌ Doctor Review API: Not configured\nPlease check .env file")
            status_label.setStyleSheet("color: red; font-weight: normal;")
        
        config_layout.addWidget(status_label)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # File selection group
        files_group = QGroupBox("Select Files")
        files_layout = QVBoxLayout()
        
        # PDF selection
        pdf_row = QHBoxLayout()
        self.pdf_label = QLabel("No PDF selected")
        self.pdf_label.setStyleSheet("color: #666; font-weight: normal;")
        pdf_btn = QPushButton("📄 Select PDF Report")
        pdf_btn.clicked.connect(self.select_pdf)
        pdf_row.addWidget(self.pdf_label, 1)
        pdf_row.addWidget(pdf_btn)
        files_layout.addLayout(pdf_row)
        
        # ECG data selection
        ecg_row = QHBoxLayout()
        self.ecg_label = QLabel("No ECG data selected (optional)")
        self.ecg_label.setStyleSheet("color: #666; font-weight: normal;")
        ecg_btn = QPushButton("📊 Select ECG Data JSON")
        ecg_btn.clicked.connect(self.select_ecg_data)
        ecg_row.addWidget(self.ecg_label, 1)
        ecg_row.addWidget(ecg_btn)
        files_layout.addLayout(ecg_row)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        # Patient info group
        patient_group = QGroupBox("Patient Information (Optional)")
        patient_layout = QVBoxLayout()
        
        self.patient_info = QTextEdit()
        self.patient_info.setPlaceholderText(
            "Enter patient information in JSON format:\n"
            "{\n"
            '  "patient_name": "John Doe",\n'
            '  "patient_age": "45",\n'
            '  "patient_gender": "Male"\n'
            "}"
        )
        self.patient_info.setMaximumHeight(120)
        patient_layout.addWidget(self.patient_info)
        
        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)
        
        # Send button
        self.send_btn = QPushButton("📤 Send for Doctor Review")
        self.send_btn.setMinimumHeight(50)
        self.send_btn.setFont(QFont("Arial", 14, QFont.Bold))
        self.send_btn.clicked.connect(self.send_report)
        self.send_btn.setEnabled(False)
        layout.addWidget(self.send_btn)
        
        # Status log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.append("Ready to send reports for doctor review...")
        log_layout.addWidget(self.log)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF Report",
            "reports",
            "PDF Files (*.pdf)"
        )
        if file_path:
            self.selected_pdf = file_path
            self.pdf_label.setText(f"✅ {Path(file_path).name}")
            self.pdf_label.setStyleSheet("color: green; font-weight: normal;")
            self.log.append(f"📄 Selected PDF: {Path(file_path).name}")
            self.update_send_button()
    
    def select_ecg_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ECG Data JSON",
            "reports/ecg_data",
            "JSON Files (*.json)"
        )
        if file_path:
            self.selected_ecg_data = file_path
            self.ecg_label.setText(f"✅ {Path(file_path).name}")
            self.ecg_label.setStyleSheet("color: green; font-weight: normal;")
            self.log.append(f"📊 Selected ECG data: {Path(file_path).name}")
    
    def update_send_button(self):
        # Enable send button if PDF is selected and API is configured
        can_send = (
            self.selected_pdf and
            self.uploader.doctor_review_enabled and
            self.uploader.doctor_review_api_url
        )
        self.send_btn.setEnabled(can_send)
    
    def send_report(self):
        if not self.selected_pdf:
            QMessageBox.warning(self, "Error", "Please select a PDF report first")
            return
        
        # Parse patient info if provided
        patient_data = None
        patient_text = self.patient_info.toPlainText().strip()
        if patient_text:
            try:
                patient_data = json.loads(patient_text)
                self.log.append("✅ Patient information parsed successfully")
            except json.JSONDecodeError as e:
                QMessageBox.warning(
                    self,
                    "Invalid JSON",
                    f"Patient information is not valid JSON:\n{str(e)}"
                )
                return
        
        self.log.append("\n" + "="*50)
        self.log.append("📤 Sending report to doctor review API...")
        self.send_btn.setEnabled(False)
        QApplication.processEvents()
        
        # Send to doctor review
        result = self.uploader.send_to_doctor_review(
            pdf_path=self.selected_pdf,
            patient_data=patient_data,
            ecg_data_file=self.selected_ecg_data
        )
        
        # Display result
        self.log.append(f"\nStatus: {result.get('status')}")
        self.log.append(f"Message: {result.get('message')}")
        
        if result.get('status') == 'success':
            self.log.append("✅ Report sent successfully!")
            QMessageBox.information(
                self,
                "Success",
                "Report successfully sent for doctor review!"
            )
            # Reset selections
            self.selected_pdf = None
            self.selected_ecg_data = None
            self.pdf_label.setText("No PDF selected")
            self.pdf_label.setStyleSheet("color: #666; font-weight: normal;")
            self.ecg_label.setText("No ECG data selected (optional)")
            self.ecg_label.setStyleSheet("color: #666; font-weight: normal;")
            self.patient_info.clear()
        elif result.get('status') == 'queued':
            self.log.append("📥 Report queued for sending when online")
            QMessageBox.information(
                self,
                "Queued",
                "Report queued for doctor review when internet connection is restored"
            )
        else:
            self.log.append(f"❌ Failed to send report")
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to send report:\n{result.get('message')}"
            )
        
        self.log.append("="*50 + "\n")
        self.update_send_button()


def main():
    app = QApplication(sys.argv)
    window = DoctorReviewSender()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
