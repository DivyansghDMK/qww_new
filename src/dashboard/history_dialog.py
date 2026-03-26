
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt5.QtCore import Qt
import os
from datetime import datetime

class HistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report History")
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.report_types = ["HRV", "Hyperkalemia", "Holter", "Analysis"]
        self.tables = {}

        for report_type in self.report_types:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(["File", "Date", "Size (KB)"])
            table.horizontalHeader().setStretchLastSection(True)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            hh = table.horizontalHeader()
            hh.setSectionResizeMode(0, QHeaderView.Stretch)
            hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            tab_layout.addWidget(table)
            self.tabs.addTab(tab, f"{report_type} Reports")
            self.tables[report_type] = table

        self.load_reports()

    def load_reports(self):
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return

        for report_type in self.report_types:
            self.tables[report_type].setRowCount(0)

        all_reports = []
        for filename in os.listdir(reports_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(reports_dir, filename)
                try:
                    stat = os.stat(file_path)
                    all_reports.append({
                        "path": file_path,
                        "name": filename,
                        "date": datetime.fromtimestamp(stat.st_mtime),
                        "size": stat.st_size
                    })
                except Exception:
                    continue

        all_reports.sort(key=lambda x: x["date"], reverse=True)

        for report in all_reports:
            report_type = self.categorize_report(report["name"])
            if report_type in self.tables:
                table = self.tables[report_type]
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(report["name"]))
                table.setItem(row, 1, QTableWidgetItem(report["date"].strftime("%Y-%m-%d %H:%M:%S")))
                table.setItem(row, 2, QTableWidgetItem(str(int(report["size"] / 1024))))

    def categorize_report(self, filename):
        filename_lower = filename.lower()
        if "hrv" in filename_lower:
            return "HRV"
        elif "hyperkalemia" in filename_lower:
            return "Hyperkalemia"
        elif "holter" in filename_lower:
            return "Holter"
        else:
            return "Analysis"
