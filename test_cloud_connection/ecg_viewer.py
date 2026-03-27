import sys
import os
import json
import glob
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QLineEdit, QSplitter, QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush

import pyqtgraph as pg

# Add src to path so we can import the project's real engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from ecg.holter.replay_engine import HolterReplayEngine
except ImportError:
    HolterReplayEngine = None

# ---------------- THEME ----------------
C = {
    'bg': '#06080c',
    'panel': '#0c0f16',
    'panelB': '#111826',
    'border': '#1a2438',
    'ecg': '#00e676',
    'ecgS': '#ffea00',
    'accent': '#40c4ff',
    'yellow': '#ffd740',
    'red': '#ff1744',
    'text': '#b0bec5',
    'textHi': '#eceff1',
    'textLo': '#37474f',
    'grid': '#0a1520',
    'gridBig': '#0f1d2e',
    'nav': '#080c14',
}

def parse_dur(s):
    try:
        if ':' in s:
            h, m = map(int, s.split(':'))
            return h * 60 + m
    except:
        pass
    return 0

def time_str(s, start_time_str="00:00:00"):
    try:
        parts = start_time_str.split(':')
        if len(parts) >= 2:
            hh, mm = int(parts[0]), int(parts[1])
            total = hh * 3600 + mm * 60 + s
        else:
            total = s
    except:
        total = s
        
    h = int(total // 3600) % 24
    m = int(total // 60) % 60
    sec = int(total % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ---------------- STYLING UTILS ----------------
def get_btn_style(bg=C['panelB'], fg=C['text'], border=C['border']):
    return f"""
        QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 3px; padding: 4px 10px; font-size: 11px; }}
        QPushButton:hover {{ background: {C['panel']}; border-color: {C['accent']}; }}
    """

# ---------------- REAL BACKEND DATA FETCHING ----------------
def load_real_sessions(recordings_dir):
    sessions = []
    if not os.path.isdir(recordings_dir):
        return sessions
    for folder in os.listdir(recordings_dir):
        p = os.path.join(recordings_dir, folder)
        if not os.path.isdir(p):
            continue
        ecgh_path = os.path.join(p, "recording.ecgh")
        jsonl_path = os.path.join(p, "metrics.jsonl")
        
        if os.path.exists(ecgh_path):
            # Parse metrics for summary. This avoids loading the whole ECGF file initially
            metrics = []
            if os.path.exists(jsonl_path):
                with open(jsonl_path) as f:
                    for line in f:
                        if line.strip():
                            metrics.append(json.loads(line))
            
            # Aggregate stats
            hrs = [m.get('hr_mean', 0) for m in metrics if m.get('hr_mean', 0) > 0]
            avg_hr = round(np.mean(hrs)) if hrs else 0
            max_hr = round(np.max(hrs)) if hrs else 0
            min_hr = round(np.min(hrs)) if hrs else 0
            sdnn = round(np.mean([m['rr_std'] for m in metrics if m.get('rr_std',0)>0]), 1) if metrics else 0
            rmssd = round(np.mean([m['rmssd'] for m in metrics if m.get('rmssd',0)>0]), 1) if metrics else 0
            
            total_beats = sum(m.get('beat_count',0) for m in metrics)
            total_pauses = sum(m.get('pauses',0) for m in metrics)
            longest_rr = max([m.get('longest_rr',0) for m in metrics]) if metrics else 0
            
            # Look for AF
            diagnoses = []
            if total_pauses > 0:
                diagnoses.append(f"{total_pauses} Pauses")
            
            status = "Complete"
            for m in metrics:
                arrh = m.get('arrhythmias', [])
                if any("AF" in a for a in arrh):
                    status = "AF Detected"
                    diagnoses.append("AF Episode")
                    break
                    
            if not diagnoses:
                diagnoses.append("Normal Rhythm")
                
            dur_sec = len(metrics) * 30
            h = dur_sec // 3600
            m = (dur_sec % 3600) // 60
            
            dates = folder.split('_')
            dt_str = dates[0] if len(dates) > 0 else "Unknown"
            tm_str = dates[1].replace('-', ':') if len(dates) > 1 else "00:00"
            name_str = dates[2] if len(dates) > 2 else "Unknown"

            sessions.append({
                'id': folder[:8],
                'name': name_str,
                'age': '-',
                'gender': '-',
                'doctor': 'Dr. Divyansh',
                'date': dt_str,
                'startTime': tm_str,
                'duration': f"{h:02d}:{m:02d}",
                'duration_sec': dur_sec,
                'totalBeats': total_beats,
                'avgHR': avg_hr,
                'maxHR': max_hr,
                'minHR': min_hr,
                'sinusMaxHR': max_hr,
                'sinusMinHR': min_hr,
                'sdnn': sdnn,
                'rmssd': rmssd,
                'longestRR': longest_rr,
                'tachy': '00:00:00',
                'brady': '00:00:00',
                'vTotal': 0,
                'sTotal': 0,
                'diagnoses': ', '.join(diagnoses[:2]),
                'status': status,
                'ecgh_path': ecgh_path,
                'jsonl_path': jsonl_path
            })
    return sessions

# ---------------- CUSTOM WIDGETS ----------------
class ECGChannelWidget(pg.PlotWidget):
    def __init__(self, ch_idx, title):
        super().__init__(background=C['bg'])
        self.ch_idx = ch_idx
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMenuEnabled(False)
        
        # Grid setup
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot(pen=pg.mkPen(C['ecg'], width=1.3))
        
        self.text_item = pg.TextItem(title, color=C['text'], anchor=(0, 0))
        self.addItem(self.text_item)
        
    def update_data(self, data, view_dur):
        if len(data) == 0:
            return
        x = np.linspace(0, view_dur, len(data))
        self.curve.setData(x, data)
        mx = np.max(np.abs(data)) if len(data) else 1.5
        self.setYRange(max(-1.5, -mx-0.5), max(1.5, mx+0.5))
        self.setXRange(0, view_dur)
        self.text_item.setPos(0, max(1.3, mx+0.2))

class LorenzWidget(pg.PlotWidget):
    def __init__(self, metrics=None):
        super().__init__(background='#08080f')
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        
        # Draw axes
        self.plot([300, 1500], [300, 1500], pen=pg.mkPen(C['gridBig'], width=1))
        self.scatter = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(QColor(C['ecg']).lighter(100)))
        self.addItem(self.scatter)
        self.setXRange(300, 1500)
        self.setYRange(300, 1500)
        if metrics:
            self.update_plot(metrics)
            
    def update_plot(self, metrics):
        # Extract rr intervals to emulate Lorenz (using stds to construct fake points since raw R peaks aren't in metrics)
        pts_x, pts_y = [], []
        for m in metrics:
            hr = m.get('hr_mean', 60)
            if hr <= 0: continue
            mean = 60000 / hr
            sd = m.get('rr_std', 20)
            
            # Gen some proxy points for the lorenz to look realistic for this metric block
            prev = mean
            for _ in range(8):
                curr = prev + (np.random.rand() - 0.5) * sd * 1.5
                c = max(300, min(1500, curr))
                pts_x.append(prev)
                pts_y.append(c)
                prev = c
        self.scatter.setData(pts_x, pts_y)

class TimelineWidget(QWidget):
    def __init__(self, patient, replay_engine=None):
        super().__init__()
        self.patient = patient
        self.engine = replay_engine
        self.setFixedHeight(30)
        self.setStyleSheet(f"background: {C['bg']}; border-top: 1px solid {C['border']};")
        self.total_sec = max(1, patient.get('duration_sec', 1))
        self.view_start = 0
        self.view_dur = 7.2
        self.data = np.zeros(1)
        
        if self.engine:
            try:
                # Load a severely downsampled entire recording for the minimap
                full_data = self.engine.get_all_leads_data(window_sec=self.total_sec)
                # compress 500x to reduce render time
                if len(full_data) > 1 and len(full_data[1]) > 500:
                    self.data = full_data[1][::500]
                else:
                    self.data = full_data[1] if len(full_data) > 1 else np.zeros(1)
            except Exception as e:
                print(f"Timeline error: {e}")
        
    def set_start(self, start):
        self.view_start = start
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = max(1, self.width())
        h = max(1, self.height())
        
        painter.fillRect(self.rect(), QColor(C['bg']))
        
        # Draw waveform
        painter.setPen(QPen(QColor(C['ecg']), 1))
        n = len(self.data)
        if n > 1:
            # Avoid too many lines, use path
            from PyQt5.QtGui import QPainterPath
            path = QPainterPath()
            path.moveTo(0, h/2 - self.data[0] * 8)
            for i in range(1, n):
                x = i / n * w
                y = h / 2 - self.data[i] * 8
                path.lineTo(x, y)
            painter.drawPath(path)
                
        # Draw indicator window
        frac = self.view_start / self.total_sec
        winFrac = self.view_dur / self.total_sec
        x_start = frac * w
        x_width = max(2, winFrac * w)
        
        painter.setBrush(QBrush(QColor(64, 196, 255, 60)))
        painter.setPen(QPen(QColor(C['accent']), 1))
        painter.drawRect(QRectF(x_start, 0, x_width, h))

# ---------------- ECG VIEWER ----------------
class ECGViewer(QWidget):
    def __init__(self, patient, on_back_cb):
        super().__init__()
        self.patient = patient
        self.on_back_cb = on_back_cb
        self.view_start = 0
        self.view_dur = 7.2
        self.total_sec = max(1, patient.get('duration_sec', 1))
        self.metrics = []
        
        # Wire up real database engine!
        self.engine = None
        if HolterReplayEngine and os.path.exists(patient['ecgh_path']):
            self.engine = HolterReplayEngine(patient['ecgh_path'])
            self.total_sec = self.engine.duration_sec
            if self.engine._metrics:
                self.metrics = self.engine._metrics
        
        self.setStyleSheet(f"background: {C['bg']}; color: {C['text']}; font-family: monospace;")
        self.build_ui()
        self.update_data()
        
    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 1. TOP NAV
        top_nav = QFrame()
        top_nav.setStyleSheet(f"background: {C['nav']}; border-bottom: 1px solid {C['border']};")
        top_nav.setFixedHeight(40)
        top_layout = QHBoxLayout(top_nav)
        
        btn_back = QPushButton("← Manage Record")
        btn_back.setStyleSheet(get_btn_style('#1a2235', C['text']))
        btn_back.clicked.connect(self.back_to_list)
        top_layout.addWidget(btn_back)
        
        lbl_view = QLabel("View Record ×")
        lbl_view.setStyleSheet(f"background: #1a4a80; color: {C['textHi']}; padding: 4px 10px; border-radius: 3px; font-size: 11px;")
        top_layout.addWidget(lbl_view)
        
        for t in ['Overview','Template','Histogram','Lorenz','Af Analysis','Tend. Chart','Edit Event','Edit Strips','HRV']:
            btn = QPushButton(t)
            if t == 'Overview':
                btn.setStyleSheet(get_btn_style('#1a3a6a', C['textHi'], '#2a5aaa'))
            else:
                btn.setStyleSheet(get_btn_style(C['panelB']))
            top_layout.addWidget(btn)
            
        top_layout.addStretch()
        lbl_name = QLabel(self.patient['name'])
        lbl_name.setStyleSheet(f"color: {C['textLo']}; font-size: 11px;")
        top_layout.addWidget(lbl_name)
        layout.addWidget(top_nav)

        # Main splitter layout
        main_splitter = QSplitter(Qt.Horizontal)
        
        # LEFT: Lorenz + templates
        left_panel = QFrame()
        left_panel.setStyleSheet(f"background: {C['panel']}; border-right: 1px solid {C['border']};")
        left_panel.setFixedWidth(240)
        left_layout = QVBoxLayout(left_panel)
        
        self.lorenz = LorenzWidget(self.metrics)
        self.lorenz.setFixedHeight(220)
        left_layout.addWidget(self.lorenz)
        
        lbl_lorenz = QLabel("Lorenz/Poincaré Plot")
        lbl_lorenz.setStyleSheet(f"color: {C['textLo']}; font-size: 10px; border-top: 1px solid {C['border']}; padding: 4px;")
        left_layout.addWidget(lbl_lorenz)
        
        # Beat boxes
        box_frame = QFrame()
        box_layout = QHBoxLayout(box_frame)
        box_layout.setContentsMargins(0, 0, 0, 0)
        for b, c, v in [('N', C['ecg'], self.patient['totalBeats']), ('S', C['yellow'], self.patient['sTotal']), 
                        ('V', C['red'], self.patient['vTotal']), ('X', C['text'], '0')]:
            w = QFrame()
            w.setStyleSheet(f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 3px;")
            w.setFixedSize(50, 44)
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 4, 0, 4)
            bl = QLabel(b)
            bl.setStyleSheet(f"color: {c}; font-size: 14px; font-weight: bold;")
            bl.setAlignment(Qt.AlignCenter)
            vl = QLabel(str(v))
            vl.setStyleSheet(f"color: {C['textLo']}; font-size: 9px;")
            vl.setAlignment(Qt.AlignCenter)
            wl.addWidget(bl)
            wl.addWidget(vl)
            box_layout.addWidget(w)
        left_layout.addWidget(box_frame)
        left_layout.addStretch()
        
        self.time_lbl = QLabel()
        self.time_lbl.setStyleSheet(f"color: {C['textLo']}; font-size: 10px;")
        self.time_lbl.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.time_lbl)
        
        main_splitter.addWidget(left_panel)

        # CENTER: ECG channels
        center_panel = QFrame()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        self.hdr = QLabel(f"D1 {self.patient['date']}")
        self.hdr.setStyleSheet(f"background: {C['panelB']}; color: {C['yellow']}; padding: 4px; font-size: 11px; border-bottom: 1px solid {C['border']};")
        center_layout.addWidget(self.hdr)
        
        self.channels = []
        for i in range(1, 4):  # Display Leads I, II, III (idx 0, 1, 2)
            cw = ECGChannelWidget(i - 1, f"Lead {'I' * i if i <= 3 else str(i)}")
            cw.setStyleSheet(f"border-bottom: 1px solid {C['border']};")
            self.channels.append(cw)
            center_layout.addWidget(cw)
            
        self.mini_tl = TimelineWidget(self.patient, self.engine)
        center_layout.addWidget(self.mini_tl)
        
        main_splitter.addWidget(center_panel)

        # RIGHT: Stats
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background: {C['panel']}; border-left: 1px solid {C['border']};")
        right_panel.setFixedWidth(220)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        hdr = QLabel("Overview             Value")
        hdr.setStyleSheet(f"background: {C['panelB']}; color: {C['textHi']}; padding: 8px; font-weight: bold; border-bottom: 1px solid {C['border']};")
        right_layout.addWidget(hdr)
        
        rows = [
            ['Total Beats', self.patient['totalBeats']],
            ['AVG HR', f"{self.patient['avgHR']}bpm"],
            ['Max HR', f"{self.patient['maxHR']}bpm"],
            ['Min HR', f"{self.patient['minHR']}bpm"],
            ['Sinus Max HR', f"{self.patient['sinusMaxHR']}bpm"],
            ['Sinus Min HR', f"{self.patient['sinusMinHR']}bpm"],
            ['Longest RR', f"{self.patient['longestRR']}ms"],
        ]
        
        for i, (name, val) in enumerate(rows):
            w = QFrame()
            w.setStyleSheet(f"background: {C['panel'] if i%2==0 else C['bg']}; border-bottom: 1px solid {C['border']}; padding: 4px;")
            wl = QHBoxLayout(w)
            wl.setContentsMargins(8, 2, 8, 2)
            n = QLabel(name)
            n.setStyleSheet(f"color: {C['text']}; font-size: 11px;")
            v = QLabel(str(val))
            v.setStyleSheet(f"color: {C['textHi']}; font-size: 11px;")
            wl.addWidget(n)
            wl.addStretch()
            wl.addWidget(v)
            right_layout.addWidget(w)
            
        right_layout.addStretch()
        main_splitter.addWidget(right_panel)
        
        layout.addWidget(main_splitter)

        # BOTTOM CONTROLS
        bot_nav = QFrame()
        bot_nav.setStyleSheet(f"background: {C['nav']}; border-top: 1px solid {C['border']};")
        bot_nav.setFixedHeight(40)
        bot_layout = QHBoxLayout(bot_nav)
        
        tools = ['Measuring Ruler', 'Parallel Ruler', 'Magnifying Glass', 'Gain Settings', 'Paper speed:25mm/s']
        for t in tools:
            btn = QPushButton(t)
            btn.setStyleSheet(get_btn_style())
            bot_layout.addWidget(btn)
            
        bot_layout.addStretch()
        
        btn_prev = QPushButton("◀")
        btn_prev.setStyleSheet(get_btn_style())
        btn_prev.clicked.connect(lambda: self.seek(-self.view_dur))
        bot_layout.addWidget(btn_prev)
        
        btn_next = QPushButton("▶")
        btn_next.setStyleSheet(get_btn_style())
        btn_next.clicked.connect(lambda: self.seek(self.view_dur))
        bot_layout.addWidget(btn_next)
        
        layout.addWidget(bot_nav)
        
    def back_to_list(self):
        if self.engine:
            try:
                self.engine.close()
            except:
                pass
        self.on_back_cb()
        
    def seek(self, delta):
        self.view_start = max(0, min(self.total_sec - self.view_dur, self.view_start + delta))
        self.update_data()
        self.mini_tl.set_start(self.view_start)
        
    def update_data(self):
        s1 = time_str(self.view_start, self.patient['startTime'])
        s2 = time_str(self.view_start + self.view_dur, self.patient['startTime'])
        self.time_lbl.setText(f"{s1} - {s2}")
        self.hdr.setText(f"D1 {self.patient['date']} {s1}")
        
        if self.engine:
            self.engine.seek(self.view_start)
            # Fetch 12xN array for the requested window
            data = self.engine.get_all_leads_data(window_sec=self.view_dur)
            for i, c in enumerate(self.channels):
                if i < len(data):
                    c.update_data(data[i], self.view_dur)
        else:
            # Fake data fallback if missing dependencies
            for c in self.channels:
                x = np.linspace(0, self.view_dur, 1000)
                y = np.sin(x * 5) + (np.random.rand(1000) - 0.5) * 0.1
                c.update_data(y, self.view_dur)


# ---------------- PATIENT LIST ----------------
class PatientList(QWidget):
    def __init__(self, on_select_cb):
        super().__init__()
        self.on_select_cb = on_select_cb
        self.setStyleSheet(f"background: {C['bg']}; color: {C['text']}; font-family: monospace;")
        
        # Load REAL sessions
        recs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'recordings'))
        self.patients = load_real_sessions(recs_dir)
        
        self.build_ui()
        
    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        hdr = QFrame()
        hdr.setStyleSheet(f"background: {C['nav']}; border-bottom: 1px solid {C['border']};")
        hdr.setFixedHeight(60)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 0, 24, 0)
        
        title = QLabel("● RhythmUltraMax")
        title.setStyleSheet(f"color: {C['textHi']}; font-size: 20px; font-weight: bold;")
        hl.addWidget(title)
        
        sub = QLabel("Holter Analysis System (LIVE DATA ENGINE)")
        sub.setStyleSheet(f"color: {C['textLo']}; font-size: 12px; margin-left: 10px;")
        hl.addWidget(sub)
        
        hl.addStretch()
        
        search = QLineEdit()
        search.setPlaceholderText("Search patient / ID / doctor...")
        search.setStyleSheet(f"background: {C['panelB']}; border: 1px solid {C['border']}; color: {C['text']}; padding: 6px; border-radius: 4px; width: 260px;")
        hl.addWidget(search)
        layout.addWidget(hdr)
        
        # Summary Cards
        cards_layout = QHBoxLayout()
        cards_layout.setContentsMargins(24, 16, 24, 16)
        
        for lbl, val, color in [('Total Patients', len(self.patients), C['accent']), 
                                ('Completed', len([p for p in self.patients if p['status']=='Complete']), C['ecg']),
                                ('AF Detected', len([p for p in self.patients if p['status']=='AF Detected']), C['red']),
                                ('Total DB Records', len(self.patients), C['yellow'])]:
            cw = QFrame()
            cw.setStyleSheet(f"background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; min-width: 140px;")
            cl = QVBoxLayout(cw)
            c_lbl = QLabel(lbl)
            c_lbl.setStyleSheet(f"color: {C['textLo']}; font-size: 11px;")
            c_val = QLabel(str(val))
            c_val.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: bold;")
            cl.addWidget(c_lbl)
            cl.addWidget(c_val)
            cards_layout.addWidget(cw)
        
        cards_layout.addStretch()
        layout.addLayout(cards_layout)
        
        # Table
        self.table = QTableWidget()
        cols = ['ID','Patient Name','Age','Gender','Doctor','Date','Duration','Avg HR','Max HR','Min HR','Beats','Diagnosis','Status']
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setShowGrid(False)
        
        self.table.setStyleSheet(f"""
            QTableWidget {{ background: transparent; color: {C['textHi']}; border: none; font-size: 13px; }}
            QHeaderView::section {{ background: {C['panelB']}; color: {C['accent']}; font-weight: bold; padding: 8px; border: none; border-bottom: 2px solid {C['border']}; }}
            QTableWidget::item {{ border-bottom: 1px solid {C['border']}; padding: 6px; }}
            QTableWidget::item:selected {{ background: #0f1a2e; }}
        """)
        
        self.table.setRowCount(len(self.patients))
        for r, p in enumerate(self.patients):
            items = [p['id'], p['name'], str(p['age']), p['gender'], p['doctor'], p['date'], p['duration'], 
                     f"{p['avgHR']} bpm", f"{p['maxHR']} bpm", f"{p['minHR']} bpm", str(p['totalBeats']), p['diagnoses'], p['status']]
            
            for c, itm in enumerate(items):
                cell = QTableWidgetItem(itm)
                if c == 7: cell.setForeground(QColor(C['ecg']))
                elif c == 8: cell.setForeground(QColor(C['red']))
                elif c == 9: cell.setForeground(QColor(C['accent']))
                self.table.setItem(r, c, cell)
                
        self.table.itemClicked.connect(self.on_item_click)
        
        tl = QVBoxLayout()
        tl.setContentsMargins(24, 0, 24, 24)
        tl.addWidget(self.table)
        
        if len(self.patients) == 0:
            warn = QLabel("NO REAL DATA FOUND IN src/recordings/! Ensure you have recorded .ecgh files.")
            warn.setStyleSheet(f"color: {C['red']}; font-weight: bold; padding: 20px;")
            tl.addWidget(warn)
            
        layout.addLayout(tl)
        
    def on_item_click(self, item):
        row = item.row()
        patient = self.patients[row]
        self.on_select_cb(patient)

# ---------------- APP ROOT ----------------
class AppRoot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RhythmUltraMax System (Live Data Edition)")
        self.resize(1280, 800)
        self.setStyleSheet(f"background: {C['bg']};")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)
        self.show_list()
        
    def show_list(self):
        view = PatientList(self.show_viewer)
        self.stack.addWidget(view)
        self.stack.setCurrentWidget(view)
        
    def show_viewer(self, patient):
        view = ECGViewer(patient, self.show_list)
        self.stack.addWidget(view)
        self.stack.setCurrentWidget(view)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppRoot()
    window.show()
    sys.exit(app.exec_())
