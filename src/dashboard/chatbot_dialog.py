import os
import sys
import json
import requests
# Ensure .env is loaded before anything else (revert to previous simple loader)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
except ImportError:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#') and '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ.setdefault(k, v)

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QWidget, QFrame, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QFont

# Helper for PyInstaller asset compatibility
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', relative_path)

CHAT_HISTORY_FILE = resource_path('chat_history.json')
ECG_SCOPE_KEYWORDS = {
    "ecg", "ekg", "cardio", "cardiac", "heart", "rhythm", "arrhythmia", "tachycardia",
    "bradycardia", "qt", "qtc", "qrs", "pr interval", "st segment", "t wave", "p wave",
    "lead", "leads", "ventricular", "atrial", "afib", "af", "vt", "svt", "bundle branch",
    "mi", "myocardial", "ischemia", "infarction", "sinus rhythm", "axis", "electrocardiogram",
    "electrocardiograph", "pac", "pvc", "holter", "hyperkalemia"
}

ECG_SYSTEM_PROMPT = (
    "You are an ECG-focused clinical assistant for healthcare professionals. "
    "Only answer questions about ECG interpretation, rhythm analysis, waveform morphology, "
    "intervals, leads, arrhythmias, and closely related cardiology topics. "
    "If asked about non-ECG topics, politely refuse and ask for an ECG-related question instead. "
    "Be concise, clinically careful, and practical. Do not provide a final diagnosis. "
    "When information is incomplete, say so clearly. Suggest escalation for emergency red flags."
)


def _is_ecg_topic(text: str) -> bool:
    msg = (text or "").lower()
    return any(keyword in msg for keyword in ECG_SCOPE_KEYWORDS)


def _ecg_scope_message() -> str:
    return (
        "I can only help with ECG-related questions for healthcare professionals. "
        "Please ask about ECG interpretation, leads, intervals, arrhythmias, or waveform findings."
    )

class ChatbotThread(QThread):
    response_ready = pyqtSignal(str)
    def __init__(self, prompt, api_key):
        super().__init__()
        self.prompt = prompt
        self.api_key = api_key
    def run(self):
        try:
            if not _is_ecg_topic(self.prompt):
                self.response_ready.emit(_ecg_scope_message())
                return

            api_url = os.getenv("CHATBOT_API_URL", "https://api.groq.com/openai/v1/chat/completions").strip()
            model_name = os.getenv("CHATBOT_MODEL", "llama-3.1-8b-instant").strip()
            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "temperature": 0.2,
                    "max_tokens": 400,
                    "messages": [
                        {"role": "system", "content": ECG_SYSTEM_PROMPT},
                        {"role": "user", "content": self.prompt},
                    ],
                },
                timeout=45,
            )
            if response.status_code != 200:
                self.response_ready.emit(f"Error: Chatbot request failed ({response.status_code}): {response.text}")
                return
            data = response.json()
            reply = ""
            choices = data.get("choices") or []
            if choices:
                reply = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
            if not reply:
                reply = "(No content returned by model)"
            self.response_ready.emit(reply)
        except Exception as e:
            self.response_ready.emit(f"Error: {e}")

class ChatbotDialog(QDialog):
    def __init__(self, parent=None, user_id=None, dashboard_data_func=None):
        super().__init__(parent)
        self.setWindowTitle("ECG AI Chatbot")
        self.setMinimumSize(600, 600)
        self.setStyleSheet("""
            QDialog {
                background: #f4f7fa;
                border-radius: 18px;
            }
            QLabel#HeaderTitle {
                color: #2453ff;
                font-size: 22px;
                font-weight: bold;
            }
            QLabel#HeaderDesc {
                color: #888;
                font-size: 13px;
            }
            QListWidget#ChatList {
                background: #f9fbff;
                border: none;
                border-radius: 12px;
                padding: 12px;
            }
            QTextEdit#InputBox {
                background: #fff;
                border: 2px solid #2453ff;
                border-radius: 12px;
                font-size: 15px;
                padding: 10px;
                color: #222;
            }
            QPushButton#SendBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2453ff, stop:1 #ff3380);
                color: white;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 28px;
            }
            QPushButton#SendBtn:disabled {
                background: #ccc;
                color: #fff;
            }
        """)
        self.api_key = (os.getenv("GROQ_API_KEY") or os.getenv("CHATBOT_API_KEY", "")).strip()
        self.user_id = user_id or "default"
        self.dashboard_data_func = dashboard_data_func
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        # Header
        header = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(QIcon(resource_path('assets/vheart2.png')).pixmap(40, 40))
        header.addWidget(icon)
        title_col = QVBoxLayout()
        title = QLabel("ECG AI Chatbot")
        title.setObjectName("HeaderTitle")
        desc = QLabel("Ask ECG-only questions. Built for healthcare professionals. Not a diagnosis.")
        desc.setObjectName("HeaderDesc")
        title_col.addWidget(title)
        title_col.addWidget(desc)
        header.addLayout(title_col)
        header.addStretch(1)
        layout.addLayout(header)
        # Chat area
        self.chat_list = QListWidget()
        self.chat_list.setObjectName("ChatList")
        self.chat_list.setSpacing(10)
        self.chat_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_list.setWordWrap(True)
        layout.addWidget(self.chat_list, 4)
        # History (collapsible)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(100)
        self.history_list.setObjectName("ChatList")
        layout.addWidget(QLabel("Previous Suggestions:"))
        layout.addWidget(self.history_list)
        self.load_history()
        # Input area
        input_row = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setObjectName("InputBox")
        self.input_box.setFixedHeight(48)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("SendBtn")
        self.send_btn.setFixedHeight(48)
        self.send_btn.setMinimumWidth(100)
        self.send_btn.clicked.connect(self.send_message)
        input_row.addWidget(self.input_box, 4)
        input_row.addWidget(self.send_btn, 1)
        layout.addLayout(input_row)
        self.setLayout(layout)
        self.history_list.itemClicked.connect(self.show_history_item)
        if not self.api_key:
            self.add_message("[Error: Chatbot API key not set. Please set GROQ_API_KEY in your .env file.]", sender="AI")
            self.send_btn.setEnabled(False)
    def add_message(self, text, sender="user"):
        item = QListWidgetItem()
        bubble = QWidget()
        bubble_layout = QHBoxLayout(bubble)
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setFont(QFont("Segoe UI", 13))
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        label.setMaximumWidth(self._message_max_width())
        if sender == "user":
            label.setStyleSheet("background: #2453ff; color: white; border-radius: 14px; padding: 10px 16px; margin: 2px 0 2px 40px;")
            bubble_layout.addStretch(1)
            bubble_layout.addWidget(label, 0, Qt.AlignRight)
        else:
            label.setStyleSheet("background: #fff; color: #222; border: 2px solid #ff3380; border-radius: 14px; padding: 10px 16px; margin: 2px 40px 2px 0;")
            bubble_layout.addWidget(label, 0, Qt.AlignLeft)
            bubble_layout.addStretch(1)
        bubble.setLayout(bubble_layout)
        item.setSizeHint(bubble.sizeHint())
        self.chat_list.addItem(item)
        self.chat_list.setItemWidget(item, bubble)
        self.chat_list.scrollToBottom()

    def _message_max_width(self):
        viewport_width = self.chat_list.viewport().width() if hasattr(self, "chat_list") else self.width()
        return max(260, viewport_width - 80)

    def _rewrap_messages(self):
        for i in range(self.chat_list.count()):
            item = self.chat_list.item(i)
            widget = self.chat_list.itemWidget(item)
            if not widget:
                continue
            labels = widget.findChildren(QLabel)
            if not labels:
                continue
            labels[0].setMaximumWidth(self._message_max_width())
            item.setSizeHint(widget.sizeHint())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rewrap_messages()
    def load_history(self):
        self.history = []
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    all_hist = json.load(f)
                    self.history = all_hist.get(self.user_id, [])
            except Exception:
                self.history = []
        self.history_list.clear()
        for item in self.history:
            lw_item = QListWidgetItem(item['question'][:60] + ("..." if len(item['question']) > 60 else ""))
            lw_item.setData(1000, item)
            self.history_list.addItem(lw_item)
    def save_history(self):
        all_hist = {}
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    all_hist = json.load(f)
            except Exception:
                all_hist = {}
        all_hist[self.user_id] = self.history
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(all_hist, f, indent=2)
    def send_message(self):
        user_msg = self.input_box.toPlainText().strip()
        if not user_msg:
            return
        if not _is_ecg_topic(user_msg):
            self.add_message(user_msg, sender="user")
            self.input_box.clear()
            self.add_message(_ecg_scope_message(), sender="AI")
            return
        dashboard_info = ""
        if self.dashboard_data_func:
            dashboard_info = self.dashboard_data_func()
        full_prompt = user_msg + ("\n\nDashboard Data:\n" + dashboard_info if dashboard_info else "")
        self.add_message(user_msg, sender="user")
        self.input_box.clear()
        self.send_btn.setEnabled(False)
        self.thread = ChatbotThread(full_prompt, self.api_key)
        self.thread.response_ready.connect(self.display_response)
        self.thread.start()
        self._pending_question = user_msg
    def display_response(self, reply):
        self.add_message(reply, sender="AI")
        self.history.append({'question': self._pending_question, 'answer': reply})
        self.save_history()
        self.load_history()
        self.send_btn.setEnabled(True)
    def show_history_item(self, item):
        data = item.data(1000)
        if data:
            self.chat_list.clear()
            self.add_message(data['question'], sender="user")
            self.add_message(data['answer'], sender="AI")
