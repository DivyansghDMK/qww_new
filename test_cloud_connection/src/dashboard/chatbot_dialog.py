import os
import sys
import json
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

class ChatbotThread(QThread):
    response_ready = pyqtSignal(str)
    def __init__(self, prompt, api_key):
        super().__init__()
        self.prompt = prompt
        self.api_key = api_key
    def run(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            # Prefer stable models. Allow override via CHATBOT_MODEL env.
            env_model = os.getenv("CHATBOT_MODEL", "").strip()
            ordered_full_names = [
                "models/" + env_model if env_model and not env_model.startswith("models/") else env_model,
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
            ]
            ordered_full_names = [m for m in ordered_full_names if m]
            model_name = None
            available_names = set()
            try:
                available = list(genai.list_models())
                for m in available:
                    n = getattr(m, "name", "")
                    if n:
                        available_names.add(n)
                # Prefer free/flash models if available
                flash_candidates = [n for n in available_names if "1.5-flash" in n]
                if flash_candidates:
                    model_name = sorted(flash_candidates)[0]
                # Else use first preferred present
                if not model_name:
                    for cand in ordered_full_names:
                        if cand in available_names:
                            model_name = cand
                            break
                # Else pick any model that supports generateContent
                if not model_name:
                    for m in available:
                        methods = getattr(m, "supported_generation_methods", []) or []
                        if "generateContent" in methods and getattr(m, "name", ""):
                            model_name = m.name
                            break
            except Exception:
                # Fallback to common
                model_name = "models/gemini-1.5-flash"
            if not model_name:
                model_name = "models/gemini-1.5-flash"
            # Always try plain id first (SDK examples use plain id)
            plain_id = model_name.split("/", 1)[-1]
            def build_model(name_or_id):
                return genai.GenerativeModel(name_or_id)
            # Retry once on transient/quota errors with server-provided retry delay
            def _call(active_model):
                return active_model.generate_content(
                    self.prompt,
                    generation_config={
                        "max_output_tokens": 256,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                )
            try:
                # Try plain id first
                model = build_model(plain_id)
                response = _call(model)
            except Exception as e1:
                msg = str(e1)
                # On 404 try full name, then alternate model
                if "404" in msg or "not found" in msg.lower():
                    try:
                        model = build_model(model_name)  # full form
                        response = _call(model)
                    except Exception:
                        # Try alternate between flash/pro ids (plain form)
                        alt_full = (
                            "models/gemini-1.5-pro" if "1.5-flash" in model_name else "models/gemini-1.5-flash"
                        )
                        alt_plain = alt_full.split("/", 1)[-1]
                        try:
                            model = build_model(alt_plain)
                            response = _call(model)
                        except Exception as e2:
                            # Surface available models to help user set CHATBOT_MODEL
                            try:
                                models_str = "\n".join(sorted(available_names)) if available_names else "(list_models unavailable)"
                            except Exception:
                                models_str = "(list_models unavailable)"
                            self.response_ready.emit(
                                "Error: Selected model not available.\n"
                                f"Tried (plain): {plain_id}\nTried (full): {model_name}\n"
                                f"Available on this key:\n{models_str}\n\n"
                                "Set CHATBOT_MODEL to one of the above (use full name from list)."
                            )
                            return
                elif "429" in msg or "quota" in msg.lower():
                    import re, time
                    wait_s = 30
                    m = re.search(r"retry[_ ]delay\s*{\s*seconds:\s*(\d+)", msg)
                    if not m:
                        m = re.search(r"Please retry in\s*([0-9.]+)s", msg)
                    if m:
                        try:
                            wait_s = int(float(m.group(1)))
                        except Exception:
                            wait_s = 30
                    time.sleep(min(wait_s, 60))
                    # After wait, try alternate plain id to avoid same cap
                    alt_full = (
                        "models/gemini-1.5-pro" if "1.5-flash" in model_name else "models/gemini-1.5-flash"
                    )
                    alt_plain = alt_full.split("/", 1)[-1]
                    try:
                        model = build_model(alt_plain)
                        response = _call(model)
                    except Exception as e2:
                        self.response_ready.emit(
                            "Quota exceeded or rate limited.\n"
                            "Try later, shorten prompts, or enable billing.\n"
                            "Docs: https://ai.google.dev/gemini-api/docs/rate-limits\n\n"
                            f"Details: {e2}"
                        )
                        return
                else:
                    self.response_ready.emit(f"Error: {e1}")
                    return
            # Safely extract text
            reply = getattr(response, "text", None)
            if not reply:
                try:
                    lines = []
                    for c in getattr(response, "candidates", []) or []:
                        content = getattr(c, "content", None)
                        parts = getattr(content, "parts", []) if content else []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                lines.append(t)
                    reply = "\n".join(lines) if lines else "(No content returned by model)"
                except Exception:
                    reply = "(No content returned by model)"
            self.response_ready.emit(reply)
        except Exception as e:
            self.response_ready.emit(f"Error: {e}")

class ChatbotDialog(QDialog):
    def __init__(self, parent=None, user_id=None, dashboard_data_func=None):
        super().__init__(parent)
        self.setWindowTitle("AI Health Chatbot")
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
        # Load Gemini API key from environment (.env: CHATBOT_API_KEY)
        self.api_key = os.getenv("CHATBOT_API_KEY", "")
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
        title = QLabel("AI Health Chatbot")
        title.setObjectName("HeaderTitle")
        desc = QLabel("Ask health questions. Get safe, friendly suggestions. Not a diagnosis.")
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
            self.add_message("[Error: Chatbot API key not set. Please set CHATBOT_API_KEY in your .env file.]", sender="AI")
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
