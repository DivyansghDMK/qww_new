import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class SessionRecorder:
    """Append-only JSONL recorder for per-user ECG sessions.

    Writes one JSON object per call to record(), containing:
      - timestamp
      - username and user metadata
      - live metrics
      - ECG snapshot (last N samples per lead)
      - optional events (e.g., arrhythmia detections)
    """

    def __init__(self, username: str, user_record: Optional[Dict[str, Any]] = None, base_dir: Optional[str] = None):
        self.username = username or "unknown"
        self.user_record = user_record or {}
        # Default under project reports/sessions
        if base_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.abspath(os.path.join(here, '..', '..', 'reports', 'sessions'))
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        # Create a new JSONL file per app launch for the user
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_user = ''.join(c for c in self.username if c.isalnum() or c in ('-', '_')) or 'user'
        self.file_path = os.path.join(self.base_dir, f"session_{safe_user}_{ts}.jsonl")
        # Buffered handle
        self._fh = open(self.file_path, 'a', encoding='utf-8')

    def close(self):
        try:
            if self._fh and not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

    def record(self, metrics: Dict[str, Any], ecg_snapshot: Dict[str, List[float]], events: Optional[Dict[str, Any]] = None):
        """Append one entry to the JSONL file."""
        try:
            payload = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'username': self.username,
                'user': self.user_record,
                'metrics': metrics or {},
                'ecg_snapshot': ecg_snapshot or {},
                'events': events or {},
            }
            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            # Light flush to make tail -f friendly without high IO
            self._fh.flush()
        except Exception:
            # Silent; recording should never break UI
            pass

    @staticmethod
    def snapshot_from_ecg_page(ecg_test_page, seconds: float = 5.0) -> Dict[str, List[float]]:
        """Build a dict of last N seconds per lead from ecg_test_page buffers.

        Returns: { 'I': [...], 'II': [...], ... }
        """
        try:
            leads = getattr(ecg_test_page, 'leads', []) or []
            data = getattr(ecg_test_page, 'data', []) or []
            if not leads or not data:
                return {}
            fs = 80.0
            try:
                if hasattr(ecg_test_page, 'sampler') and getattr(ecg_test_page.sampler, 'sampling_rate', None):
                    fs = float(ecg_test_page.sampler.sampling_rate)
                    if fs <= 0 or fs > 2000:
                        fs = 80.0
            except Exception:
                fs = 80.0
            window = max(1, int(seconds * fs))
            out: Dict[str, List[float]] = {}
            for i, lead_name in enumerate(leads):
                if i >= len(data):
                    continue
                buf = data[i]
                try:
                    if buf is None or len(buf) == 0:
                        continue
                    slice_data = buf[-window:]
                    # Ensure plain floats for JSON
                    out[str(lead_name)] = [float(x) for x in slice_data]
                except Exception:
                    continue
            return out
        except Exception:
            return {}


