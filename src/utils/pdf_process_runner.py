from __future__ import annotations

import multiprocessing
import os
import sys
import time
import traceback
from typing import Callable, Optional


_MP_CTX = multiprocessing.get_context("spawn")


def _child_ecg_report(queue: multiprocessing.Queue, kwargs: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")

        src_dir = kwargs.pop("_src_dir", None)
        if src_dir and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        demo_mode = bool(kwargs.pop("demo_mode", False))
        if demo_mode:
            from ecg.demo_ecg_report_generator import generate_demo_ecg_report

            generate_demo_ecg_report(**kwargs)
        else:
            from ecg.ecg_report_generator import generate_ecg_report

            generate_ecg_report(**kwargs)

        queue.put(("ok", kwargs.get("filename", "report.pdf")))
    except Exception as exc:
        queue.put(("err", f"{exc}\n{traceback.format_exc()}"))


def _child_hrv_report(queue: multiprocessing.Queue, kwargs: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")

        src_dir = kwargs.pop("_src_dir", None)
        if src_dir and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        settings_dict = kwargs.pop("settings_dict", {})

        class _DictSettingsManager:
            def __init__(self, data: dict):
                self._data = data

            def get_setting(self, key, default=None):
                return self._data.get(key, default)

            def get_wave_speed(self):
                return float(self._data.get("wave_speed", 25.0))

            def get_wave_gain(self):
                return float(self._data.get("wave_gain", 10.0))

            def get_serial_port(self):
                return self._data.get("serial_port", "")

        kwargs["settings_manager"] = _DictSettingsManager(settings_dict)

        from ecg.hrv_ecg_report_generator import generate_hrv_ecg_report

        generate_hrv_ecg_report(**kwargs)
        queue.put(("ok", kwargs.get("filename", "hrv_report.pdf")))
    except Exception as exc:
        queue.put(("err", f"{exc}\n{traceback.format_exc()}"))


class PDFProcessRunner:
    POLL_MS = 200

    def __init__(self, parent_widget=None, poll_interval_ms: int = POLL_MS):
        self._parent = parent_widget
        self._poll_ms = poll_interval_ms
        self._process: Optional[multiprocessing.Process] = None
        self._queue: Optional[multiprocessing.Queue] = None
        self._poll_timer = None
        self._on_success: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None
        self._start_ts = 0.0

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start_ecg_report(
        self,
        filename: str,
        frozen_data: dict,
        patient: dict,
        sampling_rate: float = 500.0,
        on_success: Optional[Callable[[str], None]] = None,
        on_failure: Optional[Callable[[str], None]] = None,
    ) -> bool:
        if self.is_running():
            return False

        self._on_success = on_success
        self._on_failure = on_failure

        kwargs = dict(
            filename=filename,
            data=frozen_data.get("data", {}),
            lead_images=frozen_data.get("lead_images"),
            dashboard_instance=None,
            ecg_test_page=None,
            patient=patient,
            ecg_data_file=frozen_data.get("ecg_data_file"),
            log_history=bool(frozen_data.get("log_history", False)),
            username=frozen_data.get("username", ""),
            demo_mode=bool(frozen_data.get("demo_mode", False)),
            _src_dir=self._find_src_dir(),
        )
        return self._spawn(_child_ecg_report, kwargs)

    def start_hrv_report(
        self,
        filename: str,
        captured_data: list,
        data: dict,
        patient: dict,
        settings_manager=None,
        selected_lead: str = "II",
        on_success: Optional[Callable[[str], None]] = None,
        on_failure: Optional[Callable[[str], None]] = None,
    ) -> bool:
        if self.is_running():
            return False

        self._on_success = on_success
        self._on_failure = on_failure

        kwargs = dict(
            filename=filename,
            captured_data=captured_data,
            data=data,
            patient=patient,
            settings_dict=self._serialize_settings(settings_manager),
            selected_lead=selected_lead,
            ecg_test_page=None,
            _src_dir=self._find_src_dir(),
        )
        return self._spawn(_child_hrv_report, kwargs)

    def cancel(self) -> None:
        if self._poll_timer is not None:
            try:
                self._poll_timer.stop()
            except Exception:
                pass
            self._poll_timer = None

        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
        self._process = None
        self._queue = None

    def _spawn(self, target_fn, kwargs: dict) -> bool:
        try:
            self._queue = _MP_CTX.Queue()
            self._process = _MP_CTX.Process(
                target=target_fn,
                args=(self._queue, kwargs),
                daemon=True,
            )
            self._process.start()
            self._start_ts = time.perf_counter()
            self._start_poll_timer()
            return True
        except Exception as exc:
            self._cleanup()
            if self._on_failure:
                self._on_failure(str(exc))
            return False

    def _start_poll_timer(self) -> None:
        try:
            from PyQt5.QtCore import QTimer
        except ImportError:
            from PyQt6.QtCore import QTimer

        self._poll_timer = QTimer(self._parent)
        self._poll_timer.setInterval(self._poll_ms)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

    def _poll(self) -> None:
        if self._queue is None:
            return

        try:
            status, payload = self._queue.get_nowait()
        except Exception:
            if self._process is not None and not self._process.is_alive():
                err = f"Child process exited unexpectedly (exit code {self._process.exitcode})"
                self._cleanup()
                if self._on_failure:
                    self._on_failure(err)
            return

        self._cleanup()
        if status == "ok":
            if self._on_success:
                self._on_success(payload)
        else:
            if self._on_failure:
                self._on_failure(payload)

    def _cleanup(self) -> None:
        if self._poll_timer is not None:
            try:
                self._poll_timer.stop()
            except Exception:
                pass
            self._poll_timer = None

        if self._process is not None:
            try:
                self._process.join(timeout=1)
            except Exception:
                pass
            self._process = None

        self._queue = None

    @staticmethod
    def _find_src_dir() -> str:
        this_file = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(this_file))

    @staticmethod
    def _serialize_settings(sm) -> dict:
        if sm is None:
            return {}
        if isinstance(sm, dict):
            return sm

        keys = [
            "wave_speed",
            "wave_gain",
            "serial_port",
            "baud_rate",
            "filter_ac",
            "filter_emg",
            "filter_dft",
            "report_window_seconds",
            "patient_name",
        ]
        data = {}
        for key in keys:
            try:
                data[key] = sm.get_setting(key, None)
            except Exception:
                pass

        for method_name, key in (
            ("get_wave_speed", "wave_speed"),
            ("get_wave_gain", "wave_gain"),
            ("get_serial_port", "serial_port"),
        ):
            try:
                data[key] = getattr(sm, method_name)()
            except Exception:
                pass

        return data
