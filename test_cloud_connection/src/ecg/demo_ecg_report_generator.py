from datetime import datetime
from typing import Dict, Optional
import os
import sys
import numpy as np
import pandas as pd

from ecg.ecg_report_generator import generate_ecg_report


def _default_demo_metrics() -> Dict[str, float]:
    return {
        "HR": 60.0,
        "beat": 60.0,
        "PR": 167.0,
        "QRS": 86.0,
        "QT": 357.0,
        "QTc": 357.0,
        "QTcF": 357.0,
        "ST": 0.0,
        "HR_max": 60.0,
        "HR_min": 60.0,
        "HR_avg": 60.0,
    }


def build_demo_patient() -> Dict[str, str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "first_name": "DEMO",
        "last_name": "",
        "age": "",
        "gender": "",
        "date_time": now,
    }


def _populate_demo_signals_from_dummycsv(ecg_test_page: Optional[object]) -> None:
    if ecg_test_page is None:
        return
    try:
        ecg_dir = os.path.dirname(__file__)
        src_dir = os.path.abspath(os.path.join(ecg_dir, ".."))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))

        if getattr(sys, "frozen", False):
            bundle_dir = getattr(sys, "_MEIPASS", None)
            candidates = []
            if bundle_dir:
                candidates.append(os.path.join(bundle_dir, "dummycsv.csv"))
                candidates.append(os.path.join(bundle_dir, "_internal", "dummycsv.csv"))
                candidates.append(os.path.join(os.path.dirname(sys.executable), "dummycsv.csv"))
            candidates.extend(
                [
                    os.path.join(ecg_dir, "dummycsv.csv"),
                    os.path.join(project_root, "dummycsv.csv"),
                    os.path.abspath("dummycsv.csv"),
                ]
            )
        else:
            candidates = [
                os.path.join(ecg_dir, "dummycsv.csv"),
                os.path.join(project_root, "dummycsv.csv"),
                os.path.abspath("dummycsv.csv"),
            ]

        csv_path = None
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break
        if not csv_path:
            print("Demo report: dummycsv.csv not found")
            return

        df = pd.read_csv(csv_path)
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep="\t")

        total_samples = len(df)
        if total_samples <= 0:
            print("Demo report: dummycsv.csv is empty")
            return

        target_fs = 500.0
        target_seconds = 6.0
        target_samples = int(target_fs * target_seconds)

        lead_names = getattr(ecg_test_page, "leads", [])
        data_list = getattr(ecg_test_page, "data", [])
        buffer_size = getattr(ecg_test_page, "buffer_size", target_samples)
        if buffer_size < target_samples:
            buffer_size = target_samples
            try:
                ecg_test_page.buffer_size = buffer_size
            except Exception:
                pass

        for lead in lead_names:
            col_name = "aVR" if lead == "-aVR" else lead
            if col_name in df.columns:
                src_full = df[col_name].to_numpy(dtype=float)
                if src_full.size == 0:
                    continue
                if src_full.size >= target_samples:
                    resampled = src_full[-target_samples:]
                else:
                    reps = int(np.ceil(target_samples / float(src_full.size)))
                    tiled = np.tile(src_full, reps)
                    resampled = tiled[:target_samples]
                if lead == "-aVR":
                    resampled = -resampled
                try:
                    idx = lead_names.index(lead)
                except ValueError:
                    continue
                if idx < len(data_list):
                    data_list[idx] = resampled

        ecg_test_page.data = data_list

        try:
            demo_manager = getattr(ecg_test_page, "demo_manager", None)
            if demo_manager is not None:
                demo_manager.time_window = target_seconds
                demo_manager.samples_per_second = target_fs
                print(
                    f"Demo report: forced demo_manager window={demo_manager.time_window:.2f}s, "
                    f"fs={demo_manager.samples_per_second}Hz"
                )
        except Exception as e:
            print(f"Demo report: could not update demo_manager timing: {e}")

        print(f"Demo report: loaded resampled {target_samples} samples per lead from dummycsv.csv")
        print(f"Demo report: sampling_rate={target_fs}Hz, window≈{target_seconds:.2f}s")
    except Exception as e:
        print(f"Demo report: error loading dummycsv.csv: {e}")


def generate_demo_ecg_report(
    filename: str,
    lead_images: Dict[str, str],
    dashboard_instance: Optional[object] = None,
    ecg_test_page: Optional[object] = None,
    fmt: str = "12_1",
) -> None:
    backup_data = None
    backup_buffer_size = None
    backup_time_window = None
    backup_fs = None
    try:
        if ecg_test_page is not None:
            try:
                if hasattr(ecg_test_page, "data"):
                    backup_data = [np.array(ch, copy=True) for ch in getattr(ecg_test_page, "data", [])]
                if hasattr(ecg_test_page, "buffer_size"):
                    backup_buffer_size = ecg_test_page.buffer_size
                demo_manager = getattr(ecg_test_page, "demo_manager", None)
                if demo_manager is not None:
                    backup_time_window = getattr(demo_manager, "time_window", None)
                    backup_fs = getattr(demo_manager, "samples_per_second", None)
            except Exception:
                backup_data = None
                backup_buffer_size = None
                backup_time_window = None
                backup_fs = None

        _populate_demo_signals_from_dummycsv(ecg_test_page)
        data = _default_demo_metrics()
        patient = build_demo_patient()
        username = getattr(dashboard_instance, "username", None)

        fmt = (fmt or "12_1").strip()

        if fmt in ("4_3", "6_2"):
            import importlib.util

            module_filename = "4_3_ecg_report_generator.py" if fmt == "4_3" else "6_2_ecg_report_generator.py"
            generator_names = (
                ("generate_4_3_ecg_report", "generate_ecg_report")
                if fmt == "4_3"
                else ("generate_6_2_ecg_report", "generate_ecg_report")
            )

            ecg_dir = os.path.dirname(__file__)
            module_file = os.path.join(ecg_dir, module_filename)

            if not os.path.exists(module_file):
                raise FileNotFoundError(f"{module_filename} not found for demo report")

            spec = importlib.util.spec_from_file_location("ecg_demo_format_generator", module_file)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            gen = None
            for name in generator_names:
                if hasattr(mod, name):
                    gen = getattr(mod, name)
                    break

            if gen is None:
                raise RuntimeError(f"No generator function found in {module_filename}")

            gen(
                filename,
                data,
                lead_images,
                dashboard_instance,
                ecg_test_page,
                patient,
                None,
            )
        else:
            generate_ecg_report(
                filename=filename,
                data=data,
                lead_images=lead_images,
                dashboard_instance=dashboard_instance,
                ecg_test_page=ecg_test_page,
                patient=patient,
                ecg_data_file=None,
                log_history=False,
                username=username,
            )
    finally:
        try:
            if ecg_test_page is not None:
                if backup_data is not None:
                    ecg_test_page.data = backup_data
                if backup_buffer_size is not None:
                    try:
                        ecg_test_page.buffer_size = backup_buffer_size
                    except Exception:
                        pass
                demo_manager = getattr(ecg_test_page, "demo_manager", None)
                if demo_manager is not None:
                    if backup_time_window is not None:
                        try:
                            demo_manager.time_window = backup_time_window
                        except Exception:
                            pass
                    if backup_fs is not None:
                        try:
                            demo_manager.samples_per_second = backup_fs
                        except Exception:
                            pass
        except Exception as e:
            print(f"Demo report: error while restoring live demo state: {e}")
