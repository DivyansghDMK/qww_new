"""
ECG Unified Payload Builder  ·  Schema Version 1.0.0
======================================================
Single endpoint:  POST /api/v1/ecg/reports
Supports:         12_lead  |  hrv  |  hyperkalemia
Linked by:        machine_serial  (unique device identifier)

Payload structure follows the ECG UNIFIED PAYLOAD specification exactly.
All field rules are enforced here:
  - NO unit strings inside numeric values
  - Axis as int[3] array, not string
  - ALL 12 leads always present in ecg_data
  - hrv_result_reading / rr_intervals only for HRV
  - hyperkalemia[] only populated for hyperkalemia reports
  - reserve_1/2/3 always null
  - lead_arrangement always present
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading (mirrors cloud_uploader.py approach)
# ---------------------------------------------------------------------------
load_dotenv()
try:
    _this_file = Path(__file__).resolve()
    _project_root = _this_file.parents[2] if len(_this_file.parents) >= 3 else _this_file.parent
    _root_env = _project_root / '.env'
    if _root_env.exists():
        load_dotenv(dotenv_path=str(_root_env), override=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"
EVENT_TYPE = "ecg_report"

DEVICE_NAME = os.getenv("ECG_DEVICE_NAME", "RhythmPro ECG")
MANUFACTURER = os.getenv("ECG_MANUFACTURER", "Deckmount Electronics Pvt Ltd")
APP_VERSION = os.getenv("ECG_APP_VERSION", "4.10.1")
REPORT_VERSION = os.getenv("ECG_REPORT_VERSION", "1.0.18")

ECG_UNIFIED_API_URL = os.getenv(
    "ECG_UNIFIED_API_URL",
    ""   # Set ECG_UNIFIED_API_URL in .env to enable posting
)
ECG_UNIFIED_API_KEY = os.getenv("ECG_UNIFIED_API_KEY", "")

LEAD_NAMES: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMPLES_PER_LEAD = 5000
SAMPLING_RATE = 500
DURATION_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default   # NaN check
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _report_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %I:%M %p")


def _report_id(user_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(user_id or "0").replace(" ", "")
    return f"{ts}_{uid}"


def _leads_from_ecg_test_page(ecg_test_page: Any, n: int = SAMPLES_PER_LEAD) -> Dict[str, List[int]]:
    """Extract the latest `n` samples for all 12 leads from a live ECGTestPage."""
    out: Dict[str, List[int]] = {}
    if ecg_test_page is None:
        return out
    try:
        import numpy as np
        data_buf = getattr(ecg_test_page, 'data', None)
        if not isinstance(data_buf, (list, tuple)) or len(data_buf) < 12:
            return out
        for idx, name in enumerate(LEAD_NAMES):
            arr = data_buf[idx] if idx < len(data_buf) else []
            if isinstance(arr, np.ndarray):
                arr = arr.tolist()
            samples = list(arr[-n:] if len(arr) > n else arr)
            # Pad with zeros if we have fewer than n samples
            if len(samples) < n:
                samples = [0] * (n - len(samples)) + samples
            out[name] = [int(round(v)) for v in samples]
    except Exception as e:
        print(f"[ECGPayload] leads_from_ecg_test_page error: {e}")
    return out


def _leads_from_file(ecg_data_file: str, n: int = SAMPLES_PER_LEAD) -> Dict[str, List[int]]:
    """Load lead data from a saved ECG JSON file."""
    out: Dict[str, List[int]] = {}
    if not ecg_data_file or not os.path.isfile(ecg_data_file):
        return out
    try:
        with open(ecg_data_file, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        leads = saved.get('leads') if isinstance(saved, dict) else {}
        if not isinstance(leads, dict):
            return out
        for name in LEAD_NAMES:
            arr = leads.get(name) or leads.get(f'Lead_{name}') or []
            if not isinstance(arr, list):
                arr = list(arr)
            samples = arr[-n:] if len(arr) > n else arr
            if len(samples) < n:
                samples = [0] * (n - len(samples)) + samples
            out[name] = [int(round(v)) for v in samples]
    except Exception as e:
        print(f"[ECGPayload] leads_from_file error: {e}")
    return out


def _ensure_all_leads(leads: Dict[str, List[int]], n: int = SAMPLES_PER_LEAD) -> Dict[str, List[int]]:
    """Guarantee all 12 leads are present (zero-padded if missing)."""
    for name in LEAD_NAMES:
        if name not in leads or not leads[name]:
            leads[name] = [0] * n
        elif len(leads[name]) < n:
            leads[name] = [0] * (n - len(leads[name])) + leads[name]
        elif len(leads[name]) > n:
            leads[name] = leads[name][-n:]
    return leads


def _device_data_string(leads: Dict[str, List[int]]) -> str:
    """Build pipe-separated device data string as frame-wise [12 values] chunks."""
    try:
        # Required compact format:
        #   [I0,II0,III0,aVR0,aVL0,aVF0,V10,V20,V30,V40,V50,V60]|
        #   [I1,II1,III1,...]
        # One frame per sample, each frame has 12 lead values.
        frames = []
        n = 0
        for name in LEAD_NAMES:
            n = max(n, len(leads.get(name, [])))
        for s_idx in range(n):
            vals = []
            for name in LEAD_NAMES:
                arr = leads.get(name, [])
                v = arr[s_idx] if s_idx < len(arr) else 0
                vals.append(str(int(round(v))))
            frames.append("[" + ",".join(vals) + "]")
        return "|".join(frames)
    except Exception:
        return ""


def _machine_serial(patient: Dict, data: Dict, signup: Dict) -> str:
    """Resolve machine serial from all available sources."""
    serial = (
        str(patient.get('serial_number') or '').strip()
        or str(patient.get('serial_id') or '').strip()
        or str(data.get('machine_serial') or '').strip()
        or str(signup.get('serial_id') or '').strip()
        or str(signup.get('serial_number') or '').strip()
        or os.getenv('MACHINE_SERIAL_ID', '')
    )
    if not serial:
        # Last resort: crash logger
        try:
            from utils.crash_logger import get_crash_logger
            serial = get_crash_logger().machine_serial_id or ''
        except Exception:
            pass
    return serial or 'UNKNOWN'


def _load_signup_details(signup_details: Optional[Dict], patient: Dict, data: Dict) -> Dict:
    """
    Resolve signup/profile details from the provided dict or users.json.

    This keeps report payloads linked with the saved user profile for:
    - normal signup users
    - phone OTP login users
    """
    if isinstance(signup_details, dict) and signup_details:
        return dict(signup_details)

    try:
        project_root = Path(__file__).resolve().parents[2]
        users_path = project_root / "users.json"
        if not users_path.exists():
            return {}

        with users_path.open("r", encoding="utf-8") as f:
            users = json.load(f)
        if not isinstance(users, dict):
            return {}

        candidate_keys = [
            str(data.get("username") or "").strip(),
            str(data.get("user_id") or "").strip(),
            str(patient.get("username") or "").strip(),
            str(patient.get("phone") or patient.get("mobile_no") or data.get("mobile_no") or "").strip(),
            str(patient.get("name") or patient.get("patient_name") or "").strip(),
        ]
        candidate_keys = [v for v in candidate_keys if v]

        for key in candidate_keys:
            rec = users.get(key)
            if isinstance(rec, dict):
                return dict(rec)

        for _, rec in users.items():
            if not isinstance(rec, dict):
                continue
            rec_phone = str(rec.get("phone") or rec.get("contact") or "").strip()
            rec_name = str(rec.get("full_name") or "").strip().lower()
            for key in candidate_keys:
                if rec_phone and key == rec_phone:
                    return dict(rec)
                if rec_name and key.lower() == rec_name:
                    return dict(rec)
        return {}
    except Exception:
        return {}


def _settings_details(settings_manager: Any) -> Dict[str, str]:
    """Extract ECG settings from SettingsManager."""
    defaults = {
        "paper_speed": "25 mm/s",
        "gain": "10 mm/mV",
        "filter_band": "0.5-150 Hz",
        "ac_filter": "50 Hz",
        "lead_arrangement": "standard",
    }
    if settings_manager is None:
        return defaults
    try:
        speed = settings_manager.get_setting('wave_speed', '25')
        gain = settings_manager.get_setting('wave_gain', '10')
        ac = settings_manager.get_setting('filter_ac', '50')
        emg = settings_manager.get_setting('filter_emg', '25')
        dft = settings_manager.get_setting('filter_dft', '0.5')
        lead_seq = str(settings_manager.get_setting('lead_sequence', 'Standard') or 'Standard').lower()
        arrangement = 'cabrera' if lead_seq == 'cabrera' else 'standard'
        return {
            "paper_speed": f"{speed} mm/s",
            "gain": f"{gain} mm/mV",
            "filter_band": f"{dft}-{emg} Hz",
            "ac_filter": f"{ac} Hz",
            "lead_arrangement": arrangement,
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Core payload builders
# ---------------------------------------------------------------------------

def _build_patient_block(patient: Dict, data: Dict, report_id: str, signup: Optional[Dict] = None) -> Dict:
    data_patient = data.get('patient') if isinstance(data.get('patient'), dict) else {}

    first = str(
        patient.get('first_name')
        or data_patient.get('first_name')
        or ''
    ).strip()
    last = str(
        patient.get('last_name')
        or data_patient.get('last_name')
        or ''
    ).strip()
    full = str(
        patient.get('patient_name')
        or patient.get('name')
        or data_patient.get('patient_name')
        or data_patient.get('name')
        or data.get('patient_name')
        or data.get('name')
        or (signup or {}).get('full_name')
        or ''
    ).strip()
    if not full:
        full = f"{first} {last}".strip()

    report_date = (
        str(
            data.get('report_date')
            or patient.get('report_date')
            or data_patient.get('report_date')
            or ''
        ).strip()
        or _report_date_str()
    )

    return {
        "user_id": str(
            patient.get('serial_id')
            or patient.get('serial_number')
            or data_patient.get('serial_id')
            or data_patient.get('serial_number')
            or data.get('user_id')
            or data.get('serial_id')
            or data.get('serial_number')
            or (signup or {}).get('serial_id')
            or (signup or {}).get('serial_number')
            or (signup or {}).get('phone')
            or ''
        ).strip() or "0",
        "report_id": report_id,
        "name": full or "Unknown",
        "age": _safe_int(
            patient.get('age')
            or data_patient.get('age')
            or data.get('age')
            or (signup or {}).get('age')
        ),
        "gender": str(
            patient.get('gender')
            or data_patient.get('gender')
            or data.get('gender')
            or (signup or {}).get('gender')
            or 'Unknown'
        ).strip(),
        "mobile_no": str(
            patient.get('phone')
            or patient.get('mobile_no')
            or data_patient.get('phone')
            or data_patient.get('mobile_no')
            or data.get('mobile_no')
            or data.get('phone')
            or (signup or {}).get('phone')
            or (signup or {}).get('contact')
            or ''
        ).strip(),
        "report_date": report_date,
    }


def _build_device_block(machine_serial: str) -> Dict:
    return {
        "device_name": DEVICE_NAME,
        "manufacturer": MANUFACTURER,
        "app_version": APP_VERSION,
        "report_version": REPORT_VERSION,
        "machine_serial": machine_serial,
    }


def _build_data_details() -> Dict:
    return {
        "format": "int16_le",
        "leads": 12,
        "sampling_rate": SAMPLING_RATE,
        "samples_per_lead": SAMPLES_PER_LEAD,
        "duration_seconds": DURATION_SECONDS,
        "reserve_1": None,
        "reserve_2": None,
        "reserve_3": None,
    }


def _build_result_reading(data: Dict) -> Dict:
    """Build result_reading block — NO unit strings inside values."""
    hr = _safe_int(data.get('HR_bpm') or data.get('Heart_Rate') or data.get('HR'))
    rr = _safe_int(data.get('RR_ms') or data.get('RR'))
    pr = _safe_int(data.get('PR') or data.get('PR_ms'))
    qrs = _safe_int(data.get('QRS') or data.get('QRS_ms'))
    qt = _safe_int(data.get('QT') or data.get('QT_ms'))
    qtc = _safe_int(data.get('QTc') or data.get('QTc_ms'))
    qtcf = _safe_int(data.get('QTc_Fridericia') or data.get('QTcF') or data.get('QTCF_ms'))
    st = _safe_int(data.get('ST') or data.get('ST_ms'))

    # Axis: must be int[3]
    raw_axis = data.get('P_QRS_T_deg') or data.get('P_QRS_T') or []
    if isinstance(raw_axis, list) and len(raw_axis) >= 3:
        p_qrs_t = [_safe_int(raw_axis[0]), _safe_int(raw_axis[1]), _safe_int(raw_axis[2])]
    else:
        p_axis = _safe_int(data.get('P_axis') or data.get('P_deg'))
        qrs_axis = _safe_int(data.get('QRS_axis') or data.get('QRS_deg'))
        t_axis = _safe_int(data.get('T_axis') or data.get('T_deg'))
        p_qrs_t = [p_axis, qrs_axis, t_axis]

    rv5 = _safe_float(data.get('RV5_mV') or data.get('RV5'))
    sv1 = _safe_float(data.get('SV1_mV') or data.get('SV1'))
    rv5_sv1 = _safe_float(data.get('RV5_plus_SV1_mV') or data.get('RV5_SV1'))
    if rv5_sv1 == 0.0 and (rv5 != 0.0 or sv1 != 0.0):
        rv5_sv1 = round(rv5 + abs(sv1), 4)

    rv6 = _safe_float(data.get('RV6_mV') or data.get('RV6'))
    sv2 = _safe_float(data.get('SV2_mV') or data.get('SV2'))
    rv6_sv2 = _safe_float(data.get('RV6_plus_SV2_mV') or data.get('RV6_SV2'))
    if rv6_sv2 == 0.0 and (rv6 != 0.0 or sv2 != 0.0):
        rv6_sv2 = round(rv6 + abs(sv2), 4)

    return {
        "HR_bpm": hr,
        "RR_ms": rr,
        "PR_ms": pr,
        "QRS_ms": qrs,
        "QT_ms": qt,
        "QTc_ms": qtc,
        "QTCF_ms": qtcf,
        "ST_ms": st,
        "P_QRS_T_deg": p_qrs_t,
        "RV5_mV": rv5,
        "SV1_mV": sv1,
        "RV5_plus_SV1_mV": rv5_sv1,
        "RV6_mV": rv6,
        "SV2_mV": sv2,
        "RV6_plus_SV2_mV": rv6_sv2,
    }


def _build_hrv_result_reading(data: Dict) -> Dict:
    """HRV-specific result_reading (HR only + HRV metrics)."""
    hr = _safe_int(data.get('HR_bpm') or data.get('Heart_Rate') or data.get('HR'))
    return {
        "HR_bpm": hr,
        "SDNN_ms": _safe_float(data.get('SDNN_ms') or data.get('SDNN')),
        "RMSSD_ms": _safe_float(data.get('RMSSD_ms') or data.get('RMSSD')),
        "pNN50_percent": _safe_float(data.get('pNN50_percent') or data.get('pNN50')),
    }


def _build_rr_intervals_block(data: Dict) -> Dict:
    """Build rr_intervals block from HRV data."""
    rr_ms = data.get('rr_intervals_ms') or data.get('rr_intervals') or []
    hr_bpm = data.get('heart_rate_bpm') or []
    nn50 = _safe_int(data.get('nn50_count') or data.get('nn50'))

    # Time domain
    td = {
        "SDNN_ms": _safe_float(data.get('SDNN_ms') or data.get('SDNN')),
        "RMSSD_ms": _safe_float(data.get('RMSSD_ms') or data.get('RMSSD')),
        "pNN50_pct": _safe_float(data.get('pNN50_percent') or data.get('pNN50')),
    }

    # Frequency domain
    fd = {
        "LF_ms2": _safe_float(data.get('LF_ms2') or data.get('LF')),
        "HF_ms2": _safe_float(data.get('HF_ms2') or data.get('HF')),
        "LF_HF_ratio": _safe_float(data.get('LF_HF_ratio') or data.get('LF_HF')),
    }

    return {
        "rr_intervals_ms": [_safe_int(v) for v in rr_ms],
        "heart_rate_bpm": [_safe_int(v) for v in hr_bpm],
        "nn50_count": nn50,
        "time_domain": td,
        "frequency_domain": fd,
    }


def _build_clinical_findings(
    report_type: str,
    data: Dict,
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
    hyperkalemia_findings: Optional[List[str]] = None,
) -> Dict:
    _conclusions = list(conclusions or data.get('conclusion') or data.get('conclusions') or [])
    _arrhythmia = list(arrhythmia or data.get('arrhythmia') or data.get('arrhythmias') or [])

    if report_type == 'hyperkalemia':
        _hyper = list(
            hyperkalemia_findings
            or data.get('hyperkalemia')
            or data.get('hyperkalemia_findings')
            or []
        )
    else:
        _hyper = []

    return {
        "conclusion": _conclusions,
        "arrhythmia": _arrhythmia,
        "hyperkalemia": _hyper,
    }


def _resolve_unified_api_url() -> str:
    """Resolve the ECG unified API URL from env, with a migration fallback."""
    explicit_url = os.getenv("ECG_UNIFIED_API_URL", ECG_UNIFIED_API_URL).strip()
    if explicit_url:
        return explicit_url

    legacy_url = (
        os.getenv("DOCTOR_UPLOAD_API_URL")
        or os.getenv("DOCTOR_REVIEW_API_URL")
        or ""
    ).strip()
    if not legacy_url:
        return ""

    if legacy_url.endswith("/api/doctor/upload"):
        return legacy_url[: -len("/api/doctor/upload")] + "/api/v1/ecg/reports"

    return legacy_url.rstrip("/") + "/api/v1/ecg/reports"


def _resolve_unified_api_key() -> str:
    """Resolve the ECG unified API key from env, with doctor API key fallback."""
    return (
        os.getenv("ECG_UNIFIED_API_KEY", ECG_UNIFIED_API_KEY).strip()
        or os.getenv("DOCTOR_UPLOAD_API_KEY", "").strip()
        or os.getenv("DOCTOR_REVIEW_API_KEY", "").strip()
    )


def _resolve_pdf_backend_base_url() -> str:
    """
    Resolve the PDF backend base URL.

    Expected base form:
        https://host/api/v1
    Final upload target becomes:
        <base>/reports/upload
    """
    explicit = os.getenv("BACKEND_API_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")

    unified = _resolve_unified_api_url()
    if "/api/v1/ecg/reports" in unified:
        return unified.split("/api/v1/ecg/reports", 1)[0].rstrip("/") + "/api/v1"

    return ""


def _resolve_pdf_backend_api_key() -> str:
    return (
        os.getenv("BACKEND_API_KEY", "").strip()
        or _resolve_unified_api_key()
    )


def _pdf_upload_metadata_from_payload(payload: Dict) -> Dict[str, Any]:
    patient = payload.get("patient_details", {}) or {}
    device = payload.get("device_details", {}) or {}
    findings = payload.get("clinical_findings", {}) or {}

    metadata: Dict[str, Any] = {
        "schema_version": payload.get("schema_version"),
        "event_type": payload.get("event_type"),
        "generated_at": payload.get("generated_at"),
        "report_type": payload.get("report_type"),
        "report_format": payload.get("report_format"),
        "report_id": patient.get("report_id"),
        "user_id": patient.get("user_id"),
        "patient_name": patient.get("name"),
        "patient_age": patient.get("age"),
        "patient_gender": patient.get("gender"),
        "patient_phone": patient.get("mobile_no"),
        "report_date": patient.get("report_date"),
        "machine_serial": device.get("machine_serial"),
        "conclusion": findings.get("conclusion", []),
        "arrhythmia": findings.get("arrhythmia", []),
        "hyperkalemia": findings.get("hyperkalemia", []),
    }
    return metadata


def _backend_report_json_from_payload(payload: Dict) -> Dict[str, Any]:
    """
    Compact JSON sent alongside PDF upload.
    Keeps clinically relevant report data + patient details, without large waveform blobs.
    """
    return {
        "schema_version": payload.get("schema_version"),
        "event_type": payload.get("event_type"),
        "generated_at": payload.get("generated_at"),
        "report_type": payload.get("report_type"),
        "report_format": payload.get("report_format"),
        "source_report_file": payload.get("source_report_file"),
        "patient_details": payload.get("patient_details", {}) or {},
        "device_details": payload.get("device_details", {}) or {},
        "result_reading": payload.get("result_reading", {}) or {},
        "hrv_result_reading": payload.get("hrv_result_reading", {}) or {},
        "rr_intervals": payload.get("rr_intervals", {}) or {},
        "clinical_findings": payload.get("clinical_findings", {}) or {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_12lead_payload(
    *,
    data: Dict,
    patient: Dict,
    settings_manager: Any = None,
    signup_details: Optional[Dict] = None,
    ecg_test_page: Any = None,
    ecg_data_file: Optional[str] = None,
    report_format: str = "12_1",
    source_report_file: str = "",
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
) -> Dict:
    """
    Build a schema-compliant payload for a 12-lead ECG report.

    Parameters
    ----------
    data              : dict with HR/PR/QRS/QT/QTc/ST/RV5/SV1/axis …
    patient           : dict with name/age/gender/phone/serial_id …
    settings_manager  : SettingsManager instance (optional)
    signup_details    : raw user record from users.json (optional)
    ecg_test_page     : live ECGTestPage instance (optional, for lead data)
    ecg_data_file     : path to saved ECG JSON file (optional)
    report_format     : "12_1" | "4_3" | "6_2"
    source_report_file: path to the generated PDF
    conclusions       : list of clinical conclusion strings
    arrhythmia        : list of arrhythmia strings

    Returns
    -------
    dict — JSON-serialisable unified payload
    """
    signup = _load_signup_details(signup_details, patient, data)
    machine_serial = _machine_serial(patient, data, signup)
    rid = _report_id(patient.get('serial_id') or machine_serial)

    # Lead data
    leads: Dict[str, List[int]] = {}
    if ecg_data_file:
        leads = _leads_from_file(ecg_data_file)
    if not leads and ecg_test_page is not None:
        leads = _leads_from_ecg_test_page(ecg_test_page)
    leads = _ensure_all_leads(leads)

    ecg_settings = _settings_details(settings_manager)

    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "generated_at": _now_iso(),
        "report_type": "12_lead",
        "report_format": report_format,
        "source_report_file": str(source_report_file),
        "patient_details": _build_patient_block(patient, data, rid, signup),
        "device_details": _build_device_block(machine_serial),
        "data_details": _build_data_details(),
        "ecg_settings": ecg_settings,
        "ecg_data": {
            "device_data_encoding": "pipe_separated_frames",
            "device_data": _device_data_string(leads),
            "leads_data": leads,
        },
        "result_reading": _build_result_reading(data),
        "clinical_findings": _build_clinical_findings(
            "12_lead", data, conclusions=conclusions, arrhythmia=arrhythmia
        ),
    }


def build_hrv_payload(
    *,
    data: Dict,
    patient: Dict,
    settings_manager: Any = None,
    signup_details: Optional[Dict] = None,
    ecg_test_page: Any = None,
    ecg_data_file: Optional[str] = None,
    source_report_file: str = "",
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
    hrv_metrics: Optional[Dict] = None,
) -> Dict:
    """
    Build a schema-compliant payload for an HRV report.

    Additional HRV keys in `data` (or `hrv_metrics`):
        SDNN_ms, RMSSD_ms, pNN50_percent, nn50_count
        rr_intervals_ms, heart_rate_bpm
        LF_ms2, HF_ms2, LF_HF_ratio
    """
    signup = _load_signup_details(signup_details, patient, data)
    machine_serial = _machine_serial(patient, data, signup)
    rid = _report_id(patient.get('serial_id') or machine_serial)
    merged = {**data, **(hrv_metrics or {})}

    # Lead data (HRV captures 5 min of Lead II but we include all 12 if available)
    leads: Dict[str, List[int]] = {}
    if ecg_data_file:
        leads = _leads_from_file(ecg_data_file)
    if not leads and ecg_test_page is not None:
        leads = _leads_from_ecg_test_page(ecg_test_page)
    leads = _ensure_all_leads(leads)

    ecg_settings = _settings_details(settings_manager)

    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "generated_at": _now_iso(),
        "report_type": "hrv",
        "report_format": "hrv",
        "source_report_file": str(source_report_file),
        "patient_details": _build_patient_block(patient, merged, rid, signup),
        "device_details": _build_device_block(machine_serial),
        "data_details": _build_data_details(),
        "ecg_settings": ecg_settings,
        "ecg_data": {
            "device_data_encoding": "pipe_separated_frames",
            "device_data": _device_data_string(leads),
            "leads_data": leads,
        },
        "result_reading": _build_result_reading(merged),
        "hrv_result_reading": _build_hrv_result_reading(merged),
        "rr_intervals": _build_rr_intervals_block(merged),
        "clinical_findings": _build_clinical_findings(
            "hrv", merged, conclusions=conclusions, arrhythmia=arrhythmia
        ),
    }


def build_hyperkalemia_payload(
    *,
    data: Dict,
    patient: Dict,
    settings_manager: Any = None,
    signup_details: Optional[Dict] = None,
    ecg_test_page: Any = None,
    ecg_data_file: Optional[str] = None,
    source_report_file: str = "",
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
    hyperkalemia_findings: Optional[List[str]] = None,
) -> Dict:
    """
    Build a schema-compliant payload for a hyperkalemia report.

    hyperkalemia_findings should be a list selecting from:
        "Peaked T waves", "Widened QRS", "Flattened P waves", "Sine wave pattern"
    """
    signup = _load_signup_details(signup_details, patient, data)
    machine_serial = _machine_serial(patient, data, signup)
    rid = _report_id(patient.get('serial_id') or machine_serial)

    # Lead data
    leads: Dict[str, List[int]] = {}
    if ecg_data_file:
        leads = _leads_from_file(ecg_data_file)
    if not leads and ecg_test_page is not None:
        leads = _leads_from_ecg_test_page(ecg_test_page)
    leads = _ensure_all_leads(leads)

    ecg_settings = _settings_details(settings_manager)

    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "generated_at": _now_iso(),
        "report_type": "hyperkalemia",
        "report_format": "hyperkalemia",
        "source_report_file": str(source_report_file),
        "patient_details": _build_patient_block(patient, data, rid, signup),
        "device_details": _build_device_block(machine_serial),
        "data_details": _build_data_details(),
        "ecg_settings": ecg_settings,
        "ecg_data": {
            "device_data_encoding": "pipe_separated_frames",
            "device_data": _device_data_string(leads),
            "leads_data": leads,
        },
        "result_reading": _build_result_reading(data),
        "clinical_findings": _build_clinical_findings(
            "hyperkalemia",
            data,
            conclusions=conclusions,
            arrhythmia=arrhythmia,
            hyperkalemia_findings=hyperkalemia_findings,
        ),
    }


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

def send_ecg_payload(payload: Dict, *, async_mode: bool = True) -> Optional[Dict]:
    """
    POST the unified ECG payload to ECG_UNIFIED_API_URL.

    Parameters
    ----------
    payload    : the dict returned by one of the build_* functions
    async_mode : if True (default) the POST runs in a background thread
                 so it never blocks report generation.

    Returns
    -------
    None if async_mode=True, else the response dict.
    """
    url = _resolve_unified_api_url()
    if not url:
        print("[ECGPayload] ECG_UNIFIED_API_URL not set — payload not sent.")
        return None

    def _post() -> Dict:
        try:
            api_key = _resolve_unified_api_key()
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["x-api-key"] = api_key

            serial = (
                payload.get("device_details", {}).get("machine_serial", "UNKNOWN")
                or "UNKNOWN"
            )
            report_type = payload.get("report_type", "unknown")
            print(f"[ECGPayload] Sending {report_type} payload (serial={serial}) → {url}")

            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            result = {
                "status_code": resp.status_code,
                "ok": resp.status_code in (200, 201, 202),
            }
            try:
                result["body"] = resp.json()
            except Exception:
                result["body"] = resp.text[:500]

            if result["ok"]:
                print(f"[ECGPayload] ✅ Payload accepted (HTTP {resp.status_code})")
            else:
                print(f"[ECGPayload] ⚠️  Server returned HTTP {resp.status_code}: {result['body']}")
            return result

        except requests.exceptions.ConnectionError:
            print("[ECGPayload] ⚠️  No internet — payload could not be sent (connection error)")
            return {"ok": False, "error": "connection_error"}
        except requests.exceptions.Timeout:
            print("[ECGPayload] ⚠️  Request timed out")
            return {"ok": False, "error": "timeout"}
        except Exception as exc:
            print(f"[ECGPayload] ❌ Unexpected error: {exc}")
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    if async_mode:
        t = threading.Thread(target=_post, daemon=True)
        t.start()
        return None
    else:
        return _post()


def send_pdf_report_to_backend(
    pdf_path: str,
    payload: Dict,
    *,
    async_mode: bool = True,
) -> Optional[Dict]:
    """
    Upload the generated PDF report to the backend multipart endpoint.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[ECGPayload] PDF not found — backend PDF upload skipped: {pdf_path}")
        return None

    base_url = _resolve_pdf_backend_base_url()
    if not base_url:
        print("[ECGPayload] BACKEND_API_URL not set — PDF backend upload not sent.")
        return None

    url = f"{base_url.rstrip('/')}/reports/upload"
    metadata = _pdf_upload_metadata_from_payload(payload)
    report_json = _backend_report_json_from_payload(payload)

    def _post() -> Dict:
        try:
            headers: Dict[str, str] = {}
            api_key = _resolve_pdf_backend_api_key()
            if api_key:
                headers["X-API-Key"] = api_key

            report_type = payload.get("report_type", "unknown")
            serial = payload.get("device_details", {}).get("machine_serial", "UNKNOWN")
            print(f"[ECGPayload] Uploading PDF for {report_type} (serial={serial}) → {url}")

            with open(pdf_path, "rb") as fh:
                files = {"file": (os.path.basename(pdf_path), fh, "application/pdf")}
                data = {
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "report_json": json.dumps(report_json, ensure_ascii=False),
                }
                resp = requests.post(url, files=files, data=data, headers=headers, timeout=60)

            result = {
                "status_code": resp.status_code,
                "ok": resp.status_code in (200, 201, 202),
            }
            try:
                result["body"] = resp.json()
            except Exception:
                result["body"] = resp.text[:500]

            if result["ok"]:
                print(f"[ECGPayload] ✅ PDF accepted (HTTP {resp.status_code})")
            else:
                print(f"[ECGPayload] ⚠️  PDF upload failed (HTTP {resp.status_code}): {result['body']}")
            return result
        except requests.exceptions.ConnectionError:
            print("[ECGPayload] ⚠️  PDF upload connection error")
            return {"ok": False, "error": "connection_error"}
        except requests.exceptions.Timeout:
            print("[ECGPayload] ⚠️  PDF upload timeout")
            return {"ok": False, "error": "timeout"}
        except Exception as exc:
            print(f"[ECGPayload] ❌ PDF upload error: {exc}")
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    if async_mode:
        t = threading.Thread(target=_post, daemon=True)
        t.start()
        return None
    return _post()


def save_payload_locally(payload: Dict, reports_dir: str = "reports") -> Optional[str]:
    """
    Save the payload JSON to disk under reports/ecg_payloads/.
    Returns the file path, or None on failure.
    """
    try:
        serial = payload.get("device_details", {}).get("machine_serial", "UNKNOWN")
        report_type = payload.get("report_type", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = os.path.join(reports_dir, "ecg_payloads")
        os.makedirs(subdir, exist_ok=True)
        fname = os.path.join(subdir, f"{report_type}_{serial}_{ts}.json")
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[ECGPayload] Saved payload locally → {fname}")
        return fname
    except Exception as e:
        print(f"[ECGPayload] Could not save payload locally: {e}")
        return None


# ---------------------------------------------------------------------------
# Convenience one-shot helpers (build + save + send)
# ---------------------------------------------------------------------------

def dispatch_12lead_report(
    *,
    data: Dict,
    patient: Dict,
    pdf_path: str,
    settings_manager: Any = None,
    signup_details: Optional[Dict] = None,
    ecg_test_page: Any = None,
    ecg_data_file: Optional[str] = None,
    report_format: str = "12_1",
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
    reports_dir: str = "reports",
    save_local: bool = True,
    send: bool = True,
) -> Optional[str]:
    """Build, optionally save, and optionally send a 12-lead payload. Returns local file path."""
    pdf_path = os.path.abspath(str(pdf_path))
    payload = build_12lead_payload(
        data=data, patient=patient, settings_manager=settings_manager,
        signup_details=signup_details, ecg_test_page=ecg_test_page,
        ecg_data_file=ecg_data_file, report_format=report_format,
        source_report_file=pdf_path, conclusions=conclusions, arrhythmia=arrhythmia,
    )
    fpath = save_payload_locally(payload, reports_dir) if save_local else None
    if send:
        send_ecg_payload(payload, async_mode=True)
        send_pdf_report_to_backend(pdf_path, payload, async_mode=True)
    return fpath


def dispatch_hrv_report(
    *,
    data: Dict,
    patient: Dict,
    pdf_path: str,
    settings_manager: Any = None,
    signup_details: Optional[Dict] = None,
    ecg_test_page: Any = None,
    ecg_data_file: Optional[str] = None,
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
    hrv_metrics: Optional[Dict] = None,
    reports_dir: str = "reports",
    save_local: bool = True,
    send: bool = True,
) -> Optional[str]:
    """Build, optionally save, and optionally send an HRV payload. Returns local file path."""
    pdf_path = os.path.abspath(str(pdf_path))
    payload = build_hrv_payload(
        data=data, patient=patient, settings_manager=settings_manager,
        signup_details=signup_details, ecg_test_page=ecg_test_page,
        ecg_data_file=ecg_data_file, source_report_file=pdf_path,
        conclusions=conclusions, arrhythmia=arrhythmia, hrv_metrics=hrv_metrics,
    )
    fpath = save_payload_locally(payload, reports_dir) if save_local else None
    if send:
        send_ecg_payload(payload, async_mode=True)
        send_pdf_report_to_backend(pdf_path, payload, async_mode=True)
    return fpath


def dispatch_hyperkalemia_report(
    *,
    data: Dict,
    patient: Dict,
    pdf_path: str,
    settings_manager: Any = None,
    signup_details: Optional[Dict] = None,
    ecg_test_page: Any = None,
    ecg_data_file: Optional[str] = None,
    conclusions: Optional[List[str]] = None,
    arrhythmia: Optional[List[str]] = None,
    hyperkalemia_findings: Optional[List[str]] = None,
    reports_dir: str = "reports",
    save_local: bool = True,
    send: bool = True,
) -> Optional[str]:
    """Build, optionally save, and optionally send a hyperkalemia payload. Returns local file path."""
    pdf_path = os.path.abspath(str(pdf_path))
    payload = build_hyperkalemia_payload(
        data=data, patient=patient, settings_manager=settings_manager,
        signup_details=signup_details, ecg_test_page=ecg_test_page,
        ecg_data_file=ecg_data_file, source_report_file=pdf_path,
        conclusions=conclusions, arrhythmia=arrhythmia,
        hyperkalemia_findings=hyperkalemia_findings,
    )
    fpath = save_payload_locally(payload, reports_dir) if save_local else None
    if send:
        send_ecg_payload(payload, async_mode=True)
        send_pdf_report_to_backend(pdf_path, payload, async_mode=True)
    return fpath
