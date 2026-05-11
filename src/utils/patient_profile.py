import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.app_paths import data_file

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
USERS_FILE = data_file("users.json")
ALL_PATIENTS_FILE = PROJECT_ROOT / "all_patients.json"


def _safe_read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _split_name(full_name: str) -> Dict[str, str]:
    parts = [part for part in str(full_name or "").strip().split() if part]
    if not parts:
        return {"first_name": "", "last_name": "", "patient_name": ""}
    return {
        "first_name": parts[0],
        "last_name": " ".join(parts[1:]),
        "patient_name": " ".join(parts),
    }


def _find_user_record(username: str = "", user_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(user_details, dict) and user_details:
        return dict(user_details)

    users = _safe_read_json(USERS_FILE)
    if not isinstance(users, dict):
        return {}

    key = str(username or "").strip()
    if key and isinstance(users.get(key), dict):
        return dict(users[key])

    key_lower = key.lower()
    for uname, record in users.items():
        if not isinstance(record, dict):
            continue
        if key and uname == key:
            return dict(record)
        full_name = str(record.get("full_name", "")).strip()
        phone = str(record.get("phone", "")).strip()
        if key and (full_name == key or phone == key):
            return dict(record)
        if key_lower and (full_name.lower() == key_lower or phone.lower() == key_lower):
            return dict(record)

    return {}


def _is_valid_name(name: str) -> bool:
    """Return True only if the name contains at least one alphabetic character.
    This prevents raw numeric usernames (e.g. '12', '007') from being
    displayed as patient names in ECG reports."""
    return bool(name) and any(c.isalpha() for c in name)


def patient_from_user_profile(username: str = "", user_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    record = _find_user_record(username=username, user_details=user_details)
    if not record:
        return {}

    raw_name = (
        str(record.get("patient_name", "")).strip()
        or str(record.get("full_name", "")).strip()
        or " ".join(
            part for part in [str(record.get("first_name", "")).strip(), str(record.get("last_name", "")).strip()] if part
        ).strip()
    )
    # Reject purely numeric names (login IDs masquerading as patient names)
    full_name = raw_name if _is_valid_name(raw_name) else ""
    name_parts = _split_name(full_name)

    patient = {
        "first_name": name_parts["first_name"],
        "last_name": name_parts["last_name"],
        "patient_name": name_parts["patient_name"],
        "age": str(record.get("age", "") or "").strip(),
        "gender": str(record.get("gender", "") or "").strip(),
        "phone": str(record.get("phone", "") or "").strip(),
        "serial_id": str(record.get("serial_id", "") or "").strip(),
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {k: v for k, v in patient.items() if _is_present(v) or k == "date_time"}


def get_latest_saved_patient() -> Dict[str, Any]:
    payload = _safe_read_json(ALL_PATIENTS_FILE)
    if isinstance(payload, dict):
        patients = payload.get("patients")
        if isinstance(patients, list) and patients:
            last_patient = patients[-1]
            if isinstance(last_patient, dict):
                return dict(last_patient)
    return {}


def merge_patient_profile(base_patient: Optional[Dict[str, Any]], fallback_patient: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base_patient or {})
    fallback = dict(fallback_patient or {})

    for key, value in fallback.items():
        if not _is_present(merged.get(key)):
            merged[key] = value

    if not _is_present(merged.get("patient_name")):
        combined_name = " ".join(
            part for part in [str(merged.get("first_name", "")).strip(), str(merged.get("last_name", "")).strip()] if part
        ).strip()
        if combined_name:
            merged["patient_name"] = combined_name

    if not _is_present(merged.get("first_name")) or not _is_present(merged.get("last_name")):
        name_parts = _split_name(str(merged.get("patient_name", "")).strip())
        if not _is_present(merged.get("first_name")):
            merged["first_name"] = name_parts["first_name"]
        if not _is_present(merged.get("last_name")):
            merged["last_name"] = name_parts["last_name"]

    merged["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return merged


def resolve_patient_profile(
    explicit_patient: Optional[Dict[str, Any]] = None,
    username: str = "",
    user_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current = dict(explicit_patient or {})
    if current:
        return merge_patient_profile(current, patient_from_user_profile(username=username, user_details=user_details))

    user_patient = patient_from_user_profile(username=username, user_details=user_details)
    if user_patient:
        return merge_patient_profile(user_patient, {})

    latest_patient = get_latest_saved_patient()
    if latest_patient:
        return merge_patient_profile(latest_patient, {})

    return {"date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
