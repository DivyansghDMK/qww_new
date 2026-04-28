import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()


class ECGAuthAPI:
    """Client for the ECG OTP/JWT backend."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "ECG_API_BASE_URL",
            "https://pmltkfluqk.execute-api.us-east-1.amazonaws.com",
        ).rstrip("/")
        self.auth_prefix = os.getenv("ECG_AUTH_API_PREFIX", "/dev/api")
        self.api_prefix = os.getenv("ECG_API_PREFIX", "/api")
        self.timeout = int(os.getenv("ECG_API_TIMEOUT_SECONDS", "20"))
        self.session_file = str(self._resolve_session_file())
        self._session_cache: Optional[Dict[str, Any]] = None

    def _resolve_session_file(self) -> Path:
        runtime_dir = os.getenv("ECG_RUNTIME_DIR", "").strip()
        if runtime_dir:
            base = Path(runtime_dir)
        else:
            base = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "Deckmount" / "ECGMonitor"
        base.mkdir(parents=True, exist_ok=True)
        return base / "ecg_auth_session.json"

    def normalize_phone(self, phone_no: str) -> str:
        digits = "".join(ch for ch in str(phone_no or "") if ch.isdigit())
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        return digits

    def send_otp(self, phone_no: str) -> Dict[str, Any]:
        payload = {"mobile_number": self.normalize_phone(phone_no)}
        if len(payload["mobile_number"]) != 10:
            raise ValueError("Phone number must be 10 digits.")
        return self._request("POST", f"{self.auth_prefix}/auth/send-otp", json=payload)

    def verify_otp(self, phone_no: str, otp: str) -> Dict[str, Any]:
        payload = {
            "mobile_number": self.normalize_phone(phone_no),
            "otp": str(otp).strip(),
        }
        if len(payload["mobile_number"]) != 10:
            raise ValueError("Phone number must be 10 digits.")
        if not payload["otp"]:
            raise ValueError("OTP is required.")

        response = self._request("POST", f"{self.auth_prefix}/auth/verify-otp", json=payload)
        token = self._extract_token(response)
        if not token:
            raise ValueError("OTP verified but JWT token was not found in the response.")

        session = {
            "mobile_number": payload["mobile_number"],
            "token": token,
            "verified_at": datetime.utcnow().isoformat() + "Z",
        }
        self._save_session(session)

        result = dict(response)
        result["token"] = token
        return result

    def get_token(self) -> Optional[str]:
        session = self._load_session()
        if not session:
            return None
        return session.get("token")

    def get_auth_headers(self) -> Dict[str, str]:
        token = self.get_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get_reports(self) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_prefix}/report", headers=self.get_auth_headers())

    def get_user_details(self) -> Dict[str, Any]:
        return self._request("GET", f"{self.api_prefix}/user/details", headers=self.get_auth_headers())

    def save_user_details(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"{self.api_prefix}/user/details",
            headers=self.get_auth_headers(),
            json=payload,
        )

    def check_mobile_availability(self, phone_no: str) -> Dict[str, Any]:
        normalized_phone = self.normalize_phone(phone_no)
        return self._request(
            "GET",
            f"{self.api_prefix}/user/check-mobile?mobile_number={normalized_phone}",
        )

    def clear_session(self) -> None:
        self._session_cache = None
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
        except OSError:
            pass

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, timeout=self.timeout, **kwargs)

        try:
            payload = response.json()
        except ValueError:
            payload = {
                "status": "error" if response.status_code >= 400 else "success",
                "message": response.text.strip(),
            }

        if response.status_code >= 400:
            message = self._extract_message(payload) or response.text or "Request failed"
            raise requests.HTTPError(
                f"{response.status_code} {message}",
                response=response,
            )

        if isinstance(payload, dict):
            return payload
        return {"status": "success", "data": payload}

    def _extract_token(self, payload: Dict[str, Any]) -> Optional[str]:
        candidate_keys = ("token", "jwt", "access_token", "id_token")
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        data = payload.get("data")
        if isinstance(data, dict):
            for key in candidate_keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _extract_message(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("message", "error", "detail"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return ""

    def _load_session(self) -> Optional[Dict[str, Any]]:
        if self._session_cache is not None:
            return self._session_cache
        try:
            with open(self.session_file, "r", encoding="utf-8") as handle:
                self._session_cache = json.load(handle)
                return self._session_cache
        except (OSError, ValueError):
            return None

    def _save_session(self, session: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
        with open(self.session_file, "w", encoding="utf-8") as handle:
            json.dump(session, handle, indent=2)
        self._session_cache = session


_ecg_auth_api: Optional[ECGAuthAPI] = None


def get_ecg_auth_api() -> ECGAuthAPI:
    global _ecg_auth_api
    if _ecg_auth_api is None:
        _ecg_auth_api = ECGAuthAPI()
    return _ecg_auth_api
