"""
Support Complaint API integration (CardioX Support API).

Provides:
- submit_complaint(): POST complaint with x-api-key header
  - offline-first via OfflineQueue
  - local rate limit enforcement (10 requests/min per machine_id)
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .offline_queue import get_offline_queue

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:
    pass


def _safe_json_load(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        return default
    return default


def _safe_json_write(path: str, data: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except Exception:
        return


@dataclass(frozen=True)
class SupportConfig:
    url: str
    api_key: str
    timeout_s: int = 12


class SupportAPI:
    def __init__(self):
        self.config = SupportConfig(
            url=os.getenv(
                "SUPPORT_COMPLAINT_URL",
                "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/prod/support/complaint",
            ).strip(),
            api_key=os.getenv("SUPPORT_API_KEY", "CX-SW-2026-Deckmount-Key").strip(),
            timeout_s=int(os.getenv("SUPPORT_API_TIMEOUT", "12").strip() or "12"),
        )
        self.offline_queue = get_offline_queue()
        self._rate_lock = threading.Lock()
        self._rate_file = os.path.join("logs", "support_rate_limit.json")

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
        }

    def _rate_limit_check_and_record(self, machine_id: str) -> bool:
        """
        Enforce 10 requests per rolling 60 seconds per machine_id.
        Returns True if allowed.
        """
        machine_id = str(machine_id or "").strip()
        if not machine_id:
            # If unknown machine id, don't block; API itself may reject anyway.
            return True

        now = time.time()
        with self._rate_lock:
            data = _safe_json_load(self._rate_file, default={})
            if not isinstance(data, dict):
                data = {}
            entries = data.get(machine_id, [])
            if not isinstance(entries, list):
                entries = []

            # Keep only last 60 seconds
            recent = [ts for ts in entries if isinstance(ts, (int, float)) and (now - float(ts)) < 60.0]
            if len(recent) >= 10:
                data[machine_id] = recent
                _safe_json_write(self._rate_file, data)
                return False

            recent.append(now)
            data[machine_id] = recent
            _safe_json_write(self._rate_file, data)
            return True

    def _post_complaint(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            self.config.url,
            headers=self._headers(),
            json=payload,
            timeout=(3, max(3, int(self.config.timeout_s))),
        )
        try:
            data = resp.json()
        except Exception:
            data = {"success": False, "message": resp.text}

        if resp.status_code >= 200 and resp.status_code < 300 and isinstance(data, dict):
            return data

        return {
            "success": False,
            "status_code": resp.status_code,
            "message": data.get("message") if isinstance(data, dict) else resp.text,
            "raw": data,
        }

    def get_complaint_status(self, complaint_id: str) -> Dict[str, Any]:
        complaint_id = str(complaint_id or "").strip()
        if not complaint_id:
            return {"success": False, "status": "error", "message": "Complaint ID is required"}

        url = os.getenv(
            "SUPPORT_COMPLAINT_STATUS_URL",
            "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/prod/support/complaint/status",
        ).strip()

        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                params={"complaint_id": complaint_id},
                timeout=(3, max(3, int(self.config.timeout_s))),
            )
            try:
                data = resp.json()
            except Exception:
                data = {"success": False, "message": resp.text}

            if resp.status_code >= 200 and resp.status_code < 300 and isinstance(data, dict):
                return data

            return {
                "success": False,
                "status": "error",
                "status_code": resp.status_code,
                "message": data.get("message") if isinstance(data, dict) else resp.text,
                "raw": data,
            }
        except requests.exceptions.Timeout:
            return {"success": False, "status": "error", "message": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "status": "error", "message": "Connection error"}
        except Exception as e:
            return {"success": False, "status": "error", "message": str(e)}

    def submit_complaint(
        self,
        *,
        name: str,
        machine_id: str,
        complaint: str,
        source: str = "software",
        queue_if_offline: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "name": str(name or "").strip(),
            "machine_id": str(machine_id or "").strip(),
            "complaint": str(complaint or "").strip(),
            "source": str(source or "software").strip() or "software",
        }

        if not payload["machine_id"]:
            return {"success": False, "status": "error", "message": "Machine ID is required"}

        if not payload["complaint"]:
            return {"success": False, "status": "error", "message": "Complaint cannot be empty"}

        if not self._rate_limit_check_and_record(payload["machine_id"]):
            return {
                "success": False,
                "status": "rate_limited",
                "message": "Rate limit exceeded (10 requests/minute for this machine_id). Please wait and try again.",
            }

        if queue_if_offline and not self.offline_queue.is_online():
            queued_id = self.offline_queue.queue_data("support_complaint", payload, priority=2)
            return {"success": True, "status": "queued", "queued_id": queued_id}

        try:
            data = self._post_complaint(payload)
            if isinstance(data, dict) and data.get("success") is True:
                status = data.get("status")
                return {
                    "success": True,
                    "status": "success",
                    "complaint_id": data.get("complaint_id", ""),
                    "complaint_status": status,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "resolved_at": data.get("resolved_at"),
                }
            return {"success": False, "status": "error", **(data if isinstance(data, dict) else {"raw": data})}
        except requests.exceptions.Timeout:
            if queue_if_offline:
                queued_id = self.offline_queue.queue_data("support_complaint", payload, priority=2)
                return {"success": True, "status": "queued", "queued_id": queued_id, "message": "Request timed out; queued"}
            return {"success": False, "status": "error", "message": "Request timed out"}
        except requests.exceptions.ConnectionError:
            if queue_if_offline:
                queued_id = self.offline_queue.queue_data("support_complaint", payload, priority=2)
                return {"success": True, "status": "queued", "queued_id": queued_id, "message": "Offline; queued"}
            return {"success": False, "status": "error", "message": "Connection error"}
        except Exception as e:
            return {"success": False, "status": "error", "message": str(e)}


_support_api: Optional[SupportAPI] = None


def get_support_api() -> SupportAPI:
    global _support_api
    if _support_api is None:
        _support_api = SupportAPI()
    return _support_api
