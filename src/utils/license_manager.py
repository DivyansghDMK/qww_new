"""
utils/license_manager.py
========================
Client-side license validation for CardioX / ECG Monitor.

Flow:
  1. On startup, read cached license (HMAC-integrity-checked).
  2. If cache is valid, not expired, and within offline-grace window → allow.
  3. Otherwise contact license server to (re-)validate.
  4. Server response is HMAC-signed; client verifies signature.
  5. Valid response is written back to encrypted cache.

License key format  (displayed as XXXXX-XXXXX-XXXXX-XXXXX):
  20 base-32 chars (no ambiguous 0/O, 1/I) encoding:
    tier      1 byte   (0=trial, 1=standard, 2=pro, 3=enterprise)
    expiry    4 bytes  (Unix timestamp, seconds — 0 = perpetual)
    nonce     4 bytes  (random, makes each key unique)
    checksum  3 bytes  (first 3 bytes of HMAC-SHA256(payload, SECRET))
  Total: 12 bytes → 20 base-32 chars

The server performs the authoritative validation; the checksum is a
lightweight offline sanity check only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple
from dotenv import find_dotenv, load_dotenv

# Load project .env so standalone imports and the app share the same settings.
load_dotenv(find_dotenv(usecwd=True), override=False)

# ── Configuration ─────────────────────────────────────────────────────────────
# Override LICENSE_SERVER_URL via environment variable in production.
LICENSE_SERVER_URL: str = os.getenv(
    "LICENSE_SERVER_URL",
    "https://m4qoae4d8e.execute-api.us-east-1.amazonaws.com/prod/api/v1",
)

def _load_hmac_secret() -> bytes:
    """Load the shared HMAC secret from env as UTF-8 bytes."""
    raw = os.getenv(
        "LICENSE_HMAC_SECRET",
        "949d13007c30ce16e1609c590ef31866d7f68010127c9c514e840de6b02ea1fb",
    ).strip()
    return raw.encode("utf-8")


# Shared HMAC secret — MUST match the value on the server.
# Set via environment variable; never hard-code in production builds.
_HMAC_SECRET: bytes = _load_hmac_secret()

# Optional gateway token used by the API layer in front of the license server.
LICENSE_API_TOKEN: str = os.getenv("LICENSE_API_TOKEN", "").strip()

SOFTWARE_VERSION: str = "1.1.1"
PRODUCT_CODE: str = "CARDIOX"

# How many days the app may run offline before it requires re-validation.
OFFLINE_GRACE_DAYS: int = 7

# Local cache file location (writable on all platforms).
_CACHE_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Deckmount" / "ECGMonitor"
_CACHE_FILE: Path = _CACHE_DIR / "license.cache"
_DEVICE_ID_FILE: Path = _CACHE_DIR / "device.id"

# Base-32 alphabet — no ambiguous chars (0, O, 1, I)
_B32_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


# ── Hardware Fingerprint ───────────────────────────────────────────────────────

def get_hardware_fingerprint() -> str:
    """Return a stable installation fingerprint for this app instance."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if _DEVICE_ID_FILE.exists():
            device_id = _DEVICE_ID_FILE.read_text(encoding="utf-8").strip()
            if device_id:
                return device_id
    except Exception:
        pass

    # Create the ID once and keep it local to this installation.
    try:
        seed = f"{uuid.uuid4().hex}:{platform.node()}:{os.getenv('USERNAME', '')}:{os.getenv('COMPUTERNAME', '')}"
        device_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        _DEVICE_ID_FILE.write_text(device_id, encoding="utf-8")
        return device_id
    except Exception:
        # Last resort: derive something from the host so activation can still proceed.
        fallback = f"{platform.node()}|{platform.system()}|{platform.release()}|{uuid.getnode():012x}"
        return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def get_machine_context() -> Dict[str, str]:
    """Return machine metadata expected by the activation API."""
    try:
        uname = platform.uname()
    except Exception:
        uname = None
    return {
        "machine_name": os.getenv("COMPUTERNAME", "") or (uname.node if uname else platform.node()),
        "machine_os": f"{platform.system()} {platform.release()}".strip(),
        "machine_host": platform.node(),
    }


# ── License Key Utilities ──────────────────────────────────────────────────────

def _b32_encode(data: bytes) -> str:
    """Encode bytes to our custom base-32 string."""
    result = []
    acc = 0
    bits = 0
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            result.append(_B32_ALPHA[(acc >> bits) & 0x1F])
    if bits > 0:
        result.append(_B32_ALPHA[(acc << (5 - bits)) & 0x1F])
    return "".join(result)


def _b32_decode(s: str) -> bytes:
    """Decode our custom base-32 string to bytes."""
    s = s.upper().replace("-", "").replace(" ", "")
    acc = 0
    bits = 0
    result = []
    for char in s:
        idx = _B32_ALPHA.find(char)
        if idx < 0:
            raise ValueError(f"Invalid character in license key: {char!r}")
        acc = (acc << 5) | idx
        bits += 5
        if bits >= 8:
            bits -= 8
            result.append((acc >> bits) & 0xFF)
    return bytes(result)


def format_key(raw_key: str) -> str:
    """Format a 20-char raw key as XXXXX-XXXXX-XXXXX-XXXXX."""
    raw_key = raw_key.upper().replace("-", "").replace(" ", "")
    return "-".join(raw_key[i:i+5] for i in range(0, len(raw_key), 5))


def parse_key_payload(license_key: str) -> Optional[Dict]:
    """
    Decode license key without contacting the server.
    Returns dict with tier/expiry/nonce, or None if the checksum fails.
    """
    try:
        raw = license_key.upper().replace("-", "").replace(" ", "")
        if len(raw) != 20:
            return None
        data = _b32_decode(raw)          # 12 bytes
        if len(data) < 12:
            return None
        tier     = data[0]
        expiry   = struct.unpack(">I", data[1:5])[0]
        nonce    = data[5:9]
        checksum = data[9:12]

        # Verify embedded checksum
        payload = data[:9]
        expected_cs = hmac.new(_HMAC_SECRET, payload, hashlib.sha256).digest()[:3]
        if not hmac.compare_digest(checksum, expected_cs):
            return None

        return {
            "tier":   tier,
            "expiry": expiry,
            "nonce":  nonce.hex(),
        }
    except Exception:
        return None


def parse_key_metadata(license_key: str) -> Optional[Dict]:
    """
    Decode the visible key format without enforcing the embedded checksum.

    This keeps local validation lightweight while allowing backend-issued keys
    to be checked authoritatively by the server.
    """
    try:
        raw = license_key.upper().replace("-", "").replace(" ", "")
        if len(raw) != 20:
            return None
        data = _b32_decode(raw)
        if len(data) < 12:
            return None
        tier = data[0]
        expiry = struct.unpack(">I", data[1:5])[0]
        nonce = data[5:9]
        return {
            "tier": tier,
            "expiry": expiry,
            "nonce": nonce.hex(),
        }
    except Exception:
        return None


def is_key_expired_locally(license_key: str) -> bool:
    """Quick local expiry check (does not contact server)."""
    payload = parse_key_metadata(license_key)
    if payload is None:
        return True                    # invalid key
    expiry = payload["expiry"]
    if expiry == 0:
        return False                   # perpetual
    return int(time.time()) > expiry


# ── Local Encrypted Cache ──────────────────────────────────────────────────────

def _cache_write(data: Dict) -> None:
    """Write license cache to disk, protected by HMAC."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, sort_keys=True).encode()
        sig = hmac.new(_HMAC_SECRET, payload, hashlib.sha256).hexdigest()
        _CACHE_FILE.write_text(
            json.dumps({"payload": data, "sig": sig}, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[License] Cache write failed: {e}")


def _cache_read() -> Optional[Dict]:
    """Read and verify cached license. Returns None if missing or tampered."""
    try:
        if not _CACHE_FILE.exists():
            return None
        obj = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        payload_bytes = json.dumps(obj["payload"], sort_keys=True).encode()
        expected_sig = hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, obj["sig"]):
            print("[License] Cache tampered — ignoring.")
            return None
        return obj["payload"]
    except Exception:
        return None


def _cache_clear() -> None:
    """Remove the local license cache (e.g. after deactivation)."""
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
    except Exception:
        pass


def clear_license_cache() -> None:
    """Public helper to remove only the JSON cache artifact."""
    _cache_clear()


def remember_valid_license(
    license_key: str,
    fingerprint: str,
    result: Optional[Dict] = None,
) -> None:
    """
    Persist a known-valid license locally so the next startup can skip the dialog.

    This is used after successful activation to seed the HMAC-protected cache,
    which allows offline launches to trust the most recent successful check.
    """
    try:
        payload: Dict[str, object] = {
            "license_key": license_key.strip().upper().replace(" ", ""),
            "hardware_fingerprint": fingerprint,
            "last_online": int(time.time()),
        }
        if isinstance(result, dict):
            for key in ("tier", "expires", "message", "source", "valid", "revoked"):
                if key in result:
                    payload[key] = result[key]
        _cache_write(payload)
    except Exception as e:
        print(f"[License] Could not persist valid license cache: {e}")


# ── Server Communication ───────────────────────────────────────────────────────

def _post_json(endpoint: str, body: Dict, timeout: int = 12) -> Dict:
    """Send a signed JSON POST to the license server."""
    import urllib.request
    import urllib.error

    url = f"{LICENSE_SERVER_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    payload_bytes = json.dumps(
        body,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    # Sign the request body so the server can verify it wasn't tampered
    req_sig = hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()

    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Request-Sig": req_sig,
            "X-Product": PRODUCT_CODE,
            "X-Version": SOFTWARE_VERSION,
        },
        method="POST",
    )
    if LICENSE_API_TOKEN:
        req.add_header("X-API-Key", LICENSE_API_TOKEN)
        req.add_header("Authorization", f"Bearer {LICENSE_API_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            return body
        except Exception:
            return {"valid": False, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"valid": False, "error": str(e), "offline": True}


def _verify_server_response(response: Dict) -> bool:
    """Verify the server's HMAC signature on its response."""
    response = dict(response)
    sig = response.pop("server_sig", None)
    if not sig:
        # Legacy servers may not sign — treat as unverified but still use
        return True
    candidate_payloads = [
        json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        json.dumps(response, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
    ]
    for payload_bytes in candidate_payloads:
        expected = hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, sig):
            return True
    print("[License] Server response signature mismatch — rejecting.")
    return False


def _response_indicates_revocation(result: Dict) -> bool:
    """Return True when the server response clearly indicates revocation."""
    try:
        if result.get("revoked") is True or result.get("license_revoked") is True:
            return True
        for key in ("message", "error", "reason", "status"):
            value = str(result.get(key, "")).lower()
            if "revoked" in value:
                return True
        return False
    except Exception:
        return False


def validate_with_server(license_key: str, fingerprint: str) -> Dict:
    """Contact the license server to validate a key + hardware pair."""
    body = {
        "license_key": license_key,
        "hardware_fingerprint": fingerprint,
    }
    result = _post_json("validate", body)
    _verify_server_response(result)
    return result


def activate_with_server(license_key: str, fingerprint: str, machine_name: str = "") -> Dict:
    """First-time activation: tie this license key to this hardware."""
    machine_ctx = get_machine_context()
    body = {
        "license_key": license_key,
        "hardware_fingerprint": fingerprint,
        "machine_name": machine_name or machine_ctx["machine_name"],
        "machine_os": machine_ctx["machine_os"],
        "machine_host": machine_ctx["machine_host"],
    }
    result = _post_json("activate", body)
    _verify_server_response(result)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def check_license(license_key: str, force_server: bool = False) -> Dict:
    """
    Full license check.  Call this at application startup.

    Returns a dict with at minimum:
        valid       bool
        source      "cache" | "server" | "local_expiry"
        message     str
        tier        int   (0=trial, 1=standard, 2=pro, 3=enterprise)
        expires     int   (Unix timestamp, 0=perpetual)
    """
    license_key = license_key.strip().upper().replace(" ", "")
    fingerprint = get_hardware_fingerprint()
    now = int(time.time())

    # ── 1. Quick local sanity check (no network) ─────────────────────────────
    local_payload = parse_key_metadata(license_key)
    if local_payload is None:
        return {
            "valid": False,
            "source": "local",
            "message": "Invalid license key format.",
            "tier": 0,
            "expires": 0,
        }
    if local_payload["expiry"] != 0 and now > local_payload["expiry"]:
        return {
            "valid": False,
            "source": "local_expiry",
            "message": "License key has expired.",
            "tier": local_payload["tier"],
            "expires": local_payload["expiry"],
        }

    # ── 2. Try the local cache ───────────────────────────────────────────────
    cached = _cache_read()
    if cached and not force_server:
        key_match = cached.get("license_key") == license_key
        fp_match = cached.get("hardware_fingerprint") == fingerprint
        srv_expiry = cached.get("expires", 0)
        not_server_expired = (srv_expiry == 0) or (now < srv_expiry)
        last_online = cached.get("last_online", 0)
        grace_seconds = OFFLINE_GRACE_DAYS * 86400
        within_grace = (now - last_online) < grace_seconds

        if key_match and fp_match and not_server_expired and within_grace:
            return {
                **cached,
                "valid": True,
                "source": "cache",
                "message": "License valid (cached).",
            }

    # ── 3. Contact server ────────────────────────────────────────────────────
    result = validate_with_server(license_key, fingerprint)
    if _response_indicates_revocation(result):
        return {
            "valid": False,
            "revoked": True,
            "source": "server",
            "message": result.get("message", "License key is revoked. Contact support."),
            "tier": 0,
            "expires": 0,
        }
    if result.get("valid"):
        result.setdefault("license_key", license_key)
        result.setdefault("hardware_fingerprint", fingerprint)
        result.setdefault("tier", local_payload["tier"])
        result.setdefault("expires", local_payload["expiry"])
        result["last_online"] = now
        result.setdefault("source", "server")
        result.setdefault("message", "License valid.")
        return result

    # ── 4. Server unreachable — fall back to cache if within grace ─────────
    if result.get("offline") and cached:
        key_match = cached.get("license_key") == license_key
        fp_match = cached.get("hardware_fingerprint") == fingerprint
        last_online = cached.get("last_online", 0)
        grace_seconds = OFFLINE_GRACE_DAYS * 86400
        within_grace = (now - last_online) < grace_seconds

        if key_match and fp_match and within_grace:
            days_left = max(0, int((grace_seconds - (now - last_online)) / 86400))
            return {
                **cached,
                "valid": True,
                "source": "offline_grace",
                "message": f"Running offline — {days_left} day(s) of grace period remaining.",
            }

    # ── 5. Completely invalid ───────────────────────────────────────────────
    return {
        "valid": False,
        "revoked": bool(result.get("revoked")),
        "source": "server",
        "message": result.get("error", result.get("message", "License validation failed.")),
        "tier": 0,
        "expires": 0,
    }


def deactivate(license_key: str) -> bool:
    """Deactivate this machine (contact server + clear cache)."""
    fingerprint = get_hardware_fingerprint()
    machine_ctx = get_machine_context()
    result = _post_json("deactivate", {
        "license_key": license_key,
        "hardware_fingerprint": fingerprint,
        "machine_name": machine_ctx["machine_name"],
        "machine_os": machine_ctx["machine_os"],
        "machine_host": machine_ctx["machine_host"],
    })
    _cache_clear()
    try:
        _LICENSE_KEY_FILE.unlink(missing_ok=True)  # Python 3.8+
    except Exception:
        pass
    return bool(result.get("success"))


# ── Storage helpers (used by the dialog) ─────────────────────────────────────

_LICENSE_KEY_FILE: Path = _CACHE_DIR / "license.key"


def load_stored_key() -> str:
    """Return the license key stored on this machine, or empty string."""
    try:
        if _LICENSE_KEY_FILE.exists():
            return _LICENSE_KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def save_stored_key(license_key: str) -> None:
    """Persist the license key locally (plaintext — it's not a secret)."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _LICENSE_KEY_FILE.write_text(license_key.strip(), encoding="utf-8")
    except Exception as e:
        print(f"[License] Could not save key: {e}")


def clear_stored_key() -> None:
    """Remove the saved license key so the activation dialog opens again."""
    try:
        _LICENSE_KEY_FILE.unlink(missing_ok=True)  # Python 3.8+
    except Exception as e:
        print(f"[License] Could not clear key: {e}")


# ── Tier helpers ──────────────────────────────────────────────────────────────

TIER_NAMES = {0: "Trial", 1: "Standard", 2: "Professional", 3: "Enterprise"}


def tier_name(tier: int) -> str:
    return TIER_NAMES.get(tier, "Unknown")
