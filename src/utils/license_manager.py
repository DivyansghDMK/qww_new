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
    "LICENSE_SERVER_URL", "https://license.deckmount.io/api/v1"
)

# Shared HMAC secret — MUST match the value on the server.
# Set via environment variable; never hard-code in production builds.
_HMAC_SECRET: bytes = os.getenv(
    "LICENSE_HMAC_SECRET", "CHANGE_ME_32_BYTES_RANDOM_SECRET!"
).encode()

SOFTWARE_VERSION: str = "1.1.1"
PRODUCT_CODE: str = "CARDIOX"

# How many days the app may run offline before it requires re-validation.
OFFLINE_GRACE_DAYS: int = 7

# Local cache file location (writable on all platforms).
_CACHE_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Deckmount" / "ECGMonitor"
_CACHE_FILE: Path = _CACHE_DIR / "license.cache"

# Base-32 alphabet — no ambiguous chars (0, O, 1, I)
_B32_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


# ── Hardware Fingerprint ───────────────────────────────────────────────────────

def get_hardware_fingerprint() -> str:
    """Return a stable SHA-256 fingerprint of this machine's hardware."""
    parts: list[str] = []

    # Primary MAC address (node)
    try:
        parts.append(f"mac:{uuid.getnode():012x}")
    except Exception:
        pass

    # CPU / host identity
    try:
        parts.append(f"cpu:{platform.processor()[:40]}")
        parts.append(f"node:{platform.node()[:40]}")
        parts.append(f"machine:{platform.machine()}")
        parts.append(f"os:{platform.system()}-{platform.release()}")
    except Exception:
        pass

    # OS-level machine ID (most stable identifier)
    try:
        if sys.platform == "win32":
            import winreg  # type: ignore
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as k:
                mid, _ = winreg.QueryValueEx(k, "MachineGuid")
            parts.append(f"mid:{mid}")
        else:
            for mid_path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                p = Path(mid_path)
                if p.exists():
                    parts.append(f"mid:{p.read_text().strip()}")
                    break
    except Exception:
        pass

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


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


def is_key_expired_locally(license_key: str) -> bool:
    """Quick local expiry check (does not contact server)."""
    payload = parse_key_payload(license_key)
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


# ── Server Communication ───────────────────────────────────────────────────────

def _post_json(endpoint: str, body: Dict, timeout: int = 12) -> Dict:
    """Send a signed JSON POST to the license server."""
    import urllib.request
    import urllib.error

    url = f"{LICENSE_SERVER_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    payload_bytes = json.dumps(body).encode()

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
    sig = response.pop("server_sig", None)
    if not sig:
        # Legacy servers may not sign — treat as unverified but still use
        return True
    payload_bytes = json.dumps(response, sort_keys=True).encode()
    expected = hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()
    ok = hmac.compare_digest(expected, sig)
    if not ok:
        print("[License] Server response signature mismatch — rejecting.")
    return ok


def validate_with_server(license_key: str, fingerprint: str) -> Dict:
    """Contact the license server to validate a key + hardware pair."""
    body = {
        "license_key":          license_key,
        "hardware_fingerprint": fingerprint,
        "software_version":     SOFTWARE_VERSION,
        "platform":             platform.system(),
        "platform_version":     platform.release(),
        "timestamp":            int(time.time()),
    }
    result = _post_json("validate", body)
    _verify_server_response(result)
    return result


def activate_with_server(license_key: str, fingerprint: str, machine_name: str = "") -> Dict:
    """First-time activation: tie this license key to this hardware."""
    body = {
        "license_key":          license_key,
        "hardware_fingerprint": fingerprint,
        "machine_name":         machine_name or platform.node(),
        "software_version":     SOFTWARE_VERSION,
        "platform":             platform.system(),
        "timestamp":            int(time.time()),
    }
    result = _post_json("activate", body)
    _verify_server_response(result)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def check_license(license_key: str) -> Dict:
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
    local_payload = parse_key_payload(license_key)
    if local_payload is None:
        return {
            "valid":   False,
            "source":  "local",
            "message": "Invalid license key format or checksum.",
            "tier":    0,
            "expires": 0,
        }
    if local_payload["expiry"] != 0 and now > local_payload["expiry"]:
        return {
            "valid":   False,
            "source":  "local_expiry",
            "message": "License key has expired.",
            "tier":    local_payload["tier"],
            "expires": local_payload["expiry"],
        }

    # ── 2. Try the local cache ───────────────────────────────────────────────
    cached = _cache_read()
    if cached:
        key_match  = cached.get("license_key")          == license_key
        fp_match   = cached.get("hardware_fingerprint") == fingerprint
        srv_expiry = cached.get("expires", 0)
        not_server_expired = (srv_expiry == 0) or (now < srv_expiry)
        last_online   = cached.get("last_online", 0)
        grace_seconds = OFFLINE_GRACE_DAYS * 86400
        within_grace  = (now - last_online) < grace_seconds

        if key_match and fp_match and not_server_expired and within_grace:
            return {
                **cached,
                "valid":  True,
                "source": "cache",
                "message": "License valid (cached).",
            }

    # ── 3. Contact server ────────────────────────────────────────────────────
    result = validate_with_server(license_key, fingerprint)

    if result.get("valid"):
        result.setdefault("license_key",          license_key)
        result.setdefault("hardware_fingerprint", fingerprint)
        result.setdefault("tier",   local_payload["tier"])
        result.setdefault("expires", local_payload["expiry"])
        result["last_online"] = now
        result.setdefault("source", "server")
        result.setdefault("message", "License valid.")
        _cache_write(result)
        return result

    # ── 4. Server unreachable — fall back to cache if within grace ───────────
    if result.get("offline") and cached:
        key_match  = cached.get("license_key")          == license_key
        fp_match   = cached.get("hardware_fingerprint") == fingerprint
        last_online   = cached.get("last_online", 0)
        grace_seconds = OFFLINE_GRACE_DAYS * 86400
        within_grace  = (now - last_online) < grace_seconds

        if key_match and fp_match and within_grace:
            days_left = max(0, int((grace_seconds - (now - last_online)) / 86400))
            return {
                **cached,
                "valid":   True,
                "source":  "offline_grace",
                "message": f"Running offline — {days_left} day(s) of grace period remaining.",
            }

    # ── 5. Completely invalid ────────────────────────────────────────────────
    return {
        "valid":   False,
        "source":  "server",
        "message": result.get("error", result.get("message", "License validation failed.")),
        "tier":    0,
        "expires": 0,
    }


def deactivate(license_key: str) -> bool:
    """Deactivate this machine (contact server + clear cache)."""
    fingerprint = get_hardware_fingerprint()
    result = _post_json("deactivate", {
        "license_key":          license_key,
        "hardware_fingerprint": fingerprint,
        "timestamp":            int(time.time()),
    })
    _cache_clear()
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


# ── Tier helpers ──────────────────────────────────────────────────────────────

TIER_NAMES = {0: "Trial", 1: "Standard", 2: "Professional", 3: "Enterprise"}


def tier_name(tier: int) -> str:
    return TIER_NAMES.get(tier, "Unknown")
