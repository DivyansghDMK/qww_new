#!/usr/bin/env python3
"""
license_server/generate_license.py
===================================
Admin CLI to generate, inspect, and revoke CardioX license keys.

Usage:
    # Generate 5 Standard (tier=1) keys valid for 365 days:
    python generate_license.py generate --tier 1 --days 365 --count 5

    # Generate 1 Enterprise perpetual key:
    python generate_license.py generate --tier 3 --days 0

    # Inspect / decode a key (no network needed):
    python generate_license.py inspect ABCDE-FGHJK-LMNPQ-RSTUV

    # Revoke a key via the license server:
    python generate_license.py revoke ABCDE-FGHJK-LMNPQ-RSTUV

Set LICENSE_HMAC_SECRET and ADMIN_TOKEN environment variables to match your server.
"""

import argparse
import hashlib
import hmac as _hmac
import json
import os
import secrets
import struct
import sys
import time
import urllib.request
from datetime import datetime, timezone
from dotenv import find_dotenv, load_dotenv

# Load .env from the project tree so the generator matches the server secret.
load_dotenv(find_dotenv(usecwd=True), override=False)

# ── Config ────────────────────────────────────────────────────────────────────
HMAC_SECRET: bytes = os.getenv(
    "LICENSE_HMAC_SECRET", "CHANGE_ME_32_BYTES_RANDOM_SECRET!"
).encode()
SERVER_URL: str = os.getenv("LICENSE_SERVER_URL", "http://localhost:5000")
ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "CHANGE_THIS_ADMIN_TOKEN")

TIER_NAMES = {0: "Trial", 1: "Standard", 2: "Professional", 3: "Enterprise"}
_B32_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


# ── Codec ─────────────────────────────────────────────────────────────────────

def _b32_encode(data: bytes) -> str:
    result, acc, bits = [], 0, 0
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            result.append(_B32_ALPHA[(acc >> bits) & 0x1F])
    if bits:
        result.append(_B32_ALPHA[(acc << (5 - bits)) & 0x1F])
    return "".join(result)


def _b32_decode(s: str) -> bytes:
    s = s.upper().replace("-", "").replace(" ", "")
    acc, bits, result = 0, 0, []
    for ch in s:
        idx = _B32_ALPHA.find(ch)
        if idx < 0:
            raise ValueError(f"Invalid character: {ch!r}")
        acc = (acc << 5) | idx
        bits += 5
        if bits >= 8:
            bits -= 8
            result.append((acc >> bits) & 0xFF)
    return bytes(result)


def generate_key(tier: int, validity_days: int) -> str:
    """Generate a signed license key locally (no server needed)."""
    expiry = 0 if validity_days == 0 else int(time.time()) + validity_days * 86400
    nonce  = secrets.token_bytes(4)
    payload = bytes([tier & 0xFF]) + struct.pack(">I", expiry) + nonce
    checksum = _hmac.new(HMAC_SECRET, payload, hashlib.sha256).digest()[:3]
    raw_key = _b32_encode(payload + checksum)
    return "-".join(raw_key[i:i+5] for i in range(0, 20, 5))


def decode_key(license_key: str) -> dict | None:
    try:
        raw = license_key.upper().replace("-", "").replace(" ", "")
        if len(raw) != 20:
            return None
        data = _b32_decode(raw)
        if len(data) < 12:
            return None
        tier     = data[0]
        expiry   = struct.unpack(">I", data[1:5])[0]
        checksum = data[9:12]
        expected = _hmac.new(HMAC_SECRET, data[:9], hashlib.sha256).digest()[:3]
        if not _hmac.compare_digest(checksum, expected):
            return None
        return {"tier": tier, "expiry": expiry, "nonce": data[5:9].hex()}
    except Exception:
        return None


# ── Server helpers ─────────────────────────────────────────────────────────────

def _admin_post(endpoint: str, body: dict) -> dict:
    url = f"{SERVER_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {ADMIN_TOKEN}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # type: ignore
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def _admin_get(endpoint: str) -> dict:
    url = f"{SERVER_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


# ── CLI Commands ───────────────────────────────────────────────────────────────

def cmd_generate(args):
    print(f"\nGenerating {args.count} × {TIER_NAMES.get(args.tier,'?')} key(s) "
          f"({'perpetual' if args.days == 0 else f'{args.days} days'})...\n")
    keys = []
    for _ in range(args.count):
        key = generate_key(args.tier, args.days)
        keys.append(key)
        print(f"  {key}")

    if args.output:
        with open(args.output, "a") as f:
            for key in keys:
                f.write(key + "\n")
        print(f"\n  ✓ Appended to {args.output}")

    print()


def cmd_inspect(args):
    key = args.key.strip()
    info = decode_key(key)
    if info is None:
        print(f"\n  ✗ Invalid key: {key}\n")
        sys.exit(1)

    expiry = info["expiry"]
    now = int(time.time())
    if expiry == 0:
        expiry_str = "Perpetual"
        expired    = False
    else:
        expiry_str = datetime.fromtimestamp(expiry, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        expired    = now > expiry

    print(f"""
  Key      : {key}
  Tier     : {info['tier']} — {TIER_NAMES.get(info['tier'], '?')}
  Expiry   : {expiry_str}{'  ⚠ EXPIRED' if expired else ''}
  Nonce    : {info['nonce']}
  Checksum : OK ✓
""")


def cmd_list(args):
    result = _admin_get("admin/keys")
    if "error" in result:
        print(f"\n  ✗ {result['error']}\n")
        sys.exit(1)

    keys = result.get("keys", [])
    print(f"\n  {'KEY':<28} {'TIER':<16} {'EXPIRY':<28} {'SEATS':>5}  STATUS")
    print("  " + "-" * 90)
    for k in keys:
        status = "REVOKED" if k["revoked"] else ("EXPIRED" if k["expired"] else "Active")
        print(f"  {k['key']:<28} {k['tier']:<16} {k['expiry']:<28} {k['activations']:>5}  {status}")
    print(f"\n  Total: {result.get('total', len(keys))} key(s)\n")


def cmd_revoke(args):
    key = args.key.strip()
    r = _admin_post("admin/keys/revoke", {"license_key": key})
    if r.get("success"):
        print(f"\n  ✓ Revoked: {key}\n")
    else:
        print(f"\n  ✗ {r.get('message', r.get('error', 'Unknown error'))}\n")
        sys.exit(1)


def cmd_unrevoke(args):
    key = args.key.strip()
    r = _admin_post("admin/keys/unrevoke", {"license_key": key})
    if r.get("success"):
        print(f"\n  ✓ Un-revoked: {key}\n")
    else:
        print(f"\n  ✗ {r.get('message', r.get('error', 'Unknown error'))}\n")
        sys.exit(1)


# ── Argument Parser ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CardioX License Key Admin CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate license key(s) locally")
    p_gen.add_argument("--tier",  type=int, default=1, choices=[0,1,2,3],
                       help="0=Trial 1=Standard 2=Pro 3=Enterprise")
    p_gen.add_argument("--days",  type=int, default=365,
                       help="Validity in days (0=perpetual)")
    p_gen.add_argument("--count", type=int, default=1,
                       help="Number of keys to generate")
    p_gen.add_argument("--output", metavar="FILE",
                       help="Append generated keys to this file")
    p_gen.set_defaults(func=cmd_generate)

    # inspect
    p_ins = sub.add_parser("inspect", help="Decode and verify a key locally")
    p_ins.add_argument("key", help="License key (e.g. ABCDE-FGHJK-LMNPQ-RSTUV)")
    p_ins.set_defaults(func=cmd_inspect)

    # list
    p_lst = sub.add_parser("list", help="List all keys from server")
    p_lst.set_defaults(func=cmd_list)

    # revoke
    p_rev = sub.add_parser("revoke", help="Revoke a key on the server")
    p_rev.add_argument("key", help="License key to revoke")
    p_rev.set_defaults(func=cmd_revoke)

    # unrevoke
    p_unr = sub.add_parser("unrevoke", help="Un-revoke a key on the server")
    p_unr.add_argument("key", help="License key to un-revoke")
    p_unr.set_defaults(func=cmd_unrevoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
