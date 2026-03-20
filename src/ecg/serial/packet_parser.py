"""ECG packet parsing utilities"""
import os
import re
from typing import Dict, Tuple, Optional

# ── Packet structure constants ────────────────────────────────────────────────
# Hardware sends: START(1) + COUNTER(1) + FLAGS(2) + LEADS(8×2) + CRC(1) + END(1) = 22 bytes
PACKET_SIZE = 22
START_BYTE = 0xE8
END_BYTE = 0x8E
LEAD_NAMES_DIRECT = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
PACKET_REGEX = re.compile(r"(?i)(E8(?:[0-9A-F\s]{2,})?8E)")

_DEBUG_PACKETS = os.getenv("ECG_DEBUG_PACKETS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def hex_string_to_bytes(hex_str: str) -> bytes:
    """Convert hex string to bytes"""
    cleaned = re.sub(r"[^0-9A-Fa-f]", "", hex_str)
    if len(cleaned) % 2 != 0:
        raise ValueError("Hex string must have even length")
    return bytes(int(cleaned[i : i + 2], 16) for i in range(0, len(cleaned), 2))


def decode_lead(msb: int, lsb: int) -> Tuple[int, bool]:
    """Decode lead value from MSB and LSB bytes"""
    lower7 = lsb & 0x7F
    upper5 = msb & 0x1F
    value = (upper5 << 7) | lower7
    connected = (msb & 0x20) != 0
    return value, connected


def parse_packet(raw: bytes) -> Dict[str, Optional[int]]:
    """
    Parse ECG packet and return dictionary of lead values.

    BUG-19 FIX: Returns None for disconnected leads instead of plotting electrode noise.
    BUG-15 FIX: aVL and aVF now use correct Goldberger formulas.

    Returns:
        dict mapping lead name → int (ADC value) or None (lead disconnected)
    """
    if len(raw) != PACKET_SIZE or raw[0] != START_BYTE or raw[-1] != END_BYTE:
        return {}

    # Extract packet counter (byte 1) - sequence number 0-63
    packet_counter = raw[1] & 0x3F  # Counter is in lower 6 bits (0-63)

    lead_values: Dict[str, Optional[int]] = {}
    idx = 5  # first MSB position

    if _DEBUG_PACKETS:
        print(f"---- New Packet (Counter: {packet_counter}) ----")

    for name in LEAD_NAMES_DIRECT:
        msb = raw[idx]
        lsb = raw[idx + 1]
        idx += 2

        value, connected = decode_lead(msb, lsb)

        if _DEBUG_PACKETS:
            print(f"{name}: MSB={msb:02X}, LSB={lsb:02X}, value={value}, connected={connected}")

        # BUG-19 FIX: respect the connected flag.
        # When connected=False, electrode is off → return None so display shows "LEAD OFF"
        # instead of plotting garbage ADC noise as an ECG waveform.
        if connected:
            lead_values[name] = value
        else:
            lead_values[name] = None  # Caller must handle None (show flat line / LEAD OFF indicator)

    # Derived limb leads — only calculate when source leads are connected
    lead_i  = lead_values.get("I")
    lead_ii = lead_values.get("II")

    if lead_i is not None and lead_ii is not None:
        # ── BUG-15 FIX: Correct Goldberger/Einthoven formulas ────────────────
        # OLD (wrong): aVL = (I - III) / 2,  aVF = (II + III) / 2
        # NEW (correct Goldberger):
        lead_iii = lead_ii - lead_i                    # Einthoven's law ✅
        avr      = -(lead_i + lead_ii) / 2             # Goldberger ✅
        avl      = lead_i  - lead_ii / 2               # Goldberger ✅ (was wrong)
        avf      = lead_ii - lead_i  / 2               # Goldberger ✅ (was wrong)

        lead_values["III"] = int(round(lead_iii))
        lead_values["aVR"] = int(round(avr))
        lead_values["aVL"] = int(round(avl))
        lead_values["aVF"] = int(round(avf))
    else:
        # If source limb leads are disconnected, derived leads are also invalid
        lead_values["III"] = None
        lead_values["aVR"] = None
        lead_values["aVL"] = None
        lead_values["aVF"] = None

    if _DEBUG_PACKETS:
        print("Derived:", {
            "III": lead_values.get("III"),
            "aVR": lead_values.get("aVR"),
            "aVL": lead_values.get("aVL"),
            "aVF": lead_values.get("aVF"),
        })
        print("---------------------\n")

    return lead_values
