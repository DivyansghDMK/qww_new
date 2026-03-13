"""ECG packet parsing utilities"""
import os
import re
from typing import Dict, Tuple

# Packet parsing constants
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


def parse_packet(raw: bytes) -> Dict[str, int]:
    """Parse ECG packet and return dictionary of lead values"""
    if len(raw) != PACKET_SIZE or raw[0] != START_BYTE or raw[-1] != END_BYTE:
        return {}

    # Extract packet counter (byte 1) - sequence number 0-63
    packet_counter = raw[1] & 0x3F  # Counter is in lower 6 bits (0-63)
    
    lead_values: Dict[str, int] = {}
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

        lead_values[name] = value

    # Derived limb leads
    lead_i = lead_values.get("I", 0)
    lead_ii = lead_values.get("II", 0)

    lead_values["III"] = lead_ii - lead_i
    lead_values["aVR"] = -(lead_i + lead_ii) / 2
    lead_values["aVL"] = (lead_i - lead_values["III"]) / 2
    lead_values["aVF"] = (lead_ii + lead_values["III"]) / 2

    if _DEBUG_PACKETS:
        print("Derived:", {
            "III": lead_values["III"],
            "aVR": lead_values["aVR"],
            "aVL": lead_values["aVL"],
            "aVF": lead_values["aVF"],
        })

        print("---------------------\n")

    return lead_values
