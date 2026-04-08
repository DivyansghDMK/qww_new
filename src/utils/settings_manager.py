import json
import os
import re

class SettingsManager:
    def __init__(self):
        self.settings_file = "ecg_settings.json"
        self.default_settings = {
            "wave_speed": "25",  # mm/s (default)
            "wave_gain": "10",   # mm/mV
            "lead_sequence": "Standard",
            "serial_port": "Select Port",
            "baud_rate": "115200",
            "hardware_version": "",

            # Report Setup settings
            "printer_average_wave": "on",
            "lead_sequence": "Standard",

            # Filter settings
            "filter_ac": "50",
            "filter_emg": "150",
            "filter_dft": "0.5",

            # System Setup settings
            "system_beat_vol": "off",
            "system_language": "en",

            # Factory Maintain settings
            "factory_calibration": "skip",
            "factory_self_test": "skip",
            "factory_memory_reset": "keep",
            "factory_reset": "cancel"
        }
        self.settings = self.load_settings()

    def _normalize_filter_value(self, key, value):
        """
        Normalize persisted filter settings so all consumers receive canonical values.
        Handles legacy/free-form values like "50 hz", "50Hz", etc.
        """
        if value is None:
            return value

        text = str(value).strip().lower()
        if key == "filter_ac":
            if text in ("off", "", "none", "0"):
                return "off"
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                hz = match.group(1)
                if hz in ("50", "50.0"):
                    return "50"
                if hz in ("60", "60.0"):
                    return "60"
            return "off"

        if key == "filter_emg":
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                hz = match.group(1).rstrip("0").rstrip(".")
                if hz in {"25", "35", "40", "75", "100", "150"}:
                    return hz
            return self.default_settings.get("filter_emg", "150") if hasattr(self, "default_settings") else "150"

        if key == "filter_dft":
            if text in ("off", "", "none", "0"):
                return "off"
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            if match:
                val = float(match.group(1))
                if abs(val - 0.05) < 1e-6:
                    return "0.05"
                if abs(val - 0.5) < 1e-6:
                    return "0.5"
            return self.default_settings.get("filter_dft", "0.5") if hasattr(self, "default_settings") else "0.5"

        return value
    
    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    
                    merged_settings = self.default_settings.copy()
                    merged_settings.update(loaded_settings)
                    return merged_settings
            except:
                return self.default_settings.copy()
        return self.default_settings.copy()
    
    def save_settings(self):
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def get_setting(self, key, default=None):
        value = self.settings.get(key, self.default_settings.get(key, default))
        if key in {"filter_ac", "filter_emg", "filter_dft"}:
            return self._normalize_filter_value(key, value)
        return value
    
    def set_setting(self, key, value):
        if key in {"filter_ac", "filter_emg", "filter_dft"}:
            value = self._normalize_filter_value(key, value)
        self.settings[key] = value
        self.save_settings()
        print(f"Setting updated: {key} = {value}")  # Terminal verification

    def reset_to_defaults(self):
        """Restore every persisted setting to its original factory default, preserving hardware version."""
        current_hw_version = self.settings.get("hardware_version", "")
        self.settings = self.default_settings.copy()
        self.settings["hardware_version"] = current_hw_version
        self.save_settings()
        return self.settings.copy()
    
    def get_wave_speed(self):
        return float(self.get_setting("wave_speed"))
    
    def get_wave_gain(self):
        return float(self.get_setting("wave_gain"))

    def get_serial_port(self):
        return self.get_setting("serial_port")
    
    def get_baud_rate(self):
        return self.get_setting("baud_rate")
    
    def set_serial_port(self, port):
        self.set_setting("serial_port", port)
    
    def set_baud_rate(self, baud_rate):
        self.set_setting("baud_rate", baud_rate)

    def get_calibration_notch_boxes(self):
        """Calculate calibration notch boxes based on wave gain"""
        wave_gain = self.get_wave_gain()
        if wave_gain == 20:
            return 4.0
        elif wave_gain == 10:
            return 2.0
        elif wave_gain == 5:
            return 1.0
        elif wave_gain == 2.5:
            return 0.5
        else:
            # Default calculation for other values
            return wave_gain / 5.0  # 5mm = 1 box baseline
