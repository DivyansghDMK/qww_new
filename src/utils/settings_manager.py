from email.policy import default
import json
import os

class SettingsManager:
    def __init__(self):
        self.settings_file = "ecg_settings.json"
        self.default_settings = {
            "wave_speed": "25",  # mm/s (default)
            "wave_gain": "10",   # mm/mV
            "lead_sequence": "Standard",
            "serial_port": "Select Port",
            "baud_rate": "115200",

            # Report Setup settings
            "printer_average_wave": "on",

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

        # Force EMG filter to 150Hz on every application start
        # This overrides any saved setting from previous sessions.
        self.settings["filter_emg"] = "150"
    
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
        return self.settings.get(key, self.default_settings.get(key, default))
    
    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()
        print(f"Setting updated: {key} = {value}")  # Terminal verification

    def reset_to_defaults(self):
        """Restore every persisted setting to its original factory default."""
        self.settings = self.default_settings.copy()
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
