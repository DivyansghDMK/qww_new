"""
Application settings and configuration management
"""

import os
import json
from typing import Dict, Any, Optional


class AppConfig:
    """Centralized configuration management for the ECG application"""
    
    def __init__(self, config_file: str = "ecg_settings.json"):
        self.config_file = config_file
        self._config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            "ecg": {
                "sampling_rate": 80,
                "buffer_size": 1000,
                "update_interval": 50,  # ms
                "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "filtering": {
                    "bandpass_low": 0.5,
                    "bandpass_high": 40.0,
                    "notch_frequency": 50.0,
                    "smoothing_enabled": True
                }
            },
            "ui": {
                "theme": "default",
                "font_family": "Arial",
                "font_size": 12,
                "window_size": [1200, 800],
                "dashboard_refresh_rate": 20  # Hz
            },
            "hardware": {
                "serial_port": "COM3",
                "baud_rate": 9600,
                "timeout": 1.0,
                "max_readings_per_cycle": 20
            },
            "audio": {
                "heartbeat_enabled": True,
                "volume": 100,
                "sound_file": "heartbeat.wav"
            },
            "logging": {
                "level": "INFO",
                "file": "ecg_app.log",
                "max_size": "10MB",
                "backup_count": 5
            }
        }
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _merge_config(self, file_config: Dict[str, Any]) -> None:
        """Merge file configuration with default configuration"""
        def merge_dict(default: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dict(default[key], value)
                else:
                    default[key] = value
            return default
        
        self._config = merge_dict(self._config, file_config)
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving config file: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'ecg.sampling_rate')"""
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_ecg_config(self) -> Dict[str, Any]:
        """Get ECG-specific configuration"""
        return self.get('ecg', {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-specific configuration"""
        return self.get('ui', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-specific configuration"""
        return self.get('hardware', {})
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio-specific configuration"""
        return self.get('audio', {})


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and PyInstaller"""
    if hasattr(os.sys, '_MEIPASS'):
        return os.path.join(os.sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
