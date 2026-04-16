"""
Logging configuration for ECG application
"""

import logging
import logging.handlers
import os
from typing import Optional
from .constants import LOG_FILE, LOG_ROTATION_SIZE


class ECGLogger:
    """Centralized logging for ECG application"""
    
    def __init__(self, name: str = "ECGApp", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.log_file = log_file or LOG_FILE
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with file and console handlers"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set logging level
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=LOG_ROTATION_SIZE,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            self.logger.warning(f"Could not setup file logging: {e}")
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str) -> None:
        """Log exception with traceback"""
        self.logger.exception(message)


# Global logger instance
app_logger = ECGLogger()


def get_logger(name: str = "ECGApp") -> ECGLogger:
    """Get logger instance"""
    return ECGLogger(name)


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        app_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            app_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            app_logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_ecg_metrics(metrics: dict) -> None:
    """Log ECG metrics for debugging"""
    app_logger.info("ECG Metrics:")
    for key, value in metrics.items():
        app_logger.info(f"  {key}: {value}")


def log_performance_stats(stats: dict) -> None:
    """Log performance statistics"""
    app_logger.info("Performance Stats:")
    for key, value in stats.items():
        app_logger.info(f"  {key}: {value}")
