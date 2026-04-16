import sys
from typing import Any, TextIO


def format_ecg_data(ecg_data):
    # Function to format ECG data for display
    return [round(value, 2) for value in ecg_data]


def validate_user_input(input_data, expected_type):
    # Function to validate user input
    if not isinstance(input_data, expected_type):
        raise ValueError(f"Input must be of type {expected_type.__name__}")
    return True


def calculate_average(values):
    # Function to calculate the average of a list of values
    if not values:
        return 0
    return sum(values) / len(values)


def safe_print(*args: Any, sep: str = " ", end: str = "\n", file: TextIO | None = None, flush: bool = False) -> None:
    """
    Cross-platform print helper that strips or replaces characters the active console
    encoding cannot represent (e.g., emoji on legacy Windows code pages).
    """
    stream = file if file is not None else getattr(sys, "stdout", None)
    if stream is None:
        return

    encoding = getattr(stream, "encoding", None) or "utf-8"
    text = sep.join(str(arg) for arg in args)

    def _write(msg: str) -> None:
        stream.write(msg + end)

    try:
        _write(text)
    except UnicodeEncodeError:
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        _write(sanitized)
    except Exception:
        return
    finally:
        if flush:
            try:
                stream.flush()
            except Exception:
                pass


# Automatically replace the built-in print with safe_print once this module is imported.
try:
    import builtins

    builtins.print = safe_print
except Exception:
    pass
