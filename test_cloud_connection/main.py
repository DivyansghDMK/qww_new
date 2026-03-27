#!/usr/bin/env python3
"""
ECG Monitor Application Launcher
This script launches the ECG Monitor application from the root directory.
"""

import sys
import os

# Fix encoding for Windows PowerShell (cp1252 can't handle emojis)
if sys.platform == 'win32':
    import io
    try:
        if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
    try:
        if sys.stderr is not None and hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add the src directory to the Python path
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Change to the src directory
try:
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
except Exception:
    pass

# Import and run the main application
try:
    from main import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f" Error importing main application: {e}")
    print(" Make sure you're running from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f" Error running application: {e}")
    sys.exit(1)
