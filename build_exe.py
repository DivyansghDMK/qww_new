"""
Build script for creating ECG Monitor executable
Run: python build_exe.py
"""

import PyInstaller.__main__
import os
import sys

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
assets_dir = os.path.join(project_root, 'assets')

# Main script path
main_script = os.path.join(src_dir, 'main.py')

# Collect all asset files
asset_files = []
if os.path.exists(assets_dir):
    for root, dirs, files in os.walk(assets_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, project_root)
            asset_files.append((full_path, 'assets'))

# PyInstaller arguments
args = [
    main_script,
    '--name=ECG_Monitor',
    '--onefile',  # Create a single executable file
    '--windowed',  # No console window (GUI app)
    '--icon=NONE',  # Add icon path if you have one
    '--add-data', f'{assets_dir};assets',  # Include assets folder
    '--hidden-import=PyQt5',
    '--hidden-import=PyQt5.QtCore',
    '--hidden-import=PyQt5.QtGui',
    '--hidden-import=PyQt5.QtWidgets',
    '--hidden-import=numpy',
    '--hidden-import=scipy',
    '--hidden-import=scipy.signal',
    '--hidden-import=matplotlib',
    '--hidden-import=pyqtgraph',
    '--hidden-import=serial',
    '--hidden-import=pandas',
    '--hidden-import=PIL',
    '--hidden-import=reportlab',
    '--hidden-import=boto3',
    '--hidden-import=psutil',
    '--hidden-import=pyparsing',
    '--collect-all=PyQt5',
    '--collect-all=matplotlib',
    '--collect-all=scipy',
    '--collect-all=pyqtgraph',
    '--noconfirm',  # Overwrite output directory without asking
    '--clean',  # Clean cache before building
]

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("Building ECG Monitor executable...")
print(f"Project root: {project_root}")
print(f"Main script: {main_script}")
print(f"Assets directory: {assets_dir}")

try:
    PyInstaller.__main__.run(args)
    print("\nBuild complete! Executable should be in the 'dist' folder.")
    print("Location: dist/ECG_Monitor.exe")
except Exception as e:
    print(f"\nBuild failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

