# ═══════════════════════════════════════════════════════════════════════════════
# BUILD COMMANDS — ECGMonitor EXE (STANDARDIZED)
# Run these in PowerShell from your project root.
# This flow uses build_exe.py (default: ONEDIR) for better cross-system stability.
# ═══════════════════════════════════════════════════════════════════════════════

# STEP 0: Go to project root
cd C:\Users\DELL\Downloads\dfg\merge

# STEP 1: Activate venv
.\.venv\Scripts\Activate.ps1

# STEP 2: Install pinned build dependencies (recommended)
# pip install -r src\requirements.txt

# STEP 3: Clean old artifacts
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist  -ErrorAction SilentlyContinue

# STEP 4: Build (ONEDIR recommended for consistent runtime)
python build_exe.py --name ECGMonitor

# STEP 5: Verify critical runtime files in output
Test-Path "dist\ECGMonitor\ECGMonitor.exe"
Test-Path "dist\ECGMonitor\_internal\assets\Deckmountimg.png"

# STEP 6: Run
.\dist\ECGMonitor\ECGMonitor.exe

# ───────────────────────────────────────────────────────────────────────────────
# DEBUG BUILD (if app closes immediately)
# python build_exe.py --name ECGMonitor --console
# Then run from PowerShell to see traceback:
# .\dist\ECGMonitor\ECGMonitor.exe
# ───────────────────────────────────────────────────────────────────────────────

# DISTRIBUTION RULE
# Zip/share entire folder: dist\ECGMonitor\
# Do NOT send only the EXE for onedir builds.

# COMMON ISSUES
# 1) ModuleNotFoundError:
#    - Build from clean venv
#    - Install pinned deps: pip install -r src\requirements.txt
#
# 2) Works on one PC, fails on another:
#    - Use same Python major/minor on builder machine
#    - Rebuild with --clean
#    - Share complete dist\ECGMonitor\ folder
#
# 3) COM port mismatch on target PC:
#    - ECG settings may be fixed to a specific COM (e.g., COM8)
#    - Select correct COM on target machine or use auto-detect in app
