# ═══════════════════════════════════════════════════════════════════════════════
# BUILD COMMANDS — ECGMonitor EXE
# Run these in PowerShell from your project root:
#   C:\Users\DELL\Downloads\dfg\merge>
# ═══════════════════════════════════════════════════════════════════════════════


# ─── STEP 0: Make sure you are in the right folder ────────────────────────────
cd C:\Users\DELL\Downloads\dfg\merge


# ─── STEP 1: Activate your virtual environment ────────────────────────────────
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks scripts, run this first (once):
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


# ─── STEP 2: Copy the spec file into your project root ────────────────────────
# (Copy ECGMonitor.spec from this file into C:\Users\DELL\Downloads\dfg\merge\)


# ─── STEP 3: Clean old build artifacts ───────────────────────────────────────
Remove-Item -Recurse -Force build   -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist    -ErrorAction SilentlyContinue


# ─── STEP 4: BUILD THE EXE ───────────────────────────────────────────────────
pyinstaller ECGMonitor.spec --clean --noconfirm

# What this does:
#   --clean         deletes PyInstaller cache before building (fresh build)
#   --noconfirm     don't ask "overwrite dist/?" — just do it


# ─── STEP 5: Verify dummycsv.csv is in the output ────────────────────────────
Test-Path "dist\ECGMonitor\dummycsv.csv"
# Should print: True
# If False → manually copy: Copy-Item dummycsv.csv dist\ECGMonitor\


# ─── STEP 6: Run the EXE to test ─────────────────────────────────────────────
.\dist\ECGMonitor\ECGMonitor.exe


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ONE-LINER (paste this whole block and run) 
# ═══════════════════════════════════════════════════════════════════════════════
cd C:\Users\DELL\Downloads\dfg\merge; .\.venv\Scripts\Activate.ps1; Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue; pyinstaller ECGMonitor.spec --clean --noconfirm; if (Test-Path "dist\ECGMonitor\ECGMonitor.exe") { Write-Host "BUILD SUCCESS" -ForegroundColor Green; .\dist\ECGMonitor\ECGMonitor.exe } else { Write-Host "BUILD FAILED" -ForegroundColor Red }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: Install UPX for smaller EXE (reduces size by ~30%)
# Download from: https://github.com/upx/upx/releases
# Extract upx.exe to C:\upx\ and add to PATH, OR place upx.exe in project root
# Then rebuild — PyInstaller picks it up automatically
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# WHERE OUTPUT GOES
# ═══════════════════════════════════════════════════════════════════════════════
#
#   dist\
#   └── ECGMonitor\                  ← share THIS entire folder
#       ├── ECGMonitor.exe           ← double-click to launch
#       ├── dummycsv.csv             ← demo ECG data (bundled)
#       ├── users.json               ← user accounts
#       ├── ecg_settings.json        ← settings
#       ├── assets\                  ← logo images for PDF reports
#       │   ├── Deckmountimg.png
#       │   ├── v.gif
#       │   └── v1.png
#       ├── ecg\                     ← ECG processing modules
#       ├── dashboard\               ← UI modules
#       ├── PyQt5\                   ← Qt libraries
#       ├── scipy\                   ← signal processing
#       └── [other DLLs...]
#
#   To distribute: ZIP the entire dist\ECGMonitor\ folder
#   Recipient just extracts and double-clicks ECGMonitor.exe — no install needed


# ═══════════════════════════════════════════════════════════════════════════════
# IF YOU GET ERRORS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Error: "ModuleNotFoundError: No module named 'X'" when running EXE
#   → Add 'X' to hiddenimports in ECGMonitor.spec, rebuild
#
# Error: "dummycsv.csv not found"
#   → Run: Copy-Item dummycsv.csv dist\ECGMonitor\
#
# Error: "assets/Deckmountimg.png not found" in PDF
#   → Check assets\ folder exists at project root with the PNG files
#   → Run: Copy-Item -Recurse assets dist\ECGMonitor\
#
# Error: "Failed to execute script main" (onefile crash)
#   → You are using onefile — switch to onedir (already done in this spec)
#
# Black console window appears behind app
#   → console=False is already set in spec — make sure you're using THIS spec
#
# App works in Python but crashes in EXE
#   → Run with console=True temporarily to see the error:
#      Change console=False to console=True in spec, rebuild, run from PowerShell
