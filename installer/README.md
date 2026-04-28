# Installer (Windows Setup EXE)

This folder contains an **Inno Setup** installer script that creates a standard Windows `Setup_*.exe` for the app.

## Prerequisites

- Build the app with the recommended PyInstaller **ONEDIR** output:
  - `python build_exe.py --name ECGMonitor`
- Install **Inno Setup** (so you have `ISCC.exe` available).

## Build the installer

From the project root in PowerShell:

```powershell
python build_exe.py --name ECGMonitor
.\installer\build_installer.ps1
```

The setup EXE is written to:

- `dist_installer\`

## What gets installed

- Everything under `dist\ECGMonitor\` is installed to `C:\Program Files\ECG Monitor\` by default.
- Start Menu shortcut (and optional Desktop shortcut) is created with working directory set to the install folder.

