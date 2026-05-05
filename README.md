# 🫀 RhythmPro - Comprehensive ECG & Holter Analysis Platform

A clinical-grade, cross-platform desktop application for real-time ECG monitoring and advanced Holter analysis. Designed to process, classify, and report cardiac data with professional medical-grade accuracy.

---

## 🌟 Key Features

### 1. Advanced Holter Analysis Engine
- **Clinical-Grade Reporting:** Fully automated generation of comprehensive multi-page PDF reports mirroring professional medical standards.
- **Dynamic Full Disclosure (CH1):** Compact, multi-line raster rendering of long-term ECG recordings (up to 30 minutes of continuous waveform per page).
- **Asynchronous Processing:** Background PDF compilation using robust `QThread` architecture ensures the UI remains buttery smooth and responsive during heavy data processing.
- **Smart Auto-Scaling:** Intelligently scales report templates to accommodate short diagnostic tests without leaving blank pages.

### 2. Professional Medical Dashboard (3-Pane Layout)
- **High-Fidelity ECG Trace Viewer:** Sub-pixel waveform rendering using PyQtGraph with strict grid snapping and dynamic overlay pins.
- **Morphology Thumbnail Grid:** Side-by-side clustering of abnormal beats (VE, SVE) for rapid physician review.
- **Real-time Event List:** Absolute-time synchronization mapping annotations directly to the patient's physical recording timeline.
- **Dark Mode Clinical Aesthetics:** High-contrast, fatigue-reducing dark UI tailored for extensive clinical review sessions.

### 3. Medical-Grade Signal Processing
- **Pan-Tompkins R-Peak Detection:** Robust QRS complex detection and real-time beat annotation (N, V, S, AF).
- **Adaptive Filtering Pipeline:** 8-stage filtering system featuring Wiener filters, Gaussian smoothing, and adaptive median noise removal.
- **HRV Analytics:** Computes Time Domain parameters (SDNN, rMSSD, pNN50) and comprehensive interval statistics (PR, QRS, QT, QTc).

### 4. Robust Data Management & Cloud Integration
- **Zero-Collision Architecture:** Timestamp-appended filenames entirely prevent OS-level `PermissionError` conflicts during workflow multitasking.
- **Global Background Uploads:** Automatic cloud synchronization to AWS S3 every 15 seconds without interrupting the medical workflow.
- **Offline Queueing:** Robust caching system safely queues data offline and automatically uploads missing records the moment an internet connection is restored.

---

## 🛠️ Tech Stack
- **Core Engine:** Python 3.10
- **User Interface:** PyQt5, PyQtGraph (Hardware-accelerated rendering)
- **Signal Analysis:** NumPy, SciPy (FFT, digital filtering)
- **Report Generation:** ReportLab, Matplotlib (Agg background renderer)
- **Cloud Infrastructure:** Boto3 (AWS S3)

---

## 🚀 Quick Start (Developers)

```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r src\requirements.txt

# 3. Launch the application
python src\main.py
```

---

## 📦 Building the Executable

Use the provided build tools to create a standalone Windows application:

```powershell
# Build standard directory
python build_exe.py --name RhythmPro

# Build with debug console
python build_exe.py --name RhythmPro --console
```
This generates `dist\RhythmPro\RhythmPro.exe`. For deployment, zip and share the entire `RhythmPro` folder to ensure all `_internal` dependencies are included.

## 📝 Admin & Demo Configuration

**Admin Panel Access**
Access the internal cloud dashboard using credentials: `admin` / `adminsd`

**Demo Mode**
The application supports a hardware-free simulation mode. A static `recording.ecgh` dataset can be replayed to test UI responses and PDF generation without needing an active device connection.

---
**Status:** 🟢 Production Ready | **Last Updated:** May 2026
