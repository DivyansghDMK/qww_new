"""
Deterministic Windows build script for ECG Monitor executable.

Goals:
Produce consistent output across machines.
Default to ONEDIR packaging (more stable than onefile for this app).
Include required runtime files/assets explicitly.

Usage:
    python build_exe.py
    python build_exe.py --onefile        # optional, less stable for this app
    python build_exe.py --console        # debug startup crashes
    python build_exe.py --name ECGMonitor
    python build_exe.py --onefile --runtime-tmpdir "C:\\Temp\\ECGMonitor"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import PyInstaller.__main__


def _encode_stdio_utf8_on_windows() -> None:
    if sys.platform != "win32":
        return
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _data_sep() -> str:
    return ";" if os.name == "nt" else ":"


def _stage_runtime_file(src: Path, staging_dir: Path, dst_name: str | None = None) -> Path | None:
    """Copy a runtime file into a staging dir so build steps never mutate the repo."""
    if not src.exists():
        return None
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / (dst_name or src.name)
    try:
        shutil.copy2(src, target)
    except Exception:
        return None
    return target


def _stage_json_placeholder(staging_dir: Path, dst_name: str) -> Path:
    """Create a minimal JSON file in the staging dir for missing optional runtime data."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / dst_name
    if not target.exists():
        try:
            target.write_text("{}", encoding="utf-8")
        except Exception:
            pass
    return target


_DEFAULT_DIST_LICENSE_SERVER_URL = (
    "https://m4qoae4d8e.execute-api.us-east-1.amazonaws.com/prod/api/v1"
)


def _stage_env_for_distribution(src: Path, staging_dir: Path) -> Path | None:
    """
    Stage a .env file for distribution.

    Why:
    - The repo's `.env` commonly points LICENSE_SERVER_URL to localhost for dev.
    - Bundling that into the EXE makes activation fail on other PCs (WinError 10061).

    Behavior:
    - Copy `.env` into staging, but rewrite LICENSE_SERVER_URL if it points to localhost/127.*.
    - You can override the distributed URL via BUILD_LICENSE_SERVER_URL at build time.
    """
    if not src.exists():
        return None

    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / src.name

    try:
        raw = src.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return _stage_runtime_file(src, staging_dir)

    dist_url = os.getenv("BUILD_LICENSE_SERVER_URL", "").strip() or _DEFAULT_DIST_LICENSE_SERVER_URL

    out_lines: list[str] = []
    replaced = False
    for line in raw:
        stripped = line.strip()
        if stripped.startswith("LICENSE_SERVER_URL="):
            value = stripped.split("=", 1)[1].strip()
            lower = value.lower()
            if lower.startswith("http://127.0.0.1") or lower.startswith("http://localhost"):
                out_lines.append(
                    f"LICENSE_SERVER_URL={dist_url}"
                )
                replaced = True
                continue
        out_lines.append(line)

    if not replaced:
        # Ensure the key exists for consistent behavior in frozen apps.
        out_lines.append(f"\nLICENSE_SERVER_URL={dist_url}")

    try:
        target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    except Exception:
        return _stage_runtime_file(src, staging_dir)
    return target


def _add_data_args(project_root: Path, build_root: Path) -> list[str]:
    sep = _data_sep()
    staging_dir = build_root / "_staging"
    pairs: list[tuple[Path, str]] = []

    # Assets used by PDF/report UI
    pairs.append((project_root / "assets", "assets"))

    # Runtime config/demo files often expected in working directory
    for filename in [
        ".env",
        "customer_channels.json",
        "dummycsv.csv",
        "users.json",
        "ecg_settings.json",
        "last_conclusions.json",
        "ecg_auth_session.json",
    ]:
        file_path = project_root / filename
        if filename == ".env":
            staged = _stage_env_for_distribution(file_path, staging_dir)
        else:
            staged = _stage_runtime_file(file_path, staging_dir)
        if staged is None and filename.endswith(".json"):
            staged = _stage_json_placeholder(staging_dir, filename)
        if staged is not None:
            pairs.append((staged, "."))

    # Some deployments use src-side settings files as fallbacks
    for filename in [
        "src/ecg_settings.json",
        "src/users.json",
        "src/ecg_auth_session.json",
    ]:
        file_path = project_root / filename
        staged = _stage_runtime_file(file_path, staging_dir, Path(filename).name)
        if staged is None:
            staged = _stage_json_placeholder(staging_dir, Path(filename).name)
        if staged is not None:
            pairs.append((staged, "src"))

    args: list[str] = []
    for src, dst in pairs:
        if src.exists():
            args.extend(["--add-data", f"{src}{sep}{dst}"])
    return args


def build_args(project_root: Path, name: str, onefile: bool, console: bool) -> list[str]:
    main_script = project_root / "src" / "main.py"
    if not main_script.exists():
        raise FileNotFoundError(f"Main script not found: {main_script}")

    build_dir = project_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = project_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    args: list[str] = [
        str(main_script),
        f"--name={name}",
        "--noconfirm",
        "--clean",
        "--windowed" if not console else "--console",
        "--onedir" if not onefile else "--onefile",
        # FIX: src/ on path so PyInstaller finds organization.py and all src modules
        f"--paths={project_root / 'src'}",
        # FIX: write spec to build/ so stale local spec is never used on CI
        f"--specpath={build_dir}",
        f"--workpath={build_dir / 'work'}",
        f"--distpath={dist_dir}",
        # Imported dynamically via importlib in src/main.py
        "--hidden-import=organization",
        # `importlib` is stdlib, but include explicitly for deterministic PyInstaller analysis.
        "--hidden-import=importlib",
        "--hidden-import=serial",
        "--hidden-import=serial.tools.list_ports",
        "--hidden-import=serial.tools.list_ports.windows",
        "--hidden-import=serial.tools.list_ports.posix",
        # ECG modules imported dynamically
        "--hidden-import=ecg.twelve_lead_test",
        "--hidden-import=ecg.expanded_lead_view",
        "--hidden-import=ecg.hrv_test",
        "--hidden-import=ecg.hyperkalemia_test",
        "--hidden-import=ecg.ecg_report_generator",
        "--hidden-import=ecg.clinical_measurements",
        "--hidden-import=ecg.arrhythmia_detector",
        "--hidden-import=ecg.pan_tompkins",
        "--hidden-import=ecg.recording",
        "--hidden-import=ecg.demo_manager",
        "--hidden-import=ecg.signal_paths",
        "--hidden-import=ecg.ecg_calculations",
        "--hidden-import=ecg.ecg_filters",
        "--hidden-import=ecg.clinical_validation",
        "--collect-submodules=ecg",
        "--collect-submodules=ecg.serial",
        "--collect-submodules=ecg.signal",
        "--collect-submodules=ecg.metrics",
        "--collect-submodules=ecg.plotting",
        "--collect-submodules=ecg.ui",
        "--collect-submodules=ecg.utils",
        "--collect-submodules=ecg.holter",
        "--collect-submodules=ecg.arrhythmia_engine",
        "--collect-submodules=dashboard",
        "--collect-submodules=utils",
        "--collect-submodules=auth",
        "--collect-submodules=config",
        "--collect-submodules=core",
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.sip",
        "--hidden-import=fitz",
        "--hidden-import=pymupdf",
        "--collect-submodules=PyQt5",
        "--collect-all=pyqtgraph",
        "--exclude-module=PyQt6",
        "--exclude-module=PySide2",
        "--exclude-module=PySide6",
        "--exclude-module=notebook",
        "--exclude-module=jinja2",
        "--exclude-module=IPython",
    ]
    args.extend(_add_data_args(project_root, build_dir))
    return args


def main() -> int:
    _encode_stdio_utf8_on_windows()

    parser = argparse.ArgumentParser(description="Build ECG Monitor executable with PyInstaller")
    parser.add_argument("--name", default="ECGMonitor", help="Output application name")
    parser.add_argument("--onefile", action="store_true", help="Build as onefile (not recommended)")
    parser.add_argument("--console", action="store_true", help="Build with console for debugging")
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Request UAC admin privileges on launch (adds PyInstaller --uac-admin; Windows only).",
    )
    parser.add_argument(
        "--runtime-tmpdir",
        default=None,
        help=(
            "Where PyInstaller onefile extracts runtime files. "
            "Example (Windows): TEMP\\ECGMonitor (use your TEMP environment variable when running the command). "
            "If omitted, PyInstaller default is used."
        ),
    )
    ns = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    args = build_args(project_root, ns.name, ns.onefile, ns.console)

    # Only relevant for onefile builds. If not set, PyInstaller will use its default temp location.
    if ns.onefile and ns.runtime_tmpdir:
        runtime_tmpdir_raw = ns.runtime_tmpdir.strip()
        # IMPORTANT:
        # - PyInstaller embeds --runtime-tmpdir into the executable at build time.
        # - Using %TEMP% / $env:TEMP would get expanded on the build machine and break on other PCs.
        if "%" in runtime_tmpdir_raw or "$env:" in runtime_tmpdir_raw or "$" in runtime_tmpdir_raw:
            raise SystemExit(
                "ERROR: --runtime-tmpdir must be an absolute path and must not contain environment variables "
                "(e.g. %TEMP% or $env:TEMP). Omit --runtime-tmpdir to use the per-user temp directory on each PC."
            )
        runtime_tmpdir = Path(runtime_tmpdir_raw)
        if not runtime_tmpdir.is_absolute():
            raise SystemExit(
                "ERROR: --runtime-tmpdir must be an absolute path (e.g. C:\\Temp\\ECGMonitor). "
                "Omit --runtime-tmpdir to use the per-user temp directory on each PC."
            )
        args.insert(5, f"--runtime-tmpdir={runtime_tmpdir}")

    print("=" * 70)
    print("Building ECG Monitor executable")
    print(f"Project root : {project_root}")
    print(f"Mode         : {'onefile' if ns.onefile else 'onedir (recommended)'}")
    print(f"Console      : {'ON' if ns.console else 'OFF'}")
    print(f"Admin        : {'ON' if ns.admin else 'OFF'}")
    print("=" * 70)
    print("PyInstaller args:")
    for a in args:
        print(f"  {a}")

    try:
        if ns.admin:
            args.append("--uac-admin")
        PyInstaller.__main__.run(args)
    except Exception as exc:
        print(f"\nBuild failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    if ns.onefile:
        print(f"\nBuild complete: dist/{ns.name}.exe")
    else:
        print(f"\nBuild complete: dist/{ns.name}/{ns.name}.exe")
        print("Distribute the entire dist folder for this app (not exe alone).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
