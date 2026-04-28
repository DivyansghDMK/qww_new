"""
Build Windows setup.exe (Inno Setup) from an existing PyInstaller onedir build.

Usage:
    python build_setup.py
    python build_setup.py --name ECGMonitor --version 1.0.0
    python build_setup.py --iscc "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _find_iscc(explicit_path: str | None = None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        return p if p.exists() else None

    # Check PATH first
    from_path = shutil.which("ISCC.exe") or shutil.which("ISCC")
    if from_path:
        return Path(from_path).resolve()

    # Common install locations
    candidates = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _default_version() -> str:
    return datetime.now().strftime("%Y.%m.%d.%H%M")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ECGMonitor setup.exe with Inno Setup")
    parser.add_argument("--name", default="ECGMonitor", help="App name / exe name without extension")
    parser.add_argument("--version", default=_default_version(), help="Installer version label")
    parser.add_argument("--channel", default="stable", help="Update channel label")
    parser.add_argument("--repository", default="", help="GitHub repository in owner/name form")
    parser.add_argument("--publisher", default="Deckmount Electronics", help="Publisher name")
    parser.add_argument("--dist-dir", default="", help="Path to PyInstaller onedir folder")
    parser.add_argument("--output-dir", default="", help="Path where setup exe should be written")
    parser.add_argument("--iscc", default="", help="Path to ISCC.exe (optional)")
    ns = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dist_dir = Path(ns.dist_dir).resolve() if ns.dist_dir else (project_root / "dist" / ns.name).resolve()
    output_dir = Path(ns.output_dir).resolve() if ns.output_dir else (project_root / "dist" / "installers").resolve()
    iss_file = (project_root / "installer" / "ECGMonitor.iss").resolve()

    if not iss_file.exists():
        print(f"Installer script not found: {iss_file}")
        return 1
    if not dist_dir.exists():
        print(f"Build output not found: {dist_dir}")
        print("Run: python build_exe.py --name ECGMonitor")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    iscc = _find_iscc(ns.iscc or None)
    if not iscc:
        print("Inno Setup compiler (ISCC.exe) not found.")
        print("Install Inno Setup 6, then run this command again.")
        print(f"Expected script ready at: {iss_file}")
        return 1

    defines = [
        f"/DMyAppName={ns.name}",
        f"/DMyAppVersion={ns.version}",
        f"/DMyAppChannel={ns.channel}",
        f"/DMyAppRepository={ns.repository}",
        f"/DMyAppURL={'https://github.com/' + ns.repository + '/releases' if ns.repository else 'https://example.com'}",
        f"/DMyAppPublisher={ns.publisher}",
        f"/DMyAppExeName={ns.name}.exe",
        f"/DMyAppDistDir={str(dist_dir)}",
        f"/DMyAppOutputDir={str(output_dir)}",
    ]
    cmd = [str(iscc), *defines, str(iss_file)]

    print("=" * 70)
    print("Building setup.exe")
    print(f"ISCC       : {iscc}")
    print(f"Dist dir   : {dist_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Version    : {ns.version}")
    print(f"Channel    : {ns.channel}")
    print(f"Repository : {ns.repository or '(none)'}")
    print("=" * 70)

    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"Failed to run ISCC: {exc}")
        return 1

    if proc.returncode != 0:
        print(f"ISCC failed with exit code {proc.returncode}")
        return proc.returncode

    print("Setup build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

