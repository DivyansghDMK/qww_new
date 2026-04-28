from __future__ import annotations

import os
import re
import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any

import requests
from PyQt5.QtWidgets import QMessageBox


def _load_version_metadata() -> tuple[str, str, str]:
    try:
        from version import APP_VERSION, UPDATE_CHANNEL, GITHUB_REPOSITORY

        return str(APP_VERSION), str(UPDATE_CHANNEL or "stable"), str(GITHUB_REPOSITORY or "")
    except Exception:
        return "0.0.0", "stable", ""


def _load_customer_channel_config() -> dict[str, Any]:
    candidates = []
    env_file = os.getenv("ECG_CUSTOMER_CHANNEL_FILE", "").strip()
    if env_file:
        candidates.append(Path(env_file))

    candidates.extend(
        [
            Path(os.getcwd()) / "customer_channels.json",
        ]
    )

    try:
        app_dir = Path(sys.executable).resolve().parent  # type: ignore[name-defined]
        candidates.append(app_dir / "customer_channels.json")
    except Exception:
        pass

    try:
        here = Path(__file__).resolve().parents[2]
        candidates.append(here / "customer_channels.json")
    except Exception:
        pass

    for candidate in candidates:
        try:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict):
                        return data
        except Exception:
            continue
    return {}


def _resolve_customer_channel(default_channel: str) -> str:
    config = _load_customer_channel_config()
    default_from_file = str(config.get("default_channel") or default_channel or "stable").strip() or "stable"
    allowed = {"canary", "blue", "green", "stable"}

    direct = os.getenv("ECG_UPDATE_CHANNEL", "").strip().lower()
    if direct in allowed:
        return direct

    customer_key = (
        os.getenv("ECG_CUSTOMER_ID", "").strip()
        or os.getenv("MACHINE_SERIAL_ID", "").strip()
        or os.getenv("COMPUTERNAME", "").strip()
        or os.getenv("USERNAME", "").strip()
    )
    customer_key = customer_key.lower()
    customers = config.get("customers", {})
    if isinstance(customers, dict):
        for key, channel in customers.items():
            if str(key).strip().lower() == customer_key:
                resolved = str(channel).strip().lower()
                if resolved in allowed:
                    return resolved

    return default_from_file if default_from_file in allowed else "stable"


def _version_parts(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in re.split(r"[^0-9]+", str(value)):
        if token:
            try:
                parts.append(int(token))
            except ValueError:
                continue
    return tuple(parts) if parts else (0,)


def _version_is_newer(candidate: str, current: str) -> bool:
    return _version_parts(candidate) > _version_parts(current)


def _latest_release_for_channel(repo: str, channel: str) -> dict[str, Any] | None:
    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ECGMonitor-Updater",
    }
    response = requests.get(url, headers=headers, timeout=8)
    response.raise_for_status()
    releases = response.json()
    prefix = f"{channel}-"

    for release in releases:
        if release.get("draft"):
            continue
        tag = str(release.get("tag_name") or "")
        if not tag.startswith(prefix):
            continue
        version = tag[len(prefix) :].strip()
        assets = release.get("assets") or []
        installer = None
        for asset in assets:
            name = str(asset.get("name") or "")
            if name.lower().endswith(".exe") or name.lower().endswith(".msi"):
                installer = asset
                break
        if installer is None and assets:
            installer = assets[0]
        if not version or installer is None:
            continue
        return {
            "tag_name": tag,
            "version": version,
            "name": str(release.get("name") or tag),
            "body": str(release.get("body") or ""),
            "asset_name": str(installer.get("name") or ""),
            "asset_url": str(installer.get("browser_download_url") or ""),
            "html_url": str(release.get("html_url") or ""),
            "prerelease": bool(release.get("prerelease")),
        }
    return None


def _download_asset(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)


def check_and_install_update(parent=None, quiet: bool = False) -> bool:
    current_version, channel, repo = _load_version_metadata()
    repo = os.getenv("ECG_UPDATE_REPO", repo).strip()
    channel = _resolve_customer_channel(channel)

    if not repo:
        return False

    try:
        latest = _latest_release_for_channel(repo, channel)
    except Exception:
        return False

    if not latest:
        return False

    if not _version_is_newer(latest["version"], current_version):
        return False

    if not quiet:
        text = (
            f"A newer {channel} build is available.\n\n"
            f"Current: {current_version}\n"
            f"Latest:  {latest['version']}\n\n"
            f"Install now from GitHub Releases?"
        )
        answer = QMessageBox.question(
            parent,
            "Update Available",
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return False

    asset_url = latest.get("asset_url", "")
    if not asset_url:
        return False

    temp_dir = Path(tempfile.gettempdir()) / "ecg_monitor_updates"
    temp_dir.mkdir(parents=True, exist_ok=True)
    installer_name = latest.get("asset_name") or "ECGMonitor-Setup.exe"
    installer_path = temp_dir / installer_name

    try:
        _download_asset(asset_url, installer_path)
        subprocess.Popen(
            [
                str(installer_path),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/SP-",
                "/CLOSEAPPLICATIONS",
            ],
            cwd=str(temp_dir),
        )
        return True
    except Exception:
        return False
