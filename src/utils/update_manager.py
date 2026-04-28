from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import sys
import threading
from pathlib import Path
from typing import Any
from datetime import datetime

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


def _telemetry_state_path() -> Path:
    base = os.getenv("ECG_RUNTIME_DIR", "").strip()
    if base:
        root = Path(base)
    else:
        root = Path(os.getcwd())
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return root / "update_status.json"


def _load_telemetry_state() -> dict[str, Any]:
    path = _telemetry_state_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_telemetry_state(state: dict[str, Any]) -> None:
    path = _telemetry_state_path()
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    except Exception:
        pass


def _build_update_event(current_version: str, channel: str, repo: str) -> dict[str, Any]:
    customer_key = (
        os.getenv("ECG_CUSTOMER_ID", "").strip()
        or os.getenv("MACHINE_SERIAL_ID", "").strip()
        or os.getenv("COMPUTERNAME", "").strip()
        or os.getenv("USERNAME", "").strip()
    )
    return {
        "event_type": "update_report",
        "reported_at": datetime.utcnow().isoformat() + "Z",
        "app_version": current_version,
        "channel": channel,
        "repository": repo,
        "customer_key": customer_key,
        "machine_serial_id": os.getenv("MACHINE_SERIAL_ID", "").strip(),
        "computer_name": os.getenv("COMPUTERNAME", "").strip(),
        "username": os.getenv("USERNAME", "").strip(),
        "install_type": "update" if _load_telemetry_state().get("reported_version") else "first_run",
    }


def _post_update_event(event: dict[str, Any]) -> bool:
    url = os.getenv("ECG_UPDATE_TELEMETRY_URL", "").strip()
    if not url:
        return True

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("ECG_UPDATE_TELEMETRY_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.post(url, json=event, headers=headers, timeout=8)
        return response.status_code in (200, 201, 202)
    except Exception:
        return False


def report_update_completion(current_version: str | None = None, *, async_mode: bool = True) -> bool:
    """
    Record and optionally send an update-complete heartbeat.

    The event is emitted once per app version. If a telemetry endpoint is not
    configured, the event is stored locally in update_status.json.
    """
    version, channel, repo = _load_version_metadata()
    current_version = str(current_version or version or "0.0.0").strip()
    if not current_version:
        return False

    state = _load_telemetry_state()
    if state.get("reported_version") == current_version and not state.get("pending_event"):
        return False

    event = _build_update_event(current_version, channel, repo)
    state["pending_event"] = event
    _save_telemetry_state(state)

    def _worker() -> None:
        latest_state = _load_telemetry_state()
        pending = latest_state.get("pending_event")
        if not isinstance(pending, dict):
            pending = event
        if _post_update_event(pending):
            latest_state["reported_version"] = current_version
            latest_state["reported_at"] = pending.get("reported_at")
            latest_state.pop("pending_event", None)
            _save_telemetry_state(latest_state)

    if async_mode:
        threading.Thread(target=_worker, daemon=True, name="UpdateTelemetry").start()
    else:
        _worker()
    return True


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
