"""Cheap run-status manifest preventing audits from reading stale forecasts.

A failed forecasting attempt intentionally leaves the last good forecast.parquet in
place.  This sidecar records whether the *current* application run completed and
which exact forecast file it produced, so validators never mistake an older file
for the output of a failed newer run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import settings

RUN_STATUS_FILENAME = "forecast_run_status.json"


def _path_from_out_dir(out_dir: Path) -> Path:
    return Path(out_dir) / RUN_STATUS_FILENAME


def _path_from_forecast(forecast_path: Path) -> Path:
    return Path(forecast_path).parent / RUN_STATUS_FILENAME


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def mark_running(out_dir: Path) -> None:
    _write_atomic(
        _path_from_out_dir(out_dir),
        {
            "status": "running",
            "app_version": str(getattr(settings, "APP_VERSION", "")),
            "started_ns": time.time_ns(),
        },
    )


def mark_failed(out_dir: Path, exc: BaseException) -> None:
    _write_atomic(
        _path_from_out_dir(out_dir),
        {
            "status": "failed",
            "app_version": str(getattr(settings, "APP_VERSION", "")),
            "finished_ns": time.time_ns(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        },
    )


def mark_success(forecast_path: Path) -> None:
    path = Path(forecast_path)
    stat = path.stat()
    _write_atomic(
        _path_from_forecast(path),
        {
            "status": "success",
            "app_version": str(getattr(settings, "APP_VERSION", "")),
            "finished_ns": time.time_ns(),
            "forecast_mtime_ns": int(stat.st_mtime_ns),
            "forecast_size_bytes": int(stat.st_size),
        },
    )


def validation_errors(forecast_path: Path) -> list[str]:
    """Return errors when forecast is not the successful current-version output."""
    forecast = Path(forecast_path)
    status_path = _path_from_forecast(forecast)
    if not status_path.exists():
        return [
            f"falta {RUN_STATUS_FILENAME}; ejecute forecasts.py con la versión actual antes de auditar"
        ]
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"no se puede leer {RUN_STATUS_FILENAME}: {exc}"]

    expected_version = str(getattr(settings, "APP_VERSION", ""))
    errors: list[str] = []
    if str(payload.get("status") or "") != "success":
        errors.append(
            f"última corrida no terminó correctamente (status={payload.get('status')!r})"
        )
    if str(payload.get("app_version") or "") != expected_version:
        errors.append(
            f"forecast pertenece a APP_VERSION={payload.get('app_version')!r}, no {expected_version!r}"
        )
    if not forecast.exists():
        errors.append(f"no existe forecast: {forecast}")
        return errors

    stat = forecast.stat()
    expected_mtime = int(payload.get("forecast_mtime_ns") or -1)
    expected_size = int(payload.get("forecast_size_bytes") or -1)
    if expected_mtime != int(stat.st_mtime_ns) or expected_size != int(stat.st_size):
        errors.append(
            "forecast.parquet no coincide con el archivo registrado por la última corrida exitosa"
        )
    return errors
