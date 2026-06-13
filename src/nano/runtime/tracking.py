"""Pluggable experiment-tracking backends.

The rest of the system depends only on the :class:`TrackingBackend` protocol, so
Flywheel is never a hard dependency. A failed upload must never mark the
training run itself as failed (handled by :func:`upload_run`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from nano.runtime.manifest import read_json


class TrackingBackend(Protocol):
    def create_run(self, manifest: Mapping[str, Any]) -> str: ...
    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None: ...
    def upload_artifact(self, run_id: str, path: str, artifact_type: str | None = None) -> None: ...
    def finish(self, run_id: str, status: str) -> None: ...


class NoOpBackend:
    """Discards everything. The default; lets every run complete with no creds."""

    def create_run(self, manifest: Mapping[str, Any]) -> str:
        return str(manifest.get("run_id", "noop"))

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        pass

    def upload_artifact(self, run_id: str, path: str, artifact_type: str | None = None) -> None:
        pass

    def finish(self, run_id: str, status: str) -> None:
        pass


class LocalJsonlBackend:
    """Appends tracking events to a local JSONL file -- a dependency-free mirror."""

    def __init__(self, path: str | Path = "experiments/tracking.jsonl", **_: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, event: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def create_run(self, manifest: Mapping[str, Any]) -> str:
        run_id = str(manifest.get("run_id", "run"))
        self._emit({"event": "create_run", "run_id": run_id, "manifest": dict(manifest)})
        return run_id

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        self._emit({"event": "log_metrics", "run_id": run_id, "metrics": dict(metrics)})

    def upload_artifact(self, run_id: str, path: str, artifact_type: str | None = None) -> None:
        self._emit({"event": "artifact", "run_id": run_id, "path": str(path), "type": artifact_type})

    def finish(self, run_id: str, status: str) -> None:
        self._emit({"event": "finish", "run_id": run_id, "status": status})


class FlywheelBackend:
    """Stub adapter. Wire to the Flywheel SDK/API once creds/docs are available."""

    def __init__(self, api_key: str | None = None, project: str | None = None, **_: Any):
        self.api_key = api_key
        self.project = project

    def create_run(self, manifest: Mapping[str, Any]) -> str:
        raise NotImplementedError(
            "FlywheelBackend is a stub. Wire this to the Flywheel SDK/API once "
            "credentials/docs are available (see SPECS.md Phase 5)."
        )

    def log_metrics(self, run_id: str, metrics: Mapping[str, float]) -> None:
        raise NotImplementedError("FlywheelBackend.log_metrics not implemented yet")

    def upload_artifact(self, run_id: str, path: str, artifact_type: str | None = None) -> None:
        raise NotImplementedError("FlywheelBackend.upload_artifact not implemented yet")

    def finish(self, run_id: str, status: str) -> None:
        raise NotImplementedError("FlywheelBackend.finish not implemented yet")


_BACKENDS = {
    "noop": NoOpBackend,
    "local": LocalJsonlBackend,
    "local_jsonl": LocalJsonlBackend,
    "flywheel": FlywheelBackend,
}

ARTIFACT_FILES = ("manifest.json", "features.yaml", "train_generated.py", "raw.log", "summary.json")


def get_backend(name: str | None, **kwargs: Any) -> TrackingBackend:
    name = (name or "noop").lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown tracking backend {name!r}; known: {', '.join(_BACKENDS)}")
    return _BACKENDS[name](**kwargs)


def upload_run(
    manifest_path: str | Path,
    backend: TrackingBackend | None = None,
    *,
    backend_name: str | None = None,
) -> dict[str, Any]:
    """Upload a run's artifacts to a tracking backend.

    Never raises on backend failure -- returns a result dict with ``ok`` and any
    error string so the caller can record an *upload* failure without marking the
    training run failed.
    """
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    run_dir = manifest_path.parent

    if backend is None:
        tracking = manifest.get("tracking", {}) or {}
        backend = get_backend(backend_name or tracking.get("backend"))

    result: dict[str, Any] = {"ok": True, "error": None, "backend": type(backend).__name__}
    try:
        run_id = backend.create_run(manifest)
        metrics = {k: v for k, v in (manifest.get("metrics") or {}).items() if isinstance(v, (int, float))}
        if metrics:
            backend.log_metrics(run_id, metrics)
        for name in ARTIFACT_FILES:
            artifact = run_dir / name
            if artifact.exists():
                backend.upload_artifact(run_id, str(artifact), artifact_type=name.split(".")[-1])
        backend.finish(run_id, manifest.get("status", "completed"))
    except Exception as exc:  # noqa: BLE001 -- upload must never break the run
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
