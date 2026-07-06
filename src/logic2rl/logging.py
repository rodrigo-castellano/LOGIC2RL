"""Run-bundle plumbing: where a run writes its artifacts.

Owns the canonical ``output/runs/<family>/<run_id>/...`` bundle: the run-bundle
policy (:class:`LoggingConfig` / :class:`ModelConfig`) and the run-scoped runtime
object (:class:`RunContext` — ``stdout_capture`` / ``log_event`` / ``log_metrics``
/ ``save_model`` / ``finish``). The CLI driver that builds one RunContext per run
(``run_cli``) lives in :mod:`base.runner`.

Each run writes: ``config.json`` (resolved config), ``manifest.json`` (final
summary), ``stdout.log`` (teed stdout+stderr), ``events.jsonl`` (lifecycle
timeline), ``metrics.json`` (split-grouped metrics), and — when a model is saved —
the model file plus ``model_info.json``.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import torch


@dataclass
class ModelConfig:
    """Single-model save policy."""

    mode: Literal["none", "last", "best"] = "best"
    metric: str = ""  # metric name recorded with the saved model; the app config sets it (e.g. KGE: mrr_mean)
    filename: str = "model.safetensors"


@dataclass
class LoggingConfig:
    """Run-bundle configuration."""

    family: Optional[str] = None
    output_root: str = "./output"
    model: ModelConfig = field(default_factory=ModelConfig)


def sanitize_slug(value: str) -> str:
    """Normalize an identifier into a stable path component."""
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return collapsed.strip("-._") or "run"


def _json_default(o: Any) -> Any:
    """``json.dumps`` fallback for non-native types (dataclass / Path / set / torch)."""
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.detach().cpu().tolist()
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            return repr(o)
    return repr(o)


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n"


class _TeeStream:
    """Mirror writes to the original stream and a file.

    Implements enough of the file protocol that ``logging.shutdown`` / absl can
    call ``close()`` / ``isatty()`` / ``fileno()`` at interpreter exit without
    raising (and tqdm/rich check ``isatty``). Never closes the wrapped streams on
    its own ``close()`` — that would double-close ``sys.stdout`` / the run log.
    """

    def __init__(self, original: Any, file_handle: Any) -> None:
        self._original = original
        self._file_handle = file_handle

    def write(self, data: str) -> int:
        self._file_handle.write(data)
        self._original.write(data)
        return len(data)

    def flush(self) -> None:
        self._file_handle.flush()
        self._original.flush()

    def close(self) -> None:
        pass

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._original.fileno()


class RunContext:
    """Run-scoped logging and artifact helpers.

    Computes the run id + bundle paths from ``logging.output_root`` / ``family`` /
    ``signature`` / ``seed``, creates the run dir (``self.root``), and writes
    ``config.json`` + a ``running`` manifest on construction.
    """

    def __init__(
        self,
        *,
        logging: LoggingConfig,
        family: str,
        signature: str,
        seed: int,
        resolved_config: Any,
    ) -> None:
        self.logging = logging
        self.family = family
        self.signature = signature
        self.seed = int(seed)
        self.resolved_config = resolved_config

        now = datetime.now(timezone.utc)
        self.started_at = now.isoformat()
        self.run_id = f"{now.strftime('%Y%m%d-%H%M%S')}_{sanitize_slug(signature)}_s{self.seed}"
        self.experiment_root = Path(logging.output_root).expanduser() / "runs" / sanitize_slug(family)
        self.root = self.experiment_root / self.run_id
        self.artifacts_dir = self.root / "artifacts"

        self._model_filename = sanitize_slug(logging.model.filename)
        self._saved_model_path: Optional[Path] = None
        self._saved_model_info: Optional[dict[str, Any]] = None
        self._metrics: dict[str, list] = {"train": [], "val": [], "test": []}

        for directory in (self.experiment_root, self.root, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(_dumps(resolved_config))
        self._write_manifest(status="running")

    def _write_manifest(self, *, status: str, error: Optional[str] = None, final_metrics: Any = None) -> None:
        manifest = {
            "run": {
                "id": self.run_id,
                "family": self.family,
                "signature": self.signature,
                "seed": self.seed,
                "status": status,
                "started_at": self.started_at,
                "ended_at": datetime.now(timezone.utc).isoformat() if status != "running" else None,
                "error": error,
            },
            "paths": {
                "run_root": str(self.root),
                "model_path": str(self._saved_model_path) if self._saved_model_path else None,
            },
            "model": self._saved_model_info,
            "final_metrics": final_metrics,
        }
        (self.root / "manifest.json").write_text(_dumps(manifest))

    @contextmanager
    def stdout_capture(self):
        """Tee stdout+stderr into ``stdout.log``.

        Also rebinds stray logging ``StreamHandler``s (created before this redirect,
        e.g. by ``basicConfig()`` at import time) to the tee'd stderr, so logger
        output also lands in the bundle and not only the original console.
        """
        with (self.root / "stdout.log").open("a", encoding="utf-8", buffering=1) as handle:
            tee_out = _TeeStream(sys.stdout, handle)
            tee_err = _TeeStream(sys.stderr, handle)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                import logging  # stdlib; this module is base.logging (absolute import)
                rebound = []
                for h in logging.getLogger().handlers:
                    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                        rebound.append((h, h.stream))
                        h.stream = sys.stderr
                try:
                    yield
                finally:
                    for h, orig in rebound:
                        h.stream = orig

    def log_event(self, name: str, **payload: Any) -> None:
        """Append a lifecycle event line to ``events.jsonl``."""
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": name, **payload}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")

    @staticmethod
    def _canonical_split(split: Optional[str]) -> str:
        if split is None:
            return "other"
        s = sanitize_slug(split).lower()
        if s in {"eval", "valid", "validation", "val"}:
            return "val"
        return s if s in {"train", "test"} else (s or "other")

    def log_metrics(
        self, metrics: Mapping[str, Any], *, step: Optional[int] = None, split: Optional[str] = None,
    ) -> None:
        """Append one metric snapshot row to ``metrics.json`` under its split bucket."""
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(metrics)}
        if step is not None:
            row["step"] = int(step)
        self._metrics.setdefault(self._canonical_split(split), []).append(row)
        (self.root / "metrics.json").write_text(_dumps(self._metrics))

    def save_model(
        self,
        serializer: Any,
        *,
        saved_as: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        global_step: Optional[int] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist the single model artifact + ``model_info.json`` for this run."""
        if self.logging.model.mode == "none":
            return None
        model_path = self.root / self._model_filename
        serializer(model_path)
        self._saved_model_path = model_path
        info: dict[str, Any] = {
            "save_mode": saved_as or self.logging.model.mode,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "global_step": int(global_step) if global_step is not None else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            info.update(dict(extra_metadata))
        self._saved_model_info = info
        (self.root / "model_info.json").write_text(_dumps(info))
        return model_path

    def finish(self, *, status: str, final_metrics: Mapping[str, Any], error: Optional[str] = None) -> None:
        """Finalize the run manifest."""
        self._write_manifest(status=status, error=error, final_metrics=dict(final_metrics))


__all__ = ["LoggingConfig", "ModelConfig", "RunContext", "sanitize_slug"]
