"""What a model really takes: the memory peak of its generations, kept per model.

The registry's minimum for a model is an estimate written once, and some are
far off — Boogu Image Turbo is listed at 10 GB, and on the M3 Ultra a one-off
image of it took 20 GB and more. So every generation records its own peak,
the highest seen is kept in ``~/.ollamadiffuser/memory/<model>.json``, and the
memory check (memory_check.py) goes by that once a model has one.

The peak is the backend's own count: MLX's for mflux models, which includes
the weights; PyTorch's for CUDA; the Metal driver's allocation for PyTorch on
a Mac, which holds on to what it has used. A model that renders in another
process (LTX-2 through the ltx-2-mlx CLI) is not measured: this process would
see almost nothing and record a need that is far too small.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Below this a reading is not a model's peak but a backend that measured
# nothing — a strategy that ran elsewhere, or one that never touched the GPU.
_MIN_PLAUSIBLE_BYTES = 500_000_000


def _folder() -> Path:
    from ..config.settings import settings
    return Path(settings.config_dir) / "memory"


def _path(model: str) -> Path:
    return _folder() / (re.sub(r"[^A-Za-z0-9._-]", "_", model) + ".json")


class Meter:
    """Measures the peak memory of the work done inside it (``peak_bytes``)."""

    def __init__(self, backend: Optional[str]):
        self.backend = backend
        self.peak_bytes: Optional[int] = None

    def __enter__(self):
        try:
            if self.backend == "mlx":
                import mlx.core as mx
                mx.reset_peak_memory()
            elif self.backend == "cuda":
                import torch
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            logger.debug(f"Not measuring memory ({self.backend}): {e}")
            self.backend = None
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            return False  # a failed generation measures nothing
        try:
            if self.backend == "mlx":
                import mlx.core as mx
                self.peak_bytes = int(mx.get_peak_memory())
            elif self.backend == "cuda":
                import torch
                self.peak_bytes = int(torch.cuda.max_memory_allocated())
            elif self.backend == "mps":
                import torch
                self.peak_bytes = int(torch.mps.driver_allocated_memory())
        except Exception as e:
            logger.debug(f"Could not read the memory peak ({self.backend}): {e}")
        return False


def record(model: Optional[str], peak_bytes: Optional[int],
           width: Optional[int] = None, height: Optional[int] = None) -> None:
    """Keep ``peak_bytes`` for ``model`` if it is the highest seen."""
    if not model or not peak_bytes or peak_bytes < _MIN_PLAUSIBLE_BYTES:
        return
    path = _path(model)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked(path.parent / ".lock"):
            old = _read(path) or {}
            peak_gb = round(peak_bytes / 1e9, 2)
            entry = dict(old)
            entry["runs"] = int(old.get("runs", 0)) + 1
            if peak_gb >= float(old.get("peak_gb", 0)):
                entry.update(peak_gb=peak_gb, width=width, height=height,
                             measured_at=time.strftime("%Y-%m-%d"))
            _write(path, entry)
    except Exception as e:  # a measurement is never worth failing a generation over
        logger.debug(f"Could not record the memory peak of {model}: {e}")


def measured(model: Optional[str]) -> Optional[Dict]:
    """The highest peak recorded for ``model`` ({"peak_gb", "width", "height",
    "measured_at", "runs"}), or None when it has none."""
    if not model:
        return None
    entry = _read(_path(model))
    if not entry or not entry.get("peak_gb"):
        return None
    return entry


def _read(path: Path) -> Optional[Dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, data: Dict) -> None:
    handle, temp = tempfile.mkstemp(prefix=".memory-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


@contextlib.contextmanager
def _locked(lock_path: Path):
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is None:
        yield
        return
    with open(lock_path, "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
