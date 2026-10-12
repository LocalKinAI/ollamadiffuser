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

That count is the whole process's, so a server keeping several models in
memory (`serve --max-loaded`) would credit one model with the weights of all
of them. The engine tells the meter what the other models hold and what this
one held when the job began (``in_use`` before and after each load and job,
engine.py), and the meter records the model's own share.
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


def in_use(backend: Optional[str]) -> Optional[int]:
    """Bytes ``backend`` holds for this process now — MLX's arrays, PyTorch's
    CUDA tensors, the Metal driver's allocation for PyTorch on a Mac — or None
    when there is nothing to ask."""
    try:
        if backend == "mlx":
            import mlx.core as mx
            return int(mx.get_active_memory())
        if backend == "cuda":
            import torch
            return int(torch.cuda.memory_allocated())
        if backend == "mps":
            import torch
            return int(torch.mps.driver_allocated_memory())
    except Exception as e:
        logger.debug(f"Could not read the memory in use ({backend}): {e}")
    return None


def _reset_peak(backend: Optional[str]) -> bool:
    """Start counting ``backend``'s peak afresh; False when it cannot be counted."""
    try:
        if backend == "mlx":
            import mlx.core as mx
            mx.reset_peak_memory()
        elif backend == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats()
        return backend in ("mlx", "cuda", "mps")
    except Exception as e:
        logger.debug(f"Not measuring memory ({backend}): {e}")
        return False


def _peak(backend: Optional[str]) -> Optional[int]:
    """``backend``'s peak since _reset_peak — for PyTorch on a Mac, what the
    Metal driver holds, which keeps what it has used."""
    try:
        if backend == "mlx":
            import mlx.core as mx
            return int(mx.get_peak_memory())
        if backend == "cuda":
            import torch
            return int(torch.cuda.max_memory_allocated())
        if backend == "mps":
            import torch
            return int(torch.mps.driver_allocated_memory())
    except Exception as e:
        logger.debug(f"Could not read the memory peak ({backend}): {e}")
    return None


class Meter:
    """Measures the peak memory of the work done inside it (``peak_bytes``).

    ``others``: what other models loaded in this process hold on the same
    backend, left out of the peak. ``held``: what this model held when the
    work began; ``held_after`` is what it holds when the work is done (the
    first job materialises weights MLX loaded lazily).
    """

    def __init__(self, backend: Optional[str], held: Optional[int] = None, others: int = 0):
        self.backend = backend
        self.held, self.others = held, others or 0
        self.peak_bytes: Optional[int] = None
        self.held_after: Optional[int] = held
        self._start: Optional[int] = None

    def __enter__(self):
        self._start = in_use(self.backend)
        if not _reset_peak(self.backend):
            self.backend = None
        return self

    def __exit__(self, kind, value, traceback):
        end = in_use(self.backend)
        if None not in (self.held, self._start, end):
            self.held_after = max(0, self.held + end - self._start)
        peak = _peak(self.backend) if kind is None else None  # a failed generation measures nothing
        if peak is None:
            return False
        # The process's peak without the other models' weights; or what this
        # model held plus what the job added. Both are this model's share when
        # the counts are right, and the larger is the safer when they drift.
        shares = [peak - self.others]
        if None not in (self.held, self._start):
            shares.append(self.held + max(0, peak - self._start))
        self.peak_bytes = max(shares)
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
