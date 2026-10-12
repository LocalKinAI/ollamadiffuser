"""Will a model fit in the memory that is free right now?

On Apple Silicon the GPU shares the machine's memory, so a model that does
not fit does not fail cleanly: the system compresses and swaps until it
stalls. The M3 Ultra this was measured on went down with a watchdog panic
(free memory 0, 58 GB compressed) when a LoRA rebuild pushed it there beside
Ollama's 22 GB, and several agents each starting a model server can get there
just as well. So a load is refused when free memory is below what the model
needs — its measured peak once a generation has recorded one, the registry's
``hardware_requirements.min_vram_gb`` until then — with the numbers and the
way around it. NVIDIA cards fail with an ordinary out-of-memory error
instead, which the caller sees, so they are not checked.
"""
from __future__ import annotations

import logging
import os
import platform
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SKIP_ENV = "OLLAMADIFFUSER_SKIP_MEMORY_CHECK"


def unified_memory() -> bool:
    """Apple Silicon, where the model and everything else share one memory."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def free_gb() -> Optional[float]:
    """Memory the system could hand out now (free plus reclaimable), in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def needed_gb(info: Optional[Dict]) -> Optional[float]:
    """The memory a registry entry says its model needs, or None when it says nothing."""
    hardware = (info or {}).get("hardware_requirements") or {}
    try:
        need = float(hardware.get("min_vram_gb") or 0)
    except (TypeError, ValueError):
        return None
    return need or None


# What a backend's own count of a model's peak leaves out: Python, decoded
# pictures, files being read, and the 2 GB MLX may keep cached.
MARGIN_GB = 3.0


def need_for(model_name: str, info: Optional[Dict]) -> Tuple[Optional[float], str]:
    """The memory ``model_name`` needs, and where the number comes from: its
    measured peak once a generation has recorded one (memory_record.py),
    otherwise the registry's estimate."""
    from . import memory_record
    seen = memory_record.measured(model_name)
    if seen:
        peak = float(seen["peak_gb"])
        return round(peak + MARGIN_GB, 1), f"its measured peak, {peak:g} GB, plus {MARGIN_GB:g}"
    return needed_gb(info), "the registry's estimate"


def refusal(model_name: str, info: Optional[Dict], freeable_gb: float = 0.0) -> Optional[str]:
    """Why ``model_name`` should not be loaded now, or None when it may be.

    ``freeable_gb``: what unloading the other models this process holds would
    give back — a server keeping several makes room that way first, so the
    load is refused only when even that would not be enough.
    """
    if os.environ.get(SKIP_ENV) or not unified_memory():
        return None
    (need, source), free = need_for(model_name, info), free_gb()
    if need is None or free is None or free + freeable_gb >= need:
        return None
    others = (f", {free + freeable_gb:.1f} GB with the other models loaded here unloaded"
              if freeable_gb > 0 else "")
    return (
        f"{model_name} needs about {need:g} GB of memory ({source}) and {free:.1f} GB is free{others}. "
        "Free some first — `ollamadiffuser ps` lists the model servers running, "
        "`ollamadiffuser stop --port N` stops one — or load it anyway with --force "
        f"({SKIP_ENV}=1 for a server or an agent)."
    )


def room_gb() -> Optional[float]:
    """What a model could take now, in GB: free memory on Apple Silicon, free
    GPU memory on NVIDIA (with what PyTorch keeps cached but unused); None
    where there is no telling."""
    if unified_memory():
        return free_gb()
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            return (free + max(0, cached)) / 1e9
    except Exception as e:
        logger.debug(f"Could not read the free GPU memory: {e}")
    return None


def short_gb(model_name: str, info: Optional[Dict], held_gb: Optional[float] = 0.0) -> float:
    """How much more memory than is free ``model_name`` needs now, in GB: 0
    when it fits, or when there is no telling. ``held_gb``: what it holds
    already — a model in memory about to run needs only the rest. What a
    server unloads other models by; not a refusal, so not skipped on request."""
    need, room = need_for(model_name, info)[0], room_gb()
    if need is None or room is None:
        return 0.0
    return max(0.0, need - (held_gb or 0.0) - room)
