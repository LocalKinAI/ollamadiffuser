"""Will a model fit in the memory that is free right now?

On Apple Silicon the GPU shares the machine's memory, so a model that does
not fit does not fail cleanly: the system compresses and swaps until it
stalls. The M3 Ultra this was measured on went down with a watchdog panic
(free memory 0, 58 GB compressed) when a LoRA rebuild pushed it there beside
Ollama's 22 GB, and several agents each starting a model server can get there
just as well. So a load is refused when free memory is below the minimum the
registry gives for the model (``hardware_requirements.min_vram_gb``), with
the numbers and the way around it. NVIDIA cards fail with an ordinary
out-of-memory error instead, which the caller sees, so they are not checked.
"""
from __future__ import annotations

import logging
import os
import platform
from typing import Dict, Optional

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


def refusal(model_name: str, info: Optional[Dict]) -> Optional[str]:
    """Why ``model_name`` should not be loaded now, or None when it may be."""
    if os.environ.get(SKIP_ENV) or not unified_memory():
        return None
    need, free = needed_gb(info), free_gb()
    if need is None or free is None or free >= need:
        return None
    return (
        f"{model_name} needs about {need:g} GB of memory and {free:.1f} GB is free. "
        "Free some first — `ollamadiffuser ps` lists the model servers running, "
        "`ollamadiffuser stop --port N` stops one — or load it anyway with --force "
        f"({SKIP_ENV}=1 for a server or an agent)."
    )
