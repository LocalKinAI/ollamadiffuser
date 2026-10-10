"""Sample a checkpoint on the sigma schedule it ships with, through mflux.

Qwen-Image-2.1-Turbo (2026-10-09) is the same transformer as Qwen-Image-2.1,
distilled to 8 steps; what makes it a turbo is a fixed table of 8 noise
levels in its ``model_index.json`` (``sample_sigmas``), which diffusers'
pipeline uses whatever step count it is asked for. mflux has no such
schedule for it: its Qwen-Image-2.1 sampler spaces the steps linearly and
shifts them by resolution, which is the base model's schedule and not the
one the turbo was distilled on.

mflux resolves a scheduler by name from a registry it lets callers extend
(``register_contrib``), so this registers one that walks the checkpoint's own
table: the nodes as given — the turbo's scheduler config has shift 1.0 and no
dynamic shifting, so no shift applies — then 0, with the same first-order
Euler step mflux's linear scheduler takes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def read_sample_sigmas(model_dir: Optional[str]) -> Optional[List[float]]:
    """The ``sample_sigmas`` a diffusers checkpoint's model_index.json carries, if any."""
    if not model_dir:
        return None
    try:
        data = json.loads((Path(model_dir) / "model_index.json").read_text())
    except (OSError, ValueError):
        return None
    nodes = data.get("sample_sigmas")
    if not isinstance(nodes, list) or not nodes:
        return None
    try:
        nodes = [float(x) for x in nodes]
    except (TypeError, ValueError):
        return None
    # Noise levels run from 1 toward 0 and never repeat.
    if not all(0.0 < x <= 1.0 for x in nodes) or any(b >= a for a, b in zip(nodes, nodes[1:])):
        logger.warning(f"Ignoring sample_sigmas in {model_dir}: not a decreasing table in (0, 1]")
        return None
    return nodes


def scheduler_name(nodes: List[float]) -> str:
    digest = hashlib.sha1(json.dumps(nodes).encode()).hexdigest()[:10]
    return f"ollamadiffuser_sample_sigmas_{digest}"


def register(nodes: List[float]) -> str:
    """Register a scheduler walking ``nodes`` with mflux; returns its name."""
    import mlx.core as mx
    from mflux.models.common.schedulers import SCHEDULER_REGISTRY, register_contrib
    from mflux.models.common.schedulers.base_scheduler import BaseScheduler

    name = scheduler_name(nodes)
    if name in SCHEDULER_REGISTRY:
        return name
    table = tuple(nodes)

    class SampleSigmasScheduler(BaseScheduler):
        NODES = table

        def __init__(self, config):
            self.config = config
            steps = config.num_inference_steps
            if steps != len(self.NODES):
                raise ValueError(
                    f"this checkpoint samples on its own {len(self.NODES)}-step schedule; "
                    f"num_inference_steps is {steps}. Use {len(self.NODES)}."
                )
            self._sigmas = mx.array(list(self.NODES) + [0.0], dtype=mx.float32)
            self._timesteps = mx.arange(steps, dtype=mx.float32)

        @property
        def sigmas(self):
            return self._sigmas

        @property
        def timesteps(self):
            return self._timesteps

        def step(self, noise, timestep: int, latents, **kwargs):
            dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
            return latents + noise.astype(latents.dtype) * dt

    SampleSigmasScheduler.__name__ = name
    register_contrib(SampleSigmasScheduler, name)
    return name
