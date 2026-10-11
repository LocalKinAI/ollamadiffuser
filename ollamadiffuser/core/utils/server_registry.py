"""Which OllamaDiffuser servers are running on this machine.

Every ``ollamadiffuser run`` / ``serve`` is its own process holding one model,
and several can run at once — an image model on one port, a video model on
another. ``ps`` used to read a single "current model" field that each ``run``
overwrote, and to ask only the default port, so it showed whichever started
last. Each server now leaves a small file under ``~/.ollamadiffuser/servers/``
while it is up; ``ps`` reads them, drops the ones whose process is gone, and
asks each live server what it holds.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def servers_dir() -> Path:
    from ..config.settings import settings
    return Path(settings.config_dir) / "servers"


def _record_path(port: int) -> Path:
    return servers_dir() / f"{int(port)}.json"


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # someone else's process, but it exists
    except (OSError, ValueError, TypeError):
        return False
    return True


def register(host: str, port: int, model: Optional[str] = None) -> Path:
    """Record this process as the server on ``port``."""
    path = _record_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pid": os.getpid(), "host": host, "port": int(port),
        "model": model, "started": time.time(),
    }))
    return path


def unregister(port: int) -> None:
    """Remove this process's record for ``port`` (another's is left alone)."""
    path = _record_path(port)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if record.get("pid") == os.getpid():
        try:
            path.unlink()
        except OSError:
            pass


def running() -> List[Dict]:
    """The recorded servers whose process is still alive; stale records go."""
    folder = servers_dir()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if _alive(record.get("pid")):
            found.append(record)
        else:
            try:
                path.unlink()
            except OSError:
                pass
    return sorted(found, key=lambda r: r.get("port", 0))


def ask(host: str, port: int, timeout: float = 2.0) -> Optional[Dict]:
    """What the server on ``host:port`` says it is running, or None."""
    import requests
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", None) else host
    try:
        response = requests.get(f"http://{probe_host}:{port}/api/models/running", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# Where `run` and the ComfyUI nodes put servers.
USUAL_PORTS = range(8000, 8020)


def discover(ports: Iterable[int] = USUAL_PORTS, timeout: float = 1.0,
             host: Optional[str] = None) -> List[Dict]:
    """Every OllamaDiffuser server on this machine, and what each holds.

    The records first, then the usual ports: a server started by an older
    version, or with another config directory — an agent's sandbox — leaves
    no record here but still answers. Each entry has ``port``, ``host``,
    ``url``, ``pid`` (None when found by probing), ``answering``, ``loaded``,
    ``model``, ``type`` and ``device``.

    ``host`` looks at another machine instead: its usual ports only, since
    the records are this machine's. Only servers started with ``--host
    0.0.0.0`` answer from there.
    """
    from concurrent.futures import ThreadPoolExecutor

    recorded = {} if host else {int(r["port"]): r for r in running() if r.get("port") is not None}
    candidates = sorted(set(recorded) | {int(p) for p in ports})

    def probe(port: int):
        where = host or recorded.get(port, {}).get("host") or "127.0.0.1"
        answer = ask(where, port, timeout=timeout)
        # Something else may listen on these ports; ours answers this shape.
        return port, where, answer if isinstance(answer, dict) and "loaded" in answer else None

    with ThreadPoolExecutor(max_workers=max(1, min(16, len(candidates)))) as pool:
        answers = list(pool.map(probe, candidates))

    found = []
    for port, where, answer in answers:
        record = recorded.get(port)
        if answer is None and record is None:
            continue
        loaded = bool(answer and answer.get("loaded"))
        info = (answer or {}).get("info") or {}
        url_host = "127.0.0.1" if where in ("0.0.0.0", "", None) else where
        found.append({
            "port": port,
            "host": where,
            "url": f"http://{url_host}:{port}",
            "pid": record.get("pid") if record else None,
            "answering": answer is not None,
            "loaded": loaded,
            "model": ((answer or {}).get("model") or info.get("name")) if loaded else None,
            "type": info.get("type") if loaded else None,
            "device": info.get("device") if loaded else None,
        })
    return found
