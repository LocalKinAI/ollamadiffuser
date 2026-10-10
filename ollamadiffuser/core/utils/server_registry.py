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
from typing import Dict, List, Optional

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
