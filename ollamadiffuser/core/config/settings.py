import contextlib
import copy
import os
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from dataclasses import dataclass, fields

def _stored_hf_token():
    """The token from a `hf auth login`, or None. Never raises."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


logger = logging.getLogger(__name__)

_MISSING = object()
# Threads of one process (the MCP server runs tools in threads) save one at a time.
_SAVE_LOCK = threading.RLock()


def _merge(base: Dict[str, Any], mine: Dict[str, Any], disk: Dict[str, Any]) -> Dict[str, Any]:
    """The file as it is now (``disk``) with this process's changes applied:
    what differs between ``base`` (the file as this process last read or wrote
    it) and ``mine`` (its state now). A key it did not change keeps the
    file's value, whoever wrote that. Dicts are merged key by key, so two
    processes changing different models — or different parameters of one —
    both keep their change."""
    out = copy.deepcopy(disk)
    for key in set(base) | set(mine):
        if key not in mine:                       # removed here
            out.pop(key, None)
            continue
        old, new = base.get(key, _MISSING), mine[key]
        if isinstance(new, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(old if isinstance(old, dict) else {}, new, out[key])
        elif old is _MISSING or old != new:       # added or changed here
            out[key] = copy.deepcopy(new)
    return out


def _known(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """The fields of ``data`` that ``cls`` takes; a hand-added extra no longer
    makes the whole file fail to load (and stays in the file)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    path: str
    model_type: str  # "sd15", "sdxl", "sd3", "flux", etc.
    variant: Optional[str] = None  # "fp16", "fp32", etc.
    components: Optional[Dict[str, str]] = None  # LoRA, VAE, etc.
    parameters: Optional[Dict[str, Any]] = None  # default generation parameters

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "localhost"
    port: int = 8000
    max_queue_size: int = 100
    timeout: int = 600
    enable_cors: bool = True

class Settings:
    """Global application settings"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".ollamadiffuser"
        self.models_dir = self.config_dir / "models"
        self.cache_dir = self.config_dir / "cache"
        self.config_file = self.config_dir / "config.json"
        
        # Ensure directories exist (check first — mkdir fails on symlinks in Python 3.10)
        for d in (self.config_dir, self.models_dir, self.cache_dir):
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.server = ServerConfig()
        self.models: Dict[str, ModelConfig] = {}
        self.current_model: Optional[str] = None
        # `hf auth login` stores a token under ~/.cache/huggingface rather
        # than in the environment, and get_token() reads both. Reading only
        # HF_TOKEN reported "no token found" to someone who had just logged
        # in, and then failed the gated download it had already warned about.
        self.hf_token: Optional[str] = os.environ.get('HF_TOKEN') or _stored_hf_token()
        # The file as this process last read or wrote it; save_config writes
        # only what changed since (see there).
        self._base: Dict[str, Any] = {}
        
        # Load configuration file
        self.load_config()
    
    def load_config(self):
        """Load settings from configuration file"""
        if self.config_file.exists():
            try:
                self._apply(self._read_file())
                self._base = self._to_dict()
                logger.info(f"Configuration file loaded: {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration file: {e}")

    def save_config(self):
        """Save this process's changes to the configuration file.

        Several processes share the file — servers, one-off jobs, agents adding
        an entry — and each holds the copy it read when it started. Writing that
        copy back wiped whatever another process had changed in between: on the
        GPU box another session's new model entry and parameter edits survived
        only by timing. So what this process changed since it last read or wrote
        the file is applied to the file as it is now, under a lock, and written
        in one rename; then this copy takes in what the others changed.
        """
        with _SAVE_LOCK:
            try:
                with self._file_lock():
                    try:
                        disk = self._read_file()
                    except ValueError:
                        broken = self.config_file.with_name(
                            f"{self.config_file.name}.broken-{time.strftime('%Y%m%d-%H%M%S')}")
                        os.replace(self.config_file, broken)
                        logger.warning(f"{self.config_file} was not valid JSON; kept it as {broken}")
                        disk = {}
                    merged = _merge(getattr(self, "_base", None) or {}, self._to_dict(), disk)
                    self._write_file(merged)
                self._apply(merged)
                self._base = self._to_dict()
                logger.info(f"Configuration saved to: {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to save configuration file: {e}")

    def _to_dict(self) -> Dict[str, Any]:
        """This object as the configuration file holds it."""
        config_data = {
            'server': {
                'host': self.server.host,
                'port': self.server.port,
                'max_queue_size': self.server.max_queue_size,
                'timeout': self.server.timeout,
                'enable_cors': self.server.enable_cors
            },
            'models': {
                name: {
                    'name': model.name,
                    'path': model.path,
                    'model_type': model.model_type,
                    'variant': model.variant,
                    'components': model.components,
                    'parameters': model.parameters
                }
                for name, model in self.models.items()
            },
            'current_model': self.current_model
        }

        # Only persist path overrides when they differ from defaults
        default_models_dir = self.config_dir / "models"
        default_cache_dir = self.config_dir / "cache"
        paths = {}
        if self.models_dir != default_models_dir:
            paths['models_dir'] = str(self.models_dir)
        if self.cache_dir != default_cache_dir:
            paths['cache_dir'] = str(self.cache_dir)
        if paths:
            config_data['paths'] = paths
        return config_data

    def _apply(self, config_data: Dict[str, Any]) -> None:
        """Take the configuration file's contents into this object."""
        if 'server' in config_data:
            self.server = ServerConfig(**_known(ServerConfig, config_data['server'] or {}))

        if 'models' in config_data:
            models = {}
            for name, model_data in (config_data['models'] or {}).items():
                try:
                    models[name] = ModelConfig(**_known(ModelConfig, model_data))
                except TypeError as e:
                    # Left out here and left alone in the file.
                    logger.warning(f"Skipping model entry {name!r} in {self.config_file}: {e}")
            self.models = models

        self.current_model = config_data.get('current_model')

        # Load custom path overrides
        if 'paths' in config_data:
            paths_data = config_data['paths'] or {}
            if 'models_dir' in paths_data:
                self.models_dir = Path(paths_data['models_dir'])
            if 'cache_dir' in paths_data:
                self.cache_dir = Path(paths_data['cache_dir'])
            # Ensure custom directories exist
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _read_file(self) -> Dict[str, Any]:
        """The configuration file as it is now; {} when there is none.

        Versions before this one write the file in place, so a read can catch
        one mid-write and find it cut short: a few tries before calling it
        broken (ValueError)."""
        for attempt in range(3):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except FileNotFoundError:
                return {}
            except ValueError:
                if attempt == 2:
                    raise
                time.sleep(0.05)
        return {}

    def _write_file(self, data: Dict[str, Any]) -> None:
        """Replace the configuration file in one rename: a reader sees the old
        file or the new one, never half of one."""
        folder = self.config_file.parent
        folder.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=folder)
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            with contextlib.suppress(OSError):  # keep the file's permissions
                os.chmod(temp, stat.S_IMODE(os.stat(self.config_file).st_mode))
            os.replace(temp, self.config_file)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp)
            raise

    @contextlib.contextmanager
    def _file_lock(self):
        """Hold the configuration's lock file while the file is read and
        replaced, so two processes saving at once take turns. Where flock is
        missing (Windows) the one-rename write still keeps the file whole."""
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is None:
            yield
            return
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file.with_name(self.config_file.name + ".lock"), 'a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def add_model(self, model_config: ModelConfig):
        """Add model configuration"""
        self.models[model_config.name] = model_config
        self.save_config()
    
    def remove_model(self, model_name: str):
        """Remove model configuration"""
        if model_name in self.models:
            del self.models[model_name]
            if self.current_model == model_name:
                self.current_model = None
            self.save_config()
    
    def set_current_model(self, model_name: str):
        """Set current model to use"""
        if model_name in self.models:
            self.current_model = model_name
            self.save_config()
        else:
            raise ValueError(f"Model '{model_name}' does not exist")
    
    def get_model_path(self, model_name: str) -> Path:
        """Get storage path for model"""
        return self.models_dir / model_name

# Global settings instance
settings = Settings() 