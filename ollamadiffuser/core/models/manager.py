import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import hashlib
from huggingface_hub import login
from ..config.settings import settings, ModelConfig
from ..config.model_registry import model_registry
from ..utils.download_utils import robust_snapshot_download, robust_file_download
from .gguf_loader import gguf_loader, GGUF_AVAILABLE

logger = logging.getLogger(__name__)

# Known pipeline directory patterns by model_type.
# Downloading only these directories skips root-level monolithic checkpoints,
# ONNX/Flax/OpenVINO exports, and other non-diffusers files.
_PIPELINE_ALLOW_PATTERNS = {
    "sd15": [
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "unet/*",
        "vae/*",
    ],
    "sdxl": [
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "text_encoder_2/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "unet/*",
        "vae/*",
    ],
    "sd3": [
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "text_encoder_2/*",
        "text_encoder_3/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "tokenizer_3/*",
        "transformer/*",
        "vae/*",
    ],
    "flux": [
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "text_encoder_2/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "transformer/*",
        "vae/*",
    ],
}

# Default files to skip for all non-GGUF models.
# These are safe to ignore: we never load safety_checker (always None),
# and we only use PyTorch — not ONNX, Flax, or OpenVINO.
_DEFAULT_IGNORE_PATTERNS = [
    "*.ckpt",
    "*.onnx",
    "*.onnx_data",
    "*.msgpack",
    "*.xml",
    "*.pb",
    "*.git*",
    "safety_checker/*",
    "feature_extractor/*",
    "comfyui/*",
    "text_encoders/*",
    "__pycache__/*",
]


class ModelManager:
    """Model manager with dynamic registry support and GGUF compatibility"""
    
    def __init__(self):
        self.loaded_model: Optional[object] = None
        self.current_model_name: Optional[str] = None
        self.current_model_type: Optional[str] = None  # Track model type
        self.last_error: Optional[str] = None  # why the last load_model() failed
        # How many models may stay in memory at once: one, unless a server
        # keeps more (`serve --max-loaded N`). The current model is the one
        # above; the others wait here, least recently used first.
        self.max_loaded: int = 1
        self._others: "OrderedDict[str, Tuple[object, Optional[str]]]" = OrderedDict()
    
    @property
    def model_registry(self):
        """Get the current model registry (for backward compatibility)"""
        return model_registry.get_all_models()
    
    def list_available_models(self) -> List[str]:
        """List all available models"""
        return model_registry.get_model_names()
    
    def list_installed_models(self) -> List[str]:
        """List installed models"""
        return list(settings.models.keys())
    
    def is_model_installed(self, model_name: str) -> bool:
        """Check if model is installed"""
        return model_name in settings.models
    
    def is_gguf_model(self, model_name: str) -> bool:
        """Check if a model is a GGUF model"""
        if not model_name:
            return False
        model_info = model_registry.get_model(model_name)
        if model_info:
            return gguf_loader.is_gguf_model(model_name, model_info)
        return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information"""
        info = model_registry.get_model(model_name)
        if info:
            # Create a copy to avoid modifying the original
            info = info.copy()
            info['installed'] = self.is_model_installed(model_name)
            info['is_gguf'] = self.is_gguf_model(model_name)
            info['gguf_supported'] = GGUF_AVAILABLE
            if info['installed']:
                config = settings.models[model_name]
                info['local_path'] = config.path
                info['size'] = self._get_model_size(config.path)
            return info
        return None
    
    def _installed_copies(self, model_name: str, repo_id: str) -> List[str]:
        """Folders of other installed models pulled from ``repo_id`` — the
        sources `pull` links from before it downloads (see link_from_local_dirs)."""
        folders = []
        for name, config in settings.models.items():
            if name == model_name:
                continue
            info = model_registry.get_model(name) or {}
            if info.get("repo_id") == repo_id and getattr(config, "path", None):
                folders.append(str(config.path))
        return folders

    def _get_model_size(self, model_path: str) -> str:
        """Get model size"""
        try:
            path = Path(model_path)
            if path.is_file():
                size = path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} PB"
        except Exception:
            return "Unknown"
    
    def pull_model(self, model_name: str, force: bool = False, progress_callback=None) -> bool:
        """Download model using robust download utilities with detailed progress tracking"""
        if not force and self.is_model_installed(model_name):
            logger.info(f"Model {model_name} already exists")
            if progress_callback:
                progress_callback(f"✅ Model {model_name} already installed")
            return True
        
        model_info = model_registry.get_model(model_name)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            if progress_callback:
                progress_callback(f"❌ Error: Unknown model {model_name}")
            return False
        
        model_path = settings.get_model_path(model_name)
        
        # Show model information before download
        if progress_callback:
            license_info = model_info.get("license_info", {})
            progress_callback(f"📦 Model: {model_name}")
            progress_callback(f"🔗 Repository: {model_info['repo_id']}")
            if license_info:
                progress_callback(f"📄 License: {license_info.get('type', 'Unknown')}")
                if license_info.get('requires_agreement', False):
                    progress_callback(
                        "🔑 Gated: needs `hf auth login`, and access granted on "
                        f"https://huggingface.co/{model_info['repo_id']}")
                else:
                    progress_callback(f"✅ No HuggingFace token required")
        
        # Check if partial download exists and is valid
        if not force and model_path.exists():
            if progress_callback:
                progress_callback(f"🔍 Checking existing download...")
            
            from ..utils.download_utils import check_download_integrity
            if check_download_integrity(str(model_path), model_info["repo_id"]):
                if progress_callback:
                    progress_callback(f"✅ Found complete download, adding to configuration...")
                
                # Add to configuration
                model_config = ModelConfig(
                    name=model_name,
                    path=str(model_path),
                    model_type=model_info["model_type"],
                    variant=model_info.get("variant"),
                    components=model_info.get("components"),
                    parameters=model_info.get("parameters")
                )
                
                settings.add_model(model_config)
                logger.info(f"Model {model_name} configuration updated")
                if progress_callback:
                    progress_callback(f"✅ {model_name} ready to use!")
                return True
            else:
                if progress_callback:
                    progress_callback(f"⚠️ Incomplete download detected, will resume...")
        
        try:
            # Ensure HuggingFace token is set
            # A token in the environment is worth installing into the hub's
            # own store; one that is already there needs nothing doing. Only
            # the absence of both is worth warning about.
            if os.environ.get('HF_TOKEN'):
                login(token=os.environ['HF_TOKEN'])
                if progress_callback:
                    progress_callback(f"🔑 Authenticated with HuggingFace (HF_TOKEN)")
            elif settings.hf_token:
                if progress_callback:
                    progress_callback(f"🔑 Using the token from `hf auth login`")
            else:
                if progress_callback:
                    progress_callback(
                        "⚠️ Not logged in to HuggingFace — gated models will "
                        "refuse. Log in with: hf auth login")
            
            logger.info(f"Downloading model: {model_name}")
            if progress_callback:
                progress_callback(f"🚀 Starting download of {model_name}")
            
            # Determine download patterns for GGUF models
            download_kwargs = {
                "repo_id": model_info["repo_id"],
                "local_dir": str(model_path),
                "cache_dir": str(settings.cache_dir),
                "max_retries": 5,  # Increased retries for large models
                "initial_workers": 4,  # More workers for faster download
                "force_download": force,
                "progress_callback": progress_callback
            }
            
            # Add download filtering
            if self.is_gguf_model(model_name):
                variant = model_info.get("variant", "gguf")
                patterns = gguf_loader.get_gguf_download_patterns(variant)
                download_kwargs["allow_patterns"] = patterns["allow_patterns"]
                download_kwargs["ignore_patterns"] = patterns["ignore_patterns"]

                if progress_callback:
                    progress_callback(f"🔍 GGUF model detected - downloading only required files for {variant}")
                    progress_callback(f"📦 Required files: {len(patterns['allow_patterns'])} files")
                    progress_callback(f"🚫 Ignoring: {len(patterns['ignore_patterns'])} other GGUF variants")
            else:
                # Model-specific allow_patterns override (e.g. single-file models)
                if "allow_patterns" in model_info:
                    download_kwargs["allow_patterns"] = model_info["allow_patterns"]
                else:
                    # Use model_type-based pipeline patterns to skip root-level
                    # monolithic checkpoints, ONNX/Flax exports, etc.
                    model_type = model_info.get("model_type", "")
                    if model_type in _PIPELINE_ALLOW_PATTERNS:
                        download_kwargs["allow_patterns"] = _PIPELINE_ALLOW_PATTERNS[model_type]

                # Always apply ignore patterns (ckpt, onnx, flax, safety_checker, etc.)
                download_kwargs["ignore_patterns"] = model_info.get(
                    "ignore_patterns", _DEFAULT_IGNORE_PATTERNS
                )
                if progress_callback:
                    progress_callback(f"📦 Filtering downloads: skipping non-PyTorch files and safety_checker")
            
            # Another installed entry from the same repo already has these files.
            download_kwargs["local_sources"] = self._installed_copies(model_name, model_info["repo_id"])

            # Download main model using robust downloader with enhanced progress
            from ..utils.download_utils import robust_snapshot_download
            robust_snapshot_download(**download_kwargs)
            
            # Download components (such as LoRA)
            if "components" in model_info:
                components_path = model_path / "components"
                components_path.mkdir(exist_ok=True)
                
                for comp_name, comp_info in model_info["components"].items():
                    comp_path = components_path / comp_name
                    comp_path.mkdir(exist_ok=True)
                    
                    if progress_callback:
                        progress_callback(f"📦 Downloading component: {comp_name}")
                    
                    robust_snapshot_download(
                        repo_id=comp_info["repo_id"],
                        local_dir=str(comp_path),
                        cache_dir=str(settings.cache_dir),
                        allow_patterns=comp_info.get("allow_patterns"),
                        ignore_patterns=comp_info.get("ignore_patterns", ["*.git*", "README.md", "*.txt"]),
                        max_retries=3,
                        initial_workers=2,
                        progress_callback=progress_callback
                    )
            
            # Create model configuration
            model_config = ModelConfig(
                name=model_name,
                path=str(model_path),
                model_type=model_info["model_type"],
                variant=model_info.get("variant"),
                components=model_info.get("components"),
                parameters=model_info.get("parameters")
            )
            
            # Add to settings
            settings.add_model(model_config)
            
            logger.info(f"Model {model_name} downloaded successfully")
            if progress_callback:
                progress_callback(f"✅ {model_name} downloaded and configured successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if progress_callback:
                progress_callback(f"❌ Download failed: {str(e)}")
            
            # Clean up partial download
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial download: {cleanup_error}")
            
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove model"""
        if not self.is_model_installed(model_name):
            logger.error(f"Model {model_name} is not installed")
            return False
        
        try:
            # If it is in memory, unload it first
            if model_name in self.resident_models():
                self.unload_model(name=model_name)
            
            # Delete model files
            model_config = settings.models[model_name]
            model_path = Path(model_config.path)
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Remove from configuration
            settings.remove_model(model_name)
            
            logger.info(f"Model {model_name} has been removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model: {e}")
            return False
    
    def load_model(self, model_name: str, force: bool = False, persist: bool = True) -> bool:
        """Load model into memory (supports both regular and GGUF models).

        A model already in memory is made the current one, with no load. A
        new one gets room from the models in memory, least recently used
        first: for the count (``max_loaded`` in all), and for the memory it
        needs. A load that is refused or fails leaves memory as it was — the
        models unloaded for it are loaded back.

        On Apple Silicon the load is refused when free memory, with what
        unloading the others would give back, is below what the model needs
        (utils/memory_check.py); ``force`` loads it anyway. ``persist=False``
        leaves config.json's current model alone — a one-off job that loads,
        generates and exits is nobody's current model. Why a load failed is
        left in ``last_error`` for the caller to show.
        """
        self.last_error = None
        if not self.is_model_installed(model_name):
            self.last_error = (f"Model {model_name} is not installed. "
                               f"Install it first: ollamadiffuser pull {model_name}")
            logger.error(self.last_error)
            return False

        # If the same model is already loaded, return directly
        if self.current_model_name == model_name:
            logger.info(f"Model {model_name} is already loaded")
            return True
        if model_name in self._others:
            self._come_forward(model_name)
            if persist:
                settings.set_current_model(model_name)
            logger.info(f"Model {model_name} is in memory already; now the current one")
            return True

        from ..utils import memory_check
        info = model_registry.get_model(model_name)
        # Refused before anything is unloaded when unloading everything
        # would still not make room: a request that cannot be served should
        # not cost the others their place.
        freeable = self._freeable_gb()
        if not force and freeable is not None:
            reason = memory_check.refusal(model_name, info, freeable_gb=freeable)
            if reason:
                self.last_error = reason
                logger.error(reason)
                return False

        previous = self._step_aside()
        order = list(self._others)
        unloaded = []
        if self.is_gguf_model(model_name):
            # stable-diffusion.cpp holds one model per process.
            for name in [n for n, (_, kind) in self._others.items() if kind == 'gguf']:
                unloaded.append(self._unload_other(name))
        while self._others and len(self._others) >= self.max_loaded:
            unloaded.append(self._unload_other(next(iter(self._others))))
        while self._others and memory_check.short_gb(model_name, info) > 0:
            unloaded.append(self._unload_other(next(iter(self._others))))

        # Checked after the unloads, which gave their memory back.
        reason = None if force else memory_check.refusal(model_name, info)
        if reason:
            self.last_error = reason
            logger.error(reason)
        elif self._load_new(model_name, persist):
            return True
        self._put_back(unloaded, order, previous)
        return False

    def _load_new(self, model_name: str, persist: bool) -> bool:
        """Load ``model_name`` as the current model, with no room made for it."""
        try:
            model_config = settings.models[model_name]

            # Refresh parameters from registry so new fields (e.g. single_file,
            # allow_patterns) are picked up without requiring a re-pull.
            registry_info = model_registry.get_model(model_name)
            if registry_info and registry_info.get("parameters"):
                saved_params = model_config.parameters or {}
                registry_params = registry_info["parameters"]
                # Registry is the base; saved overrides (preserves user customizations)
                merged = {**registry_params, **saved_params}
                if merged != saved_params:
                    model_config.parameters = merged
                    logger.info(f"Refreshed model parameters from registry for {model_name}")

            # Check if this is a GGUF model
            if self.is_gguf_model(model_name):
                if not GGUF_AVAILABLE:
                    self.last_error = "GGUF support not available. Enable it with: ollamadiffuser enable gguf"
                    logger.error(self.last_error)
                    return False

                # Load GGUF model
                model_config_dict = {
                    'name': model_name,
                    'path': model_config.path,
                    'variant': model_config.variant,
                    'model_type': model_config.model_type,
                    'parameters': model_config.parameters
                }
                
                if gguf_loader.load_model(model_config_dict):
                    self.loaded_model = gguf_loader
                    self.current_model_name = model_name
                    self.current_model_type = 'gguf'
                    if persist:
                        settings.set_current_model(model_name)
                    logger.info(f"GGUF model {model_name} loaded successfully")
                    return True
                else:
                    self.last_error = f"GGUF model {model_name} failed to load (see the log above)"
                    logger.error(self.last_error)
                    return False
            else:
                # Load regular diffusion model
                from ..inference.engine import InferenceEngine
                
                engine = InferenceEngine()
                
                if engine.load_model(model_config):
                    self.loaded_model = engine
                    self.current_model_name = model_name
                    self.current_model_type = 'diffusion'
                    if persist:
                        settings.set_current_model(model_name)
                    logger.info(f"Model {model_name} loaded successfully")
                    return True
                else:
                    self.last_error = f"Model {model_name} failed to load (see the log above)"
                    logger.error(self.last_error)
                    return False

        except Exception as e:
            self.last_error = f"Failed to load model {model_name}: {e}"
            logger.error(self.last_error)
            return False

    def unload_model(self, persist: bool = True, name: Optional[str] = None):
        """Unload ``name``, or every model in memory (supports both regular
        and GGUF models). When the current model goes and others stay, the
        most recently used of them becomes the current one.

        ``persist=False`` leaves config.json's current model as it is.
        """
        if name is not None and name != self.current_model_name:
            if name in self._others:
                self._unload_other(name)
            return
        if name is None:
            while self._others:
                self._unload_other(next(iter(self._others)))
        if self.loaded_model is not None:
            try:
                self._release(self.loaded_model, self.current_model_type, self.current_model_name)
            finally:
                self.loaded_model = None
                self.current_model_name = None
                self.current_model_type = None
        if self._others:
            self._come_forward(next(reversed(self._others)))

        # Also clear the persisted state
        if persist:
            settings.current_model = self.current_model_name
            settings.save_config()

    # --- Models kept in memory beside the current one ---

    def resident_models(self) -> List[str]:
        """The models in memory: the current one, then the others from the
        most recently used."""
        if self.loaded_model is None:
            return []
        return [self.current_model_name, *reversed(self._others)]

    def engine_of(self, model_name: str) -> Optional[object]:
        """The engine of ``model_name`` if it is in memory."""
        if model_name is not None and model_name == self.current_model_name:
            return self.loaded_model
        kept = self._others.get(model_name)
        return kept[0] if kept else None

    def held_gb(self, model_name: str) -> Optional[float]:
        """What ``model_name`` holds while it stays in memory (the engine's
        count, inference/engine.py), or None when there is no telling."""
        held = getattr(self.engine_of(model_name), "held_bytes", None)
        return held / 1e9 if isinstance(held, (int, float)) else None

    def make_room(self, model_name: str) -> List[str]:
        """Before a job on ``model_name``, the current model: unload others,
        least recently used first, while what the job adds to what the model
        holds would not fit beside them. Returns the ones unloaded."""
        if not self._others or model_name != self.current_model_name:
            return []
        from ..utils import memory_check
        info = model_registry.get_model(model_name)
        held = self.held_gb(model_name)
        unloaded = []
        while self._others and memory_check.short_gb(model_name, info, held_gb=held) > 0:
            unloaded.append(self._unload_other(next(iter(self._others))))
        return unloaded

    def _freeable_gb(self) -> Optional[float]:
        """What unloading every model in memory would give back; None when
        one of them cannot tell."""
        held = [self.held_gb(name) for name in self.resident_models()]
        return None if None in held else sum(held)

    def _step_aside(self) -> Optional[str]:
        """Move the current model in with the others, as the most recently used."""
        if self.loaded_model is None:
            return None
        name = self.current_model_name
        self._others[name] = (self.loaded_model, self.current_model_type)
        self._others.move_to_end(name)
        self.loaded_model = self.current_model_name = self.current_model_type = None
        return name

    def _come_forward(self, model_name: str) -> None:
        """Make ``model_name``, one of the others, the current model."""
        engine, kind = self._others.pop(model_name)
        self._step_aside()
        self.loaded_model, self.current_model_name, self.current_model_type = engine, model_name, kind

    def _unload_other(self, model_name: str) -> str:
        engine, kind = self._others.pop(model_name)
        self._release(engine, kind, model_name)
        return model_name

    @staticmethod
    def _release(engine, kind: Optional[str], model_name: Optional[str]) -> None:
        try:
            if kind == 'gguf':
                # Unload GGUF model
                gguf_loader.unload_model()
                logger.info(f"GGUF model {model_name} unloaded")
            else:
                # Unload regular model
                engine.unload()
                logger.info(f"Model {model_name} unloaded")
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")

    def _put_back(self, unloaded: List[str], order: List[str], previous: Optional[str]) -> None:
        """Load back the models unloaded for a load that did not happen, in
        their old order, and make ``previous`` the current model again."""
        reason = self.last_error
        for name in unloaded:
            if self._load_new(name, persist=False):
                self._step_aside()
            else:
                logger.error(f"Could not load {name} back: {self.last_error}")
        for name in order:
            if name in self._others:
                self._others.move_to_end(name)
        if previous in self._others:
            self._come_forward(previous)
        self.last_error = reason
    
    def get_current_model(self) -> Optional[str]:
        """Get current loaded model name"""
        # First check in-memory state
        if self.current_model_name:
            return self.current_model_name
        # Then check persisted state
        return settings.current_model
    
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded in memory"""
        # Only check in-memory state - a model is truly loaded only if it's in memory
        return self.loaded_model is not None
    
    def has_current_model(self) -> bool:
        """Check if there's a current model set (may not be loaded in memory)"""
        return settings.current_model is not None
    
    def is_server_running(self) -> bool:
        """Check if the server is actually running"""
        try:
            import requests
            host = settings.server.host
            port = settings.server.port
            response = requests.get(f"http://{host}:{port}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_current_model_info(self) -> Optional[Dict]:
        """Get information about the currently loaded model"""
        if not self.loaded_model or not self.current_model_name:
            return None
            
        model_info = self.get_model_info(self.current_model_name)
        if model_info:
            model_info['loaded'] = True
            model_info['type'] = self.current_model_type
            
            # Add GGUF-specific info if applicable
            if self.current_model_type == 'gguf':
                model_info.update(gguf_loader.get_model_info())
                
        return model_info

# Global model manager instance
model_manager = ModelManager() 