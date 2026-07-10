"""
Model Registry Configuration

This file contains the default model definitions and provides a system
for managing models externally without hardcoding them in the manager.
"""

import os
import json
import yaml
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from .settings import settings
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Dynamic model registry that supports external model definitions"""
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._external_registries: List[str] = []
        self._external_api_models: Dict[str, Dict[str, Any]] = {}
        self._load_default_models()
        self._load_external_models()
        # Load external API models on initialization
        self._refresh_external_api_models()
    
    def _get_model_manager(self):
        """Get model manager instance (lazy import to avoid circular imports)"""
        try:
            from ..models.manager import ModelManager
            if not hasattr(self, '_model_manager'):
                self._model_manager = ModelManager()
            return self._model_manager
        except ImportError:
            return None
    
    def _load_default_models(self):
        """Load built-in models from the bundled models.yaml data file.

        The model list lives in ``models.yaml`` (data, not code) so that
        adding a model is a pure-data change — no Python edits, no code
        review of logic. User overrides still load afterward via
        :meth:`_load_external_models`.
        """
        models_file = Path(__file__).parent / "models.yaml"
        try:
            with open(models_file, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError:
            logger.error("Bundled models.yaml not found at %s", models_file)
            self._registry = {}
            return
        self._registry = (data or {}).get("models", {}) or {}

    def _load_external_models(self):
        """Load models from external configuration files"""
        # Check for user-defined model configurations
        config_paths = [
            settings.config_dir / "models.json",
            settings.config_dir / "models.yaml",
            settings.config_dir / "models.yml",
            Path.home() / ".ollamadiffuser" / "models.json",
            Path.home() / ".ollamadiffuser" / "models.yaml",
            Path.home() / ".ollamadiffuser" / "models.yml",
        ]
        
        # Also check environment variable for custom model config path
        if "OLLAMADIFFUSER_MODEL_CONFIG" in os.environ:
            config_paths.append(Path(os.environ["OLLAMADIFFUSER_MODEL_CONFIG"]))
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    self._load_config_file(config_path)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to load model config from {config_path}: {e}")
    
    def _load_config_file(self, config_path: Path):
        """Load models from a configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                data = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        if 'models' in data:
            for model_name, model_config in data['models'].items():
                self._registry[model_name] = model_config
        
        self._external_registries.append(str(config_path))
    
    def _refresh_external_api_models(self):
        """Refresh external API models cache"""
        self._external_api_models = self._fetch_external_api_models()
    
    def _get_combined_models(self) -> Dict[str, Dict[str, Any]]:
        """Get combined models (local + external API), with local taking precedence"""
        combined_models = {}
        combined_models.update(self._external_api_models)  # Add external API models first
        combined_models.update(self._registry)  # Local models override external ones
        return combined_models
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered models including external API models"""
        return self._get_combined_models()
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific model configuration from local or external sources"""
        # Check local registry first
        if model_name in self._registry:
            return self._registry[model_name]
        
        # Check external API models
        if model_name in self._external_api_models:
            return self._external_api_models[model_name]
        
        return None
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names including external API models"""
        return list(self._get_combined_models().keys())
    
    def get_installed_models(self) -> Dict[str, Dict[str, Any]]:
        """Get only actually installed models (from settings.models)"""
        model_manager = self._get_model_manager()
        if model_manager is None:
            return {}
        
        installed_models = {}
        installed_model_names = model_manager.list_installed_models()
        
        for model_name in installed_model_names:
            model_config = self.get_model(model_name)
            if model_config:
                installed_models[model_name] = model_config
        
        return installed_models
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available but not installed models"""
        model_manager = self._get_model_manager()
        if model_manager is None:
            return self._get_combined_models()
        
        installed_model_names = set(model_manager.list_installed_models())
        all_models = self._get_combined_models()
        
        available_models = {}
        for model_name, model_config in all_models.items():
            if model_name not in installed_model_names:
                available_models[model_name] = model_config
        
        return available_models
    
    def is_model_installed(self, model_name: str) -> bool:
        """Check if a model is actually installed"""
        model_manager = self._get_model_manager()
        if model_manager is None:
            return False
        return model_manager.is_model_installed(model_name)
    
    def add_model(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """Add a new model to the local registry (runtime only)"""
        try:
            # Validate required fields
            required_fields = ['repo_id', 'model_type']
            for field in required_fields:
                if field not in model_config:
                    raise ValueError(f"Missing required field: {field}")
            
            self._registry[model_name] = model_config
            return True
        except Exception:
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the local registry (runtime only)"""
        if model_name in self._registry:
            del self._registry[model_name]
            return True
        return False
    
    def reload(self):
        """Reload the model registry including external API models"""
        self._registry.clear()
        self._external_registries.clear()
        self._external_api_models.clear()
        self._load_default_models()
        self._load_external_models()
        self._refresh_external_api_models()
    
    def save_user_config(self, models: Dict[str, Dict[str, Any]], config_path: Optional[Path] = None):
        """Save user-defined models to a configuration file"""
        if config_path is None:
            config_path = settings.config_dir / "models.json"
        
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {"models": models}
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def get_external_registries(self) -> List[str]:
        """Get list of external registry files that were loaded"""
        return self._external_registries.copy()
    
    def refresh_external_models(self):
        """Manually refresh external API models"""
        self._refresh_external_api_models()
    
    def get_local_models_only(self) -> Dict[str, Dict[str, Any]]:
        """Get only locally defined models (from registry, not necessarily installed)"""
        return self._registry.copy()
    
    def get_external_api_models_only(self) -> Dict[str, Dict[str, Any]]:
        """Get only external API models"""
        return self._external_api_models.copy()

    def _fetch_external_api_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch models from external API"""
        try:
            response = requests.get("https://www.ollamadiffuser.com/api/models", timeout=10)
            if response.status_code == 200:
                api_data = response.json()
                # Expected format: {"models": {"model_name": {...}, ...}}
                if "models" in api_data:
                    return api_data["models"]
                else:
                    # If the API returns a different format, adapt accordingly
                    return api_data if isinstance(api_data, dict) else {}
            else:
                return {}
        except Exception as e:
            # Log the error but don't fail completely
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to fetch external API models: {e}")
            return {}
    
    def get_all_models_with_external(self) -> Dict[str, Dict[str, Any]]:
        """Get all models including those from external API (deprecated - use get_all_models)"""
        # This method is now redundant since get_all_models includes external by default
        return self.get_all_models()


# Global model registry instance
model_registry = ModelRegistry() 