import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.server = ServerConfig()
        self.models: Dict[str, ModelConfig] = {}
        self.current_model: Optional[str] = None
        self.hf_token: Optional[str] = os.environ.get('HF_TOKEN')
        
        # Load configuration file
        self.load_config()
    
    def load_config(self):
        """Load settings from configuration file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load server configuration
                if 'server' in config_data:
                    server_data = config_data['server']
                    self.server = ServerConfig(**server_data)
                
                # Load model configuration
                if 'models' in config_data:
                    self.models = {
                        name: ModelConfig(**model_data)
                        for name, model_data in config_data['models'].items()
                    }
                
                self.current_model = config_data.get('current_model')

                # Load custom path overrides
                if 'paths' in config_data:
                    paths_data = config_data['paths']
                    if 'models_dir' in paths_data:
                        self.models_dir = Path(paths_data['models_dir'])
                    if 'cache_dir' in paths_data:
                        self.cache_dir = Path(paths_data['cache_dir'])
                    # Ensure custom directories exist
                    self.models_dir.mkdir(parents=True, exist_ok=True)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Configuration file loaded: {self.config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration file: {e}")
    
    def save_config(self):
        """Save settings to configuration file"""
        try:
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

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration file: {e}")
    
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