#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) manager for downloading and managing LoRA weights
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
from huggingface_hub import hf_hub_download, login

from ..config.settings import settings
from .download_utils import robust_file_download

logger = logging.getLogger(__name__)

class LoRAManager:
    """Manager for LoRA weights"""
    
    def __init__(self):
        self.lora_dir = settings.config_dir / "loras"
        self.lora_dir.mkdir(exist_ok=True)
        self.config_file = self.lora_dir / "loras.json"
        self.current_lora = None
        self._load_config()
    
    def _load_config(self):
        """Load LoRA configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load LoRA config: {e}")
                self.config = {}
        else:
            self.config = {}
    
    def _save_config(self):
        """Save LoRA configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save LoRA config: {e}")
    
    def _get_lora_path(self, lora_name: str) -> Path:
        """Get path for LoRA storage"""
        return self.lora_dir / lora_name
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")
        return total_size
    
    def _is_server_running(self) -> bool:
        """Check if the API server is running"""
        try:
            import requests
            response = requests.get(f"http://{settings.server.host}:{settings.server.port}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _try_load_lora_via_api(self, lora_name: str, scale: float = 1.0) -> bool:
        """Try to load LoRA via API server"""
        try:
            if not self._is_server_running():
                return False
            
            # Check if LoRA exists
            if lora_name not in self.config:
                logger.error(f"LoRA {lora_name} not found")
                return False
            
            lora_info = self.config[lora_name]
            
            import requests
            
            # Prepare API request
            api_data = {
                "lora_name": lora_name,
                "repo_id": lora_info["repo_id"],
                "scale": scale
            }
            
            if "weight_name" in lora_info:
                api_data["weight_name"] = lora_info["weight_name"]
            
            # Make API request to load LoRA
            response = requests.post(
                f"http://{settings.server.host}:{settings.server.port}/api/lora/load",
                json=api_data,
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_lora = lora_name
                logger.info(f"LoRA {lora_name} loaded successfully via API with scale {scale}")
                return True
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load LoRA via API: {e}")
            return False
    
    def _try_unload_lora_via_api(self) -> bool:
        """Try to unload LoRA via API server"""
        try:
            if not self._is_server_running():
                return False
            
            import requests
            
            # Make API request to unload LoRA
            response = requests.post(
                f"http://{settings.server.host}:{settings.server.port}/api/lora/unload",
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_lora = None
                logger.info("LoRA unloaded successfully via API")
                return True
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload LoRA via API: {e}")
            return False
    
    def pull_lora(self, repo_id: str, weight_name: Optional[str] = None, 
                  alias: Optional[str] = None, progress_callback: Optional[Callable] = None) -> bool:
        """Download LoRA weights from Hugging Face Hub"""
        try:
            # Determine local name
            lora_name = alias if alias else repo_id.replace('/', '_')
            lora_path = self._get_lora_path(lora_name)
            
            # Check if already exists
            if lora_name in self.config and lora_path.exists():
                if progress_callback:
                    progress_callback(f"✅ LoRA {lora_name} already exists")
                logger.info(f"LoRA {lora_name} already exists")
                return True
            
            # Create directory
            lora_path.mkdir(exist_ok=True)
            
            # Ensure HuggingFace token is set
            if settings.hf_token:
                login(token=settings.hf_token)
                if progress_callback:
                    progress_callback(f"🔑 Authenticated with HuggingFace")
            
            if progress_callback:
                progress_callback(f"📥 Downloading LoRA from {repo_id}")
            
            # Download specific weight file or all files
            if weight_name:
                # Download specific file
                downloaded_file = robust_file_download(
                    repo_id=repo_id,
                    filename=weight_name,
                    local_dir=str(lora_path),
                    cache_dir=str(settings.cache_dir),
                    progress_callback=progress_callback
                )
                
                # Store metadata
                lora_info = {
                    "repo_id": repo_id,
                    "weight_name": weight_name,
                    "path": str(lora_path),
                    "downloaded_at": datetime.now().isoformat(),
                    "size": self._format_size(self._get_directory_size(lora_path))
                }
            else:
                # Download all files (fallback)
                from .download_utils import robust_snapshot_download
                robust_snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(lora_path),
                    cache_dir=str(settings.cache_dir),
                    progress_callback=progress_callback
                )
                
                # Store metadata
                lora_info = {
                    "repo_id": repo_id,
                    "path": str(lora_path),
                    "downloaded_at": datetime.now().isoformat(),
                    "size": self._format_size(self._get_directory_size(lora_path))
                }
            
            # Update configuration
            self.config[lora_name] = lora_info
            self._save_config()
            
            logger.info(f"LoRA {lora_name} downloaded successfully")
            if progress_callback:
                progress_callback(f"✅ LoRA {lora_name} downloaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download LoRA: {e}")
            if progress_callback:
                progress_callback(f"❌ Failed to download LoRA: {e}")
            
            # Clean up failed download
            if lora_path.exists():
                try:
                    shutil.rmtree(lora_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up failed download: {cleanup_error}")
            
            return False
    
    def load_lora(self, lora_name: str, scale: float = 1.0) -> bool:
        """Load LoRA weights into the current model"""
        try:
            from ..models.manager import model_manager
            
            # Check if model is loaded locally
            if not model_manager.is_model_loaded():
                # Try to load via API if server is running
                if self._try_load_lora_via_api(lora_name, scale):
                    return True
                logger.error("No model is currently loaded")
                return False
            
            # Check if LoRA exists
            if lora_name not in self.config:
                logger.error(f"LoRA {lora_name} not found")
                return False
            
            lora_info = self.config[lora_name]
            lora_path = Path(lora_info["path"])
            
            if not lora_path.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return False
            
            # Get the inference engine
            engine = model_manager.loaded_model
            if not engine:
                logger.error("No inference engine available")
                return False
            
            # Load LoRA weights
            if "weight_name" in lora_info:
                # The file `lora pull` downloaded, when it is there: handing
                # over the hub id instead sends the backend to fetch it again
                # (mflux into a cache of its own).
                pulled = lora_path / lora_info["weight_name"]
                success = engine.load_lora_runtime(
                    repo_id=str(lora_path) if pulled.is_file() else lora_info["repo_id"],
                    weight_name=lora_info["weight_name"],
                    scale=scale
                )
            else:
                # Load from local directory
                weight_files = list(lora_path.glob("*.safetensors"))
                if not weight_files:
                    weight_files = list(lora_path.glob("*.bin"))
                
                if not weight_files:
                    logger.error(f"No weight files found in {lora_path}")
                    return False
                
                # Use the first weight file found
                weight_file = weight_files[0]
                success = engine.load_lora_runtime(
                    repo_id=str(lora_path),
                    weight_name=weight_file.name,
                    scale=scale
                )
            
            if success:
                self.current_lora = lora_name
                logger.info(f"LoRA {lora_name} loaded successfully with scale {scale}")
                return True
            else:
                logger.error(f"Failed to load LoRA {lora_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False
    
    def unload_lora(self) -> bool:
        """Unload current LoRA weights"""
        try:
            from ..models.manager import model_manager
            
            # Check if model is loaded locally
            if not model_manager.is_model_loaded():
                # Try to unload via API if server is running
                if self._try_unload_lora_via_api():
                    return True
                logger.error("No model is currently loaded")
                return False
            
            # Get the inference engine
            engine = model_manager.loaded_model
            if not engine:
                logger.error("No inference engine available")
                return False
            
            # Unload LoRA weights
            success = engine.unload_lora()
            
            if success:
                self.current_lora = None
                logger.info("LoRA weights unloaded successfully")
                return True
            else:
                logger.error("Failed to unload LoRA weights")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            return False
    
    def remove_lora(self, lora_name: str) -> bool:
        """Remove LoRA weights"""
        try:
            # Check if LoRA exists
            if lora_name not in self.config:
                logger.error(f"LoRA {lora_name} not found")
                return False
            
            # Unload if currently loaded
            if self.current_lora == lora_name:
                self.unload_lora()
            
            # Remove files
            lora_info = self.config[lora_name]
            lora_path = Path(lora_info["path"])
            
            if lora_path.exists():
                shutil.rmtree(lora_path)
            
            # Remove from configuration
            del self.config[lora_name]
            self._save_config()
            
            logger.info(f"LoRA {lora_name} removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove LoRA: {e}")
            return False
    
    def list_installed_loras(self) -> Dict[str, Dict]:
        """List all installed LoRA weights"""
        return self.config.copy()
    
    def get_current_lora(self) -> Optional[str]:
        """Get currently loaded LoRA name"""
        return self.current_lora
    
    def get_lora_info(self, lora_name: str) -> Optional[Dict]:
        """Get information about a specific LoRA"""
        return self.config.get(lora_name)
    
    def is_lora_installed(self, lora_name: str) -> bool:
        """Check if LoRA is installed"""
        return lora_name in self.config

# Global LoRA manager instance
lora_manager = LoRAManager() 