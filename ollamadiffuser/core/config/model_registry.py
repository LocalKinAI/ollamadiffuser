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
        """Load default hardcoded models for backward compatibility"""
        self._registry = {
            "flux.1-dev": {
                "repo_id": "black-forest-labs/FLUX.1-dev",
                "model_type": "flux",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 20,
                    "guidance_scale": 3.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 20,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 32,
                    "recommended_ram_gb": 64,
                    "disk_space_gb": 24,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "Requires significant VRAM. Consider using smaller variants for lower-end hardware."
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "flux.1-schnell": {
                "repo_id": "black-forest-labs/FLUX.1-schnell",
                "model_type": "flux",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 256
                },
                "hardware_requirements": {
                    "min_vram_gb": 16,
                    "recommended_vram_gb": 20,
                    "min_ram_gb": 24,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 24,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Faster variant with fewer steps. Good balance of speed and quality."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "flux.1-dev-gguf-q2k": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q2k",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 3,
                    "recommended_vram_gb": 6,
                    "min_ram_gb": 8,
                    "recommended_ram_gb": 16,
                    "disk_space_gb": 3,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Heavily quantized, lowest quality but very fast"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q3ks": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q3ks",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 4,
                    "recommended_vram_gb": 8,
                    "min_ram_gb": 10,
                    "recommended_ram_gb": 16,
                    "disk_space_gb": 4,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Light quantization, good speed/quality balance"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q4ks": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q4ks",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 6,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Recommended quantization level - good quality and speed"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q4-0": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q4-0",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 6,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Q4_0 quantization - fast inference with good quality"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q4-1": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q4-1",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 6,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Q4_1 quantization - improved Q4_0 with better accuracy"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q5ks": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q5ks",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 8,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Higher quality quantization, slower but better results"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q5-0": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q5-0",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 8,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Q5_0 quantization - good balance of size and quality"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q5-1": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q5-1",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 8,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Q5_1 quantization - improved Q5_0 with better accuracy"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q6k": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q6k",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 10,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 20,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 10,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "High quality quantization, close to original"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-q8": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-q8",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 18,
                    "min_ram_gb": 24,
                    "recommended_ram_gb": 36,
                    "disk_space_gb": 12,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Very high quality, minimal quantization loss"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },
            
            "flux.1-dev-gguf-f16": {
                "repo_id": "city96/FLUX.1-dev-gguf",
                "model_type": "flux",
                "variant": "gguf-f16",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 16,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 32,
                    "recommended_ram_gb": 48,
                    "disk_space_gb": 16,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Full precision, best quality but largest size"
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "stable-diffusion-3.5-medium": {
                "repo_id": "stabilityai/stable-diffusion-3.5-medium",
                "model_type": "sd3",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 10,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Best on NVIDIA RTX 3080+ or Apple M2 Pro+"
                }
            },
            "stable-diffusion-xl-base": {
                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "model_type": "sdxl",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Good on NVIDIA RTX 3070+ or Apple M1 Pro+"
                }
            },
            "stable-diffusion-1.5": {
                "repo_id": "runwayml/stable-diffusion-v1-5",
                "model_type": "sd15",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                },
                "hardware_requirements": {
                    "min_vram_gb": 4,
                    "recommended_vram_gb": 6,
                    "min_ram_gb": 8,
                    "recommended_ram_gb": 16,
                    "disk_space_gb": 5,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Runs well on most modern GPUs, including GTX 1060+"
                }
            },

            # ControlNet models for SD 1.5
            "controlnet-canny-sd15": {
                "repo_id": "lllyasviel/sd-controlnet-canny",
                "model_type": "controlnet_sd15",
                "base_model": "stable-diffusion-1.5",
                "controlnet_type": "canny",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "controlnet_conditioning_scale": 1.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 8,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Requires base SD 1.5 model + ControlNet model. Good for edge detection."
                }
            },

            "controlnet-depth-sd15": {
                "repo_id": "lllyasviel/sd-controlnet-depth",
                "model_type": "controlnet_sd15",
                "base_model": "stable-diffusion-1.5",
                "controlnet_type": "depth",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "controlnet_conditioning_scale": 1.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 8,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Requires base SD 1.5 model + ControlNet model. Good for depth-based control."
                }
            },

            "controlnet-openpose-sd15": {
                "repo_id": "lllyasviel/sd-controlnet-openpose",
                "model_type": "controlnet_sd15",
                "base_model": "stable-diffusion-1.5",
                "controlnet_type": "openpose",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "controlnet_conditioning_scale": 1.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 8,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Requires base SD 1.5 model + ControlNet model. Good for pose control."
                }
            },

            "controlnet-scribble-sd15": {
                "repo_id": "lllyasviel/sd-controlnet-scribble",
                "model_type": "controlnet_sd15",
                "base_model": "stable-diffusion-1.5",
                "controlnet_type": "scribble",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "controlnet_conditioning_scale": 1.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 8,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 20,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Requires base SD 1.5 model + ControlNet model. Good for sketch-based control."
                }
            },

            # ControlNet models for SDXL
            "controlnet-canny-sdxl": {
                "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
                "model_type": "controlnet_sdxl",
                "base_model": "stable-diffusion-xl-base",
                "controlnet_type": "canny",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "controlnet_conditioning_scale": 1.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 28,
                    "disk_space_gb": 10,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Requires base SDXL model + ControlNet model. Good for edge detection with SDXL quality."
                }
            },

            "controlnet-depth-sdxl": {
                "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
                "model_type": "controlnet_sdxl",
                "base_model": "stable-diffusion-xl-base",
                "controlnet_type": "depth",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "controlnet_conditioning_scale": 1.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 28,
                    "disk_space_gb": 10,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Requires base SDXL model + ControlNet model. Good for depth-based control with SDXL quality."
                }
            },

            # ===== Tier 1: Additional base models (use existing strategies) =====

            "stable-diffusion-3.5-large": {
                "repo_id": "stabilityai/stable-diffusion-3.5-large",
                "model_type": "sd3",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5
                },
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 24,
                    "recommended_ram_gb": 48,
                    "disk_space_gb": 16,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Large SD3.5 model. Higher quality than medium variant."
                },
                "license_info": {
                    "type": "Stability AI Community License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "stable-diffusion-3.5-large-turbo": {
                "repo_id": "stabilityai/stable-diffusion-3.5-large-turbo",
                "model_type": "sd3",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 4,
                    "guidance_scale": 0.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 24,
                    "recommended_ram_gb": 48,
                    "disk_space_gb": 16,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Turbo variant with 4-step generation. Fast inference."
                },
                "license_info": {
                    "type": "Stability AI Community License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "realvisxl-v4": {
                "repo_id": "SG161222/RealVisXL_V4.0",
                "model_type": "sdxl",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 5.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Photorealistic SDXL finetune. Very popular community model."
                },
                "license_info": {
                    "type": "CreativeML Open RAIL++-M",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "dreamshaper": {
                "repo_id": "Lykon/DreamShaper",
                "model_type": "sd15",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 4,
                    "recommended_vram_gb": 6,
                    "min_ram_gb": 8,
                    "recommended_ram_gb": 16,
                    "disk_space_gb": 5,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Popular community SD 1.5 model. Runs on most GPUs."
                },
                "license_info": {
                    "type": "CreativeML Open RAIL-M",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "realistic-vision-v6": {
                "repo_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                "model_type": "sd15",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 30,
                    "guidance_scale": 5.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 4,
                    "recommended_vram_gb": 6,
                    "min_ram_gb": 8,
                    "recommended_ram_gb": 16,
                    "disk_space_gb": 5,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Photorealistic SD 1.5 model. Great for portraits."
                },
                "license_info": {
                    "type": "CreativeML Open RAIL-M",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "sdxl-turbo": {
                "repo_id": "stabilityai/sdxl-turbo",
                "model_type": "sdxl",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 1,
                    "guidance_scale": 0.0
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Distilled SDXL. Single-step inference, extremely fast."
                },
                "license_info": {
                    "type": "Stability AI Non-Commercial Research Community License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            # ===== Tier 2: Scheduler override models =====

            "sdxl-lightning-4step": {
                "repo_id": "ByteDance/SDXL-Lightning",
                "model_type": "sdxl",
                "variant": "fp16",
                "parameters": {
                    "num_inference_steps": 4,
                    "guidance_scale": 0.0,
                    "scheduler_class": "EulerDiscreteScheduler",
                    "scheduler_kwargs": {
                        "timestep_spacing": "trailing"
                    }
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 12,
                    "recommended_ram_gb": 24,
                    "disk_space_gb": 7,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Lightning-fast SDXL with 4 steps."
                },
                "license_info": {
                    "type": "OpenRAIL++",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            # ===== Tier 3: FLUX pipeline variants =====

            "flux.1-fill-dev": {
                "repo_id": "black-forest-labs/FLUX.1-Fill-dev",
                "model_type": "flux",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "FluxFillPipeline",
                    "num_inference_steps": 28,
                    "guidance_scale": 30.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 20,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 32,
                    "recommended_ram_gb": 64,
                    "disk_space_gb": 24,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "FLUX inpainting/outpainting model."
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "flux.1-canny-dev": {
                "repo_id": "black-forest-labs/FLUX.1-Canny-dev",
                "model_type": "flux",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "FluxControlPipeline",
                    "num_inference_steps": 28,
                    "guidance_scale": 30.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 20,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 32,
                    "recommended_ram_gb": 64,
                    "disk_space_gb": 24,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "FLUX canny edge control model."
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "flux.1-depth-dev": {
                "repo_id": "black-forest-labs/FLUX.1-Depth-dev",
                "model_type": "flux",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "FluxControlPipeline",
                    "num_inference_steps": 28,
                    "guidance_scale": 10.0,
                    "max_sequence_length": 512
                },
                "hardware_requirements": {
                    "min_vram_gb": 20,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 32,
                    "recommended_ram_gb": 64,
                    "disk_space_gb": 24,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "FLUX depth control model."
                },
                "license_info": {
                    "type": "FLUX.1 Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            # ===== Tier 4: New-generation models (GenericPipelineStrategy) =====

            "flux.2-dev": {
                "repo_id": "black-forest-labs/FLUX.2-dev",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "Flux2Pipeline",
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5,
                    "torch_dtype": "bfloat16",
                    "enable_cpu_offload": True
                },
                "hardware_requirements": {
                    "min_vram_gb": 14,
                    "recommended_vram_gb": 24,
                    "min_ram_gb": 32,
                    "recommended_ram_gb": 64,
                    "disk_space_gb": 36,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "FLUX.2 32B parameter model. Requires CPU offloading on consumer GPUs. Requires diffusers from source."
                },
                "license_info": {
                    "type": "FLUX.2 Dev Non-Commercial License",
                    "requires_agreement": True,
                    "commercial_use": False
                }
            },

            "flux.2-klein-4b": {
                "repo_id": "black-forest-labs/FLUX.2-klein-4B",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "Flux2KleinPipeline",
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5,
                    "torch_dtype": "bfloat16"
                },
                "hardware_requirements": {
                    "min_vram_gb": 10,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 10,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Compact 4B FLUX.2 model. Apache 2.0 licensed. Requires diffusers from source."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "z-image-turbo": {
                "repo_id": "Tongyi-MAI/Z-Image-Turbo",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "ZImagePipeline",
                    "num_inference_steps": 8,
                    "guidance_scale": 5.0,
                    "torch_dtype": "bfloat16"
                },
                "hardware_requirements": {
                    "min_vram_gb": 10,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 14,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "Alibaba 6B model. 8-step turbo generation. Bilingual Chinese/English. Requires diffusers from source."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "sana-1.5": {
                "repo_id": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "SanaPipeline",
                    "num_inference_steps": 20,
                    "guidance_scale": 5.0,
                    "torch_dtype": "float16"
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 12,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 4,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "NVIDIA 1.6B model. Very efficient, competitive with 12B models."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "cogview4": {
                "repo_id": "THUDM/CogView4-6B",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "CogView4Pipeline",
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "torch_dtype": "bfloat16"
                },
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 18,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 14,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Zhipu AI 6B model with GLM-4 text encoder. Bilingual Chinese/English. Tight fit on 16GB Apple Silicon."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "kolors": {
                "repo_id": "Kwai-Kolors/Kolors-diffusers",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "KolorsPipeline",
                    "num_inference_steps": 50,
                    "guidance_scale": 5.0,
                    "torch_dtype": "float16",
                    "enable_cpu_offload": True
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 18,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 18,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Kuaishou 8.6B model with ChatGLM3 encoder. Bilingual Chinese/English. Midjourney-v6 level quality."
                },
                "license_info": {
                    "type": "Kolors License",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "hunyuan-dit": {
                "repo_id": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "HunyuanDiTPipeline",
                    "num_inference_steps": 50,
                    "guidance_scale": 5.0,
                    "torch_dtype": "float16",
                    "enable_cpu_offload": True
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 12,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Tencent 1.5B model with dual text encoders (mT5 + bilingual CLIP). Fine-grained Chinese understanding."
                },
                "license_info": {
                    "type": "Tencent Hunyuan Community License",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "lumina-2": {
                "repo_id": "Alpha-VLLM/Lumina-Image-2.0",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "Lumina2Pipeline",
                    "num_inference_steps": 30,
                    "guidance_scale": 4.0,
                    "torch_dtype": "bfloat16",
                    "enable_cpu_offload": True
                },
                "hardware_requirements": {
                    "min_vram_gb": 8,
                    "recommended_vram_gb": 16,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 6,
                    "supported_devices": ["CUDA", "MPS"],
                    "performance_notes": "Alpha-VLLM 2B model. Unified text+image token architecture. Works on Apple Silicon (tight on 16GB)."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "pixart-sigma": {
                "repo_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "PixArtSigmaPipeline",
                    "num_inference_steps": 20,
                    "guidance_scale": 4.5,
                    "torch_dtype": "float16"
                },
                "hardware_requirements": {
                    "min_vram_gb": 6,
                    "recommended_vram_gb": 10,
                    "min_ram_gb": 8,
                    "recommended_ram_gb": 16,
                    "disk_space_gb": 3,
                    "supported_devices": ["CUDA", "MPS", "CPU"],
                    "performance_notes": "Very lightweight 0.6B model. 4K capable. Outperforms much larger models."
                },
                "license_info": {
                    "type": "PixArt Open License",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "auraflow": {
                "repo_id": "fal/AuraFlow-v0.3",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "AuraFlowPipeline",
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "torch_dtype": "float16",
                    "enable_cpu_offload": True
                },
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 20,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 14,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "Fal 6.8B model. Largest Apache 2.0 licensed text-to-image model."
                },
                "license_info": {
                    "type": "Apache 2.0",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            },

            "omnigen": {
                "repo_id": "Shitao/OmniGen-v1-diffusers",
                "model_type": "generic",
                "variant": "fp16",
                "parameters": {
                    "pipeline_class": "OmniGenPipeline",
                    "num_inference_steps": 50,
                    "guidance_scale": 2.5,
                    "torch_dtype": "bfloat16",
                    "enable_cpu_offload": True,
                    "supports_negative_prompt": False
                },
                "hardware_requirements": {
                    "min_vram_gb": 12,
                    "recommended_vram_gb": 18,
                    "min_ram_gb": 16,
                    "recommended_ram_gb": 32,
                    "disk_space_gb": 10,
                    "supported_devices": ["CUDA"],
                    "performance_notes": "BAAI 3.8B unified model. Text-to-image, editing, subject-driven generation without extra plugins."
                },
                "license_info": {
                    "type": "MIT",
                    "requires_agreement": False,
                    "commercial_use": True
                }
            }
        }
    
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