"""OllamaDiffuser for ComfyUI: its MLX models as nodes, through its HTTP API.

Install by putting this folder (or a link to it) in ComfyUI's custom_nodes.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
