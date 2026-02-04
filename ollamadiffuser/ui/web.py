import asyncio
import io
import base64
import logging
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from ..core.models.manager import model_manager
from ..core.utils.lora_manager import lora_manager
from ..core.utils.controlnet_preprocessors import controlnet_preprocessor

logger = logging.getLogger(__name__)

# Get templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def ensure_samples_exist(samples_dir: Path):
    """Ensure ControlNet sample images exist, create them if missing"""
    try:
        # Check if samples directory exists and has content
        if not samples_dir.exists() or not any(samples_dir.iterdir()):
            logger.info("Creating ControlNet sample images...")
            samples_dir.mkdir(exist_ok=True)
            
            # Import and run the sample creation functions from the standalone script
            # We'll reimplement the functions here to avoid importing the standalone script
            _create_controlnet_samples(samples_dir)
            logger.info("ControlNet sample images created successfully")
        else:
            # Check if all required sample types exist
            required_types = ['canny', 'depth', 'openpose', 'scribble']
            missing_types = []
            
            for sample_type in required_types:
                type_dir = samples_dir / sample_type
                if not type_dir.exists() or not any(type_dir.iterdir()):
                    missing_types.append(sample_type)
            
            if missing_types:
                logger.info(f"Creating missing sample types: {missing_types}")
                _create_controlnet_samples(samples_dir, only_types=missing_types)
                logger.info("Missing sample types created successfully")
    except Exception as e:
        logger.warning(f"Failed to create sample images: {e}")
        
def _create_controlnet_samples(samples_dir: Path, only_types=None):
    """Create ControlNet sample images"""
    import numpy as np
    import math
    from PIL import ImageDraw
    
    # Create sample directories
    sample_types = only_types or ['canny', 'depth', 'openpose', 'scribble']
    for sample_type in sample_types:
        (samples_dir / sample_type).mkdir(exist_ok=True)
    
    if 'canny' in sample_types:
        # 1. Simple geometric shapes
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 150], outline='black', width=3)
        draw.ellipse([300, 50, 450, 200], outline='black', width=3)
        draw.polygon([(100, 300), (200, 200), (300, 300)], outline='black', width=3)
        draw.polygon([(400, 250), (450, 300), (400, 350), (350, 300)], outline='black', width=3)
        img.save(samples_dir / 'canny' / 'geometric_shapes.png')
        
        # 2. Simple house outline
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([150, 250, 350, 400], outline='black', width=3)
        draw.polygon([(130, 250), (250, 150), (370, 250)], outline='black', width=3)
        draw.rectangle([220, 320, 280, 400], outline='black', width=2)
        draw.rectangle([170, 280, 210, 320], outline='black', width=2)
        draw.rectangle([290, 280, 330, 320], outline='black', width=2)
        draw.rectangle([300, 170, 330, 220], outline='black', width=2)
        img.save(samples_dir / 'canny' / 'house_outline.png')
        
        # 3. Portrait silhouette
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([180, 100, 330, 280], outline='black', width=3)
        draw.rectangle([235, 280, 275, 320], outline='black', width=3)
        draw.arc([150, 300, 360, 450], start=0, end=180, fill='black', width=3)
        img.save(samples_dir / 'canny' / 'portrait_outline.png')
    
    if 'depth' in sample_types:
        # 1. Radial gradient
        img = Image.new('RGB', (512, 512), 'white')
        pixels = np.zeros((512, 512, 3), dtype=np.uint8)
        center_x, center_y = 256, 256
        max_distance = 200
        for y in range(512):
            for x in range(512):
                distance = min(np.sqrt((x - center_x)**2 + (y - center_y)**2), max_distance)
                intensity = int(255 * (1 - distance / max_distance))
                pixels[y, x] = [intensity, intensity, intensity]
        Image.fromarray(pixels).save(samples_dir / 'depth' / 'radial_gradient.png')
        
        # 2. Linear perspective
        pixels = np.zeros((512, 512, 3), dtype=np.uint8)
        for y in range(512):
            intensity = int(255 * (y / 512))
            pixels[y, :] = [intensity, intensity, intensity]
        Image.fromarray(pixels).save(samples_dir / 'depth' / 'linear_perspective.png')
        
        # 3. Simple 3D sphere
        pixels = np.zeros((512, 512, 3), dtype=np.uint8)
        center_x, center_y = 256, 256
        radius = 150
        for y in range(512):
            for x in range(512):
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    z = np.sqrt(radius**2 - distance**2)
                    intensity = int(255 * (z / radius))
                    pixels[y, x] = [intensity, intensity, intensity]
        Image.fromarray(pixels).save(samples_dir / 'depth' / 'sphere_3d.png')
    
    if 'openpose' in sample_types:
        # 1. Standing pose
        img = Image.new('RGB', (512, 512), 'black')
        draw = ImageDraw.Draw(img)
        draw.ellipse([240, 80, 270, 110], fill='white')
        draw.line([255, 110, 255, 250], fill='white', width=4)
        draw.line([255, 150, 200, 200], fill='white', width=4)
        draw.line([255, 150, 310, 200], fill='white', width=4)
        draw.line([255, 250, 220, 350], fill='white', width=4)
        draw.line([255, 250, 290, 350], fill='white', width=4)
        img.save(samples_dir / 'openpose' / 'standing_pose.png')
        
        # 2. Running pose
        img = Image.new('RGB', (512, 512), 'black')
        draw = ImageDraw.Draw(img)
        draw.ellipse([240, 80, 270, 110], fill='white')
        draw.line([255, 110, 270, 250], fill='white', width=4)
        draw.line([255, 150, 180, 180], fill='white', width=4)
        draw.line([255, 150, 320, 120], fill='white', width=4)
        draw.line([270, 250, 240, 350], fill='white', width=4)
        draw.line([270, 250, 320, 320], fill='white', width=4)
        img.save(samples_dir / 'openpose' / 'running_pose.png')
        
        # 3. Sitting pose
        img = Image.new('RGB', (512, 512), 'black')
        draw = ImageDraw.Draw(img)
        draw.ellipse([240, 100, 270, 130], fill='white')
        draw.line([255, 130, 255, 220], fill='white', width=4)
        draw.line([255, 170, 200, 220], fill='white', width=4)
        draw.line([255, 170, 310, 220], fill='white', width=4)
        draw.line([255, 220, 220, 280], fill='white', width=4)
        draw.line([220, 280, 200, 350], fill='white', width=4)
        draw.line([255, 220, 290, 280], fill='white', width=4)
        draw.line([290, 280, 310, 350], fill='white', width=4)
        img.save(samples_dir / 'openpose' / 'sitting_pose.png')
    
    if 'scribble' in sample_types:
        # 1. Simple tree sketch
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        draw.line([256, 400, 256, 250], fill='black', width=8)
        points = []
        for i in range(20):
            angle = (i / 20) * 2 * math.pi
            radius = 80 + 20 * math.sin(i * 3)
            x = 256 + radius * math.cos(angle)
            y = 200 + radius * math.sin(angle) * 0.8
            points.append((x, y))
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            draw.line([points[i], points[next_i]], fill='black', width=3)
        img.save(samples_dir / 'scribble' / 'tree_sketch.png')
        
        # 2. Simple face sketch
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([180, 150, 330, 320], outline='black', width=3)
        draw.ellipse([210, 200, 230, 220], outline='black', width=2)
        draw.ellipse([280, 200, 300, 220], outline='black', width=2)
        draw.line([255, 230, 255, 250], fill='black', width=2)
        draw.line([255, 250, 245, 260], fill='black', width=2)
        draw.arc([230, 270, 280, 300], start=0, end=180, fill='black', width=2)
        img.save(samples_dir / 'scribble' / 'face_sketch.png')
        
        # 3. Simple car sketch
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 250, 400, 320], outline='black', width=3)
        draw.rectangle([150, 200, 350, 250], outline='black', width=3)
        draw.ellipse([130, 320, 170, 360], outline='black', width=3)
        draw.ellipse([330, 320, 370, 360], outline='black', width=3)
        draw.rectangle([170, 210, 220, 240], outline='black', width=2)
        draw.rectangle([280, 210, 330, 240], outline='black', width=2)
        img.save(samples_dir / 'scribble' / 'car_sketch.png')
    
    # Create metadata file
    metadata = {
        "canny": {
            "geometric_shapes.png": {
                "title": "Geometric Shapes",
                "description": "Perfect for generating architectural elements, logos, or geometric art",
                "good_for": ["architecture", "logos", "geometric patterns", "modern art"]
            },
            "house_outline.png": {
                "title": "House Outline", 
                "description": "Great for generating buildings, houses, or architectural scenes",
                "good_for": ["buildings", "houses", "architecture", "real estate"]
            },
            "portrait_outline.png": {
                "title": "Portrait Silhouette",
                "description": "Ideal for generating portraits, characters, or people",
                "good_for": ["portraits", "characters", "people", "headshots"]
            }
        },
        "depth": {
            "radial_gradient.png": {
                "title": "Radial Depth",
                "description": "Perfect for centered subjects with depth, like portraits or objects",
                "good_for": ["portraits", "centered objects", "product photography", "focus effects"]
            },
            "linear_perspective.png": {
                "title": "Linear Perspective", 
                "description": "Great for landscapes, roads, or scenes with distance",
                "good_for": ["landscapes", "roads", "horizons", "perspective scenes"]
            },
            "sphere_3d.png": {
                "title": "3D Sphere",
                "description": "Ideal for round objects, balls, or 3D elements",
                "good_for": ["spheres", "balls", "3D objects", "rounded elements"]
            }
        },
        "openpose": {
            "standing_pose.png": {
                "title": "Standing Pose",
                "description": "Basic standing position, great for portraits and character art",
                "good_for": ["standing portraits", "character design", "fashion", "formal poses"]
            },
            "running_pose.png": {
                "title": "Running Pose",
                "description": "Dynamic action pose, perfect for sports or movement scenes",
                "good_for": ["sports", "action scenes", "dynamic poses", "movement"]
            },
            "sitting_pose.png": {
                "title": "Sitting Pose",
                "description": "Relaxed sitting position, ideal for casual or indoor scenes",
                "good_for": ["casual portraits", "indoor scenes", "relaxed poses", "sitting figures"]
            }
        },
        "scribble": {
            "tree_sketch.png": {
                "title": "Tree Sketch",
                "description": "Simple tree drawing, great for nature and landscape scenes",
                "good_for": ["nature", "landscapes", "trees", "outdoor scenes"]
            },
            "face_sketch.png": {
                "title": "Face Sketch",
                "description": "Basic face outline, perfect for portrait generation",
                "good_for": ["portraits", "faces", "character art", "headshots"]
            },
            "car_sketch.png": {
                "title": "Car Sketch",
                "description": "Simple vehicle outline, ideal for automotive or transportation themes",
                "good_for": ["cars", "vehicles", "transportation", "automotive"]
            }
        }
    }
    
    with open(samples_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def create_ui_app() -> FastAPI:
    """Create Web UI application"""
    app = FastAPI(title="OllamaDiffuser Web UI")
    
    # Mount static files for samples
    samples_dir = Path(__file__).parent / "samples"
    logger.info(f"Samples directory: {samples_dir}")
    logger.info(f"Samples directory exists: {samples_dir.exists()}")
    
    # Ensure samples exist before mounting
    ensure_samples_exist(samples_dir)
    
    if samples_dir.exists():
        logger.info(f"Mounting samples directory: {samples_dir}")
        app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")
    else:
        logger.warning(f"Samples directory not found: {samples_dir}")
    
    def get_template_context(request: Request):
        """Get common template context"""
        models = model_manager.list_available_models()
        installed_models = model_manager.list_installed_models()
        current_model = model_manager.get_current_model()
        model_loaded = model_manager.is_model_loaded()
        
        # Get LoRA information
        installed_loras = lora_manager.list_installed_loras()
        current_lora = lora_manager.get_current_lora()
        
        # Check if current model is ControlNet
        is_controlnet_model = False
        controlnet_type = None
        model_parameters = {}
        if current_model and model_loaded:
            engine = model_manager.loaded_model
            if hasattr(engine, 'is_controlnet_pipeline'):
                is_controlnet_model = engine.is_controlnet_pipeline
                if is_controlnet_model:
                    # Get ControlNet type from model info
                    model_info = model_manager.get_model_info(current_model)
                    controlnet_type = model_info.get('controlnet_type', 'canny') if model_info else 'canny'
            
            # Get model parameters for current model
            model_info = model_manager.get_model_info(current_model)
            if model_info and 'parameters' in model_info:
                model_parameters = model_info['parameters']
        
        # Get available ControlNet preprocessors (without initializing)
        available_preprocessors = controlnet_preprocessor.get_available_types()
        
        # Load sample metadata
        sample_metadata = {}
        metadata_file = samples_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    sample_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sample metadata: {e}")
        
        return {
            "request": request,
            "models": models,
            "installed_models": installed_models,
            "current_model": current_model,
            "model_loaded": model_loaded,
            "installed_loras": installed_loras,
            "current_lora": current_lora,
            "is_controlnet_model": is_controlnet_model,
            "controlnet_type": controlnet_type,
            "available_preprocessors": available_preprocessors,
            "controlnet_available": controlnet_preprocessor.is_available(),
            "controlnet_initialized": controlnet_preprocessor.is_initialized(),
            "sample_metadata": sample_metadata,
            "model_parameters": model_parameters
        }
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page"""
        return templates.TemplateResponse("index.html", get_template_context(request))
    
    @app.post("/generate")
    async def generate_image_ui(
        request: Request,
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality, low resolution"),
        num_inference_steps: int = Form(28),
        guidance_scale: float = Form(3.5),
        width: int = Form(1024),
        height: int = Form(1024),
        seed: Optional[int] = Form(None),
        control_image: UploadFile = File(None),
        controlnet_conditioning_scale: float = Form(1.0),
        control_guidance_start: float = Form(0.0),
        control_guidance_end: float = Form(1.0),
    ):
        """Generate image (Web UI) - runs generation in thread pool"""
        error_message = None
        image_b64 = None
        control_image_b64 = None

        try:
            if not model_manager.is_model_loaded():
                error_message = "No model loaded. Please load a model first using the model management section above."

            if not error_message:
                engine = model_manager.loaded_model
                if engine is None:
                    error_message = "Model engine is not available. Please reload the model."
                else:
                    # Process control image if provided
                    control_image_pil = None
                    if control_image and control_image.filename:
                        if not controlnet_preprocessor.is_initialized():
                            if not controlnet_preprocessor.initialize():
                                error_message = "Failed to initialize ControlNet preprocessors."

                        if not error_message:
                            image_data = await control_image.read()
                            control_image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
                            img_buffer = io.BytesIO()
                            control_image_pil.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            control_image_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                    if not error_message:
                        image = await asyncio.to_thread(
                            engine.generate_image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            seed=seed,
                            control_image=control_image_pil,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            control_guidance_start=control_guidance_start,
                            control_guidance_end=control_guidance_end,
                        )

                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        image_b64 = base64.b64encode(img_buffer.getvalue()).decode()

        except Exception as e:
            error_message = f"Image generation failed: {str(e)}"

        context = get_template_context(request)
        context.update({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "image_b64": image_b64,
            "control_image_b64": control_image_b64,
            "error_message": error_message,
        })

        return templates.TemplateResponse("index.html", context)

    @app.post("/generate/img2img")
    async def generate_img2img_ui(
        request: Request,
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality, low resolution"),
        num_inference_steps: int = Form(28),
        guidance_scale: float = Form(3.5),
        seed: Optional[int] = Form(None),
        strength: float = Form(0.75),
        image: UploadFile = File(...),
    ):
        """Image-to-image generation (Web UI)"""
        error_message = None
        image_b64 = None
        input_image_b64 = None

        try:
            if not model_manager.is_model_loaded():
                error_message = "No model loaded."
            else:
                engine = model_manager.loaded_model
                image_data = await image.read()
                input_image = Image.open(io.BytesIO(image_data)).convert('RGB')

                # Show the input image
                buf = io.BytesIO()
                input_image.save(buf, format='PNG')
                buf.seek(0)
                input_image_b64 = base64.b64encode(buf.getvalue()).decode()

                result = await asyncio.to_thread(
                    engine.generate_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=input_image.width,
                    height=input_image.height,
                    seed=seed,
                    image=input_image,
                    strength=strength,
                )

                buf = io.BytesIO()
                result.save(buf, format='PNG')
                buf.seek(0)
                image_b64 = base64.b64encode(buf.getvalue()).decode()

        except Exception as e:
            error_message = f"img2img generation failed: {str(e)}"

        context = get_template_context(request)
        context.update({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "image_b64": image_b64,
            "input_image_b64": input_image_b64,
            "error_message": error_message,
        })

        return templates.TemplateResponse("index.html", context)
    
    @app.post("/preprocess_control_image")
    async def preprocess_control_image_ui(
        request: Request,
        control_type: str = Form(...),
        image: UploadFile = File(...)
    ):
        """Preprocess control image (Web UI)"""
        try:
            # Initialize ControlNet preprocessors if needed
            if not controlnet_preprocessor.is_initialized():
                logger.info("Initializing ControlNet preprocessors for image preprocessing...")
                if not controlnet_preprocessor.initialize():
                    return {"error": "Failed to initialize ControlNet preprocessors. Please check your installation."}
            
            # Read uploaded image
            image_data = await image.read()
            input_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess image
            processed_image = controlnet_preprocessor.preprocess(input_image, control_type)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return StreamingResponse(io.BytesIO(img_buffer.getvalue()), media_type="image/png")
            
        except Exception as e:
            # Return error as JSON
            return {"error": f"Image preprocessing failed: {str(e)}"}
    
    @app.post("/load_model")
    async def load_model_ui(request: Request, model_name: str = Form(...)):
        """Load model (Web UI) - runs loading in thread pool"""
        success = False
        error_message = None

        try:
            if await asyncio.to_thread(model_manager.load_model, model_name):
                success = True
            else:
                error_message = f"Failed to load model {model_name}"
        except Exception as e:
            error_message = f"Error loading model: {str(e)}"

        context = get_template_context(request)
        context.update({
            "success_message": f"Model {model_name} loaded successfully!" if success else None,
            "error_message": error_message,
        })

        return templates.TemplateResponse("index.html", context)
    
    @app.post("/unload_model")
    async def unload_model_ui(request: Request):
        """Unload current model (Web UI)"""
        try:
            current_model = model_manager.get_current_model()
            model_manager.unload_model()
            success_message = f"Model {current_model} unloaded successfully!" if current_model else "Model unloaded!"
            error_message = None
        except Exception as e:
            success_message = None
            error_message = f"Error unloading model: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": success_message,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/load_lora")
    async def load_lora_ui(request: Request, lora_name: str = Form(...), scale: float = Form(1.0)):
        """Load LoRA (Web UI)"""
        success = False
        error_message = None
        
        try:
            if lora_manager.load_lora(lora_name, scale=scale):
                success = True
            else:
                error_message = f"Failed to load LoRA {lora_name}"
        except Exception as e:
            error_message = f"Error loading LoRA: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": f"LoRA {lora_name} loaded successfully with scale {scale}!" if success else None,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/unload_lora")
    async def unload_lora_ui(request: Request):
        """Unload current LoRA (Web UI)"""
        try:
            current_lora_name = lora_manager.get_current_lora()
            lora_manager.unload_lora()
            success_message = f"LoRA {current_lora_name} unloaded successfully!" if current_lora_name else "LoRA unloaded!"
            error_message = None
        except Exception as e:
            success_message = None
            error_message = f"Error unloading LoRA: {str(e)}"
        
        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": success_message,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)
    
    @app.post("/pull_lora")
    async def pull_lora_ui(request: Request, repo_id: str = Form(...), weight_name: str = Form(""), alias: str = Form("")):
        """Pull LoRA from Hugging Face Hub (Web UI)"""
        success = False
        error_message = None
        final_name = repo_id

        try:
            # Use alias if provided, otherwise use repo_id
            lora_alias = alias if alias.strip() else None
            weight_file = weight_name if weight_name.strip() else None

            if lora_manager.pull_lora(repo_id, weight_name=weight_file, alias=lora_alias):
                success = True
                final_name = lora_alias if lora_alias else repo_id.replace('/', '_')
            else:
                error_message = f"Failed to download LoRA {repo_id}"
        except Exception as e:
            error_message = f"Error downloading LoRA: {str(e)}"

        # Return result page
        context = get_template_context(request)
        context.update({
            "success_message": f"LoRA {final_name} downloaded successfully!" if success else None,
            "error_message": error_message
        })
        
        return templates.TemplateResponse("index.html", context)

    @app.post("/api/controlnet/initialize")
    async def initialize_controlnet_api():
        """Initialize ControlNet preprocessors (API endpoint)"""
        try:
            success = controlnet_preprocessor.initialize()
            return {
                "success": success,
                "initialized": controlnet_preprocessor.is_initialized(),
                "message": "ControlNet preprocessors initialized successfully!" if success else "Failed to initialize ControlNet preprocessors"
            }
        except Exception as e:
            logger.error(f"Error initializing ControlNet: {e}")
            return {
                "success": False,
                "initialized": False,
                "message": f"Error initializing ControlNet: {str(e)}"
            }

    return app 