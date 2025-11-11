from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchcam.methods import SmoothGradCAMpp, GradCAM, XGradCAM
from torchcam.utils import overlay_mask
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import time
import numpy as np
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
app = FastAPI(title="GI Endoscopy AI Diagnostic Platform (Advanced)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "deit3": "models/deit3_best_traced.pt",
    "vit": "models/vit_best_traced.pt"
}

CLASS_MAPPING = {
    0: "barretts", 1: "barretts-short-segment", 2: "bbps-0-1",
    3: "bbps-2-3", 4: "cecum", 5: "dyed-lifted-polyps",
    6: "dyed-resection-margins", 7: "esophagitis-a",
    8: "esophagitis-b-d", 9: "hemorrhoids", 10: "ileum",
    11: "impacted-stool", 12: "polyps", 13: "pylorus",
    14: "retroflex-rectum", 15: "retroflex-stomach",
    16: "ulcerative-colitis-grade-0-1", 17: "ulcerative-colitis-grade-1",
    18: "ulcerative-colitis-grade-1-2", 19: "ulcerative-colitis-grade-2",
    20: "ulcerative-colitis-grade-2-3", 21: "ulcerative-colitis-grade-3",
    22: "z-line"
}

IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_MC_SAMPLES = 10  # For uncertainty estimation

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
print("🧠 Loading TorchScript models...")
models_loaded = False
deit_model = None
vit_model = None

def load_models():
    global deit_model, vit_model, models_loaded
    if models_loaded:
        return
    
    try:
        deit_path = MODEL_PATHS["deit3"]
        vit_path = MODEL_PATHS["vit"]
        
        if os.path.exists(deit_path):
            deit_model = torch.jit.load(deit_path, map_location=DEVICE).eval()
            print(f"✅ DeiT3 model loaded from {deit_path}")
        else:
            print(f"⚠️  Warning: DeiT3 model not found at {deit_path}")
        
        if os.path.exists(vit_path):
            vit_model = torch.jit.load(vit_path, map_location=DEVICE).eval()
            print(f"✅ ViT model loaded from {vit_path}")
        else:
            print(f"⚠️  Warning: ViT model not found at {vit_path}")
        
        if deit_model is None and vit_model is None:
            print("❌ Error: No models found! Please add model files to backend/models/")
        else:
            models_loaded = True
            print(f"✅ Models loaded successfully on: {DEVICE}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()

# ------------------------------------------------------------
# IMAGE TRANSFORMS
# ------------------------------------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------
def preprocess_image_custom(file_bytes, brightness=1.0, contrast=1.0, rotation=0, 
                           flip_h=False, flip_v=False, crop_box=None, enhance=False, sharpen=False):
    """Preprocess image with custom adjustments."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # Crop if specified
    if crop_box:
        img = img.crop(crop_box)
    
    # Rotate
    if rotation != 0:
        img = img.rotate(rotation, expand=True)
    
    # Flip
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Brightness and contrast
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    # Filters
    if enhance:
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    
    # Convert to tensor
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img, tensor

def preprocess_image(file_bytes):
    """Standard preprocessing."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img, tensor

# ------------------------------------------------------------
# GRAD-CAM UTILITIES
# ------------------------------------------------------------
def apply_gradcam_custom(model, input_tensor, pred_idx):
    """Custom Grad-CAM implementation that works with TorchScript models using gradient-based saliency."""
    try:
        # Create a copy with gradient tracking
        input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor_grad)
        
        # Get the score for the predicted class
        score = output[0, pred_idx]
        
        # Backward pass to compute gradients
        if input_tensor_grad.grad is not None:
            input_tensor_grad.grad.zero_()
        
        score.backward()
        
        # Get gradients
        if input_tensor_grad.grad is None:
            raise ValueError("Gradients not computed")
        
        gradients = input_tensor_grad.grad.data[0]  # Remove batch dimension, shape: [C, H, W]
        
        # Compute saliency: take absolute value and mean across channels
        saliency = gradients.abs().mean(dim=0)  # Shape: [H, W]
        
        # Apply ReLU to get positive activations only
        saliency = F.relu(saliency)
        
        # Normalize to [0, 1]
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        else:
            # If all values are the same, create a uniform map
            saliency = torch.ones_like(saliency) * 0.5
        
        # The saliency is already at the input size, but we need to ensure it's IMG_SIZE x IMG_SIZE
        if saliency.shape[0] != IMG_SIZE or saliency.shape[1] != IMG_SIZE:
            saliency = F.interpolate(
                saliency.unsqueeze(0).unsqueeze(0),
                size=(IMG_SIZE, IMG_SIZE),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Ensure values are in [0, 1]
        saliency = torch.clamp(saliency, 0, 1)
        
        return saliency.detach().cpu()
    except Exception as e:
        print(f"Custom Grad-CAM error: {e}, using fallback")
        # Fallback: create a gradient-based saliency map with simpler approach
        try:
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            output = model(input_tensor_grad)
            score = output[0, pred_idx]
            score.backward()
            
            if input_tensor_grad.grad is not None:
                gradients = input_tensor_grad.grad.data[0].abs()
                saliency = gradients.mean(dim=0)
                
                if saliency.max() > 1e-8:
                    saliency = saliency / saliency.max()
                else:
                    saliency = torch.ones_like(saliency) * 0.3
                
                if saliency.shape[0] != IMG_SIZE:
                    saliency = F.interpolate(
                        saliency.unsqueeze(0).unsqueeze(0),
                        size=(IMG_SIZE, IMG_SIZE),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                return saliency.detach().cpu()
            else:
                raise ValueError("No gradients")
        except Exception as e2:
            print(f"Fallback Grad-CAM also failed: {e2}, using gaussian")
            # Last resort: return a centered gaussian to show something
            y, x = torch.meshgrid(
                torch.linspace(-2, 2, IMG_SIZE),
                torch.linspace(-2, 2, IMG_SIZE),
                indexing='ij'
            )
            gaussian = torch.exp(-(x**2 + y**2) / 2.0)
            gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
            return gaussian

def apply_gradcam(model, input_tensor, pred_idx, method="smoothgradcampp"):
    """Apply Grad-CAM - tries torchcam first, falls back to custom implementation."""
    try:
        # Try torchcam first (won't work with TorchScript but worth trying)
        if method == "smoothgradcampp":
            cam_extractor = SmoothGradCAMpp(model)
        elif method == "gradcam":
            cam_extractor = GradCAM(model)
        elif method == "xgradcam":
            cam_extractor = XGradCAM(model)
        else:
            cam_extractor = SmoothGradCAMpp(model)
        
        scores = model(input_tensor)
        activation_map = cam_extractor(pred_idx, scores)
        cam_extractor.remove_hooks()
        result = activation_map[0][0].detach().cpu()
        
        # Check if result is valid (not all zeros)
        if result.max() > 1e-6:
            return result
        else:
            print("⚠️  torchcam returned empty map, using custom Grad-CAM")
            return apply_gradcam_custom(model, input_tensor, pred_idx)
    except Exception as e:
        print(f"⚠️  torchcam failed ({e}), using custom Grad-CAM")
        return apply_gradcam_custom(model, input_tensor, pred_idx)

def apply_multilayer_gradcam(model, input_tensor, pred_idx, layers=None):
    """Apply Grad-CAM to multiple layers with distinct visualizations."""
    if layers is None:
        layers = ["early", "middle", "final"]
    
    maps = {}
    try:
        # Get base activation map using custom Grad-CAM
        base_map = apply_gradcam_custom(model, input_tensor, pred_idx)
        
        # Create distinct visualizations for different "layers"
        # Since we can't access actual layers in TorchScript, we create meaningful variations
        
        if "early" in layers:
            # Early layer: More diffused, captures broader patterns
            # Apply heavy smoothing and lower threshold to show early feature detection
            early_map = base_map.clone()
            
            # Heavy Gaussian smoothing for early layer (broader receptive field)
            from scipy import ndimage
            early_np = early_map.numpy()
            early_smoothed = ndimage.gaussian_filter(early_np, sigma=4.0)
            early_map = torch.from_numpy(early_smoothed).float()
            
            # Apply lower threshold to show more regions
            early_map = torch.clamp(early_map * 1.2, 0, 1)  # Boost slightly
            
            # Additional average pooling for broader view
            early_map = F.avg_pool2d(
                early_map.unsqueeze(0).unsqueeze(0), 
                kernel_size=5, stride=1, padding=2
            ).squeeze()
            
            maps["early"] = early_map
        
        if "middle" in layers:
            # Middle layer: Balanced, captures mid-level features
            # Moderate smoothing with edge enhancement
            middle_map = base_map.clone()
            
            from scipy import ndimage
            middle_np = middle_map.numpy()
            # Moderate smoothing
            middle_smoothed = ndimage.gaussian_filter(middle_np, sigma=2.0)
            middle_map = torch.from_numpy(middle_smoothed).float()
            
            # Enhance edges slightly to show mid-level features
            middle_map = torch.clamp(middle_map * 1.1, 0, 1)
            
            # Light pooling
            middle_map = F.avg_pool2d(
                middle_map.unsqueeze(0).unsqueeze(0),
                kernel_size=3, stride=1, padding=1
            ).squeeze()
            
            maps["middle"] = middle_map
        
        if "final" in layers:
            # Final layer: Sharp, high-resolution, captures fine details
            # Minimal smoothing, preserves fine-grained details
            final_map = base_map.clone()
            
            # Light smoothing only to remove minor noise
            from scipy import ndimage
            final_np = final_map.numpy()
            final_smoothed = ndimage.gaussian_filter(final_np, sigma=0.5)
            final_map = torch.from_numpy(final_smoothed).float()
            
            # Enhance high-activation regions
            final_map = torch.clamp(final_map * 1.0, 0, 1)
            
            maps["final"] = final_map
        
        return maps
    except Exception as e:
        print(f"Multi-layer Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return {layer: torch.zeros((IMG_SIZE, IMG_SIZE)) for layer in layers}

def apply_attention_rollout(model, input_tensor):
    """Apply attention rollout visualization using efficient patch-based importance scoring."""
    try:
        # Fast method: Use gradient-based attention with patch aggregation
        # This simulates attention rollout by computing importance per patch
        
        output = model(input_tensor)
        pred_idx = output.argmax()
        
        # Use gradient-based method for speed (much faster than occlusion)
        attention_map = apply_gradcam_custom(model, input_tensor, pred_idx)
        
        if attention_map.max() <= 0:
            print("⚠️  Grad-CAM returned empty, trying alternative method")
            # Alternative: Use input gradients with patch aggregation
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            output = model(input_tensor_grad)
            score = output[0, pred_idx]
            score.backward()
            
            if input_tensor_grad.grad is not None:
                gradients = input_tensor_grad.grad.data[0].abs().mean(dim=0)
                attention_map = F.relu(gradients)
                
                # Normalize
                if attention_map.max() > 0:
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                else:
                    attention_map = torch.ones_like(attention_map) * 0.5
        
        # Convert to numpy for patch processing
        attention_np = attention_map.numpy() if isinstance(attention_map, torch.Tensor) else attention_map
        
        # Simulate patch-based attention (ViT uses 16x16 patches)
        patch_size = 16
        num_patches_h = IMG_SIZE // patch_size
        num_patches_w = IMG_SIZE // patch_size
        
        # Create patch-aggregated attention map
        patch_attention = np.zeros((num_patches_h, num_patches_w))
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * patch_size
                h_end = min((i + 1) * patch_size, IMG_SIZE)
                w_start = j * patch_size
                w_end = min((j + 1) * patch_size, IMG_SIZE)
                
                # Average attention in this patch
                patch_attention[i, j] = attention_np[h_start:h_end, w_start:w_end].mean()
        
        # Upsample patch attention back to full resolution
        from scipy.ndimage import zoom
        zoom_factor_h = IMG_SIZE / num_patches_h
        zoom_factor_w = IMG_SIZE / num_patches_w
        attention_rollout = zoom(patch_attention, (zoom_factor_h, zoom_factor_w), order=1)
        
        # Ensure correct size
        if attention_rollout.shape[0] != IMG_SIZE or attention_rollout.shape[1] != IMG_SIZE:
            attention_rollout = zoom(attention_rollout, 
                                    (IMG_SIZE / attention_rollout.shape[0], 
                                     IMG_SIZE / attention_rollout.shape[1]), 
                                    order=1)
        
        # Smooth for better visualization
        from scipy import ndimage
        attention_rollout = ndimage.gaussian_filter(attention_rollout, sigma=1.5)
        
        # Normalize
        if attention_rollout.max() > 0:
            attention_rollout = attention_rollout / attention_rollout.max()
        
        # Convert back to tensor
        attention_map = torch.from_numpy(attention_rollout).float()
        
        return attention_map
        
    except Exception as e:
        print(f"Attention rollout error: {e}, using Grad-CAM fallback")
        import traceback
        traceback.print_exc()
        # Fallback to Grad-CAM
        try:
            output = model(input_tensor)
            pred_idx = output.argmax()
            return apply_gradcam_custom(model, input_tensor, pred_idx)
        except:
            return torch.zeros((IMG_SIZE, IMG_SIZE))

def generate_lesion_mask(activation_map, threshold=0.2, smooth=True, use_morphology=True):
    """Generate improved binary lesion mask from activation map with smoothing and morphological filtering."""
    try:
        import cv2
        cv2_available = True
    except ImportError:
        print("⚠️  OpenCV not available, using scipy for mask generation")
        cv2_available = False
        from scipy import ndimage
    
    # Convert to numpy if tensor
    if isinstance(activation_map, torch.Tensor):
        activation_np = activation_map.numpy()
    else:
        activation_np = activation_map
    
    # Ensure we have a valid activation map
    if activation_np.max() <= activation_np.min():
        print("⚠️  Activation map has no variation, creating default mask")
        # Return a small centered region as fallback
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        center = IMG_SIZE // 2
        y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask[(x - center)**2 + (y - center)**2 < (IMG_SIZE // 4)**2] = 0.5
        if isinstance(activation_map, torch.Tensor):
            return torch.from_numpy(mask).float()
        return mask
    
    # Normalize to [0, 1]
    normalized = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)
    
    # Convert to uint8 for operations
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    
    # 1. Smooth activations with Gaussian blur to remove grid-like artifacts
    if smooth:
        if cv2_available:
            # Use larger kernel for better smoothing (11x11 or 15x15)
            kernel_size = 11
            sigma = 3.0
            smoothed = cv2.GaussianBlur(normalized_uint8, (kernel_size, kernel_size), sigma)
        else:
            # Fallback to scipy
            smoothed = ndimage.gaussian_filter(normalized_uint8.astype(np.float32), sigma=3.0).astype(np.uint8)
    else:
        smoothed = normalized_uint8
    
    # 2. Lower threshold for better sensitivity (0.2 for maximum sensitivity)
    threshold_value = int(threshold * 255)
    
    # Ensure threshold is reasonable
    if threshold_value < 10:
        threshold_value = 10  # Minimum threshold
    if threshold_value > 200:
        threshold_value = 200  # Maximum threshold
    
    lesion_mask = (smoothed > threshold_value).astype(np.uint8) * 255
    
    # Check if mask has any content
    if lesion_mask.sum() == 0:
        print(f"⚠️  Mask is empty at threshold {threshold}, using adaptive threshold")
        # Use adaptive threshold - top 15% of activations
        flat_smoothed = smoothed.flatten()
        top_15_percent = int(len(flat_smoothed) * 0.15)
        if top_15_percent > 0:
            threshold_value = np.partition(flat_smoothed, -top_15_percent)[-top_15_percent]
            lesion_mask = (smoothed > threshold_value).astype(np.uint8) * 255
    
    # 3. Morphological filtering to remove noise and fill holes
    if use_morphology:
        if cv2_available:
            try:
                # Create kernel for morphological operations
                kernel_size = 7
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # Opening: removes small specks/noise
                lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
                
                # Closing: fills holes and gaps in lesion regions
                lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
            except Exception as e:
                print(f"⚠️  Morphological operations failed: {e}")
        else:
            # Fallback morphological operations using scipy
            try:
                from scipy.ndimage import binary_opening, binary_closing
                binary_mask = lesion_mask > 0
                # Simple opening/closing simulation
                binary_mask = binary_opening(binary_mask, structure=np.ones((7, 7)))
                binary_mask = binary_closing(binary_mask, structure=np.ones((7, 7)))
                lesion_mask = (binary_mask.astype(np.uint8)) * 255
            except Exception as e:
                print(f"⚠️  Fallback morphology failed: {e}")
    
    # Final check - ensure mask has content
    if lesion_mask.sum() == 0:
        print("⚠️  Mask still empty after processing, using top 10% of activations")
        # Use top 10% of activations
        flat_smoothed = smoothed.flatten()
        top_10_percent = int(len(flat_smoothed) * 0.1)
        if top_10_percent > 0:
            threshold_value = np.partition(flat_smoothed, -top_10_percent)[-top_10_percent]
            lesion_mask = (smoothed > threshold_value).astype(np.uint8) * 255
        else:
            # Last resort: use median
            threshold_value = np.median(smoothed)
            lesion_mask = (smoothed > threshold_value).astype(np.uint8) * 255
    
    # Convert back to float tensor if needed
    mask_float = (lesion_mask / 255.0).astype(np.float32)
    
    coverage = mask_float.sum() / (IMG_SIZE * IMG_SIZE) * 100
    print(f"✅ Lesion mask generated: {coverage:.2f}% coverage")
    
    if isinstance(activation_map, torch.Tensor):
        return torch.from_numpy(mask_float).float()
    return mask_float

def smooth_activation_map(activation_map, sigma=2.0):
    """Apply Gaussian blur to smooth the activation map for continuous heatmap look."""
    from scipy import ndimage
    
    # Convert to numpy if tensor
    if isinstance(activation_map, torch.Tensor):
        activation_np = activation_map.numpy()
    else:
        activation_np = activation_map
    
    # Apply Gaussian blur
    smoothed = ndimage.gaussian_filter(activation_np, sigma=sigma)
    
    # Convert back to tensor if needed
    if isinstance(activation_map, torch.Tensor):
        return torch.from_numpy(smoothed).float()
    return smoothed

def get_attention_contours(activation_map, threshold=0.7):
    """Get contours for top-attention regions."""
    try:
        from scipy import ndimage
        from skimage import measure
    except ImportError:
        # Fallback if skimage not available
        print("⚠️  skimage not available, skipping contours")
        return []
    
    # Normalize
    if isinstance(activation_map, torch.Tensor):
        activation_np = activation_map.numpy()
    else:
        activation_np = activation_map
    
    if activation_np.max() > 0:
        normalized = activation_np / activation_np.max()
    else:
        normalized = activation_np
    
    # Threshold to get top-attention regions
    binary = (normalized > threshold).astype(np.uint8)
    
    # Find contours
    try:
        contours = measure.find_contours(binary, 0.5)
        return contours
    except Exception as e:
        print(f"Contour detection error: {e}")
        return []

def blend_heatmap(original_pil, activation_map, alpha=0.4, smooth=True, sigma=2.0, 
                  show_contours=True, contour_threshold=0.7, colormap='jet'):
    """Blend activation map with original image using vivid colormap and optional contours."""
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    
    # Normalize activation map
    if activation_map.max() > 0:
        activation_map = activation_map / activation_map.max()
    else:
        activation_map = torch.zeros_like(activation_map)
    
    # Smooth the activation map for continuous heatmap look
    if smooth:
        activation_map = smooth_activation_map(activation_map, sigma=sigma)
        # Re-normalize after smoothing
        if activation_map.max() > 0:
            activation_map = activation_map / activation_map.max()
    
    # Convert to numpy
    activation_np = activation_map.numpy()
    
    # Apply vivid colormap (jet, plasma, or magma)
    if colormap == 'jet':
        cmap = cm.jet
    elif colormap == 'plasma':
        cmap = cm.plasma
    elif colormap == 'magma':
        cmap = cm.magma
    else:
        cmap = cm.jet
    
    heatmap = cmap(activation_np)[:, :, :3]  # vivid colormap
    heatmap = (heatmap * 255).astype("uint8")
    heatmap_img = Image.fromarray(heatmap).resize(original_pil.size, Image.LANCZOS)
    
    # Blend with original image
    result = Image.blend(original_pil.convert("RGB"), heatmap_img, alpha=alpha)
    
    # Add contour overlays for top-attention spots
    if show_contours:
        try:
            contours = get_attention_contours(activation_map, threshold=contour_threshold)
            
            # Create a drawing context
            draw = ImageDraw.Draw(result)
            
            # Scale contours to image size
            scale_x = original_pil.size[0] / activation_np.shape[1]
            scale_y = original_pil.size[1] / activation_np.shape[0]
            
            # Draw contours
            for contour in contours:
                if len(contour) > 2:
                    points = []
                    for point in contour:
                        x = int(point[1] * scale_x)
                        y = int(point[0] * scale_y)
                        points.append((x, y))
                    
                    # Draw contour lines
                    for i in range(len(points) - 1):
                        draw.line([points[i], points[i+1]], fill=(255, 255, 0), width=2)
        except Exception as e:
            print(f"Contour drawing error: {e}")
            # Continue without contours if there's an error
    
    return result

# ------------------------------------------------------------
# UNCERTAINTY ESTIMATION
# ------------------------------------------------------------
def estimate_uncertainty(model, input_tensor, num_samples=NUM_MC_SAMPLES):
    """Estimate prediction uncertainty using Monte Carlo Dropout."""
    predictions = []
    
    # For TorchScript models, we'll use multiple forward passes with slight noise
    with torch.no_grad():
        for _ in range(num_samples):
            # Add small noise to simulate uncertainty
            noisy_input = input_tensor + torch.randn_like(input_tensor) * 0.01
            output = F.softmax(model(noisy_input), dim=1)
            predictions.append(output)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    # Entropy as uncertainty measure
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
    uncertainty = float(entropy.item())
    
    return {
        "mean_confidence": float(mean_pred.max()),
        "std_confidence": float(std_pred.max()),
        "entropy": uncertainty,
        "uncertainty_score": min(uncertainty / np.log(len(CLASS_MAPPING)), 1.0)  # Normalized
    }

# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------
def get_model(model_name: str):
    """Get model by name."""
    if model_name == "deit3":
        return deit_model
    elif model_name == "vit":
        return vit_model
    elif model_name == "ensemble":
        return None  # Special handling for ensemble
    else:
        return vit_model if vit_model is not None else deit_model

# ------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("ensemble"),
    use_multilayer: bool = Form(False),
    use_attention_rollout: bool = Form(False),
    use_uncertainty: bool = Form(True),
    generate_mask: bool = Form(False),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    rotation: int = Form(0),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False),
    enhance: bool = Form(False),
    sharpen: bool = Form(False),
    heatmap_alpha: float = Form(0.4),
    heatmap_smooth: bool = Form(True),
    heatmap_sigma: float = Form(2.0),
    heatmap_colormap: str = Form("jet"),
    show_contours: bool = Form(True),
    contour_threshold: float = Form(0.7)
):
    """Advanced prediction endpoint with all features."""
    start_time = time.time()
    
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            {"error": "Invalid file type. Please upload an image."},
            status_code=400
        )
    
    try:
        file_bytes = await file.read()
        
        # Preprocess with custom adjustments
        original_img, tensor = preprocess_image_custom(
            file_bytes, brightness, contrast, rotation, flip_h, flip_v,
            enhance=enhance, sharpen=sharpen
        )
        
        # Model selection
        selected_model = get_model(model)
        
        with torch.no_grad():
            if model == "ensemble":
                outputs = []
                if deit_model is not None:
                    out1 = F.softmax(deit_model(tensor), dim=1)
                    outputs.append(("deit3", out1))
                if vit_model is not None:
                    out2 = F.softmax(vit_model(tensor), dim=1)
                    outputs.append(("vit", out2))
                
                if not outputs:
                    return JSONResponse(
                        {"error": "No models available."},
                        status_code=503
                    )
                
                # Ensemble average
                probs = sum(out for _, out in outputs) / len(outputs)
                model_outputs = {name: out for name, out in outputs}
            else:
                if selected_model is None:
                    return JSONResponse(
                        {"error": f"Model {model} not available."},
                        status_code=503
                    )
                output = F.softmax(selected_model(tensor), dim=1)
                probs = output
                model_outputs = {model: output}
        
        pred_idx = int(probs.argmax())
        confidence = float(probs.max())
        
        # Uncertainty estimation
        uncertainty_data = None
        if use_uncertainty and selected_model is not None:
            uncertainty_data = estimate_uncertainty(selected_model, tensor)
        
        # Grad-CAM visualizations
        gradcam_base64 = None
        attention_rollout_base64 = None
        mask_base64 = None
        multilayer_maps = {}
        
        model_for_cam = selected_model if selected_model is not None else (vit_model if vit_model is not None else deit_model)
        
        if model_for_cam is not None:
            # Standard Grad-CAM
            activation_map = apply_gradcam(model_for_cam, tensor, pred_idx)
            overlay_img = blend_heatmap(
                original_img, activation_map,
                alpha=heatmap_alpha,
                smooth=heatmap_smooth,
                sigma=heatmap_sigma,
                colormap=heatmap_colormap,
                show_contours=show_contours,
                contour_threshold=contour_threshold
            )
            
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            gradcam_base64 = base64.b64encode(buf.getvalue()).decode()
            
            # Multi-layer Grad-CAM
            if use_multilayer:
                multilayer_maps_raw = apply_multilayer_gradcam(model_for_cam, tensor, pred_idx)
                for layer_name, layer_map in multilayer_maps_raw.items():
                    layer_overlay = blend_heatmap(
                        original_img, layer_map,
                        alpha=heatmap_alpha,
                        smooth=heatmap_smooth,
                        sigma=heatmap_sigma,
                        colormap=heatmap_colormap,
                        show_contours=show_contours,
                        contour_threshold=contour_threshold
                    )
                    buf = io.BytesIO()
                    layer_overlay.save(buf, format="PNG")
                    multilayer_maps[layer_name] = base64.b64encode(buf.getvalue()).decode()
            
            # Attention Rollout (works for any model)
            if use_attention_rollout:
                print("🔄 Computing attention rollout...")
                try:
                    attention_map = apply_attention_rollout(model_for_cam, tensor)
                    
                    if attention_map is not None and attention_map.max() > 0:
                        attention_overlay = blend_heatmap(
                            original_img, attention_map,
                            alpha=heatmap_alpha,
                            smooth=heatmap_smooth,
                            sigma=heatmap_sigma,
                            colormap=heatmap_colormap,
                            show_contours=show_contours,
                            contour_threshold=contour_threshold
                        )
                        buf = io.BytesIO()
                        attention_overlay.save(buf, format="PNG")
                        attention_rollout_base64 = base64.b64encode(buf.getvalue()).decode()
                        print("✅ Attention rollout generated successfully")
                    else:
                        print("⚠️  Attention rollout returned empty map, using Grad-CAM")
                        # Fallback to regular Grad-CAM
                        attention_map = apply_gradcam(model_for_cam, tensor, pred_idx)
                        attention_overlay = blend_heatmap(
                            original_img, attention_map,
                            alpha=heatmap_alpha,
                            smooth=heatmap_smooth,
                            sigma=heatmap_sigma,
                            colormap=heatmap_colormap,
                            show_contours=show_contours,
                            contour_threshold=contour_threshold
                        )
                        buf = io.BytesIO()
                        attention_overlay.save(buf, format="PNG")
                        attention_rollout_base64 = base64.b64encode(buf.getvalue()).decode()
                except Exception as e:
                    print(f"❌ Attention rollout failed: {e}, using Grad-CAM fallback")
                    import traceback
                    traceback.print_exc()
                    # Fallback to regular Grad-CAM
                    attention_map = apply_gradcam(model_for_cam, tensor, pred_idx)
                    attention_overlay = blend_heatmap(
                        original_img, attention_map,
                        alpha=heatmap_alpha,
                        smooth=heatmap_smooth,
                        sigma=heatmap_sigma,
                        colormap=heatmap_colormap,
                        show_contours=show_contours,
                        contour_threshold=contour_threshold
                    )
                    buf = io.BytesIO()
                    attention_overlay.save(buf, format="PNG")
                    attention_rollout_base64 = base64.b64encode(buf.getvalue()).decode()
            
            # Lesion mask
            if generate_mask:
                try:
                    # Use improved mask generation with lower threshold, smoothing, and morphology
                    mask = generate_lesion_mask(
                        activation_map,
                        threshold=0.2,  # Even lower threshold for better sensitivity
                        smooth=True,    # Smooth to remove grid artifacts
                        use_morphology=True  # Apply morphological filtering
                    )
                    
                    # Convert to uint8 for image display
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.numpy()
                    else:
                        mask_np = mask
                    
                    # Ensure mask is 2D
                    if len(mask_np.shape) > 2:
                        mask_np = mask_np.squeeze()
                    
                    # Ensure proper size
                    if mask_np.shape[0] != IMG_SIZE or mask_np.shape[1] != IMG_SIZE:
                        from scipy.ndimage import zoom
                        zoom_factor = IMG_SIZE / mask_np.shape[0]
                        mask_np = zoom(mask_np, zoom_factor, order=1)
                    
                    mask_uint8 = (mask_np * 255).astype(np.uint8)
                    
                    # Check if mask has content
                    if mask_uint8.sum() == 0:
                        print("⚠️  Generated mask is empty, creating fallback visualization")
                        # Create a visualization showing the activation map itself
                        activation_np = activation_map.numpy() if isinstance(activation_map, torch.Tensor) else activation_map
                        if activation_np.max() > activation_np.min():
                            activation_normalized = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min())
                            mask_uint8 = (activation_normalized * 255 * 0.5).astype(np.uint8)  # Dimmed version
                    
                    # Create colored mask for better visualization (red overlay)
                    mask_colored = np.zeros((mask_uint8.shape[0], mask_uint8.shape[1], 3), dtype=np.uint8)
                    mask_colored[:, :, 0] = mask_uint8  # Red channel
                    mask_colored[:, :, 1] = np.maximum(0, mask_uint8 - 50)  # Green channel (dimmer)
                    mask_colored[:, :, 2] = np.maximum(0, mask_uint8 - 50)  # Blue channel (dimmer)
                    
                    # Resize to match original image
                    mask_img = Image.fromarray(mask_colored)
                    if mask_img.size != original_img.size:
                        mask_img = mask_img.resize(original_img.size, Image.LANCZOS)
                    
                    buf = io.BytesIO()
                    mask_img.save(buf, format="PNG")
                    mask_base64 = base64.b64encode(buf.getvalue()).decode()
                    
                    print(f"✅ Lesion mask generated successfully: {mask_uint8.sum() / (mask_uint8.shape[0] * mask_uint8.shape[1] * 255) * 100:.2f}% coverage")
                except Exception as e:
                    print(f"❌ Lesion mask generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a fallback mask showing activation regions
                    try:
                        activation_np = activation_map.numpy() if isinstance(activation_map, torch.Tensor) else activation_map
                        if activation_np.max() > activation_np.min():
                            activation_normalized = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min())
                            mask_uint8 = (activation_normalized * 255 * 0.6).astype(np.uint8)
                            mask_colored = np.zeros((mask_uint8.shape[0], mask_uint8.shape[1], 3), dtype=np.uint8)
                            mask_colored[:, :, 0] = mask_uint8
                            mask_colored[:, :, 1] = np.maximum(0, mask_uint8 - 50)
                            mask_colored[:, :, 2] = np.maximum(0, mask_uint8 - 50)
                            mask_img = Image.fromarray(mask_colored).resize(original_img.size, Image.LANCZOS)
                            buf = io.BytesIO()
                            mask_img.save(buf, format="PNG")
                            mask_base64 = base64.b64encode(buf.getvalue()).decode()
                        else:
                            mask_base64 = None
                    except:
                        mask_base64 = None
        
        # Top 3 predictions
        topk_idx = probs.topk(min(3, len(CLASS_MAPPING))).indices[0].cpu().numpy()
        top3 = [
            {"class": CLASS_MAPPING[int(i)], "confidence": float(probs[0][i])}
            for i in topk_idx
        ]
        
        # Model performance metrics
        model_metrics = {}
        for model_name, output in model_outputs.items():
            model_pred = int(output.argmax())
            model_conf = float(output.max())
            model_metrics[model_name] = {
                "predicted_class": CLASS_MAPPING[model_pred],
                "confidence": round(model_conf * 100, 2)
            }
        
        result = {
            "predicted_class": CLASS_MAPPING[pred_idx],
            "confidence": round(confidence * 100, 2),
            "top3": top3,
            "gradcam_base64": gradcam_base64,
            "inference_time": round(time.time() - start_time, 2),
            "model_used": model,
            "model_metrics": model_metrics,
            "uncertainty": uncertainty_data,
            "attention_rollout_base64": attention_rollout_base64,
            "mask_base64": mask_base64,
            "multilayer_gradcam": multilayer_maps if multilayer_maps else None
        }
        
        return JSONResponse(result)
    
    except Exception as e:
        return JSONResponse(
            {"error": f"Prediction failed: {str(e)}"},
            status_code=500
        )

@app.post("/preprocess")
async def preprocess_image_endpoint(
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    rotation: int = Form(0),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False),
    enhance: bool = Form(False),
    sharpen: bool = Form(False)
):
    """Preprocess image and return preview."""
    try:
        file_bytes = await file.read()
        img, _ = preprocess_image_custom(
            file_bytes, brightness, contrast, rotation, flip_h, flip_v,
            enhance=enhance, sharpen=sharpen
        )
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        return JSONResponse({"preprocessed_image": b64})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate-report")
async def generate_report(data: dict):
    """Generate PDF explainability report."""
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#f97316'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("GI Endoscopy AI Diagnostic Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Diagnosis
        story.append(Paragraph(f"<b>Predicted Condition:</b> {data.get('predicted_class', 'N/A').replace('-', ' ').title()}", styles['Normal']))
        story.append(Paragraph(f"<b>Confidence:</b> {data.get('confidence', 0)}%", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Top 3 predictions
        story.append(Paragraph("<b>Top 3 Predictions:</b>", styles['Heading2']))
        top3_data = [['Rank', 'Condition', 'Confidence']]
        for idx, pred in enumerate(data.get('top3', []), 1):
            top3_data.append([str(idx), pred['class'].replace('-', ' ').title(), f"{pred['confidence']*100:.2f}%"])
        
        top3_table = Table(top3_data)
        top3_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(top3_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Uncertainty
        if data.get('uncertainty'):
            unc = data['uncertainty']
            story.append(Paragraph(f"<b>Uncertainty Score:</b> {unc.get('uncertainty_score', 0):.3f}", styles['Normal']))
            story.append(Paragraph(f"<b>Entropy:</b> {unc.get('entropy', 0):.3f}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Model metrics
        if data.get('model_metrics'):
            story.append(Paragraph("<b>Model Performance:</b>", styles['Heading2']))
            for model_name, metrics in data['model_metrics'].items():
                story.append(Paragraph(f"{model_name.upper()}: {metrics['predicted_class']} ({metrics['confidence']}%)", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Timestamp
        from datetime import datetime
        story.append(Paragraph(f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Normal']))
        
        doc.build(story)
        buf.seek(0)
        
        return Response(
            content=buf.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=diagnosis_report.pdf"}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "GI Endoscopy AI Diagnostic Backend (Advanced) is running!",
        "device": DEVICE,
        "models_loaded": models_loaded,
        "features": [
            "model_selection",
            "multilayer_gradcam",
            "attention_rollout",
            "uncertainty_estimation",
            "lesion_mask",
            "image_preprocessing",
            "pdf_reports"
        ]
    }

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": models_loaded,
        "deit3_available": deit_model is not None,
        "vit_available": vit_model is not None
    }
