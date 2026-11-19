"""
NutriVision Preprocessing Pipeline and Visualization Utilities.
Matches Step 2 of the project architecture:
Resize & Crop -> Grayscale -> Normalize.
"""

from typing import Tuple, Dict, Any
from PIL import Image, ImageOps
import numpy as np


def preprocess_food_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    """
    Applies the full preprocessing pipeline on an input food image and produces
    both visual stages (for the UI) and normalized tensor data for the CNN.
    
    Stages:
    1. Original RGB image
    2. Resized & Center Cropped image (target_size)
    3. Grayscale representation (as shown in Step 2 of the infographic)
    4. Normalized feature map visualization
    5. Final normalized array for PyTorch tensor conversion
    """
    # 1. Ensure RGB
    if image.mode != "RGB":
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image.copy()

    # 2. Resize and Center Crop maintaining aspect ratio
    # High-quality Lanczos resampling
    cropped_image = ImageOps.fit(rgb_image, target_size, method=Image.Resampling.LANCZOS)

    # 3. Grayscale representation (matching infographic stage 2)
    gray_image = cropped_image.convert("L")

    # 4. Normalization using standard ImageNet parameters
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img_array = np.array(cropped_image, dtype=np.float32) / 255.0  # (224, 224, 3)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized_array = (img_array - mean) / std

    # 5. Create a visual representation of normalized data for dashboard display
    # Map back to [0, 255] for visual rendering
    norm_vis = ((normalized_array - normalized_array.min()) / (normalized_array.max() - normalized_array.min()) * 255.0).astype(np.uint8)
    norm_image = Image.fromarray(norm_vis)

    # 6. Prepare PyTorch-ready tensor format (Channels First: 1, 3, 224, 224)
    tensor_array = np.transpose(normalized_array, (2, 0, 1))  # (3, 224, 224)
    tensor_array = np.expand_dims(tensor_array, axis=0)      # (1, 3, 224, 224)

    return {
        "original_image": rgb_image,
        "cropped_image": cropped_image,
        "gray_image": gray_image,
        "normalized_image": norm_image,
        "tensor_array": tensor_array,
        "width": cropped_image.width,
        "height": cropped_image.height
    }
