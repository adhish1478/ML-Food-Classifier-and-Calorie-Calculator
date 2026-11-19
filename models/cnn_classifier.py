"""
NutriVision CNN-Based Food Recognition Model.
Matches Step 3 and Section 1 ('CNN for Food Recognition') of the infographic:
Input Image -> Conv + ReLU -> Pooling -> Flatten -> FC Layer (Output).
Utilizes MobileNetV2 / ResNet Transfer Learning for high accuracy spatial
feature extraction and lightweight cross-platform inference (M1 Mac & RTX 3050).
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nutrition_db import FOOD_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, FOOD_DATABASE
from utils.preprocessing import preprocess_food_image


def get_optimal_device() -> torch.device:
    """
    Returns optimal accelerator: CUDA on Windows RTX 3050, MPS on M1 Mac, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class FoodClassifierCNN(nn.Module):
    """
    Deep Convolutional Neural Network for Food Category Classification.
    Built upon MobileNetV2 with inverted residual blocks for real-time spatial
    feature learning (edges, textures, colors, culinary geometry).
    """
    def __init__(self, num_classes: int = len(FOOD_CLASSES), pretrained: bool = True):
        super(FoodClassifierCNN, self).__init__()
        self.num_classes = num_classes

        # Initialize base MobileNetV2 architecture
        base_model = models.mobilenet_v2(weights=None)

        if pretrained:
            local_weights = os.path.join(os.path.dirname(__file__), "mobilenet_v2.pth")
            hub_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth")

            if os.path.exists(local_weights):
                state_dict = torch.load(local_weights, map_location="cpu")
                base_model.load_state_dict(state_dict)
            elif os.path.exists(hub_cache):
                state_dict = torch.load(hub_cache, map_location="cpu")
                base_model.load_state_dict(state_dict)
            else:
                try:
                    weights = models.MobileNet_V2_Weights.DEFAULT
                    base_model = models.mobilenet_v2(weights=weights)
                except Exception:
                    pass

        # Retain feature extractor (Conv + Inverted Residual blocks + Pooling)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # Custom high-capacity classification head
        in_features = base_model.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: (batch_size, 3, 224, 224)
        output: (batch_size, num_classes) logits
        """
        feats = self.features(x)
        pooled = self.pool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits


def load_food_classifier(model_path: str = None, device: torch.device = None) -> FoodClassifierCNN:
    """
    Initializes and loads the FoodClassifierCNN.
    If saved weights exist at model_path, loads them; otherwise initializes
    with pretrained MobileNetV2 backbone weights.
    """
    if device is None:
        device = get_optimal_device()

    model = FoodClassifierCNN(num_classes=len(FOOD_CLASSES), pretrained=True)

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[NutriVision CNN] Loaded trained weights from {model_path}")
    else:
        print("[NutriVision CNN] Initialized with pretrained backbone feature extractor.")

    model.to(device)
    model.eval()
    return model


def classify_food_image(
    model: FoodClassifierCNN,
    image: Image.Image,
    top_k: int = 3,
    device: torch.device = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs the complete Step 2 & 3 pipeline on an input image:
    1. Preprocesses image (Resize, Grayscale, Normalize)
    2. Runs CNN forward pass
    3. Computes Softmax probability distribution
    4. Returns Top-1 prediction, Top-k candidates, and preprocessing visualization dict.
    """
    if device is None:
        device = get_optimal_device()

    # Step 2: Preprocessing
    prep_data = preprocess_food_image(image)
    tensor = torch.tensor(prep_data["tensor_array"], dtype=torch.float32, device=device)

    # Step 3: Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    top_probs, top_indices = torch.topk(probs, min(top_k, len(FOOD_CLASSES)))

    top_candidates = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
        cls_key = IDX_TO_CLASS[idx]
        food_meta = FOOD_DATABASE[cls_key]
        top_candidates.append({
            "food_key": cls_key,
            "display_name": food_meta["display_name"],
            "cuisine": food_meta["cuisine"],
            "category": food_meta["category"],
            "confidence": round(prob * 100.0, 2),
            "healthy_alternative": food_meta["healthy_alternative"]
        })

    top1 = top_candidates[0]

    return top1, top_candidates, prep_data
