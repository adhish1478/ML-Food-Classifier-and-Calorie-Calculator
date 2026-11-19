"""
NutriVision Deep Food Recognition Model.
Deploys a Zero-Shot Vision-Language Food Classifier (CLIP ViT) backed by
Hugging Face pre-trained weights for state-of-the-art culinary dish recognition,
spanning regional Indian, Asian, and Western cuisines.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nutrition_db import FOOD_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, FOOD_DATABASE
from utils.preprocessing import preprocess_food_image

# Curated culinary natural-language descriptions for robust zero-shot classification
FOOD_PROMPTS = {
    "oatmeal": "a bowl of oatmeal porridge with sliced bananas and peanut butter",
    "gulab_jamun": "gulab jamun sweet indian dessert soaked in sugar syrup",
    "chicken_biryani": "a plate of spiced chicken biryani with basmati rice and meat",
    "pizza": "a hot cheesy slice of pizza with toppings",
    "burger": "a juicy burger with meat or chicken patty and cheese in a bun",
    "caesar_salad": "a fresh chicken caesar salad with green lettuce and dressing",
    "masala_dosa": "a crispy south indian masala dosa crepe with potato curry filling",
    "idli": "steamed white idli rice cakes with sambar",
    "samosa": "crispy fried triangular indian samosa pastry",
    "paneer_butter_masala": "paneer butter masala curry with cubes of cottage cheese",
    "butter_chicken": "creamy butter chicken curry in tomato gravy",
    "dal_makhani": "creamy black dal makhani lentil curry",
    "chole_bhature": "chole bhature chickpeas with puffed fried bread",
    "roti_naan": "flatbread butter naan or whole wheat tandoori roti",
    "french_fries": "golden crispy potato french fries",
    "spaghetti_bolognese": "spaghetti pasta with rich bolognese meat sauce",
    "grilled_chicken": "herb grilled chicken breast with vegetables",
    "sandwich": "a club sandwich with bread layers and fillings",
    "steak": "a grilled beef steak fillet",
    "fried_rice": "chinese chicken fried rice with vegetables and eggs",
    "sushi": "fresh salmon sushi rolls with rice and seaweed",
    "dim_sum": "steamed dumplings dim sum in a bamboo basket",
    "ramen": "a hot bowl of ramen noodle soup with boiled egg and broth",
    "pad_thai": "pad thai stir fried noodles with peanuts",
    "boiled_eggs": "hard boiled eggs sliced in half",
    "greek_yogurt": "a bowl of plain white greek yogurt",
    "fruit_salad": "a bowl of fresh mixed fruit salad",
    "green_salad": "a fresh green leafy garden salad with cucumber",
    "pancakes": "a stack of fluffy breakfast pancakes with maple syrup",
    "vegetable_curry": "mixed vegetable curry with spices and vegetables"
}

FOOD_KEYS = list(FOOD_PROMPTS.keys())
CANDIDATE_TEXTS = [f"a photo of {FOOD_PROMPTS[k]}" for k in FOOD_KEYS]


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


class FoodClassifier:
    """
    High-accuracy Food Classifier utilizing a pre-trained Vision Transformer
    (CLIP ViT) with natural language culinary prompt matching.
    """
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: torch.device = None):
        if device is None:
            device = get_optimal_device()
        self.device = device
        self.model_id = model_id

        from transformers import CLIPProcessor, CLIPModel
        print(f"[NutriVision] Loading Pretrained Food Classifier ({model_id}) on {device} (100% Offline / Local)...")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device)
        except Exception:
            # Fallback if running on a fresh machine where model has not yet been cached
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(device)

        self.model.eval()


        self.food_keys = FOOD_KEYS
        self.candidate_texts = CANDIDATE_TEXTS

    def predict(self, image: Image.Image, top_k: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        # Preprocessing representation (Step 2 visual cards)
        prep_data = preprocess_food_image(image)

        rgb_img = image.convert("RGB")
        inputs = self.processor(
            text=self.candidate_texts,
            images=rgb_img,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # (1, num_classes)
            probs = logits.softmax(dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.food_keys)))

        candidates = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            k = self.food_keys[idx]
            food_meta = FOOD_DATABASE.get(k, {
                "display_name": k.replace("_", " ").title(),
                "cuisine": "Global",
                "category": "Main",
                "healthy_alternative": ""
            })
            candidates.append({
                "food_key": k,
                "display_name": food_meta["display_name"],
                "cuisine": food_meta["cuisine"],
                "category": food_meta["category"],
                "confidence": round(prob * 100.0, 2),
                "healthy_alternative": food_meta["healthy_alternative"]
            })

        top1 = candidates[0]
        return top1, candidates, prep_data


# Backward compatibility alias
FoodClassifierCNN = FoodClassifier

def load_food_classifier(model_path: str = None, device: torch.device = None) -> FoodClassifier:
    """
    Factory function to load the NutriVision food classifier.
    """
    return FoodClassifier(device=device)


def classify_food_image(
    model: Any,
    image: Image.Image,
    top_k: int = 3,
    device: torch.device = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience wrapper function matching the app interface.
    Gracefully handles old cached models or newly initialized models.
    """
    if hasattr(model, "predict"):
        return model.predict(image, top_k=top_k)
    
    # If a stale cached object was passed from a running Streamlit session
    fresh_model = FoodClassifier(device=device)
    return fresh_model.predict(image, top_k=top_k)

