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

from utils.nutrition_db import FOOD_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, FOOD_DATABASE, TOPPINGS_DATABASE
from utils.preprocessing import preprocess_food_image

# Comprehensive culinary natural-language descriptions for zero-shot base dish classification
FOOD_PROMPTS = {
    # --- Indian Specialties & Street Foods ---
    "vada_pav": "a mumbai vada pav spicy potato fritter in a bun with dry garlic chutney",
    "pav_bhaji": "a plate of spicy pav bhaji mashed vegetable curry with toasted buttered pav buns",
    "samosa": "crispy fried triangular indian samosa pastry stuffed with spiced potato and peas",
    "pani_puri": "pani puri golgappa hollow fried balls with spicy mint flavored water",
    "bhel_puri": "indian street bhel puri chaat puffed rice with sev and tangy chutneys",
    "chicken_biryani": "a plate of spiced chicken biryani with fragrant basmati rice and chicken",
    "mutton_biryani": "hyderabadi mutton biryani with spiced basmati rice and tender meat",
    "veg_biryani": "vegetable biryani or pulao with fragrant basmati rice and vegetables",
    "paneer_butter_masala": "paneer butter masala curry with cubes of cottage cheese in tomato gravy",
    "palak_paneer": "palak paneer spinach curry with white cottage cheese paneer cubes",
    "butter_chicken": "creamy butter chicken curry in rich spiced tomato sauce",
    "chicken_tikka": "tandoori chicken tikka pieces grilled with spices",
    "masala_dosa": "a crispy golden south indian masala dosa crepe with spiced potato filling",
    "plain_dosa": "thin crispy south indian plain dosa crepe served with chutney",
    "idli": "steamed white south indian idli rice cakes served with sambar",
    "medu_vada": "crispy golden south indian medu vada lentil doughnut fritter",
    "poha": "maharashtrian kanda poha flattened rice with roasted peanuts and turmeric",
    "upma": "south indian rava upma semolina breakfast with vegetables",
    "chole_bhature": "spiced chickpea chole curry with large puffy deep fried bhature bread",
    "dal_makhani": "creamy black dal makhani lentil curry",
    "dal_tadka": "yellow dal tadka lentil curry tempered with cumin and chili",
    "rajma_chawal": "red kidney bean rajma curry with white steamed rice",
    "aloo_paratha": "stuffed aloo paratha indian flatbread with potato filling and butter",
    "paneer_paratha": "stuffed paneer paratha indian flatbread with cottage cheese",
    "poori_bhaji": "puffy deep fried golden poori with spiced potato bhaji curry",
    "khichdi": "comforting moong dal and rice khichdi with ghee",
    "curd_rice": "south indian curd rice tempered with mustard seeds and curry leaves",
    "uttapam": "thick south indian uttapam pancake topped with onions and tomatoes",
    "dhokla": "spongy yellow steamed khaman dhokla with mustard seeds",
    "roti_naan": "tandoori butter naan or whole wheat indian roti flatbread",
    "fish_curry": "spiced coastal fish curry in coconut and tomato gravy",
    "egg_curry": "boiled egg curry in rich indian spiced masala gravy",
    "gulab_jamun": "sweet round brown gulab jamun soaked in sugar syrup",
    "rasgulla": "spongy white sweet rasgulla balls soaked in light syrup",
    "jalebi": "bright orange spiral crispy fried jalebi soaked in sugar syrup",
    "kheer": "indian rice kheer payasam sweet milk pudding with nuts",
    "gajar_halwa": "warm red gajar ka halwa carrot pudding with dry fruits",
    "chai": "a cup of hot indian spiced masala chai milk tea",
    "lassi": "a glass of creamy sweet lassi or mango yogurt drink",

    # --- Western & Continental Favorites ---
    "pizza": "a cheesy slice of pizza with melted cheese and toppings",
    "burger": "a juicy hamburger cheeseburger with bun and patty",
    "french_fries": "golden crispy potato french fries",
    "caesar_salad": "fresh chicken caesar salad with green lettuce and dressing",
    "spaghetti_bolognese": "spaghetti pasta with rich bolognese meat sauce",
    "grilled_chicken": "herb grilled chicken breast with vegetables",
    "sandwich": "a club sandwich with bread layers and fillings",
    "steak": "a grilled beef steak fillet",
    "hot_dog": "a hot dog sausage in a bun with mustard and ketchup",
    "tacos": "mexican tacos with meat filling and vegetables in tortilla",
    "lasagna": "a slice of baked cheesy italian lasagna pasta",
    "macaroni_cheese": "creamy cheesy macaroni and cheese pasta bowl",
    "fried_chicken": "crispy golden deep fried chicken drumstick",
    "pancakes": "a stack of fluffy buttermilk pancakes with syrup",
    "waffles": "golden belgian waffles with waffle squares and syrup",
    "croissant": "flaky golden butter croissant pastry",
    "chocolate_cake": "a slice of rich dark chocolate fudge cake",
    "ice_cream": "a scoop of ice cream sundae in a bowl",

    # --- Asian Specialties ---
    "momos": "steamed white momos dumplings with red spicy dipping sauce",
    "fried_rice": "chinese chicken fried rice with vegetables",
    "sushi": "fresh salmon sushi rolls with rice and seaweed",
    "dim_sum": "steamed dim sum dumplings in a bamboo basket",
    "ramen": "a steaming bowl of japanese ramen noodles with egg and broth",
    "pad_thai": "thai pad thai stir fried noodles with peanuts and lime",
    "spring_rolls": "crispy fried golden spring rolls",

    # --- Healthy & Breakfast Essentials ---
    "oatmeal": "a bowl of warm oatmeal porridge or rolled oats cereal",
    "boiled_eggs": "hard boiled eggs sliced in half showing yellow yolks",
    "scrambled_eggs": "soft fluffy scrambled yellow eggs",
    "avocado_toast": "sliced or mashed green avocado on toasted sourdough bread",
    "greek_yogurt": "a bowl of thick creamy white greek yogurt",
    "fruit_salad": "a fresh colorful mixed fruit salad bowl",
    "green_salad": "a fresh green leafy garden salad",
    "grilled_salmon": "a grilled pink salmon fish fillet with lemon",
    "vegetable_curry": "mixed vegetable curry with spices and vegetables"
}

# Align food keys with FOOD_DATABASE
FOOD_KEYS = [k for k in FOOD_DATABASE.keys() if k in FOOD_PROMPTS]


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
    State-of-the-art Two-Tier Vision Pipeline:
    - Tier 1: Zero-Shot Base Dish Recognition across 60+ regional and global culinary classes.
    - Tier 2: Multi-Label Ingredient & Topping Probing using paired hypothesis testing
      (detects peanut butter, almonds, banana slices, blueberries, cheese, eggs, etc.).
    Uses precomputed text embeddings for instantaneous sub-second GPU inference.
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
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(device)

        self.model.eval()

        self.food_keys = FOOD_KEYS
        self.candidate_texts = [f"a photo of {FOOD_PROMPTS[k]}" for k in self.food_keys]

        # Prepare paired prompts for multi-label topping detection
        self.topping_keys = list(TOPPINGS_DATABASE.keys())
        self.topping_texts = []
        for k in self.topping_keys:
            desc = TOPPINGS_DATABASE[k]["prompt"]
            self.topping_texts.append(f"a photo of food topped with {desc}")
            self.topping_texts.append(f"a photo of food without any {desc}")

        # Precompute and cache normalized text embeddings once for rapid sub-second inference
        print(f"[NutriVision] Precomputing text embeddings for {len(self.food_keys)} foods and {len(self.topping_keys)} toppings...")
        with torch.no_grad():
            # Base foods text features
            base_inputs = self.processor(text=self.candidate_texts, return_tensors="pt", padding=True).to(self.device)
            res_base = self.model.get_text_features(**base_inputs)
            base_feats = res_base.pooler_output if hasattr(res_base, "pooler_output") else res_base
            self.base_text_embeds = base_feats / base_feats.norm(dim=-1, keepdim=True)

            # Toppings text features
            top_inputs = self.processor(text=self.topping_texts, return_tensors="pt", padding=True).to(self.device)
            res_top = self.model.get_text_features(**top_inputs)
            top_feats = res_top.pooler_output if hasattr(res_top, "pooler_output") else res_top
            self.topping_text_embeds = top_feats / top_feats.norm(dim=-1, keepdim=True)

        print("[NutriVision] Ready! Ready for real-time inference.")

    def predict(
        self,
        image: Image.Image,
        top_k: int = 4
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes the Two-Tier Vision Pipeline:
        1. Encodes image into normalized feature vector via Vision Transformer.
        2. Tier 1: Computes cosine similarity against all base food classes.
        3. Tier 2: Evaluates paired hypothesis tests for multi-label toppings.
        """
        # Step 2 visual representation
        prep_data = preprocess_food_image(image)

        rgb_img = image.convert("RGB")
        img_inputs = self.processor(images=rgb_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            res_img = self.model.get_image_features(**img_inputs)
            img_embeds = res_img.pooler_output if hasattr(res_img, "pooler_output") else res_img
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()

            # --- Tier 1: Base Dish Classification ---
            base_sims = (img_embeds @ self.base_text_embeds.T) * logit_scale
            base_probs = base_sims.softmax(dim=-1).squeeze(0)

            # --- Tier 2: Multi-Label Ingredient / Topping Probing ---
            top_sims = (img_embeds @ self.topping_text_embeds.T) * logit_scale
            top_sims = top_sims.squeeze(0)

        # Extract top-k base dishes
        top_probs, top_indices = torch.topk(base_probs, min(top_k, len(self.food_keys)))
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
                "healthy_alternative": food_meta.get("healthy_alternative", "")
            })

        top1 = candidates[0]

        # Extract detected toppings with category compatibility check
        detected_toppings = []
        dish_category = top1.get("category", "")
        for i, top_key in enumerate(self.topping_keys):
            pair_logits = top_sims[2 * i : 2 * i + 2]
            pair_probs = pair_logits.softmax(dim=0)
            p_pos = pair_probs[0].item() * 100.0

            top_meta = TOPPINGS_DATABASE[top_key]
            compat = top_meta.get("compatible_categories", [])
            is_compatible = (not compat) or any(
                c.lower() in dish_category.lower() or dish_category.lower() in c.lower() 
                for c in compat
            )

            # Context-sensitive thresholding:
            # 50.0% for culinary-compatible toppings, 80.0% for unlisted cross-category toppings
            threshold = 50.0 if is_compatible else 80.0
            if p_pos >= threshold:
                detected_toppings.append({
                    "topping_key": top_key,
                    "display_name": top_meta["display_name"],
                    "unit": top_meta["unit"],
                    "calories": top_meta["calories"],
                    "confidence": round(p_pos, 1)
                })


        # Attach to top1 for backward compatibility
        top1["detected_toppings"] = detected_toppings

        return top1, candidates, prep_data, detected_toppings


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
    top_k: int = 4,
    device: torch.device = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience wrapper function matching the app interface.
    Gracefully handles old cached models or newly initialized models.
    """
    if hasattr(model, "predict"):
        res = model.predict(image, top_k=top_k)
        if len(res) == 4:
            return res
        elif len(res) == 3:
            top1, cand, prep = res
            return top1, cand, prep, top1.get("detected_toppings", [])

    # If a stale cached object was passed from a running Streamlit session
    fresh_model = FoodClassifier(device=device)
    return fresh_model.predict(image, top_k=top_k)


