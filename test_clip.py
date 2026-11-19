import os
import sys
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nutrition_db import FOOD_CLASSES, FOOD_DATABASE

MODEL_ID = "openai/clip-vit-base-patch32"
print(f"Loading CLIP model {MODEL_ID}...")

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"Running on device: {device}")

model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

# Create rich, natural food prompts for all 30+ food classes
FOOD_PROMPTS = {
    "oatmeal": "a bowl of oatmeal porridge with sliced bananas and peanut butter",
    "gulab_jamun": "gulab jamun sweet indian dessert soaked in sugar syrup",
    "chicken_biryani": "a plate of spiced chicken biryani with basmati rice",
    "pizza": "a slice of hot cheesy pizza with toppings",
    "burger": "a burger with beef or chicken patty and cheese in a bun",
    "caesar_salad": "a fresh chicken caesar salad with green lettuce",
    "masala_dosa": "a crispy south indian masala dosa with potato curry filling",
    "idli": "steamed white idli rice cakes with sambar",
    "samosa": "crispy fried indian samosa pastry",
    "paneer_butter_masala": "paneer butter masala curry with cubes of cottage cheese",
    "butter_chicken": "creamy butter chicken curry",
    "dal_makhani": "creamy black dal makhani lentil soup",
    "chole_bhature": "chole bhature chickpeas with puffed bread",
    "roti_naan": "flatbread naan or roti",
    "french_fries": "golden crispy french fries",
    "spaghetti_bolognese": "spaghetti pasta with bolognese meat sauce",
    "grilled_chicken": "grilled chicken breast with vegetables",
    "sandwich": "a sandwich with bread, lettuce, and fillings",
    "steak": "a grilled beef steak",
    "fried_rice": "fried rice with vegetables and eggs",
    "sushi": "fresh sushi rolls with fish and seaweed",
    "dim_sum": "steamed dumplings dim sum in a basket",
    "ramen": "a hot bowl of ramen noodle soup with broth",
    "pad_thai": "pad thai stir fried noodles",
    "boiled_eggs": "hard boiled eggs cut in half",
    "greek_yogurt": "a bowl of white greek yogurt",
    "fruit_salad": "a bowl of fresh mixed fruit salad",
    "green_salad": "a fresh green garden salad with leaves",
    "pancakes": "a stack of pancakes with syrup",
    "vegetable_curry": "mixed vegetable curry with gravy"
}

food_keys = list(FOOD_PROMPTS.keys())
candidate_texts = [f"a photo of {FOOD_PROMPTS[k]}" for k in food_keys]

for fname in ["1.jpg", "2.JPG"]:
    img_path = os.path.join(PROJECT_ROOT, "my_samples", fname)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=candidate_texts, images=img, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, num_classes)
            probs = logits_per_image.softmax(dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, 3)

        print(f"\n==================== {fname} ====================")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            k = food_keys[idx.item()]
            name = FOOD_DATABASE[k]["display_name"]
            print(f"Rank {i+1}: {name} ({prob.item()*100:.2f}%) -> Prompt: '{FOOD_PROMPTS[k]}'")
