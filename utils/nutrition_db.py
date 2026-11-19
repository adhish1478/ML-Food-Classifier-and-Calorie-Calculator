"""
NutriVision Nutritional Knowledge Base and Dataset Generator.
Contains reference nutritional data for 35+ popular food classes across
Indian, Asian, and Global cuisines.
"""

import os
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Master Food Database with Reference Nutritional Values per 100g
FOOD_DATABASE = {
    # --- Indian Specialties ---
    "chicken_biryani": {
        "display_name": "Chicken Biryani",
        "cuisine": "Indian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 165.0, "protein": 9.2, "carbs": 21.5, "fat": 4.8, "fiber": 1.2},
        "healthy_alternative": "Brown rice chicken biryani with raita and double chicken breast.",
        "tags": ["High Protein", "Comfort Food"]
    },
    "paneer_butter_masala": {
        "display_name": "Paneer Butter Masala",
        "cuisine": "Indian",
        "category": "Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 210.0, "protein": 6.8, "carbs": 8.5, "fat": 16.5, "fiber": 1.8},
        "healthy_alternative": "Tofu tikka masala or low-fat paneer cooked in a tomato-onion gravy without heavy cream.",
        "tags": ["Vegetarian", "Rich in Calcium"]
    },
    "masala_dosa": {
        "display_name": "Masala Dosa",
        "cuisine": "South Indian",
        "category": "Breakfast / Snack",
        "default_portion_g": 200,
        "per_100g": {"calories": 175.0, "protein": 4.1, "carbs": 29.5, "fat": 4.6, "fiber": 2.1},
        "healthy_alternative": "Oats or Ragi dosa with coconut-mint chutney and vegetable sambar.",
        "tags": ["Vegetarian", "Fermented Probiotic"]
    },
    "idli": {
        "display_name": "Idli with Sambar",
        "cuisine": "South Indian",
        "category": "Breakfast",
        "default_portion_g": 220,
        "per_100g": {"calories": 115.0, "protein": 3.8, "carbs": 23.0, "fat": 0.8, "fiber": 2.4},
        "healthy_alternative": "Steamed multigrain or quinoa idli with lentil-rich sambar.",
        "tags": ["Low Fat", "Heart Healthy", "Vegetarian"]
    },
    "samosa": {
        "display_name": "Samosa",
        "cuisine": "Indian",
        "category": "Snack",
        "default_portion_g": 120,
        "per_100g": {"calories": 262.0, "protein": 4.5, "carbs": 32.0, "fat": 13.0, "fiber": 2.5},
        "healthy_alternative": "Air-fried or baked vegetable samosa with mint chutney.",
        "tags": ["High Carb", "Snack"]
    },
    "chole_bhature": {
        "display_name": "Chole Bhature",
        "cuisine": "North Indian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 245.0, "protein": 6.5, "carbs": 34.0, "fat": 9.8, "fiber": 4.5},
        "healthy_alternative": "Boiled chole with whole-wheat baked roti and onion salad.",
        "tags": ["High Energy", "High Fiber"]
    },
    "dal_makhani": {
        "display_name": "Dal Makhani",
        "cuisine": "North Indian",
        "category": "Lentil / Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 140.0, "protein": 5.5, "carbs": 16.0, "fat": 6.2, "fiber": 4.0},
        "healthy_alternative": "Yellow tadka dal cooked with minimal ghee and fresh cilantro.",
        "tags": ["Vegetarian", "High Fiber"]
    },
    "butter_chicken": {
        "display_name": "Butter Chicken",
        "cuisine": "North Indian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 185.0, "protein": 12.0, "carbs": 6.5, "fat": 12.5, "fiber": 1.1},
        "healthy_alternative": "Tandoori chicken breast with spiced yogurt dip and mint sauce.",
        "tags": ["High Protein", "Keto Friendly"]
    },
    "gulab_jamun": {
        "display_name": "Gulab Jamun",
        "cuisine": "Indian",
        "category": "Dessert",
        "default_portion_g": 100,
        "per_100g": {"calories": 320.0, "protein": 4.0, "carbs": 52.0, "fat": 11.0, "fiber": 0.5},
        "healthy_alternative": "Steamed sandesh or honey-baked yogurt (Bhapa Doi).",
        "tags": ["High Sugar", "Dessert"]
    },
    "roti_naan": {
        "display_name": "Butter Naan / Flatbread",
        "cuisine": "Indian",
        "category": "Breads",
        "default_portion_g": 100,
        "per_100g": {"calories": 290.0, "protein": 8.0, "carbs": 50.0, "fat": 6.5, "fiber": 2.2},
        "healthy_alternative": "100% whole wheat tandoori roti without butter.",
        "tags": ["High Carb"]
    },

    # --- Western & Fast Food ---
    "pizza": {
        "display_name": "Cheese Pizza",
        "cuisine": "Italian",
        "category": "Fast Food",
        "default_portion_g": 250,
        "per_100g": {"calories": 266.0, "protein": 11.0, "carbs": 33.0, "fat": 10.0, "fiber": 2.3},
        "healthy_alternative": "Thin-crust cauliflower base pizza with part-skim mozzarella and bell peppers.",
        "tags": ["High Carb", "Comfort Food"]
    },
    "burger": {
        "display_name": "Cheeseburger",
        "cuisine": "American",
        "category": "Fast Food",
        "default_portion_g": 220,
        "per_100g": {"calories": 250.0, "protein": 13.0, "carbs": 24.0, "fat": 11.5, "fiber": 1.4},
        "healthy_alternative": "Grilled turkey or black bean burger on a lettuce wrap.",
        "tags": ["High Protein", "Fast Food"]
    },
    "french_fries": {
        "display_name": "French Fries",
        "cuisine": "American",
        "category": "Snack",
        "default_portion_g": 150,
        "per_100g": {"calories": 312.0, "protein": 3.4, "carbs": 41.5, "fat": 15.0, "fiber": 3.8},
        "healthy_alternative": "Air-fried sweet potato wedges seasoned with paprika.",
        "tags": ["High Fat", "High Carb"]
    },
    "caesar_salad": {
        "display_name": "Chicken Caesar Salad",
        "cuisine": "Continental",
        "category": "Salad / Healthy",
        "default_portion_g": 250,
        "per_100g": {"calories": 125.0, "protein": 8.5, "carbs": 4.5, "fat": 8.0, "fiber": 2.0},
        "healthy_alternative": "Caesar salad with olive oil & lemon vinaigrette, omitting croutons.",
        "tags": ["Keto Friendly", "Low Carb"]
    },
    "spaghetti_bolognese": {
        "display_name": "Spaghetti Bolognese",
        "cuisine": "Italian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 145.0, "protein": 7.5, "carbs": 18.0, "fat": 4.8, "fiber": 1.8},
        "healthy_alternative": "Zucchini noodles (zoodles) or whole-wheat pasta with lean beef/lentil sauce.",
        "tags": ["High Carb", "Balanced"]
    },
    "grilled_chicken": {
        "display_name": "Herb Grilled Chicken Breast",
        "cuisine": "Continental",
        "category": "Lean Protein",
        "default_portion_g": 200,
        "per_100g": {"calories": 165.0, "protein": 31.0, "carbs": 0.5, "fat": 3.6, "fiber": 0.0},
        "healthy_alternative": "Already an optimal lean protein source! Pair with steamed broccoli.",
        "tags": ["High Protein", "Keto Friendly", "Zero Sugar"]
    },
    "sandwich": {
        "display_name": "Club Sandwich",
        "cuisine": "Continental",
        "category": "Snack / Lunch",
        "default_portion_g": 220,
        "per_100g": {"calories": 220.0, "protein": 10.5, "carbs": 26.0, "fat": 8.0, "fiber": 2.5},
        "healthy_alternative": "Multigrain open-faced sandwich with avocado, boiled egg, and spinach.",
        "tags": ["Balanced"]
    },
    "steak": {
        "display_name": "Grilled Beef Steak",
        "cuisine": "Continental",
        "category": "Main Course",
        "default_portion_g": 250,
        "per_100g": {"calories": 215.0, "protein": 26.0, "carbs": 0.0, "fat": 12.0, "fiber": 0.0},
        "healthy_alternative": "Grass-fed lean sirloin or grilled portobello mushrooms.",
        "tags": ["High Protein", "Iron Rich", "Low Carb"]
    },

    # --- Asian Specialties ---
    "fried_rice": {
        "display_name": "Chicken Fried Rice",
        "cuisine": "Asian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 163.0, "protein": 6.2, "carbs": 24.5, "fat": 4.5, "fiber": 1.5},
        "healthy_alternative": "Cauliflower fried rice with edamame and scrambled egg whites.",
        "tags": ["High Carb"]
    },
    "sushi": {
        "display_name": "Salmon Sushi Roll",
        "cuisine": "Japanese",
        "category": "Seafood",
        "default_portion_g": 200,
        "per_100g": {"calories": 142.0, "protein": 6.5, "carbs": 22.0, "fat": 2.8, "fiber": 1.1},
        "healthy_alternative": "Salmon sashimi or brown rice sushi with cucumber and avocado.",
        "tags": ["Omega-3 Rich", "Heart Healthy"]
    },
    "dim_sum": {
        "display_name": "Steamed Chicken Dim Sum / Dumplings",
        "cuisine": "Asian",
        "category": "Snack",
        "default_portion_g": 180,
        "per_100g": {"calories": 135.0, "protein": 8.0, "carbs": 18.0, "fat": 3.2, "fiber": 1.2},
        "healthy_alternative": "Steamed crystal vegetable dumplings with light soy sauce.",
        "tags": ["Steamed", "Low Fat"]
    },
    "ramen": {
        "display_name": "Japanese Ramen Bowl",
        "cuisine": "Japanese",
        "category": "Soup / Noodles",
        "default_portion_g": 450,
        "per_100g": {"calories": 110.0, "protein": 4.5, "carbs": 15.0, "fat": 3.8, "fiber": 1.0},
        "healthy_alternative": "Shirataki or soba noodles in a clear miso-vegetable broth with bok choy.",
        "tags": ["Comfort Food", "High Sodium"]
    },
    "pad_thai": {
        "display_name": "Pad Thai Noodles",
        "cuisine": "Thai",
        "category": "Noodles",
        "default_portion_g": 320,
        "per_100g": {"calories": 178.0, "protein": 7.0, "carbs": 26.0, "fat": 5.0, "fiber": 2.0},
        "healthy_alternative": "Glass noodle stir-fry with tofu, sprouts, and reduced peanut sauce.",
        "tags": ["High Energy"]
    },

    # --- Healthy & Breakfast Essentials ---
    "oatmeal": {
        "display_name": "Oatmeal with Berries & Nuts",
        "cuisine": "Global",
        "category": "Breakfast",
        "default_portion_g": 250,
        "per_100g": {"calories": 95.0, "protein": 3.2, "carbs": 16.5, "fat": 1.8, "fiber": 3.2},
        "healthy_alternative": "Already an exceptional superfood! Add chia seeds and protein powder.",
        "tags": ["High Fiber", "Heart Healthy", "Low Glycemic"]
    },
    "boiled_eggs": {
        "display_name": "Boiled Eggs (2 Large)",
        "cuisine": "Global",
        "category": "Breakfast / Protein",
        "default_portion_g": 100,
        "per_100g": {"calories": 155.0, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "fiber": 0.0},
        "healthy_alternative": "Steamed egg whites with black pepper and turmeric.",
        "tags": ["High Protein", "Keto Friendly", "Choline Rich"]
    },
    "greek_yogurt": {
        "display_name": "Greek Yogurt Bowl",
        "cuisine": "Mediterranean",
        "category": "Dairy / Breakfast",
        "default_portion_g": 200,
        "per_100g": {"calories": 73.0, "protein": 10.0, "carbs": 3.6, "fat": 1.9, "fiber": 0.0},
        "healthy_alternative": "Zero-fat plain Greek yogurt with fresh blueberries and walnuts.",
        "tags": ["High Protein", "Probiotic", "Calcium Rich"]
    },
    "fruit_salad": {
        "display_name": "Fresh Fruit Medley",
        "cuisine": "Global",
        "category": "Salad / Dessert",
        "default_portion_g": 250,
        "per_100g": {"calories": 58.0, "protein": 0.8, "carbs": 14.5, "fat": 0.2, "fiber": 2.4},
        "healthy_alternative": "Add mint leaves and a squeeze of lime; limit high-glycemic fruits if diabetic.",
        "tags": ["Vitamins A & C", "Hydrating", "Antioxidant Rich"]
    },
    "green_salad": {
        "display_name": "Mediterranean Green Salad",
        "cuisine": "Mediterranean",
        "category": "Salad / Healthy",
        "default_portion_g": 200,
        "per_100g": {"calories": 45.0, "protein": 1.8, "carbs": 4.2, "fat": 2.5, "fiber": 2.8},
        "healthy_alternative": "Top with pumpkin seeds and cold-pressed extra virgin olive oil.",
        "tags": ["Super Low Calorie", "High Fiber", "Keto Friendly"]
    },
    "pancakes": {
        "display_name": "Fluffy Buttermilk Pancakes",
        "cuisine": "American",
        "category": "Breakfast / Dessert",
        "default_portion_g": 180,
        "per_100g": {"calories": 227.0, "protein": 6.4, "carbs": 38.0, "fat": 5.5, "fiber": 1.3},
        "healthy_alternative": "Banana oat protein pancakes with sugar-free maple syrup.",
        "tags": ["High Carb", "Breakfast"]
    },
    "vegetable_curry": {
        "display_name": "Mixed Vegetable Curry",
        "cuisine": "Indian",
        "category": "Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 95.0, "protein": 2.8, "carbs": 11.5, "fat": 4.2, "fiber": 3.5},
        "healthy_alternative": "Cook in coconut water with minimal oil; use green leafy vegetables.",
        "tags": ["Vegetarian", "High Fiber", "Antioxidant Rich"]
    }
}

# Ordered class list for model index alignment
FOOD_CLASSES = sorted(list(FOOD_DATABASE.keys()))
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(FOOD_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(FOOD_CLASSES)}


def get_nutrition_for_portion(food_key: str, portion_grams: float):
    """
    Computes absolute nutrition values for a given food and portion in grams.
    """
    food = FOOD_DATABASE.get(food_key)
    if not food:
        # Fallback to chicken_biryani
        food = FOOD_DATABASE["chicken_biryani"]

    factor = max(portion_grams, 10.0) / 100.0
    p100 = food["per_100g"]

    return {
        "food_key": food_key,
        "display_name": food["display_name"],
        "category": food["category"],
        "cuisine": food["cuisine"],
        "portion_grams": round(portion_grams, 1),
        "calories": round(p100["calories"] * factor, 1),
        "protein": round(p100["protein"] * factor, 1),
        "carbs": round(p100["carbs"] * factor, 1),
        "fat": round(p100["fat"] * factor, 1),
        "fiber": round(p100["fiber"] * factor, 1),
        "healthy_alternative": food["healthy_alternative"],
        "tags": food["tags"]
    }


def generate_ann_training_dataset(num_samples: int = 5000, random_seed: int = 42):
    """
    Generates a realistic synthetic tabular dataset for training the Calorie
    Regression ANN model, following the biological Atwater energy conversion
    factors with natural preparation and culinary variance.
    """
    np.random.seed(random_seed)
    data = []

    for _ in range(num_samples):
        food_key = np.random.choice(FOOD_CLASSES)
        food = FOOD_DATABASE[food_key]
        class_idx = CLASS_TO_IDX[food_key]

        # Portion size sampled realistically around default portion
        def_portion = food["default_portion_g"]
        portion_g = np.random.uniform(max(50.0, def_portion * 0.4), def_portion * 2.2)

        factor = portion_g / 100.0
        p100 = food["per_100g"]

        # Culinary variance (+/- 10%)
        var_p = np.random.normal(1.0, 0.08)
        var_c = np.random.normal(1.0, 0.08)
        var_f = np.random.normal(1.0, 0.10)
        var_fib = np.random.normal(1.0, 0.08)

        protein = max(0.2, p100["protein"] * factor * var_p)
        carbs = max(0.2, p100["carbs"] * factor * var_c)
        fat = max(0.1, p100["fat"] * factor * var_f)
        fiber = max(0.0, p100["fiber"] * factor * var_fib)

        # Standard Atwater regression target with realistic thermal variance (+/- 4%)
        # 4 kcal/g protein, 4 kcal/g carb, 9 kcal/g fat, 2 kcal/g fiber
        base_kcal = (protein * 4.0) + (carbs * 4.0) + (fat * 9.0) + (fiber * 2.0)
        noise = np.random.normal(0.0, 0.03 * base_kcal)
        calories = max(10.0, base_kcal + noise)

        data.append({
            "class_idx": class_idx,
            "food_name": food_key,
            "portion_grams": round(portion_g, 2),
            "protein_g": round(protein, 2),
            "carbs_g": round(carbs, 2),
            "fat_g": round(fat, 2),
            "fiber_g": round(fiber, 2),
            "calories_kcal": round(calories, 2)
        })

    df = pd.DataFrame(data)
    return df


import csv

def generate_and_save_dataset_csv(csv_path: str, num_samples: int = 6000, random_seed: int = 42):
    """
    Generates synthetic nutrition dataset and writes directly to CSV file.
    Does not require pandas to be installed.
    """
    np.random.seed(random_seed)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["class_idx", "food_name", "portion_grams", "protein_g", "carbs_g", "fat_g", "fiber_g", "calories_kcal"]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_samples):
            food_key = np.random.choice(FOOD_CLASSES)
            food = FOOD_DATABASE[food_key]
            class_idx = CLASS_TO_IDX[food_key]

            def_portion = food["default_portion_g"]
            portion_g = np.random.uniform(max(50.0, def_portion * 0.4), def_portion * 2.2)

            factor = portion_g / 100.0
            p100 = food["per_100g"]

            var_p = np.random.normal(1.0, 0.08)
            var_c = np.random.normal(1.0, 0.08)
            var_f = np.random.normal(1.0, 0.10)
            var_fib = np.random.normal(1.0, 0.08)

            protein = max(0.2, p100["protein"] * factor * var_p)
            carbs = max(0.2, p100["carbs"] * factor * var_c)
            fat = max(0.1, p100["fat"] * factor * var_f)
            fiber = max(0.0, p100["fiber"] * factor * var_fib)

            base_kcal = (protein * 4.0) + (carbs * 4.0) + (fat * 9.0) + (fiber * 2.0)
            noise = np.random.normal(0.0, 0.03 * base_kcal)
            calories = max(10.0, base_kcal + noise)

            writer.writerow({
                "class_idx": class_idx,
                "food_name": food_key,
                "portion_grams": round(portion_g, 2),
                "protein_g": round(protein, 2),
                "carbs_g": round(carbs, 2),
                "fat_g": round(fat, 2),
                "fiber_g": round(fiber, 2),
                "calories_kcal": round(calories, 2)
            })

    print(f"Successfully generated {num_samples} samples -> {csv_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    csv_file = os.path.join(data_dir, "food_nutrition_dataset.csv")
    generate_and_save_dataset_csv(csv_file, num_samples=6000)

