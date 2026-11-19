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
    # ==================== Indian Specialties & Street Foods ====================
    "vada_pav": {
        "display_name": "Vada Pav",
        "cuisine": "Indian (Street Food)",
        "category": "Snack",
        "default_portion_g": 150,
        "per_100g": {"calories": 265.0, "protein": 5.2, "carbs": 38.0, "fat": 10.5, "fiber": 2.8},
        "healthy_alternative": "Air-fried batata vada in a whole-wheat pav bun with roasted garlic-chili dry chutney.",
        "tags": ["Street Food", "High Carb", "Vegetarian"]
    },
    "pav_bhaji": {
        "display_name": "Pav Bhaji",
        "cuisine": "Indian (Street Food)",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 160.0, "protein": 3.8, "carbs": 24.5, "fat": 5.5, "fiber": 3.2},
        "healthy_alternative": "Cauliflower-beetroot-carrot bhaji cooked with cold-pressed oil and multigrain toasted pav.",
        "tags": ["Street Food", "High Fiber", "Vegetarian"]
    },
    "samosa": {
        "display_name": "Samosa",
        "cuisine": "Indian",
        "category": "Snack",
        "default_portion_g": 120,
        "per_100g": {"calories": 262.0, "protein": 4.5, "carbs": 32.0, "fat": 13.0, "fiber": 2.5},
        "healthy_alternative": "Air-fried or baked vegetable samosa with mint coriander chutney.",
        "tags": ["High Carb", "Snack", "Vegetarian"]
    },
    "pani_puri": {
        "display_name": "Pani Puri / Golgappa",
        "cuisine": "Indian (Street Food)",
        "category": "Snack",
        "default_portion_g": 180,
        "per_100g": {"calories": 145.0, "protein": 2.8, "carbs": 26.0, "fat": 3.5, "fiber": 2.1},
        "healthy_alternative": "Baked whole wheat puris filled with boiled sprouted moong and fresh herb water.",
        "tags": ["Street Food", "Hydrating", "Vegetarian"]
    },
    "bhel_puri": {
        "display_name": "Bhel Puri / Chaat",
        "cuisine": "Indian (Street Food)",
        "category": "Snack",
        "default_portion_g": 160,
        "per_100g": {"calories": 168.0, "protein": 4.2, "carbs": 30.5, "fat": 3.8, "fiber": 2.6},
        "healthy_alternative": "Puffed rice and roasted chana bhel with raw mango, onions, tomatoes, and tamarind.",
        "tags": ["Street Food", "Low Fat", "Vegetarian"]
    },
    "chicken_biryani": {
        "display_name": "Chicken Biryani",
        "cuisine": "Indian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 165.0, "protein": 9.2, "carbs": 21.5, "fat": 4.8, "fiber": 1.2},
        "healthy_alternative": "Brown rice chicken biryani with cucumber raita and lean chicken breast.",
        "tags": ["High Protein", "Comfort Food"]
    },
    "mutton_biryani": {
        "display_name": "Mutton Biryani",
        "cuisine": "Indian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 195.0, "protein": 11.5, "carbs": 21.0, "fat": 7.5, "fiber": 1.0},
        "healthy_alternative": "Lean trimmed lamb or mutton biryani served with high-protein cucumber onion raita.",
        "tags": ["High Protein", "Iron Rich"]
    },
    "veg_biryani": {
        "display_name": "Vegetable Biryani / Pulao",
        "cuisine": "Indian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 140.0, "protein": 3.6, "carbs": 25.0, "fat": 3.0, "fiber": 2.4},
        "healthy_alternative": "Quinoa or brown basmati pulao packed with green peas, carrots, and french beans.",
        "tags": ["Vegetarian", "Balanced"]
    },
    "paneer_butter_masala": {
        "display_name": "Paneer Butter Masala",
        "cuisine": "Indian",
        "category": "Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 210.0, "protein": 6.8, "carbs": 8.5, "fat": 16.5, "fiber": 1.8},
        "healthy_alternative": "Low-fat paneer or firm tofu in a slow-simmered tomato-cashew sauce without heavy butter.",
        "tags": ["Vegetarian", "Calcium Rich"]
    },
    "palak_paneer": {
        "display_name": "Palak Paneer",
        "cuisine": "Indian",
        "category": "Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 145.0, "protein": 7.5, "carbs": 6.2, "fat": 10.5, "fiber": 3.4},
        "healthy_alternative": "Blanched spinach puree with grilled tofu or low-fat cottage cheese.",
        "tags": ["Vegetarian", "Iron Rich", "Low Carb"]
    },
    "butter_chicken": {
        "display_name": "Butter Chicken",
        "cuisine": "North Indian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 185.0, "protein": 12.0, "carbs": 6.5, "fat": 12.5, "fiber": 1.1},
        "healthy_alternative": "Tandoori chicken breast pieces in a spiced tomato yogurt gravy without heavy cream.",
        "tags": ["High Protein", "Comfort Food"]
    },
    "chicken_tikka": {
        "display_name": "Tandoori Chicken Tikka",
        "cuisine": "Indian",
        "category": "Lean Protein",
        "default_portion_g": 200,
        "per_100g": {"calories": 150.0, "protein": 24.0, "carbs": 2.5, "fat": 4.8, "fiber": 0.5},
        "healthy_alternative": "Clay-oven grilled chicken breast with lemon wedge and mint dip. Already an ideal protein choice!",
        "tags": ["High Protein", "Keto Friendly", "Low Carb"]
    },
    "masala_dosa": {
        "display_name": "Masala Dosa",
        "cuisine": "South Indian",
        "category": "Breakfast / Snack",
        "default_portion_g": 200,
        "per_100g": {"calories": 175.0, "protein": 4.1, "carbs": 29.5, "fat": 4.6, "fiber": 2.1},
        "healthy_alternative": "Ragi (finger millet) or oats dosa with vegetable sambar and coriander chutney.",
        "tags": ["Vegetarian", "Fermented Probiotic"]
    },
    "plain_dosa": {
        "display_name": "Crispy Plain Dosa",
        "cuisine": "South Indian",
        "category": "Breakfast",
        "default_portion_g": 160,
        "per_100g": {"calories": 165.0, "protein": 3.8, "carbs": 31.0, "fat": 3.2, "fiber": 1.8},
        "healthy_alternative": "Mung bean (Pesarattu) crepe with fresh ginger-chili chutney.",
        "tags": ["Vegetarian", "Low Fat"]
    },
    "idli": {
        "display_name": "Idli with Sambar",
        "cuisine": "South Indian",
        "category": "Breakfast",
        "default_portion_g": 220,
        "per_100g": {"calories": 115.0, "protein": 3.8, "carbs": 23.0, "fat": 0.8, "fiber": 2.4},
        "healthy_alternative": "Steamed multigrain or quinoa idli with high-protein pigeon pea sambar.",
        "tags": ["Low Fat", "Heart Healthy", "Vegetarian", "Fermented Probiotic"]
    },
    "medu_vada": {
        "display_name": "Medu Vada",
        "cuisine": "South Indian",
        "category": "Breakfast / Snack",
        "default_portion_g": 140,
        "per_100g": {"calories": 235.0, "protein": 6.8, "carbs": 25.0, "fat": 12.0, "fiber": 3.5},
        "healthy_alternative": "Appe-pan or air-fried urad dal vadas served with spicy tomato-onion chutney.",
        "tags": ["High Protein", "Vegetarian"]
    },
    "poha": {
        "display_name": "Kanda Poha",
        "cuisine": "Indian (West / Central)",
        "category": "Breakfast",
        "default_portion_g": 200,
        "per_100g": {"calories": 130.0, "protein": 2.8, "carbs": 24.5, "fat": 2.8, "fiber": 2.2},
        "healthy_alternative": "Red rice poha with roasted peanuts, curry leaves, turmeric, and lemon squeeze.",
        "tags": ["Iron Rich", "Low Fat", "Vegetarian"]
    },
    "upma": {
        "display_name": "Rava Vegetable Upma",
        "cuisine": "South Indian",
        "category": "Breakfast",
        "default_portion_g": 200,
        "per_100g": {"calories": 120.0, "protein": 3.0, "carbs": 22.0, "fat": 2.5, "fiber": 2.0},
        "healthy_alternative": "Broken wheat (dalia) or rolled oats upma with mixed peas, carrots, and beans.",
        "tags": ["Vegetarian", "Fiber Rich"]
    },
    "chole_bhature": {
        "display_name": "Chole Bhature",
        "cuisine": "North Indian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 245.0, "protein": 6.5, "carbs": 34.0, "fat": 9.8, "fiber": 4.5},
        "healthy_alternative": "Boiled kabuli chana with whole-wheat baked roti or phulka and onion salad.",
        "tags": ["High Energy", "High Fiber", "Vegetarian"]
    },
    "dal_makhani": {
        "display_name": "Dal Makhani",
        "cuisine": "North Indian",
        "category": "Lentil / Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 140.0, "protein": 5.5, "carbs": 16.0, "fat": 6.2, "fiber": 4.0},
        "healthy_alternative": "Slow-cooked black lentils with whole spices, substituting cream with low-fat yogurt.",
        "tags": ["Vegetarian", "High Fiber"]
    },
    "dal_tadka": {
        "display_name": "Yellow Dal Tadka",
        "cuisine": "Indian",
        "category": "Lentil / Curry",
        "default_portion_g": 250,
        "per_100g": {"calories": 110.0, "protein": 6.2, "carbs": 15.0, "fat": 3.2, "fiber": 3.8},
        "healthy_alternative": "Arhar/Toor dal tempered with cumin, garlic, and fresh coriander using 1 tsp ghee.",
        "tags": ["High Protein", "Vegetarian", "Heart Healthy"]
    },
    "rajma_chawal": {
        "display_name": "Rajma Chawal",
        "cuisine": "North Indian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 135.0, "protein": 5.2, "carbs": 23.5, "fat": 2.8, "fiber": 3.9},
        "healthy_alternative": "Spiced kidney bean curry served with fiber-rich brown rice.",
        "tags": ["High Fiber", "Vegetarian", "Balanced Meal"]
    },
    "aloo_paratha": {
        "display_name": "Aloo Paratha with Butter",
        "cuisine": "North Indian",
        "category": "Breakfast / Bread",
        "default_portion_g": 180,
        "per_100g": {"calories": 240.0, "protein": 4.8, "carbs": 36.0, "fat": 8.8, "fiber": 3.2},
        "healthy_alternative": "Whole wheat stuffed flatbread dry-roasted on tawa, served with plain probiotic curd.",
        "tags": ["High Carb", "Vegetarian"]
    },
    "paneer_paratha": {
        "display_name": "Stuffed Paneer Paratha",
        "cuisine": "North Indian",
        "category": "Breakfast / Bread",
        "default_portion_g": 180,
        "per_100g": {"calories": 265.0, "protein": 9.5, "carbs": 32.0, "fat": 11.2, "fiber": 3.0},
        "healthy_alternative": "Whole wheat flatbread stuffed with low-fat paneer, green chilies, and mint.",
        "tags": ["High Protein", "Vegetarian"]
    },
    "poori_bhaji": {
        "display_name": "Poori with Aloo Bhaji",
        "cuisine": "Indian",
        "category": "Breakfast",
        "default_portion_g": 260,
        "per_100g": {"calories": 255.0, "protein": 4.2, "carbs": 35.0, "fat": 11.0, "fiber": 2.8},
        "healthy_alternative": "Baked whole wheat poori or phulka with cumin-turmeric potato sabzi.",
        "tags": ["Vegetarian", "High Energy"]
    },
    "khichdi": {
        "display_name": "Moong Dal Khichdi",
        "cuisine": "Indian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 115.0, "protein": 4.5, "carbs": 19.5, "fat": 2.2, "fiber": 2.8},
        "healthy_alternative": "Equal-ratio yellow moong dal and brown rice/millet khichdi with cumin ghee tadka.",
        "tags": ["Easily Digestible", "Gut Friendly", "Vegetarian"]
    },
    "curd_rice": {
        "display_name": "South Indian Curd Rice",
        "cuisine": "South Indian",
        "category": "Main Course",
        "default_portion_g": 250,
        "per_100g": {"calories": 130.0, "protein": 3.5, "carbs": 21.0, "fat": 3.6, "fiber": 1.2},
        "healthy_alternative": "Probiotic curd rice with pomegranate seeds and mustard-curry leaf tempering.",
        "tags": ["Probiotic", "Gut Friendly", "Vegetarian"]
    },
    "uttapam": {
        "display_name": "Onion Tomato Uttapam",
        "cuisine": "South Indian",
        "category": "Breakfast",
        "default_portion_g": 200,
        "per_100g": {"calories": 155.0, "protein": 4.2, "carbs": 27.0, "fat": 3.5, "fiber": 2.5},
        "healthy_alternative": "Oats or vegetable-loaded uttapam with coconut cilantro dip.",
        "tags": ["Vegetarian", "Fermented"]
    },
    "dhokla": {
        "display_name": "Khaman Dhokla",
        "cuisine": "Gujarati",
        "category": "Snack",
        "default_portion_g": 150,
        "per_100g": {"calories": 160.0, "protein": 5.8, "carbs": 26.5, "fat": 3.8, "fiber": 2.5},
        "healthy_alternative": "Steamed gram flour dhokla tempered with mustard seeds and curry leaves. Naturally healthy!",
        "tags": ["Steamed", "Low Fat", "Vegetarian"]
    },
    "roti_naan": {
        "display_name": "Butter Naan / Roti",
        "cuisine": "Indian",
        "category": "Breads",
        "default_portion_g": 100,
        "per_100g": {"calories": 290.0, "protein": 8.0, "carbs": 50.0, "fat": 6.5, "fiber": 2.2},
        "healthy_alternative": "100% whole wheat tandoori roti or jowar bhakri without butter.",
        "tags": ["High Carb", "Vegetarian"]
    },
    "fish_curry": {
        "display_name": "Spiced Fish Curry",
        "cuisine": "Coastal Indian",
        "category": "Main Course",
        "default_portion_g": 250,
        "per_100g": {"calories": 135.0, "protein": 15.0, "carbs": 4.5, "fat": 6.5, "fiber": 1.2},
        "healthy_alternative": "Coastal fish curry cooked in kokum and light coconut milk broth.",
        "tags": ["High Protein", "Omega-3 Rich"]
    },
    "egg_curry": {
        "display_name": "Boiled Egg Curry",
        "cuisine": "Indian",
        "category": "Main Course",
        "default_portion_g": 250,
        "per_100g": {"calories": 140.0, "protein": 10.5, "carbs": 5.5, "fat": 8.8, "fiber": 1.4},
        "healthy_alternative": "Hard-boiled eggs simmered in roasted onion-tomato-garlic gravy.",
        "tags": ["High Protein", "Keto Friendly"]
    },
    "gulab_jamun": {
        "display_name": "Gulab Jamun",
        "cuisine": "Indian",
        "category": "Dessert",
        "default_portion_g": 100,
        "per_100g": {"calories": 320.0, "protein": 4.0, "carbs": 52.0, "fat": 11.0, "fiber": 0.5},
        "healthy_alternative": "Steamed Bengali sandesh or honey-baked yogurt (Bhapa Doi).",
        "tags": ["High Sugar", "Dessert", "Vegetarian"]
    },
    "rasgulla": {
        "display_name": "Spongy Rasgulla",
        "cuisine": "Indian (East)",
        "category": "Dessert",
        "default_portion_g": 100,
        "per_100g": {"calories": 185.0, "protein": 4.5, "carbs": 38.0, "fat": 1.8, "fiber": 0.2},
        "healthy_alternative": "Steamed fresh chenna balls with cardamom and light jaggery syrup.",
        "tags": ["Low Fat", "Dessert", "Vegetarian"]
    },
    "jalebi": {
        "display_name": "Crispy Jalebi",
        "cuisine": "Indian",
        "category": "Dessert",
        "default_portion_g": 100,
        "per_100g": {"calories": 370.0, "protein": 2.5, "carbs": 68.0, "fat": 10.5, "fiber": 0.3},
        "healthy_alternative": "Baked apple rings dusted with cinnamon or jaggery ragi halwa.",
        "tags": ["High Sugar", "High Carb", "Dessert"]
    },
    "kheer": {
        "display_name": "Rice Kheer / Payasam",
        "cuisine": "Indian",
        "category": "Dessert",
        "default_portion_g": 150,
        "per_100g": {"calories": 175.0, "protein": 4.8, "carbs": 26.5, "fat": 6.0, "fiber": 0.8},
        "healthy_alternative": "Oats or foxnut (makhana) kheer sweetened with dates and almond milk.",
        "tags": ["Calcium Rich", "Dessert"]
    },
    "gajar_halwa": {
        "display_name": "Gajar Ka Halwa",
        "cuisine": "North Indian",
        "category": "Dessert",
        "default_portion_g": 150,
        "per_100g": {"calories": 260.0, "protein": 4.2, "carbs": 34.0, "fat": 12.0, "fiber": 2.5},
        "healthy_alternative": "Grated carrots simmered in low-fat cow milk and sweetened with dates.",
        "tags": ["Vitamin A Rich", "Dessert"]
    },
    "chai": {
        "display_name": "Masala Chai / Milk Tea",
        "cuisine": "Indian",
        "category": "Beverage",
        "default_portion_g": 180,
        "per_100g": {"calories": 65.0, "protein": 2.1, "carbs": 8.5, "fat": 2.5, "fiber": 0.0},
        "healthy_alternative": "Ginger-tulsi black tea or green tea without refined sugar.",
        "tags": ["Beverage", "Antioxidant Rich"]
    },
    "lassi": {
        "display_name": "Sweet Mango / Yogurt Lassi",
        "cuisine": "North Indian",
        "category": "Beverage",
        "default_portion_g": 250,
        "per_100g": {"calories": 95.0, "protein": 3.6, "carbs": 14.5, "fat": 2.8, "fiber": 0.4},
        "healthy_alternative": "Chilled spiced chaas (buttermilk) with roasted cumin and mint.",
        "tags": ["Probiotic", "Refreshing", "Calcium Rich"]
    },

    # ==================== Western & Continental Favorites ====================
    "pizza": {
        "display_name": "Cheese Pizza",
        "cuisine": "Italian",
        "category": "Fast Food",
        "default_portion_g": 250,
        "per_100g": {"calories": 266.0, "protein": 11.0, "carbs": 33.0, "fat": 10.0, "fiber": 2.3},
        "healthy_alternative": "Thin-crust sourdough or cauliflower crust pizza with part-skim mozzarella.",
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
        "healthy_alternative": "Air-fried sweet potato wedges seasoned with smoked paprika and sea salt.",
        "tags": ["High Fat", "High Carb"]
    },
    "caesar_salad": {
        "display_name": "Chicken Caesar Salad",
        "cuisine": "Continental",
        "category": "Salad / Healthy",
        "default_portion_g": 250,
        "per_100g": {"calories": 125.0, "protein": 8.5, "carbs": 4.5, "fat": 8.0, "fiber": 2.0},
        "healthy_alternative": "Caesar salad with olive oil & lemon vinaigrette, omitting toasted croutons.",
        "tags": ["Keto Friendly", "Low Carb", "High Protein"]
    },
    "spaghetti_bolognese": {
        "display_name": "Spaghetti Bolognese",
        "cuisine": "Italian",
        "category": "Main Course",
        "default_portion_g": 350,
        "per_100g": {"calories": 145.0, "protein": 7.5, "carbs": 18.0, "fat": 4.8, "fiber": 1.8},
        "healthy_alternative": "Zucchini noodles or whole-wheat pasta with lean beef/lentil bolognese sauce.",
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
    "hot_dog": {
        "display_name": "Hot Dog with Bun",
        "cuisine": "American",
        "category": "Fast Food",
        "default_portion_g": 140,
        "per_100g": {"calories": 290.0, "protein": 10.2, "carbs": 22.0, "fat": 18.0, "fiber": 1.2},
        "healthy_alternative": "Lean chicken sausage on a whole-wheat bun with mustard.",
        "tags": ["Fast Food", "High Sodium"]
    },
    "tacos": {
        "display_name": "Mexican Tacos",
        "cuisine": "Mexican",
        "category": "Main Course",
        "default_portion_g": 200,
        "per_100g": {"calories": 210.0, "protein": 10.0, "carbs": 20.0, "fat": 10.0, "fiber": 2.8},
        "healthy_alternative": "Corn tortilla tacos with grilled fish or shredded chicken breast and pico de gallo.",
        "tags": ["Balanced", "High Protein"]
    },
    "lasagna": {
        "display_name": "Baked Cheesy Lasagna",
        "cuisine": "Italian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 175.0, "protein": 9.5, "carbs": 16.5, "fat": 8.5, "fiber": 1.5},
        "healthy_alternative": "Layered eggplant and zucchini lasagna with ricotta and lean turkey mince.",
        "tags": ["Comfort Food", "Calcium Rich"]
    },
    "macaroni_cheese": {
        "display_name": "Macaroni and Cheese",
        "cuisine": "American",
        "category": "Comfort Food",
        "default_portion_g": 250,
        "per_100g": {"calories": 215.0, "protein": 7.5, "carbs": 26.0, "fat": 9.2, "fiber": 1.4},
        "healthy_alternative": "Butternut squash & nutritional yeast blended sauce over chickpea macaroni.",
        "tags": ["Comfort Food", "High Carb"]
    },
    "fried_chicken": {
        "display_name": "Crispy Fried Chicken",
        "cuisine": "American",
        "category": "Fast Food",
        "default_portion_g": 200,
        "per_100g": {"calories": 280.0, "protein": 22.0, "carbs": 12.0, "fat": 16.5, "fiber": 0.8},
        "healthy_alternative": "Oven-baked cornflake or panko-crusted chicken tenders.",
        "tags": ["High Protein", "High Fat"]
    },
    "pancakes": {
        "display_name": "Fluffy Buttermilk Pancakes",
        "cuisine": "American",
        "category": "Breakfast / Dessert",
        "default_portion_g": 180,
        "per_100g": {"calories": 227.0, "protein": 6.4, "carbs": 38.0, "fat": 5.5, "fiber": 1.3},
        "healthy_alternative": "Oat banana protein pancakes topped with fresh berries and Greek yogurt.",
        "tags": ["High Carb", "Breakfast"]
    },
    "waffles": {
        "display_name": "Belgian Waffles with Syrup",
        "cuisine": "Belgian / American",
        "category": "Breakfast / Dessert",
        "default_portion_g": 180,
        "per_100g": {"calories": 290.0, "protein": 7.0, "carbs": 42.0, "fat": 10.5, "fiber": 1.5},
        "healthy_alternative": "Whole-grain spelt waffles topped with almond butter and cinnamon.",
        "tags": ["High Carb", "High Energy"]
    },
    "croissant": {
        "display_name": "Butter Croissant",
        "cuisine": "French",
        "category": "Bakery / Breakfast",
        "default_portion_g": 80,
        "per_100g": {"calories": 406.0, "protein": 8.2, "carbs": 45.8, "fat": 21.0, "fiber": 2.6},
        "healthy_alternative": "Whole wheat sourdough toast with light whipped ricotta cheese.",
        "tags": ["High Fat", "Bakery"]
    },
    "chocolate_cake": {
        "display_name": "Chocolate Fudge Cake",
        "cuisine": "Western",
        "category": "Dessert",
        "default_portion_g": 120,
        "per_100g": {"calories": 371.0, "protein": 4.5, "carbs": 53.0, "fat": 16.5, "fiber": 3.0},
        "healthy_alternative": "Flourless dark chocolate avocado mousse cake with 85% cocoa.",
        "tags": ["Dessert", "High Sugar"]
    },
    "ice_cream": {
        "display_name": "Ice Cream Scoop / Sundae",
        "cuisine": "Global",
        "category": "Dessert",
        "default_portion_g": 120,
        "per_100g": {"calories": 207.0, "protein": 3.5, "carbs": 24.0, "fat": 11.0, "fiber": 0.7},
        "healthy_alternative": "Blended frozen banana nice-cream with a splash of almond milk.",
        "tags": ["Dessert", "Calcium Rich"]
    },

    # ==================== Asian Specialties ====================
    "momos": {
        "display_name": "Steamed Momos / Dumplings",
        "cuisine": "Tibetan / Indo-Chinese",
        "category": "Snack",
        "default_portion_g": 180,
        "per_100g": {"calories": 140.0, "protein": 7.8, "carbs": 21.0, "fat": 3.0, "fiber": 1.5},
        "healthy_alternative": "Steamed whole-wheat vegetable or chicken momos with garlic-tomato dip.",
        "tags": ["Steamed", "Low Fat"]
    },
    "fried_rice": {
        "display_name": "Chicken Fried Rice",
        "cuisine": "Asian",
        "category": "Main Course",
        "default_portion_g": 300,
        "per_100g": {"calories": 163.0, "protein": 6.2, "carbs": 24.5, "fat": 4.5, "fiber": 1.5},
        "healthy_alternative": "Cauliflower-quinoa fried rice with edamame and scrambled egg whites.",
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
        "display_name": "Steamed Dim Sum Dumplings",
        "cuisine": "Asian",
        "category": "Snack",
        "default_portion_g": 180,
        "per_100g": {"calories": 135.0, "protein": 8.0, "carbs": 18.0, "fat": 3.2, "fiber": 1.2},
        "healthy_alternative": "Steamed crystal vegetable dumplings with light soy dipping sauce.",
        "tags": ["Steamed", "Low Fat"]
    },
    "ramen": {
        "display_name": "Japanese Ramen Bowl",
        "cuisine": "Japanese",
        "category": "Soup / Noodles",
        "default_portion_g": 450,
        "per_100g": {"calories": 110.0, "protein": 4.5, "carbs": 15.0, "fat": 3.8, "fiber": 1.0},
        "healthy_alternative": "Soba or shirataki noodles in clear miso broth with bok choy and soft boiled egg.",
        "tags": ["Comfort Food"]
    },
    "pad_thai": {
        "display_name": "Pad Thai Noodles",
        "cuisine": "Thai",
        "category": "Noodles",
        "default_portion_g": 320,
        "per_100g": {"calories": 178.0, "protein": 7.0, "carbs": 26.0, "fat": 5.0, "fiber": 2.0},
        "healthy_alternative": "Zucchini/glass noodle stir-fry with tofu, bean sprouts, and lime.",
        "tags": ["High Energy"]
    },
    "spring_rolls": {
        "display_name": "Crispy Spring Rolls",
        "cuisine": "Asian",
        "category": "Snack",
        "default_portion_g": 150,
        "per_100g": {"calories": 210.0, "protein": 4.5, "carbs": 28.0, "fat": 9.2, "fiber": 2.0},
        "healthy_alternative": "Fresh Vietnamese rice paper summer rolls packed with herbs and shrimp.",
        "tags": ["Snack", "High Carb"]
    },

    # ==================== Healthy & Breakfast Essentials ====================
    "oatmeal": {
        "display_name": "Oatmeal Porridge Bowl",
        "cuisine": "Global",
        "category": "Breakfast / Healthy",
        "default_portion_g": 250,
        "per_100g": {"calories": 95.0, "protein": 3.2, "carbs": 16.5, "fat": 1.8, "fiber": 3.2},
        "healthy_alternative": "Already an exceptional superfood! Top with chia seeds and cinnamon.",
        "tags": ["High Fiber", "Heart Healthy", "Low Glycemic"]
    },
    "boiled_eggs": {
        "display_name": "Boiled Eggs (2 Large)",
        "cuisine": "Global",
        "category": "Breakfast / Protein",
        "default_portion_g": 100,
        "per_100g": {"calories": 155.0, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "fiber": 0.0},
        "healthy_alternative": "Hard-boiled egg whites with black pepper and olive oil drizzle.",
        "tags": ["High Protein", "Keto Friendly", "Choline Rich"]
    },
    "scrambled_eggs": {
        "display_name": "Fluffy Scrambled Eggs",
        "cuisine": "Global",
        "category": "Breakfast / Protein",
        "default_portion_g": 150,
        "per_100g": {"calories": 149.0, "protein": 10.0, "carbs": 1.6, "fat": 11.0, "fiber": 0.0},
        "healthy_alternative": "Scrambled egg whites with baby spinach and mushrooms.",
        "tags": ["High Protein", "Keto Friendly"]
    },
    "avocado_toast": {
        "display_name": "Avocado Toast on Sourdough",
        "cuisine": "Continental",
        "category": "Breakfast / Healthy",
        "default_portion_g": 150,
        "per_100g": {"calories": 190.0, "protein": 5.0, "carbs": 18.0, "fat": 11.0, "fiber": 4.5},
        "healthy_alternative": "Whole-grain rye toast with mashed avocado, chili flakes, and hemp seeds.",
        "tags": ["Heart Healthy", "Monounsaturated Fats", "High Fiber"]
    },
    "greek_yogurt": {
        "display_name": "Greek Yogurt Bowl",
        "cuisine": "Mediterranean",
        "category": "Dairy / Breakfast",
        "default_portion_g": 200,
        "per_100g": {"calories": 73.0, "protein": 10.0, "carbs": 3.6, "fat": 1.9, "fiber": 0.0},
        "healthy_alternative": "Zero-fat plain Greek yogurt with crushed walnuts and blueberries.",
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
    "grilled_salmon": {
        "display_name": "Grilled Salmon Fillet",
        "cuisine": "Continental",
        "category": "Lean Protein",
        "default_portion_g": 200,
        "per_100g": {"calories": 208.0, "protein": 22.0, "carbs": 0.0, "fat": 13.0, "fiber": 0.0},
        "healthy_alternative": "Already an optimal superfood! High in DHA & EPA omega-3 fatty acids.",
        "tags": ["High Protein", "Omega-3 Rich", "Keto Friendly"]
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

# =========================================================================
# Multi-Label Toppings & Add-ons Knowledge Base
# Accurate individual nutritional values per typical serving
# =========================================================================
TOPPINGS_DATABASE = {
    "peanut_butter": {
        "display_name": "Peanut Butter",
        "unit": "1 tbsp (20g)",
        "grams": 20.0,
        "calories": 118.0,
        "protein": 5.0,
        "carbs": 4.0,
        "fat": 10.0,
        "fiber": 1.5,
        "prompt": "creamy peanut butter or nut butter spread",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dairy / Breakfast", "Bakery / Breakfast", "Snack"]
    },
    "banana_slices": {
        "display_name": "Sliced Bananas",
        "unit": "1/2 banana (50g)",
        "grams": 50.0,
        "calories": 45.0,
        "protein": 0.6,
        "carbs": 11.5,
        "fat": 0.2,
        "fiber": 1.3,
        "prompt": "fresh sliced bananas or banana rounds",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dairy / Breakfast", "Salad / Dessert", "Dessert", "Bakery / Breakfast"]
    },
    "almonds": {
        "display_name": "Crushed Almonds / Nuts",
        "unit": "handful (15g)",
        "grams": 15.0,
        "calories": 87.0,
        "protein": 3.2,
        "carbs": 3.2,
        "fat": 7.5,
        "fiber": 1.9,
        "prompt": "sliced almonds or crushed roasted nuts",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dairy / Breakfast", "Salad / Dessert", "Dessert", "Curry", "Main Course"]
    },
    "blueberries": {
        "display_name": "Fresh Blueberries",
        "unit": "scoop (30g)",
        "grams": 30.0,
        "calories": 17.0,
        "protein": 0.2,
        "carbs": 4.3,
        "fat": 0.1,
        "fiber": 0.7,
        "prompt": "fresh blueberries or dark berries",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dairy / Breakfast", "Salad / Dessert", "Dessert", "Bakery / Breakfast"]
    },
    "chia_seeds": {
        "display_name": "Chia / Flax Seeds",
        "unit": "1 tbsp (10g)",
        "grams": 10.0,
        "calories": 49.0,
        "protein": 1.7,
        "carbs": 4.2,
        "fat": 3.1,
        "fiber": 3.4,
        "prompt": "black chia seeds or flax seeds sprinkled",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dairy / Breakfast", "Salad / Dessert"]
    },
    "honey_syrup": {
        "display_name": "Honey / Maple Syrup",
        "unit": "1 tbsp (15g)",
        "grams": 15.0,
        "calories": 46.0,
        "protein": 0.0,
        "carbs": 12.4,
        "fat": 0.0,
        "fiber": 0.0,
        "prompt": "golden honey or maple syrup drizzle",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dairy / Breakfast", "Dessert", "Bakery / Breakfast"]
    },
    "chocolate_chips": {
        "display_name": "Chocolate Chips / Drizzle",
        "unit": "1 tbsp (15g)",
        "grams": 15.0,
        "calories": 75.0,
        "protein": 0.8,
        "carbs": 9.5,
        "fat": 4.2,
        "fiber": 0.6,
        "prompt": "dark chocolate chips or chocolate drizzle",
        "compatible_categories": ["Breakfast", "Breakfast / Healthy", "Dessert", "Bakery / Breakfast"]
    },
    "cheese": {
        "display_name": "Grated / Melted Cheese",
        "unit": "slice (25g)",
        "grams": 25.0,
        "calories": 100.0,
        "protein": 6.5,
        "carbs": 0.8,
        "fat": 8.0,
        "fiber": 0.0,
        "prompt": "melted cheese or shredded cheddar cheese",
        "compatible_categories": ["Fast Food", "Main Course", "Snack", "Breakfast", "Breads", "Noodles"]
    },
    "fried_egg": {
        "display_name": "Fried / Poached Egg",
        "unit": "1 egg (50g)",
        "grams": 50.0,
        "calories": 90.0,
        "protein": 6.3,
        "carbs": 0.4,
        "fat": 7.0,
        "fiber": 0.0,
        "prompt": "fried egg sunny side up on top",
        "compatible_categories": ["Fast Food", "Main Course", "Breakfast", "Breakfast / Protein", "Noodles", "Snack"]
    },
    "paneer_cubes": {
        "display_name": "Paneer / Tofu Cubes",
        "unit": "portion (40g)",
        "grams": 40.0,
        "calories": 105.0,
        "protein": 7.2,
        "carbs": 1.4,
        "fat": 8.0,
        "fiber": 0.0,
        "prompt": "white paneer or tofu cubes",
        "compatible_categories": ["Curry", "Main Course", "Salad / Healthy", "Snack", "Lean Protein"]
    },
    "spicy_chutney": {
        "display_name": "Spicy Chutney / Garlic Powder",
        "unit": "1 tbsp (20g)",
        "grams": 20.0,
        "calories": 18.0,
        "protein": 0.5,
        "carbs": 2.5,
        "fat": 0.7,
        "fiber": 0.9,
        "prompt": "spicy green chutney or red garlic chutney powder",
        "compatible_categories": ["Snack", "Street Food", "Breakfast / Snack", "Breakfast", "Main Course", "Breads"]
    },
    "whipped_cream": {
        "display_name": "Whipped Cream",
        "unit": "dollop (20g)",
        "grams": 20.0,
        "calories": 52.0,
        "protein": 0.4,
        "carbs": 2.5,
        "fat": 4.5,
        "fiber": 0.0,
        "prompt": "white whipped cream dollop",
        "compatible_categories": ["Dessert", "Breakfast / Dessert", "Beverage", "Bakery / Breakfast"]
    }
}


TOPPINGS_CLASSES = sorted(list(TOPPINGS_DATABASE.keys()))

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
        food = FOOD_DATABASE.get("chicken_biryani", list(FOOD_DATABASE.values())[0])

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


def calculate_composite_nutrition(food_key: str, portion_grams: float, selected_toppings: list = None):
    """
    Calculates combined nutrition for base meal plus all detected or selected toppings.
    Returns:
        base_nutri: Dict of base meal nutrition
        toppings_list: List of dicts describing each added topping
        total_nutri: Aggregated composite nutrition (kcal, protein, carbs, fat, fiber)
    """
    base = get_nutrition_for_portion(food_key, portion_grams)
    if not selected_toppings:
        return base, [], base

    toppings_list = []
    tot_cal = base["calories"]
    tot_p = base["protein"]
    tot_c = base["carbs"]
    tot_f = base["fat"]
    tot_fib = base["fiber"]
    tot_g = base["portion_grams"]

    for top_key in selected_toppings:
        top_info = TOPPINGS_DATABASE.get(top_key)
        if top_info:
            toppings_list.append(top_info)
            tot_cal += top_info["calories"]
            tot_p += top_info["protein"]
            tot_c += top_info["carbs"]
            tot_f += top_info["fat"]
            tot_fib += top_info["fiber"]
            tot_g += top_info["grams"]

    total_nutri = {
        "food_key": food_key,
        "display_name": base["display_name"],
        "category": base["category"],
        "cuisine": base["cuisine"],
        "portion_grams": round(tot_g, 1),
        "calories": round(tot_cal, 1),
        "protein": round(tot_p, 1),
        "carbs": round(tot_c, 1),
        "fat": round(tot_f, 1),
        "fiber": round(tot_fib, 1),
        "healthy_alternative": base["healthy_alternative"],
        "tags": base["tags"]
    }

    return base, toppings_list, total_nutri



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

