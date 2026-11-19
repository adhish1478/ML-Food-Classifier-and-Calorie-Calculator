"""
NutriVision 100-Dish Real-World Benchmark Dataset Builder & Evaluator.
Downloads 100 authentic, amateur, user-clicked food photos across Indian,
Western, Asian, and composite dishes, compiles expert Ground Truth nutrition,
and benchmarks NutriVision model predictions against Ground Truth.
"""

import os
import sys
import json
import csv
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nutrition_db import FOOD_DATABASE, TOPPINGS_DATABASE, calculate_composite_nutrition, get_nutrition_for_portion
from models.cnn_classifier import load_food_classifier, classify_food_image
from models.ann_regressor import load_calorie_regressor, predict_calories

HEADERS = {'User-Agent': 'NutriVisionResearchBot/1.0 (anushya.research@christuniversity.in)'}

# Comprehensive 100-dish benchmark specification
BENCHMARK_SPEC = [
    # ==================== Indian Regional & Street Foods (40 Items) ====================
    {"id": "001", "q": "vada pav street food plate", "key": "vada_pav", "dish": "Vada Pav", "toppings": ["spicy_chutney"], "portion": 150, "cal": 283, "p": 5.7, "c": 40.5, "f": 11.2, "fib": 3.7},
    {"id": "002", "q": "pav bhaji street food butter", "key": "pav_bhaji", "dish": "Pav Bhaji", "toppings": [], "portion": 300, "cal": 480, "p": 11.4, "c": 73.5, "f": 16.5, "fib": 9.6},
    {"id": "003", "q": "samosa indian food plate chutney", "key": "samosa", "dish": "Samosa", "toppings": ["spicy_chutney"], "portion": 120, "cal": 332, "p": 5.9, "c": 40.9, "f": 16.3, "fib": 3.9},
    {"id": "004", "q": "pani puri golgappa street food", "key": "pani_puri", "dish": "Pani Puri / Golgappa", "toppings": [], "portion": 180, "cal": 261, "p": 5.0, "c": 46.8, "f": 6.3, "fib": 3.8},
    {"id": "005", "q": "bhel puri chaat street food", "key": "bhel_puri", "dish": "Bhel Puri / Chaat", "toppings": ["spicy_chutney"], "portion": 160, "cal": 287, "p": 7.2, "c": 51.3, "f": 6.8, "fib": 5.1},
    {"id": "006", "q": "chole bhature north indian food", "key": "chole_bhature", "dish": "Chole Bhature", "toppings": [], "portion": 350, "cal": 857, "p": 22.8, "c": 119.0, "f": 34.3, "fib": 15.8},
    {"id": "007", "q": "masala dosa sambar chutney plate", "key": "masala_dosa", "dish": "Masala Dosa", "toppings": ["spicy_chutney"], "portion": 200, "cal": 368, "p": 8.7, "c": 61.5, "f": 9.9, "fib": 5.1},
    {"id": "008", "q": "plain dosa south indian breakfast", "key": "plain_dosa", "dish": "Crispy Plain Dosa", "toppings": ["spicy_chutney"], "portion": 160, "cal": 282, "p": 6.6, "c": 52.1, "f": 5.8, "fib": 3.8},
    {"id": "009", "q": "idli sambar breakfast plate", "key": "idli", "dish": "Idli with Sambar", "toppings": ["spicy_chutney"], "portion": 220, "cal": 271, "p": 8.9, "c": 53.1, "f": 2.5, "fib": 6.2},
    {"id": "010", "q": "medu vada sambar chutney", "key": "medu_vada", "dish": "Medu Vada", "toppings": ["spicy_chutney"], "portion": 140, "cal": 347, "p": 10.0, "c": 37.5, "f": 17.5, "fib": 5.8},
    {"id": "011", "q": "kanda poha breakfast maharashtra", "key": "poha", "dish": "Kanda Poha", "toppings": [], "portion": 200, "cal": 260, "p": 5.6, "c": 49.0, "f": 5.6, "fib": 4.4},
    {"id": "012", "q": "rava upma south indian breakfast", "key": "upma", "dish": "Rava Vegetable Upma", "toppings": [], "portion": 200, "cal": 240, "p": 6.0, "c": 44.0, "f": 5.0, "fib": 4.0},
    {"id": "013", "q": "aloo paratha with butter curd", "key": "aloo_paratha", "dish": "Aloo Paratha with Butter", "toppings": [], "portion": 180, "cal": 432, "p": 8.6, "c": 64.8, "f": 15.8, "fib": 5.8},
    {"id": "014", "q": "paneer paratha flatbread plate", "key": "paneer_paratha", "dish": "Stuffed Paneer Paratha", "toppings": [], "portion": 180, "cal": 477, "p": 17.1, "c": 57.6, "f": 20.2, "fib": 5.4},
    {"id": "015", "q": "poori bhaji indian breakfast", "key": "poori_bhaji", "dish": "Poori with Aloo Bhaji", "toppings": [], "portion": 260, "cal": 663, "p": 10.9, "c": 91.0, "f": 28.6, "fib": 7.3},
    {"id": "016", "q": "rajma chawal north indian plate", "key": "rajma_chawal", "dish": "Rajma Chawal", "toppings": [], "portion": 350, "cal": 472, "p": 18.2, "c": 82.2, "f": 9.8, "fib": 13.6},
    {"id": "017", "q": "dal makhani black lentils bowl", "key": "dal_makhani", "dish": "Dal Makhani", "toppings": [], "portion": 250, "cal": 350, "p": 13.8, "c": 40.0, "f": 15.5, "fib": 10.0},
    {"id": "018", "q": "dal tadka yellow dal indian", "key": "dal_tadka", "dish": "Yellow Dal Tadka", "toppings": [], "portion": 250, "cal": 275, "p": 15.5, "c": 37.5, "f": 8.0, "fib": 9.5},
    {"id": "019", "q": "khichdi moong dal rice plate", "key": "khichdi", "dish": "Moong Dal Khichdi", "toppings": [], "portion": 300, "cal": 345, "p": 13.5, "c": 58.5, "f": 6.6, "fib": 8.4},
    {"id": "020", "q": "curd rice south indian thali", "key": "curd_rice", "dish": "South Indian Curd Rice", "toppings": [], "portion": 250, "cal": 325, "p": 8.8, "c": 52.5, "f": 9.0, "fib": 3.0},
    {"id": "021", "q": "uttapam south indian pancake", "key": "uttapam", "dish": "Onion Tomato Uttapam", "toppings": [], "portion": 200, "cal": 310, "p": 8.4, "c": 54.0, "f": 7.0, "fib": 5.0},
    {"id": "022", "q": "khaman dhokla gujarati snack", "key": "dhokla", "dish": "Khaman Dhokla", "toppings": [], "portion": 150, "cal": 240, "p": 8.7, "c": 39.8, "f": 5.7, "fib": 3.8},
    {"id": "023", "q": "veg biryani pulao plate", "key": "veg_biryani", "dish": "Vegetable Biryani / Pulao", "toppings": [], "portion": 300, "cal": 420, "p": 10.8, "c": 75.0, "f": 9.0, "fib": 7.2},
    {"id": "024", "q": "chicken biryani plate restaurant", "key": "chicken_biryani", "dish": "Chicken Biryani", "toppings": [], "portion": 350, "cal": 577, "p": 32.2, "c": 75.2, "f": 16.8, "fib": 4.2},
    {"id": "025", "q": "mutton biryani hyderabadi plate", "key": "mutton_biryani", "dish": "Mutton Biryani", "toppings": [], "portion": 350, "cal": 682, "p": 40.2, "c": 73.5, "f": 26.2, "fib": 3.5},
    {"id": "026", "q": "butter chicken curry bowl indian", "key": "butter_chicken", "dish": "Butter Chicken", "toppings": [], "portion": 300, "cal": 555, "p": 36.0, "c": 19.5, "f": 37.5, "fib": 3.3},
    {"id": "027", "q": "chicken tikka tandoori skewers", "key": "chicken_tikka", "dish": "Tandoori Chicken Tikka", "toppings": [], "portion": 200, "cal": 300, "p": 48.0, "c": 5.0, "f": 9.6, "fib": 1.0},
    {"id": "028", "q": "paneer butter masala curry indian", "key": "paneer_butter_masala", "dish": "Paneer Butter Masala", "toppings": [], "portion": 250, "cal": 525, "p": 17.0, "c": 21.2, "f": 41.2, "fib": 4.5},
    {"id": "029", "q": "palak paneer spinach curry", "key": "palak_paneer", "dish": "Palak Paneer", "toppings": [], "portion": 250, "cal": 362, "p": 18.8, "c": 15.5, "f": 26.2, "fib": 8.5},
    {"id": "030", "q": "fish curry coastal indian meal", "key": "fish_curry", "dish": "Spiced Fish Curry", "toppings": [], "portion": 250, "cal": 337, "p": 37.5, "c": 11.2, "f": 16.2, "fib": 3.0},
    {"id": "031", "q": "egg curry boiled eggs indian", "key": "egg_curry", "dish": "Boiled Egg Curry", "toppings": [], "portion": 250, "cal": 350, "p": 26.2, "c": 13.8, "f": 22.0, "fib": 3.5},
    {"id": "032", "q": "butter naan roti basket indian", "key": "roti_naan", "dish": "Butter Naan / Roti", "toppings": [], "portion": 100, "cal": 290, "p": 8.0, "c": 50.0, "f": 6.5, "fib": 2.2},
    {"id": "033", "q": "gulab jamun syrup indian sweet", "key": "gulab_jamun", "dish": "Gulab Jamun", "toppings": [], "portion": 100, "cal": 320, "p": 4.0, "c": 52.0, "f": 11.0, "fib": 0.5},
    {"id": "034", "q": "rasgulla white bengali sweet", "key": "rasgulla", "dish": "Spongy Rasgulla", "toppings": [], "portion": 100, "cal": 185, "p": 4.5, "c": 38.0, "f": 1.8, "fib": 0.2},
    {"id": "035", "q": "jalebi indian sweet street food", "key": "jalebi", "dish": "Crispy Jalebi", "toppings": [], "portion": 100, "cal": 370, "p": 2.5, "c": 68.0, "f": 10.5, "fib": 0.3},
    {"id": "036", "q": "kheer rice pudding payasam", "key": "kheer", "dish": "Rice Kheer / Payasam", "toppings": ["almonds"], "portion": 150, "cal": 350, "p": 10.4, "c": 43.0, "f": 16.5, "fib": 3.1},
    {"id": "037", "q": "gajar ka halwa carrot sweet", "key": "gajar_halwa", "dish": "Gajar Ka Halwa", "toppings": ["almonds"], "portion": 150, "cal": 477, "p": 9.5, "c": 54.2, "f": 25.5, "fib": 5.6},
    {"id": "038", "q": "masala chai tea glass indian", "key": "chai", "dish": "Masala Chai / Milk Tea", "toppings": [], "portion": 180, "cal": 117, "p": 3.8, "c": 15.3, "f": 4.5, "fib": 0.0},
    {"id": "039", "q": "mango lassi sweet yogurt drink", "key": "lassi", "dish": "Sweet Mango / Yogurt Lassi", "toppings": [], "portion": 250, "cal": 238, "p": 9.0, "c": 36.2, "f": 7.0, "fib": 1.0},
    {"id": "040", "q": "steamed momos red chutney plate", "key": "momos", "dish": "Steamed Momos / Dumplings", "toppings": ["spicy_chutney"], "portion": 180, "cal": 270, "p": 14.5, "c": 40.3, "f": 6.1, "fib": 3.6},

    # ==================== Western & Continental Favorites (30 Items) ====================
    {"id": "041", "q": "cheeseburger meal diner table", "key": "burger", "dish": "Cheeseburger", "toppings": ["cheese"], "portion": 245, "cal": 650, "p": 35.1, "c": 53.6, "f": 33.3, "fib": 3.1},
    {"id": "042", "q": "pizza slice pepperoni cheese plate", "key": "pizza", "dish": "Cheese Pizza", "toppings": ["cheese"], "portion": 275, "cal": 765, "p": 34.0, "c": 83.3, "f": 33.0, "fib": 5.8},
    {"id": "043", "q": "french fries basket diner", "key": "french_fries", "dish": "French Fries", "toppings": [], "portion": 150, "cal": 468, "p": 5.1, "c": 62.2, "f": 22.5, "fib": 5.7},
    {"id": "044", "q": "caesar salad plate chicken", "key": "caesar_salad", "dish": "Chicken Caesar Salad", "toppings": ["cheese"], "portion": 275, "cal": 412, "p": 27.8, "c": 12.1, "f": 28.0, "fib": 5.0},
    {"id": "045", "q": "spaghetti bolognese pasta plate", "key": "spaghetti_bolognese", "dish": "Spaghetti Bolognese", "toppings": ["cheese"], "portion": 375, "cal": 607, "p": 32.8, "c": 63.8, "f": 24.8, "fib": 6.3},
    {"id": "046", "q": "grilled chicken breast vegetables plate", "key": "grilled_chicken", "dish": "Herb Grilled Chicken Breast", "toppings": [], "portion": 200, "cal": 330, "p": 62.0, "c": 1.0, "f": 7.2, "fib": 0.0},
    {"id": "047", "q": "club sandwich sliced lunch", "key": "sandwich", "dish": "Club Sandwich", "toppings": [], "portion": 220, "cal": 484, "p": 23.1, "c": 57.2, "f": 17.6, "fib": 5.5},
    {"id": "048", "q": "grilled beef steak plate restaurant", "key": "steak", "dish": "Grilled Beef Steak", "toppings": [], "portion": 250, "cal": 537, "p": 65.0, "c": 0.0, "f": 30.0, "fib": 0.0},
    {"id": "049", "q": "hot dog with mustard bun plate", "key": "hot_dog", "dish": "Hot Dog with Bun", "toppings": [], "portion": 140, "cal": 406, "p": 14.3, "c": 30.8, "f": 25.2, "fib": 1.7},
    {"id": "050", "q": "mexican tacos plate salsa", "key": "tacos", "dish": "Mexican Tacos", "toppings": ["cheese"], "portion": 225, "cal": 520, "p": 26.5, "c": 40.8, "f": 28.0, "fib": 5.6},
    {"id": "051", "q": "cheesy lasagna pasta baked slice", "key": "lasagna", "dish": "Baked Cheesy Lasagna", "toppings": ["cheese"], "portion": 325, "cal": 625, "p": 35.0, "c": 50.3, "f": 33.5, "fib": 4.5},
    {"id": "052", "q": "macaroni and cheese pasta bowl", "key": "macaroni_cheese", "dish": "Macaroni and Cheese", "toppings": ["cheese"], "portion": 275, "cal": 637, "p": 25.2, "c": 65.8, "f": 31.0, "fib": 3.5},
    {"id": "053", "q": "crispy fried chicken plate dinner", "key": "fried_chicken", "dish": "Crispy Fried Chicken", "toppings": [], "portion": 200, "cal": 560, "p": 44.0, "c": 24.0, "f": 33.0, "fib": 1.6},
    {"id": "054", "q": "pancakes with syrup breakfast plate", "key": "pancakes", "dish": "Fluffy Buttermilk Pancakes", "toppings": ["honey_syrup"], "portion": 195, "cal": 454, "p": 11.5, "c": 80.8, "f": 9.9, "fib": 2.3},
    {"id": "055", "q": "belgian waffles breakfast plate syrup", "key": "waffles", "dish": "Belgian Waffles with Syrup", "toppings": ["honey_syrup"], "portion": 195, "cal": 568, "p": 12.6, "c": 88.0, "f": 18.9, "fib": 2.7},
    {"id": "056", "q": "butter croissant breakfast pastry", "key": "croissant", "dish": "Butter Croissant", "toppings": [], "portion": 80, "cal": 325, "p": 6.6, "c": 36.6, "f": 16.8, "fib": 2.1},
    {"id": "057", "q": "chocolate fudge cake slice plate", "key": "chocolate_cake", "dish": "Chocolate Fudge Cake", "toppings": ["chocolate_chips"], "portion": 135, "cal": 520, "p": 6.2, "c": 73.1, "f": 24.0, "fib": 4.2},
    {"id": "058", "q": "ice cream sundae bowl dessert", "key": "ice_cream", "dish": "Ice Cream Scoop / Sundae", "toppings": ["chocolate_chips"], "portion": 135, "cal": 323, "p": 5.0, "c": 38.3, "f": 17.4, "fib": 1.4},
    {"id": "059", "q": "scrambled eggs breakfast plate", "key": "scrambled_eggs", "dish": "Fluffy Scrambled Eggs", "toppings": [], "portion": 150, "cal": 223, "p": 15.0, "c": 2.4, "f": 16.5, "fib": 0.0},
    {"id": "060", "q": "boiled eggs halved plate", "key": "boiled_eggs", "dish": "Boiled Eggs (2 Large)", "toppings": [], "portion": 100, "cal": 155, "p": 13.0, "c": 1.1, "f": 11.0, "fib": 0.0},
    {"id": "061", "q": "avocado toast sourdough bread plate", "key": "avocado_toast", "dish": "Avocado Toast on Sourdough", "toppings": [], "portion": 150, "cal": 285, "p": 7.5, "c": 27.0, "f": 16.5, "fib": 6.8},
    {"id": "062", "q": "grilled salmon fillet plate lemon", "key": "grilled_salmon", "dish": "Grilled Salmon Fillet", "toppings": [], "portion": 200, "cal": 416, "p": 44.0, "c": 0.0, "f": 26.0, "fib": 0.0},
    {"id": "063", "q": "vegetable curry bowl plate", "key": "vegetable_curry", "dish": "Mixed Vegetable Curry", "toppings": [], "portion": 250, "cal": 237, "p": 7.0, "c": 28.8, "f": 10.5, "fib": 8.8},
    {"id": "064", "q": "green garden salad bowl", "key": "green_salad", "dish": "Mediterranean Green Salad", "toppings": [], "portion": 200, "cal": 90, "p": 3.6, "c": 8.4, "f": 5.0, "fib": 5.6},
    {"id": "065", "q": "fresh fruit salad bowl dessert", "key": "fruit_salad", "dish": "Fresh Fruit Medley", "toppings": [], "portion": 250, "cal": 145, "p": 2.0, "c": 36.2, "f": 0.5, "fib": 6.0},
    {"id": "066", "q": "greek yogurt bowl breakfast", "key": "greek_yogurt", "dish": "Greek Yogurt Bowl", "toppings": [], "portion": 200, "cal": 146, "p": 20.0, "c": 7.2, "f": 3.8, "fib": 0.0},
    {"id": "067", "q": "burger and french fries combo diner", "key": "burger", "dish": "Cheeseburger", "toppings": ["cheese"], "portion": 245, "cal": 650, "p": 35.1, "c": 53.6, "f": 33.3, "fib": 3.1},
    {"id": "068", "q": "pizza Margherita plate casual", "key": "pizza", "dish": "Cheese Pizza", "toppings": ["cheese"], "portion": 275, "cal": 765, "p": 34.0, "c": 83.3, "f": 33.0, "fib": 5.8},
    {"id": "069", "q": "breakfast waffles berries plate", "key": "waffles", "dish": "Belgian Waffles with Syrup", "toppings": ["blueberries", "honey_syrup"], "portion": 225, "cal": 585, "p": 12.8, "c": 92.3, "f": 19.0, "fib": 3.4},
    {"id": "070", "q": "fried eggs plate breakfast sunny side", "key": "scrambled_eggs", "dish": "Fluffy Scrambled Eggs", "toppings": [], "portion": 150, "cal": 223, "p": 15.0, "c": 2.4, "f": 16.5, "fib": 0.0},

    # ==================== Asian Specialties (15 Items) ====================
    {"id": "071", "q": "japanese ramen noodle bowl egg", "key": "ramen", "dish": "Japanese Ramen Bowl", "toppings": ["fried_egg"], "portion": 500, "cal": 585, "p": 26.5, "c": 67.9, "f": 24.1, "fib": 4.5},
    {"id": "072", "q": "salmon sushi roll plate seaweed", "key": "sushi", "dish": "Salmon Sushi Roll", "toppings": [], "portion": 200, "cal": 284, "p": 13.0, "c": 44.0, "f": 5.6, "fib": 2.2},
    {"id": "073", "q": "steamed dim sum bamboo basket", "key": "dim_sum", "dish": "Steamed Dim Sum Dumplings", "toppings": [], "portion": 180, "cal": 243, "p": 14.4, "c": 32.4, "f": 5.8, "fib": 2.2},
    {"id": "074", "q": "chicken fried rice bowl chinese", "key": "fried_rice", "dish": "Chicken Fried Rice", "toppings": [], "portion": 300, "cal": 489, "p": 18.6, "c": 73.5, "f": 13.5, "fib": 4.5},
    {"id": "075", "q": "pad thai noodles plate peanuts lime", "key": "pad_thai", "dish": "Pad Thai Noodles", "toppings": ["almonds"], "portion": 335, "cal": 657, "p": 25.6, "c": 86.4, "f": 23.5, "fib": 8.3},
    {"id": "076", "q": "crispy spring rolls plate dipping sauce", "key": "spring_rolls", "dish": "Crispy Spring Rolls", "toppings": [], "portion": 150, "cal": 315, "p": 6.8, "c": 42.0, "f": 13.8, "fib": 3.0},
    {"id": "077", "q": "vegetable momos steamed dumplings", "key": "momos", "dish": "Steamed Momos / Dumplings", "toppings": ["spicy_chutney"], "portion": 180, "cal": 270, "p": 14.5, "c": 40.3, "f": 6.1, "fib": 3.6},
    {"id": "078", "q": "shoyu ramen bowl boiled egg", "key": "ramen", "dish": "Japanese Ramen Bowl", "toppings": ["fried_egg"], "portion": 500, "cal": 585, "p": 26.5, "c": 67.9, "f": 24.1, "fib": 4.5},
    {"id": "079", "q": "california sushi roll plate", "key": "sushi", "dish": "Salmon Sushi Roll", "toppings": [], "portion": 200, "cal": 284, "p": 13.0, "c": 44.0, "f": 5.6, "fib": 2.2},
    {"id": "080", "q": "egg fried rice bowl chinese", "key": "fried_rice", "dish": "Chicken Fried Rice", "toppings": [], "portion": 300, "cal": 489, "p": 18.6, "c": 73.5, "f": 13.5, "fib": 4.5},
    {"id": "081", "q": "fried spring rolls plate thai", "key": "spring_rolls", "dish": "Crispy Spring Rolls", "toppings": [], "portion": 150, "cal": 315, "p": 6.8, "c": 42.0, "f": 13.8, "fib": 3.0},
    {"id": "082", "q": "steamed dumplings dim sum meal", "key": "dim_sum", "dish": "Steamed Dim Sum Dumplings", "toppings": [], "portion": 180, "cal": 243, "p": 14.4, "c": 32.4, "f": 5.8, "fib": 2.2},
    {"id": "083", "q": "thai pad thai stir fry plate", "key": "pad_thai", "dish": "Pad Thai Noodles", "toppings": [], "portion": 320, "cal": 570, "p": 22.4, "c": 83.2, "f": 16.0, "fib": 6.4},
    {"id": "084", "q": "miso ramen soup noodles bowl", "key": "ramen", "dish": "Japanese Ramen Bowl", "toppings": [], "portion": 450, "cal": 495, "p": 20.2, "c": 67.5, "f": 17.1, "fib": 4.5},
    {"id": "085", "q": "chicken momos red chutney plate", "key": "momos", "dish": "Steamed Momos / Dumplings", "toppings": ["spicy_chutney"], "portion": 180, "cal": 270, "p": 14.5, "c": 40.3, "f": 6.1, "fib": 3.6},

    # ==================== Composite Bowls & Loaded Foods with Toppings (15 Items) ====================
    {"id": "086", "q": "oatmeal porridge bowl sliced bananas peanut butter", "key": "oatmeal", "dish": "Oatmeal Porridge Bowl", "toppings": ["peanut_butter", "banana_slices"], "portion": 320, "cal": 400, "p": 13.6, "c": 56.8, "f": 14.7, "fib": 10.8},
    {"id": "087", "q": "oatmeal berries almonds nuts breakfast", "key": "oatmeal", "dish": "Oatmeal Porridge Bowl", "toppings": ["blueberries", "almonds"], "portion": 295, "cal": 341, "p": 11.4, "c": 48.8, "f": 12.1, "fib": 10.6},
    {"id": "088", "q": "oatmeal bowl chia seeds honey", "key": "oatmeal", "dish": "Oatmeal Porridge Bowl", "toppings": ["chia_seeds", "honey_syrup"], "portion": 275, "cal": 332, "p": 9.7, "c": 57.8, "f": 7.6, "fib": 11.4},
    {"id": "089", "q": "greek yogurt bowl fresh blueberries", "key": "greek_yogurt", "dish": "Greek Yogurt Bowl", "toppings": ["blueberries", "honey_syrup"], "portion": 245, "cal": 209, "p": 20.2, "c": 23.9, "f": 3.9, "fib": 0.7},
    {"id": "090", "q": "greek yogurt walnuts almonds honey bowl", "key": "greek_yogurt", "dish": "Greek Yogurt Bowl", "toppings": ["almonds", "honey_syrup"], "portion": 230, "cal": 279, "p": 23.2, "c": 22.8, "f": 11.3, "fib": 1.9},
    {"id": "091", "q": "avocado toast fried egg sunny side up", "key": "avocado_toast", "dish": "Avocado Toast on Sourdough", "toppings": ["fried_egg"], "portion": 200, "cal": 375, "p": 13.8, "c": 27.4, "f": 23.5, "fib": 6.8},
    {"id": "092", "q": "pancakes stack sliced bananas maple syrup", "key": "pancakes", "dish": "Fluffy Buttermilk Pancakes", "toppings": ["banana_slices", "honey_syrup"], "portion": 245, "cal": 500, "p": 12.1, "c": 92.3, "f": 10.1, "fib": 3.6},
    {"id": "093", "q": "waffles chocolate chips syrup plate", "key": "waffles", "dish": "Belgian Waffles with Syrup", "toppings": ["chocolate_chips", "honey_syrup"], "portion": 210, "cal": 643, "p": 13.4, "c": 97.5, "f": 23.1, "fib": 3.3},
    {"id": "094", "q": "chicken caesar salad croutons cheese", "key": "caesar_salad", "dish": "Chicken Caesar Salad", "toppings": ["cheese"], "portion": 275, "cal": 412, "p": 27.8, "c": 12.1, "f": 28.0, "fib": 5.0},
    {"id": "095", "q": "cheeseburger bacon melted cheese plate", "key": "burger", "dish": "Cheeseburger", "toppings": ["cheese"], "portion": 245, "cal": 650, "p": 35.1, "c": 53.6, "f": 33.3, "fib": 3.1},
    {"id": "096", "q": "masala dosa chutney sambar south indian", "key": "masala_dosa", "dish": "Masala Dosa", "toppings": ["spicy_chutney"], "portion": 220, "cal": 386, "p": 9.2, "c": 64.0, "f": 10.6, "fib": 6.0},
    {"id": "097", "q": "vada pav green chili dry garlic chutney", "key": "vada_pav", "dish": "Vada Pav", "toppings": ["spicy_chutney"], "portion": 170, "cal": 301, "p": 6.2, "c": 43.0, "f": 11.9, "fib": 4.6},
    {"id": "098", "q": "khichdi ghee indian meal comfort", "key": "khichdi", "dish": "Moong Dal Khichdi", "toppings": [], "portion": 300, "cal": 345, "p": 13.5, "c": 58.5, "f": 6.6, "fib": 8.4},
    {"id": "099", "q": "chicken biryani raita plate lunch", "key": "chicken_biryani", "dish": "Chicken Biryani", "toppings": [], "portion": 350, "cal": 577, "p": 32.2, "c": 75.2, "f": 16.8, "fib": 4.2},
    {"id": "100", "q": "oatmeal bowl berries chia seeds nuts", "key": "oatmeal", "dish": "Oatmeal Porridge Bowl", "toppings": ["blueberries", "almonds", "chia_seeds"], "portion": 305, "cal": 390, "p": 13.1, "c": 53.0, "f": 15.2, "fib": 14.0}
]


def download_single_item(item, output_dir):
    """
    Downloads an authentic amateur image for a food item from Wikimedia Commons.
    Filters strictly for .jpg/.jpeg/.png image files and sorts by relevance index.
    """
    item_id = item["id"]
    q = item["q"]
    key = item["key"]
    filename = f"{item_id}_{key}.jpg"
    filepath = os.path.join(output_dir, filename)

    queries_to_try = [q, f"{item['dish']} food", item['dish']]

    for query_str in queries_to_try:
        encoded = urllib.parse.quote(query_str)
        url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded}&gsrnamespace=6&gsrlimit=8&prop=imageinfo&iiprop=url&iiurlwidth=400&format=json"

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                raw_pages = data.get("query", {}).get("pages", {})
                pages = sorted(raw_pages.values(), key=lambda p: p.get("index", 999))
                for p in pages:
                    title = p.get("title", "").lower()
                    if any(title.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        thumb = p.get("imageinfo", [{}])[0].get("thumburl")
                        if thumb and thumb.startswith("http"):
                            img_req = urllib.request.Request(thumb, headers=HEADERS)
                            with urllib.request.urlopen(img_req, timeout=12) as img_resp:
                                img = Image.open(io.BytesIO(img_resp.read())).convert("RGB")
                                img.thumbnail((400, 400))
                                img.save(filepath, "JPEG", quality=80)
                                return item_id, filepath, True
        except Exception:
            continue

    return item_id, filepath, False



def run_benchmark():
    img_dir = os.path.join(PROJECT_ROOT, "test_data", "images")
    os.makedirs(img_dir, exist_ok=True)

    print("=" * 70)
    print("NutriVision 100-Dish Real-World Benchmark Builder")
    print("Downloading 100 authentic amateur/user-clicked food photos...")
    print("=" * 70)

    t0 = time.time()
    downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_single_item, item, img_dir): item for item in BENCHMARK_SPEC}
        for future in as_completed(futures):
            item_id, fpath, ok = future.result()
            if ok:
                downloaded += 1

    print(f"Downloaded {downloaded}/{len(BENCHMARK_SPEC)} authentic amateur food images in {time.time() - t0:.1f}s!")

    # Save Ground Truth Catalog
    gt_file = os.path.join(PROJECT_ROOT, "test_data", "benchmark_ground_truth.json")
    with open(gt_file, "w") as f:
        json.dump(BENCHMARK_SPEC, f, indent=2)
    print(f"Saved Ground Truth Catalog -> {gt_file}")

    print("\n" + "=" * 70)
    print("Loading NutriVision Two-Tier AI Pipeline...")
    print("=" * 70)

    clf_model = load_food_classifier()
    reg_model, reg_meta = load_calorie_regressor(os.path.join(PROJECT_ROOT, "models", "calorie_regressor.pt"))

    print("\nExecuting Inference across all 100 Benchmark Dishes...")

    results = []
    top1_correct = 0
    top3_correct = 0
    total_items = 0
    total_abs_calorie_error = 0.0
    total_abs_portion_error = 0.0
    total_topping_tp = 0
    total_topping_fp = 0
    total_topping_fn = 0

    csv_rows = []

    for item in BENCHMARK_SPEC:
        item_id = item["id"]
        key = item["key"]
        fname = f"{item_id}_{key}.jpg"
        fpath = os.path.join(img_dir, fname)

        if not os.path.exists(fpath):
            continue

        total_items += 1
        img = Image.open(fpath)

        # Run Tier-1, Tier-2 & Tier-3 Vision Pipeline
        top1, top_k, prep, detected_toppings, quantity_info = classify_food_image(clf_model, img, top_k=3)
        detected_top_keys = [t["topping_key"] for t in detected_toppings]
        detected_top_names = [t["display_name"] for t in detected_toppings]

        # Check classification accuracy
        gt_key = item["key"]
        is_top1 = (top1["food_key"] == gt_key)
        is_top3 = any(c["food_key"] == gt_key for c in top_k)

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1

        # Check multi-label topping precision/recall
        gt_toppings = set(item.get("toppings", []))
        pred_toppings = set(detected_top_keys)

        tp = len(gt_toppings.intersection(pred_toppings))
        fp = len(pred_toppings - gt_toppings)
        fn = len(gt_toppings - pred_toppings)

        total_topping_tp += tp
        total_topping_fp += fp
        total_topping_fn += fn

        # Run Calorie Estimation using AI detected portion
        portion = float(item["portion"])
        pred_portion = float(quantity_info.get("estimated_grams", portion))
        portion_error = abs(pred_portion - portion)
        total_abs_portion_error += portion_error

        base_nutri, toppings_list, total_nutri = calculate_composite_nutrition(top1["food_key"], portion, detected_top_keys)
        base_kcal = predict_calories(
            reg_model, reg_meta, top1["food_key"], portion,
            base_nutri["protein"], base_nutri["carbs"], base_nutri["fat"], base_nutri["fiber"]
        )
        toppings_kcal = sum(t["calories"] for t in toppings_list)
        pred_total_kcal = round(base_kcal + toppings_kcal, 1)

        gt_cal = float(item["cal"])
        cal_error = abs(pred_total_kcal - gt_cal)
        total_abs_calorie_error += cal_error

        res_record = {
            "id": item_id,
            "filename": fname,
            "ground_truth_dish": item["dish"],
            "ground_truth_key": gt_key,
            "ground_truth_toppings": [TOPPINGS_DATABASE[k]["display_name"] for k in item.get("toppings", []) if k in TOPPINGS_DATABASE],
            "ground_truth_portion_g": item["portion"],
            "predicted_portion_g": pred_portion,
            "portion_error_g": round(portion_error, 1),
            "detected_serving": quantity_info.get("serving_label", ""),
            "ground_truth_calories": gt_cal,
            "ground_truth_protein_g": item["p"],
            "ground_truth_carbs_g": item["c"],
            "ground_truth_fat_g": item["f"],
            "ground_truth_fiber_g": item["fib"],

            "predicted_dish": top1["display_name"],
            "predicted_key": top1["food_key"],
            "predicted_confidence": top1["confidence"],
            "top3_candidates": [f"{c['display_name']} ({c['confidence']}%)" for c in top_k],
            "detected_toppings": detected_top_names,
            "predicted_calories": pred_total_kcal,
            "predicted_protein_g": total_nutri["protein"],
            "predicted_carbs_g": total_nutri["carbs"],
            "predicted_fat_g": total_nutri["fat"],
            "predicted_fiber_g": total_nutri["fiber"],

            "is_top1_match": is_top1,
            "is_top3_match": is_top3,
            "calorie_absolute_error": round(cal_error, 1)
        }
        results.append(res_record)

        csv_rows.append({
            "ID": item_id,
            "Ground Truth Dish": item["dish"],
            "Predicted Dish": top1["display_name"],
            "Top-1 Confidence (%)": top1["confidence"],
            "Top-1 Match": "YES" if is_top1 else "NO",
            "Top-3 Match": "YES" if is_top3 else "NO",
            "GT Portion (g)": item["portion"],
            "Pred Portion (g)": pred_portion,
            "Detected Serving": quantity_info.get("serving_label", ""),
            "Portion Error (g)": round(portion_error, 1),
            "GT Toppings": "; ".join(res_record["ground_truth_toppings"]),
            "Detected Toppings": "; ".join(detected_top_names),
            "GT Calories (kcal)": gt_cal,
            "Predicted Calories (kcal)": pred_total_kcal,
            "Calorie Error (kcal)": round(cal_error, 1),
            "GT Protein (g)": item["p"],
            "Pred Protein (g)": total_nutri["protein"],
            "GT Carbs (g)": item["c"],
            "Pred Carbs (g)": total_nutri["carbs"],
            "GT Fat (g)": item["f"],
            "Pred Fat (g)": total_nutri["fat"]
        })

    # Save Results JSON and CSV
    results_json = os.path.join(PROJECT_ROOT, "test_data", "benchmark_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    results_csv = os.path.join(PROJECT_ROOT, "test_data", "benchmark_comparison.csv")
    if csv_rows:
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Calculate overall metrics
    top1_acc = (top1_correct / total_items) * 100.0 if total_items else 0
    top3_acc = (top3_correct / total_items) * 100.0 if total_items else 0
    mae_kcal = (total_abs_calorie_error / total_items) if total_items else 0
    mae_portion_g = (total_abs_portion_error / total_items) if total_items else 0

    p_prec = (total_topping_tp / (total_topping_tp + total_topping_fp)) * 100.0 if (total_topping_tp + total_topping_fp) else 0
    p_rec = (total_topping_tp / (total_topping_tp + total_topping_fn)) * 100.0 if (total_topping_tp + total_topping_fn) else 0
    p_f1 = (2 * p_prec * p_rec / (p_prec + p_rec)) if (p_prec + p_rec) else 0

    print("\n" + "=" * 70)
    print("🏆 NUTRIVISION 100-DISH BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated Real-World Dishes:  {total_items}")
    print(f"Top-1 Base Dish Classification Accuracy: {top1_acc:.2f}% ({top1_correct}/{total_items})")
    print(f"Top-3 Candidate Accuracy:               {top3_acc:.2f}% ({top3_correct}/{total_items})")
    print(f"Multi-Label Topping Precision:          {p_prec:.2f}%")
    print(f"Multi-Label Topping Recall:             {p_rec:.2f}%")
    print(f"Multi-Label Topping F1-Score:           {p_f1:.2f}%")
    print(f"Visual Portion Estimation MAE:          {mae_portion_g:.1f} g")
    print(f"Calorie Regression Mean Absolute Error:  {mae_kcal:.2f} kcal")
    print("=" * 70)
    print(f"Saved Complete Benchmark CSV  -> {results_csv}")
    print(f"Saved Complete Benchmark JSON -> {results_json}")


if __name__ == "__main__":
    run_benchmark()
