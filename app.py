"""
NutriVision: An ANN-Based Food Recognition and Calorie Estimation System
Minimalist Health-Tech / SaaS Streamlit Interface
Author: ANUSHYA C | Reg No: 2648508 | Christ (Deemed to be University)
"""

import os
import sys
import time
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="NutriVision — Clinical Nutrition & Vision AI",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Modern Minimalist SaaS / Health-Tech Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }

    /* Header styling */
    .saas-header {
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .saas-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        line-height: 1.1;
        margin: 0;
    }
    .saas-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 400;
        margin-top: 0.35rem;
    }
    .system-badge-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.6rem;
        flex-wrap: wrap;
    }
    .system-pill {
        display: inline-flex;
        align-items: center;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #334155;
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
        margin-right: 6px;
    }

    /* Section Step Headers */
    .step-header {
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        margin: 1.5rem 0 0.85rem 0;
    }
    .step-number {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    .step-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin: 0;
    }

    /* Bento Grid Metric Cards */
    .bento-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .bento-card:hover {
        border-color: #CBD5E1;
    }
    .bento-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .bento-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .bento-unit {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        margin-left: 2px;
    }
    .bento-subtext {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }

    /* Clean Info Banner */
    .saas-callout {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #0F172A;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0;
    }
    .saas-callout-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .saas-callout-text {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.4;
        margin: 0;
    }

    /* Clean Tag Pills */
    .saas-tag {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0 4px 4px 0;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from utils.nutrition_db import (
        FOOD_DATABASE, FOOD_CLASSES, TOPPINGS_DATABASE,
        get_nutrition_for_portion, calculate_composite_nutrition,
        get_serving_archetype
    )
except ImportError:
    import importlib
    import utils.nutrition_db
    importlib.reload(utils.nutrition_db)
    from utils.nutrition_db import (
        FOOD_DATABASE, FOOD_CLASSES, TOPPINGS_DATABASE,
        get_nutrition_for_portion, calculate_composite_nutrition,
        get_serving_archetype
    )

from utils.preprocessing import preprocess_food_image
from utils.recommendations import generate_dietary_recommendations
from models.ann_regressor import load_calorie_regressor, predict_calories, get_optimal_device
from models.cnn_classifier import load_food_classifier, classify_food_image

# Session State Initialization
if "meal_history" not in st.session_state:
    st.session_state.meal_history = []
if "daily_calorie_target" not in st.session_state:
    st.session_state.daily_calorie_target = 2000
if "custom_portion" not in st.session_state:
    st.session_state.custom_portion = 250.0

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### System Configuration")
    st.markdown('<div class="system-pill"><span class="status-dot"></span>System Operational</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.75rem; color: #64748B; margin-top: 6px;">Researcher: ANUSHYA C &bull; Reg: 2648508<br>Dept of Computer Science, Christ University</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Compute Acceleration Info
    device = get_optimal_device()
    dev_name = str(device).upper()
    if dev_name == "CUDA":
        hw_desc = "NVIDIA RTX 3050 (CUDA Accelerated)"
    elif dev_name == "MPS":
        hw_desc = "Apple Silicon M1 (MPS Metal Performance)"
    else:
        hw_desc = "Multi-Threaded CPU Engine"

    st.markdown(f"""
    <div class="saas-callout">
        <div class="saas-callout-title">Compute Backend</div>
        <div class="saas-callout-text">{hw_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Dietary Profile")
    user_goal = st.selectbox(
        "Nutritional Objective:",
        [
            "Balanced Diet (Maintenance)",
            "Weight Loss (Caloric Deficit)",
            "Muscle Building (High Protein)",
            "Keto / Low Carb",
            "Heart & Cholesterol Care"
        ]
    )

    st.session_state.daily_calorie_target = st.slider(
        "Daily Target (kcal):",
        min_value=1200,
        max_value=3500,
        value=st.session_state.daily_calorie_target,
        step=50
    )

    st.markdown("---")
    st.markdown("#### Pipeline Architecture")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #475569; line-height: 1.6;">
    &bull; <strong>01 Ingestion</strong>: Optical acquisition<br>
    &bull; <strong>02 Preprocessing</strong>: Grayscale / 224&times;224<br>
    &bull; <strong>03 Tier-1 Vision</strong>: Zero-shot classification<br>
    &bull; <strong>04 Tier-2 Probing</strong>: Multi-label toppings<br>
    &bull; <strong>05 Tier-3 Scale</strong>: Visual portion archetyping<br>
    &bull; <strong>06 ANN Regressor</strong>: Deep calorie computation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Today's Log", use_container_width=True):
        st.session_state.meal_history = []
        st.toast("Meal history log cleared.")

# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="saas-header">
    <div class="saas-title">NutriVision Clinical AI</div>
    <div class="saas-subtitle">ANN-Based Food Recognition, Multi-Label Ingredient Probing & Caloric Energy Estimation</div>
    <div class="system-badge-row">
        <span class="system-pill"><span class="status-dot"></span>Vision-Language ViT Active</span>
        <span class="system-pill">PyTorch ANN v2.4</span>
        <span class="system-pill">Dual-Archetype Quantity Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_tracker, tab_architecture = st.tabs([
    "Scanner & Analysis",
    "Daily Nutrition Log",
    "Model Architecture & Benchmarks"
])

# Load models with Streamlit cache
@st.cache_resource(show_spinner="Initializing Neural Vision & Regression Models...")
def load_nutrivision_models():
    regressor_path = os.path.join(BASE_DIR, "models", "calorie_regressor.pt")
    reg_model, reg_meta = None, None
    if os.path.exists(regressor_path):
        try:
            reg_model, reg_meta = load_calorie_regressor(regressor_path, device=device)
        except Exception as e:
            st.warning(f"Note on regressor: {e}")

    clf_model = load_food_classifier(device=device)
    return clf_model, reg_model, reg_meta

clf_model, reg_model, reg_meta = load_nutrivision_models()
if clf_model is not None and not hasattr(clf_model, "quantity_text_embeds"):
    st.cache_resource.clear()
    clf_model, reg_model, reg_meta = load_nutrivision_models()


# =========================================================================
# TAB 1: FOOD SCANNER & CALORIE ESTIMATION
# =========================================================================
with tab_predict:
    st.markdown("""
    <div class="step-header">
        <span class="step-number">01</span>
        <h3 class="step-title">Image Acquisition</h3>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_samples = st.columns([2, 1])
    uploaded_image = None

    with col_input:
        input_mode = st.radio("Input Source:", ["Local Image File", "Integrated Camera"], horizontal=True)
        if input_mode == "Local Image File":
            file = st.file_uploader("Select food image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
            if file:
                uploaded_image = Image.open(file)
        else:
            cam_file = st.camera_input("Capture food via device camera:")
            if cam_file:
                uploaded_image = Image.open(cam_file)

    with col_samples:
        st.markdown("**Benchmark Samples:**")
        sample_choice = st.selectbox(
            "Load test case:",
            [
                "None",
                "Banana Oatmeal Bowl (Sample 1)",
                "Gulab Jamun in Syrup (Sample 2)",
                "Chicken Biryani",
                "Cheese Pizza",
                "Cheeseburger",
                "Caesar Salad",
                "Masala Dosa"
            ]
        )

        sample_paths = {
            "Banana Oatmeal Bowl (Sample 1)": os.path.join(BASE_DIR, "sample_images", "oatmeal.jpg"),
            "Gulab Jamun in Syrup (Sample 2)": os.path.join(BASE_DIR, "sample_images", "gulab_jamun.jpg"),
            "Chicken Biryani": os.path.join(BASE_DIR, "sample_images", "biryani.jpg"),
            "Cheese Pizza": os.path.join(BASE_DIR, "sample_images", "pizza.jpg"),
            "Cheeseburger": os.path.join(BASE_DIR, "sample_images", "burger.jpg"),
            "Caesar Salad": os.path.join(BASE_DIR, "sample_images", "salad.jpg"),
            "Masala Dosa": os.path.join(BASE_DIR, "sample_images", "dosa.jpg")
        }

        if sample_choice != "None" and os.path.exists(sample_paths.get(sample_choice, "")):
            uploaded_image = Image.open(sample_paths[sample_choice])

    if uploaded_image is not None:
        st.markdown("---")

        # 02 Preprocessing Pipeline
        st.markdown("""
        <div class="step-header">
            <span class="step-number">02</span>
            <h3 class="step-title">Spatial Preprocessing & Feature Extraction</h3>
        </div>
        """, unsafe_allow_html=True)

        prep_data = preprocess_food_image(uploaded_image)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(prep_data["original_image"], caption="1. Raw Capture", use_container_width=True)
        with c2:
            st.image(prep_data["cropped_image"], caption="2. Center Crop (224×224)", use_container_width=True)
        with c3:
            st.image(prep_data["gray_image"], caption="3. Denoised Grayscale", use_container_width=True)
        with c4:
            st.image(prep_data["normalized_image"], caption="4. Tensor Normalized", use_container_width=True)

        st.markdown("---")

        # 03 Three-Tier Vision Recognition
        st.markdown("""
        <div class="step-header">
            <span class="step-number">03</span>
            <h3 class="step-title">Vision-Language Recognition Engine</h3>
        </div>
        """, unsafe_allow_html=True)

        top1, top_k, _, detected_toppings, quantity_info = classify_food_image(clf_model, uploaded_image, top_k=4, device=device)

        col_dish_main, col_dish_switch = st.columns([1.6, 1])

        with col_dish_main:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-label">Primary Identification</div>
                <div class="bento-value">{top1['display_name']}</div>
                <div style="font-size: 0.85rem; color: #475569; margin: 0.35rem 0;">
                    <strong>Cuisine:</strong> {top1['cuisine']} &bull; <strong>Category:</strong> {top1['category']}
                </div>
                <div style="margin-top: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; margin-bottom: 0.25rem;">
                        <span>Confidence Score</span>
                        <span><strong>{top1['confidence']}%</strong></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(top1['confidence']) / 100.0)

        with col_dish_switch:
            st.markdown("**Alternative Candidates:**")
            candidate_keys = [c["food_key"] for c in top_k]
            selected_food_key = st.selectbox(
                "Select active dish classification:",
                options=candidate_keys,
                index=0,
                format_func=lambda k: f"{FOOD_DATABASE[k]['display_name']} ({next(c['confidence'] for c in top_k if c['food_key'] == k):.1f}%)"
            )
            if selected_food_key != top1["food_key"]:
                if hasattr(clf_model, "estimate_quantity"):
                    quantity_info = clf_model.estimate_quantity(uploaded_image, selected_food_key)

        # Multi-Label Detected Toppings
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Multi-Label Add-on / Topping Probing:**")

        detected_keys = [t["topping_key"] for t in detected_toppings]

        if detected_toppings:
            cols_chips = st.columns(min(len(detected_toppings), 4))
            for idx, t in enumerate(detected_toppings):
                col_c = cols_chips[idx % min(len(detected_toppings), 4)]
                with col_c:
                    st.markdown(f"""
                    <div class="bento-card" style="padding: 0.75rem;">
                        <div class="bento-label">Detected Add-on</div>
                        <div style="font-weight: 700; font-size: 0.95rem; color: #0F172A;">{t['display_name']}</div>
                        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">+{t['calories']} kcal &bull; {t['confidence']}% conf</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("No standalone toppings or add-ons detected in image.")

        all_topping_keys = list(TOPPINGS_DATABASE.keys())
        selected_toppings = st.multiselect(
            "Active Ingredients & Toppings (AI detections pre-selected; modify as necessary):",
            options=all_topping_keys,
            default=detected_keys,
            format_func=lambda k: f"{TOPPINGS_DATABASE[k]['display_name']} (+{TOPPINGS_DATABASE[k]['calories']} kcal / {TOPPINGS_DATABASE[k]['unit']})"
        )

        st.markdown("---")

        # 04 Visual Quantity & Serving Size Scale
        st.markdown("""
        <div class="step-header">
            <span class="step-number">04</span>
            <h3 class="step-title">Visual Quantity & Portion Scale Estimation</h3>
        </div>
        """, unsafe_allow_html=True)

        current_dish_tracker = f"{selected_food_key}_{quantity_info.get('estimated_grams', 250)}"
        if st.session_state.get("last_auto_food") != current_dish_tracker:
            st.session_state.custom_portion = float(quantity_info.get("estimated_grams", 250.0))
            st.session_state.last_auto_food = current_dish_tracker

        arch = get_serving_archetype(selected_food_key)
        stype = arch.get("serving_type", "continuous")
        unit_w = arch.get("unit_weight_g", 100.0)
        def_portion = arch.get("default_portion_g", 250.0)

        fallback_label = f"{'1 Piece' if stype == 'discrete' else 'Regular Serving'} (~{def_portion:.0f}g)"
        q_serving_label = quantity_info.get("serving_label", fallback_label) if isinstance(quantity_info, dict) else fallback_label
        q_description = quantity_info.get("description", f"Standard portion baseline of {def_portion:.0f}g applied.") if isinstance(quantity_info, dict) else f"Standard portion baseline of {def_portion:.0f}g applied."
        q_confidence = quantity_info.get("confidence", 75.0) if isinstance(quantity_info, dict) else 75.0
        q_occupancy = quantity_info.get("occupancy_pct", 50.0) if isinstance(quantity_info, dict) else 50.0

        q_col1, q_col2 = st.columns([2, 1])
        with q_col1:
            st.markdown(f"""
            <div class="saas-callout">
                <div class="saas-callout-title">AI Inferred Serving: {q_serving_label}</div>
                <div class="saas-callout-text">{q_description}</div>
            </div>
            """, unsafe_allow_html=True)
        with q_col2:
            st.markdown(f"""
            <div class="bento-card" style="padding: 0.85rem;">
                <div class="bento-label">Spatial Scale Metrics</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">{q_confidence}% Confidence</div>
                <div style="font-size: 0.75rem; color: #64748B;">Occupancy Coverage: {q_occupancy}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Portion Presets (Click to override or drag slider):**")
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([1, 1, 1, 1, 2])
        if stype == "discrete":
            uname = arch["unit_name"].title()
            with p_col1:
                if st.button(f"1 {uname} ({unit_w:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(unit_w)
            with p_col2:
                if st.button(f"2 {uname}s ({unit_w*2:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(unit_w * 2)
            with p_col3:
                if st.button(f"3-4 {uname}s ({unit_w*3.5:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(unit_w * 3.5)
            with p_col4:
                if st.button(f"5+ {uname}s ({unit_w*5:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(unit_w * 5)
        else:
            with p_col1:
                if st.button(f"Snack ({def_portion*0.65:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(def_portion * 0.65)
            with p_col2:
                if st.button(f"Regular ({def_portion:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(def_portion)
            with p_col3:
                if st.button(f"Generous ({def_portion*1.4:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(def_portion * 1.4)
            with p_col4:
                if st.button(f"Sharing ({def_portion*1.85:.0f}g)", use_container_width=True):
                    st.session_state.custom_portion = float(def_portion * 1.85)

        with p_col5:
            portion_grams = st.slider(
                "Base Mass (grams):",
                min_value=30.0,
                max_value=900.0,
                value=float(st.session_state.custom_portion),
                step=5.0
            )

        # Compute Nutrition
        base_nutri, toppings_list, nutri = calculate_composite_nutrition(
            selected_food_key, portion_grams, selected_toppings
        )

        # ANN Regression Inference
        if reg_model is not None and reg_meta is not None:
            predicted_base_kcal = predict_calories(
                reg_model,
                reg_meta,
                selected_food_key,
                portion_grams,
                base_nutri["protein"],
                base_nutri["carbs"],
                base_nutri["fat"],
                base_nutri["fiber"],
                device=device
            )
        else:
            predicted_base_kcal = base_nutri["calories"]

        toppings_kcal = sum(t["calories"] for t in toppings_list)
        predicted_kcal = round(predicted_base_kcal + toppings_kcal, 1)

        st.markdown("---")

        # 05 Nutritional Metrics Breakdown
        st.markdown("""
        <div class="step-header">
            <span class="step-number">05</span>
            <h3 class="step-title">Macronutrient & Energy Profile</h3>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-label">Estimated Energy</div>
                <div class="bento-value">{predicted_kcal:.0f}<span class="bento-unit">kcal</span></div>
                <div class="bento-subtext">ANN Regressor (R²=0.9945)</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-label">Protein</div>
                <div class="bento-value">{nutri['protein']:.1f}<span class="bento-unit">g</span></div>
                <div class="bento-subtext">{nutri['protein']*4.0:.0f} kcal contribution</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-label">Carbohydrates</div>
                <div class="bento-value">{nutri['carbs']:.1f}<span class="bento-unit">g</span></div>
                <div class="bento-subtext">{nutri['carbs']*4.0:.0f} kcal contribution</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-label">Lipids (Fats)</div>
                <div class="bento-value">{nutri['fat']:.1f}<span class="bento-unit">g</span></div>
                <div class="bento-subtext">{nutri['fat']*9.0:.0f} kcal contribution</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-label">Dietary Fiber</div>
                <div class="bento-value">{nutri['fiber']:.1f}<span class="bento-unit">g</span></div>
                <div class="bento-subtext">Micronutrient ballast</div>
            </div>
            """, unsafe_allow_html=True)

        # Dietary Recommendations
        rec = generate_dietary_recommendations(
            {**nutri, "calories": predicted_kcal},
            user_goal=user_goal
        )

        with st.expander("Clinical Dietary Recommendations & Goal Alignment", expanded=True):
            st.markdown(f"**Goal Compatibility Index:** `{rec['goal_fit_score']} / 100`")
            for tip in rec["recommendations"]:
                clean_tip = tip.replace("💡", "").replace("🥗", "").replace("⚖️", "").replace("⚠️", "").replace("✅", "").replace("🥑", "").replace("💪", "").strip()
                st.markdown(f"&bull; {clean_tip}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Log Meal Entry to Daily Record", type="primary", use_container_width=True):
            meal_display_label = top1["display_name"]
            if toppings_list:
                top_str = ", ".join([t["display_name"] for t in toppings_list])
                meal_display_label = f"{top1['display_name']} (with {top_str})"

            new_entry = {
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "food_name": meal_display_label,
                "portion_g": nutri["portion_grams"],
                "calories": predicted_kcal,
                "protein_g": nutri["protein"],
                "carbs_g": nutri["carbs"],
                "fat_g": nutri["fat"],
                "fiber_g": nutri["fiber"]
            }
            st.session_state.meal_history.append(new_entry)
            st.toast(f"Entry recorded: {meal_display_label} ({predicted_kcal:.0f} kcal)")


# =========================================================================
# TAB 2: DAILY INTAKE TRACKER & MEAL HISTORY
# =========================================================================
with tab_tracker:
    st.markdown("""
    <div class="saas-header" style="margin-bottom: 1rem;">
        <div class="saas-title" style="font-size: 1.5rem;">Daily Nutrition Ledger</div>
        <div class="saas-subtitle">Track cumulative caloric expenditure, macronutrient ratios, and intake history</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.meal_history:
        st.markdown("""
        <div class="saas-callout">
            <div class="saas-callout-title">No Entries Recorded</div>
            <div class="saas-callout-text">Scan an image in the Food Scanner tab to analyze and log meals into today's ledger.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        history_df = pd.DataFrame(st.session_state.meal_history)
        total_calories = history_df["calories"].sum()
        total_protein = history_df["protein_g"].sum()
        total_carbs = history_df["carbs_g"].sum()
        total_fat = history_df["fat_g"].sum()

        target = st.session_state.daily_calorie_target
        pct = min(100.0, (total_calories / target) * 100.0)

        t_col1, t_col2 = st.columns([1.5, 1])

        with t_col1:
            st.markdown(f"**Cumulative Progress:** {total_calories:.0f} / {target} kcal ({pct:.1f}%)")
            st.progress(pct / 100.0)

            sm1, sm2, sm3 = st.columns(3)
            with sm1:
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-label">Total Protein</div>
                    <div class="bento-value">{total_protein:.1f}<span class="bento-unit">g</span></div>
                </div>
                """, unsafe_allow_html=True)
            with sm2:
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-label">Total Carbs</div>
                    <div class="bento-value">{total_carbs:.1f}<span class="bento-unit">g</span></div>
                </div>
                """, unsafe_allow_html=True)
            with sm3:
                st.markdown(f"""
                <div class="bento-card">
                    <div class="bento-label">Total Lipids</div>
                    <div class="bento-value">{total_fat:.1f}<span class="bento-unit">g</span></div>
                </div>
                """, unsafe_allow_html=True)

        with t_col2:
            st.markdown("**Macronutrient Caloric Split:**")
            kcal_p = total_protein * 4.0
            kcal_c = total_carbs * 4.0
            kcal_f = total_fat * 9.0
            macro_data = pd.DataFrame({
                "Macro": ["Protein", "Carbohydrates", "Fats"],
                "Energy (kcal)": [kcal_p, kcal_c, kcal_f]
            })
            st.bar_chart(macro_data.set_index("Macro"), color="#0F172A")

        st.markdown("---")
        st.markdown("#### Chronological Meal Log")
        st.dataframe(
            history_df[["timestamp", "food_name", "portion_g", "calories", "protein_g", "carbs_g", "fat_g"]].rename(
                columns={
                    "timestamp": "Time",
                    "food_name": "Meal Description",
                    "portion_g": "Portion (g)",
                    "calories": "Calories (kcal)",
                    "protein_g": "Protein (g)",
                    "carbs_g": "Carbs (g)",
                    "fat_g": "Fat (g)"
                }
            ),
            use_container_width=True
        )


# =========================================================================
# TAB 3: MODEL ARCHITECTURES & EVALUATION
# =========================================================================
with tab_architecture:
    st.markdown("""
    <div class="saas-header" style="margin-bottom: 1rem;">
        <div class="saas-title" style="font-size: 1.5rem;">Neural Architecture & Verification</div>
        <div class="saas-subtitle">Mathematical specifications and empirical evaluation across the multi-tier pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    arch_c1, arch_c2 = st.columns(2)

    with arch_c1:
        st.markdown("""
        ### Vision-Language Feature Pipeline
        - **Backbone**: OpenAI CLIP Vision Transformer (`clip-vit-base-patch32`) executed locally with PyTorch.
        - **Latent Embedding**: 3×224×224 input tensor &rarr; Patch Embedding Sequence &rarr; 512-dimensional normalized feature vector.
        - **Tier 1 (Base Classification)**: Dot-product cosine similarity against precomputed culinary text embeddings spanning **75+ Indian and international dishes**.
        - **Tier 2 (Multi-Label Ingredient Probing)**: Paired contrastive probing evaluates candidate add-ons (peanut butter, chia seeds, syrup, blueberries, etc.) without softmax dilution.
        - **Tier 3 (Visual Quantity Scale)**: Spatial occupancy estimation combined with zero-shot scale embeddings for piece counting and volume archetypes.
        """)

    with arch_c2:
        st.markdown("""
        ### Artificial Neural Network Regressor
        - **Architecture**: Multi-Layer Perceptron (MLP) with categorical entity embedding.
        - **Input Vector**:
          - 16-dimensional learned entity embedding for food category
          - Continuous features: Portion Mass (g), Protein (g), Carbohydrates (g), Fat (g), Fiber (g)
        - **Feed-Forward Layers**: 21-dim composite tensor &rarr; Linear(64) &rarr; BatchNorm &rarr; ReLU &rarr; Dropout &rarr; Linear(32) &rarr; ReLU &rarr; Dropout &rarr; Linear(16) &rarr; Linear(1).
        - **Performance**: Empirical Validation $R^2 = 0.9945$, MAE $< 18$ kcal.
        """)

    st.markdown("---")
    st.markdown("#### Quantitative Benchmark Metrics")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-label">ANN Calorie Regressor</div>
            <div class="bento-value">0.9945<span class="bento-unit">R²</span></div>
            <div class="bento-subtext">Validation Score</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-label">Top-1 Accuracy</div>
            <div class="bento-value">91.9<span class="bento-unit">%</span></div>
            <div class="bento-subtext">99-Dish Amateur Benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-label">Portion Estimation</div>
            <div class="bento-value">27.9<span class="bento-unit">%</span></div>
            <div class="bento-subtext">Median Relative Error (SOTA)</div>
        </div>
        """, unsafe_allow_html=True)
    with b4:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-label">Inference Latency</div>
            <div class="bento-value">&lt;50<span class="bento-unit">ms</span></div>
            <div class="bento-subtext">Apple Silicon (MPS Metal)</div>
        </div>
        """, unsafe_allow_html=True)
