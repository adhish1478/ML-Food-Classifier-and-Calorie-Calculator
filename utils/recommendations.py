"""
NutriVision Goal-Based Dietary Recommendation Engine.
Matches the 'Dietary Recommendations' and 'Goal Tracking' features
highlighted in the project infographic.
"""

from typing import Dict, Any, List


def generate_dietary_recommendations(nutrition_data: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
    """
    Generates intelligent personalized dietary insights and healthier alternatives
    tailored to the user's primary health/fitness goal.
    """
    calories = nutrition_data.get("calories", 0.0)
    protein = nutrition_data.get("protein", 0.0)
    carbs = nutrition_data.get("carbs", 0.0)
    fat = nutrition_data.get("fat", 0.0)
    fiber = nutrition_data.get("fiber", 0.0)
    portion_g = nutrition_data.get("portion_grams", 100.0)
    alt = nutrition_data.get("healthy_alternative", "Consider pairing with fresh green vegetables.")

    status = "info"
    recommendations: List[str] = []
    goal_fit_score = 75  # out of 100

    if user_goal == "Weight Loss (Caloric Deficit)":
        if calories > 500:
            status = "warning"
            goal_fit_score = 45
            recommendations.append(f"⚠️ High caloric density ({calories:.0f} kcal). Consider splitting into 2 meals or reducing portion from {portion_g:.0f}g to {portion_g*0.65:.0f}g.")
        else:
            goal_fit_score = 85
            recommendations.append(f"✅ Well-aligned calorie count ({calories:.0f} kcal) for a weight loss meal target (typically 400-550 kcal).")

        if fiber < 2.0:
            recommendations.append("💡 Add a side cucumber or spinach salad to boost satiety without adding excess calories.")
        if alt:
            recommendations.append(f"🌱 Healthier swap: {alt}")

    elif user_goal == "Muscle Building (High Protein)":
        protein_ratio = (protein * 4.0) / max(calories, 1.0)
        if protein >= 25.0 or protein_ratio >= 0.25:
            status = "success"
            goal_fit_score = 92
            recommendations.append(f"💪 Excellent protein content ({protein:.1f}g)! Supports muscle hypertrophy and protein synthesis.")
        else:
            status = "warning"
            goal_fit_score = 55
            recommendations.append(f"⚡ Moderate protein ({protein:.1f}g). Pair with 2 boiled egg whites, grilled chicken, or tofu to hit the 30g+ target.")

        if carbs < 30.0:
            recommendations.append("🍚 Consider adding healthy complex carbs (sweet potato, brown rice) for glycogen replenishment.")

    elif user_goal == "Keto / Low Carb":
        if carbs <= 10.0:
            status = "success"
            goal_fit_score = 95
            recommendations.append(f"🥑 Highly keto compliant with only {carbs:.1f}g net carbs.")
        else:
            status = "warning"
            goal_fit_score = 40
            recommendations.append(f"⚠️ High carbohydrate load ({carbs:.1f}g). May kick you out of ketosis. Swap carb-heavy components for leafy greens.")

    elif user_goal == "Heart & Cholesterol Care":
        if fat > 15.0:
            status = "warning"
            goal_fit_score = 50
            recommendations.append(f"🫀 Total fat is {fat:.1f}g. Opt for grilling, steaming, or air-frying rather than deep frying.")
        else:
            goal_fit_score = 88
            recommendations.append("✅ Low saturated fat profile favorable for cardiovascular health.")
        if fiber >= 3.0:
            recommendations.append(f"🌾 Good fiber content ({fiber:.1f}g) helps regulate LDL cholesterol.")

    else:  # Balanced Diet (Maintenance)
        goal_fit_score = 80
        recommendations.append("🥗 Well-rounded meal. Ensure you drink 400-500ml water alongside.")
        if alt:
            recommendations.append(f"✨ Pro-tip: {alt}")

    return {
        "status": status,
        "goal_fit_score": goal_fit_score,
        "recommendations": recommendations,
        "healthy_alternative": alt
    }
