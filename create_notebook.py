import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# NutriVision: An ANN-Based Food Recognition & Calorie Estimation System\n",
                "\n",
                "**Author**: ANUSHYA C  \n",
                "**Registration Number**: 2648508  \n",
                "**Institution**: Christ (Deemed to be University), Bangalore  \n",
                "**Department**: Department of Computer Science (MS AIML / Data Science)  \n",
                "\n",
                "---\n",
                "\n",
                "## 1. Project Overview\n",
                "\n",
                "Accurate, automated nutritional assessment from food imagery is a fundamental challenge at the intersection of Computer Vision and Preventive Healthcare. **NutriVision** introduces a two-stage deep learning pipeline:\n",
                "\n",
                "1. **Stage 1 (CNN Food Classifier)**: A Convolutional Neural Network employing Transfer Learning (MobileNetV2 / ResNet backbone) to extract spatial, textural, and chromatic features from food photographs.\n",
                "2. **Stage 2 (ANN Calorie Regressor)**: An Artificial Neural Network (Multi-Layer Perceptron) with categorical embedding that takes the predicted food class, serving weight (grams), and macronutrient components (Protein, Carbohydrates, Fats, Fiber) to predict total energy expenditure in Kilocalories (kcal)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from PIL import Image\n",
                "\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "import torchvision.models as models\n",
                "import torchvision.transforms as transforms\n",
                "\n",
                "# Cross-Platform Device Selection (M1 Mac MPS / Windows RTX 3050 CUDA / CPU)\n",
                "if torch.cuda.is_available():\n",
                "    device = torch.device('cuda')\n",
                "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
                "    device = torch.device('mps')\n",
                "else:\n",
                "    device = torch.device('cpu')\n",
                "\n",
                "print(f\"NutriVision Compute Device Active: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Image Preprocessing Pipeline (Step 2)\n",
                "\n",
                "Following Step 2 of the architecture infographic, each input image passes through:\n",
                "- **Aspect-Ratio Resizing & Center Cropping** to standard $224 \\times 224$ pixels.\n",
                "- **Grayscale Transformation** to eliminate luminance and lighting bias.\n",
                "- **Tensor Normalization** using standard ImageNet channel statistics ($\\mu=[0.485, 0.456, 0.406]$, $\\sigma=[0.229, 0.224, 0.225]$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from PIL import ImageOps\n",
                "\n",
                "def demonstrate_preprocessing(image_path):\n",
                "    raw_img = Image.open(image_path).convert('RGB')\n",
                "    cropped_img = ImageOps.fit(raw_img, (224, 224), method=Image.Resampling.LANCZOS)\n",
                "    gray_img = cropped_img.convert('L')\n",
                "    \n",
                "    # Normalization\n",
                "    arr = np.array(cropped_img, dtype=np.float32) / 255.0\n",
                "    mean = np.array([0.485, 0.456, 0.406])\n",
                "    std = np.array([0.229, 0.224, 0.225])\n",
                "    norm_arr = (arr - mean) / std\n",
                "    norm_vis = ((norm_arr - norm_arr.min()) / (norm_arr.max() - norm_arr.min()) * 255.0).astype(np.uint8)\n",
                "    \n",
                "    fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
                "    axes[0].imshow(raw_img)\n",
                "    axes[0].set_title(\"1. Original Photo\")\n",
                "    axes[1].imshow(cropped_img)\n",
                "    axes[1].set_title(\"2. Resized (224x224)\")\n",
                "    axes[2].imshow(gray_img, cmap='gray')\n",
                "    axes[2].set_title(\"3. Grayscale (Denoised)\")\n",
                "    axes[3].imshow(norm_vis)\n",
                "    axes[3].set_title(\"4. Normalized Feature Map\")\n",
                "    for ax in axes: ax.axis('off')\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "# Test on sample pizza\n",
                "sample_img_path = \"../sample_images/pizza.jpg\"\n",
                "if os.path.exists(sample_img_path):\n",
                "    demonstrate_preprocessing(sample_img_path)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. ANN Calorie & Macronutrient Regression Model (Step 4)\n",
                "\n",
                "The Calorie Regression model is a Multi-Layer Perceptron (MLP) containing an **Embedding Layer** for categorical food representations, followed by Dense layers with **Batch Normalization**, **ReLU**, and **Dropout** regularisation.\n",
                "\n",
                "$$\\text{Calories (kcal)} = f_{\\text{ANN}}(\\text{Embedding}(\\text{FoodClass}), \\text{Portion}(g), \\text{Protein}(g), \\text{Carbs}(g), \\text{Fat}(g), \\text{Fiber}(g))$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class CalorieRegressionANN(nn.Module):\n",
                "    def __init__(self, num_classes=30, embedding_dim=16, num_numerical=5):\n",
                "        super(CalorieRegressionANN, self).__init__()\n",
                "        self.class_embedding = nn.Embedding(num_classes, embedding_dim)\n",
                "        in_features = embedding_dim + num_numerical\n",
                "        \n",
                "        self.mlp = nn.Sequential(\n",
                "            nn.Linear(in_features, 64),\n",
                "            nn.BatchNorm1d(64),\n",
                "            nn.ReLU(),\n",
                "            nn.Dropout(p=0.1),\n",
                "            \n",
                "            nn.Linear(64, 32),\n",
                "            nn.ReLU(),\n",
                "            nn.Dropout(p=0.1),\n",
                "            \n",
                "            nn.Linear(32, 16),\n",
                "            nn.ReLU(),\n",
                "            \n",
                "            nn.Linear(16, 1)\n",
                "        )\n",
                "        \n",
                "    def forward(self, class_ids, num_feats):\n",
                "        emb = self.class_embedding(class_ids)\n",
                "        x = torch.cat([emb, num_feats], dim=1)\n",
                "        return self.mlp(x).squeeze(-1)\n",
                "\n",
                "print(CalorieRegressionANN())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Training & Evaluation Metrics\n",
                "\n",
                "We evaluate the regression model using standard statistical performance metrics:\n",
                "- **Mean Squared Error (MSE)**: Measures average squared prediction deviations.\n",
                "- **Mean Absolute Error (MAE)**: Direct clinical interpretability in kilocalories (kcal).\n",
                "- **Coefficient of Determination ($R^2$ Score)**: Proportion of variance explained by the model ($R^2 > 0.99$ target)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load training dataset and display distribution\n",
                "dataset_path = \"../data/food_nutrition_dataset.csv\"\n",
                "if os.path.exists(dataset_path):\n",
                "    df = pd.read_csv(dataset_path)\n",
                "    print(f\"Total training records: {len(df)}\")\n",
                "    display(df.head())\n",
                "    \n",
                "    plt.figure(figsize=(10, 4))\n",
                "    plt.hist(df['calories_kcal'], bins=30, color='#2563EB', edgecolor='black', alpha=0.7)\n",
                "    plt.title(\"Caloric Distribution in Training Set\")\n",
                "    plt.xlabel(\"Calories (kcal)\")\n",
                "    plt.ylabel(\"Frequency\")\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Live Inference Pipeline Demonstration\n",
                "\n",
                "Here we test the complete end-to-end NutriVision pipeline on unseen images, outputting food classification, confidence, estimated calories, and macro breakdown."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from utils.nutrition_db import get_nutrition_for_portion\n",
                "from models.ann_regressor import load_calorie_regressor, predict_calories\n",
                "from models.cnn_classifier import load_food_classifier, classify_food_image\n",
                "\n",
                "# Load serialized models\n",
                "model_pt = \"../models/calorie_regressor.pt\"\n",
                "if os.path.exists(model_pt):\n",
                "    reg_model, reg_meta = load_calorie_regressor(model_pt, device=device)\n",
                "    clf_model = load_food_classifier(device=device)\n",
                "    \n",
                "    test_img = Image.open(\"../sample_images/biryani.jpg\")\n",
                "    top1, top_k, _ = classify_food_image(clf_model, test_img, device=device)\n",
                "    \n",
                "    nutri = get_nutrition_for_portion(top1['food_key'], portion_grams=350)\n",
                "    pred_kcal = predict_calories(\n",
                "        reg_model, reg_meta, top1['food_key'],\n",
                "        350, nutri['protein'], nutri['carbs'], nutri['fat'], nutri['fiber'],\n",
                "        device=device\n",
                "    )\n",
                "    \n",
                "    print(f\"Predicted Food: {top1['display_name']} (Confidence: {top1['confidence']}%)\")\n",
                "    print(f\"Portion Size: 350g\")\n",
                "    print(f\"Estimated Calories: {pred_kcal:.1f} kcal\")\n",
                "    print(f\"Macros -> Protein: {nutri['protein']}g | Carbs: {nutri['carbs']}g | Fat: {nutri['fat']}g\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_dir = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(out_dir, exist_ok=True)
nb_path = os.path.join(out_dir, "NutriVision_Training.ipynb")

with open(nb_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Created Jupyter Notebook: {nb_path}")
