"""
NutriVision Calorie & Macronutrient ANN Regression Model.
Matches Step 4 and Section 2 ('ANN for Calorie Prediction') of the infographic:
Input Features: Food Category, Portion Size (g), Carbs (g), Protein (g), Fat (g), Fiber (g)
-> Multi-Layer Perceptron (ANN regression) -> Estimated Calories (kcal).
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nutrition_db import FOOD_CLASSES, CLASS_TO_IDX, generate_ann_training_dataset


def get_optimal_device() -> torch.device:
    """
    Auto-detects best available compute hardware across platforms:
    - Windows: NVIDIA CUDA (e.g. RTX 3050)
    - macOS: Apple Silicon Metal Performance Shaders (M1/M2/M3 MPS)
    - Fallback: High-performance CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class CalorieRegressionANN(nn.Module):
    """
    Deep Artificial Neural Network (Multi-Layer Perceptron) for Calorie Regression.
    Incorporates an Embedding Layer for categorical food classes and Dense Linear
    layers with Batch Normalization, ReLU non-linearities, and Dropout regularisation.
    """
    def __init__(self, num_classes: int = len(FOOD_CLASSES), embedding_dim: int = 16, num_numerical: int = 5):
        super(CalorieRegressionANN, self).__init__()
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)

        in_features = embedding_dim + num_numerical  # 16 + 5 = 21

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, class_indices: torch.Tensor, numerical_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        class_indices: (batch_size,) integer tensor of food class IDs
        numerical_feats: (batch_size, 5) float tensor [portion_g, protein, carbs, fat, fiber]
        """
        embedded = self.class_embedding(class_indices)  # (batch_size, 16)
        x = torch.cat([embedded, numerical_feats], dim=1)  # (batch_size, 21)
        return self.mlp(x).squeeze(-1)


class NutritionDataset(Dataset):
    def __init__(self, class_ids: np.ndarray, num_features: np.ndarray, targets: np.ndarray):
        self.class_ids = torch.tensor(class_ids, dtype=torch.long)
        self.num_features = torch.tensor(num_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.class_ids[idx], self.num_features[idx], self.targets[idx]


def train_calorie_regressor(
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    save_path: str = None
) -> Tuple[CalorieRegressionANN, Dict[str, Any]]:
    """
    Trains the Calorie Regression ANN model, validates metrics, and serializes
    both the weights (.pt) and standardizer parameters.
    """
    device = get_optimal_device()
    print(f"[NutriVision ANN] Training Calorie Regressor on device: {device}")

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "food_nutrition_dataset.csv")
    if os.path.exists(csv_path):
        import csv
        class_ids, raw_num_list, targets_list = [], [], []
        with open(csv_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_ids.append(int(row["class_idx"]))
                raw_num_list.append([
                    float(row["portion_grams"]),
                    float(row["protein_g"]),
                    float(row["carbs_g"]),
                    float(row["fat_g"]),
                    float(row["fiber_g"])
                ])
                targets_list.append(float(row["calories_kcal"]))
        class_ids = np.array(class_ids, dtype=np.int64)
        raw_num = np.array(raw_num_list, dtype=np.float32)
        targets = np.array(targets_list, dtype=np.float32)
    else:
        df = generate_ann_training_dataset(num_samples=6000, random_seed=42)
        class_ids = df["class_idx"].values
        raw_num = df[["portion_grams", "protein_g", "carbs_g", "fat_g", "fiber_g"]].values
        targets = df["calories_kcal"].values

    # Fit feature standardizer for stable gradient descent
    mean = raw_num.mean(axis=0)
    std = raw_num.std(axis=0) + 1e-8
    norm_num = (raw_num - mean) / std


    # Train / Validation Split (80 / 20)
    n = len(class_ids)
    indices = np.random.RandomState(42).permutation(n)

    train_idx, val_idx = indices[:int(0.8 * n)], indices[int(0.8 * n):]

    train_ds = NutritionDataset(class_ids[train_idx], norm_num[train_idx], targets[train_idx])
    val_ds = NutritionDataset(class_ids[val_idx], norm_num[val_idx], targets[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CalorieRegressionANN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_cls, b_num, b_y in train_loader:
            b_cls, b_num, b_y = b_cls.to(device), b_num.to(device), b_y.to(device)

            optimizer.zero_grad()
            preds = model(b_cls, b_num)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(b_y)

        train_loss = total_loss / len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_abs_err = 0.0
        with torch.no_grad():
            for b_cls, b_num, b_y in val_loader:
                b_cls, b_num, b_y = b_cls.to(device), b_num.to(device), b_y.to(device)
                preds = model(b_cls, b_num)
                loss = criterion(preds, b_y)
                val_loss += loss.item() * len(b_y)
                val_abs_err += torch.sum(torch.abs(preds - b_y)).item()

        val_loss = val_loss / len(val_ds)
        val_mae = val_abs_err / len(val_ds)
        scheduler.step(val_loss)
        model.train()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train MSE: {train_loss:.2f} | Val MSE: {val_loss:.2f} | Val MAE: {val_mae:.2f} kcal")

    # Compute R^2 Score on validation set
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for b_cls, b_num, b_y in val_loader:
            b_cls, b_num = b_cls.to(device), b_num.to(device)
            preds = model(b_cls, b_num)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(b_y.numpy())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    ss_res = np.sum((all_trues - all_preds) ** 2)
    ss_tot = np.sum((all_trues - np.mean(all_trues)) ** 2)
    r2_score = 1.0 - (ss_res / ss_tot)
    final_mae = np.mean(np.abs(all_trues - all_preds))

    metrics = {
        "r2_score": float(r2_score),
        "val_mae_kcal": float(final_mae),
        "val_mse": float(val_loss),
        "history": history,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist()
    }

    print(f"\n[NutriVision ANN Final Metrics] R² Score: {r2_score:.4f} | Validation MAE: {final_mae:.2f} kcal")

    # Serialize weights and scaler metadata
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        meta_path = os.path.splitext(save_path)[0] + "_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Model saved -> {save_path}")
        print(f"Metadata saved -> {meta_path}")

    return model, metrics


def load_calorie_regressor(model_path: str, device: torch.device = None) -> Tuple[CalorieRegressionANN, Dict[str, Any]]:
    """
    Loads saved model weights and standardization metadata.
    """
    if device is None:
        device = get_optimal_device()

    meta_path = os.path.splitext(model_path)[0] + "_metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file {meta_path} not found.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model = CalorieRegressionANN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, meta


def predict_calories(
    model: CalorieRegressionANN,
    metadata: Dict[str, Any],
    food_key: str,
    portion_g: float,
    protein_g: float,
    carbs_g: float,
    fat_g: float,
    fiber_g: float,
    device: torch.device = None
) -> float:
    """
    Executes inference with the ANN Regressor to predict calorie count (kcal).
    """
    if device is None:
        device = get_optimal_device()

    class_idx = CLASS_TO_IDX.get(food_key, 0)
    raw_num = np.array([portion_g, protein_g, carbs_g, fat_g, fiber_g], dtype=np.float32)

    mean = np.array(metadata["scaler_mean"], dtype=np.float32)
    std = np.array(metadata["scaler_std"], dtype=np.float32)
    norm_num = (raw_num - mean) / std

    cls_tensor = torch.tensor([class_idx], dtype=torch.long, device=device)
    num_tensor = torch.from_numpy(norm_num.reshape(1, -1)).to(device=device, dtype=torch.float32)


    model.eval()
    with torch.no_grad():
        pred = model(cls_tensor, num_tensor).item()

    return max(10.0, round(pred, 1))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_file = os.path.join(current_dir, "calorie_regressor.pt")
    print("Starting ANN Calorie Regressor Training...")
    model, metrics = train_calorie_regressor(epochs=35, batch_size=64, save_path=save_file)
    print("Training Complete! Validated Metrics:")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Validation MAE: {metrics['val_mae_kcal']:.2f} kcal")

