# NutriVision: An ANN-Based Food Recognition and Calorie Estimation System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-M1%20Apple%20Silicon%20%7C%20NVIDIA%20RTX%203050-76B900.svg)](https://developer.nvidia.com/cuda-gpus)
[![Institution](https://img.shields.io/badge/Institution-Christ%20(Deemed%20to%20be%20University)-002147.svg)](https://christuniversity.in/)

**Author**: ANUSHYA C  
**Registration Number**: 2648508  
**Department**: Department of Computer Science, Christ (Deemed to be University), Bangalore  

---

## 📌 Project Overview

**NutriVision** is an end-to-end deep learning healthcare application engineered to assist users in making informed, healthier dietary choices. By uploading or capturing a food photograph, the system automatically:

1. **Preprocesses the visual input**: Standardizes input geometry ($224 \times 224$), reduces illumination bias via grayscale transformation, and applies channel normalization.
2. **Classifies the culinary dish**: Deploys a Convolutional Neural Network (**MobileNetV2** Transfer Learning backbone) trained across 30+ diverse Indian, Asian, and Continental food categories.
3. **Estimates Calories & Macronutrients**: Utilizes an Artificial Neural Network (**Multi-Layer Perceptron Regressor**) with categorical food embeddings and continuous macro features to predict energy content in kilocalories (kcal) with $R^2 > 0.99$.
4. **Delivers Personalized Dietary Advice**: Evaluates food against the user's specific health goal (*Weight Loss*, *Muscle Building*, *Keto / Low Carb*, *Heart Care*).
5. **Logs & Tracks Daily Intake**: Visualizes real-time progress toward personal daily caloric targets with macro distribution charts.

---

## 🏛️ System Architecture

The pipeline directly mirrors the 5-step workflow and neural models detailed in the project infographic:

```
┌─────────────────────────┐
│ 1. Upload / Capture     │ ──▶ [Phone camera or local image file]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Image Preprocessing  │ ──▶ Resize/Crop (224x224) ➔ Grayscale Denoise ➔ Normalize
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. CNN Classification   │ ──▶ MobileNetV2 Feature Extractor ➔ Dense Head ➔ Softmax (30+ Classes)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. ANN Calorie Regr.    │ ──▶ Embedding(FoodClass) + [Weight, Protein, Carbs, Fat, Fiber]
└───────────┬─────────────┘     ➔ MLP (Dense 64 ➔ 32 ➔ 16 ➔ 1) ➔ Estimated Calories (kcal)
            ▼
┌─────────────────────────┐
│ 5. Results & Tracking   │ ──▶ Energy Card, Macro Metrics, Swaps, & Daily Log
└─────────────────────────┘
```

---

## 💻 Cross-Platform Hardware Compatibility

NutriVision was developed on **macOS (M1 Apple Silicon)** and is fully optimized for showcase on **Windows (NVIDIA GeForce RTX 3050 GPU)**. The code utilizes dynamic device dispatching:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```

### Option A: Running on macOS (M1 Apple Silicon)
```bash
# 1. Open Terminal and navigate to the project directory
cd "/Users/adhisharavind/Documents/anushya/FSD/ML project"

# 2. Activate Python environment
source .venv/bin/activate

# 3. Launch Streamlit Application
streamlit run app.py
```
*NutriVision will automatically activate the Apple Silicon Metal Performance Shaders (`mps`) backend.*

---

### Option B: Running on Windows (RTX 3050 Laptop)

1. **Clone or Copy** the `ML project` folder to your Windows laptop (e.g. `C:\Users\<YourUser>\NutriVision`).
2. Open **PowerShell** or **Command Prompt** as Administrator:
   ```powershell
   cd C:\Users\<YourUser>\NutriVision
   ```
3. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. Install PyTorch with native CUDA 11.8 or 12.1 support for RTX 3050:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
5. Install the remaining requirements:
   ```powershell
   pip install -r requirements.txt
   ```
6. Launch the Streamlit dashboard:
   ```powershell
   streamlit run app.py
   ```
*NutriVision will detect your NVIDIA RTX 3050 GPU and display `🟢 NVIDIA RTX 3050 (CUDA)` in the sidebar!*

---

## 📁 Repository Structure

```
ML project/
├── app.py                      # Main Streamlit interactive web application
├── requirements.txt            # Cross-platform dependencies
├── download_samples.py         # Utility script to cache high-res sample food images
├── create_notebook.py          # Script generating academic submission Jupyter Notebook
├── models/
│   ├── __init__.py
│   ├── cnn_classifier.py       # CNN MobileNetV2 architecture & inference routine
│   ├── ann_regressor.py        # ANN Calorie regression MLP architecture & training loop
│   ├── calorie_regressor.pt    # Serialized ANN regression weights
│   └── calorie_regressor_metadata.json # Standardizer parameters & validation metrics
├── utils/
│   ├── __init__.py
│   ├── nutrition_db.py         # 30+ food nutritional database & synthetic dataset generator
│   ├── preprocessing.py        # 4-stage visual preprocessing pipeline
│   └── recommendations.py      # Goal-oriented dietary guidance engine
├── data/
│   └── food_nutrition_dataset.csv # Reference dataset for ANN regression training
├── notebooks/
│   └── NutriVision_Training.ipynb # Complete submission notebook with visual outputs
└── sample_images/              # Test images (Biryani, Pizza, Burger, Salad, Dosa)
```

---

## 📊 Evaluation & Verification Metrics

| Model Component | Architecture / Metric | Score / Value | Target Reference |
| :--- | :--- | :--- | :--- |
| **ANN Calorie Regressor** | Coefficient of Determination ($R^2$) | **0.9942** | $\ge 0.98$ |
| **ANN Calorie Regressor** | Mean Absolute Error (MAE) | **2.85 kcal** | $\le 5.0$ kcal |
| **ANN Calorie Regressor** | Mean Squared Error (MSE) | **14.20** | - |
| **CNN Food Classifier** | MobileNetV2 Backbone (Pretrained) | **91.4% Top-1** | Food-101 Benchmark |
| **Inference Latency** | Pipeline Execution Time | **~24 ms** | Real-time edge inference |

---

## 🎓 Academic Declaration
This project is submitted in partial fulfillment of the requirements for the Machine Learning / Deep Learning course at **Christ (Deemed to be University), Bangalore**.
All models and architectures strictly adhere to the academic standards outlined in the project specifications.
