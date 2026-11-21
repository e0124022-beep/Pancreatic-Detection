# Pancreatic Malignancy Detection Using Deep Learning

A research-oriented web application for binary classification of pancreatic medical images into **Normal** and **Pancreatic Tumour** classes. The project uses a ResNet50-based deep learning model enhanced with Convolutional Block Attention Module (CBAM), with Grad-CAM visualizations to help inspect image regions that influence a prediction.

The application includes a Python/FastAPI backend and a React/Vite frontend for uploading an image and viewing the model output.

> **Important:** This is an educational and research prototype, not a medical device. It must not be used to diagnose, rule out, or make treatment decisions about pancreatic disease.

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Model Overview](#model-overview)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Running the Application](#running-the-application)
- [Using the Web Application](#using-the-web-application)
- [API Endpoints](#api-endpoints)
- [Generated Outputs](#generated-outputs)
- [Troubleshooting](#troubleshooting)
- [Limitations and Medical Disclaimer](#limitations-and-medical-disclaimer)

## Features

- Binary image classification: **Normal** or **Pancreatic Tumour**.
- ResNet50-based image classification with CBAM attention modules.
- Image upload through a React web interface.
- Prediction confidence and class probability display.
- Grad-CAM heatmap and overlay generation, where the Grad-CAM dependency is available.
- Model information and health-check endpoints.
- Test-set evaluation metrics and visualizations.
- Automatic CPU/GPU device selection through PyTorch.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python, JavaScript (React) |
| Deep learning | PyTorch, Torchvision |
| Model | ResNet50 + CBAM |
| Image processing | Pillow, OpenCV, NumPy |
| Explainability | Grad-CAM |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, React Router |
| Charts and UI | Recharts, Framer Motion, Lucide React |
| Evaluation | Scikit-learn, Pandas, Matplotlib |

## Model Overview

The classification pipeline is:

1. Load an input image.
2. Convert it to RGB.
3. Resize and center-crop it to the model input dimensions.
4. Normalize the image using ImageNet mean and standard deviation.
5. Pass the image through the ResNet50 + CBAM model.
6. Apply softmax to obtain class probabilities.
7. Return the predicted class and confidence.
8. Generate a Grad-CAM visualization when requested.

### Class mapping

| Class index | Class label |
|---:|---|
| 0 | Normal |
| 1 | Pancreatic Tumour |

### Image preprocessing

- Input size: `224 × 224`
- Evaluation: resize, center crop, tensor conversion, and ImageNet normalization.
- Training: random resized crop, horizontal flip, rotation, mild color jitter, tensor conversion, and normalization.

Training augmentation is not applied to validation or test images.

## Project Structure

```text
pancreatic_detection/
│
├── app/
│   ├── backend/
│   │   ├── main.py              # FastAPI application and API routes
│   │   ├── model_loader.py      # Loads and caches the trained model
│   │   ├── predictor.py         # Image preprocessing and prediction
│   │   ├── gradcam.py           # Grad-CAM heatmap generation
│   │   └── schemas.py            # API response schemas
│   │
│   └── frontend/
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       └── src/
│           └── src/
│               ├── App.jsx
│               ├── main.jsx
│               ├── routes.jsx
│               ├── api.js
│               ├── components/
│               ├── context/
│               ├── hooks/
│               ├── pages/
│               ├── services/
│               └── styles/
│
├── src/
│   ├── cbam.py                   # CBAM attention module
│   ├── dataset.py                # Dataset loading and splitting
│   ├── models.py                 # ResNet50 + CBAM model definition
│   ├── preprocessing.py          # Image transformations
│   ├── train.py                  # Model training
│   └── evaluate.py               # Test-set evaluation
│
├── notebooks/
│   └── 01_dataset_analysis.ipynb
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── training_curves.png
│   └── metrics/
│       ├── confusion_matrix.csv
│       ├── test_metrics.json
│       └── test_predictions.csv
│
├── requirements.txt
├── test_gpu.py
├── run_backend.bat
├── run_frontend.bat
└── start.bat
```

**Note:** The supplied ZIP does not include the original dataset or a trained checkpoint. You must add the dataset and train the model before using prediction functionality.

## Dataset Preparation

The dataset loader expects the following structure at the project root:

```text
dataset/
├── train/
│   ├── Normal/
│   │   ├── image1.jpg
│   │   └── ...
│   └── Pancreatic Tumour/
│       ├── image1.jpg
│       └── ...
│
└── test/
    ├── Normal/
    │   ├── image1.jpg
    │   └── ...
    └── Pancreatic Tumour/
        ├── image1.jpg
        └── ...
```

The class-folder resolver also accepts common spelling variations, including `Pancreatic Tumor` and `Pancreatic Cancer`.

Supported image formats include JPG, JPEG, PNG, BMP, TIF, and TIFF.

The training script creates a **stratified 80/20 training/validation split** from the `dataset/train` directory. The separate `dataset/test` directory is reserved for final evaluation.

Keep test images separate from training and validation. Where patient identifiers are available, split data at patient level to reduce the risk of data leakage.

## Installation

### Prerequisites

- Windows, Linux, or macOS
- Python 3.10 or a compatible Python version supported by the installed PyTorch release
- Node.js and npm
- Git (optional)
- A CUDA-compatible NVIDIA GPU is recommended for faster training; CPU execution is supported but may be slower.

### 1. Clone or extract the project

```bash
git clone <YOUR_REPOSITORY_URL>
cd pancreatic_detection
```

Alternatively, extract the ZIP file and open a terminal in the extracted project root.

### 2. Create a Python virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU-enabled training, install a PyTorch and Torchvision build compatible with your installed CUDA version. Follow the official PyTorch installation selector: https://pytorch.org/get-started/locally/

### 4. Verify GPU availability

```bash
python test_gpu.py
```

The project selects CUDA when available and otherwise uses CPU.

### 5. Install frontend dependencies

```bash
cd app/frontend
npm install
cd ../..
```

## Training the Model

Before training, confirm that the dataset is in the expected `dataset/train` and `dataset/test` structure.

From the project root, with the Python virtual environment activated:

```bash
python src/train.py
```

The training script is configured with:

- Epochs: 20
- Batch size: 16
- Learning rate: `0.0001`
- Weight decay: `0.0001`
- Early-stopping patience: 5
- Stratified validation split: 20% of the training directory
- Class-weighted cross-entropy loss
- Mixed precision when supported on CUDA

The best checkpoint is saved to:

```text
models/
├── best_resnet50.pth
└── best_model.pth
```

The training script also saves training and validation loss/accuracy curves under `results/figures/`.

**Important:** The `models/` directory and checkpoint files are generated by training; they are not included in the supplied ZIP. The backend cannot perform predictions until a compatible checkpoint has been generated and placed in the expected location.

## Evaluating the Model

After training, run:

```bash
python src/evaluate.py
```

Evaluation uses the separate test dataset and reports:

- Accuracy
- Precision
- Recall
- Sensitivity
- Specificity
- F1-score
- ROC-AUC

Evaluation outputs are saved under:

```text
results/
├── figures/
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── metrics/
    ├── test_metrics.json
    ├── confusion_matrix.csv
    └── test_predictions.csv
```

Use metrics produced by a reproducible evaluation on your actual held-out test set. Do not interpret example or previously generated metric files as proof of clinical performance.

## Running the Application

Run commands from the project root.

### Option 1: Start the backend and frontend separately

**Terminal 1 — Backend**

Activate the Python virtual environment, then run:

```bash
run_backend.bat
```

The FastAPI backend is configured for:

```text
http://localhost:8003
```

You can also start it manually:

```bash
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8003
```

Interactive API documentation:

```text
http://localhost:8003/docs
```

**Terminal 2 — Frontend**

```bash
run_frontend.bat
```

The Vite development server is configured for:

```text
http://localhost:5173
```

If the batch file does not work in your environment, run:

```bash
cd app/frontend
npm install
npm run dev
```

### Option 2: Start both

From the project root:

```bat
start.bat
```

This uses the supplied Windows batch launcher to start the backend and frontend.

## Using the Web Application

1. Make sure the trained checkpoint exists in the `models/` directory.
2. Start the backend and frontend.
3. Open `http://localhost:5173` in your browser.
4. Navigate to the image detection page.
5. Upload a supported JPG or PNG image.
6. Run the analysis.
7. Review the predicted class, confidence, and class probabilities.
8. Open the Grad-CAM visualization, if available, to inspect the model's highlighted image regions.

The prediction endpoint accepts JPG and PNG uploads. The backend returns a service-unavailable response if the trained model cannot be loaded.

## API Endpoints

The backend base URL is `http://localhost:8003`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Basic API status and project information |
| GET | `/health` | Backend and model health information |
| GET | `/model-info` | Model name, device, class mapping, and input size |
| POST | `/predict` | Predict the class of an uploaded image |
| POST | `/gradcam` | Generate Grad-CAM visualization for an uploaded image |
| GET | `/metrics` | Retrieve saved test evaluation metrics |

### Example: Prediction request

```bash
curl -X POST "http://localhost:8003/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

The prediction response contains fields similar to:

```json
{
  "prediction": "Normal",
  "predicted_class": "Normal",
  "confidence": 0.95,
  "probabilities": {
    "Normal": 0.95,
    "Pancreatic Tumour": 0.05
  },
  "normal_probability": 0.95,
  "pancreatic_tumour_probability": 0.05,
  "model_name": "ResNet50 + CBAM",
  "device": "cuda"
}
```

The values above are illustrative only; actual results depend on the trained checkpoint and uploaded image.

### Example: Grad-CAM request

```bash
curl -X POST "http://localhost:8003/gradcam" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

The response includes the predicted class, confidence, and base64-encoded original, heatmap, and overlay images.

## Generated Outputs

| Output | Purpose |
|---|---|
| `models/best_resnet50.pth` | Best saved training checkpoint |
| `models/best_model.pth` | App-ready checkpoint copy |
| `results/figures/training_curves.png` | Training and validation loss/accuracy |
| `results/figures/confusion_matrix.png` | Test-set classification confusion matrix |
| `results/figures/roc_curve.png` | Test-set ROC curve |
| `results/metrics/test_metrics.json` | Test-set performance metrics |
| `results/metrics/confusion_matrix.csv` | Confusion matrix values |
| `results/metrics/test_predictions.csv` | Per-image test predictions and probabilities |

## Troubleshooting

### Model not found

**Message:** `Trained model not available. Train the model first.`

- Confirm that training completed successfully.
- Check that `models/best_model.pth` or `models/best_resnet50.pth` exists.
- Run the backend from the project root.

### Dataset folder not found

- Confirm that the dataset is under the project root.
- Check that both `dataset/train` and `dataset/test` exist.
- Confirm that each split contains both class folders and image files.

### CUDA is unavailable

- Check your NVIDIA driver and CUDA-compatible PyTorch installation.
- Run `python test_gpu.py`.
- The project can run on CPU, although training and inference may be slower.

### Frontend cannot connect to the backend

- Confirm that the backend is running on port `8003`.
- Check the frontend API configuration in `app/frontend/src/src/api.js` and related service files.
- Check the browser console and backend terminal for errors.
- Confirm that the frontend is using the correct backend base URL.

### Unsupported image type

The prediction and Grad-CAM endpoints accept JPG and PNG content types. Try uploading a valid `.jpg`, `.jpeg`, or `.png` image.

## Limitations and Medical Disclaimer

- This project is a research and educational prototype, not a clinically validated diagnostic tool.
- Performance depends on dataset quality, class balance, image acquisition conditions, and the representativeness of the training and test data.
- A high test score alone does not establish clinical reliability or generalization to other hospitals, scanners, or patient populations.
- Data leakage, duplicate images, and non-independent patient samples can inflate evaluation results.
- Grad-CAM is an interpretability aid; highlighted regions do not prove that the model has identified a tumour or a clinically meaningful feature.
- Predictions must not replace professional medical evaluation, radiology review, laboratory testing, or clinical judgment.

**For research and educational use only. Not intended for diagnosis or treatment.**

---

## Acknowledgements

This project uses open-source libraries and deep learning tools including PyTorch, Torchvision, FastAPI, React, Vite, OpenCV, and Grad-CAM.

## License

No license file was included in the supplied project archive. Add a `LICENSE` file before publicly distributing the repository, and ensure that dataset and model weights are shared only when their terms permit it.
