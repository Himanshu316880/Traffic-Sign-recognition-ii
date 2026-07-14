# 🚦 Traffic Sign Recognition — CNN + Streamlit

A beginner-friendly **Traffic Sign Recognition** web app built with Python, TensorFlow/Keras, and Streamlit.

Upload a traffic sign image → the app tells you what sign it is and how confident it is.

> 🎓 Built as a college mini project. Every line of code is explained so you can confidently present it in a viva or interview.

---

## 📸 Demo

Run locally with:
```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```
traffic-sign-recognition/
├── dataset/            → Images for each traffic sign class (one subfolder per class)
│   └── README.md       → Instructions for replacing placeholder data with real images
├── model/              → Saved trained model + class labels
│   ├── traffic_sign_model.h5   → The trained CNN model
│   ├── labels.json             → Maps index numbers to class names
│   └── training_curves.png     → Accuracy & loss graphs from training
├── uploads/            → Temporary folder for images during local testing (gitignored)
├── generate_dataset.py → Creates synthetic placeholder images (run this if no real dataset)
├── train.py            → Builds and trains the CNN model
├── predict.py          → Predicts the class of a single image (CLI + importable function)
├── app.py              → Streamlit web application
├── requirements.txt    → Python package dependencies
├── Procfile            → For Heroku-style deployment
├── setup.sh            → Streamlit server config for Heroku-style deployment
├── .gitignore          → Files/folders to exclude from Git
└── README.md           → This file
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖼️ Dataset

> ⚠️ **Important:** The `dataset/` folder uses **synthetic placeholder images** (colored boxes with text) by default. These are only for testing the pipeline — NOT for real use.

### Option A — Use Placeholder Data (to test the pipeline)

```bash
python generate_dataset.py
```

This creates 80 fake images per class in `dataset/`.

### Option B — Use a Real Dataset

1. Download an **Indian Traffic Sign Dataset** from [Kaggle](https://www.kaggle.com/datasets) (search "Indian Traffic Sign Dataset")
2. Organize images into subfolders like this:

```
dataset/
├── Stop/
│   ├── img001.jpg
│   └── ...
├── No Entry/
├── Turn Left/
├── Turn Right/
└── Speed Limit 40/
```

See `dataset/README.md` for detailed instructions.

---

## 🧠 Training the Model

```bash
python train.py
```

This will:
- Load images from `dataset/`
- Resize and normalize all images to 64×64
- Apply data augmentation (rotation, zoom, brightness)
- Train a CNN for up to 30 epochs (stops early if it stops improving)
- Save the model to `model/traffic_sign_model.h5`
- Save class labels to `model/labels.json`
- Save training curves to `model/training_curves.png`
- Print a classification report (precision/recall per class)

**Training takes about 3–8 minutes** depending on your CPU.

---

## 🔍 Testing Prediction (Command Line)

After training, test a single image:

```bash
python predict.py dataset/Stop/Stop_001.jpg
```

Output:
```
Predicting class for: dataset/Stop/Stop_001.jpg
----------------------------------------
Predicted Sign : Stop
Confidence     : 98.72%
```

---

## 🌐 Running the Web App Locally

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

**What you'll see:**
1. Upload a traffic sign image (JPG/PNG)
2. The app displays the image
3. The CNN predicts the sign and shows confidence %

---

## 🧩 How It Works — Simple Explanation

```
User uploads image
        ↓
Image is resized to 64×64 pixels
        ↓
Pixel values normalized (0–255 → 0.0–1.0)
        ↓
Passed through the CNN:
    Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Softmax
        ↓
Model outputs probabilities for each class
  e.g. [Stop: 0.95, No Entry: 0.02, Turn Left: 0.01, ...]
        ↓
Highest probability = predicted class
```

### CNN Architecture (Simplified)

| Layer | What it does |
|-------|-------------|
| Conv2D (32 filters) | Detects basic features: edges, corners, colors |
| MaxPooling2D | Reduces image size, keeps important info |
| Conv2D (64 filters) | Detects more complex patterns |
| MaxPooling2D | Further reduces size |
| Flatten | Converts 2D feature map → 1D list of numbers |
| Dense (128 neurons) | Learns combinations of features |
| Dropout (0.5) | Randomly ignores neurons to prevent overfitting |
| Dense (softmax) | Outputs probability for each class |

---

## ➕ How to Add New Traffic Sign Classes

1. Create a new folder in `dataset/` with the sign name:
   ```
   dataset/Give Way/
   ```
2. Add at least 50 images of that sign into the folder
3. Delete the old model:
   ```bash
   del model\traffic_sign_model.h5
   ```
4. Retrain:
   ```bash
   python train.py
   ```
5. Done! The new class is automatically detected.

---

## ❗ Common Errors and Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `Model file not found` | Model hasn't been trained yet | Run `python train.py` |
| `Could not read image` | Invalid image format | Use JPG or PNG files only |
| `ModuleNotFoundError` | Missing library | Run `pip install -r requirements.txt` |
| `Port already in use` | Another app on port 8501 | Run `streamlit run app.py --server.port 8502` |
| `dataset/ not found` | Dataset missing | Run `python generate_dataset.py` or add real images |
| Streamlit page is blank | Browser cache issue | Hard refresh: Ctrl+Shift+R |
| Low accuracy | Too few/poor images | Use more varied real images |

---

## 🐙 GitHub & Deployment

### Push to GitHub

```bash
# Initialize git (already done if you cloned)
git init
git add .
git commit -m "Initial commit: traffic sign recognition app"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/traffic-sign-recognition.git
git push -u origin main
```

> ⚠️ **Important:** Make sure `model/traffic_sign_model.h5` and `model/labels.json` are committed — the deployed app needs them to make predictions.

> The `dataset/` folder is large. You don't need to push it — only the trained model needs to be in the repo for deployment.

### Deploy on Streamlit Community Cloud (Free)

1. Push your repo to GitHub (see above)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **"New app"**
5. Select your repo and set **Main file path** to `app.py`
6. Click **Deploy** — your app will be live in ~2 minutes!

> **Note:** `Procfile` and `setup.sh` are only needed if deploying to a **Heroku-style** host. Streamlit Community Cloud ignores them and runs `app.py` directly.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Programming language |
| TensorFlow / Keras | Build and train the CNN |
| OpenCV | Read and preprocess images |
| NumPy | Numerical operations |
| Matplotlib | Plot training curves |
| Streamlit | Build the web interface |
| Pillow (PIL) | Open images in Streamlit |

---

## 👨‍🎓 About This Project

This is a **beginner-level college mini project** demonstrating:
- Image classification using a Convolutional Neural Network (CNN)
- End-to-end ML pipeline: data → train → predict → deploy
- Streamlit for rapid web app development

The code is intentionally kept simple and well-commented so you can understand, modify, and explain every part of it.
