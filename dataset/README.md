# Dataset — Traffic Sign Recognition

> ⚠️ **PLACEHOLDER DATA** — The images in this folder were synthetically generated (simple colored boxes with text). They are NOT real traffic sign images. Replace them with a real dataset before actual use or presenting the project.

---

## Current Structure (Placeholder)

```
dataset/
├── Stop/               → 80 synthetic images
├── No Entry/           → 80 synthetic images
├── Turn Left/          → 80 synthetic images
├── Turn Right/         → 80 synthetic images
└── Speed Limit 40/     → 80 synthetic images
```

---

## How to Use a Real Dataset

### Option 1 — Download from Kaggle

1. Search for **"Indian Traffic Sign Dataset"** on [Kaggle](https://www.kaggle.com/datasets)
2. Download and extract the dataset
3. Organize images into the folder structure below

### Option 2 — Collect Your Own Images

Take photos of real traffic signs OR download images from Google Images.

---

## Required Folder Structure

For the model to train correctly, your dataset must look exactly like this:

```
dataset/
├── Stop/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── No Entry/
│   ├── img001.jpg
│   └── ...
├── Turn Left/
│   └── ...
├── Turn Right/
│   └── ...
└── Speed Limit 40/
    └── ...
```

**Rules:**
- Each folder name becomes a class label (must match exactly)
- Supported image formats: `.jpg`, `.jpeg`, `.png`
- Recommended minimum: **50+ images per class** (more is better)
- Images do NOT need to be the same size — `train.py` resizes them all to 64×64 automatically

---

## How to Add a New Traffic Sign Class

1. Create a new folder inside `dataset/` with the sign name (e.g., `dataset/Give Way/`)
2. Add images of that sign into the folder
3. Delete the old trained model: `model/traffic_sign_model.h5`
4. Re-run training: `python train.py`
5. The new class will automatically be detected and added!

---

## Tips for Better Accuracy

- Use **varied images**: different lighting, angles, distances, backgrounds
- Avoid blurry or very dark images
- More images = better accuracy (aim for 100–200+ per class)
- Make sure images actually show the sign clearly
