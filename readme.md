# Architectural Fingerprinting & Eagle-Eye Change Detection

A dual-module computer vision suite for unique building identification and multi-temporal change analysis.

## 1. Architectural Fingerprinting
Generates a structured, unique "ID" for buildings based on permanent physical characteristics, designed to be perspective and lighting invariant.
- **Structural Analysis:** Floor counting and window grid mapping (e.g., Floor 1: 8 windows).
- **Geometric Signature:** Width-to-height structural aspect ratio ($R = w/h$).
- **Keypoint Descriptors:** Signatures derived from permanent rooflines and structural corners.
- **Fuzzy Matching:** Robust logic to handle partial occlusions (trees, shadows) and varying angles.

## 2. Eagle-Eye (Image Difference)
A 2D processing pipeline to identify minute physical changes while ignoring environmental noise.
- **Module 1 (Registration):** Image alignment using SIFT/ORB features and Homography mapping.
- **Module 2 (Filtering):** Illumination invariant pre-processing using LBP (Local Binary Patterns) and Gradient Analysis to neutralize shadows.
- **Module 3 (Detection):** Deterministic dual-path detection for ground textures (holes/stains/debris) and discrete physical objects using classical image processing.
- **Module 4 (Reporting):** Visual output with color-coded markers:
    - 🟢 **Added** | 🔴 **Removed** | 🟡 **Moved** | 🔵 **Ground Change**.

## Tech Stack (Planned)
- **Language:** Python
- **Core Libraries:** OpenCV (Hough Transforms, Feature Matching)
- **Image Processing:** Scikit-Image, SciPy, NumPy
- **No-AI Policy:** No torch, keras, transformers, or ultralytics. Detection is based on OpenCV, NumPy, SciPy, and Euclidean geometry.

## Installation & Setup
1. Ensure Python 3.9+ is installed.
2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To test the deterministic Module 1 -> Module 2 pipeline:
```bash
python run_pipeline_demo.py
```

## Success Criteria
- Stability across different times of day (10:00 AM vs 4:00 PM).
- High sensitivity to small-scale objects (50cm x 50cm).
- Minimal false positives from moving shadows or weather.
