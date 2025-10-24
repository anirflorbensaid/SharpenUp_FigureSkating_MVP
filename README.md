# SharpenUp

Minimal prototype for detecting figure skating elements from video using pose
estimation and classical ML models.

---

## Pipeline Overview

1. **Keypoint extraction** – `main.py` uses MediaPipe Pose to convert a video
   into per-frame 3D joint coordinates stored in `keypoints/<clip>.npy`.
2. **Feature engineering** – `src/feature_engineering.py` converts raw
   keypoints into biomechanical time series (joint angles, torso tilt, joint
   velocities) and provides summary statistics for each clip.
3. **Trick classification** – `src/train_classifier.py` reads a labelled
   dataset (e.g., "spin" vs "glide") and trains a Random Forest classifier on
   the engineered features.

## Preparing Training Data

1. Record or collect short clips for the tricks you care about (start with two
   such as **spin** and **glide**).
2. Run `python main.py` for each clip to store the pose keypoints as
   `keypoints/<clip_name>.npy`.
3. Create `data/annotations.csv` with at least 20 labelled examples. The file
   should look like:

   ```csv
   clip_id,label,keypoints
   clip_spin_01,spin,clip_spin_01.npy
   clip_glide_01,glide,clip_glide_01.npy
   ...
   ```

   A template is provided in `data/annotations_example.csv`. Update the
   filenames and labels to match your actual recordings.

## Training the Classifier

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then train the model (adjust paths and FPS if needed):

```bash
python -m src.train_classifier \
    --annotations data/annotations.csv \
    --keypoint-root keypoints \
    --fps 30 \
    --model-path models/spin_glide_classifier.joblib
```

The script prints a validation report and stores the trained model together
with the feature list. The resulting `.joblib` file can be loaded later to
classify new clips using the same feature extractor.
