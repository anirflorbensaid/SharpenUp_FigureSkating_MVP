import cv2
import numpy as np
import mediapipe as mp
import joblib
import argparse
import os
from feature_engineering import summarise_clip

# === Parse command-line arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to input video")
parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
args = parser.parse_args()

VIDEO_PATH = args.input
MODEL_PATH = args.model
OUT_PATH = os.path.join("outputs", os.path.basename(VIDEO_PATH).replace(".mp4", "_labeled.mp4"))

# === Setup folders ===
os.makedirs("outputs", exist_ok=True)

# === Load trained model ===
model_data = joblib.load(MODEL_PATH)
if isinstance(model_data, dict) and "model" in model_data:
    model = model_data["model"]
else:
    model = model_data
print(f"✅ Loaded model from {MODEL_PATH}")

# === Setup Mediapipe Pose ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# === Open video ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width, height))

print(f"🎥 Processing {frames} frames from {VIDEO_PATH} ...")

frame_idx = 0
window = []  # store recent frames (each with 33 * 2 coordinates)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Extract x, y only (ignore z)
        landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark]).flatten()

        if len(landmarks) == 66:  # 33 keypoints × 2
            window.append(landmarks)
            if len(window) > 30:
                window.pop(0)

            # When we have a full 30-frame window
            if len(window) == 30:
                keypoints_window = np.array(window)  # (30, 66)

                # Expand to fake 3D since summarise_clip expects (frames, 99)
                num_landmarks = keypoints_window.shape[1] // 2
                keypoints_3d = np.zeros((keypoints_window.shape[0], num_landmarks * 3))
                keypoints_3d[:, 0::3] = keypoints_window[:, 0::2]  # x
                keypoints_3d[:, 1::3] = keypoints_window[:, 1::2]  # y

                # Compute engineered biomechanical features
                features, _ = summarise_clip(keypoints_3d, fps=fps)
                features = features.reshape(1, -1)

                try:
                    pred = model.predict(features)[0]
                    prob = np.max(model.predict_proba(features))
                    color = (0, 255, 0) if pred == "spin" else (255, 100, 0)
                    label = f"{pred.upper()} ({prob:.2f})"
                    cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                except Exception as e:
                    print(f"⚠️ Prediction error at frame {frame_idx}: {e}")

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
        )

    out.write(frame)

    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{frames} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Saved overlaid labeled video to {OUT_PATH}")
