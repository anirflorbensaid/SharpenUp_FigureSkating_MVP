import os
import cv2
import numpy as np
import mediapipe as mp

# === CONFIGURATION ===
VIDEO_PATH = "data/videos/sample_skater.mp4"
OUT_DIR = "keypoints"

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose

def extract_keypoints(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{filename}.npy")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    keypoints = []
    frames = 0
    kept = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # If a body is detected, save the (x,y,z) of each landmark
        if results.pose_landmarks:
            pts = []
            for lm in results.pose_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            keypoints.append(pts)
            kept += 1

    cap.release()

    keypoints = np.array(keypoints)
    np.save(out_path, keypoints)

    print("\n=== DONE ===")
    print(f"Video:       {video_path}")
    print(f"Frames read: {frames}")
    print(f"Frames kept: {kept}")
    print(f"Saved to:    {out_path}")
    print(f"Array shape: {keypoints.shape}\n")

if __name__ == "__main__":
    extract_keypoints(VIDEO_PATH, OUT_DIR)
