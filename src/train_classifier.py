"""Train a simple spin vs glide classifier from pose keypoints."""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from feature_engineering import summarise_clip


def load_dataset(
    annotations_path: str,
    keypoint_root: str,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load labelled clips and convert them into feature vectors."""

    annotations = pd.read_csv(annotations_path)
    if not {"clip_id", "label", "keypoints"}.issubset(annotations.columns):
        raise ValueError(
            "annotations CSV must contain clip_id, label, keypoints columns"
        )

    features: List[np.ndarray] = []
    feature_names: List[str] = []
    labels: List[str] = []

    for _, row in annotations.iterrows():
        keypoint_path = row["keypoints"]
        if not os.path.isabs(keypoint_path):
            keypoint_path = os.path.join(keypoint_root, keypoint_path)
        if not os.path.exists(keypoint_path):
            raise FileNotFoundError(
                f"Missing keypoint file for clip {row['clip_id']}: {keypoint_path}"
            )
        keypoints = np.load(keypoint_path)
        clip_features, feature_names = summarise_clip(keypoints, fps=fps)
        features.append(clip_features)
        labels.append(row["label"])

    if not features:
        raise ValueError("No training samples found. Provide at least one row")

    feature_matrix = np.vstack(features)
    return feature_matrix, np.asarray(labels), feature_names


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train and persist a RandomForest classifier."""

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes (e.g., spin vs glide)")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y) > 1 else None,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred)
    print("\n=== VALIDATION REPORT ===")
    print(report)

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": feature_names}, model_path)
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations.csv",
        help="CSV file with clip_id,label,keypoints columns",
    )
    parser.add_argument(
        "--keypoint-root",
        type=str,
        default="keypoints",
        help="Directory containing .npy keypoint files",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used to compute velocities",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/spin_glide_classifier.joblib",
        help="Where to persist the trained classifier",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples for validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X, y, feature_names = load_dataset(args.annotations, args.keypoint_root, args.fps)
    train_classifier(
        X,
        y,
        feature_names,
        args.model_path,
        test_size=args.test_size,
    )
