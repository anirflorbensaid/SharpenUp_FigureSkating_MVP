"""Feature engineering utilities for figure skating movement analysis.

This module provides helpers to derive biomechanical features such as joint
angles, torso tilt, and joint velocities from Mediapipe Pose keypoints.  The
resulting time series can be fed into classical ML models or deep learning
architectures for skill classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Mediapipe Pose landmark indices (33 total)
POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


@dataclass
class TimeSeriesFeatures:
    """Collection of per-frame feature series and their names."""

    data: np.ndarray
    feature_names: List[str]

    def summary_statistics(self) -> Tuple[np.ndarray, List[str]]:
        """Compute summary statistics for each feature time series.

        Returns
        -------
        stats : np.ndarray
            Flattened array with statistics per feature.
        stat_names : list[str]
            Human readable names for each statistic.
        """

        stats: List[float] = []
        names: List[str] = []
        for i, name in enumerate(self.feature_names):
            series = self.data[:, i]
            stats.extend(
                [
                    float(np.nanmean(series)),
                    float(np.nanstd(series)),
                    float(np.nanmin(series)),
                    float(np.nanmax(series)),
                    float(np.nanmax(series) - np.nanmin(series)),
                ]
            )
            names.extend(
                [
                    f"{name}_mean",
                    f"{name}_std",
                    f"{name}_min",
                    f"{name}_max",
                    f"{name}_range",
                ]
            )
        return np.asarray(stats, dtype=float), names


def _reshape_keypoints(keypoints: np.ndarray) -> np.ndarray:
    if keypoints.ndim != 2 or keypoints.shape[1] % 3 != 0:
        raise ValueError(
            "Expected keypoints with shape (frames, landmarks * 3) containing"
            " x/y/z coordinates for each landmark."
        )
    num_landmarks = keypoints.shape[1] // 3
    if num_landmarks != 33:
        raise ValueError(
            f"Expected 33 landmarks from Mediapipe Pose, got {num_landmarks}."
        )
    return keypoints.reshape(keypoints.shape[0], num_landmarks, 3)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two 3D vectors."""

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    cos_theta = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the interior angle ABC (in degrees)."""

    return _angle_between(a - b, c - b)


def _torso_tilt(shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
    """Angle between torso vector and the vertical axis.

    A perfectly upright torso pointing towards the camera should yield an angle
    close to 0°, while leaning forward/backward increases the angle.
    """

    torso_vec = shoulder_center - hip_center
    if np.linalg.norm(torso_vec) == 0:
        return np.nan
    vertical_axis = np.array([0.0, -1.0, 0.0])
    return _angle_between(torso_vec, vertical_axis)


def _hip_rotation(left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """Orientation of the hip line relative to the horizontal axis."""

    hip_vec = right_hip - left_hip
    horizontal_axis = np.array([1.0, 0.0, 0.0])
    return _angle_between(hip_vec, horizontal_axis)


def _compute_speed(positions: np.ndarray, fps: float) -> np.ndarray:
    """Compute instantaneous speed of a position time series."""

    diffs = np.diff(positions, axis=0, prepend=positions[[0]])
    speeds = np.linalg.norm(diffs, axis=1) * fps
    return speeds


def _compute_angular_velocity(angle_series: np.ndarray, fps: float) -> np.ndarray:
    """Angular velocity in degrees per second."""

    derivatives = np.gradient(angle_series, edge_order=2)
    return derivatives * fps


def extract_time_series(
    keypoints: np.ndarray,
    fps: float = 30.0,
) -> TimeSeriesFeatures:
    """Extract per-frame biomechanical features.

    Parameters
    ----------
    keypoints : np.ndarray
        Array with shape (frames, 33 * 3) containing Mediapipe Pose keypoints.
    fps : float, optional
        Frames-per-second of the source video, used for velocity estimates.

    Returns
    -------
    TimeSeriesFeatures
        Matrix of shape (frames, num_features) and their corresponding names.
    """

    coords = _reshape_keypoints(keypoints)

    left_shoulder = coords[:, POSE_LANDMARKS["left_shoulder"], :]
    right_shoulder = coords[:, POSE_LANDMARKS["right_shoulder"], :]
    left_elbow = coords[:, POSE_LANDMARKS["left_elbow"], :]
    right_elbow = coords[:, POSE_LANDMARKS["right_elbow"], :]
    left_wrist = coords[:, POSE_LANDMARKS["left_wrist"], :]
    right_wrist = coords[:, POSE_LANDMARKS["right_wrist"], :]
    left_hip = coords[:, POSE_LANDMARKS["left_hip"], :]
    right_hip = coords[:, POSE_LANDMARKS["right_hip"], :]
    left_knee = coords[:, POSE_LANDMARKS["left_knee"], :]
    right_knee = coords[:, POSE_LANDMARKS["right_knee"], :]
    left_ankle = coords[:, POSE_LANDMARKS["left_ankle"], :]
    right_ankle = coords[:, POSE_LANDMARKS["right_ankle"], :]
    nose = coords[:, POSE_LANDMARKS["nose"], :]

    frame_features: List[List[float]] = []
    left_knee_angles: List[float] = []
    right_knee_angles: List[float] = []
    left_elbow_angles: List[float] = []
    right_elbow_angles: List[float] = []
    torso_tilts: List[float] = []
    hip_rotations: List[float] = []

    for i in range(coords.shape[0]):
        ls, rs = left_shoulder[i], right_shoulder[i]
        le, re = left_elbow[i], right_elbow[i]
        lw, rw = left_wrist[i], right_wrist[i]
        lh, rh = left_hip[i], right_hip[i]
        lk, rk = left_knee[i], right_knee[i]
        la, ra = left_ankle[i], right_ankle[i]

        left_knee_angle = _joint_angle(lh, lk, la)
        right_knee_angle = _joint_angle(rh, rk, ra)
        left_elbow_angle = _joint_angle(ls, le, lw)
        right_elbow_angle = _joint_angle(rs, re, rw)
        shoulder_center = (ls + rs) / 2.0
        hip_center = (lh + rh) / 2.0
        torso_tilt = _torso_tilt(shoulder_center, hip_center)
        hip_rotation = _hip_rotation(lh, rh)

        left_knee_angles.append(left_knee_angle)
        right_knee_angles.append(right_knee_angle)
        left_elbow_angles.append(left_elbow_angle)
        right_elbow_angles.append(right_elbow_angle)
        torso_tilts.append(torso_tilt)
        hip_rotations.append(hip_rotation)

        frame_features.append(
            [
                left_knee_angle,
                right_knee_angle,
                left_elbow_angle,
                right_elbow_angle,
                torso_tilt,
                hip_rotation,
            ]
        )

    frame_features = np.asarray(frame_features, dtype=float)

    # Compute kinematic features derived from positions
    hip_center = (left_hip + right_hip) / 2.0
    nose_speed = _compute_speed(nose, fps)
    hip_speed = _compute_speed(hip_center, fps)
    ankle_speed = _compute_speed((left_ankle + right_ankle) / 2.0, fps)

    angles = {
        "left_knee_angle": np.asarray(left_knee_angles),
        "right_knee_angle": np.asarray(right_knee_angles),
        "left_elbow_angle": np.asarray(left_elbow_angles),
        "right_elbow_angle": np.asarray(right_elbow_angles),
        "torso_tilt": np.asarray(torso_tilts),
        "hip_rotation": np.asarray(hip_rotations),
    }

    angular_velocities = {
        f"{name}_angular_velocity": _compute_angular_velocity(series, fps)
        for name, series in angles.items()
    }

    additional_series = np.column_stack(
        [
            nose_speed,
            hip_speed,
            ankle_speed,
        ]
    )

    series_matrix = np.column_stack(
        [
            frame_features,
            additional_series,
            np.column_stack(list(angular_velocities.values())),
        ]
    )

    feature_names = [
        "left_knee_angle",
        "right_knee_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "torso_tilt",
        "hip_rotation",
        "nose_speed",
        "hip_speed",
        "ankle_speed",
    ] + list(angular_velocities.keys())

    return TimeSeriesFeatures(series_matrix, feature_names)


def summarise_clip(
    keypoints: np.ndarray,
    fps: float = 30.0,
) -> Tuple[np.ndarray, List[str]]:
    """Convenience wrapper returning summary stats for a clip."""

    time_series = extract_time_series(keypoints, fps=fps)
    return time_series.summary_statistics()
