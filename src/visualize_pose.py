import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path to your saved keypoints file
DATA_PATH = "keypoints/sample_skater.npy"

# Load the (frames, 99) array
data = np.load(DATA_PATH)
print(f"Loaded data shape: {data.shape}")

# MediaPipe Pose defines 33 landmarks per person
NUM_LANDMARKS = 33

# Reshape into (frames, 33, 3)
data = data.reshape(-1, NUM_LANDMARKS, 3)

# Define landmark connections (simplified stick figure)
CONNECTIONS = [
    (11, 13), (13, 15),   # Left arm
    (12, 14), (14, 16),   # Right arm
    (11, 12),             # Shoulders
    (23, 24),             # Hips
    (11, 23), (12, 24),   # Torso
    (23, 25), (25, 27),   # Left leg
    (24, 26), (26, 28),   # Right leg
]

def plot_frame(ax, landmarks, connections):
    ax.clear()
    xs, ys, zs = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
    ax.scatter(xs, ys, zs, c="blue", s=10)
    for (i, j) in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], "k-", lw=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # flip Y for better visual
    ax.set_zlim(-0.5, 0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("MediaPipe Pose Reconstruction", fontsize=12)

def animate(data, connections):
    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    for frame_idx in range(0, len(data), 3):  # skip frames for speed
        plot_frame(ax, data[frame_idx], connections)
        plt.pause(0.03)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    animate(data, CONNECTIONS)
