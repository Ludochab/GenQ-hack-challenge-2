# Check that graph coordinates are reasonable for QuEra device

import json
import numpy as np

def scale_and_snap_positions(positions, min_dist=4, max_abs=37.5):
    positions = np.array(positions)
    # Center
    center = positions.mean(axis=0)
    positions -= center

    # Scale to fit within [-max_abs, max_abs]
    max_dim = np.max(np.abs(positions))
    if max_dim > 0:
        scale = max_abs / max_dim
    else:
        scale = 1.0
    positions *= scale

    # Snap y to multiples of 4
    positions[:, 1] = 4 * np.round(positions[:, 1] / 4)

    # Check minimum distance and rescale if needed
    def min_pairwise_dist(pos):
        from scipy.spatial.distance import pdist
        return np.min(pdist(pos))

    while min_pairwise_dist(positions) < min_dist:
        positions *= min_dist / min_pairwise_dist(positions)

    return positions.tolist()

with open("input.json", "r") as f:
    data = json.load(f)

positions = data["positions"]
positions = scale_and_snap_positions(positions)

# check if positions are valid
for i, pos in enumerate(positions):
    if abs(pos[0]) > 37.5 or abs(pos[1]) > 37.5:
        raise ValueError(f"Position {i} = {pos} is out of bounds.")
    if pos[1] % 4 != 0:
        raise ValueError(f"Position {i} = {pos} does not have y coordinate multiple of 4.")
    for j in range(i):
        dist = np.linalg.norm(np.array(pos) - np.array(positions[j]))
        if dist < 4:
            raise ValueError(f"Positions {i} and {j} are too close: {dist} < 4.")