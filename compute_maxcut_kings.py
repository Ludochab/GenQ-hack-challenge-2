import json
import networkx as nx
from kings_graph import king_graph, remove_random_nodes, add_random_edge_weights_sum
from OtherMaxCut import maxcut_goemans_williamson
import math
import random
import numpy as np

M = 8          # number of lines in the grid
N = 8         # number of columns in the grid
MODE = "k"      # "k" to remove exactly K vertices; "p" for probability P
k_min = max(1, math.ceil(0.25 * M * N))
k_max = max(k_min, math.floor(0.50 * M * N))
K = random.randint(k_min, k_max)
P = 0.15        # used if MODE == "p" (must be in [0,1])
SEED = None       # integer for reproducibility; set to None for pure randomness
NUM_SAMPLES = 100000  # number of random graphs to generate and solve
results = []

for i in range(NUM_SAMPLES):
    if i % 1000 == 0:
        print(f"Sample {i}/{NUM_SAMPLES}")
    G = king_graph(M, N)
    H = remove_random_nodes(G, mode=MODE, k=(K if MODE=="k" else None), p=(P if MODE=="p" else None), seed=SEED)
    add_random_edge_weights_sum(H, total=100.0, seed=SEED, integer=False, decimals=3)
    W = nx.to_numpy_array(H, weight="weight")
    val, x, (S, VS), meta = maxcut_goemans_williamson(W, R=64, rng=None, verbose=False)
    results.append({"sample": i, "cut_value": val})

bin_size = 0.2  # Choose your bin size (e.g., 5 units)
cut_values = [r["cut_value"] for r in results]

# Discretize and count
discrete_counts = {}
for val in cut_values:
    # Round to nearest bin_size
    discrete_val = bin_size * round(val / bin_size)
    discrete_counts[discrete_val] = discrete_counts.get(discrete_val, 0) + 1

# Prepare for JSON
binned_data = [
    {"discrete_value": float(discrete_val), "count": count}
    for discrete_val, count in sorted(discrete_counts.items())
]

with open("classical_kings.json", "w") as f:
    json.dump(binned_data, f, indent=2)