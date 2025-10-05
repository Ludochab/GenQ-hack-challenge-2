import json
import networkx as nx
from kings_graph import king_graph, remove_random_nodes, add_random_edge_weights_sum
from OtherMaxCut import maxcut_goemans_williamson, maxcut_value_from_assignment
import math
import random
import numpy as np

M = 4
N = 4
MODE = "k"
k_min = max(1, math.ceil(0.05 * M * N))
k_max = max(k_min, math.floor(0.15 * M * N))
K = random.randint(k_min, k_max)
P = 0.15
SEED = None
NUM_SAMPLES = 10000
results = []

for i in range(NUM_SAMPLES):
    if i % 1000 == 0:
        print(f"Sample {i}/{NUM_SAMPLES}")
    G = king_graph(M, N)
    H = remove_random_nodes(G, mode=MODE, k=(K if MODE=="k" else None), p=(P if MODE=="p" else None), seed=SEED)
    add_random_edge_weights_sum(H, total=100.0, seed=SEED, integer=False, decimals=3)
    W = nx.to_numpy_array(H, weight="weight")
    
    # 1. Compute true MaxCut value
    val_maxcut, x_maxcut, (S, VS), meta = maxcut_goemans_williamson(W, R=64, rng=None, verbose=False)
    
    # 2. Compute MIS (using NetworkX's approximation)
    mis_nodes = nx.algorithms.approximation.maximum_independent_set(H)
    # Build assignment: +1 for MIS nodes, -1 for others
    node_list = list(H.nodes())
    x_mis = np.array([1.0 if node in mis_nodes else -1.0 for node in node_list])
    # 3. Compute MaxCut value for this assignment
    val_mis_cut = maxcut_value_from_assignment(W, x_mis)
    
    results.append({
        "sample": i,
        "maxcut_value": val_maxcut,
        "mis_cut_value": val_mis_cut
    })

# Optionally, bin and save as before
bin_size = 0.5
cut_values = [r["maxcut_value"] for r in results]
mis_cut_values = [r["mis_cut_value"] for r in results]

def bin_counts(values, bin_size):
    counts = {}
    for val in values:
        discrete_val = bin_size * round(val / bin_size)
        counts[discrete_val] = counts.get(discrete_val, 0) + 1
    return [
        {"discrete_value": float(discrete_val), "count": count}
        for discrete_val, count in sorted(counts.items())
    ]

binned_maxcut = bin_counts(cut_values, bin_size)
binned_mis_cut = bin_counts(mis_cut_values, bin_size)

with open("maxcut_vs_mis_cut_kings.json", "w") as f:
    json.dump({
        "maxcut": binned_maxcut,
        "mis_cut": binned_mis_cut
    }, f, indent=2)