import random
import networkx as nx
import matplotlib.pyplot as plt
import math
import random as rng

M = 14          # number of lines in the grid
N = 14          # number of columns in the grid
MODE = "k"      # "k" to remove exactly K vertices; "p" for probability P
k_min = max(1, math.ceil(0.25 * M * N))
k_max = max(k_min, math.floor(0.50 * M * N))
K = rng.randint(k_min, k_max)
P = 0.15        # used if MODE == "p" (must be in [0,1])
SEED = 42       # integer for reproducibility; set to None for pure randomness

def king_graph(m: int, n: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(m):
        for j in range(n):
            G.add_node((i, j))
    for i in range(m):
        for j in range(n):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < m and 0 <= jj < n:
                        G.add_edge((i, j), (ii, jj))
    return G

def remove_random_nodes(G: nx.Graph, *, mode: str, k: int | None, p: float | None, seed: int | None):
    rng = random.Random() 
    nodes = list(G.nodes)

    if mode == "k":
        removed = set(rng.sample(nodes, k))
    elif mode == "p":
        removed = {v for v in nodes if rng.random() < p}
    else:
        raise ValueError("mode should be 'k' or 'p'.")

    H = G.copy()
    H.remove_nodes_from(removed)
    return H

def draw_king_graph(H: nx.Graph, title="Final graph"):
    pos = {v: (v[1], -v[0]) for v in H.nodes()}
    plt.figure(figsize=(6, 6))
    nx.draw(
        H, pos,
        node_size=120, linewidths=0.5,
        with_labels=False
    )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    G = king_graph(M, N)
    H = remove_random_nodes(G, mode=MODE, k=(K if MODE=="k" else None),
                                     p=(P if MODE=="p" else None), seed=SEED)

    draw_king_graph(H, title=f"Final graph ({H.number_of_nodes()} vertices)")