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
SEED = None       # integer for reproducibility; set to None for pure randomness

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
    rng = random.Random(seed) 
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

def add_random_edge_weights_sum(
    H: nx.Graph,
    *,
    total: float = 100.0,
    seed: int | None = None,
    integer: bool = False,
    decimals: int | None = 3,
    min_int_weight: int = 0
):
    if H.number_of_edges() == 0:
        return

    r = random.Random(seed)
    edges = list(H.edges())
    m = len(edges)

    if not integer:
        raw = []
        for _ in range(m):
            u = r.random()
            u = u if u > 1e-12 else 1e-12  
            raw.append(-math.log(u))
        s = sum(raw)
        weights = [total * x / s for x in raw]

        if decimals is not None:
            weights = [round(w, decimals) for w in weights]
            fix = round(total - sum(weights), decimals)
            weights[-1] = round(weights[-1] + fix, decimals)
    else:
        S = int(round(total))
        if min_int_weight < 0:
            raise ValueError("min_int_weight doit être ≥ 0.")
        if min_int_weight * m > S:
            raise ValueError(f"Impossible: m*min_int_weight={m*min_int_weight} > total={S}.")
        S_remaining = S - min_int_weight * m
        if m == 1:
            parts = [S_remaining]
        else:
            cuts = sorted(r.randint(0, S_remaining) for _ in range(m - 1))
            parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, m - 1)] + [S_remaining - cuts[-1]]
        weights = [p + min_int_weight for p in parts]

    for (u, v), w in zip(edges, weights):
        H[u][v]["weight"] = w

def draw_edge_weights(H: nx.Graph, decimals: int = 2):
    if H.number_of_edges() == 0:
        raise ValueError("No edges to label.")
    pos = {v: (v[1], -v[0]) for v in H.nodes()}
    
    plt.figure(figsize=(6, 6))
    nx.draw(H, pos, node_size=100, linewidths=0.5, with_labels=False)
    labels = {(u, v): f"{H[u][v]['weight']:.{decimals}f}" for u, v in H.edges()}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=labels, font_size=7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Final graph (edge weights)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    G = king_graph(M, N)
    H = remove_random_nodes(G, mode=MODE, k=(K if MODE=="k" else None),
                                     p=(P if MODE=="p" else None), seed=SEED)

    draw_king_graph(H, title=f"Final graph ({H.number_of_nodes()} vertices)")

    add_random_edge_weights_sum(H, total=100.0, seed=SEED, integer=False, decimals=3)
    s = sum(nx.get_edge_attributes(H, "weight").values())
    print(f"Sum of edge weights = {s}")

    draw_edge_weights(H, decimals=2)
