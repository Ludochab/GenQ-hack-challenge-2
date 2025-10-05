import math
import numpy as np
import networkx as nx
from bloqade.analog import start
import matplotlib.pyplot as plt

# ---------- Utils ----------
def is_independent(G, S):
    S = set(S)
    return all(not (u in S and v in S) for u, v in G.edges())

def spring_embed_unit_disk(G, Rb_um, edge_margin=0.90, nonedge_margin=1.10, tries=200, seed=0):
    """
    Essaie de trouver un layout 2D + une échelle telle que:
      max distance arête <= edge_margin * Rb_um
      min distance non-arête >= nonedge_margin * Rb_um
    Renvoie: (coords_um_list, nodes_order)
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2:
        return [(0.0, 0.0)], nodes

    # Pré-calcul des paires
    E = set(map(lambda e: tuple(sorted(e)), G.edges()))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    nonE = [p for p in all_pairs if (nodes[p[0]], nodes[p[1]]) not in E]

    for t in range(tries):
        pos = nx.spring_layout(G, dim=2, seed=seed + t)  # positions ~O(1)
        P = np.array([pos[u] for u in nodes])            # shape (n,2)

        # Distances brutes
        def dist(i, j):
            d = P[i] - P[j]
            return float(np.hypot(d[0], d[1]))

        if E:
            dE_max = max(dist(i, j) for i, j in [(nodes.index(u), nodes.index(v)) for u, v in G.edges()])
        else:
            dE_max = 1e-9  # pas d'arêtes -> n'importe quel Rb fonctionne

        if nonE:
            dNE_min = min(dist(i, j) for i, j in nonE)
        else:
            dNE_min = float('inf')  # graphe complet

        # Condition de séparabilité géométrique
        if dE_max < dNE_min:  # possible de placer un Rb entre les deux
            # Échelle pour serrer les arêtes sous edge_margin * Rb
            s = (edge_margin * Rb_um) / dE_max if dE_max > 0 else 1.0
            if s * dNE_min >= nonedge_margin * Rb_um:
                coords_um = [(float(s * x), float(s * y)) for x, y in P]
                return coords_um, nodes
        # sinon on retente avec un autre seed

    # Échec : on renvoie quand même un layout rescalé grossièrement et on préviendra
    s = (edge_margin * Rb_um) / max(dE_max, 1e-9)
    coords_um = [(float(s * x), float(s * y)) for x, y in P]
    return coords_um, nodes  # mais le graphe réalisé pourra différer

def realized_unit_disk_edges(coords_um, Rb_um):
    n = len(coords_um)
    E = set()
    for i in range(n):
        xi, yi = coords_um[i]
        for j in range(i+1, n):
            xj, yj = coords_um[j]
            if math.hypot(xi - xj, yi - yj) <= Rb_um:
                E.add((i, j))
    return E

# ---------- Définis ton graphe ici ----------
# Exemples :
# G = nx.path_graph(8)
#G = nx.erdos_renyi_graph(10, 0.25, seed=1)
G = nx.cycle_graph(7)
#G = nx.grid_2d_graph(3, 4) ; G = nx.convert_node_labels_to_integers(G)

#G = nx.path_graph(8)  # <-- change ici
pos = nx.spring_layout(G, seed=42)  # calcule des positions 2D
nx.draw(G, pos, with_labels=True, node_size=600, font_size=10)
plt.show()

# ---------- Paramètres physiques / pulse identiques ----------
Rb_um = 8.0
ramp_us, plateau_us = 0.06, 1.6
Omega_max = 15.0
Delta_start, Delta_final = -30.0, "delta_f"
T = ramp_us + plateau_us + ramp_us

# ---------- Embedding générique ----------
coords_um, nodes = spring_embed_unit_disk(G, Rb_um, edge_margin=0.90, nonedge_margin=1.10, tries=300, seed=42)

# (Optionnel) Vérifier le graphe réellement induit par (coords_um, Rb_um)
E_real = realized_unit_disk_edges(coords_um, Rb_um)
# Construit E cible en indices 0..n-1
idx = {u: i for i, u in enumerate(nodes)}
E_target = set(tuple(sorted((idx[u], idx[v]))) for u, v in G.edges())
if E_real != E_target:
    diff_plus  = E_real - E_target
    diff_minus = E_target - E_real
    print("[AVERTISSEMENT] Le graphe unit-disk réalisé diffère du G demandé.")
    if diff_plus:
        print("  Arêtes en plus (géométrie):", diff_plus)
    if diff_minus:
        print("  Arêtes manquantes:", diff_minus)

# ---------- Programme analogique ----------
prog = (
    start
    .add_position(coords_um)  # <- pas add_positions !
    .rydberg
    .rabi.amplitude.uniform.piecewise_linear(
        values=[0.0, Omega_max, Omega_max, 0.0],
        durations=[ramp_us, plateau_us, ramp_us]
    )
    .phase.uniform.constant(value=0.0, duration=T)
    .detuning.uniform.piecewise_linear(
        values=[Delta_start, Delta_start, Delta_final, Delta_final],
        durations=[ramp_us, plateau_us/2, plateau_us/2 + ramp_us]
    )
)

# ---------- Sweep de la détuning finale ----------
finals = np.linspace(0.0, 80.0, 41)
batch = prog.batch_assign(delta_f=finals)
res = batch.bloqade.python().run(500)
rep = res.report()

# ---------- Analyse / remapping vers labels NetworkX ----------
dens = rep.rydberg_densities()        # (n_final, n_sites)
totals = dens.sum(axis=1).values
best_i = int(np.argmax(totals))
best_delta = finals[best_i]

p_site = dens.iloc[best_i].values
cand_nodes = {nodes[i] for i, p in enumerate(p_site) if p >= 0.5}

# "Réparation" simple pour obtenir un MIS maximal (casse les conflits locaux)
p_by_node = {nodes[i]: p_site[i] for i in range(len(nodes))}
for u, v in G.edges():
    if u in cand_nodes and v in cand_nodes:
        if p_by_node[u] >= p_by_node[v]:
            cand_nodes.discard(v)
        else:
            cand_nodes.discard(u)

# Bitstring brut et remappé en labels du graphe
bs = max(rep.counts()[best_i].items(), key=lambda kv: kv[1])[0]
labelled_bs = {nodes[k] for k, b in enumerate(bs) if b == "1"}

print(f"Best final detuning ≈ {best_delta:.1f} rad/µs")
print("Per-site excitation probabilities at best detuning:", np.round(p_site, 3))
print("Candidat (≥0.5) réparé en MIS maximal:", cand_nodes, "| valide:", is_independent(G, cand_nodes))
print("Bitstring le plus fréquent (remappé):", labelled_bs, "| brut:", bs)  # <-- parenthèse fermée