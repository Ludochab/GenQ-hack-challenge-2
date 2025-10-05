import networkx as nx
import numpy as np

def bitstring_to_partition(bitstring):
    S = {i for i, b in enumerate(bitstring) if b == "1"}
    return S

def cut_value(G, S):
    value = 0
    for u, v in G.edges(data=True):
        w = G[u][v]
        if (u in S and v not in S) or (v in S and u not in S):
            value += w
    return value

def hybrid_max_cut(G, S, max_iter=100):
    improved = True
    it = 1
    while improved and it < max_iter:
        it += 1
        improved = False
        for v in G.nodes():
            if v in S:
                new_S = S - {v}
            else:
                new_S = S + {v}
            if cut_value(G, new_S) > cut_value(G, S):
                S = new_S
                improved = True
    return S


mtl_Matrix = np.load("mtlFile.npy")
S = "1011101001110010110100"
hybrid_max_cut(mtl_Matrix ,S)