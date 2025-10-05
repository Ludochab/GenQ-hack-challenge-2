import networkx as nx
import numpy as np

def bitstring_to_partition(bitstring):
    S = [i for i, b in enumerate(bitstring) if b == "1"]
    return S

def cut_value(G, S, weight="weight"):
    value = 0
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1)
        if ((u in S) and (v not in S)) or ((v in S) and (u not in S)):
            value += w
    return value

def hybrid_max_cut(G, bs, weight="weight", max_iter=100):
    S = bitstring_to_partition(bs)
    improved = True
    it = 1
    while improved and it < max_iter:
        it += 1
        improved = False
        for v in list(G.nodes()):
            if v in S:
                new_S = S
                new_S.remove(v)
            else:
                new_S = S
                new_S.append(v)
            if cut_value(G, new_S, weight="weight") > cut_value(G, S, weight="weight"):
                S = new_S
                improved = True
    return S


#mtl_Matrix = np.load("mtl_matrix.npy")
#mtl_G = nx.from_numpy_array(mtl_Matrix)
#bs = "1011101001110010110100"
#print(hybrid_max_cut(mtl_G ,bs))