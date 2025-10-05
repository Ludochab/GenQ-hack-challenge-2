import random
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import folium
import random
import json

gdf = gpd.read_file("data/CARTO_SER_ELE_TEL_AERIEN.shp")
print(gdf.head())
print(gdf.columns)
print(gdf.crs)



# basic stats
print("Count:", len(gdf))
print(gdf.geometry.geom_type.value_counts())

# Calculate centroids in projected CRS (EPSG:2950) for accuracy
gdf_proj = gdf.to_crs(epsg=2950)
gdf['centroid'] = gdf_proj.geometry.centroid

# Reproject main GeoDataFrame and centroids to WGS84 (lat/lon) for mapping
gdf = gdf.to_crs(epsg=4326)
gdf['centroid'] = gdf['centroid'].to_crs(epsg=4326)


def round_coord(coord, decimals=4):
    return (round(coord[0], decimals), round(coord[1], decimals))

G = nx.Graph()

for geom in gdf.geometry:
    coords = [round_coord(pt) for pt in geom.coords]
    for i in range(len(coords) - 1):
        G.add_node(coords[i])
        G.add_node(coords[i+1])
        G.add_edge(coords[i], coords[i+1])

# Draw the graph
pos = {node: node for node in G.nodes()}
nx.draw(G, pos, node_size=1)
plt.savefig("debug2.png")

def merge_close_nodes(nodes, threshold):
    clusters = []
    node_to_centroid = {}
    for node in nodes:
        found = False
        for cluster in clusters:
            # Euclidean distance
            if np.linalg.norm(np.array(node) - np.array(cluster)) < threshold:
                node_to_centroid[node] = cluster
                found = True
                break
        if not found:
            clusters.append(node)
            node_to_centroid[node] = node
    return node_to_centroid

def simplify_graph(G):
    G_simple = nx.Graph()
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for node in nodes:
        if G.degree(node) != 2:
            G_simple.add_node(node)
    # For each node with degree != 2, traverse its neighbors to find other degree != 2 nodes
    for node in G_simple.nodes():
        for neighbor in G.neighbors(node):
            path = [node, neighbor]
            current = neighbor
            prev = node
            while G.degree(current) == 2:
                next_nodes = [n for n in G.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                prev, current = current, next_nodes[0]
                path.append(current)
            if current != node and current in G_simple.nodes():
                G_simple.add_edge(node, current)
    return G_simple



# Clustering nodes to find where transformers post are
nodes = list(G.nodes())
threshold = 0.012  # adjust as needed for your CRS
node_map = merge_close_nodes(nodes, threshold)

# Build new graph with merged nodes
G_merged = nx.Graph()
for u, v in G.edges():
    u_new = node_map[u]
    v_new = node_map[v]
    if u_new != v_new:
        G_merged.add_edge(u_new, v_new)

# Draw the simplified graph
pos_simplified = {node: node for node in G.nodes()}
plt.figure()
nx.draw(G_merged, pos_simplified, node_size=20)
plt.savefig("debug_simplified.png")

if nx.is_connected(G_merged):
    print("The graph is connected.")
else:
    print("The graph is NOT connected.")

G_simplified = simplify_graph(G_merged)

seed = 42

# Use spring layout for even node spacing
spring_pos = nx.spring_layout(G_simplified, seed=seed)  # seed for reproducibility

# Draw the spring layout graph
plt.figure()
nx.draw(G_simplified, spring_pos, node_size=20)
plt.savefig("debug_spring.png")

# Folium visualization
center = gdf.geometry.union_all().centroid
m = folium.Map(location=[center.y, center.x], zoom_start=12)

for u, v in G_simplified.edges():
    folium.PolyLine(locations=[(u[1], u[0]), (v[1], v[0])], color='blue', weight=2).add_to(m)

for node in G_simplified.nodes():
    folium.CircleMarker(
        location=[node[1], node[0]],
        radius=3,
        color='red',
        fill=True,
        fill_color='red'
    ).add_to(m)

m.save("map.html")



mtl_matrix = nx.to_numpy_array(G_simplified)

# add random weights to the adjacencies edges
mtl_matrix = mtl_matrix * np.random.uniform(0.5, 1.5, mtl_matrix.shape)

np.save("mtl_matrix.npy", mtl_matrix)

# Create a list of nodes in the order of positions
node_list = list(G_simplified.nodes())
positions = [list(map(float, spring_pos[n])) for n in node_list]

# Prepare the data dictionary with indices
data = {
    "indices": list(range(len(node_list))),
    "nodes": [str(n) for n in node_list],  # or just node_list if they are serializable
    "positions": positions
}

# Save to JSON
with open("input.json", "w") as f:
    json.dump(data, f)