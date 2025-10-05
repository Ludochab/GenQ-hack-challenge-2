import json
import matplotlib.pyplot as plt

# Load data
with open("classical_kings_smol.json") as f:
    data = json.load(f)

# Extract values and counts
x = [entry["discrete_value"] for entry in data]
y_counts = [entry["count"] for entry in data]
total = sum(y_counts)
y = [count / total for count in y_counts]

# Plot as a line
plt.figure(figsize=(10, 5))
plt.plot(x, y, color="blue")
plt.xlabel("Cut edges weight sum")
plt.ylabel("Probability density")
plt.title("Distribution of MaxCut Sums for Classical Kings Graphs")
plt.tight_layout()
plt.show()

# Load data
with open("maxcut_vs_mis_cut_kings.json") as f:
    data = json.load(f)

# Extract MaxCut data
x_maxcut = [entry["discrete_value"] for entry in data["maxcut"]]
y_counts_maxcut = [entry["count"] for entry in data["maxcut"]]
total_maxcut = sum(y_counts_maxcut)
y_maxcut = [count / total_maxcut for count in y_counts_maxcut]

# Extract MIS-cut data
x_mis = [entry["discrete_value"] for entry in data["mis_cut"]]
y_counts_mis = [entry["count"] for entry in data["mis_cut"]]
total_mis = sum(y_counts_mis)
y_mis = [count / total_mis for count in y_counts_mis]

# Plot both as lines
plt.figure(figsize=(10, 5))
plt.plot(x_maxcut, y_maxcut, color="blue", label="MaxCut (Goemans–Williamson)")
plt.plot(x_mis, y_mis, color="orange", label="MIS-based Cut")
plt.xlabel("Cut edges weight sum")
plt.ylabel("Probability density")
plt.title("Normalized Distribution: MaxCut vs MIS-based Cut")
plt.legend()
plt.tight_layout()
plt.show()

# Load MTL graph results
with open("mtl_maxcut_vs_mis_cut.json") as f:
    mtl_data = json.load(f)
mtl_maxcut = mtl_data["maxcut_value"]
mtl_mis_cut = mtl_data["mis_cut_value"]

# Plot both as lines
plt.figure(figsize=(10, 5))
plt.plot(x_maxcut, y_maxcut, color="blue", label="MaxCut (Goemans–Williamson)")
plt.plot(x_mis, y_mis, color="orange", label="MIS-based Cut")

# Add vertical lines for MTL graph
plt.axvline(mtl_maxcut, color="blue", linestyle="--", label="MTL MaxCut")
plt.axvline(mtl_mis_cut, color="orange", linestyle="--", label="MTL MIS-based Cut")
plt.xlabel("Cut edges weight sum")
plt.ylabel("Probability density")
plt.title("Normalized Distribution: MaxCut vs MIS-based Cut")
plt.legend()
plt.tight_layout()
plt.show()