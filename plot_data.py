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
plt.xlabel("MaxCut sum (discretized)")
plt.ylabel("Fraction of instances")
plt.title("Normalized Distribution of MaxCut Sums")
plt.tight_layout()
plt.show()