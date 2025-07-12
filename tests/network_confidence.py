# %% v1

## Dependencies
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from math import pi, cos, sin

data = {
    'Drug_A': ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Drug A', 'Drug B', 'Drug C', 'Drug D', 'Drug W', 'Drug X', 'Drug Y', 'Drug Z'],
    'Drug 1': [0, 0.935, 0.1491, 0.1463, 0.065, 0.0434, 0, 0.0054, 0.0027, 0.0786, 0.0759, 0.0054],
    'Drug 2': [0.961, 0, 0.1532, 0.1504, 0.0557, 0.0334, 0, 0.0056, 0.0028, 0.0641, 0.0613, 0.0056],
    'Drug 3': [1, 1, 0, 0, 0.0545, 0.0545, 0, 0, 0, 0.0182, 0.0364, 0],
    'Drug 4': [1, 1, 0, 0, 0.037, 0.0185, 0, 0, 0, 0, 0.037, 0],
    'Drug A': [0.0696, 0.058, 0.0087, 0.0058, 0, 0.9188, 0.1304, 0.1623, 0.0058, 0.058, 0.0725, 0.0058],
    'Drug B': [0.0482, 0.0361, 0.009, 0.003, 0.9548, 0, 0.1355, 0.1687, 0.003, 0.0452, 0.0602, 0],
    'Drug C': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0.0222, 0],
    'Drug D': [0.0357, 0.0357, 0, 0, 1, 1, 0, 0, 0, 0, 0.0179, 0],
    'Drug W': [0.0204, 0.0204, 0, 0, 0.0408, 0.0204, 0, 0, 0, 1, 1, 0],
    'Drug X': [0.0808, 0.0641, 0.0028, 0, 0.0557, 0.0418, 0, 0, 0.1365, 0, 0.9415, 0.1532],
    'Drug Y': [0.0771, 0.0606, 0.0055, 0.0055, 0.0689, 0.0551, 0.0028, 0.0028, 0.135, 0.9311, 0, 0.1515],
    'Drug Z': [0.0364, 0.0364, 0, 0, 0.0364, 0, 0, 0, 0, 1, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)
df = df.set_index('Drug_A')

# Create directed graph
G = nx.DiGraph()

# Add nodes
nodes = df.index.tolist()
G.add_nodes_from(nodes)

# Add edges with confidence as weight
# Set a threshold to filter out weak associations
confidence_threshold = 0.1  # You can adjust this threshold

for source in df.index:
    for target in df.columns:
        confidence = df.loc[source, target]
        if confidence > confidence_threshold and source != target:
            G.add_edge(source, target, weight=confidence)

# Create the visualization
plt.figure(figsize=(8, 6))

# Position nodes using spring layout
pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=2000, alpha=0.8)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Draw edges with varying thickness based on confidence
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

# Normalize weights for edge thickness
max_weight = max(weights) if weights else 1
min_weight = min(weights) if weights else 0
normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 5 + 0.5 for w in weights]

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=edges, width=normalized_weights, 
                      alpha=0.6, edge_color='gray', arrows=True, 
                      arrowsize=20, arrowstyle='->')

plt.title('Not Correct\n\nDrug Association Network Graph\n(Confidence Threshold: {})'.format(confidence_threshold), 
          fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Print some network statistics
print("Network Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Print top associations
print("\nTop 10 Strongest Associations:")
edge_weights = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
edge_weights.sort(key=lambda x: x[2], reverse=True)

for i, (source, target, weight) in enumerate(edge_weights[:10]):
    print(f"{i+1}. {source} -> {target}: {weight:.4f}")

# Alternative visualization with different layout
plt.figure(figsize=(8, 6))

# Use circular layout for better visibility
pos2 = nx.circular_layout(G)

# Draw with color-coded edges based on confidence levels
edge_colors = []
for u, v in G.edges():
    weight = G[u][v]['weight']
    if weight > 0.8:
        edge_colors.append('red')      # Very high confidence
    elif weight > 0.5:
        edge_colors.append('orange')   # High confidence
    elif weight > 0.2:
        edge_colors.append('yellow')   # Medium confidence
    else:
        edge_colors.append('lightgray') # Low confidence

nx.draw_networkx_nodes(G, pos2, node_color='lightcoral', 
                      node_size=2500, alpha=0.8)
nx.draw_networkx_labels(G, pos2, font_size=10, font_weight='bold')
nx.draw_networkx_edges(G, pos2, edge_color=edge_colors, width=2, 
                      alpha=0.7, arrows=True, arrowsize=20)

plt.title('Drug Association Network Graph - Circular Layout\n(Edge Colors: Red=Very High, Orange=High, Yellow=Medium, Gray=Low)', 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Create a heatmap for comparison
plt.figure(figsize=(12, 6))

# Create heatmap
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.astype(float), annot=True, cmap='YlOrRd', 
            center=0, square=True, fmt='.3f', cbar_kws={'label': 'Confidence'})
plt.title('Drug Association Confidence Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Target Drug')
plt.ylabel('Source Drug')
plt.tight_layout()
plt.show()
