# %% v1

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from math import pi, cos, sin

# Your data
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

plt.title('Drug Association Network Graph\n(Confidence Threshold: {})'.format(confidence_threshold), 
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



# %% v2

# Your data
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

# Parameters for customization
confidence_threshold = 0.02  # Lower threshold to show more connections
min_node_size = 15
max_node_size = 60
min_edge_width = 1
max_edge_width = 8

# Create directed graph
G = nx.DiGraph()
nodes = df.index.tolist()
G.add_nodes_from(nodes)

# Add edges with confidence as weight
edges_data = []
for source in df.index:
    for target in df.columns:
        confidence = df.loc[source, target]
        if confidence > confidence_threshold and source != target:
            G.add_edge(source, target, weight=confidence)
            edges_data.append((source, target, confidence))

# Calculate node importance (sum of outgoing and incoming confidences)
node_importance = {}
for node in nodes:
    outgoing = df.loc[node, df.columns != node].sum()
    incoming = df.loc[df.index != node, node].sum()
    node_importance[node] = outgoing + incoming

# Normalize node sizes
max_importance = max(node_importance.values())
min_importance = min(node_importance.values())
node_sizes = {}
for node in nodes:
    normalized = (node_importance[node] - min_importance) / (max_importance - min_importance)
    node_sizes[node] = min_node_size + normalized * (max_node_size - min_node_size)

# Create circular layout
def circular_layout(nodes, radius=2):
    pos = {}
    angle_step = 2 * pi / len(nodes)
    for i, node in enumerate(nodes):
        angle = i * angle_step
        pos[node] = (radius * cos(angle), radius * sin(angle))
    return pos

pos = circular_layout(nodes)

# Prepare edge traces
edge_traces = []
edge_info = []

# Sort edges by weight to draw thicker ones on top
edges_data.sort(key=lambda x: x[2])

for source, target, weight in edges_data:
    x0, y0 = pos[source]
    x1, y1 = pos[target]
    
    # Calculate arrow position (90% along the line)
    arrow_x = x0 + 0.9 * (x1 - x0)
    arrow_y = y0 + 0.9 * (y1 - y0)
    
    # Normalize edge width
    if len(edges_data) > 1:
        max_weight = max([e[2] for e in edges_data])
        min_weight = min([e[2] for e in edges_data])
        normalized_width = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
    else:
        normalized_width = 0.5
    
    edge_width = min_edge_width + normalized_width * (max_edge_width - min_edge_width)
    
    # Color based on confidence level
    # if weight > 0.8:
    #     color = 'rgba(255, 0, 0, 0.7)'  # Red for very high
    #     color_name = 'Very High'
    # elif weight > 0.5:
    #     color = 'rgba(255, 165, 0, 0.7)'  # Orange for high
    #     color_name = 'High'
    # elif weight > 0.2:
    #     color = 'rgba(255, 255, 0, 0.7)'  # Yellow for medium
    #     color_name = 'Medium'
    # else:
    #     color = 'rgba(128, 128, 128, 0.5)'  # Gray for low
    #     color_name = 'Low'

    # Create a gradient from light to dark blue
    intensity = weight  # Use weight directly for intensity
    color = f'rgba(0, 100, 200, {0.3 + intensity * 0.7})'  # Darker = higher confidence    
    
    # Edge line
    edge_trace = go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode='lines',
        line=dict(width=edge_width, color=color),
        hoverinfo='none',
        showlegend=False
    )
    edge_traces.append(edge_trace)
    
    # Arrow head
    arrow_trace = go.Scatter(
        x=[arrow_x],
        y=[arrow_y],
        mode='markers',
        marker=dict(
            symbol='triangle-right',
            size=edge_width + 5,
            color=color,
            line=dict(width=1, color='black')
        ),
        hoverinfo='text',
        hovertext=f'{source} → {target}<br>Confidence: {weight:.4f}<br>Level: {color_name}',
        showlegend=False
    )
    edge_traces.append(arrow_trace)

# Prepare node trace
node_x = []
node_y = []
node_text = []
node_info = []
node_size_list = []
node_color_list = []

for node in nodes:
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    
    # Create detailed hover info
    outgoing_edges = [(target, df.loc[node, target]) for target in df.columns if df.loc[node, target] > confidence_threshold and target != node]
    incoming_edges = [(source, df.loc[source, node]) for source in df.index if df.loc[source, node] > confidence_threshold and source != node]
    
    hover_text = f'<b>{node}</b><br>'
    hover_text += f'Importance Score: {node_importance[node]:.3f}<br>'
    hover_text += f'Node Size: {node_sizes[node]:.1f}<br><br>'
    
    if outgoing_edges:
        hover_text += '<b>Outgoing Associations:</b><br>'
        for target, conf in sorted(outgoing_edges, key=lambda x: x[1], reverse=True)[:5]: # Top 5 associations
            hover_text += f'  → {target}: {conf:.3f}<br>'
    
    if incoming_edges:
        hover_text += '<b>Incoming Associations:</b><br>'
        for source, conf in sorted(incoming_edges, key=lambda x: x[1], reverse=True)[:5]: # Top 5 associations
            hover_text += f'  ← {source}: {conf:.3f}<br>'
    
    node_info.append(hover_text)
    node_text.append(node)
    node_size_list.append(node_sizes[node])
    
    # Color nodes based on their importance
    importance_normalized = (node_importance[node] - min_importance) / (max_importance - min_importance)
    node_color_list.append(importance_normalized)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    hoverinfo='text',
    hovertext=node_info,
    text=node_text,
    textposition="middle center",
    textfont=dict(size=10, color='black'),
    marker=dict(
        size=node_size_list,
        color=node_color_list,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title="Node Importance",
            # titleside="right",
            tickmode="linear",
            thickness=20
        ),
        line=dict(width=2, color='black')
    )
)

# Create the figure
fig = go.Figure(data=[node_trace] + edge_traces,
               layout=go.Layout(
                title=dict(
                    text='Interactive Drug Association Network<br><sub>Node size = Total confidence importance | Edge thickness = Confidence value</sub>',
                    x=0.5,
                    font=dict(size=16)
                ),
                # titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=60),
                annotations=[ dict(
                    text="Hover over nodes and edges for details<br>Node size reflects total association strength<br>Edge thickness shows confidence level",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="gray", size=10)
                ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=1200,
                height=800
            ))

fig.write_html('test.html', include_plotlyjs="cdn")

# Display the interactive plot
# fig.show()

# Print summary statistics
print("Network Summary:")
print(f"Confidence threshold: {confidence_threshold}")
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges_data)}")
print(f"Node size range: {min_node_size} - {max_node_size}")
print(f"Edge width range: {min_edge_width} - {max_edge_width}")

print("\nTop 10 Most Important Nodes (by total confidence):")
sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
for i, (node, importance) in enumerate(sorted_nodes[:10]):
    print(f"{i+1}. {node}: {importance:.3f}")

print("\nTop 10 Strongest Associations:")
sorted_edges = sorted(edges_data, key=lambda x: x[2], reverse=True)
for i, (source, target, weight) in enumerate(sorted_edges[:10]):
    print(f"{i+1}. {source} → {target}: {weight:.4f}")

# Create a second visualization with adjustable threshold
def create_interactive_network(threshold=0.02):
    """Function to create network with different threshold"""
    return fig  # You can call this function with different threshold values

print("\nTo adjust the confidence threshold, change the 'confidence_threshold' variable at the top of the script.")
print(f"Current threshold: {confidence_threshold} (showing {len(edges_data)} edges)")
