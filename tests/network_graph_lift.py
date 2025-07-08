# %% v1

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from math import pi, cos, sin

# Your lift data
data = {
    'Drug_A': ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Drug A', 'Drug B', 'Drug C', 'Drug D', 'Drug W', 'Drug X', 'Drug Y', 'Drug Z'],
    'Drug 1': [1, 2.604, 2.71, 2.71, 0.189, 0.131, 1, 0.097, 0.055, 0.219, 0.209, 0.099],
    'Drug 2': [2.604, 1, 2.786, 2.786, 0.161, 0.101, 1, 0.099, 0.057, 0.178, 0.169, 0.101],
    'Drug 3': [2.71, 2.786, 1, 1, 0.158, 0.164, 1, 1, 1, 0.051, 0.1, 1],
    'Drug 4': [2.71, 2.786, 1, 1, 0.107, 0.056, 1, 1, 1, 1, 0.102, 1],
    'Drug A': [0.189, 0.161, 0.158, 0.107, 1, 2.768, 2.899, 2.899, 0.118, 0.161, 0.2, 0.105],
    'Drug B': [0.131, 0.101, 0.164, 0.056, 2.768, 1, 3.012, 3.012, 0.061, 0.126, 0.166, 1],
    'Drug C': [1, 1, 1, 1, 2.899, 3.012, 1, 1, 1, 1, 0.061, 1],
    'Drug D': [0.097, 0.099, 1, 1, 2.899, 3.012, 1, 1, 1, 1, 0.049, 1],
    'Drug W': [0.055, 0.057, 1, 1, 0.118, 0.061, 1, 1, 1, 2.786, 2.755, 1],
    'Drug X': [0.219, 0.178, 0.051, 1, 0.161, 0.126, 1, 1, 2.786, 1, 2.594, 2.786],
    'Drug Y': [0.209, 0.169, 0.1, 0.102, 0.2, 0.166, 0.061, 0.049, 2.755, 2.594, 1, 2.755],
    'Drug Z': [0.099, 0.101, 1, 1, 0.105, 1, 1, 1, 1, 2.786, 2.755, 1]
}

# Create DataFrame
df = pd.DataFrame(data)
df = df.set_index('Drug_A')

# Parameters for customization - adjusted for lift values
lift_threshold = 1.1  # Only show associations where lift > 1.1 (significant positive association)
min_node_size = 15
max_node_size = 60
min_edge_width = 1
max_edge_width = 8

# Create directed graph
G = nx.DiGraph()
nodes = df.index.tolist()
G.add_nodes_from(nodes)

# Add edges with lift as weight - only for lift > threshold
edges_data = []
for source in df.index:
    for target in df.columns:
        lift = df.loc[source, target]
        if lift > lift_threshold and source != target:
            G.add_edge(source, target, weight=lift)
            edges_data.append((source, target, lift))

# Calculate node importance (sum of significant outgoing and incoming lifts)
node_importance = {}
for node in nodes:
    outgoing = df.loc[node, (df.columns != node) & (df.loc[node] > lift_threshold)].sum()
    incoming = df.loc[(df.index != node) & (df.loc[:, node] > lift_threshold), node].sum()
    node_importance[node] = outgoing + incoming

# Normalize node sizes
max_importance = max(node_importance.values()) if node_importance.values() else 1
min_importance = min(node_importance.values()) if node_importance.values() else 0
node_sizes = {}
for node in nodes:
    if max_importance > min_importance:
        normalized = (node_importance[node] - min_importance) / (max_importance - min_importance)
    else:
        normalized = 0.5
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
    
    # Normalize edge width based on lift values
    if len(edges_data) > 1:
        max_weight = max([e[2] for e in edges_data])
        min_weight = min([e[2] for e in edges_data])
        normalized_width = (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
    else:
        normalized_width = 0.5
    
    edge_width = min_edge_width + normalized_width * (max_edge_width - min_edge_width)
    
    # Color based on lift level - adjusted for lift interpretation
    # if weight > 2.5:
    #     color = 'rgba(255, 0, 0, 0.8)'  # Red for very high lift
    #     color_name = 'Very High'
    # elif weight > 2.0:
    #     color = 'rgba(255, 165, 0, 0.7)'  # Orange for high lift
    #     color_name = 'High'
    # elif weight > 1.5:
    #     color = 'rgba(255, 255, 0, 0.7)'  # Yellow for medium lift
    #     color_name = 'Medium'
    # else:
    #     color = 'rgba(0, 128, 255, 0.6)'  # Blue for low but significant lift
    #     color_name = 'Low Significant'

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
        # hovertext=f'{source} → {target}<br>Lift: {weight:.3f}<br>Level: {color_name}<br>{"Strong positive association" if weight > 1 else "Negative association"}',
        hovertext=f'{source} → {target}<br>Lift: {weight:.3f}<br>{"Strong positive association" if weight > 1 else "Negative association"}',
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
    
    # Create detailed hover info for lift values
    outgoing_edges = [(target, df.loc[node, target]) for target in df.columns if df.loc[node, target] > lift_threshold and target != node]
    incoming_edges = [(source, df.loc[source, node]) for source in df.index if df.loc[source, node] > lift_threshold and source != node]
    
    hover_text = f'<b>{node}</b><br>'
    hover_text += f'Lift Importance Score: {node_importance[node]:.3f}<br>'
    hover_text += f'Node Size: {node_sizes[node]:.1f}<br><br>'
    
    if outgoing_edges:
        hover_text += '<b>Outgoing Associations (Lift > 1.1):</b><br>'
        for target, lift_val in sorted(outgoing_edges, key=lambda x: x[1], reverse=True)[:5]:
            association_strength = "Very Strong" if lift_val > 2.5 else "Strong" if lift_val > 2.0 else "Moderate" if lift_val > 1.5 else "Weak Positive"
            hover_text += f'  → {target}: {lift_val:.3f} ({association_strength})<br>'
    
    if incoming_edges:
        hover_text += '<b>Incoming Associations (Lift > 1.1):</b><br>'
        for source, lift_val in sorted(incoming_edges, key=lambda x: x[1], reverse=True)[:5]:
            association_strength = "Very Strong" if lift_val > 2.5 else "Strong" if lift_val > 2.0 else "Moderate" if lift_val > 1.5 else "Weak Positive"
            hover_text += f'  ← {source}: {lift_val:.3f} ({association_strength})<br>'
    
    node_info.append(hover_text)
    node_text.append(node)
    node_size_list.append(node_sizes[node])
    
    # Color nodes based on their importance
    if max_importance > min_importance:
        importance_normalized = (node_importance[node] - min_importance) / (max_importance - min_importance)
    else:
        importance_normalized = 0.5
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
            title="Node Importance<br>(Lift Sum)",
            # titleside="right",
            tickmode="linear",
            thickness=15
        ),
        line=dict(width=2, color='black')
    )
)

# Create the figure
fig = go.Figure(data=[node_trace] + edge_traces,
               layout=go.Layout(
                title=dict(
                    text='Interactive Drug Association Network - Lift Values<br><sub>Node size = Total lift importance | Edge thickness = Lift value | Only showing Lift > 1.1</sub>',
                    x=0.5,
                    font=dict(size=16)
                ),
                # titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=60),
                annotations=[ dict(
                    text="Lift > 1: Positive association | Lift = 1: Independence | Lift < 1: Negative association<br>Hover over nodes and edges for details<br>Edge thickness shows lift strength",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="gray", size=10)
                ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=1000,
                height=800
            ))

fig.write_html('test.html', include_plotlyjs="cdn")

# Display the interactive plot
# fig.show()

# Print summary statistics
print("Network Summary (Lift Values):")
print(f"Lift threshold: {lift_threshold} (showing only positive associations)")
print(f"Number of nodes: {len(nodes)}")
print(f"Number of significant edges (lift > {lift_threshold}): {len(edges_data)}")
print(f"Node size range: {min_node_size} - {max_node_size}")
print(f"Edge width range: {min_edge_width} - {max_edge_width}")

print("\nTop 10 Most Important Nodes (by total lift):")
sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
for i, (node, importance) in enumerate(sorted_nodes[:10]):
    print(f"{i+1}. {node}: {importance:.3f}")

print("\nTop 10 Strongest Associations (by lift):")
sorted_edges = sorted(edges_data, key=lambda x: x[2], reverse=True)
for i, (source, target, weight) in enumerate(sorted_edges[:10]):
    association_strength = "Very Strong" if weight > 2.5 else "Strong" if weight > 2.0 else "Moderate" if weight > 1.5 else "Weak Positive"
    print(f"{i+1}. {source} → {target}: {weight:.3f} ({association_strength})")

print("\nLift Interpretation:")
print("Lift > 2.5: Very Strong positive association (Red)")
print("Lift 2.0-2.5: Strong positive association (Orange)")
print("Lift 1.5-2.0: Moderate positive association (Yellow)")
print("Lift 1.1-1.5: Weak positive association (Blue)")
print("Lift = 1: Independence (not shown)")
print("Lift < 1: Negative association (not shown due to threshold)")
