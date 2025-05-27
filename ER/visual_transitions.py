import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import json

import json
import subprocess
import os
import sys
from pathlib import Path


def json_to_dot(json_data):
    """Convert JSON graph representation to DOT format."""
    dot_content = ['digraph G {']

    # Graph styling
    dot_content.append('    node [shape=ellipse, style=filled, fillcolor=white];')
    dot_content.append('    edge [fontsize=10];')
    dot_content.append('    rankdir=TB;')

    # Add nodes
    if 'nodes' in json_data:
        for node in json_data['nodes']:
            node_id = node['id']
            label = node.get('label', node_id)

            # Node styling
            node_attrs = []
            if 'attributes' in node:
                for attr_name, attr_value in node['attributes'].items():
                    node_attrs.append(f'{attr_name}="{attr_value}"')

            node_attrs_str = ', '.join(['label="{}"'.format(label)] + node_attrs)
            dot_content.append(f'    {node_id} [{node_attrs_str}];')

    # Add edges
    if 'edges' in json_data:
        for edge in json_data['edges']:
            source = edge['source']
            target = edge['target']

            # Edge styling
            edge_attrs = []
            if 'label' in edge:
                edge_attrs.append(f'label="{edge["label"]}"')

            if 'attributes' in edge:
                for attr_name, attr_value in edge['attributes'].items():
                    edge_attrs.append(f'{attr_name}="{attr_value}"')

            edge_attrs_str = ', '.join(edge_attrs)
            if edge_attrs_str:
                dot_content.append(f'    {source} -> {target} [{edge_attrs_str}];')
            else:
                dot_content.append(f'    {source} -> {target};')

    dot_content.append('}')
    return '\n'.join(dot_content)


def render_dot(dot_content, output_format='png', output_path='graph'):
    """Render DOT content using Graphviz."""
    # Write DOT content to file
    with open(f'{output_path}.dot', 'w') as f:
        f.write(dot_content)

    # Run Graphviz dot command
    cmd = ['dot', f'-T{output_format}', f'{output_path}.dot', '-o', f'{output_path}.{output_format}']
    print(f"Executing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Graph rendered successfully to {output_path}.{output_format}")
    except subprocess.CalledProcessError as e:
        print(f"Error rendering graph: {e}")
    except FileNotFoundError:
        print("Error: Graphviz 'dot' command not found. Please ensure Graphviz is installed.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python json_to_dot.py <json_file> [output_format] [output_path]")
        sys.exit(1)

    json_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'png'
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'graph'

    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        dot_content = json_to_dot(json_data)

        # Save DOT content to file for reference
        with open(f'{output_path}.dot', 'w') as f:
            f.write(dot_content)

        render_dot(dot_content, output_format, output_path)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file}'")


if __name__ == "__main__":
    main()























# def save_transitions(transitions, filename="transitions.json"):
#     """
#     Save transitions dictionary to a JSON file.
#
#     Args:
#         transitions: Dictionary with keys (state, label) and values (next_state, probability)
#         filename: Output filename
#     """
#     # Convert tuple keys to string representation since JSON doesn't support tuple keys
#     serializable_transitions = {}
#     for (from_state, label), (to_state, prob) in transitions.items():
#         key = f"{from_state}|{label}"
#         serializable_transitions[key] = [to_state, prob]
#
#     # Save to file
#     with open(filename, 'w') as f:
#         json.dump(serializable_transitions, f, indent=2)
#
#     print(f"Transitions saved to {filename}")
#
#
# def load_transitions(filename="transitions.json"):
#     """
#     Load transitions dictionary from a JSON file.
#
#     Args:
#         filename: Input filename
#
#     Returns:
#         Transitions dictionary
#     """
#     with open(filename, 'r') as f:
#         serializable_transitions = json.load(f)
#
#     # Convert string keys back to tuples
#     transitions = {}
#     for key, value in serializable_transitions.items():
#         from_state, label = key.split('|')
#         from_state = int(from_state) if from_state.isdigit() else from_state
#         to_state, prob = value
#         to_state = int(to_state) if isinstance(to_state, str) and to_state.isdigit() else to_state
#         transitions[(from_state, label)] = (to_state, float(prob))
#
#     return transitions
#
#
#
# def draw_process_graph(json_file_path):
#     # Load JSON data
#     with open(json_file_path, 'r') as file:
#         transitions = json.load(file)
#
#
#     # Create directed graph
#     G = nx.DiGraph()
#
#     # Extract node IDs from transitions
#     nodes = set()
#     for source, (target, probability) in transitions.items():
#         source_id = int(source.split('|')[0])
#         nodes.add(source_id)
#         nodes.add(target)
#
#     # Add nodes to the graph
#     for node in nodes:
#         G.add_node(node)
#
#     # Add edges with labels and weights
#     for source, (target, probability) in transitions.items():
#         source_id = int(source.split('|')[0])
#         action = source.split('|')[1]
#         G.add_edge(source_id, target, label=f"{action} - {probability:.3f}", weight=probability)
#
#     # Set up the figure
#     plt.figure(figsize=(14, 10))
#
#     # Use a circular layout for better spacing
#     pos = nx.circular_layout(G)
#
#     # Adjust specific node positions for better readability
#     # These positions are approximate based on the figure
#     pos[0] = np.array([0.5, 0.9])
#     pos[2] = np.array([0.5, 0.6])
#     pos[4] = np.array([0.5, 0.4])
#     pos[5] = np.array([0.5, 0.2])
#     pos[6] = np.array([0.2, 0.0])
#     pos[10] = np.array([0.8, 0.0])
#     pos[8] = np.array([0.2, -0.4])
#     pos[11] = np.array([0.8, -0.4])
#     pos[7] = np.array([0.2, -0.8])
#     pos[9] = np.array([0.5, -0.8])
#     pos[3] = np.array([0.5, -1.0])
#
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='white', edgecolors='black')
#
#     # Draw edges with curved arrows
#     for u, v, data in G.edges(data=True):
#         # Create curved edges
#         nx.draw_networkx_edges(
#             G, pos,
#             edgelist=[(u, v)],
#             width=1,
#             connectionstyle='arc3,rad=0.1',  # Add curve to the edges
#             arrowsize=15
#         )
#
#     # Draw node labels
#     nx.draw_networkx_labels(G, pos, font_size=12)
#
#     # Draw edge labels
#     edge_labels = {(u, v): data['label'] for u, v, data in G.edges(data=True)}
#     nx.draw_networkx_edge_labels(
#         G, pos,
#         edge_labels=edge_labels,
#         font_size=10,
#         label_pos=0.4,  # Position of label along the edge
#         rotate=False
#     )
#
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('process_visualization.png', dpi=300, bbox_inches='tight')
#     plt.show()


# Example usage
# draw_process_graph('BPI2017_7_transitions.json')