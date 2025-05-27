import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

import json
import time
import os
from subprocess import check_call
import networkx as nx


def visualize_dfg_from_json(json_file_or_data, output_path=None, dpi=300, figsize=(10, 8)):
    """
    Visualize a DFG using Graphviz Dot from a JSON file or data.

    Args:
        json_file_or_data: Path to JSON file or a dictionary with DFG data
                          (should contain 'nodes' and 'arcs')
        output_path: Path to save the visualization (without extension).
                    If None, a default path will be used.

    Returns:
        The created NetworkX graph
    """
    # Load the DFG from JSON if a string is provided (assuming it's a file path)
    if isinstance(json_file_or_data, str):
        with open(json_file_or_data) as file:
            dfg_data = json.load(file)
        # Use the filename (without path and extension) for default output
        default_output = os.path.splitext(os.path.basename(json_file_or_data))[0]
    else:
        # Assume it's already loaded data
        dfg_data = json_file_or_data
        default_output = f"dfg_{int(time.time())}"  # Use timestamp for default name

    # If output_path not provided, use default
    if output_path is None:
        output_path = f"./dfgs/{default_output}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a directed graph
    g = nx.DiGraph()

    # Add nodes
    for node in dfg_data['nodes']:
        node_id = node['id']
        g.add_node(node_id, label=node.get('label', node_id))

        # Add styling for special nodes
        if node.get('is_source', False):
            g.nodes[node_id]['style'] = 'filled'
            g.nodes[node_id]['fillcolor'] = 'lightblue'
            g.nodes[node_id]['shape'] = 'circle'
            g.nodes[node_id]['label'] = f"Start ({node_id})"

        if node.get('is_final', False):
            g.nodes[node_id]['peripheries'] = 2  # Double circle for final states
            prob = node.get('probability', 0)
            g.nodes[node_id]['label'] = f"End ({node_id})\n({math.exp(prob) if prob else 1.0:.3f})"
            g.nodes[node_id]['style'] = 'filled'
            g.nodes[node_id]['fillcolor'] = 'lightgreen'

    # Add arcs as edges
    for arc in dfg_data['arcs']:
        source = arc['from']
        target = arc['to']
        weight = arc.get('weight', 1.0)
        label = arc.get('freq', '')

        g.add_edge(
            source, target,
            label=f"{label}" if label else f"{weight:.3f}",
            weight=weight,
            penwidth=max(1, weight * 5)  # Scale edge thickness
        )

    # Generate DOT file and convert to PNG
    dot = nx.drawing.nx_pydot.to_pydot(g)
    dot_path = f"{output_path}.dot"
    png_path = f"{output_path}.png"

    # Save DOT file
    with open(dot_path, 'w') as file:
        file.write(str(dot))

    # Convert to PNG using Graphviz
    check_call(['dot', '-Tpng', dot_path, '-o', png_path])

    print(f"DFG visualization saved to {png_path}")

    # Optionally cleanup the dot file
    os.remove(dot_path)

    return g, png_path


def visualize_all_dfgs_combined_pdf(rolling_truth_dfgs, rolling_training_dfgs, rolling_pred_dfgs,
                                    output_file="combined_dfgs.pdf"):
    """
    Create a single high-resolution PDF containing DFG visualizations for ground truth, training and predictions.

    Args:
        rolling_truth_dfgs: Dictionary containing ground truth DFGs by time period
        rolling_training_dfgs: Dictionary containing training DFGs by time period
        rolling_pred_dfgs: Dictionary containing prediction DFGs by time period
        output_file: Path to the output PDF file

    Returns:
        Path to the created PDF file
    """


    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Create a temporary directory to store individual visualization files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    # Get sorted list of all time periods
    all_periods = sorted(set(list(rolling_truth_dfgs.keys()) +
                             list(rolling_training_dfgs.keys()) +
                             list(rolling_pred_dfgs.keys())))

    # Set higher DPI for better resolution
    DPI = 300

    # Initialize PDF with larger page size
    with PdfPages(output_file) as pdf:
        for period in all_periods:
            # Create a figure with 3 subplots and larger size
            fig, axes = plt.subplots(3, 1, figsize=(12, 20), dpi=DPI)
            fig.suptitle(f"Time Period: {period}", fontsize=20)

            # Add more space between subplots
            plt.subplots_adjust(hspace=0.3)

            # Ground Truth DFG
            axes[0].set_title("Ground Truth DFG", fontsize=16)
            axes[0].axis('off')

            if period in rolling_truth_dfgs and 'dfg_json' in rolling_truth_dfgs[period]:
                truth_json = rolling_truth_dfgs[period]['dfg_json']
                truth_output = f"{temp_dir}/truth_{period}"

                # Create high-resolution image of the DFG
                visualize_dfg_from_json(truth_json, truth_output, dpi=DPI)

                # Check if the image was created
                if Path(f"{truth_output}.png").exists():
                    img = plt.imread(f"{truth_output}.png")
                    axes[0].imshow(img)
                else:
                    axes[0].text(0.5, 0.5, "Error creating visualization",
                                 horizontalalignment='center', verticalalignment='center', fontsize=14)
            else:
                axes[0].text(0.5, 0.5, "No ground truth data available",
                             horizontalalignment='center', verticalalignment='center', fontsize=14)

            # Training DFG
            axes[1].set_title("Training DFG", fontsize=16)
            axes[1].axis('off')

            if period in rolling_training_dfgs and 'dfg_json' in rolling_training_dfgs[period]:
                training_json = rolling_training_dfgs[period]['dfg_json']
                training_output = f"{temp_dir}/training_{period}"

                visualize_dfg_from_json(training_json, training_output, dpi=DPI)

                if Path(f"{training_output}.png").exists():
                    img = plt.imread(f"{training_output}.png")
                    axes[1].imshow(img)
                else:
                    axes[1].text(0.5, 0.5, "Error creating visualization",
                                 horizontalalignment='center', verticalalignment='center', fontsize=14)
            else:
                axes[1].text(0.5, 0.5, "No training data available",
                             horizontalalignment='center', verticalalignment='center', fontsize=14)

            # Prediction DFG
            axes[2].set_title("Prediction DFG", fontsize=16)
            axes[2].axis('off')

            if period in rolling_pred_dfgs and 'dfg_json' in rolling_pred_dfgs[period]:
                pred_json = rolling_pred_dfgs[period]['dfg_json']
                pred_output = f"{temp_dir}/pred_{period}"

                visualize_dfg_from_json(pred_json, pred_output, dpi=DPI)

                if Path(f"{pred_output}.png").exists():
                    img = plt.imread(f"{pred_output}.png")
                    axes[2].imshow(img)
                else:
                    axes[2].text(0.5, 0.5, "Error creating visualization",
                                 horizontalalignment='center', verticalalignment='center', fontsize=14)
            else:
                axes[2].text(0.5, 0.5, "No prediction data available",
                             horizontalalignment='center', verticalalignment='center', fontsize=14)

            # Improve layout with more space for the title
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)

    print(f"Combined PDF created at: {output_file}")

    # Clean up temporary files
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")

    return output_file