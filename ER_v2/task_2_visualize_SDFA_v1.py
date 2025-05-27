import time

import pm4py
import pandas as pd
import math
import json
from ER_v2.utils_ER_v1 import DFGConstructor, ERCalculator, GraphVisualizer
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
from pathlib import Path
from subprocess import check_call

results_folder = 'ER_v2/er_metrics_v1/sdfa_visualization'
os.makedirs(f'{results_folder}', exist_ok=True)

model_group = 'statistical'
model_name = 'ar2'

# Define parameters
# dataset = 'BPI2017'
# horizon = '7'
# start_time = '2016-10-22 00:00:00'

# dataset = 'sepsis'
# horizon = '7'
# start_time = '2015-01-05 00:00:00'

dataset = 'Hospital_Billing'
horizon = '7'
start_time = '2015-02-05 00:00:00'

# dataset = 'BPI2019_1'
# horizon = '7'
# start_time = '2018-10-11 00:00:00'

# 1. Initialize the classes
dfg_constructor = DFGConstructor()
er_calculator = ERCalculator()

# 2. Load the event log
log_file = f'data/interim/processed_logs/{dataset}.xes'
log = pm4py.read_xes(log_file)

# 3. Extract rolling window sublogs
print("Extracting rolling window sublogs...")
seq_test_log = dfg_constructor.extract_rolling_window_sublogs(
    log, 'case:concept:name', 'concept:name', 'time:timestamp',
    start_time, horizon
)

# 4. Create ground truth DFGs from the sublogs
print("Creating ground truth DFGs...")
rolling_truth_dfgs = dfg_constructor.create_dfgs_from_rolling_window(seq_test_log)

# 5. Create training baseline DFGs (using 80% of data by time)
print("Creating training baseline DFGs...")
rolling_training_dfgs = dfg_constructor.create_dfgs_from_rolling_training(
    seq_test_log,
    log,
    'case:concept:name',
    'concept:name',
    'time:timestamp',
    time_length=int(horizon)
)

# 6. Load and process predictions
print(f"Loading predictions for {model_name}...")
prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet'
predictions_df = pd.read_parquet(prediction_file)
agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')
agg_pred_round = agg_pred.round(0).astype(int)

# 7. Create prediction DFGs
print("Creating prediction DFGs...")
rolling_pred_dfgs = dfg_constructor.create_dfgs_from_rolling_predictions(seq_test_log, agg_pred_round)

def visualize_sdfa_from_json(json_file_or_data, output_path=None, dpi=300, figsize=(10, 8)):
    """
    Visualize a Stochastic Deterministic Finite Automaton using Graphviz Dot from a JSON file or data.

    Args:
        json_file_or_data: Path to JSON file or a dictionary with DFG data
                          (should contain 'nodes' and 'arcs')
        output_path: Path to save the visualization (without extension).
                    If None, a default path will be used.
        dpi: DPI for the output image
        figsize: Figure size for matplotlib

    Returns:
        The created NetworkX graph and path to the output image
    """
    # Load the DFG from JSON if a string is provided
    if isinstance(json_file_or_data, str):
        with open(json_file_or_data) as file:
            dfg_data = json.load(file)
        default_output = os.path.splitext(os.path.basename(json_file_or_data))[0]
    else:
        dfg_data = json_file_or_data
        default_output = f"sdfa_{int(time.time())}"

    # If output_path not provided, use default
    if output_path is None:
        output_path = f"./sdfas/{default_output}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert DFG to automaton to get transition probabilities
    transitions, sources, final_states, trans_table = er_calculator.convert_dfg_into_automaton(
        dfg_data['nodes'], dfg_data['arcs']
    )

    # Create a directed graph
    g = nx.DiGraph()

    # Add nodes
    for node in dfg_data['nodes']:
        node_id = node['id']
        label = node.get('label', node_id)
        g.add_node(node_id, label=label)

        # Add styling for special nodes
        if label == '▶':
            g.nodes[node_id]['style'] = 'filled'
            g.nodes[node_id]['fillcolor'] = 'lightblue'
            g.nodes[node_id]['shape'] = 'circle'
            g.nodes[node_id]['label'] = f"▶ ({node_id})"

        if label == '■':
            g.nodes[node_id]['peripheries'] = 2  # Double circle for final states
            prob = final_states.get(node_id, 0)
            g.nodes[node_id]['label'] = f"■ ({node_id})\n({math.exp(prob) if prob else 1.0:.3f})"
            g.nodes[node_id]['style'] = 'filled'
            g.nodes[node_id]['fillcolor'] = 'lightgreen'

    # Add arcs as edges with probabilities
    for (from_node, label), (to_node, prob) in transitions.items():
        g.add_edge(
            from_node, to_node,
            # label=f"{label}\n{prob:.3f}",
            label=f"{prob:.3f}",  # Only show probability
            weight=prob,
            penwidth=max(1, prob * 5)  # Scale edge thickness based on probability
        )

    # Add missing arcs to end state
    for node in dfg_data['nodes']:
        if node['label'] == '■':
            end_node_id = node['id']
            # Find all nodes that should have transitions to end state
            for arc in dfg_data['arcs']:
                if arc['to'] == end_node_id:
                    from_node = arc['from']
                    # Only add if this transition doesn't already exist
                    if not g.has_edge(from_node, end_node_id):
                        g.add_edge(
                            from_node, end_node_id,
                            label=f"{math.exp(final_states.get(end_node_id, 0)):.3f}",  # Only show probability
                            weight=math.exp(final_states.get(end_node_id, 0)),
                            penwidth=max(1, math.exp(final_states.get(end_node_id, 0)) * 5)
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

    print(f"SDFA visualization saved to {png_path}")

    # Cleanup the dot file
    os.remove(dot_path)

    return g, png_path

def visualize_all_sdfas_combined_pdf(rolling_truth_dfgs, rolling_training_dfgs, rolling_pred_dfgs, output_file="combined_sdfas.pdf"):
    """
    Create a single high-resolution PDF containing SDFA visualizations for ground truth, training and predictions.

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

            # Ground Truth SDFA
            axes[0].set_title("Ground Truth SDFA", fontsize=16)
            axes[0].axis('off')

            if period in rolling_truth_dfgs and 'dfg_json' in rolling_truth_dfgs[period]:
                truth_json = rolling_truth_dfgs[period]['dfg_json']
                truth_output = f"{temp_dir}/truth_{period}"

                # Create high-resolution image of the SDFA
                visualize_sdfa_from_json(truth_json, truth_output, dpi=DPI)

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

            # Training SDFA
            axes[1].set_title("Training SDFA", fontsize=16)
            axes[1].axis('off')

            if period in rolling_training_dfgs and 'dfg_json' in rolling_training_dfgs[period]:
                training_json = rolling_training_dfgs[period]['dfg_json']
                training_output = f"{temp_dir}/training_{period}"

                visualize_sdfa_from_json(training_json, training_output, dpi=DPI)

                if Path(f"{training_output}.png").exists():
                    img = plt.imread(f"{training_output}.png")
                    axes[1].imshow(img)
                else:
                    axes[1].text(0.5, 0.5, "Error creating visualization",
                               horizontalalignment='center', verticalalignment='center', fontsize=14)
            else:
                axes[1].text(0.5, 0.5, "No training data available",
                           horizontalalignment='center', verticalalignment='center', fontsize=14)

            # Prediction SDFA
            axes[2].set_title("Prediction SDFA", fontsize=16)
            axes[2].axis('off')

            if period in rolling_pred_dfgs and 'dfg_json' in rolling_pred_dfgs[period]:
                pred_json = rolling_pred_dfgs[period]['dfg_json']
                pred_output = f"{temp_dir}/pred_{period}"

                visualize_sdfa_from_json(pred_json, pred_output, dpi=DPI)

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

# Create combined visualization
output_pdf = visualize_all_sdfas_combined_pdf(
    rolling_truth_dfgs,
    rolling_training_dfgs,
    rolling_pred_dfgs,
    f"{results_folder}/{dataset}_horizon_{horizon}_combined_sdfas.pdf"
)

print(f"Visualization saved to {output_pdf}")
