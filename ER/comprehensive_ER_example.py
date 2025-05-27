import pm4py
import pandas as pd
import math
import json
from ER.utils_calculate_ER import DFGConstructor, ERCalculator, GraphVisualizer

# Define parameters
dataset = 'BPI2017'
horizon = '7'
start_time = '2016-10-22 00:00:00'
# dataset = 'sepsis'
# horizon = '7'
# # horizon = '28'
# start_time = '2015-01-05 00:00:00'
model_group = 'regression'
model_name = 'random_forest'
# model_group = 'baseline'
# model_name = 'persistence'

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
# Using the new method with horizon directly specified
rolling_training_dfgs = dfg_constructor.create_dfgs_from_rolling_training(
    seq_test_log,
    log,
    'case:concept:name',
    'concept:name',
    'time:timestamp',
    time_length=int(horizon)  # Convert horizon to int if necessary
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

# 8. Combine ground truth, prediction, and training DFGs
print("Combining truth, prediction, and training DFGs...")
combined_rolling_dfgs = dfg_constructor.reformat_rolling_dfgs(
    rolling_truth_dfgs,
    rolling_pred_dfgs,
    rolling_training_dfgs
)

# 9. Calculate entropic relevance for all time windows
print("Calculating entropic relevance metrics...")
rolling_er_results = er_calculator.calculate_rolling_entropic_relevance(combined_rolling_dfgs, seq_test_log)

# 10. generate report
# Add an output prefix to the method call
report = er_calculator.generate_er_metric_report(rolling_er_results, output_prefix="results/er_metrics")




###########
# 10. Calculate evaluation metrics for prediction vs truth
print("Computing overall metrics...")
er_metrics = er_calculator.calculate_er_metrics(rolling_er_results)

# 11. Print the results
print("\n===== ENTROPIC RELEVANCE EVALUATION METRICS =====")
print(f"Number of comparable windows: {er_metrics['n']}")
print(f"Mean Absolute Error (MAE): {er_metrics['mae']:.4f}")
print(f"Root Mean Square Error (RMSE): {er_metrics['rmse']:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {er_metrics['mape']:.2f}%")

# 12. Window-by-window comparison with training baseline and fitting/non-fitting traces
print("\n===== WINDOW-BY-WINDOW COMPARISON =====")
print(
    f"{'Window':20} {'Truth ER':10} {'Truth Fit':10} {'Truth Non-Fit':10} {'Pred ER':10} {'Pred Fit':10} {'Pred Non-Fit':10} {'Train ER':10} {'Train Fit':10} {'Train Non-Fit':10} {'Abs Error':10} {'% Error':10}")
print("-" * 130)

metrics_result = {
    'overall': er_metrics,
    'window_metrics': {}
}

for window_key, metrics in rolling_er_results.items():
    # Extract overall entropic relevance values
    truth_er = metrics['truth']['entropic_relevance']
    pred_er = metrics['pred']['entropic_relevance']
    training_er = metrics['training']['entropic_relevance']

    # Extract or calculate fitting and non-fitting ER components
    truth_fit_er = metrics['truth'].get('fit_er', 0)
    truth_nonfit_er = metrics['truth'].get('nonfit_er', 0)

    pred_fit_er = metrics['pred'].get('fit_er', 0)
    pred_nonfit_er = metrics['pred'].get('nonfit_er', 0)

    training_fit_er = metrics['training'].get('fit_er', 0)
    training_nonfit_er = metrics['training'].get('nonfit_er', 0)

    if not (math.isnan(truth_er) or math.isnan(pred_er)):
        abs_error = abs(truth_er - pred_er)
        pct_error = abs_error / truth_er * 100 if truth_er != 0 else float('nan')

        print(f"{window_key:20} {truth_er:<10.4f} {truth_fit_er:<10.4f} {truth_nonfit_er:<10.4f} "
              f"{pred_er:<10.4f} {pred_fit_er:<10.4f} {pred_nonfit_er:<10.4f} "
              f"{training_er:<10.4f} {training_fit_er:<10.4f} {training_nonfit_er:<10.4f} "
              f"{abs_error:<10.4f} {pct_error:<10.2f}%")

        metrics_result['window_metrics'][window_key] = {
            'truth_er': truth_er,
            'truth_fit_er': truth_fit_er,
            'truth_nonfit_er': truth_nonfit_er,
            'pred_er': pred_er,
            'pred_fit_er': pred_fit_er,
            'pred_nonfit_er': pred_nonfit_er,
            'training_er': training_er,
            'training_fit_er': training_fit_er,
            'training_nonfit_er': training_nonfit_er,
            'abs_error': abs_error,
            'pct_error': pct_error
        }
    else:
        print(f"{window_key:20} {truth_er if not math.isnan(truth_er) else 'N/A':<10} {'N/A':<10} {'N/A':<10} "
              f"{pred_er if not math.isnan(pred_er) else 'N/A':<10} {'N/A':<10} {'N/A':<10} "
              f"{training_er if not math.isnan(training_er) else 'N/A':<10} {'N/A':<10} {'N/A':<10}")

# 13. Save to file
output_file = f'{dataset}_{horizon}_{model_group}_{model_name}_er_metrics.json'
with open(output_file, 'w') as f:
    json.dump(metrics_result, f, indent=2)

# to csv
output_file = f'{dataset}_{horizon}_{model_group}_{model_name}_er_metrics.csv'
df = pd.DataFrame.from_dict(metrics_result['window_metrics'], orient='index')
df.to_csv(output_file, index_label='window')

print(f"\nResults saved to {output_file}")

# Alternative: Use the all-in-one method from ERCalculator
# results = er_calculator.run_er_evaluation(dataset, horizon, model_group, model_name, start_time)





#### Visualization
import time
import os
from subprocess import check_call
import networkx as nx


def visualize_dfg_from_json(json_file_or_data, output_path=None):
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


def visualize_transition_from_json(json_file):
    g = nx.DiGraph()
    with open(json_file) as file:
        transitions = json.load(file)
    for (t_from, label), (t_to, prob) in transitions.items():
        g.add_edge(t_from, t_to, label=label+' - ' + str(round(prob,3)))

    dot = nx.drawing.nx_pydot.to_pydot(g)
    file_name = './dfgs/temp3'
    with open(file_name + '.dot', 'w') as file:
        file.write(str(dot))
    check_call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png'])
    os.remove(file_name + '.dot')