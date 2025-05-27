import pm4py
import pandas as pd
import math
import json
from ER.utils_calculate_ER import DFGConstructor, ERCalculator, GraphVisualizer
from ER.visual_dfg_json import visualize_all_dfgs_combined_pdf
import os

# Create necessary directories
# os.makedirs('results/er_metrics_with_end_count', exist_ok=True)
# results_folder = 'results/er_metrics_with_end_count'
results_folder = 'results/er_metrics_v1/dfg_visualization'
os.makedirs(f'{results_folder}', exist_ok=True)

# Define parameters
dataset = 'BPI2017'
horizon = '7'
# horizon = '28'
start_time = '2016-10-22 00:00:00'

# dataset = 'sepsis'
# horizon = '7'
# # horizon = '28'
# start_time = '2015-01-05 00:00:00'

# dataset = 'Hospital_Billing'
# # horizon = '7'
# horizon = '28'
# start_time = '2015-02-05 00:00:00'

# dataset = 'BPI2019_1'
# # horizon = '7'
# horizon = '28'
# # start_time = '2018-11-19 00:00:00'
# start_time = '2018-10-11 00:00:00'

# dataset = 'RTFMP'
# # horizon = '7'
# horizon = '28'
# start_time = '2009-04-27 00:00:00'

# model_group = 'deep_learning'
# base_model_name = 'deepar'  # original model

# Add all models to evaluate
models_to_evaluate = [
    # ('baseline', 'persistence'),
    # ('baseline', 'naive_seasonal'),
    ('statistical', 'ar2'),
    # ('regression', 'random_forest'),
    # ('regression', 'xgboost'),
    # ('deep_learning', 'rnn'),
    # ('deep_learning', 'deepar')  # Original model included for completeness
]

# 1. Initialize the classes
dfg_constructor = DFGConstructor()
er_calculator = ERCalculator()

# 2. Load the event log
log_file = f'data/interim/processed_logs/{dataset}.xes'
log = pm4py.read_xes(log_file)

# 3. Extract rolling window sublogs (do this only once)
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

# Store all results here
all_model_results = {}
all_window_metrics = {}

# Process each model
for model_info in models_to_evaluate:
    current_model_group, current_model_name = model_info

    print(f"\n===== Processing {current_model_group}/{current_model_name} =====")

    # 6. Load and process predictions
    print(f"Loading predictions for {current_model_name}...")
    prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{current_model_group}/{current_model_name}_all_predictions.parquet'

    if not os.path.exists(prediction_file):
        print(f"Warning: Prediction file {prediction_file} not found. Skipping model.")
        continue

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

    # 10. Generate individual model report
    report_prefix = f"results/er_metrics_with_end_count/{dataset}_{horizon}_{current_model_group}_{current_model_name}"
    er_calculator.generate_er_metric_report(rolling_er_results, output_prefix=report_prefix)

    # 10. Calculate evaluation metrics for prediction vs truth
    print("Computing overall metrics...")
    er_metrics = er_calculator.calculate_er_metrics(rolling_er_results)

    # Store metrics
    model_key = f"{current_model_group}_{current_model_name}"
    all_model_results[model_key] = er_metrics

    # Extract window metrics
    for window_key, metrics in rolling_er_results.items():
        # Create window entry if it doesn't exist
        if window_key not in all_window_metrics:
            all_window_metrics[window_key] = {
                'truth_er': metrics['truth']['entropic_relevance'],
                'truth_fit_er': metrics['truth'].get('fitting_traces_ER', 0),
                'truth_nonfit_er': metrics['truth'].get('non_fitting_traces_ER', 0),
                'truth_fitting_ratio': metrics['truth'].get('fitting_ratio', 0),
                'truth_total_traces': metrics['truth'].get('total_traces', 0),
                'training_er': metrics['training']['entropic_relevance'],
                'training_fit_er': metrics['training'].get('fitting_traces_ER', 0),
                'training_nonfit_er': metrics['training'].get('non_fitting_traces_ER', 0),
                'training_fitting_ratio': metrics['training'].get('fitting_ratio', 0),
                'training_total_traces': metrics['training'].get('total_traces', 0)
            }

        # Add current model data
        truth_er = metrics['truth']['entropic_relevance']
        pred_er = metrics['pred']['entropic_relevance']

        if not (math.isnan(truth_er) or math.isnan(pred_er)):
            abs_error = abs(truth_er - pred_er)
            pct_error = abs_error / truth_er * 100 if truth_er != 0 else float('nan')
        else:
            abs_error = float('nan')
            pct_error = float('nan')

        all_window_metrics[window_key][f"{model_key}_er"] = pred_er
        all_window_metrics[window_key][f"{model_key}_fit_er"] = metrics['pred'].get('fitting_traces_ER', 0)
        all_window_metrics[window_key][f"{model_key}_nonfit_er"] = metrics['pred'].get('non_fitting_traces_ER', 0)
        all_window_metrics[window_key][f"{model_key}_fitting_ratio"] = metrics['pred'].get('fitting_ratio', 0)
        all_window_metrics[window_key][f"{model_key}_total_traces"] = metrics['pred'].get('total_traces', 0)
        # all_window_metrics[window_key][f"{model_key}_abs_error"] = abs_error
        # all_window_metrics[window_key][f"{model_key}_pct_error"] = pct_error

    # 11. Print the results
    print("\n===== ENTROPIC RELEVANCE EVALUATION METRICS =====")
    print(f"Model: {current_model_group}/{current_model_name}")
    print(f"Number of comparable windows: {er_metrics['n']}")
    print(f"Mean Absolute Error (MAE): {er_metrics['mae']:.4f}")
    print(f"Root Mean Square Error (RMSE): {er_metrics['rmse']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {er_metrics['mape']:.2f}%")

# Create summary table with overall metrics
summary_data = []
for model_key, metrics in all_model_results.items():
    model_group, model_name = model_key.split('_', 1)

    # Calculate averages for additional metrics
    avg_fit_er = 0
    avg_nonfit_er = 0
    avg_fitting_ratio = 0
    avg_total_traces = 0
    valid_windows = 0

    for window_metrics in all_window_metrics.values():
        if f"{model_key}_er" in window_metrics and not math.isnan(window_metrics[f"{model_key}_er"]):
            avg_fit_er += window_metrics.get(f"{model_key}_fit_er", 0)
            avg_nonfit_er += window_metrics.get(f"{model_key}_nonfit_er", 0)
            avg_fitting_ratio += window_metrics.get(f"{model_key}_fitting_ratio", 0)
            avg_total_traces += window_metrics.get(f"{model_key}_total_traces", 0)
            valid_windows += 1

    if valid_windows > 0:
        avg_fit_er /= valid_windows
        avg_nonfit_er /= valid_windows
        avg_fitting_ratio /= valid_windows
        avg_total_traces /= valid_windows

    summary_data.append({
        'model_group': model_group,
        'model_name': model_name,
        'n': metrics['n'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'avg_er': metrics.get('avg_er', float('nan')),
        'avg_fit_er': avg_fit_er,
        'avg_nonfit_er': avg_nonfit_er,
        'avg_fitting_ratio': avg_fitting_ratio,
        'avg_total_traces': avg_total_traces
    })

# Create combined results structure
combined_results = {
    'dataset': dataset,
    'horizon': horizon,
    'models': all_model_results,
    'window_metrics': all_window_metrics,
    'summary': {
        model_key: {
            'n': metrics['n'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'avg_er': metrics.get('avg_er', float('nan')),
            'avg_fit_er': summary_data[i].get('avg_fit_er', 0),
            'avg_nonfit_er': summary_data[i].get('avg_nonfit_er', 0),
            'avg_fitting_ratio': summary_data[i].get('avg_fitting_ratio', 0),
            'avg_total_traces': summary_data[i].get('avg_total_traces', 0)
        } for i, (model_key, metrics) in enumerate(all_model_results.items())
    }
}

# Save combined results to JSON
combined_output_file = f'{results_folder}/{dataset}_horizon_{horizon}_combined_er_metrics.json'
with open(combined_output_file, 'w') as f:
    json.dump(combined_results, f, indent=2)

# Save combined results to CSV
df_combined = pd.DataFrame.from_dict(all_window_metrics, orient='index')
combined_csv_file = f'{results_folder}/{dataset}_horizon_{horizon}_combined_er_metrics.csv'
df_combined.to_csv(combined_csv_file, index_label='window')

summary_df = pd.DataFrame(summary_data)
summary_csv_file = f'{results_folder}/{dataset}_horizon_{horizon}_er_metrics_summary.csv'
summary_df.to_csv(summary_csv_file, index=False)

print(f"\nCombined results saved to:")
print(f"  - JSON: {combined_output_file}")
print(f"  - CSV: {combined_csv_file}")
print(f"  - Summary: {summary_csv_file}")


output_pdf = visualize_all_dfgs_combined_pdf(
    rolling_truth_dfgs,
    rolling_training_dfgs,
    rolling_pred_dfgs,
    f"{results_folder}/{dataset}_horizon_{horizon}_combined_dfgs.pdf"
)