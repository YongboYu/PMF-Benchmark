import pm4py
import pandas as pd
import math
import json
from ER_v2.utils_ER_v1 import DFGConstructor, ERCalculator, GraphVisualizer
from ER_v2.utils_dfg_visualize import visualize_all_dfgs_combined_pdf
import os

results_folder = 'ER_v2/er_metrics_v1/dfg_visualization'
os.makedirs(f'{results_folder}', exist_ok=True)

model_group = 'statistical'
model_name = 'ar2'

# Define parameters
# dataset = 'BPI2017'
# horizon = '7'
# # horizon = '28'
# start_time = '2016-10-22 00:00:00'

# dataset = 'sepsis'
# horizon = '7'
# # horizon = '28'
# start_time = '2015-01-05 00:00:00'

# dataset = 'Hospital_Billing'
# horizon = '7'
# # horizon = '28'
# start_time = '2015-02-05 00:00:00'

dataset = 'BPI2019_1'
horizon = '7'
# horizon = '28'
# start_time = '2018-11-19 00:00:00'
start_time = '2018-10-11 00:00:00'

# dataset = 'RTFMP'
# # horizon = '7'
# horizon = '28'
# start_time = '2009-04-27 00:00:00'

# model_group = 'deep_learning'
# base_model_name = 'deepar'  # original model

# # Add all models to evaluate
# models_to_evaluate = [
#     # ('baseline', 'persistence'),
#     # ('baseline', 'naive_seasonal'),
#     ('statistical', 'ar2'),
#     # ('regression', 'random_forest'),
#     # ('regression', 'xgboost'),
#     # ('deep_learning', 'rnn'),
#     # ('deep_learning', 'deepar')  # Original model included for completeness
# ]

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

# 6. Load and process predictions
print(f"Loading predictions for {model_name}...")
prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet'
predictions_df = pd.read_parquet(prediction_file)
agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')
agg_pred_round = agg_pred.round(0).astype(int)

# 7. Create prediction DFGs
print("Creating prediction DFGs...")
rolling_pred_dfgs = dfg_constructor.create_dfgs_from_rolling_predictions(seq_test_log, agg_pred_round)

output_pdf = visualize_all_dfgs_combined_pdf(
    rolling_truth_dfgs,
    rolling_training_dfgs,
    rolling_pred_dfgs,
    f"{results_folder}/{dataset}_horizon_{horizon}_combined_dfgs.pdf"
)