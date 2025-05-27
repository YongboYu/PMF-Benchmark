import pandas as pd
import numpy as np
import os
import json
import glob
import sys
import logging
from pathlib import Path
from datetime import datetime
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog, Trace, Event
from calculate_entropic_relevance_corrected import calculate_entropic_relevance as calc_er_cor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"er_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("ER_Evaluation")


def create_dfg_from_predictions(predictions_df, forecast_confidence=None):
    """
    Create DFG structure from predictions dataframe

    Args:
        predictions_df: DataFrame with predictions
        forecast_confidence: Confidence level (80, 90, 95, 100) or None for direct prediction values

    Returns:
        DFG in JSON format
    """
    # Extract directly-follows relations
    node_map = {}
    reverse_map = {}
    reverse_map['Start'] = 0
    reverse_map['End'] = 1

    # Process each column in predictions_df
    # Each column represents a directly-follows relation
    for df_relation in predictions_df.columns:
        # Some columns might already be using '▶' or '■' as symbols for start/end
        source, end = df_relation.replace('▶', 'Start').replace('■', 'End').split('->')
        source = source.strip()
        end = end.strip()
        
        if source not in reverse_map:
            reverse_map[source] = len(reverse_map)
        if end not in reverse_map:
            reverse_map[end] = len(reverse_map)

    # Create arcs
    arcs = []
    node_freq = {node: 0 for node in reverse_map.keys()}
    
    # Process each directly-follows relation
    for df_relation in predictions_df.columns:
        clean_relation = df_relation.replace('▶', 'Start').replace('■', 'End')
        source, end = [part.strip() for part in clean_relation.split('->')]
        
        # For each time step in the horizon, extract the prediction value
        for _, row in predictions_df.iterrows():
            freq = round(float(row[df_relation]))
            if freq <= 0:
                continue
                
            arcs.append({
                'from': reverse_map[source], 
                'to': reverse_map[end], 
                'freq': freq
            })
            
            if source == 'Start':
                node_freq[source] += freq
            else:
                node_freq[end] += freq

    # Create nodes
    nodes = []
    for node, freq in node_freq.items():
        nodes.append({'label': node, 'id': reverse_map[node], 'freq': round(freq)})

    return {'nodes': nodes, 'arcs': arcs}


def evaluate_predictions(prediction_file, ground_truth_file, output_file='er_results.csv'):
    """
    Evaluate predictions using entropic relevance

    Args:
        prediction_file: Path to prediction parquet file
        ground_truth_file: Path to ground truth data (XES)
        output_file: Path to output CSV file
    """
    # Create results file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write('dataset,horizon,model_group,model_name,forecast_confidence,er,non_fitting_traces,total_traces,timestamp\n')

    # Load predictions
    try:
        predictions_df = pd.read_parquet(prediction_file)
        logger.info(f"Loaded predictions from {prediction_file}")
    except Exception as e:
        logger.error(f"Error loading prediction file {prediction_file}: {e}")
        return

    # Extract metadata from file path
    path_parts = str(prediction_file).split('/')
    dataset = path_parts[1] if len(path_parts) >= 2 else 'unknown'
    horizon = path_parts[2].replace('horizon_', '') if len(path_parts) >= 3 else 'unknown'
    model_group = path_parts[4] if len(path_parts) >= 5 else 'unknown'
    model_name = path_parts[5].replace('_all_predictions.parquet', '') if len(path_parts) >= 6 else 'unknown'

    # Load ground truth
    try:
        variant = xes_importer.Variants.ITERPARSE
        parameters = {variant.value.Parameters.MAX_TRACES: 100000}
        log = xes_importer.apply(ground_truth_file, parameters=parameters)
        logger.info(f"Loaded ground truth from {ground_truth_file}")
    except Exception as e:
        logger.error(f"Error loading ground truth file {ground_truth_file}: {e}")
        return

    # Calculate entropic relevance
    try:
        # Create DFG
        dfg = create_dfg_from_predictions(predictions_df)

        # Save DFG to temporary JSON file
        temp_json = f'temp_{dataset}_{horizon}_{model_group}_{model_name}.json'
        with open(temp_json, 'w') as f:
            json.dump(dfg, f, indent=1)

        # Calculate entropic relevance
        er, non_fitting, total_traces = calc_er_cor(temp_json, log, model_name)

        # Write results
        with open(output_file, 'a') as f:
            f.write(f'{dataset},{horizon},{model_group},{model_name},direct,{er},{non_fitting},{total_traces},{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        logger.info(f"Calculated ER for {dataset}, {horizon}, {model_group}, {model_name}: {er}")

        # Remove temporary file
        if os.path.exists(temp_json):
            os.remove(temp_json)

    except Exception as e:
        logger.error(f"Error calculating ER for {dataset}, {horizon}, {model_group}, {model_name}: {e}")
        # Remove temporary file if it exists
        if os.path.exists(temp_json):
            os.remove(temp_json)


def main():
    """
    Main function to process all prediction files
    """
    # Find all prediction files ending with '_all_predictions.parquet'
    prediction_files = glob.glob('results/*/horizon_*/predictions/*/*_all_predictions.parquet')

    if not prediction_files:
        logger.error("No prediction files found. Make sure the results directory structure is correct.")
        return

    logger.info(f"Found {len(prediction_files)} prediction files to process.")

    # Create output directory if it doesn't exist
    os.makedirs("er_results", exist_ok=True)

    # Process each prediction file
    for prediction_file in prediction_files:
        # Extract dataset name from path
        path_parts = prediction_file.split('/')
        dataset = path_parts[1] if len(path_parts) >= 2 else 'unknown'

        # Find corresponding ground truth file
        ground_truth_file = f"data/raw/{dataset}.xes"

        if not os.path.exists(ground_truth_file):
            logger.warning(f"Ground truth file not found for {dataset}. Skipping...")
            continue

        # Calculate entropic relevance
        output_file = f'er_results/{dataset}_er_results.csv'
        evaluate_predictions(prediction_file, ground_truth_file, output_file)

    logger.info("Evaluation complete!")
    
    # Merge all results into a single file
    merge_er_results()


def merge_er_results():
    """
    Merge all ER result files into a single CSV
    """
    all_results = []
    er_files = glob.glob('er_results/*_er_results.csv')
    
    for file in er_files:
        try:
            results = pd.read_csv(file)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
    
    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        all_df.to_csv('er_results/all_er_results.csv', index=False)
        logger.info(f"Merged {len(er_files)} ER result files into er_results/all_er_results.csv")
    else:
        logger.warning("No ER result files found to merge.")


if __name__ == "__main__":
    main()