import pandas as pd
import numpy as np
from pathlib import Path
from darts import TimeSeries
from typing import Dict, List, Any, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset: str) -> TimeSeries:
    """Load dataset from HDF5 file"""
    data_path = Path("data/processed/time_series_df.h5")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = pd.read_hdf(data_path, key=dataset)
    return TimeSeries.from_dataframe(data)


def get_top_frequent_components(series: TimeSeries, n: int = 10) -> List[str]:
    """Get top N most frequent components based on training set means"""
    # Use first 80% as training
    train_size = int(len(series) * 0.8)
    train_data = series[:train_size]

    # Calculate means for each component
    means = {
        component: np.mean(train_data[component].values().flatten())
        for component in series.components
    }

    # Sort components by mean value and get top N
    top_components = sorted(means.items(), key=lambda x: x[1], reverse=True)[:n]

    logger.info("\nTop components and their mean frequencies:")
    for comp, mean_val in top_components:
        logger.info(f"{comp}: {mean_val:.4f}")

    return [comp[0] for comp in top_components]


def load_predictions(dataset: str, horizon: int, model_group: str, model_name: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """Load predictions from parquet files"""
    base_path = Path(f"results/{dataset}/horizon_{horizon}/predictions/{model_group}")
    all_pred_path = base_path / f"{model_name}_all_predictions.parquet"
    last_pred_path = base_path / f"{model_name}_last_predictions.parquet"

    if not all_pred_path.exists() or not last_pred_path.exists():
        raise FileNotFoundError(f"Prediction files not found for {model_name}")

    # Load the dataframes
    all_predictions = pd.read_parquet(all_pred_path)
    last_predictions = pd.read_parquet(last_pred_path)

    # Set debug level to INFO temporarily to show actual dataframe structure
    logger.info(f"{model_name} all_predictions columns: {all_predictions.columns.tolist()}")
    logger.info(
        f"{model_name} all_predictions index names: {getattr(all_predictions.index, 'names', ['Not MultiIndex'])}")
    logger.info(f"{model_name} last_predictions columns: {last_predictions.columns.tolist()}")

    # For regression models, there's no 'component' column - the components are likely the columns directly
    if model_group == "regression":
        # Ensure proper index structure for all_predictions
        if not isinstance(all_predictions.index, pd.MultiIndex):
            if 'sequence_start_time' in all_predictions.columns and 'horizon_step' in all_predictions.columns:
                all_predictions = all_predictions.set_index(['sequence_start_time', 'horizon_step'])

        # Ensure proper index structure for last_predictions
        if not isinstance(last_predictions.index, pd.MultiIndex):
            if 'sequence_start_time' in last_predictions.columns:
                last_predictions = last_predictions.set_index('sequence_start_time')

        return all_predictions, last_predictions

    # For other model types, we need to pivot
    # Handle all_predictions
    if 'component' in all_predictions.columns:
        # Check for required index columns
        index_cols = ['sequence_start_time', 'horizon_step']
        if all(col in all_predictions.columns for col in index_cols):
            # Pivot the dataframe to make components as columns
            all_predictions = all_predictions.pivot_table(
                index=index_cols,
                columns='component',
                values='value',
                aggfunc='first'  # Use 'first' in case of duplicates
            )
        else:
            raise ValueError(f"Missing required columns in all_predictions: {index_cols}")
    elif isinstance(all_predictions.index, pd.MultiIndex) and 'component' in all_predictions.index.names:
        # If component is in the index, reset and pivot
        all_predictions = all_predictions.reset_index()
        all_predictions = all_predictions.pivot_table(
            index=['sequence_start_time', 'horizon_step'],
            columns='component',
            values='value',
            aggfunc='first'
        )

    # Handle last_predictions
    if 'component' in last_predictions.columns:
        if 'sequence_start_time' in last_predictions.columns:
            last_predictions = last_predictions.pivot_table(
                index='sequence_start_time',
                columns='component',
                values='value',
                aggfunc='first'
            )
        else:
            raise ValueError("Missing 'sequence_start_time' column in last_predictions")
    elif isinstance(last_predictions.index, pd.MultiIndex) and 'component' in last_predictions.index.names:
        last_predictions = last_predictions.reset_index()
        last_predictions = last_predictions.pivot_table(
            index='sequence_start_time',
            columns='component',
            values='value',
            aggfunc='first'
        )

    # Handle MultiIndex columns if they exist
    if isinstance(all_predictions.columns, pd.MultiIndex):
        all_predictions.columns = all_predictions.columns.get_level_values(-1)

    if isinstance(last_predictions.columns, pd.MultiIndex):
        last_predictions.columns = last_predictions.columns.get_level_values(-1)

    return all_predictions, last_predictions

def create_test_sequences(series: TimeSeries, horizon: int) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """Create test sequences for evaluation"""
    train_size = int(len(series) * 0.8)
    test = series[train_size:]

    n_sequences = len(test) - horizon + 1
    logger.info(f"Creating {n_sequences} test sequences")

    all_point_sequences = []
    last_point_sequences = []

    for i in range(n_sequences):
        seq = test[i:i + horizon]
        all_point_sequences.append(seq)
        last_point_sequences.append(TimeSeries.from_values(seq.values()[-1]))

    return all_point_sequences, last_point_sequences


def evaluate_predictions(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """Calculate MAE and RMSE metrics"""
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    return {'mae': mae, 'rmse': rmse}


def evaluate_model_top_components(
        all_predictions: pd.DataFrame,
        last_predictions: pd.DataFrame,
        all_ground_truth: List[TimeSeries],
        last_ground_truth: List[TimeSeries],
        top_components: List[str]
) -> Dict[str, Any]:
    """Evaluate model performance on top frequent components"""

    metrics = {
        'per_component': {},
        'average': {
            'all_points': {'mae': 0.0, 'rmse': 0.0},
            'last_point': {'mae': 0.0, 'rmse': 0.0}
        }
    }

    available_components = [comp for comp in top_components if comp in all_predictions.columns]
    if not available_components:
        raise ValueError("No top components found in predictions")

    for component in available_components:
        # Extract predictions and ground truth for the component
        all_pred_values = all_predictions[component].values
        all_true_values = np.concatenate([seq[component].values().flatten()
                                          for seq in all_ground_truth])

        last_pred_values = last_predictions[component].values
        last_true_values = np.array([seq[component].values()[0][0]
                                     for seq in last_ground_truth])

        # Calculate metrics
        all_points_metrics = evaluate_predictions(all_pred_values, all_true_values)
        last_point_metrics = evaluate_predictions(last_pred_values, last_true_values)

        metrics['per_component'][component] = {
            'all_points': all_points_metrics,
            'last_point': last_point_metrics
        }

    # Calculate average metrics
    metrics['average'] = {
        'all_points': {
            'mae': np.mean([m['all_points']['mae'] for m in metrics['per_component'].values()]),
            'rmse': np.mean([m['all_points']['rmse'] for m in metrics['per_component'].values()])
        },
        'last_point': {
            'mae': np.mean([m['last_point']['mae'] for m in metrics['per_component'].values()]),
            'rmse': np.mean([m['last_point']['rmse'] for m in metrics['per_component'].values()])
        }
    }

    return metrics


def main():
    dataset = "BPI2017"
    horizon = 7
    model_groups = ["baseline", "statistical", "deep_learning", "foundation", "regression"]

    # Load dataset and get top components
    logger.info("Loading dataset...")
    full_series = load_dataset(dataset)
    top_components = get_top_frequent_components(full_series, n=10)

    # Create test sequences
    logger.info("\nCreating test sequences...")
    all_ground_truth, last_ground_truth = create_test_sequences(full_series, horizon)

    results = {}

    # Process each model group
    for model_group in model_groups:
        logger.info(f"\nProcessing {model_group} models...")
        pred_dir = Path(f"results/{dataset}/horizon_{horizon}/predictions/{model_group}")

        if not pred_dir.exists():
            logger.warning(f"No predictions directory found for {model_group}")
            continue

        model_files = list(pred_dir.glob("*_all_predictions.parquet"))

        for pred_file in model_files:
            model_name = pred_file.stem.replace("_all_predictions", "")
            logger.info(f"Evaluating {model_name}...")

            try:
                # Load and filter predictions
                all_pred, last_pred = load_predictions(dataset, horizon, model_group, model_name)

                # Evaluate
                metrics = evaluate_model_top_components(
                    all_pred, last_pred,
                    all_ground_truth, last_ground_truth,
                    top_components
                )

                results[f"{model_group}_{model_name}"] = metrics
                logger.info(f"Average MAE (all points): {metrics['average']['all_points']['mae']:.4f}")
                logger.info(f"Average MAE (last point): {metrics['average']['last_point']['mae']:.4f}")

            except Exception as e:
                logger.error(f"Error processing {model_name}: {str(e)}")
                continue

    # Save results
    output_dir = Path("results/top_frequent_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        # Create summary DataFrame
        summary_data = []
        for model, metrics in results.items():
            summary_data.append({
                'model': model,
                'all_points_mae': metrics['average']['all_points']['mae'],
                'all_points_rmse': metrics['average']['all_points']['rmse'],
                'last_point_mae': metrics['average']['last_point']['mae'],
                'last_point_rmse': metrics['average']['last_point']['rmse']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('all_points_mae')

        # Save results
        summary_df.to_csv(output_dir / f"{dataset}_top_frequent_summary.csv", index=False)

        with open(output_dir / f"{dataset}_top_frequent_detailed.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_dir}")
        logger.info("\nTop 5 models by all-points MAE:")
        print(summary_df[['model', 'all_points_mae', 'last_point_mae']].head().to_string())
    else:
        logger.warning("No results were generated")


if __name__ == "__main__":
    main()