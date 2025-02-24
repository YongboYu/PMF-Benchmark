from darts.metrics import mae, rmse
from typing import Dict, Any, Optional, Union, List
from darts import TimeSeries
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import logging
import numpy as np
import time


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path("results")
        self.temp_results_dir = self.results_dir / "temp"
        self.temp_results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(self, predictions: TimeSeries, actuals: TimeSeries) -> Dict[str, float]:
        """Evaluate predictions against actuals on their current scale"""
        return {
            'mae': mae(actuals, predictions),
            'rmse': rmse(actuals, predictions)
        }

    def evaluate_model_group(self, predictions: Dict[str, TimeSeries], test: TimeSeries,
                           transformer: Optional[object] = None) -> dict[str, dict[str, float]]:
        """Evaluate all models in a group"""
        return {
            model_name: self.evaluate_predictions(pred, test)
            for model_name, pred in predictions.items()
        }

    def get_results_paths(self, dataset: str, horizon: int, model_group: str) -> Dict[str, Path]:
        """Get all relevant paths for saving results with the new structure
        """
        base_path = self.results_dir
        
        paths = {
            'evaluation': base_path / 'evaluation' / dataset / f'horizon_{horizon}' / model_group,
            'models': base_path / 'models' / dataset / f'horizon_{horizon}' / model_group,
            'predictions': base_path / 'predictions' / dataset / f'horizon_{horizon}' / model_group
        }
        
        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        return paths

    def save_model(self, model: Any, dataset: str, model_group: str, 
                  model_name: str, horizon: int) -> Optional[str]:
        """Save trained model"""
        try:
            paths = self.get_results_paths(dataset, horizon, model_group)
            model_name_str = str(model_name).replace('/', '_')
            save_path = paths['models'] / model_name_str
            
            # Add .pkl extension if not present
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pkl')
            
            # Create all parent directories including model-specific subdirectories
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if hasattr(model, 'save'):
                model.save(str(save_path))
                return str(save_path)
            else:
                logger.warning(f"Model {model_name} does not support saving")
                return None
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            return None

    def save_predictions(self, predictions: List[TimeSeries], dataset: str, 
                        model_group: str, model_name: str, horizon: int) -> Dict[str, str]:
        """Save both all-points and last-point predictions"""
        paths = self.get_results_paths(dataset, horizon, model_group)
        
        # Save all-points predictions
        all_points_df = self.transform_predictions_to_dataframe(predictions, last_only=False)
        all_points_path = paths['predictions'] / f'{model_name}_all_predictions.csv'
        all_points_df.to_csv(all_points_path)
        
        # Save last-point predictions
        last_point_df = self.transform_predictions_to_dataframe(predictions, last_only=True)
        last_point_path = paths['predictions'] / f'{model_name}_last_predictions.csv'
        last_point_df.to_csv(last_point_path)
        
        return {
            'all_points': str(all_points_path),
            'last_point': str(last_point_path)
        }

    def save_metrics(self, metrics_df: pd.DataFrame, dataset: str, 
                    model_group: str, model_name: str, horizon: int) -> str:
        """Save both all-points and last-point metrics in a single CSV file"""
        paths = self.get_results_paths(dataset, horizon, model_group)
        
        # Create a single row with all metrics
        combined_metrics = pd.DataFrame([{
            'dataset': dataset,
            'horizon': horizon,
            'model': f"{model_group}_{model_name}",
            'all_points_mae': metrics_df['all_points']['mae'],
            'all_points_rmse': metrics_df['all_points']['rmse'],
            'last_point_mae': metrics_df['last_point']['mae'],
            'last_point_rmse': metrics_df['last_point']['rmse'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        # Save metrics
        metrics_path = paths['evaluation'] / f'{model_name}_combined_metrics.csv'
        
        # If file exists, append; otherwise create new
        if metrics_path.exists():
            existing_metrics = pd.read_csv(metrics_path)
            combined_metrics = pd.concat([existing_metrics, combined_metrics], ignore_index=True)
        
        combined_metrics.to_csv(metrics_path, index=False)
        
        return str(metrics_path)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def save_model_result(self, dataset: str, model_group: str, model_name: str, 
                         horizon: int, metrics: Dict[str, Any], training_time: float):
        """Save individual model results to temporary file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert metrics dictionary numpy types to native Python types
        converted_metrics = {}
        for metric_type, metric_values in metrics.items():
            converted_metrics[metric_type] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metric_values.items()
            }
        
        result = {
            "dataset": dataset,
            "horizon": horizon,
            "model_group": model_group,
            "model_name": model_name,
            "metrics": converted_metrics,
            "training_time": float(training_time),
            "timestamp": timestamp
        }
        
        # Create unique filename for this model run
        temp_file = self.temp_results_dir / f"{dataset}_{horizon}_{model_group}_{model_name}.json"
        
        with open(temp_file, 'w') as f:
            json.dump(result, f, indent=4)

    def merge_results(self):
        """Merge all temporary result files into final results_log.json"""
        final_results = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {}
        }

        # Read all temporary files
        for temp_file in self.temp_results_dir.glob("*.json"):
            with open(temp_file, 'r') as f:
                result = json.load(f)

            dataset = result["dataset"]
            horizon = str(result["horizon"])
            model_group = result["model_group"]
            model_name = result["model_name"]

            # Build nested structure for datasets
            if dataset not in final_results["datasets"]:
                final_results["datasets"][dataset] = {
                    "last_updated": result["timestamp"],
                    "horizons": {}
                }

            if horizon not in final_results["datasets"][dataset]["horizons"]:
                final_results["datasets"][dataset]["horizons"][horizon] = {
                    "last_updated": result["timestamp"],
                    "model_groups": {}
                }

            if model_group not in final_results["datasets"][dataset]["horizons"][horizon]["model_groups"]:
                final_results["datasets"][dataset]["horizons"][horizon]["model_groups"][model_group] = {
                    "last_updated": result["timestamp"],
                    "models": {}
                }

            # Add model results
            final_results["datasets"][dataset]["horizons"][horizon]["model_groups"][model_group]["models"][
                model_name] = {
                "metrics": result["metrics"],
                "training_time": result["training_time"],
                "timestamp": result["timestamp"]
            }

        # Save merged results
        results_file = self.results_dir / "results_log.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4)
            
        # Cleanup temporary files
        # for temp_file in self.temp_results_dir.glob("*.json"):
        #     temp_file.unlink()

    def save_results(self, model_results: Dict[str, Any], dataset: str, 
                    horizon: int, predictions: List[TimeSeries]) -> None:
        """Main function to save all results
        
        Args:
            model_results: Dictionary containing model results and metrics
            dataset: Name of the dataset
            horizon: Forecast horizon
            predictions: List of TimeSeries predictions
        """
        # Save trained model if provided and supports saving
        if model_results.get('model') and self.config['evaluation']['save_models']:
            if isinstance(model_results['model'], list):  # Handle multiple models (statistical case)
                saved_paths = []
                for idx, component_model in enumerate(model_results['model']):
                    model_path = self.save_model(
                        model=component_model,
                        dataset=dataset,
                        model_group=model_results['model_name'],
                        model_name=f"{model_results['specific_model']}/component_{idx}",
                        horizon=horizon
                    )
                    if model_path:
                        saved_paths.append(model_path)
                if saved_paths:
                    model_results['model_paths'] = saved_paths
            else:  # Single model case
                model_path = self.save_model(
                    model=model_results['model'],
                    dataset=dataset,
                    model_group=model_results['model_name'],
                    model_name=model_results['specific_model'],
                    horizon=horizon
                )
                if model_path:
                    model_results['model_path'] = model_path

        # Save predictions
        if predictions and self.config['evaluation']['save_predictions']:
            pred_paths = self.save_predictions(
                predictions=predictions,
                dataset=dataset,
                model_group=model_results['model_name'],
                model_name=model_results['specific_model'],
                horizon=horizon
            )
            model_results['predictions_paths'] = pred_paths

        # Save metrics
        if 'metrics' in model_results:
            metrics_path = self.save_metrics(
                metrics_df=model_results['metrics'],
                dataset=dataset,
                model_group=model_results['model_name'],
                model_name=model_results['specific_model'],
                horizon=horizon
            )
            model_results['metrics_path'] = metrics_path

        # Update results log
        self.save_model_result(dataset, model_results['model_name'], model_results['specific_model'], 
                              horizon, model_results['metrics'], model_results.get('training_time', 0))

    def load_model(self, dataset: str, model_group: str, model_name: str, horizon: int) -> Optional[Any]:
        """Load a saved model from disk
        
        Args:
            dataset: Name of the dataset
            model_group: Model group (e.g., 'baseline', 'statistical', etc.)
            model_name: Name of the specific model
            horizon: Forecast horizon
            
        Returns:
            The loaded model if successful, None otherwise
        """
        try:
            paths = self.get_results_paths(dataset, horizon, model_group)
            load_path = paths['models'] / f"{model_name}.pkl"
            
            if not load_path.exists():
                logger.warning(f"No saved model found at {load_path}")
                return None
            
            # For Darts models
            if model_group in ['deep_learning', 'regression']:
                from darts.models import load_model
                return load_model(load_path)
            
            # For other models (using joblib)
            import joblib
            return joblib.load(load_path)
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None

    def evaluate_sequence_predictions(self, predictions: List[TimeSeries], 
                                   actuals: List[TimeSeries]) -> Dict[str, Dict[str, float]]:
        """Evaluate sequence predictions using both all-points and last-point methods
        
        Args:
            predictions: List of prediction TimeSeries
            actuals: List of actual TimeSeries
            horizon: Prediction horizon
        
        Returns:
            Dictionary containing both all-points and last-point metrics
        """
        # Initialize metrics containers
        all_points_metrics = []
        last_point_metrics = []
        
        # 1. All-points evaluation (all 980 points)
        for pred, true in zip(predictions, actuals):
            metrics = self.evaluate_predictions(pred, true)
            all_points_metrics.append(metrics)
        
        # 2. Last-point evaluation (140 points - last point of each sequence)
        for pred, true in zip(predictions, actuals):
            last_pred = TimeSeries.from_values(pred.values()[-1])
            last_true = TimeSeries.from_values(true.values()[-1])
            metrics = self.evaluate_predictions(last_pred, last_true)
            last_point_metrics.append(metrics)
        
        # Calculate average metrics
        avg_all_points = {
            metric: np.mean([m[metric] for m in all_points_metrics])
            for metric in ['mae', 'rmse']
        }
        
        avg_last_point = {
            metric: np.mean([m[metric] for m in last_point_metrics])
            for metric in ['mae', 'rmse']
        }
        
        return {
            'all_points': avg_all_points,
            'last_point': avg_last_point
        }

    def transform_predictions_to_dataframe(self, predictions: List[TimeSeries], last_only: bool = False) -> pd.DataFrame:
        """Transform predictions into a structured DataFrame format.
        
        Args:
            predictions: List of TimeSeries predictions
            last_only: If True, only return the last point of each sequence
            
        Returns:
            DataFrame with:
            - If last_only=False: 3D structure (sequence_start_time × horizon × variables)
            - If last_only=True: 2D structure (sequence_start_time × variables)
        """
        if not predictions:
            raise ValueError("Empty predictions list")
        
        # Get basic information
        n_sequences = len(predictions)
        horizon = len(predictions[0])
        n_components = predictions[0].width
        components = predictions[0].components
        
        if last_only:
            # Create 2D DataFrame for last points only
            data = {
                components[i]: [pred.values()[-1][i] for pred in predictions]
                for i in range(n_components)
            }
            
            # Create index using the sequence start time + horizon
            index = [pred.start_time() + pd.Timedelta(days=horizon-1) for pred in predictions]
            
            df = pd.DataFrame(data, index=index)
            df.index.name = 'time'
            
        else:
            # Create 3D DataFrame using MultiIndex
            rows = []
            
            for seq_idx, pred in enumerate(predictions):
                sequence_start = pred.start_time()
                
                for step in range(horizon):
                    row = {
                        'sequence_start_time': sequence_start,
                        'horizon_step': step,
                    }
                    
                    # Add values for each component
                    for comp_idx in range(n_components):
                        row[components[comp_idx]] = pred.values()[step][comp_idx]
                    
                    rows.append(row)
            
            # Create DataFrame with MultiIndex
            df = pd.DataFrame(rows)
            df = df.set_index(['sequence_start_time', 'horizon_step'])
        
        return df

    def generate_results_csv(self, json_path: Union[str, Path], output_path: Union[str, Path]):
        """Convert results JSON to a flattened CSV format

        Args:
            json_path: Path to the results JSON file
            output_path: Path to save the CSV file
        """
        # Convert string paths to Path objects if needed
        json_path = Path(json_path)
        output_path = Path(output_path)

        # Read JSON file
        with open(json_path, 'r') as f:
            results = json.load(f)

        # Initialize list to store flattened records
        records = []

        # Iterate through the nested structure
        for dataset, dataset_data in results['datasets'].items():
            for horizon, horizon_data in dataset_data['horizons'].items():
                for group_name, group_data in horizon_data['model_groups'].items():
                    for model_name, model_data in group_data['models'].items():
                        record = {
                            'dataset': dataset,
                            'horizon': horizon,
                            'model_group': group_name,
                            'model': model_name,
                            'timestamp': model_data['timestamp'],
                            'training_time': model_data['training_time']
                        }

                        # Add all metrics (both all_points and last_point)
                        for metric_type, metric_values in model_data['metrics'].items():
                            for metric_name, value in metric_values.items():
                                record[f'{metric_type}_{metric_name}'] = value

                        records.append(record)

        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(records)

        # Reorder columns for better readability
        base_columns = ['dataset', 'horizon', 'model_group', 'model', 'timestamp', 'training_time']
        metric_columns = [col for col in df.columns if col not in base_columns]
        df = df[base_columns + sorted(metric_columns)]

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        return df

