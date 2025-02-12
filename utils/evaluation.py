from darts.metrics import mae, rmse
from typing import Dict, Any, Optional, Union
from darts import TimeSeries
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import logging


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_base_path = Path(self.config['paths']['results_dir'])

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
        base_path = self.results_base_path
        
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

    def save_predictions(self, predictions: TimeSeries, dataset: str, 
                        model_group: str, model_name: str, horizon: int) -> str:
        """Save model predictions in the new structure"""
        paths = self.get_results_paths(dataset, horizon, model_group)
        save_path = paths['predictions'] / f'{model_name}_predictions.csv'
        predictions.to_csv(save_path)
        return str(save_path)

    def save_metrics(self, metrics_df: pd.DataFrame, dataset: str, 
                    model_group: str, model_name: str, horizon: int) -> str:
        """Save metrics in the new structure"""
        paths = self.get_results_paths(dataset, horizon, model_group)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = paths['evaluation'] / f'{model_name}_metrics.csv'
        metrics_df.to_csv(save_path, index=False)
        return str(save_path)

    def update_results_log(self, dataset: str, results: Dict[str, Any], horizon: int):
        """Update or create results log for dataset and horizon"""
        log_path = self.results_base_path / 'results_log.json'
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                all_results = json.load(f)
                if 'horizons' not in all_results:
                    current_horizon = all_results.get('horizon')
                    current_model_groups = all_results.get('model_groups', {})
                    all_results = {
                        'dataset': all_results.get('dataset'),
                        'last_updated': all_results.get('last_updated', ''),
                        'horizons': {}
                    }
                    if current_horizon is not None:
                        all_results['horizons'][str(current_horizon)] = {
                            'last_updated': all_results.get('last_updated', ''),
                            'model_groups': current_model_groups
                        }
        else:
            all_results = {
                'dataset': dataset,
                'last_updated': '',
                'horizons': {}
            }

        if str(horizon) not in all_results['horizons']:
            all_results['horizons'][str(horizon)] = {
                'last_updated': '',
                'model_groups': {}
            }

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_group = results['model_name']
        specific_model = results.get('specific_model', 'default')

        horizon_dict = all_results['horizons'][str(horizon)]
        
        if model_group not in horizon_dict['model_groups']:
            horizon_dict['model_groups'][model_group] = {
                'last_updated': timestamp,
                'models': {}
            }
        
        horizon_dict['model_groups'][model_group]['last_updated'] = timestamp
        
        horizon_dict['model_groups'][model_group]['models'][specific_model] = {
            'metrics': results['metrics'],
            'training_time': results['training_time'],
            'timestamp': timestamp
        }
        
        horizon_dict['last_updated'] = timestamp
        all_results['last_updated'] = timestamp

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
    def save_results(self, model_results: Dict[str, Any], dataset: str, 
                    horizon: int, model: Any = None, predictions: TimeSeries = None) -> None:
        """Main function to save all results"""
        # Save trained model(s) if provided and supports saving
        if model is not None and self.config['evaluation']['save_models']:
            if isinstance(model, list):  # Handle multiple models (statistical case)
                saved_paths = []
                for idx, component_model in enumerate(model):
                    model_path = self.save_model(
                        model=component_model,
                        dataset=dataset,
                        model_group=model_results['model_name'],
                        model_name=f"{model_results['specific_model']}/{model_results['specific_model']}_component_{idx}",
                        horizon=horizon
                    )
                    if model_path:
                        saved_paths.append(model_path)
                if saved_paths:
                    model_results['model_paths'] = saved_paths
            else:  # Single model case
                model_path = self.save_model(
                    model=model,
                    dataset=dataset,
                    model_group=model_results['model_name'],
                    model_name=model_results.get('specific_model', 'default'),
                    horizon=horizon
                )
                if model_path:
                    model_results['model_path'] = model_path

        # Save predictions if provided
        if predictions is not None and self.config['evaluation']['save_predictions']:
            pred_path = self.save_predictions(
                predictions=predictions,
                dataset=dataset,
                model_group=model_results['model_name'],
                model_name=model_results.get('specific_model', 'default'),
                horizon=horizon
            )
            model_results['predictions_path'] = pred_path

        # Create and save metrics DataFrame if metrics exist
        if 'metrics' in model_results:
            metrics_df = pd.DataFrame([{
                'dataset': dataset,
                'horizon': horizon,
                'model': f"{model_results['model_name']}_{model_results.get('specific_model', 'default')}",
                **model_results['metrics']
            }])
            metrics_path = self.save_metrics(
                metrics_df=metrics_df,
                dataset=dataset,
                model_group=model_results['model_name'],
                model_name=model_results.get('specific_model', 'default'),
                horizon=horizon
            )
            model_results['metrics_path'] = metrics_path

        # Update results log
        self.update_results_log(dataset, model_results, horizon)

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


def generate_results_csv(json_path: Path, output_path: Path):
    """Convert results JSON to a flattened CSV format
    
    Args:
        json_path: Path to the results JSON file
        output_path: Path to save the CSV file
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Initialize list to store flattened records
    records = []
    
    # Extract dataset name
    dataset = results['dataset']
    
    # Iterate through horizons
    for horizon, horizon_data in results['horizons'].items():
        # Iterate through model groups and their models
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
                
                # Add all metrics
                for metric_name, metric_value in model_data['metrics'].items():
                    record[metric_name] = metric_value
                
                records.append(record)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(records)
    
    # Reorder columns for better readability
    column_order = ['dataset', 'horizon', 'model_group', 'model', 'timestamp', 'training_time'] + \
                  [col for col in df.columns if col not in ['dataset', 'horizon', 'model_group', 'model', 'timestamp', 'training_time']]
    df = df[column_order]
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df

