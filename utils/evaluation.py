# utils/evaluation.py
from darts.metrics import mae, rmse
from typing import Dict, Any, Optional
from darts import TimeSeries
from datetime import datetime
from pathlib import Path
import pandas as pd


class Evaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evaluate_predictions(
            self,
            predictions: TimeSeries,
            actuals: TimeSeries,
            transformer: Optional[object] = None
    ) -> Dict[str, float]:
        """Evaluate predictions against actuals"""
        if transformer is not None:
            predictions = transformer.inverse_transform(predictions)
            actuals = transformer.inverse_transform(actuals)

        return {
            'mae': mae(actuals, predictions),
            'rmse': rmse(actuals, predictions)
        }

    def evaluate_model_group(
            self,
            predictions: Dict[str, Dict[str, TimeSeries]],
            test: TimeSeries,
            model_group: str,
            transformer: Optional[object] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all models in a group"""
        results = {}

        for horizon, horizon_preds in predictions.items():
            results[horizon] = {}
            for model_name, pred in horizon_preds['predictions'].items():
                metrics = self.evaluate_predictions(pred, test, transformer)
                results[horizon][f"{model_group}_{model_name}"] = {
                    **metrics,
                    "predictions": transformer.inverse_transform(pred) if transformer else pred
                }

        return results

    def save_results(self, results: Dict[str, Any], dataset: str, time_interval: str):
        """Save evaluation results and predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics to CSV
        metrics_df = []
        for horizon, horizon_results in results.items():
            for model_name, model_results in horizon_results.items():
                metrics_df.append({
                    'dataset': dataset,
                    'time_interval': time_interval,
                    'horizon': horizon,
                    'model': model_name,
                    'mae': model_results['mae'],
                    'rmse': model_results['rmse']
                })

        pd.DataFrame(metrics_df).to_csv(
            results_dir / f'metrics_{dataset}_{time_interval}_{timestamp}.csv',
            index=False
        )

        # Save predictions
        predictions_dir = Path(self.config['paths']['predictions_dir'])
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for horizon, horizon_results in results.items():
            for model_name, model_results in horizon_results.items():
                pred_series = model_results['predictions']
                pred_series.to_csv(
                    predictions_dir / f'pred_{dataset}_{time_interval}_{model_name}_{horizon}_{timestamp}.csv'
                )

