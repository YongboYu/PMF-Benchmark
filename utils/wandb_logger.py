import wandb
import optuna
import plotly
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice
)


class WandbLogger:
    def __init__(self, config: Dict[str, Any], project_name: str):
        self.config = config
        self.project_name = project_name

    def init_run(self, run_name: Optional[str] = None):
        """Initialize wandb run"""
        wandb.init(
            project=self.project_name,
            config=self.config,
            name=run_name,
            reinit=True
        )

    def log_optuna_study(self, study: optuna.Study, model_name: str):
        """Log Optuna study visualizations to wandb"""
        # Optimization history
        fig_history = plot_optimization_history(study)
        wandb.log({f"{model_name}/optimization_history": self._plotly_to_wandb(fig_history)})

        # Parameter importance
        fig_importance = plot_param_importances(study)
        wandb.log({f"{model_name}/parameter_importance": self._plotly_to_wandb(fig_importance)})

        # Parallel coordinate plot
        fig_parallel = plot_parallel_coordinate(study)
        wandb.log({f"{model_name}/parallel_coordinate": self._plotly_to_wandb(fig_parallel)})

        # Parameter relationships
        fig_slice = plot_slice(study)
        wandb.log({f"{model_name}/parameter_slice": self._plotly_to_wandb(fig_slice)})

        # Log hyperparameter importance as table
        param_importance = optuna.importance.get_param_importances(study)
        importance_table = wandb.Table(
            columns=["parameter", "importance"],
            data=[[param, importance] for param, importance in param_importance.items()]
        )
        wandb.log({f"{model_name}/hyperparameter_importance": importance_table})

    def log_predictions(self, predictions: Dict[str, Any], actual: Any, model_name: str):
        """Log prediction plots"""
        # Create prediction vs actual plot
        fig, ax = plt.subplots(figsize=(12, 6))
        actual.plot(ax=ax, label='Actual')
        for name, pred in predictions.items():
            pred.plot(ax=ax, label=f'{name}_prediction')
        plt.legend()
        plt.title(f'{model_name} Predictions vs Actual')
        wandb.log({f"{model_name}/predictions": wandb.Image(fig)})
        plt.close()

    def log_metrics(self, metrics: Dict[str, float], model_name: str):
        """Log evaluation metrics"""
        wandb.log({f"{model_name}/{k}": v for k, v in metrics.items()})

    def log_model_config(self, model_config: Dict[str, Any], model_name: str):
        """Log model configuration"""
        wandb.config.update({f"{model_name}_config": model_config}, allow_val_change=True)

    def log_training_progress(self, epoch: int, metrics: Dict[str, float], model_name: str):
        """Log training progress metrics"""
        wandb.log({
            f"{model_name}/epoch": epoch,
            **{f"{model_name}/{k}": v for k, v in metrics.items()}
        })

    def _plotly_to_wandb(self, fig: plotly.graph_objs.Figure) -> wandb.Image:
        """Convert plotly figure to wandb image"""
        return wandb.Image(fig.to_image(format="png"))

    def finish(self):
        """End wandb run"""
        wandb.finish()