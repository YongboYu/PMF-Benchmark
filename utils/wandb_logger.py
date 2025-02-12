import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, Optional, Union
from darts import TimeSeries
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import numpy as np


class WandbLogger:
    def __init__(self, config: Dict[str, Any]):
        self.project_name = config['project']['wandb_project']
        self.log_dir = Path('logs/wandb')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trial_metrics = {}
        self.param_history = {}
        self.trial_history = {}
        self._run = None

    @property
    def run(self):
        return self._run

    def init_run(self, dataset: str, horizon: int, model_group: str,
                 model_name: str, config: Dict[str, Any]):
        """Initialize wandb run with consistent structure"""
        self._run = wandb.init(
            project=self.project_name,
            name=f"{dataset}_h{horizon}_{model_name}",
            group=dataset,
            job_type=f"h{horizon}",
            tags=[model_group, model_name],
            config=config,
            reinit=True,
            dir=str(self.log_dir)
        )
        return self._run

    def log_metrics(self, metrics: Dict[str, float], step: Optional[Union[int, float]] = None,
                   prefix: Optional[str] = None):
        """Log metrics with optional prefix"""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # If step is a float, convert it to None to avoid the error
        if isinstance(step, float):
            step = None
        
        wandb.log(metrics, step=step)

    def log_model_artifacts(self, model_name: str, artifacts: Dict[str, Any]):
        """Log model-related artifacts"""
        wandb.log({
            f"{model_name}/best_params": artifacts.get('best_params', {}),
            f"{model_name}/best_value": artifacts.get('best_value', None),
            f"{model_name}/n_trials": artifacts.get('n_trials', 0)
        })

    def log_training_progress(self, model_name: str, metrics: Dict[str, float],
                            epoch: int, is_validation: bool = False):
        """Log training progress metrics"""
        prefix = f"{model_name}/{'val' if is_validation else 'train'}"
        self.log_metrics(metrics, step=epoch, prefix=prefix)

    def log_optuna_visualizations(self, study: optuna.Study, model_name: str):
        """Log Optuna study visualizations"""
        try:
            # Optimization history
            history_fig = plot_optimization_history(study)
            wandb.log({f"{model_name}/optimization_history": wandb.Plot(history_fig)})

            if len(study.trials) > 1:
                # Parameter importance
                importance_fig = plot_param_importances(study)
                wandb.log({f"{model_name}/parameter_importance": wandb.Plot(importance_fig)})

                # Parallel coordinates
                parallel_fig = plot_parallel_coordinate(study)
                wandb.log({f"{model_name}/parallel_coordinates": wandb.Plot(parallel_fig)})

        except Exception as e:
            wandb.log({f"{model_name}/visualization_error": str(e)})

    def finish(self):
        """Finish the current wandb run"""
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log_predictions(self, predictions: TimeSeries, actual: TimeSeries, model_name: str):
        """Log prediction plots for all time series components in a single media object using matplotlib

        Args:
            predictions: Model predictions (TimeSeries or list of TimeSeries)
            actual: Actual values (TimeSeries or list of TimeSeries)
            model_name: Name of the model
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
            actual = [actual]

        # Create subplot grid based on number of components
        n_components = actual[0].width
        n_cols = min(3, n_components)  # Reduced max columns to 3 for wider plots
        n_rows = (n_components + n_cols - 1) // n_cols
        
        # Create figure and subplots with increased width
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))  # Increased width from 20 to 24
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Plot each component
        for i in range(n_components):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            time_series_name = actual[0].columns[i]
            
            for act, pred in zip(actual, predictions):
                # Plot actual values
                ax.plot(act.time_index, 
                       act.univariate_component(time_series_name).values(),
                       'b-', label='Actual' if i == 0 else "_nolegend_",
                       linewidth=2)  # Increased line width
                
                # Plot predictions
                ax.plot(pred.time_index,
                       pred.univariate_component(time_series_name).values(),
                       'r-', label='Forecast' if i == 0 else "_nolegend_",
                       linewidth=2)  # Increased line width
            
            ax.set_title(time_series_name, pad=10)  # Added padding to title
            if row == n_rows - 1:  # Bottom row
                ax.set_xlabel('Time', fontsize=10)
            if col == 0:  # Leftmost column
                ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)  # Lighter grid
            ax.tick_params(axis='both', which='major', labelsize=9)  # Adjusted tick label size

        # Hide empty subplots
        for i in range(n_components, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        # Add legend to the figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=2, fancybox=True, shadow=True, fontsize=10)

        # Adjust layout and title
        fig.suptitle(f"{model_name} Predictions", y=1.02, fontsize=12)
        plt.tight_layout()

        # Log to wandb
        wandb.log({f"{model_name}_predictions": wandb.Image(fig)})
        
        # Close the figure to free memory
        plt.close(fig)

    def log_trial_metrics(self, metrics: Dict[str, float], params: Dict[str, Any], model_name: str):
        """Log metrics and parameters for individual trials"""
        val_score = metrics.get('mae', 0)
        trial_num = metrics.get('trial_number', 0)
        
        # Log hyperparameter values for this trial
        for param_name, param_value in params.items():
            if param_name != 'trial_number':
                wandb.log({
                    f"hyperparameters/{param_name}": param_value,
                    "trial_number": trial_num
                })
        
        # Log validation score
        wandb.log({
            "optimization/val_score": val_score,
            "trial_number": trial_num
        })
        
        # Log scalar metrics
        wandb.log({
            "metrics/min_val_error": min(self.trial_history.get(model_name, {}).get('scores', [val_score])),
            "metrics/training_time": metrics.get('training_time', 0),
            "metrics/rmse": metrics.get('rmse', 0),
            "metrics/mae": metrics.get('mae', 0)
        })

    def _log_parameter_plots(self, model_name: str):
        """Create and log plots for each hyperparameter"""
        for param_name, history in self.param_history.items():
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(history['values'], history['scores'])
            plt.xlabel(param_name)
            plt.ylabel('MAE Score')
            plt.title(f'{param_name} vs MAE')
            
            if isinstance(history['values'][0], (int, float)):
                plt.grid(True)
            else:
                plt.xticks(rotation=45)
            
            wandb.log({f"{model_name}/parameter_plots/{param_name}": wandb.Image(fig)})
            plt.close(fig)

    def log_study_results(self, study: optuna.Study, model_name: str, dataset: str, horizon: int):
        """Log final study results and visualizations"""
        try:
            # Create parallel coordinates plot using wandb
            param_names = list(study.best_params.keys())
            data = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    row = [trial.params[param] for param in param_names]
                    row.append(trial.value)  # Add val_score
                    data.append(row)
            
            wandb.log({"optimization/parallel_coordinates": wandb.plot.parallel_coordinates(
                wandb.Table(data=data, columns=param_names + ["val_score"]),
                param_names + ["val_score"]
            )})

            # Log Optuna visualizations
            history_fig = plot_optimization_history(study)
            importance_fig = plot_param_importances(study)
            parallel_fig = plot_parallel_coordinate(study)
            
            wandb.log({
                "optuna/optimization_history": history_fig,
                "optuna/parameter_importance": importance_fig,
                "optuna/parallel_coordinates": parallel_fig
            })
            plt.close('all')

        except Exception as e:
            wandb.log({f"{model_name}/visualization_error": str(e)})

    def log_optuna_study(self, study: optuna.Study, model_name: str, dataset: str, horizon: int):
        """Create a callback for logging Optuna study progress"""
        def callback(study: optuna.Study, trial: optuna.Trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Log trial information with proper grouping
                wandb.log({
                    "optimization/current_trial_value": trial.value,
                    "optimization/best_value_so_far": study.best_value,
                    "trial_number": trial.number
                })
        return callback