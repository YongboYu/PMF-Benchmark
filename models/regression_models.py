import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import yaml
import time

import optuna
import wandb
from darts.models import (
    LinearRegressionModel,
    RandomForest,
    XGBModel,
    LightGBMModel
)
from utils.optuna_manager import OptunaManager
from darts.metrics import mae
from utils.wandb_logger import WandbLogger

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

class RegressionModels:
    def __init__(self, config: Union[Dict[str, Any], str]):
        """Initialize regression models with configuration
        
        Args:
            config: Either config dictionary or path to config file
        """
        # Load configurations
        if isinstance(config, dict):
            self.config = config['model_configs']['regression_models']
            self.full_config = config  # Store full config for OptunaManager
        else:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
                self.full_config = self.config
            
        # Initialize OptunaManager
        self.optuna_manager = OptunaManager(self.full_config)
        
        # Map model names to their Darts implementations
        self.models = {
            "linear": {
                "model": LinearRegressionModel,
                "enabled": self.config['models']['linear']['enabled'],
                "params": {}
            },
            "random_forest": {
                "model": RandomForest,
                "enabled": self.config['models']['random_forest']['enabled'],
                "params": self.config['models']['random_forest']['hyperparameter_ranges']
            },
            "xgboost": {
                "model": XGBModel,
                "enabled": self.config['models']['xgboost']['enabled'],
                "params": self.config['models']['xgboost']['hyperparameter_ranges']
            },
            "lightgbm": {
                "model": LightGBMModel,
                "enabled": self.config['models']['lightgbm']['enabled'],
                "params": self.config['models']['lightgbm']['hyperparameter_ranges']
            }
        }

        self.wandb_logger = None

    def _get_model_params(self, trial: optuna.Trial, model_name: str, horizon: int) -> Dict[str, Any]:
        """Get hyperparameters for a specific model from Optuna trial"""
        params = {}
        param_ranges = self.models[model_name]["params"]
        
        # Get lags as a hyperparameter to optimize
        min_lag = self.config['common']['lags']['min_lag']
        max_lag = self.config['common']['lags']['max_lag']
        step_size = self.config['common']['lags']['step_size']
        
        # Create list of possible lag values
        possible_lags = list(range(min_lag, max_lag + 1, step_size))
        
        # Always include lags for all models
        params['lags'] = trial.suggest_categorical('lags', possible_lags)
        
        # Always add output_chunk_length parameter (not optimized)
        params['output_chunk_length'] = horizon
        
        # Get model-specific hyperparameters
        if param_ranges:
            for param_name, param_values in param_ranges.items():
                if isinstance(param_values, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple):
                    if any(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name, *param_values)
                    else:
                        params[param_name] = trial.suggest_int(param_name, *param_values)
                    
        return params

    def objective(self, trial: optuna.Trial, model_name: str, model_class,
                 train, val, horizon: int) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            params = self._get_model_params(trial, model_name, horizon)
            model = model_class(**params)
            model.fit(train)
            val_pred = model.predict(n=len(val))

            val_score = mae(val, val_pred)
            
            # Log metrics and parameters
            if self.wandb_logger is not None:
                self.wandb_logger.log_trial_metrics(
                    metrics={'mae': val_score},
                    params=params,
                    model_name=model_name
                )
            
            return val_score
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def train_and_predict(self, model_name: str, train, val, test, transformer,
                         horizon: int, dataset: str, study: Optional[optuna.Study] = None,
                         wandb_logger: WandbLogger = None) -> Dict[str, Any]:
        """Train model and generate predictions"""
        # Set wandb_logger as instance attribute
        self.wandb_logger = wandb_logger

        if not self.models[model_name]["enabled"]:
            logger.info(f"Model {model_name} is disabled in config")
            return {}

        wandb.log({f"regression_training_model": model_name})
        best_params = {}

        try:
            # Create study using OptunaManager if not provided
            if study is None:
                study = self.optuna_manager.create_study(
                    dataset=dataset,
                    horizon=horizon,
                    model_group='regression',
                    model_name=model_name
                )
            
            # Get wandb callback if logger is provided
            wandb_callback = None
            if wandb_logger is not None:
                wandb_callback = wandb_logger.log_optuna_study(study, model_name, dataset, horizon)
            
            # Run optimization with callback if available
            optimize_args = {
                'func': lambda trial: self.objective(
                    trial,
                    model_name,
                    self.models[model_name]["model"],
                    train,
                    val,
                    horizon
                ),
                'n_trials': self.config['common']['n_trials'],
                'catch': (Exception,)
            }
            
            if wandb_callback is not None:
                optimize_args['callbacks'] = [wandb_callback]
            
            study.optimize(**optimize_args)

            # Log study results after optimization
            if wandb_logger is not None:
                wandb_logger.log_study_results(study, model_name, dataset, horizon)

            # Get the best parameters if available
            if study.best_trial is not None:
                best_params = study.best_params
                best_params['output_chunk_length'] = horizon
                
                if 'lags' not in best_params:
                    best_params['lags'] = study.best_trial.params.get('lags')

                # Combine train and validation sets for final training
                combined_train = train
                if val is not None:
                    combined_train = train.concatenate(val)

                # Train final model on combined data
                model = self.models[model_name]["model"](**best_params)
                model.fit(combined_train)
                pred = model.predict(n=len(test))

                wandb.log({
                    f"regression_{model_name}_best_params": best_params,
                    f"regression_{model_name}_trained": True,
                    f"regression_{model_name}_n_trials": len(study.trials),
                    f"regression_{model_name}_best_value": study.best_value
                })

                return {
                    'predictions': pred,
                    'model': model,
                    'best_params': best_params,
                    'study': study
                }
            
            # Return empty results if no successful trials
            return {
                'predictions': None,
                'model': None,
                'best_params': {},
                'study': study
            }

        except Exception as e:
            error_msg = f"Error training {model_name}: {str(e)}"
            logger.error(error_msg)
            wandb.log({
                f"regression_{model_name}_error": error_msg,
                f"regression_{model_name}_failed": True
            })
            raise

    def get_model_names(self) -> list:
        """Return list of enabled model names"""
        return [name for name, info in self.models.items() if info["enabled"]]

