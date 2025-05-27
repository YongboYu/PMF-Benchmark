import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import yaml
import time
import numpy as np
import pandas as pd
import torch

import optuna
import wandb
from darts.models import RNNModel
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

from utils.optuna_manager import OptunaManager
from darts.metrics import mae
from utils.wandb_logger import WandbLogger
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from darts.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class CovariateDLModels:
    def __init__(self, config: Union[Dict[str, Any], str]):
        """Initialize covariate deep learning models with configuration

        Args:
            config: Either config dictionary or path to config file
        """
        # Load configurations
        if isinstance(config, dict):
            self.config = config['model_configs']['covariate_dl_models']
            self.full_config = config
        else:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
                self.full_config = self.config

        # Initialize utilities
        self.optuna_manager = OptunaManager(self.full_config)
        self.data_loader = DataLoader(self.full_config)
        self.evaluator = Evaluator(self.full_config)

        # Map model names to their Darts implementations
        self.models = {
            "cov_rnn": {
                "model": RNNModel,
                "enabled": self.config['models']['rnn']['enabled'],
                "params": self.config['models']['rnn']['hyperparameter_ranges']
            },
            "cov_deepar": {
                "model": RNNModel,
                "enabled": self.config['models']['deepar']['enabled'],
                "params": self.config['models']['deepar']['hyperparameter_ranges']
            }
        }

        self.wandb_logger = None
        
        # Set default device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _prepare_covariates(self, series: TimeSeries, target_component: str) -> TimeSeries:
        """Prepare covariates by stacking all other components except the target"""
        covariate_components = [comp for comp in series.components if comp != target_component]
        if not covariate_components:
            return None
        
        covariates = series[covariate_components[0]]
        for comp in covariate_components[1:]:
            covariates = covariates.stack(series[comp])
            
        return covariates

    def _get_model_params(self, trial: optuna.Trial, model_name: str, horizon: int) -> Dict[str, Any]:
        """Get hyperparameters for a specific model from Optuna trial"""
        params = {}
        param_ranges = self.models[model_name]["params"]

        # Get horizon-specific lags configuration
        horizon_key = f'horizon_{horizon}'
        input_chunk_config = self.config['common']['input_chunk_length'][horizon_key]

        # Create list of possible input length values
        possible_input_lengths = list(range(
            input_chunk_config['min_length'],
            input_chunk_config['max_length'] + 1,
            input_chunk_config['step_size']
        ))

        # For deep learning models, use input_chunk_length instead of lags
        input_chunk_length = trial.suggest_categorical('input_chunk_length', possible_input_lengths)
        params['input_chunk_length'] = input_chunk_length

        # Always add output_chunk_length parameter (not optimized)
        params['training_length'] = input_chunk_length + horizon

        # # Add device parameter
        # params['force_reset'] = True
        # params['save_checkpoints'] = True
        # params['device'] = self.device
        
        # Get training parameters
        params['n_epochs'] = self.config['common']['n_epochs']
        params['batch_size'] = trial.suggest_categorical('batch_size', 
                                                        self.config['common']['batch_size'])
        params['dropout'] = trial.suggest_categorical('dropout', self.config['common']['dropout'])

        # Set optimizer parameters
        params['optimizer_kwargs'] = {
            'lr': trial.suggest_categorical('lr', self.config['common']['training']['optimizer_kwargs']['lr'])
        }

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

    def _ensure_float32(self, data):
        """Ensure data is in float32 format"""
        if isinstance(data, TimeSeries):
            return data.astype(np.float32)
        return data

    def objective(self, trial: optuna.Trial, model_name: str, model_class,
                 train_component: TimeSeries, train_covariates: TimeSeries,
                 val_component: Optional[TimeSeries], val_covariates: Optional[TimeSeries],
                 horizon: int) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            params = self._get_model_params(trial, model_name, horizon)
            model = model_class(**params)
            model.fit(train_component, past_covariates=train_covariates, val_series=val_component, 
                     val_past_covariates=val_covariates)

            if val_component is None:
                # If no validation set, use a portion of training data
                train_size = int(len(train_component) * 0.8)
                val_component = train_component[train_size:]
                val_covariates = train_covariates[train_size:]
                train_component_subset = train_component[:train_size]
                train_covariates_subset = train_covariates[:train_size]
                model.fit(train_component_subset, past_covariates=train_covariates_subset)

            # Create seq2seq validation dataset
            val_input_seq, val_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train_component,
                val=val_component,
                test=None,
                input_length=params['input_chunk_length'],
                output_length=horizon,
                dataset_type='val'
            )

            # Create corresponding covariate sequences
            val_cov_seq, _ = self.data_loader.create_seq2seq_io_data(
                train=train_covariates,
                val=val_covariates,
                test=None,
                input_length=params['input_chunk_length'],
                output_length=horizon,
                dataset_type='val'
            )

            # Generate predictions
            val_predictions = []
            for input_seq, cov_seq in zip(val_input_seq, val_cov_seq):
                pred = model.predict(n=horizon, series=input_seq, past_covariates=cov_seq)
                val_predictions.append(pred)

            # Calculate average MAE
            val_scores = []
            for pred, true in zip(val_predictions, val_output_seq):
                val_scores.append(mae(true, pred))
            val_score = np.mean(val_scores)

            wandb.log({'val_mae': val_score})

            return val_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def train_and_predict(self, model_name: str, train, val, test, transformer,
                         horizon: int, dataset: str, study: Optional[optuna.Study] = None,
                         wandb_logger: WandbLogger = None) -> Dict[str, Any]:
        """Train separate models for each component using other components as covariates"""
        self.wandb_logger = wandb_logger

        if not self.models[model_name]["enabled"]:
            logger.info(f"Model {model_name} is disabled in config")
            return {}

        training_start_time = time.time()
        wandb_kwargs = self.config.get('wandb', {}).copy()

        try:
            # Convert data to float32
            train = self._ensure_float32(train)
            if val is not None:
                val = self._ensure_float32(val)
            test = self._ensure_float32(test)

            # Train separate model for each component
            all_component_predictions = []
            trained_models = []

            for target_component in train.components:
                logger.info(f"Training covariate {model_name} for target: {target_component}")

                # Prepare covariates for each dataset split
                train_covariates = self._prepare_covariates(train, target_component)
                val_covariates = self._prepare_covariates(val, target_component) if val is not None else None
                test_covariates = self._prepare_covariates(test, target_component)

                # Create study for this component
                component_study = self.optuna_manager.create_study(
                    dataset=f"{dataset}_{target_component}",
                    horizon=horizon,
                    model_group='covariate_dl',
                    model_name=f"cov_{model_name}"
                )

                wandb_callback = WeightsAndBiasesCallback(
                    metric_name=f'val_mae_{target_component}',
                    wandb_kwargs=wandb_kwargs
                )

                # Optimize for this component
                component_study.optimize(
                    func=lambda trial: self.objective(
                        trial,
                        model_name,
                        self.models[model_name]["model"],
                        train[target_component],
                        train_covariates,
                        val[target_component] if val is not None else None,
                        val_covariates,
                        horizon
                    ),
                    n_trials=self.config['common']['n_trials'],
                    callbacks=[wandb_callback]
                )

                # Log optimization results
                try:
                    wandb.log({
                        f"cov_{model_name}/{target_component}/optimization_history": plot_optimization_history(component_study)
                    })
                except Exception as e:
                    logger.warning(f"Could not plot optimization history for {target_component}: {str(e)}")

                # Train final model with best parameters
                best_params = component_study.best_params
                best_params['training_length'] = best_params['input_chunk_length'] + horizon
                best_params['lags_past_covariates'] = best_params['input_chunk_length']
                # best_params['force_reset'] = True
                # best_params['save_checkpoints'] = True
                # best_params['device'] = self.device
                best_params['n_epochs'] = self.config['common']['n_epochs']

                lr = best_params.pop('lr')
                best_params['optimizer_kwargs'] = {'lr': lr}

                # Train on combined data
                if val is not None:
                    combined_train_component = train[target_component].concatenate(val[target_component])
                    combined_train_covariates = train_covariates.concatenate(val_covariates)
                else:
                    combined_train_component = train[target_component]
                    combined_train_covariates = train_covariates

                component_model = self.models[model_name]["model"](**best_params)
                component_model.fit(combined_train_component, past_covariates=combined_train_covariates)

                # Generate test predictions using seq2seq approach
                test_input_seq, test_output_seq = self.data_loader.create_seq2seq_io_data(
                    train=train[target_component],
                    val=val[target_component] if val is not None else None,
                    test=test[target_component],
                    input_length=best_params['input_chunk_length'],
                    output_length=horizon,
                    dataset_type='test'
                )

                test_cov_seq, _ = self.data_loader.create_seq2seq_io_data(
                    train=train_covariates,
                    val=val_covariates if val is not None else None,
                    test=test_covariates,
                    input_length=best_params['input_chunk_length'],
                    output_length=horizon,
                    dataset_type='test'
                )

                component_predictions = []
                for input_seq, cov_seq in zip(test_input_seq, test_cov_seq):
                    pred = component_model.predict(n=horizon, series=input_seq, past_covariates=cov_seq)
                    component_predictions.append(pred)

                all_component_predictions.append(component_predictions)
                trained_models.append(component_model)

            # Combine predictions from all components
            all_predictions = []
            for seq_idx in range(len(test_input_seq)):
                seq_predictions = [comp_preds[seq_idx] for comp_preds in all_component_predictions]
                combined_pred = seq_predictions[0]
                for pred in seq_predictions[1:]:
                    combined_pred = combined_pred.stack(pred)
                all_predictions.append(combined_pred)

            total_training_time = time.time() - training_start_time
            wandb.log({f"cov_{model_name}_training_time": total_training_time})

            test_input_seq, test_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train,
                val=val,
                test=test,
                input_length=56,  # Default value
                output_length=horizon,
                dataset_type='test'
            )

            return {
                'predictions': all_predictions,
                'actuals': test_output_seq,
                'model': trained_models,
                'training_time': total_training_time,
                'model_name': f"cov_{model_name}"
            }

        except Exception as e:
            error_msg = f"Error training covariate {model_name}: {str(e)}"
            logger.error(error_msg)
            raise

    def get_model_names(self) -> list:
        """Return list of enabled model names"""
        return [name for name, info in self.models.items() if info["enabled"]] 