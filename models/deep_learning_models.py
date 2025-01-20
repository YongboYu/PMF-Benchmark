# models/deep_learning_models.py
from darts.models import (
    RNNModel,
    NBEATSModel,
    NHiTSModel,
    TCNModel,
    TransformerModel,
    TFTModel,
    DLinearModel,
    NLinearModel,
)
from darts.utils.likelihood_models import GaussianLikelihood
from darts.metrics import mae
import torch
import optuna
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging


class DeepLearningModels:
    def __init__(self, base_config_path: str = "config/base_config.yaml",
                 model_config_path: str = "config/model_configs/deep_learning_models.yaml"):
        # Load configurations
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create directories
        for path in self.model_config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def _get_model_class(self, model_name: str):
        """Get the appropriate model class"""
        return {
            "rnn": RNNModel,
            "lstm": RNNModel,
            "gru": RNNModel,
            "deepar": RNNModel,
            "nbeats": NBEATSModel,
            "nhits": NHiTSModel,
            "tcn": TCNModel,
            "transformer": TransformerModel,
            "tft": TFTModel,
            "dlinear": DLinearModel,
            "nlinear": NLinearModel
        }[model_name]

    def _get_model_params(self, trial: optuna.Trial, model_name: str, common_params: dict) -> dict:
        """Get model-specific parameters"""
        model_params = self.model_config['models'][model_name]['hyperparameter_ranges']
        params = common_params.copy()

        # Get model-specific parameters
        for param_name, param_range in model_params.items():
            params[param_name] = trial.suggest_categorical(param_name, param_range)

        return params

    def _objective(self, trial: optuna.Trial, model_name: str, train, val, horizon: int, transformer)  -> float:
        """Optuna objective function for hyperparameter optimization"""
        # Get horizon-specific lag range with step interval
        horizon_key = f"horizon_{horizon}"
        lag_config = self.model_config['common']['lags'][horizon_key]

        # Create list of possible lag values based on min, max, and step
        lag_values = list(range(
            lag_config['min_lag'],
            lag_config['max_lag'] + 1,
            lag_config['step']
        ))

        # Common parameters
        common_params = {
            'input_chunk_length': trial.suggest_categorical('input_chunk_length', lag_values),
            'output_chunk_length': horizon,
            'batch_size': trial.suggest_categorical('batch_size',
                                                    self.model_config['common']['batch_size']),
            'n_epochs': self.model_config['common']['n_epochs'],
            'dropout': trial.suggest_categorical('dropout',
                                                 self.model_config['common']['dropout']),
            'optimizer_kwargs': {
                'lr': trial.suggest_categorical('lr',
                                                self.model_config['common']['optimizer_kwargs']['lr'])
            }
        }

        # Get model-specific parameters
        params = self._get_model_params(trial, model_name, common_params)

        # Add training_length for RNN-based models
        if model_name in ['rnn', 'lstm', 'gru', 'deepar']:
            training_length = trial.suggest_categorical('training_length',
                                                        self.model_config['common']['training_length'])
            # Ensure training_length is larger than input_chunk_length
            while training_length <= params['input_chunk_length']:
                training_length *= 2
            params['training_length'] = training_length

        try:
            # Initialize model
            model_class = self._get_model_class(model_name)
            model = model_class(
                **params,
                random_state=self.base_config['project']['seed'],
                force_reset=True,
                save_checkpoints=True,
                pl_trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1
                }
            )

            # Train model
            model.fit(train, val_series=val)
            val_pred = model.predict(len(val))

            # Inverse transform predictions and validation data before scoring
            val_pred_original = transformer.inverse_transform(val_pred)
            val_original = transformer.inverse_transform(val)

            # Calculate MAE on original scale
            val_score = mae(val_original, val_pred_original)

            return val_score

        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            return float('inf')

    def _train_final_model(self, model_name: str, params: dict, train, val, horizon: int) -> object:
        """Train final model with best parameters"""
        model_class = self._get_model_class(model_name)

        # Add common parameters that weren't part of the optimization
        final_params = {
            **params,
            'random_state': self.base_config['project']['seed'],
            'force_reset': True,
            'save_checkpoints': True,
            'pl_trainer_kwargs': {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1
            }
        }

        # Initialize and train model
        model = model_class(**final_params)
        model.fit(train, val_series=val)

        # Save model if configured
        if self.base_config['evaluation'].get('save_models', False):
            model_path = Path(self.model_config['paths']['model_save_dir']) / f"{model_name}_h{horizon}.pt"
            model.save(model_path)

        return model

    def train_and_predict(self, train, val, test, transformer) -> Dict[str, Dict[str, Any]]:
        """Train models and generate predictions for each horizon"""
        results = {}

        for horizon in self.model_config['common']['horizons']:
            self.logger.info(f"\nTraining models for horizon {horizon}")
            predictions = {}
            best_params = {}

            for model_name in self.model_config['models'].keys():
                if not self.model_config['models'][model_name]['enabled']:
                    continue

                self.logger.info(f"Training {model_name} model...")
                wandb.log({
                    f"dl_training_model": model_name,
                    f"horizon": horizon
                })

                try:
                    # Optimize hyperparameters
                    study = optuna.create_study(direction="minimize")
                    study.optimize(
                        lambda trial: self._objective(trial, model_name, train, val, horizon, transformer),
                        n_trials=self.model_config['common']['n_trials']
                    )

                    # Get best parameters
                    best_params[model_name] = study.best_params

                    # Train final model with best parameters
                    final_model = self._train_final_model(
                        model_name=model_name,
                        params=best_params[model_name],
                        train=train,
                        val=val,
                        horizon=horizon
                    )

                    # Generate predictions
                    pred = final_model.predict(len(test))
                    predictions[model_name] = pred

                    # Log to wandb
                    wandb.log({
                        f"dl_{model_name}_h{horizon}_best_params": best_params[model_name],
                        f"dl_{model_name}_h{horizon}_trained": True
                    })

                except Exception as e:
                    error_msg = f"Error training {model_name}: {str(e)}"
                    self.logger.error(error_msg)
                    wandb.log({f"dl_{model_name}_h{horizon}_error": error_msg})

            results[f"horizon_{horizon}"] = {
                "predictions": predictions,
                "best_params": best_params
            }

        return results

    def load_model(self, model_name: str, horizon: int) -> Optional[object]:
        """Load a saved model if it exists"""
        model_path = Path(self.model_config['paths']['model_save_dir']) / f"{model_name}_h{horizon}.pt"
        if model_path.exists():
            model_class = self._get_model_class(model_name)
            return model_class.load(model_path)
        return None