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
from typing import Dict, Any, Optional, Union, List
import logging
from utils.optuna_manager import OptunaManager
import numpy as np
from darts import TimeSeries
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_pareto_front
from utils.wandb_logger import WandbLogger
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator

logger = logging.getLogger(__name__)

class DeepLearningModels:
    def __init__(self, config: Union[Dict[str, Any], str]):
        """Initialize deep learning models with configuration
        
        Args:
            config: Either config dictionary or path to config file
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load model-specific config
        model_config_path = Path("config/model_configs/deep_learning_models.yaml")
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
            
        self.full_config = self.config
        self.config = self.model_config
        
        # Initialize DataLoader and Evaluator
        self.data_loader = DataLoader(self.full_config)
        self.evaluator = Evaluator(self.full_config)
        
        # Map model names to their Darts implementations
        self.models = {
            "rnn": {
                "model": RNNModel,
                "enabled": self.config['models']['rnn']['enabled'],
                "params": self.config['models']['rnn']['hyperparameter_ranges']
            },
            "lstm": {
                "model": RNNModel,  # LSTM is a type of RNN
                "enabled": self.config['models']['lstm']['enabled'],
                "params": self.config['models']['lstm']['hyperparameter_ranges']
            },
            "gru": {
                "model": RNNModel,  # GRU is a type of RNN
                "enabled": self.config['models']['gru']['enabled'],
                "params": self.config['models']['gru']['hyperparameter_ranges']
            },
            "deepar": {
                "model": RNNModel,  # DeepAR uses RNN base
                "enabled": self.config['models']['deepar']['enabled'],
                "params": self.config['models']['deepar']['hyperparameter_ranges']
            },
            "nbeats": {
                "model": NBEATSModel,
                "enabled": self.config['models']['nbeats']['enabled'],
                "params": self.config['models']['nbeats']['hyperparameter_ranges']
            },
            "nhits": {
                "model": NHiTSModel,
                "enabled": self.config['models']['nhits']['enabled'],
                "params": self.config['models']['nhits']['hyperparameter_ranges']
            },
            "tcn": {
                "model": TCNModel,
                "enabled": self.config['models']['tcn']['enabled'],
                "params": self.config['models']['tcn']['hyperparameter_ranges']
            },
            "transformer": {
                "model": TransformerModel,
                "enabled": self.config['models']['transformer']['enabled'],
                "params": self.config['models']['transformer']['hyperparameter_ranges']
            },
            "tft": {
                "model": TFTModel,
                "enabled": self.config['models']['tft']['enabled'],
                "params": self.config['models']['tft']['hyperparameter_ranges']
            },
            "dlinear": {
                "model": DLinearModel,
                "enabled": self.config['models']['dlinear']['enabled'],
                "params": self.config['models']['dlinear']['hyperparameter_ranges']
            },
            "nlinear": {
                "model": NLinearModel,
                "enabled": self.config['models']['nlinear']['enabled'],
                "params": self.config['models']['nlinear']['hyperparameter_ranges']
            }
        }
        
        self.optuna_manager = OptunaManager(self.full_config)
        self.wandb_logger = None

        # Set device and force float32 precision
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            torch.set_default_dtype(torch.float32)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

    def _get_model_params(self, trial: optuna.Trial, model_name: str, horizon: int) -> Dict[str, Any]:
        """Get hyperparameters for a specific model from Optuna trial"""
        params = {}
        param_ranges = self.models[model_name]["params"]
        
        # Get input chunk length configuration
        input_chunk_config = self.config['common']['input_chunk_length']
        possible_input_lengths = list(range(
            input_chunk_config['min_length'],
            input_chunk_config['max_length'] + 1,
            input_chunk_config['step_size']
        ))
        
        # Set input_chunk_length for all models
        input_chunk_length = trial.suggest_categorical('input_chunk_length', possible_input_lengths)
        params['input_chunk_length'] = input_chunk_length
        
        # Handle RNN-based models (RNN, LSTM, GRU, DeepAR)
        rnn_based_models = ['rnn', 'lstm', 'gru', 'deepar']
        if model_name in rnn_based_models:
            params['training_length'] = input_chunk_length + horizon
        
        # Set output_chunk_length for non-RNN models
        if model_name not in rnn_based_models:
            params['output_chunk_length'] = horizon
        
        # Common parameters for all models
        params['batch_size'] = trial.suggest_categorical('batch_size', self.config['common']['batch_size'])
        params['n_epochs'] = self.config['common']['n_epochs']
        
        # Add dropout only for models that support it (exclude DLinear and NLinear)
        if model_name not in ['dlinear', 'nlinear']:
            params['dropout'] = trial.suggest_categorical('dropout', self.config['common']['dropout'])
        
        # Set optimizer parameters directly in model parameters
        # params['optimizer_cls'] = self.config['common']['training']['optimizer']
        params['optimizer_kwargs'] = {
            'lr': trial.suggest_categorical('lr', self.config['common']['training']['optimizer_kwargs']['lr'])
        }

        # Get model-specific parameters
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
                 train, val, horizon: int, transformer) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            params = self._get_model_params(trial, model_name, horizon)
            model = model_class(**params)
            model.fit(train)
            
            # Create seq2seq validation dataset
            val_input_seq, val_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train, val=val, test=None, 
                input_length=params['input_chunk_length'], 
                output_length=horizon, 
                dataset_type='val'
            )
            
            # Generate predictions using ground truth inputs
            val_predictions = []
            for input_seq in val_input_seq:
                pred = model.predict(n=horizon, series=input_seq)
                val_predictions.append(pred)
            
            # Calculate average MAE across all sequences
            val_scores = []
            for pred, true in zip(val_predictions, val_output_seq):
                val_scores.append(mae(true, pred))
            val_score = np.mean(val_scores)
            
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

    def _ensure_float32(self, data):
        """Ensure data is in float32 format"""
        if isinstance(data, TimeSeries):
            return data.astype(np.float32)
        return data

    def train_and_predict(self, model_name: str, train, val, test, transformer,
                         horizon: int, dataset: str, study: Optional[optuna.Study] = None,
                         wandb_logger: WandbLogger = None) -> Dict[str, Any]:
        """Train model and generate predictions using seq2seq approach"""
        if not self.models[model_name]["enabled"]:
            logger.info(f"Model {model_name} is disabled in config")
            return {}

        # Set wandb_logger as instance attribute
        self.wandb_logger = wandb_logger

        if wandb_logger:
            wandb_logger.log_metrics({"current_model": model_name}, prefix="training")
        
        try:
            # Convert data to float32
            train = self._ensure_float32(train)
            if val is not None:
                val = self._ensure_float32(val)
            test = self._ensure_float32(test)

            # Create study using OptunaManager if not provided
            if study is None:
                study = self.optuna_manager.create_study(
                    dataset=dataset,
                    horizon=horizon,
                    model_group='deep_learning',
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
                    horizon,
                    transformer
                ),
                'n_trials': self.config['common']['n_trials'],
                'catch': (Exception,)
            }
            
            if wandb_logger is not None:
                optimize_args['callbacks'] = [wandb_callback]

            study.optimize(**optimize_args)

            if wandb_logger is not None:
                wandb_logger.log_study_results(study, model_name, dataset, horizon)

            # Get the best parameters and train final model
            best_params = study.best_params

            # Handle RNN-based models (RNN, LSTM, GRU, DeepAR)
            rnn_based_models = ['rnn', 'lstm', 'gru', 'deepar']
            if model_name in rnn_based_models:
                # For RNN-based models, directly set training_length = input_chunk_length + horizon
                best_params['training_length'] = best_params['input_chunk_length'] + horizon

            # Set output_chunk_length for non-RNN models
            if model_name not in rnn_based_models:
                best_params['output_chunk_length'] = horizon

            # Add common parameters from config
            best_params['n_epochs'] = self.config['common']['n_epochs']

            # Extract lr and move it to optimizer_kwargs
            if 'lr' in best_params:
                lr = best_params.pop('lr')
                best_params['optimizer_kwargs'] = {'lr': lr}



            # Train final model on combined data
            combined_train = train.concatenate(val) if val is not None else train
            final_model = self.models[model_name]["model"](**best_params)
            final_model.fit(combined_train)

            # Create seq2seq test dataset for final evaluation
            test_input_seq, test_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train, val=val, test=test,
                input_length=best_params['input_chunk_length'],
                output_length=horizon,
                dataset_type='test'
            )

            # Generate final predictions
            all_predictions = []
            for input_seq in test_input_seq:
                pred = final_model.predict(n=horizon, series=input_seq)
                all_predictions.append(pred)

            # Log the study results
            if wandb_logger:
                wandb_logger.log_study_results(study, model_name, dataset, horizon)

            # Log detailed metrics
            if wandb_logger:
                training_time = study.trials[-1].duration.total_seconds() if study.trials else 0
                wandb_logger.log_metrics({
                    'mae': study.best_value,
                    'n_trials': len(study.trials),
                    'training_time': training_time,
                    'input_chunk_length': best_params.get('input_chunk_length'),
                    'batch_size': best_params.get('batch_size'),
                    'n_epochs': best_params.get('n_epochs'),
                    'learning_rate': best_params.get('optimizer_kwargs', {}).get('lr')
                }, prefix=model_name)

                # Log model artifacts
                wandb_logger.log_model_artifacts(model_name, {
                    'best_params': best_params,
                    'best_value': study.best_value,
                    'n_trials': len(study.trials),
                    'model_type': str(self.models[model_name]["model"].__name__),
                    'device': str(self.device),
                    'training_time': training_time
                })

            return {
                'predictions': all_predictions,
                'actuals': test_output_seq,
                'model': final_model,
                'best_params': best_params,
                'study': study
            }
            
        except Exception as e:
            logger.error(f"An error occurred during training and prediction: {e}")
            if wandb_logger:
                wandb_logger.log_metrics({
                    "error": str(e),
                    "failed": True
                }, prefix=model_name)
            return {}

    def get_model_names(self) -> List[str]:
        """Return list of enabled model names"""
        return [name for name, info in self.models.items() if info["enabled"]]