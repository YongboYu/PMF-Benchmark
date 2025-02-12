from darts.models import (
    NaiveMean,
    NaiveSeasonal,
    NaiveDrift,
    NaiveMovingAverage
)
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import time


class BaselineModels:
    def __init__(self, config: Union[Dict[str, Any], str], model_name: Optional[str] = None):
        """Initialize baseline models.

        Args:
            config: Either config dictionary or path to base config file
            model_name: Optional name of a specific model to initialize
        """
        # Load configurations
        if isinstance(config, dict):
            self.base_config = config
            self.model_config = config['model_configs']['baseline_models']
        else:
            with open(config, 'r') as f:
                self.base_config = yaml.safe_load(f)
            with open(Path(config).parent / 'model_configs' / 'baseline_models.yaml', 'r') as f:
                self.model_config = yaml.safe_load(f)

        # Initialize models
        self.models = {}
        if model_name:
            # Initialize single model if specified
            model_info = self.model_config['models'].get(model_name)
            if not model_info or not model_info['enabled']:
                raise ValueError(f"Model {model_name} not found or not enabled")
            self.models[model_name] = self._initialize_model(model_name, model_info)
        else:
            # Initialize all enabled models
            for model_name, model_info in self.model_config['models'].items():
                if model_info['enabled']:
                    self.models[model_name] = self._initialize_model(model_name, model_info)

        self.training_time = 0

    def _initialize_model(self, model_name: str, model_info: Dict[str, Any]):
        """Initialize baseline model with or without parameters"""
        params = model_info.get('params', {})

        if model_name == "naive_mean":
            return NaiveMean()
        elif model_name == "persistence":
            return NaiveSeasonal(**params)
        elif model_name == "naive_seasonal":
            return NaiveSeasonal(**params)
        elif model_name == "naive_drift":
            return NaiveDrift()
        elif model_name == "naive_moving_average":
            return NaiveMovingAverage(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train_and_predict(self, model_name: str, train, val, test, transformer,
                          horizon: int, study: Optional[Any] = None) -> Dict[str, Any]:
        """Train models and generate predictions

        Args:
            model_name: Name of the specific model to train
            train: Training data
            val: Validation data (not used for baseline models)
            test: Test data
            transformer: Data transformer (not used for baseline models)
            horizon: Forecast horizon
            study: Optional; Optuna study (not used for baseline models)

        Returns:
            Dictionary containing predictions, trained model, and training info
        """
        start_time = time.time()

        try:
            model = self.models[model_name]
            # Train model
            model.fit(train)

            # Generate predictions
            pred = model.predict(len(test))

            # Store results
            training_time = time.time() - start_time
            return {
                'predictions': pred,
                'model': model,
                'training_time': training_time,
                'model_name': model_name
            }

        except Exception as e:
            error_msg = f"Error training {model_name}: {str(e)}"
            print(error_msg)
            wandb.log({f"baseline_{model_name}_error": error_msg})
            raise

    def get_model_names(self) -> List[str]:
        """Get list of enabled model names."""
        return list(self.models.keys())
