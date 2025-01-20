from darts.models import (
    NaiveMean,
    NaiveSeasonal,
    NaiveDrift,
    NaiveMovingAverage
)
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class BaselineModels:
    def __init__(self, base_config_path: str = "config/base_config.yaml",
                 model_config_path: str = "config/model_configs/baseline_models.yaml"):
        # Load configurations
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)

        # Create directories
        for path in self.model_config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Initialize models based on config
        self.models = {}
        for model_name, model_info in self.model_config['models'].items():
            if model_info['enabled']:
                self.models[model_name] = self._initialize_model(model_name, model_info)

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

    def train_and_predict(self, train, test, horizon: int, transformer) -> Dict[str, Any]:
        """Train models and generate predictions"""
        predictions = {}

        for name, model in self.models.items():
            wandb.log({f"baseline_training_model": name})

            try:
                # Train model
                model.fit(train)

                # Generate predictions
                pred = model.predict(horizon)
                predictions[name] = pred

                # Save predictions
                pred_path = Path(self.model_config['paths']['predictions_dir']) / f"{name}_h{horizon}.csv"
                pred.to_csv(pred_path)

                # Log to wandb
                wandb.log({
                    f"baseline_{name}_trained": True,
                    f"baseline_{name}_predictions_saved": str(pred_path)
                })

            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                wandb.log({f"baseline_{name}_error": str(e)})

        return predictions