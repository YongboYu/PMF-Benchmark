from darts.models import (
    AutoARIMA,
    ExponentialSmoothing,
    TBATS,
    Theta,
    FourTheta,
    Prophet
)
from darts.utils.utils import ModelMode, SeasonalityMode

import wandb


class StatisticalModels:
    def __init__(self):
        self.models = {
            "auto_arima": AutoARIMA(seasonal=True, m=7),
            "exp_smoothing": ExponentialSmoothing(
                trend=ModelMode.ADDITIVE,
                seasonal=SeasonalityMode.MULTIPLICATIVE
            ),
            "tbats": TBATS(use_trend=True, use_box_cox=False),
            "theta": Theta(theta=2),
            "four_theta": FourTheta(theta=2),
            "prophet": Prophet()
        }

    def train_and_predict(self, train, horizon: int):
        predictions = {}
        for name, model in self.models.items():
            # Train univariate models on each component
            component_preds = []
            for component in train.components:
                model.fit(train[component])
                pred = model.predict(horizon)
                component_preds.append(pred)

            # Combine predictions
            predictions[name] = component_preds
            wandb.log({f"statistical_{name}_trained": True})

        return predictions


# models/statistical_models.py
# models/statistical_models.py
from darts.models import (
    AutoARIMA,
    ExponentialSmoothing,
    TBATS,
    Theta,
    FourTheta,
    Prophet
)
from darts.utils.utils import ModelMode, SeasonalityMode
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging


class StatisticalModels:
    def __init__(self, base_config_path: str = "config/base_config.yaml",
                 model_config_path: str = "config/model_configs/statistical_models.yaml"):
        # Load configurations
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize models based on config
        self.models = {}
        for model_name, model_info in self.model_config['models'].items():
            if model_info['enabled']:
                self.models[model_name] = self._initialize_model(model_name, model_info)

    def _initialize_model(self, model_name: str, model_info: Dict[str, Any]):
        """Initialize statistical model with parameters"""
        params = model_info.get('params', {})

        try:
            if model_name == "auto_arima":
                return AutoARIMA(**params)

            elif model_name == "exp_smoothing":
                # Convert string parameters to ModelMode enums
                if 'trend' in params:
                    params['trend'] = (ModelMode.ADDITIVE
                                       if params['trend'] == 'additive'
                                       else ModelMode.MULTIPLICATIVE)
                if 'seasonal' in params:
                    params['seasonal'] = (SeasonalityMode.ADDITIVE
                                          if params['seasonal'] == 'additive'
                                          else SeasonalityMode.MULTIPLICATIVE)
                return ExponentialSmoothing(**params)

            elif model_name == "tbats":
                return TBATS(**params)

            elif model_name == "theta":
                return Theta(**params)

            elif model_name == "four_theta":
                return FourTheta(**params)

            elif model_name == "prophet":
                return Prophet(**params)

            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            self.logger.error(f"Error initializing {model_name}: {str(e)}")
            raise

    def train_and_predict(self, train, test, horizon: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train models and generate predictions"""
        predictions = {}
        model_info = {}

        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            wandb.log({f"statistical_training_model": name})

            try:
                # Train model
                model.fit(train)

                # Generate predictions
                pred = model.predict(horizon)
                predictions[name] = pred

                # Get model-specific information
                model_info[name] = {
                    "fitted": True,
                    "model_type": name
                }

                # Add specific model parameters if available
                if hasattr(model, "get_params"):
                    model_info[name]["parameters"] = model.get_params()

                # Save model if configured
                if self.base_config['evaluation'].get('save_models', False):
                    model_path = Path(self.model_config['paths']['model_save_dir']) / f"{name}_h{horizon}.pkl"
                    model.save(model_path)
                    model_info[name]["model_path"] = str(model_path)

                # Log to wandb
                wandb.log({
                    f"statistical_{name}_trained": True,
                    f"statistical_{name}_info": model_info[name]
                })

            except Exception as e:
                error_msg = f"Error training {name}: {str(e)}"
                self.logger.error(error_msg)
                wandb.log({f"statistical_{name}_error": error_msg})
                model_info[name] = {"error": error_msg}

        return predictions, model_info