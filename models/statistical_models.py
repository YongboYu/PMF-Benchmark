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
from utils.evaluation import Evaluator
import time
from utils.logging_manager import get_logging_manager


class StatisticalModels:
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        self.base_config = config
        self.model_config = config['model_configs']['statistical_models']
        self.evaluator = Evaluator(config)

        # Setup logging
        self.logger = get_logging_manager(config).get_logger(__name__)

        # Initialize models based on config
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
                # Convert season_mode to SeasonalityMode enum
                if 'season_mode' in params:
                    params['season_mode'] = (SeasonalityMode.ADDITIVE
                                           if params['season_mode'] == 'additive'
                                           else SeasonalityMode.MULTIPLICATIVE)
                return Theta(**params)

            elif model_name == "four_theta":
                # Convert season_mode and model_mode to SeasonalityMode enum
                if 'season_mode' in params:
                    params['season_mode'] = (SeasonalityMode.ADDITIVE
                                           if params['season_mode'] == 'additive'
                                           else SeasonalityMode.MULTIPLICATIVE)
                if 'model_mode' in params:
                    params['model_mode'] = (ModelMode.ADDITIVE
                                          if params['model_mode'] == 'additive'
                                          else ModelMode.MULTIPLICATIVE)
                return FourTheta(**params)

            elif model_name == "prophet":
                return Prophet(**params)

            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            self.logger.error(f"Error initializing {model_name}: {str(e)}")
            raise

    def get_model_names(self) -> list:
        """Return list of enabled model names"""
        return list(self.models.keys())

    def train_and_predict(self, model_name: str, train, val, test, transformer, 
                         horizon: int, study: Optional[Any] = None) -> Dict[str, Any]:
        """Train model and generate predictions for multivariate time series"""
        self.logger.info(f"Training {model_name} model...")
        wandb.log({f"statistical_training_model": model_name})
        
        start_time = time.time()

        try:
            model = self.models[model_name]
            predictions = []
            trained_models = []
            
            # Train separate model for each component
            for component in train.components:
                # Extract univariate series for current component
                train_component = train[component]
                test_component = test[component] if test is not None else None
                val_component = val[component] if val is not None else None
                
                # Create new model instance for each component
                component_model = self._initialize_model(model_name, self.model_config['models'][model_name])
                
                # Train model on component
                component_model.fit(train_component)
                
                # Generate predictions
                pred = component_model.predict(len(test_component))
                predictions.append(pred)
                trained_models.append(component_model)

            # Combine component predictions into multivariate series
            combined_predictions = predictions[0]
            for pred in predictions[1:]:
                combined_predictions = combined_predictions.stack(pred)

            training_time = time.time() - start_time

            # Get model-specific information
            model_info = {
                "fitted": True,
                "model_type": model_name,
                "n_components": len(train.components),
                "training_time": training_time
            }

            # Add specific model parameters if available
            if hasattr(model, "get_params"):
                model_info["parameters"] = model.get_params()

            # Log to wandb
            wandb.log({
                f"statistical_{model_name}_trained": True,
                f"statistical_{model_name}_info": model_info
            })

            return {
                'predictions': combined_predictions,
                'models': trained_models,
                'model_info': model_info,
                'training_time': training_time
            }

        except Exception as e:
            error_msg = f"Error training {model_name}: {str(e)}"
            self.logger.error(error_msg)
            wandb.log({f"statistical_{model_name}_error": error_msg})
            raise