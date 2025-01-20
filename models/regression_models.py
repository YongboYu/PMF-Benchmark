from darts.models import (
    LinearRegressionModel,
    RandomForest,
    XGBModel,
    LightGBMModel
)
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import optuna
import wandb
from typing import Dict, Any, Tuple


class RegressionModels:
    def __init__(self):
        """Initialize regression models for both univariate and multivariate forecasting"""
        self.models = {
            "linear": {
                "model": LinearRegressionModel,
                "params": {}
            },
            "random_forest": {
                "model": RandomForest,
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "min_samples_split": (2, 10)
                }
            },
            "xgboost": {
                "model": XGBModel,
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.1),
                    "subsample": (0.6, 1.0)
                }
            },
            "lightgbm": {
                "model": LightGBMModel,
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.1),
                    "num_leaves": (20, 100)
                }
            }
        }

    def objective(self, trial: optuna.Trial, model_name: str, model_class,
                  train, val, horizon: int) -> float:
        """Optuna objective function for hyperparameter optimization"""
        # Get parameter ranges for the specific model
        param_ranges = self.models[model_name]["params"]
        params = {}

        # Define parameters based on model type
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *param_ranges["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *param_ranges["max_depth"]),
                "min_samples_split": trial.suggest_int("min_samples_split",
                                                       *param_ranges["min_samples_split"])
            }
        elif model_name in ["xgboost", "lightgbm"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *param_ranges["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *param_ranges["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate",
                                                     *param_ranges["learning_rate"],
                                                     log=True)
            }
            if model_name == "xgboost":
                params["subsample"] = trial.suggest_float("subsample",
                                                          *param_ranges["subsample"])
            else:  # lightgbm
                params["num_leaves"] = trial.suggest_int("num_leaves",
                                                         *param_ranges["num_leaves"])

        # Create and train model
        regressor = model_class(**params)
        model = RegressionModel(
            model=regressor,
            lags=12  # You might want to make this configurable
        )

        try:
            model.fit(train, val_series=val)
            val_pred = model.predict(n=horizon)
            val_score = model.evaluate(val, val_pred)
            return val_score
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return float('inf')

    def train_and_predict(self, train, val, test, horizon: int,
                          multivariate: bool = True) -> Tuple[Dict, Dict]:
        """
        Train models and generate predictions
        Args:
            multivariate: If True, train on multivariate data; if False, train separate models for each component
        """
        predictions = {}
        best_params = {}

        for name, model_info in self.models.items():
            wandb.log({f"regression_training_model": name})

            if name == "linear":
                # Linear regression doesn't need optimization
                regressor = model_info["model"]()
                model = RegressionModel(model=regressor, lags=12)

                if multivariate:
                    model.fit(train, val_series=val)
                    pred = model.predict(horizon)
                    predictions[name] = pred
                else:
                    # Train separate models for each component
                    component_preds = []
                    for component in train.components:
                        model.fit(train[component], val_series=val[component] if val is not None else None)
                        pred = model.predict(horizon)
                        component_preds.append(pred)
                    predictions[name] = component_preds

                best_params[name] = model.model.get_params()

            else:
                # Optimize hyperparameters
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: self.objective(trial, name, model_info["model"],
                                                 train, val, horizon),
                    n_trials=50
                )

                # Train final model with best parameters
                best_params[name] = study.best_params
                regressor = model_info["model"](**study.best_params)
                model = RegressionModel(model=regressor, lags=12)

                if multivariate:
                    model.fit(train, val_series=val)
                    pred = model.predict(horizon)
                    predictions[name] = pred
                else:
                    # Train separate models for each component
                    component_preds = []
                    for component in train.components:
                        model.fit(train[component], val_series=val[component] if val is not None else None)
                        pred = model.predict(horizon)
                        component_preds.append(pred)
                    predictions[name] = component_preds

            # Log to wandb
            wandb.log({
                f"regression_{name}_best_params": best_params[name],
                f"regression_{name}_trained": True
            })

        return predictions, best_params


# models/regression_models.py
from darts.models import RegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from utils.lag_utils import get_lags_for_horizon


class RegressionModels:
    def __init__(self, base_config_path: str = "config/base_config.yaml",
                 model_config_path: str = "config/model_configs/regression_models.yaml"):
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

        # Initialize model classes
        self.models = {
            "linear": {
                "model": LinearRegression,
                "params": self.model_config['models']['linear']['params']
            },
            "random_forest": {
                "model": RandomForestRegressor,
                "params": self.model_config['models']['random_forest']
            },
            "xgboost": {
                "model": XGBRegressor,
                "params": self.model_config['models']['xgboost']
            },
            "lightgbm": {
                "model": LGBMRegressor,
                "params": self.model_config['models']['lightgbm']
            }
        }

    def _objective(self, trial: optuna.Trial, model_name: str, model_class,
                   train, val, horizon: int, transformer) -> float:
        """Optuna objective function for hyperparameter optimization"""
        # Get parameter ranges from config
        param_ranges = self.models[model_name]['params']['hyperparameter_ranges']
        params = {}

        # Define parameters based on model type
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", param_ranges["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", param_ranges["max_depth"]),
                "min_samples_split": trial.suggest_categorical("min_samples_split",
                                                               param_ranges["min_samples_split"])
            }
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", param_ranges["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", param_ranges["max_depth"]),
                "learning_rate": trial.suggest_categorical("learning_rate", param_ranges["learning_rate"]),
                "subsample": trial.suggest_categorical("subsample", param_ranges["subsample"])
            }
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", param_ranges["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", param_ranges["max_depth"]),
                "learning_rate": trial.suggest_categorical("learning_rate", param_ranges["learning_rate"]),
                "num_leaves": trial.suggest_categorical("num_leaves", param_ranges["num_leaves"])
            }

        try:
            # Get lags based on horizon
            lags = get_lags_for_horizon(horizon, self.model_config['common'])

            # Create and train model
            regressor = model_class(**params)
            model = RegressionModel(
                model=regressor,
                lags=lags
            )

            model.fit(train, val_series=val)
            val_pred = model.predict(len(val))
            val_score = model.evaluate(val, val_pred)
            return val_score

        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            return float('inf')

    def train_and_predict(self, train, val, test, horizon: int, transformer) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train models and generate predictions"""
        predictions = {}
        best_params = {}

        for name, model_info in self.models.items():
            self.logger.info(f"Training {name} model...")
            wandb.log({f"regression_training_model": name})

            try:
                # Get lags based on horizon
                lags = get_lags_for_horizon(horizon, self.model_config['common'])

                if name == "linear":
                    # Linear regression doesn't need optimization
                    regressor = model_info["model"]()
                    model = RegressionModel(
                        model=regressor,
                        lags=lags
                    )

                    model.fit(train, val_series=val)
                    pred = model.predict(len(test))
                    predictions[name] = pred
                    best_params[name] = model.model.get_params()

                else:
                    # Optimize hyperparameters
                    study = optuna.create_study(direction="minimize")
                    study.optimize(
                        lambda trial: self._objective(trial, name, model_info["model"],
                                                      train, val, horizon),
                        n_trials=self.model_config['common']['n_trials']
                    )

                    # Train final model with best parameters
                    best_params[name] = study.best_params
                    regressor = model_info["model"](**study.best_params)
                    model = RegressionModel(
                        model=regressor,
                        lags=lags
                    )

                    model.fit(train, val_series=val)
                    pred = model.predict(len(test))
                    predictions[name] = pred

                # Save model if configured
                if self.base_config['evaluation'].get('save_models', False):
                    model_path = Path(self.model_config['paths']['model_save_dir']) / f"{name}_h{horizon}.pkl"
                    model.save(model_path)

                # Log to wandb
                wandb.log({
                    f"regression_{name}_best_params": best_params[name],
                    f"regression_{name}_trained": True
                })

            except Exception as e:
                error_msg = f"Error training {name}: {str(e)}"
                self.logger.error(error_msg)
                wandb.log({f"regression_{name}_error": error_msg})

        return predictions, best_params

    def load_model(self, model_name: str, horizon: int) -> Optional[RegressionModel]:
        """Load a saved model if it exists"""
        model_path = Path(self.model_config['paths']['model_save_dir']) / f"{model_name}_h{horizon}.pkl"
        if model_path.exists():
            return RegressionModel.load(model_path)
        return None