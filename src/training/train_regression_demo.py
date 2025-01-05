import os
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from sklearn.preprocessing import StandardScaler
from darts.metrics import mae, mse, mape, smape

from darts.models import RandomForest


from src.models.model import RegressionModels

models = RegressionModels

dataset = 'BPI2019_1'
time_interval = '1-day'

data = pd.read_hdf(f'./data/time_series/{time_interval}/{dataset}.h5', key=f'df_BPI2019_1')
series = TimeSeries.from_dataframe(data)
train, val_test = series.split_after(0.6)

val, test = val_test.split_after(0.5)

# standardize the data for linear regression


STUDY_NAME = "RF-optimization"

# %% Random Forest

# Define the objective function for Optuna
def objective(trial):

    start_time = time.time()

    # model_type = trial.suggest_categorical("model_type", ["random_forest", "lightgbm", "xgboost", "catboost"])
    # model_type = trial.suggest_categorical("random_forest")

    # Suggest hyperparameters
    lags = trial.suggest_int("lags", 1, 36)  # Number of lags
    n_estimators = trial.suggest_int("n_estimators", 50, 300)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 5, 20)  # Maximum depth of the tree
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)  # Min samples to split

    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(
        project="optuna",
        entity="yongboyu",  # NOTE: this entity depends on your wandb account.
        config=config,
        group=STUDY_NAME,
        reinit=True,
    )

    # Define the model with suggested hyperparameters
    model = RandomForest(
        lags=lags,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )

    # Train the model
    model.fit(train)

    # Predict on the validation set
    forecast = model.predict(len(val))

    # Evaluate using Mean Absolute Error (MAE)
    error = mae(val, forecast)

    # Log the metric to W&B
    # wandb.log({"mae": error})
    wandb.run.summary["final accuracy"] = error
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return error

# Set up W&B integration
wandb_kwargs = {"project": "darts-random-forest-optimization"}
wandb_callback = WeightsAndBiasesCallback(
    metric_name="mae",
    wandb_kwargs=wandb_kwargs,
)

# Create and run the Optuna study
study = optuna.create_study(
    direction="minimize",
    study_name="random-forest-optimization_20241127",
    storage="sqlite:///random_forest_optuna.db"
)
study.optimize(objective, n_trials=50, callbacks=[wandb_callback])

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)
print("Best MAE:", study.best_value)

# Train the final model with the best parameters
best_params = study.best_params
final_model = RandomForest(
    lags=best_params["lags"],
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    random_state=42,
)
final_model.fit(train)

# Predict future values
forecast = final_model.predict(len(val))

# Log final results to W&B
wandb.log({"final_mae": mae(val, forecast)})
