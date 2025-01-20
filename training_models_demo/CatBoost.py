import pandas as pd
import time
import optuna
import wandb
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from darts.models import CatBoostModel
from darts.metrics import mae, rmse, smape


# Load data
# dataset = 'BPI2019_1'
# time_interval = '1-day'








def main(args):
    dataset = args.dataset
    time_interval = '1-day'
    output_length = args.horizon_length
    data = pd.read_hdf(f'../data/time_series/{time_interval}/{dataset}.h5', key=f'df_{dataset}')
    series = TimeSeries.from_dataframe(data)
    train_val, test = series.split_after(0.8)
    train, val = train_val.split_after(0.75)


    # Define the objective function for Optuna
    def objective(trial):
        start_time = time.time()

        # Suggest hyperparameters
        lags = trial.suggest_int("lags", 1, 36)
        depth = trial.suggest_int("depth", 4, 10)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 1e1, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(
            project="PMF_benchmark_CatBoost-2",
            entity="yongboyu",
            config=config,
            group="CatBoost-optimization"
        )

        # Define the model with suggested hyperparameters
        model = CatBoostModel(
            lags=lags,
            output_chunk_length=output_length,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            learning_rate=learning_rate,
            random_state=42,
        )

        # Train the model
        model.fit(train)

        # Predict on the validation set
        forecast = model.predict(len(val))

        # Evaluate using Mean Absolute Error (MAE)
        error = mae(val, forecast)

        # Log the metric to W&B
        wandb.log({"mae": error})
        wandb.run.summary["final mae"] = error
        wandb.finish(quiet=True)

        return error

    # Set up W&B integration
    wandb_kwargs = {
        "project": "PMF_benchmark_CatBoost-2",
        "entity": "yongboyu",
        "group": 'CatBoost-optimization'
    }
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="mae",
        wandb_kwargs=wandb_kwargs,
    )

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="CatBoost-20241202",
        storage="sqlite:///PMF_benchmark_demo.db"
    )
    study.optimize(objective, n_trials=100, callbacks=[wandb_callback])

    # Log Optuna figures to W&B
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    wandb.log({"optimization_history": fig1})
    wandb.log({"param_importances": fig2})

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best MAE:", study.best_value)

    wandb.log({"best_hyperparameters": study.best_params})
    wandb.log({"best_mae": study.best_value})

    # Train the final model with the best parameters
    best_params = study.best_params
    final_model = CatBoostModel(
        lags=best_params["lags"],
        output_chunk_length=output_length,
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        learning_rate=best_params["learning_rate"],
        random_state=42,
    )
    final_model.fit(train_val)

    # Predict future values
    forecast = final_model.predict(len(test))

    forecast.to_csv(f'../output/result/{dataset}/CatBoost_predictions.csv')

    shift_test = test + 1e-6
    shift_forecast = forecast + 1e-6

    # Log final results to W&B
    wandb.log({
        "final_mae": mae(test, forecast),
        "final_rmse": rmse(test, forecast),
        "final_smape": smape(shift_test, shift_forecast)
    })



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatBoost (Demo).")
    parser.add_argument(
        "--dataset", type=str,  required=True,
        help="Name of the event log."
    )
    parser.add_argument(
        "--horizon_length", type=int, default='1',
        help="Out length (horizon)."
    )


    args = parser.parse_args()
    main(args)