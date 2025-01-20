import pandas as pd
import time
import optuna
import wandb
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.metrics import mae, rmse, smape

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
# dataset = 'BPI2019_1'
# time_interval = '1-day'








def main(args):
    dataset = args.dataset
    time_interval = '1-day'
    output_length = args.horizon_length
    data = pd.read_hdf(f'../data/time_series_impute/{time_interval}/{dataset}.h5', key=f'df_{dataset}')
    series = TimeSeries.from_dataframe(data)
    train_val, test = series.split_after(0.8)
    train, val = train_val.split_after(0.75)

    # Define the objective function for Optuna
    def objective(trial):
        start_time = time.time()

        # Suggest hyperparameters
        lags = trial.suggest_int("lags", output_length, 36)
        num_leaves = trial.suggest_int("num_leaves", 20, 150)
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(
            project="PMF_benchmark_LightGBM-interpolate-7d",
            entity="yongboyu",
            config=config,
            group="LightGBM-optimization"
        )

        # Define the model with suggested hyperparameters
        model = LightGBMModel(
            lags=lags,
            output_chunk_length=output_length,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
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
        "project": "PMF_benchmark_LightGBM-interpolate-7d",
        "entity": "yongboyu",
        "group": 'LightGBM-optimization'
    }
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="mae",
        wandb_kwargs=wandb_kwargs,
    )

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="LightGBM-interpolate-7d",
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
    final_model = LightGBMModel(
        lags=best_params["lags"],
        output_chunk_length=output_length,
        num_leaves=best_params["num_leaves"],
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_child_samples=best_params["min_child_samples"],
        learning_rate=best_params["learning_rate"],
    )
    final_model.fit(train_val)

    # Predict future values
    forecast = final_model.predict(len(test))

    forecast.to_csv(f'../output/result_interpolate_7d/{dataset}/LightGBM_predictions.csv')

    pdf_path = f'../output/result_interpolate_7d/{dataset}/LightGBM_predictions.pdf'
    with PdfPages(pdf_path) as pdf:
        for i in range(test.width):
            plt.figure(figsize=(12, 6))
            time_series_name = test.columns[i]
            test.univariate_component(time_series_name).plot(label='Actual')
            forecast.univariate_component(time_series_name).plot(label='Forecast')
            plt.title(f'Time Series {i + 1}: {time_series_name}')
            plt.legend()
            pdf.savefig()  # Save the current figure into the PDF
            plt.close()  # Close the figure to free memory

    shift_test = test + 1e-6
    shift_forecast = forecast + 1e-6

    # Log final results to W&B
    wandb.log({
        "final_mae": mae(test, forecast),
        "final_rmse": rmse(test, forecast),
        "final_smape": smape(shift_test, shift_forecast)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM (Demo).")
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