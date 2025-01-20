import pandas as pd
import time
import optuna
import wandb
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from darts.models import RandomForest, LinearRegressionModel
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

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(
            project="PMF_benchmark_LinearRegression-interpolate-1d",
            entity="yongboyu",
            config=config,
            group="linear-regression-optimization"
        )

        # Define the model with suggested hyperparameters
        model = LinearRegressionModel(
            lags=lags,
            random_state=42
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
        "project": "PMF_benchmark_LinearRegression-interpolate-1d",
        "entity": "yongboyu",
        "group": 'linear-regression-optimization'
    }
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="mae",
        wandb_kwargs=wandb_kwargs,
    )

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="linear-regression-optimization-interpolate-1d",
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
    final_model = LinearRegressionModel(
        lags=best_params["lags"],
        random_state=42
    )
    final_model.fit(train_val)

    # Predict future values
    forecast = final_model.predict(len(test))

    # save the predictions and metrics
    # Save predictions to CSV and h5
    forecast.to_csv(f'../output/result_interpolate_1d/{dataset}/Linear_Regression_predictions.csv')
    # forecast.to_hdf(f'result/{dataset}_predictions.h5', key='all_predictions', mode='a')

    pdf_path = f'../output/result_interpolate_1d/{dataset}/Linear_Regression_predictions.pdf'
    with PdfPages(pdf_path) as pdf:
        for i, (forecast_series, test_series) in enumerate(zip(forecast, test)):
            plt.figure(figsize=(12, 6))
            test_series.plot(label='Actual')
            forecast_series.plot(label='Forecast')
            plt.title(f'Time Series {i + 1}: {test.columns[i]}')
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
    parser = argparse.ArgumentParser(description="Train Linear Regression (Demo).")
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