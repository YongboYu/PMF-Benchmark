import pandas as pd
import time
import optuna
import wandb
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from darts.models import NHiTSModel
from darts.metrics import mae, rmse, smape
from darts.dataprocessing.transformers import Scaler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
# dataset = 'BPI2019_1'
# time_interval = '1-day'








def main(args):
    dataset = args.dataset
    time_interval = '1-day'
    output_chunk_length = args.horizon_length
    data = pd.read_hdf(f'../data/time_series_impute/{time_interval}/{dataset}.h5', key=f'df_{dataset}')
    series = TimeSeries.from_dataframe(data)
    train_val, test = series.split_after(0.8)
    train, val = train_val.split_after(0.75)

    transformer = Scaler()
    train_scaled = transformer.fit_transform(train)
    train_val_scaled = transformer.transform(train_val)
    val_scaled = transformer.transform(val)
    test_scaled = transformer.transform(test)

    # Define the objective function for Optuna
    def objective(trial):
        start_time = time.time()

        # Suggest hyperparameters
        input_chunk_length = trial.suggest_int("input_chunk_length", output_chunk_length, 36)
        # generic_architecture = trial.suggest_categorical("generic_architecture", [True, False])
        num_stacks = trial.suggest_int("num_stacks", 1, 30)
        num_blocks = trial.suggest_int("num_blocks", 1, 5)
        num_layers = trial.suggest_int("num_layers", 1, 5)
        layer_widths = trial.suggest_int("layer_widths", 32, 512)
        # expansion_coefficient_dim = trial.suggest_int("expansion_coefficient_dim", 1, 10)
        # trend_polynomial_degree = trial.suggest_int("trend_polynomial_degree", 1, 5)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        # batch_size = trial.suggest_int("batch_size", 16, 128)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)


        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(
            project="PMF_benchmark_NHiTS-interpolate-7d",
            entity="yongboyu",
            config=config,
            group="NHiTS-optimization"
        )

        # Define the model with suggested hyperparameters
        model = NHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            dropout=dropout,
            n_epochs=100,
            optimizer_kwargs={"lr": learning_rate},
            random_state=42
        )

        # Train the model
        model.fit(train_scaled)

        # Predict on the validation set
        forecast = model.predict(len(val_scaled))

        # Evaluate using Mean Absolute Error (MAE)
        error = mae(val_scaled, forecast)

        # Log the metric to W&B
        wandb.log({"mae": error})
        wandb.run.summary["final mae"] = error
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

        return error

    # Set up W&B integration
    wandb_kwargs = {
        "project": "PMF_benchmark_NHiTS-interpolate-7d",
        "entity": "yongboyu",
        "group": 'NHiTS-optimization'
    }
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="mae",
        wandb_kwargs=wandb_kwargs,
    )

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="NHITS-optimization-interpolate-7d",
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
    final_model = NHiTSModel(
        input_chunk_length=best_params["input_chunk_length"],
        output_chunk_length=output_chunk_length,
        num_stacks=best_params["num_stacks"],
        num_blocks=best_params["num_blocks"],
        num_layers=best_params["num_layers"],
        layer_widths=best_params["layer_widths"],
        dropout=best_params["dropout"],
        n_epochs=100,
        optimizer_kwargs={"lr": best_params["learning_rate"]},
        random_state=42
    )
    final_model.fit(train_val_scaled)

    # Predict future values
    forecast_scaled = final_model.predict(len(test_scaled))
    forecast = transformer.inverse_transform(forecast_scaled)

    forecast.to_csv(f'../output/result_interpolate_7d/{dataset}/NHiTS_predictions.csv')

    pdf_path = f'../output/result_interpolate_7d/{dataset}/NHiTS_predictions.pdf'
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
    parser = argparse.ArgumentParser(description="Train NHiTS (Demo).")
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