import pandas as pd
import time
import optuna
import wandb
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from darts.models import TransformerModel
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
    data = pd.read_hdf(f'../data/time_series/{time_interval}/{dataset}.h5', key=f'df_{dataset}')
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
        input_chunk_length = trial.suggest_int("input_chunk_length", 1, 36)
        d_model = trial.suggest_int("d_model", 16, 512)
        n_head = trial.suggest_int("n_head", 1, 8)
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 6)
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 6)
        dim_feedforward = trial.suggest_int("dim_feedforward", 128, 2048)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_size = trial.suggest_int("batch_size", 16, 128)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb.init(
            project="PMF_benchmark_Transformer",
            entity="yongboyu",
            config=config,
            group="Transformer-optimization"
        )

        # Define the model with suggested hyperparameters
        model = TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size=batch_size,
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
        "project": "PMF_benchmark_Transformer",
        "entity": "yongboyu",
        "group": 'Transformer-optimization'
    }
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="mae",
        wandb_kwargs=wandb_kwargs,
    )

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="Transformer-optimization-20241203",
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
    final_model = TransformerModel(
        input_chunk_length=best_params["input_chunk_length"],
        output_chunk_length=output_chunk_length,
        d_model=best_params["d_model"],
        nhead=best_params["n_head"],
        num_encoder_layers=best_params["num_encoder_layers"],
        num_decoder_layers=best_params["num_decoder_layers"],
        dim_feedforward=best_params["dim_feedforward"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"],
        n_epochs=100,
        optimizer_kwargs={"lr": best_params["learning_rate"]},
        random_state=42
    )
    final_model.fit(train_val_scaled)

    # Predict future values
    forecast_scaled = final_model.predict(len(test_scaled))
    forecast = transformer.inverse_transform(forecast_scaled)
    forecast.to_csv(f'../output/result/{dataset}/Transformer_predictions.csv')

    pdf_path = f'../output/result/{dataset}/Transformer_predictions.pdf'
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
    parser = argparse.ArgumentParser(description="Train Transformer (Demo).")
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