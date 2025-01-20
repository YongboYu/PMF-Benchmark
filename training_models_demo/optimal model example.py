import pandas as pd
import time
import optuna
import wandb
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration.wandb import WeightsAndBiasesCallback

from darts import TimeSeries
from darts.models import RandomForest
from darts.metrics import mae, rmse, smape

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
# dataset = 'BPI2019_1'
# time_interval = '1-day'









dataset = 'BPI2019_1'
time_interval = '1-day'
output_length = 1
data = pd.read_hdf(f'../data/time_series_impute/{time_interval}/{dataset}.h5', key=f'df_{dataset}')
series = TimeSeries.from_dataframe(data)
train_val, test = series.split_after(0.8)
train, val = train_val.split_after(0.75)



final_model = RandomForest(
    lags=35,
    n_estimators=152,
    max_depth=9,
    min_samples_split=2,
    random_state=42,
)
final_model.fit(train_val)

# Predict future values
forecast = final_model.predict(len(test))

# save the predictions and metrics
# Save predictions to CSV and h5
forecast.to_csv(f'../output/result_interpolate_1d/{dataset}/RandomForest_predictions_EXAMPLE.csv')
# forecast.to_hdf(f'result/{dataset}_predictions.h5', key='all_predictions', mode='a')

pdf_path = f'../output/result_interpolate_1d/{dataset}/RandomForest_predictions_EXAMPLE.pdf'
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
mae(test, forecast)
rmse(test, forecast)
smape(shift_test, shift_forecast)





# BEST N-HiTS model

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

dataset = "BPI2019_1"
time_interval = '1-day'
output_chunk_length = 1
data = pd.read_hdf(f'./data/time_series_impute/{time_interval}/{dataset}.h5', key=f'df_{dataset}')
series = TimeSeries.from_dataframe(data)
train_val, test = series.split_after(0.8)
train, val = train_val.split_after(0.75)

transformer = Scaler()
train_scaled = transformer.fit_transform(train)
train_val_scaled = transformer.transform(train_val)
val_scaled = transformer.transform(val)
test_scaled = transformer.transform(test)

final_model = NHiTSModel(
    input_chunk_length=7,
    output_chunk_length=output_chunk_length,
    num_stacks=1,
    num_blocks=2,
    num_layers=3,
    layer_widths=485,
    dropout=0.06763,
    n_epochs=100,
    optimizer_kwargs={"lr": 0.00296},
    random_state=42
)

final_model.fit(train_val_scaled)

# Predict future values
forecast_scaled = final_model.predict(len(test_scaled))
forecast = transformer.inverse_transform(forecast_scaled)

forecast.to_csv(f'./output/result_interpolate_1d/{dataset}/NHiTS_predictions.csv')

pdf_path = f'./output/result_interpolate_1d/{dataset}/NHiTS_predictions.pdf'
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
mae(test, forecast)
rmse(test, forecast)
smape(shift_test, shift_forecast)
