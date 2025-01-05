import os
import argparse
import pandas as pd

from darts import TimeSeries

from src.utils.utils import formalize_time_interval
# data = pd.read_hdf(f'./dataset/{dataset}.h5')




def create_seq2seq_dataset(series: TimeSeries, input_length: int, output_length: int) \
        -> Tuple[List[TimeSeries], List[TimeSeries]]:
    input_series_list = []
    output_series_list = []

    for i in range(len(series) - input_length - output_length + 1):
        input_series = series[i:i + input_length]
        output_series = series[i + input_length:i + input_length + output_length]

        input_series_list.append(input_series)
        output_series_list.append(output_series)

    return input_series_list, output_series_list



def main(args):
    time_interval, time_interval_str = formalize_time_interval(args.timestep)
    time_series_path = os.path.join(args.input_path, time_interval_str)
    data = pd.read_hdf(f'{time_series_path}/{args.dataset}.h5', key=f'df_{args.dataset}')
    series = TimeSeries.from_dataframe(data)

    train, val, test = series.split_after([0.6, 0.8])

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)


    test_input_seq, test_output_seq = create_seq2seq_dataset(test, input_length, output_length)
    test_scaled_input_seq, test_scaled_output_seq = create_seq2seq_dataset(test_scaled, input_length, output_length)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seq2seq IO time series data.")
    parser.add_argument(
        "--dataset", type=str,  required=True,
        help="Name of the event log."
    )
    parser.add_argument(
        "--timestep", type=str, default='1D',
        help="Time step with the time unit."
    )
    parser.add_argument(
        "--input_length", type=str, default='./data/processed',
        help="Input length (window size)."
    )
    parser.add_argument(
        "--output_length", type=str, default='./data/processed',
        help="Out length (horizon)."
    )
    parser.add_argument(
        "--input_path", type=str, default='./data/time_series',
        help="Input directory of the DF time series data."
    )
    parser.add_argument(
        "--output_path", type=str, default='./data/seq2seq_IO',
        help="Out directory of the seq2seq IO time series data."
    )


    args = parser.parse_args()
    main(args)