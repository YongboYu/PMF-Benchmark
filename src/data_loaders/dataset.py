import os
import re
import json
import argparse


import numpy as np
import pandas as pd
import pm4py

from datetime import datetime, timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer
from src.utils.utils import formalize_time_interval


def adjust_log_time_length(time_interval, log):

    timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
    start_time = min(timestamps)
    end_time = max(timestamps)
    total_time_span = end_time - start_time
    no_interval = total_time_span / time_interval
    print("Number of intervals:", no_interval)

    adj_start_time = start_time.replace(hour=0, minute=0, second=0)
    adj_end_time = end_time.replace(hour=0, minute=0, second=0) + timedelta(days=1)
    adj_total_time_span = adj_end_time - adj_start_time
    adj_no_interval = adj_total_time_span / time_interval

    if not adj_no_interval.is_integer():
        # Calculate the time length to remove from the beginning
        time_to_remove = adj_total_time_span % time_interval
        adj_start_time += time_to_remove

    # Recalculate the number of intervals
    trim_no_interval = int((adj_end_time - adj_start_time) / time_interval)
    print("Adjusted number of intervals:", trim_no_interval)

    return adj_start_time, adj_end_time, trim_no_interval


def gen_df_dict(log):
    # Dictionary to store Directly-Follows relations with timestamps
    df_dict = {}

    # Iterate over each trace (case) in the log
    for trace in log:
        # Iterate over pairs of consecutive events in the trace
        for i in range(len(trace) - 1):
            event_a = trace[i]
            event_b = trace[i + 1]

            # Extract activity names and timestamps
            activity_a = event_a['concept:name']
            activity_b = event_b['concept:name']
            start_time = event_a['time:timestamp']
            end_time = event_b['time:timestamp']

            # Calculate the duration between the events
            duration = end_time - start_time

            # Use the DF relation as the key and timestamp as the value
            df_key = f"{activity_a} -> {activity_b}"
            if df_key not in df_dict:
                df_dict[df_key] = []

            # Append timestamp of the first event in the DF relation
            df_dict[df_key].append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })

    print("DF dictionary is created.")
    return df_dict


def create_df_time_series(df_dict, start_time, end_time, time_interval):

    time_range = pd.date_range(start=start_time, end=end_time, freq=time_interval)

    # Initialize a DataFrame with the time range as the index and DF relations as columns
    df_name = list(df_dict.keys())
    df_time_series = pd.DataFrame(index=time_range, columns=df_name).fillna(0)

    # Iterate over the df_dict and populate the DataFrame
    for unique_df, occurrences in df_dict.items():
        for occurrence in occurrences:
            # Find the corresponding time interval for the start time
            interval_start = pd.Timestamp(occurrence['start_time']).floor(time_interval)
            if interval_start in df_time_series.index:
                df_time_series.at[interval_start, unique_df] += 1

    print("DF time series is created.")
    return df_time_series



def main(args):

    # Load the event log
    log_path = os.path.join(args.input_path, f'{args.dataset}.xes')
    log = xes_importer.apply(log_path)

    # Check the timestep
    time_interval, time_interval_str = formalize_time_interval(args.timestep)

    # Adjust the time length of the log
    adj_start_time, adj_end_time, trim_no_interval = adjust_log_time_length(time_interval, log)

    # Generate the DF dictionary
    df_dict = gen_df_dict(log)
    dict_save_path = os.path.join(args.output_dict_path, f"{args.dataset}.json")
    with open(dict_save_path, 'w') as f:
        json.dump(df_dict, f, default=str)

    # Create the DF time series
    df_time_series = create_df_time_series(df_dict, adj_start_time, adj_end_time, time_interval)
    time_series_path = os.path.join(args.output_time_series_path, time_interval_str)
    if not os.path.exists(time_series_path):
        os.makedirs(time_series_path)
    output_path = os.path.join(time_series_path, args.dataset)
    df_time_series.to_csv(f"{output_path}.csv")
    df_time_series.to_hdf(f"{output_path}.h5", key=f'df_{args.dataset}', mode='w')
    np.save(f"{output_path}.npy", df_time_series.to_numpy())

    print(f"Creating DF time series of {args.dataset} is done!")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a DFG sequence.")
    parser.add_argument(
        "--dataset", type=str,  required=True,
        help="Name of the event log."
    )
    parser.add_argument(
        "--timestep", type=str, default='1D',
        help="Time step with the time unit."
    )
    parser.add_argument(
        "--input_path", type=str, default='./data/processed',
        help="Input directory of the preprocessed event log."
    )
    parser.add_argument(
        "--output_dict_path", type=str, default='./data/df_dict',
        help="Output directory of DF dictionary."
    )
    parser.add_argument(
        "--output_time_series_path", type=str, default='./data/time_series',
        help="Output directory of DF time series."
    )


    args = parser.parse_args()
    main(args)