import pm4py
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Union, Set

dataset = 'BPI2017'
horizon = '7'
start_time = '2016-10-22 00:00:00'
model_group = 'deep_learning'
model_name = 'deepar'

log_path = 'data/interim/processed_logs/BPI2017.xes'
log = pm4py.read_xes(log_path)

# train
def extract_training_log(log: Union[pd.DataFrame, List],
                         case_id_col: str = 'case:concept:name',
                         activity_col: str = 'concept:name',
                         timestamp_col: str = 'time:timestamp',
                         training_ratio: float = 0.8) -> Tuple[Union[pd.DataFrame, List], int]:
    """
    Extract training log as the first X% of the event log based on time (days).
    Keeps all events within the time range, allowing partial traces.

    Parameters:
    - log: Original event log (DataFrame or pm4py log object)
    - case_id_col: Column name for case IDs
    - activity_col: Column name for activities
    - timestamp_col: Column name for timestamps
    - training_ratio: Proportion of time window to use for training (default 0.8)

    Returns:
    - Tuple of (training log in same format as input, time length in days)
    """
    # Convert pm4py log to DataFrame if needed
    if not isinstance(log, pd.DataFrame):
        print("Converting PM4Py log to DataFrame")
        df = pm4py.convert_to_dataframe(log)
    else:
        df = log.copy()

    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Get start and end date
    min_date = df[timestamp_col].min().date()
    max_date = df[timestamp_col].max().date()

    # Calculate total days and cutting date
    total_days = (max_date - min_date).days + 1  # +1 to include both start and end days
    days_for_training = int(total_days * training_ratio)
    cutoff_date = min_date + pd.Timedelta(days=days_for_training)

    print(f"Training period: {min_date} to {cutoff_date} ({days_for_training} days)")
    print(f"Total period: {min_date} to {max_date} ({total_days} days)")

    # Filter events up to the cutoff date (keeping partial traces)
    training_df = df[df[timestamp_col].dt.date < cutoff_date]

    # Replace special symbols with text representations
    training_df[activity_col] = training_df[activity_col].replace({'▶': 'Start', '■': 'End'})

    # Return in the same format as input
    if not isinstance(log, pd.DataFrame):
        # Convert back to PM4Py log
        parameters = {
            case_id_col: case_id_col,
            activity_col: activity_col,
            timestamp_col: timestamp_col
        }
        return pm4py.convert_to_event_log(training_df, parameters=parameters), days_for_training
    else:
        return training_df, days_for_training

training_log, training_days = extract_training_log(log, 'case:concept:name', 'concept:name', 'time:timestamp', 0.8)

activities = pm4py.get_event_attribute_values(
    training_log,
    'concept:name',
    case_id_key='case:concept:name'
)

full_training_dfg, sa, ea = pm4py.discover_dfg(training_log)

pm4py.view_dfg(full_training_dfg, sa, ea)

add_full_training_dfg = pm4py.insert_artificial_start_end(
    training_log,
    activity_key='concept:name',
    case_id_key='case:concept:name',
    timestamp_key='time:timestamp'
)

add_full_training_dfg, sa_add, ea_add = pm4py.discover_dfg(add_full_training_dfg)

pm4py.view_dfg(add_full_training_dfg, sa_add, ea_add)


def create_dfgs_from_rolling_training(seq_test_log: Dict[str, pd.DataFrame],
                                      raw_log: Union[pd.DataFrame, List],
                                      case_id_col: str = 'case:concept:name',
                                      activity_col: str = 'concept:name',
                                      timestamp_col: str = 'time:timestamp',
                                      time_length: int = None) -> Dict[str, Dict]:
    """
    Create training DFGs for each time window using 80% of the data based on time.
    Properly scales DFG frequencies to match the time horizon of each window.

    Parameters:
    - seq_test_log: Dictionary of test logs for each window
    - raw_log: Original complete event log
    - case_id_col: Column name for case IDs
    - activity_col: Column name for activities
    - timestamp_col: Column name for timestamps
    - time_length: Time length/horizon in days (if provided, uses this instead of calculating from window keys)

    Returns:
    - Dictionary of training DFGs for each window
    """
    # Create a dictionary to store all training DFGs
    training_dfgs = {}

    # Extract the training log (first 80% of the time) and get the training days
    print("Extracting training log (first 80% by time)")
    training_log, training_days = extract_training_log(
        raw_log, case_id_col, activity_col, timestamp_col, training_ratio=0.8
    )

    # Discover the DFG for the entire training log once
    # full_training_dfg = self.create_dfg_from_training(training_log)
    full_training_dfg, _, _ = pm4py.discover_dfg(training_log)

    # add artificial start/end events to the training log

    # Use provided time_length if available, otherwise calculate from window keys
    horizon_days = time_length
    if horizon_days is None and seq_test_log:
        # Extract time horizon from the first window key (format: YYYY-MM-DD_YYYY-MM-DD)
        first_window_key = list(seq_test_log.keys())[0]
        start_date_str, end_date_str = first_window_key.split('_')
        start_date = pd.to_datetime(start_date_str).date()
        end_date = pd.to_datetime(end_date_str).date()
        horizon_days = (end_date - start_date).days + 1  # +1 to include both start and end days

    print(f"Time horizon for each window: {horizon_days} days")

    # 1. Create horizon-scaled DFG once (scaled to match the time horizon)
    horizon_unit_training_dfg = {}
    for (source, target), freq in full_training_dfg.items():
        # Calculate daily frequency and then scale to horizon
        scaled_freq = int((freq / training_days) * horizon_days)
        if scaled_freq > 0:  # Only keep meaningful transitions
            horizon_unit_training_dfg[(source, target)] = scaled_freq

    # 2. Create JSON representation once
    dfg_json = create_dfg_from_truth(horizon_unit_training_dfg)

    # 3. Reuse the same scaled DFG and JSON for all windows
    for window_key in seq_test_log.keys():
        print(f"Creating training DFG for window: {window_key}")

        # Store in dictionary - reusing the same objects for all windows
        training_dfgs[window_key] = {
            'dfg': horizon_unit_training_dfg,
            'dfg_json': dfg_json,
            'training_log': training_log
        }

    return training_dfgs


rolling_training_dfgs = create_dfgs_from_rolling_training(
    seq_test_log,
    log,
    'case:concept:name',
    'concept:name',
    'time:timestamp',
    time_length=int(horizon)  # Convert horizon to int if necessary
)

# truth

def extract_time_period_sublog(df: pd.DataFrame,
                               case_id_col: str,
                               activity_col: str,
                               timestamp_col: str,
                               start_time: Union[pd.Timestamp, str],
                               end_time: Union[pd.Timestamp, str]) -> pd.DataFrame:
    """
    Extract a sublog containing only the directly-follows relations where the end activities
    fall within the specified time period.
    """
    # Sort the dataframe by case ID and timestamp
    df = df.sort_values(by=[case_id_col, timestamp_col])

    # Step 1: Identify cases that have at least one event within the time period
    cases_in_period = df[(df[timestamp_col] >= start_time) &
                         (df[timestamp_col] <= end_time)][case_id_col].unique()

    # Step 2: Filter to only those cases
    case_df = df[df[case_id_col].isin(cases_in_period)].copy()

    # Step 3: For each case, find all events that are needed to maintain directly-follows relations
    # where the end activity is within the time period
    result_events = []

    for case_id in cases_in_period:
        case_events = case_df[case_df[case_id_col] == case_id].copy()
        case_events['next_timestamp'] = case_events[timestamp_col].shift(-1)
        case_events['is_within_period'] = (case_events[timestamp_col] >= start_time) & (
                case_events[timestamp_col] <= end_time)
        case_events['next_within_period'] = (case_events['next_timestamp'] >= start_time) & (
                case_events['next_timestamp'] <= end_time)

        # Include events that are either within the time period or directly followed by one
        relevant_events = case_events[case_events['is_within_period'] | case_events['next_within_period']]
        result_events.append(relevant_events)

    if not result_events:
        return pd.DataFrame(columns=df.columns)

    # Combine all relevant events
    result_df = pd.concat(result_events)

    # replace symbols by texts
    result_df[activity_col] = result_df[activity_col].replace({'▶': 'Start', '■': 'End'})

    # Return the original columns only
    return result_df[df.columns].sort_values(by=[case_id_col, timestamp_col])


def extract_rolling_window_sublogs(df: pd.DataFrame,
                                   case_id_col: str,
                                   activity_col: str,
                                   timestamp_col: str,
                                   start_time: Union[pd.Timestamp, str],
                                   time_length_days: Union[int, str]) -> Dict[str, pd.DataFrame]:
    """
    Extract a series of sublogs using a rolling window of specified time length.
    """
    # Convert timestamps to datetime if they aren't already
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Convert time_length_days to int if it's a string
    time_length_days = int(time_length_days)

    # Convert start_time to datetime if it's not already
    if not isinstance(start_time, pd.Timestamp):
        start_time = pd.to_datetime(start_time)

    # Check if df timestamps have timezone and start_time doesn't
    sample_timestamp = df[timestamp_col].iloc[0]
    if hasattr(sample_timestamp, 'tz') and sample_timestamp.tz is not None and start_time.tz is None:
        # Localize start_time to match the timestamps in the dataframe
        start_time = start_time.tz_localize(sample_timestamp.tz)

    # Get the overall end time from the data
    data_end_time = df[timestamp_col].max()

    # Calculate the number of rolling windows
    time_delta = data_end_time - start_time
    num_windows = max(1, time_delta.days - time_length_days + 2)  # +2 to include the first and last window

    # Create a dictionary to store all sublogs
    sublogs = {}

    # Create rolling windows and extract sublogs
    for i in range(num_windows):
        window_start = start_time + pd.Timedelta(days=i)
        window_end = window_start + pd.Timedelta(days=time_length_days - 1)

        # Format dates for the key
        window_key = f"{window_start.strftime('%Y-%m-%d')}_{window_end.strftime('%Y-%m-%d')}"

        # Extract sublog for this window
        sublog = extract_time_period_sublog(
            df, case_id_col, activity_col, timestamp_col, window_start, window_end
        )

        # Only add the sublog if it contains events
        if len(sublog) > 0:
            # Ensure the timestamp column is recognized as datetime
            sublog[timestamp_col] = pd.to_datetime(sublog[timestamp_col])

            # Add artificial start/end events
            try:
                sublog_with_artificial = pm4py.insert_artificial_start_end(sublog)
                sublogs[window_key] = sublog_with_artificial
            except Exception as e:
                print(f"Warning: Could not add artificial start/end events for window {window_key}: {str(e)}")
                # Fall back to using the original sublog
                sublogs[window_key] = sublog

    return sublogs

seq_test_log = extract_rolling_window_sublogs(
    log, 'case:concept:name', 'concept:name', 'time:timestamp',
    start_time, horizon
)

sub_truth_log = seq_test_log['2016-10-22_2016-10-28']
sub_log_dfg, sa_sub, ea_sub = pm4py.discover_dfg(sub_truth_log)
pm4py.view_dfg(sub_log_dfg, sa_sub, ea_sub)

def create_dfg_from_truth(dfg_truth: Dict[Tuple[str, str], int]) -> Dict[str, List]:
    """
    Create DFG structure from PM4Py's DFG output
    """
    # Initialize node maps
    reverse_map = {}
    reverse_map['▶'] = 0
    reverse_map['■'] = 1

    # Map all activities to IDs
    for (source, target), freq in dfg_truth.items():
        if source not in reverse_map:
            reverse_map[source] = len(reverse_map)
        if target not in reverse_map:
            reverse_map[target] = len(reverse_map)

    # Create arcs
    arcs = []
    node_freq = {node: 0 for node in reverse_map.keys()}

    # Add directly-follows relations
    for (source, target), freq in dfg_truth.items():
        arcs.append({
            'from': reverse_map[source],
            'to': reverse_map[target],
            'freq': freq
        })
        if source == '▶':
            node_freq[source] += freq
        else:
            node_freq[target] += freq

    # Create nodes
    nodes = []
    for node, freq in node_freq.items():
        nodes.append({
            'label': node,
            'id': reverse_map[node],
            'freq': int(freq)
        })

    return {'nodes': nodes, 'arcs': arcs}


def create_dfgs_from_rolling_window(seq_test_log: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Create a series of DFGs from the rolling window sublogs.
    """
    # Create a dictionary to store all DFGs
    dfgs_dict = {}

    # For each sublog in the sequence, discover a DFG
    for window_key, sublog in seq_test_log.items():
        # Discover directly-follows graph
        dfg, _, _ = pm4py.discover_dfg(sublog)

        # Create JSON representation of the DFG
        dfg_json = create_dfg_from_truth(dfg)

        # Store in dictionary
        dfgs_dict[window_key] = {
            'dfg': dfg,
            'dfg_json': dfg_json,
            'sublog': sublog
        }

    return dfgs_dict


rolling_truth_dfgs = create_dfgs_from_rolling_window(seq_test_log)