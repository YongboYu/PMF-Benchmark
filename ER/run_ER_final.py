import pm4py
import pandas as pd
import numpy as np
import os
import json
import math
import networkx as nx
from tqdm import tqdm

def extract_time_period_sublog(df, case_id_col, activity_col, timestamp_col, start_time, end_time):
    """
    Extract a sublog containing only the directly-follows relations where the end activities
    fall within the specified time period.

    Parameters:
    - df: The event log dataframe
    - case_id_col: Column name for case IDs
    - activity_col: Column name for activities
    - timestamp_col: Column name for timestamps
    - start_time: Start of the time period
    - end_time: End of the time period

    Returns:
    - A dataframe containing the relevant events
    """
    # # Convert timestamps to datetime if they aren't already
    # df = df.copy()
    # if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
    #     df[timestamp_col] = pd.to_datetime(df[timestamp_col])


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

        # Include events that are either:
        # 1. Within the time period themselves
        # 2. Directly followed by an event within the time period
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


def extract_rolling_window_sublogs(df, case_id_col, activity_col, timestamp_col, start_time, time_length_days):
    """
    Extract a series of sublogs using a rolling window of specified time length.

    Parameters:
    - df: The event log dataframe
    - case_id_col: Column name for case IDs
    - activity_col: Column name for activities
    - timestamp_col: Column name for timestamps
    - start_time: Start of the time period (string or datetime)
    - time_length_days: Length of each time window in days (integer or string)

    Returns:
    - A dictionary of dataframes, each containing a sublog for a specific time window
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
            df,
            case_id_col,
            activity_col,
            timestamp_col,
            window_start,
            window_end
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


def create_dfg_from_truth(dfg_truth):
    """
    Create DFG structure from PM4Py's DFG output

    Args:
        dfg_truth: Dictionary containing the DFG from PM4Py (format: {(source, target): frequency})
        sa: Start activities dictionary {activity: frequency}
        ea: End activities dictionary {activity: frequency}

    Returns:
        DFG in JSON format
    """
    # Initialize node maps
    reverse_map = {}
    reverse_map['▶'] = 0
    reverse_map['■'] = 1

    # Map all activities to IDs
    for (source, target), freq in dfg_truth.items():
        # # Handle special symbols for start/end
        # source = source.replace('▶', 'Start').replace(, 'End') if isinstance(source, str) else source
        # target = target.replace('▶', 'Start').replace('■', 'End') if isinstance(target, str) else target

        if source not in reverse_map:
            reverse_map[source] = len(reverse_map)
        if target not in reverse_map:
            reverse_map[target] = len(reverse_map)

    # Create arcs
    arcs = []
    node_freq = {node: 0 for node in reverse_map.keys()}

    # # Add start activities if provided
    # if sa:
    #     for act, freq in sa.items():
    #         # Handle special symbols for act
    #         act = act.replace('▶', 'Start').replace('■', 'End') if isinstance(act, str) else act
    #
    #         if freq > 0:
    #             arcs.append({
    #                 'from': reverse_map['Start'],
    #                 'to': reverse_map[act],
    #                 'freq': freq
    #             })
    #             node_freq['Start'] += freq
    #
    # # Add end activities if provided
    # if ea:
    #     for act, freq in ea.items():
    #         # Handle special symbols for act
    #         act = act.replace('▶', 'Start').replace('■', 'End') if isinstance(act, str) else act
    #
    #         if freq > 0:
    #             arcs.append({
    #                 'from': reverse_map[act],
    #                 'to': reverse_map['End'],
    #                 'freq': freq
    #             })
    #             node_freq['End'] += freq

    # Add directly-follows relations
    for (source, target), freq in dfg_truth.items():
        # # Handle special symbols again for consistency
        # source = source.replace('▶', 'Start').replace('■', 'End') if isinstance(source, str) else source
        # target = target.replace('▶', 'Start').replace('■', 'End') if isinstance(target, str) else target

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


def create_dfgs_from_rolling_window(seq_test_log):
    """
    Create a series of DFGs from the rolling window sublogs.

    Parameters:
    - seq_test_log: Dictionary of sublogs from extract_rolling_window_sublogs
    - parameters: Optional parameters for the DFG discovery

    Returns:
    - Dictionary with time window keys and corresponding DFG objects
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


def create_dfg_from_predictions_new(predictions_df):
    """
    Create DFG structure from predictions dataframe with special start/end connections

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        DFG in JSON format
    """
    # Extract directly-follows relations
    reverse_map = {}
    reverse_map['▶'] = 0  # Start symbol
    reverse_map['■'] = 1  # End symbol
    reverse_map['Start'] = 2  # Original Start (replacement for '▶' in input)
    reverse_map['End'] = 3  # Original End (replacement for '■' in input)

    # Track original activities (excluding start/end symbols)
    original_activities = set()

    # Process each column to extract activities
    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]

            # Replace symbols in source and target
            source = source.replace('▶', 'Start').replace('■', 'End')
            target = target.replace('▶', 'Start').replace('■', 'End')

            original_activities.add(source)
            original_activities.add(target)


            # Register activities
            if source not in reverse_map:
                reverse_map[source] = len(reverse_map)

            if target not in reverse_map:
                reverse_map[target] = len(reverse_map)

    # Create arcs with original frequencies between activities
    arcs = []
    node_freq = {node: 0 for node in reverse_map.keys()}

    # Process regular activity relationships
    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]
            clean_source = source.replace('▶', 'Start').replace('■', 'End')
            clean_target = target.replace('▶', 'Start').replace('■', 'End')

            # # Skip relations involving Start/End for now
            # if clean_source in ['Start', 'End'] or clean_target in ['Start', 'End']:
            #     continue

            # Map source and target IDs
            source_id = reverse_map.get(clean_source)
            target_id = reverse_map.get(clean_target)

            if source_id is not None and target_id is not None:
                # Sum frequency across all time steps
                total_freq = 0
                for _, row in predictions_df.iterrows():
                    freq = round(float(row[df_relation]))
                    if freq > 0:
                        total_freq += freq

                if total_freq > 0:
                    arcs.append({
                        'from': source_id,
                        'to': target_id,
                        'freq': total_freq
                    })
                    node_freq[clean_target] += total_freq

    # Add connections from '▶' to each activity with frequency 1
    for activity in original_activities:
        arcs.append({
            'from': reverse_map['▶'],
            'to': reverse_map[activity],
            'freq': 1
        })
        node_freq['▶'] += 1
        # node_freq[activity] += 1

    # Add connections from each activity to '■' with frequency 1
    for activity in original_activities:
        arcs.append({
            'from': reverse_map[activity],
            'to': reverse_map['■'],
            'freq': 1
        })
        node_freq['■'] += 1

    # Update frequency for '▶' node
    # node_freq['▶'] = len(original_activities)

    # Create nodes list
    nodes = []
    for node, node_id in reverse_map.items():
        nodes.append({
            'label': node,
            'id': node_id,
            'freq': round(node_freq.get(node, 0))
        })

    return {'nodes': nodes, 'arcs': arcs}


def create_dfgs_from_rolling_predictions(seq_test_log, agg_pred_round):
    """
    Create a series of DFGs from rolling window predictions.

    Parameters:
    - seq_test_log: Dictionary of sublogs from extract_rolling_window_sublogs (for window keys)
    - agg_pred_round: DataFrame with aggregated predictions

    Returns:
    - Dictionary with time window keys and corresponding predicted DFG objects
    """
    # Create a dictionary to store all DFGs
    dfgs_dict = {}

    # For each window in the sequence, create a DFG from predictions
    for window_key in seq_test_log.keys():
        # Get timestamp from window_key (format is 'YYYY-MM-DD_YYYY-MM-DD')
        start_date = window_key.split('_')[0]

        # Find corresponding predictions for this window
        window_pred = agg_pred_round.loc[agg_pred_round.index == start_date]

        if not window_pred.empty:
            # Create JSON representation of the DFG from predictions
            dfg_json = create_dfg_from_predictions_new(window_pred)

            # Store in dictionary
            dfgs_dict[window_key] = {
                'dfg_json': dfg_json,
                'window_pred': window_pred
            }

    return dfgs_dict


def reformat_rolling_dfgs(rolling_truth_dfgs, rolling_pred_dfgs):
    """
    Reformat rolling DFGs to have both truth and prediction data for each time window.

    Parameters:
    - rolling_truth_dfgs: Dictionary with time window keys and corresponding ground truth DFGs
    - rolling_pred_dfgs: Dictionary with time window keys and corresponding predicted DFGs

    Returns:
    - Dictionary where each key is a time window and each value has 'truth' and 'pred' components
    """
    combined_dfgs = {}

    # Find all time windows that exist in either dictionary
    all_windows = set(rolling_truth_dfgs.keys()) | set(rolling_pred_dfgs.keys())

    for window_key in all_windows:
        # Initialize entry for this time window
        combined_dfgs[window_key] = {}

        # Add ground truth DFG if available
        if window_key in rolling_truth_dfgs:
            combined_dfgs[window_key]['truth'] = {
                'nodes': rolling_truth_dfgs[window_key]['dfg_json']['nodes'],
                'arcs': rolling_truth_dfgs[window_key]['dfg_json']['arcs']
            }
        else:
            combined_dfgs[window_key]['truth'] = {'nodes': [], 'arcs': []}

        # Add prediction DFG if available
        if window_key in rolling_pred_dfgs:
            combined_dfgs[window_key]['pred'] = {
                'nodes': rolling_pred_dfgs[window_key]['dfg_json']['nodes'],
                'arcs': rolling_pred_dfgs[window_key]['dfg_json']['arcs']
            }
        else:
            combined_dfgs[window_key]['pred'] = {'nodes': [], 'arcs': []}

    return combined_dfgs


class BackgroundModel:

    def __init__(self):
        self.number_of_events = 0
        self.number_of_traces = 0
        self.trace_frequency = {}
        self.labels = set()
        self.large_string = ''
        self.lprob = 0
        self.trace_size = {}
        self.log2_of_model_probability = {}
        self.total_number_non_fitting_traces = 0
        pass

    def open_trace(self):
        self.lprob = 0
        self.large_string = ''

    def process_event(self, event_label, probability):
        self.large_string += event_label
        self.number_of_events += 1
        self.labels.add(event_label)
        self.lprob += probability

    def close_trace(self, trace_length, fitting, final_state_prob):
        # print('Closing:', self.large_string)
        self.trace_size[self.large_string] = trace_length
        # print('Trace size:', trace_length)
        self.number_of_traces += 1
        if fitting:
            self.log2_of_model_probability[self.large_string] = (self.lprob + final_state_prob) / math.log(2)
        else:
            self.total_number_non_fitting_traces += 1
        tf = 0
        if self.large_string in self.trace_frequency.keys():
            tf = self.trace_frequency[self.large_string]
        self.trace_frequency[self.large_string] = tf + 1

    def h_0(self, accumulated_rho, total_number_of_traces):
        if accumulated_rho == 0 or accumulated_rho == total_number_of_traces:
            return 0
        else:
            p = (accumulated_rho / total_number_of_traces)
            return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def compute_relevance(self):
        accumulated_rho = 0
        accumulated_cost_bits = 0
        accumulated_temp_cost_bits = 0
        accumulated_prob_fitting_traces = 0

        for trace_string, trace_freq in self.trace_frequency.items():
            cost_bits = 0
            nftrace_cost_bits = 0

            if trace_string in self.log2_of_model_probability:
                cost_bits = - self.log2_of_model_probability[trace_string]
                accumulated_rho += trace_freq
            else:
                cost_bits = (1 + self.trace_size[trace_string]) * math.log2(1 + len(self.labels))
                nftrace_cost_bits += trace_freq

            accumulated_temp_cost_bits += nftrace_cost_bits * trace_freq
            accumulated_cost_bits += (cost_bits * trace_freq) / self.number_of_traces

            if trace_string in self.log2_of_model_probability:
                accumulated_prob_fitting_traces += trace_freq / self.number_of_traces

        entropic_relevance = self.h_0(accumulated_rho, self.number_of_traces) + accumulated_cost_bits
        return entropic_relevance

def convert_dfg_into_automaton_new(nodes, arcs, method='truth'):
    """
    Convert DFG to automaton with different transition probability calculation methods

    Args:
        nodes: List of node dictionaries from the DFG
        arcs: List of arc dictionaries from the DFG
        method: 'truth' or 'prediction' - determines how to calculate probabilities for start/end symbols

    Returns:
        Tuple of (transitions, sources, final_states, trans_table)
    """
    agg_outgoing_frequency = {}
    node_info = {node['id']: node['label'] for node in nodes}

    sinks = set(node_info.keys())
    sources = list(node_info.keys())

    for arc in arcs:
        if arc['freq'] > 0:
            arc_from = 0
            if arc['from'] in agg_outgoing_frequency.keys():
                arc_from = agg_outgoing_frequency[arc['from']]
            agg_outgoing_frequency[arc['from']] = arc_from + arc['freq']
            sinks.discard(arc['from'])
            if arc['to'] in sources:
                sources.remove(arc['to'])

    transitions = {}
    for arc in arcs:
        if arc['freq'] > 0:
            if arc['to'] not in sinks:
                from_node = arc['from']
                to_node = arc['to']
                label = node_info[to_node]

                # Special handling for start/end symbols in prediction mode
                if method == 'prediction' and (node_info[from_node] == '▶' or label == '■'):
                    # For predictions, always use probability 1.0 for start/end related transitions
                    transitions[(from_node, label)] = (to_node, 1.0)
                else:
                    # Normal probability calculation
                    transitions[(from_node, label)] = (to_node, arc['freq'] / agg_outgoing_frequency[from_node])

    for sink in sinks:
        del node_info[sink]

    states = set()
    outgoing_prob = {}
    trans_table = {}

    for (t_from, label), (t_to, a_prob) in transitions.items():
        trans_table[(t_from, label)] = (t_to, math.log(a_prob))
        states.add(t_from)
        states.add(t_to)
        t_f = 0
        if t_from in outgoing_prob.keys():
            t_f = outgoing_prob[t_from]
        outgoing_prob[t_from] = t_f + a_prob

    final_states = {}
    for state in states:
        if not state in outgoing_prob.keys() or 1.0 - outgoing_prob[state] > 0.000006:
            d_p = 0
            if state in outgoing_prob.keys():
                d_p = outgoing_prob[state]
            final_states[state] = math.log(1 - d_p)

    g = nx.DiGraph()
    for (t_from, label), (t_to, prob) in transitions.items():
        g.add_edge(t_from, t_to, label=label + ' - ' + str(round(prob, 3)))

    tR = set()
    for source in sources:
        available = False
        for start, end in transitions.items():
            if source in start or source in end:
                available = True
        if not available:
            tR.add(source)
    for tRe in tR:
        sources.remove(tRe)

    return transitions, sources, final_states, trans_table

def calculate_entropic_relevance_corr_new(dfg, log_truth, method='truth'):
    transitions, sources, final_states, trans_table = convert_dfg_into_automaton_new(dfg['nodes'], dfg['arcs'], method)

    if isinstance(log_truth, pd.DataFrame):
        # Convert DataFrame to a list of traces
        traces = []
        for case_id, case_events in log_truth.groupby('case:concept:name'):
            trace = []
            for _, event in case_events.iterrows():
                trace.append({'concept:name': event['concept:name']})
            traces.append(trace)
        log_truth = traces


    # assert len(sources) == 1
    ers = []

    fitting_traces = {}
    non_fitting_traces = {}

    # fitting_traces = {}
    # fitting_trace_occurrences = {}
    #
    # non_fitting_traces = {}
    # non_fitting_trace_occurrences = {}
    #
    # non_fitting_traces_key = {}
    #
    # num_non_fitting_traces = 0
    # num_fitting_traces = 0
    for source in sources:
        info_gatherer = BackgroundModel()

        initial_state = source
        for t, trace in enumerate(log_truth):
            curr = initial_state
            non_fitting = False
            info_gatherer.open_trace()
            len_trace = 0
            # print('Current state:', curr)

            # trace_key = str(t) + "_" + "_".join([event['concept:name'] for event in trace])
            trace_pattern = "_".join([event['concept:name'] for event in trace])
            for event in trace:
                label = event['concept:name']

                # print(label)
                if label in ['▶', '■']:
                    continue
                len_trace += 1
                prob = 0
                if not non_fitting and (curr, label) in trans_table.keys():
                    curr, prob = trans_table[(curr, label)]

                    # if trace_pattern not in non_fitting_traces:
                    #     fitting_traces[trace_pattern] = event['concept:name']
                    #
                    # fitting_trace_occurrences[trace_pattern] = fitting_trace_occurrences.get(trace_pattern, 0) + 1
                    #
                    # num_fitting_traces += 1

                else:
                    print('Not fitting at ', event['concept:name'])
                    # non_fitting_traces_key[trace_key] = event['concept:name']
                    # if trace_pattern not in non_fitting_traces:
                    #     non_fitting_traces[trace_pattern] = event['concept:name']
                    #
                    # non_fitting_trace_occurrences[trace_pattern] = non_fitting_trace_occurrences.get(trace_pattern, 0) + 1
                    #
                    # num_non_fitting_traces += 1
                    print('Trace:\n')
                    string_p = ''
                    for eve in trace:
                        string_p += eve['concept:name'] + ' - '
                    print(string_p)
                    non_fitting = True
                info_gatherer.process_event(label, prob)

            if not non_fitting and curr in final_states.keys():
                info_gatherer.close_trace(len_trace, True, final_states[curr])
                fitting_traces[trace_pattern] = fitting_traces.get(trace_pattern, 0) + 1

            else:
                info_gatherer.close_trace(len_trace, False, 0)
                non_fitting_traces[trace_pattern] = non_fitting_traces.get(trace_pattern, 0) + 1

        print('Non_fitting:', info_gatherer.total_number_non_fitting_traces)
        print(info_gatherer.number_of_traces)

        entropic_relevance = info_gatherer.compute_relevance()
        ers.append(entropic_relevance)

    entropic_relevance = min(ers)
    # print('Entropic relevance:', entropic_relevance)
    return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces, fitting_traces, non_fitting_traces


def calculate_rolling_entropic_relevance(combined_rolling_dfgs, seq_test_log):
    """
    Calculate entropic relevance for both truth and prediction DFGs across rolling time windows.

    Parameters:
    - combined_rolling_dfgs: Dictionary with time windows containing both 'truth' and 'pred' DFGs
    - seq_test_log: Dictionary of sublogs for each time window

    Returns:
    - Dictionary with entropic relevance metrics for each time window
    """
    results = {}

    for window_key, dfgs in combined_rolling_dfgs.items():
        print(f"\nProcessing window: {window_key}")

        if window_key not in seq_test_log:
            print(f"No log data for window {window_key}, skipping...")
            continue

        sublog = seq_test_log[window_key]

        # Calculate metrics for ground truth DFG
        print("Calculating entropic relevance for ground truth DFG...")
        if 'truth' in dfgs and dfgs['truth']['nodes'] and dfgs['truth']['arcs']:
            try:
                truth_er, truth_non_fitting, truth_total, truth_fitting_traces, truth_non_fitting_traces = calculate_entropic_relevance_corr_new(
                    dfgs['truth'], sublog, method='truth'
                )
                truth_fitting_ratio = 1 - (truth_non_fitting / truth_total) if truth_total > 0 else 0

                print(f"Truth ER: {truth_er:.4f}, Fitting ratio: {truth_fitting_ratio:.2%}")
            except Exception as e:
                print(f"Error calculating truth metrics: {str(e)}")
                truth_er = float('nan')
                truth_non_fitting = 0
                truth_total = 0
                truth_fitting_ratio = 0
                truth_fitting_traces = {}
                truth_non_fitting_traces = {}
        else:
            print("No truth DFG data available")
            truth_er = float('nan')
            truth_non_fitting = 0
            truth_total = 0
            truth_fitting_ratio = 0
            truth_fitting_traces = {}
            truth_non_fitting_traces = {}

        # Calculate metrics for prediction DFG
        print("Calculating entropic relevance for prediction DFG...")
        if 'pred' in dfgs and dfgs['pred']['nodes'] and dfgs['pred']['arcs']:
            try:
                pred_er, pred_non_fitting, pred_total, pred_fitting_traces, pred_non_fitting_traces = calculate_entropic_relevance_corr_new(
                    dfgs['pred'], sublog, method='pred'
                )
                pred_fitting_ratio = 1 - (pred_non_fitting / pred_total) if pred_total > 0 else 0

                print(f"Pred ER: {pred_er:.4f}, Fitting ratio: {pred_fitting_ratio:.2%}")
            except Exception as e:
                print(f"Error calculating prediction metrics: {str(e)}")
                pred_er = float('nan')
                pred_non_fitting = 0
                pred_total = 0
                pred_fitting_ratio = 0
                pred_fitting_traces = {}
                pred_non_fitting_traces = {}
        else:
            print("No prediction DFG data available")
            pred_er = float('nan')
            pred_non_fitting = 0
            pred_total = 0
            pred_fitting_ratio = 0
            pred_fitting_traces = {}
            pred_non_fitting_traces = {}

        # Store results for this window
        results[window_key] = {
            'truth': {
                'entropic_relevance': truth_er,
                'non_fitting_traces': truth_non_fitting,
                'total_traces': truth_total,
                'fitting_ratio': truth_fitting_ratio,
                'fitting_traces': truth_fitting_traces,
                'non_fitting_traces': truth_non_fitting_traces
            },
            'pred': {
                'entropic_relevance': pred_er,
                'non_fitting_traces': pred_non_fitting,
                'total_traces': pred_total,
                'fitting_ratio': pred_fitting_ratio,
                'fitting_traces': pred_fitting_traces,
                'non_fitting_traces': pred_non_fitting_traces
            }
        }

    return results


def calculate_er_metrics(rolling_er_results):
    """
    Calculate MAE, RMSE, and MAPE between truth and prediction entropic relevance values.

    Parameters:
    - rolling_er_results: Dictionary with entropic relevance metrics for each time window

    Returns:
    - Dictionary with calculated metrics
    """
    # Collect paired ER values from windows where both truth and pred have valid values
    truth_values = []
    pred_values = []

    for window_key, results in rolling_er_results.items():
        truth_er = results['truth']['entropic_relevance']
        pred_er = results['pred']['entropic_relevance']

        # Only include if both values are valid numbers
        if not (math.isnan(truth_er) or math.isnan(pred_er)):
            truth_values.append(truth_er)
            pred_values.append(pred_er)

    # Check if we have valid pairs to compare
    if not truth_values:
        print("No valid pairs of entropic relevance values found.")
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'mape': float('nan')
        }

    # Calculate metrics
    n = len(truth_values)

    # Mean Absolute Error
    mae = sum(abs(t - p) for t, p in zip(truth_values, pred_values)) / n

    # Root Mean Square Error
    rmse = math.sqrt(sum((t - p) ** 2 for t, p in zip(truth_values, pred_values)) / n)

    # Mean Absolute Percentage Error
    # Avoid division by zero by checking if truth value is zero
    mape_values = []
    for t, p in zip(truth_values, pred_values):
        if t != 0:  # Avoid division by zero
            mape_values.append(abs((t - p) / t))

    mape = sum(mape_values) / len(mape_values) * 100 if mape_values else float('nan')

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n': n
    }

# All datasets with their start dates
DATASETS = {
    'BPI2017': '2016-10-22 00:00:00',
    'sepsis': '2015-01-05 00:00:00',
    'Hospital_Billing': '2015-02-05 00:00:00',
    'BPI2019_1': '2018-11-19 00:00:00'
}

# All models to evaluate
MODELS = {
    'baseline': ['persistence', 'naive_seasonal'],
    'statistical': ['ar2'],
    'regression': ['random_forest', 'xgboost'],
    'deep_learning': ['rnn', 'deepar']
}

# Fixed horizon
HORIZON = '7'


def run_er_calculation(dataset, model_group, model_name, start_time, horizon):
    """
    Run entropic relevance calculation for a specific dataset and model
    """
    print(f"\n\n===== PROCESSING {dataset} with {model_name} =====")

    # Load log file
    log_file = f'data/interim/processed_logs/{dataset}.xes'
    log = pm4py.read_xes(log_file)

    # Extract rolling window sublogs
    seq_test_log = extract_rolling_window_sublogs(log, 'case:concept:name', 'concept:name', 'time:timestamp',
                                                  start_time, horizon)

    # Create ground truth DFGs
    rolling_truth_dfgs = create_dfgs_from_rolling_window(seq_test_log)

    # Load model predictions
    prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet'

    try:
        predictions_df = pd.read_parquet(prediction_file)

        # Aggregate predictions
        agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')
        agg_pred_round = agg_pred.round(0).astype(int)

        # Create prediction DFGs
        rolling_pred_dfgs = create_dfgs_from_rolling_predictions(seq_test_log, agg_pred_round)

        # Reformat rolling DFGs
        combined_rolling_dfgs = reformat_rolling_dfgs(rolling_truth_dfgs, rolling_pred_dfgs)

        # Calculate entropic relevance
        rolling_er_results = calculate_rolling_entropic_relevance(combined_rolling_dfgs, seq_test_log)

        # Calculate metrics
        er_metrics = calculate_er_metrics(rolling_er_results)

        # Save results
        output_dir = f'results/{dataset}/horizon_{horizon}/er_metrics'
        os.makedirs(output_dir, exist_ok=True)

        metrics_result = {
            'overall': er_metrics,
            'window_metrics': {}
        }

        for window_key, metrics in rolling_er_results.items():
            truth_er = metrics['truth']['entropic_relevance']
            pred_er = metrics['pred']['entropic_relevance']

            if not (math.isnan(truth_er) or math.isnan(pred_er)):
                abs_error = abs(truth_er - pred_er)
                pct_error = abs_error / truth_er * 100 if truth_er != 0 else float('nan')

                metrics_result['window_metrics'][window_key] = {
                    'truth_er': truth_er,
                    'pred_er': pred_er,
                    'abs_error': abs_error,
                    'pct_error': pct_error,
                    'truth_fitting_ratio': metrics['truth']['fitting_ratio'],
                    'pred_fitting_ratio': metrics['pred']['fitting_ratio']
                }

        output_file = f'{output_dir}/{model_group}_{model_name}_er_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics_result, f, indent=2)

        print(f"Saved metrics to {output_file}")

        return er_metrics

    except Exception as e:
        print(f"Error processing {dataset} with {model_name}: {str(e)}")
        return None


def main():
    """
    Main function to run ER calculation for all datasets and models
    """
    results = {}

    for dataset, start_time in DATASETS.items():
        results[dataset] = {}

        for model_group, models in MODELS.items():
            for model_name in models:
                metrics = run_er_calculation(dataset, model_group, model_name, start_time, HORIZON)

                if metrics:
                    results[dataset][f"{model_group}_{model_name}"] = {
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape'],
                        'n': metrics['n']
                    }

    # Save summary results
    with open(f'results/er_metrics_summary_horizon_{HORIZON}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n===== SUMMARY OF ENTROPIC RELEVANCE METRICS =====")
    print(f"{'Dataset':<15} {'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Windows':<10}")
    print("-" * 75)

    for dataset in results:
        for model, metrics in results[dataset].items():
            print(f"{dataset:<15} {model:<20} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
                  f"{metrics['mape']:<10.2f}% {metrics['n']:<10}")


if __name__ == "__main__":
    main()