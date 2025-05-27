import pm4py
import pandas as pd
import numpy as np
import os
import json
import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Union, Set



class DFGConstructor:
    """
    A class for constructing Directly-Follows Graphs from event logs and predictions.
    """

    def extract_training_log(self, log: Union[pd.DataFrame, List],
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


    def extract_time_period_sublog(self, df: pd.DataFrame,
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

    def extract_rolling_window_sublogs(self, df: pd.DataFrame,
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
            sublog = self.extract_time_period_sublog(
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

    def create_dfg_from_training(self, training_log: Union[pd.DataFrame, List]) -> Dict[Tuple[str, str], int]:
        """
        Create DFG from training log.

        Parameters:
        - training_log: Training event log (DataFrame or pm4py log object)

        Returns:
        - Directly-follows graph as dictionary
        """
        print("Discovering DFG from training log")
        dfg, _, _ = pm4py.discover_dfg(training_log)
        return dfg



    def create_dfgs_from_rolling_training(self, seq_test_log: Dict[str, pd.DataFrame],
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
        training_log, training_days = self.extract_training_log(
            raw_log, case_id_col, activity_col, timestamp_col, training_ratio=0.8
        )

        training_log_with_artifical = pm4py.insert_artificial_start_end(training_log)
        full_training_dfg, _, _ = pm4py.discover_dfg(training_log_with_artifical)

        # 2. Create JSON representation once
        dfg_json = self.create_dfg_from_truth(full_training_dfg)

        # 3. Reuse the same scaled DFG and JSON for all windows
        for window_key in seq_test_log.keys():
            print(f"Creating training DFG for window: {window_key}")

            # Store in dictionary - reusing the same objects for all windows
            training_dfgs[window_key] = {
                # 'dfg': horizon_unit_training_dfg,
                'dfg': full_training_dfg,
                'dfg_json': dfg_json,
                'training_log': training_log
            }

        return training_dfgs



    def create_dfg_from_truth(self, dfg_truth: Dict[Tuple[str, str], int]) -> Dict[str, List]:
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

    def create_dfgs_from_rolling_window(self, seq_test_log: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
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
            dfg_json = self.create_dfg_from_truth(dfg)

            # Store in dictionary
            dfgs_dict[window_key] = {
                'dfg': dfg,
                'dfg_json': dfg_json,
                'sublog': sublog
            }

        return dfgs_dict

    def create_dfg_from_predictions(self, predictions_df: pd.DataFrame) -> Dict[str, List]:
        """
        Create DFG structure from predictions dataframe with special start/end connections
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

                # Only add activities that have non-zero frequencies
                if any(predictions_df[df_relation] > 0):
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

        # Add connections from each activity to '■' with frequency 1
        for activity in original_activities:
            arcs.append({
                'from': reverse_map[activity],
                'to': reverse_map['■'],
                'freq': 1
            })
            node_freq['■'] += 1

        # Create nodes list
        nodes = []
        for node, node_id in reverse_map.items():
            nodes.append({
                'label': node,
                'id': node_id,
                'freq': round(node_freq.get(node, 0))
            })

        return {'nodes': nodes, 'arcs': arcs}

    def create_dfgs_from_rolling_predictions(self, seq_test_log: Dict[str, pd.DataFrame],
                                             agg_pred_round: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create a series of DFGs from rolling window predictions.
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
                dfg_json = self.create_dfg_from_predictions(window_pred)

                # Store in dictionary
                dfgs_dict[window_key] = {
                    'dfg_json': dfg_json,
                    'window_pred': window_pred
                }

        return dfgs_dict

    def reformat_rolling_dfgs(self, rolling_truth_dfgs: Dict[str, Dict],
                              rolling_pred_dfgs: Dict[str, Dict],
                              rolling_training_dfgs: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """
        Reformat rolling DFGs to have truth, prediction, and training data for each time window.
        """
        combined_dfgs = {}

        # Find all time windows that exist in any dictionary
        all_windows = set(rolling_truth_dfgs.keys()) | set(rolling_pred_dfgs.keys())
        if rolling_training_dfgs:
            all_windows |= set(rolling_training_dfgs.keys())

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

            # Add training DFG if available
            if rolling_training_dfgs and window_key in rolling_training_dfgs:
                combined_dfgs[window_key]['training'] = {
                    'nodes': rolling_training_dfgs[window_key]['dfg_json']['nodes'],
                    'arcs': rolling_training_dfgs[window_key]['dfg_json']['arcs']
                }
            else:
                combined_dfgs[window_key]['training'] = {'nodes': [], 'arcs': []}

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

    def open_trace(self):
        self.lprob = 0
        self.large_string = ''

    def process_event(self, event_label, probability):
        self.large_string += event_label
        self.number_of_events += 1
        self.labels.add(event_label)
        self.lprob += probability

    def close_trace(self, trace_length, fitting, final_state_prob):
        self.trace_size[self.large_string] = trace_length
        self.number_of_traces += 1
        if fitting:
            # self.log2_of_model_probability[self.large_string] = (self.lprob + final_state_prob) / math.log(2)
            self.log2_of_model_probability[self.large_string] = self.lprob
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

class ERCalculator:
    """
    A class for calculating Entropic Relevance metrics from process models.
    """

    def convert_dfg_into_automaton(self, nodes: List[Dict], arcs: List[Dict]) -> Tuple[Dict, List, Dict, Dict]:
        """
        Convert DFG to automaton with different transition probability calculation methods
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
                    if node_info[from_node] == '▶' or label == '■':
                        # Always use probability 1.0 for start/end related transitions
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
            # trans_table[(t_from, label)] = (t_to, math.log(a_prob))
            trans_table[(t_from, label)] = (t_to, math.log2(a_prob))
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
                # final_states[state] = math.log(1 - d_p)
                final_states[state] = math.log2(1 - d_p)

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

    def calculate_entropic_relevance(self, dfg: Dict, log_truth: Union[pd.DataFrame, List]) -> Tuple[float, int, int, Dict, Dict, Dict]:
        """
        Calculate entropic relevance for a DFG against a log
        """
        transitions, sources, final_states, trans_table = self.convert_dfg_into_automaton(
            dfg['nodes'], dfg['arcs']
        )

        if isinstance(log_truth, pd.DataFrame):
            # Convert DataFrame to a list of traces
            traces = []
            for case_id, case_events in log_truth.groupby('case:concept:name'):
                trace = []
                for _, event in case_events.iterrows():
                    trace.append({'concept:name': event['concept:name']})
                traces.append(trace)
            log_truth = traces

        ers = []
        fitting_traces = {}
        non_fitting_traces = {}

        for source in sources:
            info_gatherer = BackgroundModel()

            initial_state = source
            for t, trace in enumerate(log_truth):
                curr = initial_state
                non_fitting = False
                info_gatherer.open_trace()
                len_trace = 0

                trace_pattern = "_".join([event['concept:name'] for event in trace])
                for event in trace:
                    label = event['concept:name']

                    if label in ['▶', '■']:
                        continue
                    len_trace += 1
                    prob = 0
                    if not non_fitting and (curr, label) in trans_table.keys():
                        curr, prob = trans_table[(curr, label)]
                    else:
                        print('Not fitting at ', event['concept:name'])
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
        return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces, fitting_traces, non_fitting_traces, transitions

    def calculate_er_for_fitting_non_fitting(self, dfg: Dict, log_truth: Union[pd.DataFrame, List]) -> Dict[str, float]:
        """
        Calculate separate entropic relevance values for fitting and non-fitting traces.

        Parameters:
        - dfg: Dictionary containing nodes and arcs
        - log_truth: Event log to calculate ER against

        Returns:
        - Dictionary with ER values for fitting and non-fitting traces
        """
        # First identify which traces are fitting and non-fitting
        transitions, sources, final_states, trans_table = self.convert_dfg_into_automaton(
            dfg['nodes'], dfg['arcs']
        )

        if isinstance(log_truth, pd.DataFrame):
            # Convert DataFrame to a list of traces
            traces = []
            for case_id, case_events in log_truth.groupby('case:concept:name'):
                trace = []
                for _, event in case_events.iterrows():
                    trace.append({'concept:name': event['concept:name']})
                traces.append(trace)
            log_truth = traces

        # Separate traces into fitting and non-fitting
        fitting_traces = []
        non_fitting_traces = []

        # Use the first source as initial state to classify traces
        if not sources:
            return {
                'fitting_er': float('nan'),
                'non_fitting_er': float('nan')
            }

        initial_state = sources[0]

        for trace in log_truth:
            curr = initial_state
            non_fitting = False
            trace_pattern = "_".join([event['concept:name'] for event in trace])

            for event in trace:
                label = event['concept:name']
                if label in ['▶', '■']:
                    continue

                if not non_fitting and (curr, label) in trans_table.keys():
                    curr, _ = trans_table[(curr, label)]
                else:
                    non_fitting = True
                    break

            if not non_fitting and curr in final_states.keys():
                fitting_traces.append(trace)
            else:
                non_fitting_traces.append(trace)

        # Calculate ER for fitting traces
        fitting_er = self._calculate_er_for_traces(dfg, fitting_traces)

        # Calculate ER for non-fitting traces
        non_fitting_er = self._calculate_er_for_traces(dfg, non_fitting_traces)

        return {
            'fitting_er': fitting_er,
            'non_fitting_er': non_fitting_er,
            'fitting_count': len(fitting_traces),
            'non_fitting_count': len(non_fitting_traces)
        }

    def _calculate_er_for_traces(self, dfg: Dict, traces: List) -> float:
        """
        Calculate entropic relevance for a specific subset of traces.

        Parameters:
        - dfg: Dictionary containing nodes and arcs
        - traces: List of traces to calculate ER for

        Returns:
        - Entropic relevance value
        """
        if not traces:
            return float('nan')

        try:
            transitions, sources, final_states, trans_table = self.convert_dfg_into_automaton(
                dfg['nodes'], dfg['arcs']
            )

            min_er = float('inf')
            for source in sources:
                info_gatherer = BackgroundModel()

                initial_state = source
                for trace in traces:
                    curr = initial_state
                    non_fitting = False
                    info_gatherer.open_trace()
                    len_trace = 0

                    # Handle both string pattern keys and actual trace objects
                    if isinstance(trace, str):
                        # This is a trace pattern string, not an actual trace object
                        continue

                    for event in trace:
                        # Make sure event is a dictionary with concept:name
                        if isinstance(event, dict) and 'concept:name' in event:
                            label = event['concept:name']

                            if label in ['▶', '■']:
                                continue
                            len_trace += 1
                            prob = 0
                            if not non_fitting and (curr, label) in trans_table.keys():
                                curr, prob = trans_table[(curr, label)]
                            else:
                                non_fitting = True
                            info_gatherer.process_event(label, prob)

                    if not non_fitting and curr in final_states.keys():
                        info_gatherer.close_trace(len_trace, True, final_states[curr])
                    else:
                        info_gatherer.close_trace(len_trace, False, 0)

                entropic_relevance = info_gatherer.compute_relevance()
                if entropic_relevance < min_er:
                    min_er = entropic_relevance

            return min_er if min_er != float('inf') else float('nan')
        except Exception as e:
            print(f"Error calculating ER for subset of traces: {str(e)}")
            return float('nan')

    def calculate_rolling_entropic_relevance(self, combined_rolling_dfgs: Dict[str, Dict],
                                             seq_test_log: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Calculate entropic relevance for truth, prediction, and training DFGs across rolling time windows.
        Also computes separate ER values for fitting and non-fitting traces.
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
            truth_metrics = self._calculate_dfg_metrics(dfgs.get('truth', {'nodes': [], 'arcs': []}), sublog, "truth")

            # Calculate metrics for prediction DFG
            print("Calculating entropic relevance for prediction DFG...")
            pred_metrics = self._calculate_dfg_metrics(dfgs.get('pred', {'nodes': [], 'arcs': []}), sublog, "pred")

            # Calculate metrics for training DFG
            print("Calculating entropic relevance for training DFG...")
            training_metrics = self._calculate_dfg_metrics(dfgs.get('training', {'nodes': [], 'arcs': []}), sublog,
                                                           "training")

            # Store results for this window
            results[window_key] = {
                'truth': truth_metrics,
                'pred': pred_metrics,
                'training': training_metrics
            }

        return results

    def generate_er_metric_report(self, rolling_er_results: Dict[str, Dict], output_prefix: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report of entropic relevance metrics and save to JSON and CSV.

        Parameters:
        - rolling_er_results: Dictionary containing ER results for each window
        - output_prefix: Prefix for output filenames

        Returns:
        - Dictionary containing the formatted metrics report
        """
        # Calculate overall metrics
        er_metrics = self.calculate_er_metrics(rolling_er_results)

        # Calculate average and standard deviation across all windows
        truth_ers = []
        pred_ers = []
        training_ers = []
        truth_fit_ers = []
        truth_nonfit_ers = []
        pred_fit_ers = []
        pred_nonfit_ers = []
        training_fit_ers = []
        training_nonfit_ers = []
        errors = []
        pct_errors = []

        for metrics in rolling_er_results.values():
            truth_er = metrics['truth']['entropic_relevance']
            pred_er = metrics['pred']['entropic_relevance']
            training_er = metrics['training']['entropic_relevance']

            truth_fit_er = metrics['truth'].get('fitting_traces_ER', float('nan'))
            truth_nonfit_er = metrics['truth'].get('non_fitting_traces_ER', float('nan'))
            pred_fit_er = metrics['pred'].get('fitting_traces_ER', float('nan'))
            pred_nonfit_er = metrics['pred'].get('non_fitting_traces_ER', float('nan'))
            training_fit_er = metrics['training'].get('fitting_traces_ER', float('nan'))
            training_nonfit_er = metrics['training'].get('non_fitting_traces_ER', float('nan'))

            if not (math.isnan(truth_er) or math.isnan(pred_er)):
                truth_ers.append(truth_er)
                pred_ers.append(pred_er)
                training_ers.append(training_er)

                if not math.isnan(truth_fit_er):
                    truth_fit_ers.append(truth_fit_er)
                if not math.isnan(truth_nonfit_er):
                    truth_nonfit_ers.append(truth_nonfit_er)
                if not math.isnan(pred_fit_er):
                    pred_fit_ers.append(pred_fit_er)
                if not math.isnan(pred_nonfit_er):
                    pred_nonfit_ers.append(pred_nonfit_er)
                if not math.isnan(training_fit_er):
                    training_fit_ers.append(training_fit_er)
                if not math.isnan(training_nonfit_er):
                    training_nonfit_ers.append(training_nonfit_er)

                abs_error = abs(truth_er - pred_er)
                pct_error = abs_error / truth_er * 100 if truth_er != 0 else float('nan')

                if not math.isnan(pct_error):
                    errors.append(abs_error)
                    pct_errors.append(pct_error)

        # Compute statistics
        avg_truth_er = np.mean(truth_ers) if truth_ers else float('nan')
        std_truth_er = np.std(truth_ers) if truth_ers else float('nan')
        avg_pred_er = np.mean(pred_ers) if pred_ers else float('nan')
        std_pred_er = np.std(pred_ers) if pred_ers else float('nan')
        avg_training_er = np.mean(training_ers) if training_ers else float('nan')
        std_training_er = np.std(training_ers) if training_ers else float('nan')

        avg_truth_fit_er = np.mean(truth_fit_ers) if truth_fit_ers else float('nan')
        std_truth_fit_er = np.std(truth_fit_ers) if truth_fit_ers else float('nan')
        avg_truth_nonfit_er = np.mean(truth_nonfit_ers) if truth_nonfit_ers else float('nan')
        std_truth_nonfit_er = np.std(truth_nonfit_ers) if truth_nonfit_ers else float('nan')

        avg_pred_fit_er = np.mean(pred_fit_ers) if pred_fit_ers else float('nan')
        std_pred_fit_er = np.std(pred_fit_ers) if pred_fit_ers else float('nan')
        avg_pred_nonfit_er = np.mean(pred_nonfit_ers) if pred_nonfit_ers else float('nan')
        std_pred_nonfit_er = np.std(pred_nonfit_ers) if pred_nonfit_ers else float('nan')

        avg_training_fit_er = np.mean(training_fit_ers) if training_fit_ers else float('nan')
        std_training_fit_er = np.std(training_fit_ers) if training_fit_ers else float('nan')
        avg_training_nonfit_er = np.mean(training_nonfit_ers) if training_nonfit_ers else float('nan')
        std_training_nonfit_er = np.std(training_nonfit_ers) if training_nonfit_ers else float('nan')

        avg_error = np.mean(errors) if errors else float('nan')
        std_error = np.std(errors) if errors else float('nan')
        avg_pct_error = np.mean(pct_errors) if pct_errors else float('nan')
        std_pct_error = np.std(pct_errors) if pct_errors else float('nan')

        # Create a dictionary to store all metrics
        metrics_report = {
            'overall': er_metrics,
            'window_metrics': {},
            'statistics': {
                'avg_truth_er': avg_truth_er,
                'std_truth_er': std_truth_er,
                'avg_pred_er': avg_pred_er,
                'std_pred_er': std_pred_er,
                'avg_training_er': avg_training_er,
                'std_training_er': std_training_er,
                'avg_truth_fit_er': avg_truth_fit_er,
                'std_truth_fit_er': std_truth_fit_er,
                'avg_truth_nonfit_er': avg_truth_nonfit_er,
                'std_truth_nonfit_er': std_truth_nonfit_er,
                'avg_pred_fit_er': avg_pred_fit_er,
                'std_pred_fit_er': std_pred_fit_er,
                'avg_pred_nonfit_er': avg_pred_nonfit_er,
                'std_pred_nonfit_er': std_pred_nonfit_er,
                'avg_training_fit_er': avg_training_fit_er,
                'std_training_fit_er': std_training_fit_er,
                'avg_training_nonfit_er': avg_training_nonfit_er,
                'std_training_nonfit_er': std_training_nonfit_er,
                'avg_error': avg_error,
                'std_error': std_error,
                'avg_pct_error': avg_pct_error,
                'std_pct_error': std_pct_error
            }
        }

        # Build a detailed report for each window
        print("\n===== ENTROPIC RELEVANCE DETAILED REPORT =====")
        # Print averages and standard deviations first
        print("SUMMARY STATISTICS:")
        print(f"{'Metric':20} {'Average':10} {'Std Dev':10}")
        print(f"{'-' * 40}")
        print(f"{'Truth ER':20} {avg_truth_er:<10.4f} {std_truth_er:<10.4f}")
        print(f"{'Truth Fit ER':20} {avg_truth_fit_er:<10.4f} {std_truth_fit_er:<10.4f}")
        print(f"{'Truth Non-Fit ER':20} {avg_truth_nonfit_er:<10.4f} {std_truth_nonfit_er:<10.4f}")
        print(f"{'Pred ER':20} {avg_pred_er:<10.4f} {std_pred_er:<10.4f}")
        print(f"{'Pred Fit ER':20} {avg_pred_fit_er:<10.4f} {std_pred_fit_er:<10.4f}")
        print(f"{'Pred Non-Fit ER':20} {avg_pred_nonfit_er:<10.4f} {std_pred_nonfit_er:<10.4f}")
        print(f"{'Training ER':20} {avg_training_er:<10.4f} {std_training_er:<10.4f}")
        print(f"{'Training Fit ER':20} {avg_training_fit_er:<10.4f} {std_training_fit_er:<10.4f}")
        print(f"{'Training Non-Fit ER':20} {avg_training_nonfit_er:<10.4f} {std_training_nonfit_er:<10.4f}")
        print(f"{'Abs Error':20} {avg_error:<10.4f} {std_error:<10.4f}")
        print(f"{'% Error':20} {avg_pct_error:<10.2f}% {std_pct_error:<10.2f}%")

        print("\nWINDOW-BY-WINDOW DETAILS:")
        print(
            f"{'Window':20} {'Truth ER':10} {'Truth Fit':10} {'Truth Non-Fit':10} "
            f"{'Pred ER':10} {'Pred Fit':10} {'Pred Non-Fit':10} "
            f"{'Train ER':10} {'Train Fit':10} {'Train Non-Fit':10} {'Abs Error':10} {'% Error':10}"
        )
        print("-" * 130)

        for window_key, metrics in rolling_er_results.items():
            # Extract overall entropic relevance values
            truth_er = metrics['truth']['entropic_relevance']
            pred_er = metrics['pred']['entropic_relevance']
            training_er = metrics['training']['entropic_relevance']

            # Extract or calculate fitting and non-fitting ER components
            truth_total_traces = metrics['truth'].get('total_traces', 0)
            truth_fitting_ratio = metrics['truth'].get('fitting_ratio', 0)
            truth_fit_er = metrics['truth'].get('fitting_traces_ER', float('nan'))
            truth_nonfit_er = metrics['truth'].get('non_fitting_traces_ER', float('nan'))

            pred_total_traces = metrics['pred'].get('total_traces', 0)
            pred_fitting_ratio = metrics['pred'].get('fitting_ratio', 0)
            pred_fit_er = metrics['pred'].get('fitting_traces_ER', float('nan'))
            pred_nonfit_er = metrics['pred'].get('non_fitting_traces_ER', float('nan'))

            training_total_traces = metrics['training'].get('total_traces', 0)
            training_fitting_ratio = metrics['training'].get('fitting_ratio', 0)
            training_fit_er = metrics['training'].get('fitting_traces_ER', float('nan'))
            training_nonfit_er = metrics['training'].get('non_fitting_traces_ER', float('nan'))

            if not (math.isnan(truth_er) or math.isnan(pred_er)):
                abs_error = abs(truth_er - pred_er)
                pct_error = abs_error / truth_er * 100 if truth_er != 0 else float('nan')

                print(
                    f"{window_key:20} {truth_er:<10.4f} {truth_fit_er:<10.4f} {truth_nonfit_er:<10.4f} "
                    f"{pred_er:<10.4f} {pred_fit_er:<10.4f} {pred_nonfit_er:<10.4f} "
                    f"{training_er:<10.4f} {training_fit_er:<10.4f} {training_nonfit_er:<10.4f} "
                    f"{abs_error:<10.4f} {pct_error:<10.2f}%"
                )

                metrics_report['window_metrics'][window_key] = {
                    'truth_er': truth_er,
                    'truth_fit_er': truth_fit_er,
                    'truth_nonfit_er': truth_nonfit_er,
                    'truth_fitting_ratio': truth_fitting_ratio,
                    'truth_total_traces': truth_total_traces,
                    'pred_er': pred_er,
                    'pred_fit_er': pred_fit_er,
                    'pred_nonfit_er': pred_nonfit_er,
                    'pred_fitting_ratio': pred_fitting_ratio,
                    'pred_total_traces': pred_total_traces,
                    'training_er': training_er,
                    'training_fit_er': training_fit_er,
                    'training_nonfit_er': training_nonfit_er,
                    'training_fitting_ratio': training_fitting_ratio,
                    'training_total_traces': training_total_traces,
                    'abs_error': abs_error,
                    'pct_error': pct_error
                }
            else:
                print(
                    f"{window_key:20} {truth_er if not math.isnan(truth_er) else 'N/A':<10} {'N/A':<10} {'N/A':<10} "
                    f"{pred_er if not math.isnan(pred_er) else 'N/A':<10} {'N/A':<10} {'N/A':<10} "
                    f"{training_er if not math.isnan(training_er) else 'N/A':<10} {'N/A':<10} {'N/A':<10}"
                )

        # Save to JSON file
        json_output_file = f'{output_prefix}_er_metrics.json'
        with open(json_output_file, 'w') as f:
            json.dump(metrics_report, f, indent=2)

        # Save to CSV file
        csv_output_file = f'{output_prefix}_er_metrics.csv'
        df = pd.DataFrame.from_dict(metrics_report['window_metrics'], orient='index')
        df.to_csv(csv_output_file, index_label='window')

        print(f"\nDetailed metrics saved to {json_output_file} and {csv_output_file}")

        return metrics_report

    def _calculate_dfg_metrics(self, dfg: Dict, sublog: pd.DataFrame, dfg_type: str) -> Dict:
        """Helper method to calculate metrics for a DFG"""
        if dfg and dfg.get('nodes') and dfg.get('arcs'):
            try:
                er, non_fitting, total, fitting_traces, non_fitting_traces, transitions = self.calculate_entropic_relevance(
                    dfg, sublog
                )
                fitting_ratio = 1 - (non_fitting / total) if total > 0 else 0

                # Calculate separate ER values for fitting and non-fitting traces
                separate_er = self.calculate_er_for_fitting_non_fitting(dfg, sublog)
                fitting_traces_ER = separate_er['fitting_er']
                non_fitting_traces_ER = separate_er['non_fitting_er']

                print(f"{dfg_type.capitalize()} ER: {er:.4f}, Fitting ratio: {fitting_ratio:.2%}")
                print(f"{dfg_type.capitalize()} ER for fitting traces: {fitting_traces_ER:.4f}")
                print(f"{dfg_type.capitalize()} ER for non-fitting traces: {non_fitting_traces_ER:.4f}")

                return {
                    'entropic_relevance': er,
                    'non_fitting_traces': non_fitting,
                    'total_traces': total,
                    'fitting_ratio': fitting_ratio,
                    'fitting_traces': fitting_traces,
                    'non_fitting_traces': non_fitting_traces,
                    'transitions': transitions,
                    'fitting_traces_ER': fitting_traces_ER,
                    'non_fitting_traces_ER': non_fitting_traces_ER
                }
            except Exception as e:
                print(f"Error calculating {dfg_type} metrics: {str(e)}")
        else:
            print(f"No {dfg_type} DFG data available or empty DFG")

        # Return default values if no data or error occurs
        return {
            'entropic_relevance': float('nan'),
            'non_fitting_traces': 0,
            'total_traces': 0,
            'fitting_ratio': 0,
            'fitting_traces': {},
            'non_fitting_traces': {},
            'transitions': {},
            'fitting_traces_ER': float('nan'),
            'non_fitting_traces_ER': float('nan')
        }
    def calculate_er_metrics(self, rolling_er_results: Dict[str, Dict]) -> Dict[str, Union[float, int]]:
        """
        Calculate MAE, RMSE, and MAPE between truth and prediction entropic relevance values.
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

    def run_er_evaluation(self, dataset, horizon, model_group, model_name, start_time):
        """
        Run the complete entropic relevance evaluation workflow including training baseline.
        """
        print(f"Starting entropic relevance evaluation for {dataset} with {model_name} model")

        dfg_constructor = DFGConstructor()  # Create instance of DFGConstructor

        # Load log file
        log_file = f'data/interim/processed_logs/{dataset}.xes'
        log = pm4py.read_xes(log_file)

        # Extract rolling window sublogs
        print("Extracting rolling window sublogs...")
        seq_test_log = dfg_constructor.extract_rolling_window_sublogs(
            log, 'case:concept:name', 'concept:name', 'time:timestamp',
            start_time, horizon
        )

        # Create ground truth DFGs
        print("Creating ground truth DFGs...")
        rolling_truth_dfgs = dfg_constructor.create_dfgs_from_rolling_window(seq_test_log)

        # Create training DFGs (using 80% of the data)
        print("Creating training baseline DFGs...")
        rolling_training_dfgs = dfg_constructor.create_training_dfgs(
            seq_test_log,
            log,
            'case:concept:name',
            'concept:name',
            'time:timestamp'
        )

        # Load predictions
        print(f"Loading predictions from {model_name}...")
        prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet'
        predictions_df = pd.read_parquet(prediction_file)

        # Aggregate and round predictions
        agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')
        agg_pred_round = agg_pred.round(0).astype(int)

        # Create prediction DFGs
        print("Creating prediction DFGs...")
        rolling_pred_dfgs = dfg_constructor.create_dfgs_from_rolling_predictions(seq_test_log, agg_pred_round)

        # Combine ground truth, prediction, and training DFGs
        print("Combining truth, prediction, and training DFGs...")
        combined_rolling_dfgs = dfg_constructor.reformat_rolling_dfgs(
            rolling_truth_dfgs,
            rolling_pred_dfgs,
            rolling_training_dfgs
        )

        # Calculate entropic relevance
        print("Calculating entropic relevance metrics...")
        rolling_er_results = self.calculate_rolling_entropic_relevance(combined_rolling_dfgs, seq_test_log)

        # Calculate evaluation metrics for prediction vs truth
        print("Computing evaluation metrics...")
        er_metrics = self.calculate_er_metrics(rolling_er_results)

        # Print the metrics
        print("\n===== ENTROPIC RELEVANCE EVALUATION METRICS =====")
        print(f"Number of comparable windows: {er_metrics['n']}")
        print(f"Mean Absolute Error (MAE): {er_metrics['mae']:.4f}")
        print(f"Root Mean Square Error (RMSE): {er_metrics['rmse']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {er_metrics['mape']:.2f}%")

        # Window-by-window comparison
        print("\n===== WINDOW-BY-WINDOW COMPARISON =====")
        print(f"{'Window':20} {'Truth ER':10} {'Pred ER':10} {'Training ER':10} {'Abs Error':10} {'% Error':10}")
        print("-" * 70)

        metrics_result = {
            'overall': er_metrics,
            'window_metrics': {}
        }

        for window_key, metrics in rolling_er_results.items():
            truth_er = metrics['truth']['entropic_relevance']
            pred_er = metrics['pred']['entropic_relevance']
            training_er = metrics['training']['entropic_relevance']

            if not (math.isnan(truth_er) or math.isnan(pred_er)):
                abs_error = abs(truth_er - pred_er)
                pct_error = abs_error / truth_er * 100 if truth_er != 0 else float('nan')

                print(
                    f"{window_key:20} {truth_er:<10.4f} {pred_er:<10.4f} {training_er:<10.4f} {abs_error:<10.4f} {pct_error:<10.2f}%")

                metrics_result['window_metrics'][window_key] = {
                    'truth_er': truth_er,
                    'pred_er': pred_er,
                    'training_er': training_er,
                    'abs_error': abs_error,
                    'pct_error': pct_error
                }
            else:
                print(
                    f"{window_key:20} {truth_er if not math.isnan(truth_er) else 'N/A':<10} {pred_er if not math.isnan(pred_er) else 'N/A':<10} {training_er if not math.isnan(training_er) else 'N/A':<10}")

        # Save results to JSON
        output_file = f'{dataset}_{horizon}_{model_group}_{model_name}_er_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics_result, f, indent=2)

        print(f"\nResults saved to {output_file}")
        return metrics_result


class GraphVisualizer:
    """
    A class for visualizing process graphs.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def draw_graph(self):
        """
        Draw the directed graph using NetworkX and Matplotlib.
        """
        pos = nx.spring_layout(self.graph)
        labels = nx.get_edge_attributes(self.graph, 'label')
        weights = nx.get_edge_attributes(self.graph, 'weight')

        # Draw nodes and edges
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.show()