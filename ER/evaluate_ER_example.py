import pm4py
import pandas as pd
import numpy as np
import os
import json
import math
import networkx as nx

from ER.ER_calculation_final import truth_er

dataset = 'BPI2017'
horizon = '7'
model_group = 'deep_learning'
model_name = 'deepar'


prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet'

ground_truth_log = f'data/interim/processed_logs/{dataset}.xes'


#%% Ground Truth

# load test xes file
log = pm4py.read_xes(ground_truth_log)

time_length = pm4py.get_event_attribute_values(log, 'time:timestamp')
start_time = min(time_length)
end_time = max(time_length)



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

start_time = '2016-10-22 00:00:00'
end_time = '2016-10-29 00:00:00'

sub_ground_truth_log = extract_time_period_sublog(log, 'case:concept:name', 'concept:name', 'time:timestamp', start_time, end_time)

sub_ground_truth_log_start_end = pm4py.insert_artificial_start_end(sub_ground_truth_log)

dfg_truth_log, sa, ea = pm4py.discover_dfg(sub_ground_truth_log_start_end)

pm4py.save_vis_dfg(dfg_truth_log, sa, ea, 'dfg_truth_log.png')



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

dfg_truth_json = create_dfg_from_truth(dfg_truth_log)


#%% Predictions

# load predictions
predictions_df = pd.read_parquet(prediction_file)

agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')

agg_pred_round = agg_pred.round(0).astype(int)

agg_pred_first = agg_pred_round.iloc[[0]]

# generate prediction DFGs
def create_dfg_from_predictions(predictions_df):
    """
    Create DFG structure from predictions dataframe

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        DFG in JSON format
    """
    # Extract directly-follows relations
    reverse_map = {}
    reverse_map['▶'] = 0  # New Start symbol
    reverse_map['■'] = 1  # New End symbol
    reverse_map['Start'] = 2  # Original Start (replacement for '▶' in input)
    reverse_map['End'] = 3  # Original End (replacement for '■' in input)

    # Track all activities and their frequencies
    incoming_freq = {}
    outgoing_freq = {}

    # Process each column to extract the activities and map them to IDs
    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]

            # Replace symbols in source and target
            source = source.replace('▶', 'Start').replace('■', 'End')
            target = target.replace('▶', 'Start').replace('■', 'End')

            # Register activities
            if source not in reverse_map:
                reverse_map[source] = len(reverse_map)
                outgoing_freq[source] = 0
                incoming_freq[source] = 0

            if target not in reverse_map:
                reverse_map[target] = len(reverse_map)
                outgoing_freq[target] = 0
                incoming_freq[target] = 0

    # Calculate incoming and outgoing frequencies for each activity
    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]

            # Replace symbols in source and target
            source = source.replace('▶', 'Start').replace('■', 'End')
            target = target.replace('▶', 'Start').replace('■', 'End')

            # Get total frequency for this relation across all rows
            total_freq = 0
            for _, row in predictions_df.iterrows():
                freq = round(float(row[df_relation]))
                if freq > 0:
                    total_freq += freq

            # Update incoming and outgoing frequencies
            outgoing_freq[source] = outgoing_freq.get(source, 0) + total_freq
            incoming_freq[target] = incoming_freq.get(target, 0) + total_freq
            # if source not in ['Start', 'End']:
            #     outgoing_freq[source] = outgoing_freq.get(source, 0) + total_freq
            #
            # if target not in ['Start', 'End']:
            #     incoming_freq[target] = incoming_freq.get(target, 0) + total_freq

    # Create arcs from original relations
    arcs = []
    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]

            # Replace symbols in source and target
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

    # Create additional arcs from new '▶' to activities
    for activity, act_id in reverse_map.items():
        if activity not in ['▶', '■']:
        # if activity not in ['▶', '■', 'Start', 'End']:
            # Frequency from '▶' to activity = outgoing - incoming (if positive)
            diff = outgoing_freq.get(activity, 0) - incoming_freq.get(activity, 0)
            if diff > 0:
                arcs.append({
                    'from': reverse_map['▶'],
                    'to': act_id,
                    'freq': diff
                })

    # Create additional arcs from activities to new '■'
    for activity, act_id in reverse_map.items():
        if activity not in ['▶', '■']:
        # if activity not in ['▶', '■', 'Start', 'End']:
            # Frequency from activity to '■' = incoming - outgoing (if positive)
            diff = incoming_freq.get(activity, 0) - outgoing_freq.get(activity, 0)
            if diff > 0:
                arcs.append({
                    'from': act_id,
                    'to': reverse_map['■'],
                    'freq': diff
                })

    # Calculate node frequencies based on arcs
    node_freq = {node: 0 for node in reverse_map.keys()}

    # Calculate frequencies for all nodes
    for arc in arcs:
        # For start node and regular nodes, add outgoing frequency
        source_id = arc['from']
        target_id = arc['to']

        # Find node labels for these IDs
        source_label = None
        target_label = None
        for node, node_id in reverse_map.items():
            if node_id == source_id:
                source_label = node
            if node_id == target_id:
                target_label = node

        if source_label == '▶':
            node_freq['▶'] += arc['freq']

        # For all nodes, add incoming frequency
        if target_label:
            node_freq[target_label] += arc['freq']

    # Create nodes list
    nodes = []
    for node, node_id in reverse_map.items():
        nodes.append({
            'label': node,
            'id': node_id,
            'freq': round(node_freq.get(node, 0))
        })

    return {'nodes': nodes, 'arcs': arcs}

dfg_pred_json = create_dfg_from_predictions(agg_pred_first)


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

# Fix the broken function call
# dfg_pred_json = create_dfg_from_predictions_new(agg_pred_first)

dfg_pred_json_new = create_dfg_from_predictions_new(agg_pred_first)

#%% Calculate Entropic Relevance
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


def convert_dfg_into_automaton(nodes, arcs):
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
            # sources.discard(arc['to'])
            if arc['to'] in sources:
                sources.remove(arc['to'])

    # print('Outgoing frequencies:')
    # print(agg_outgoing_frequency)

    transitions = {}
    for arc in arcs:
        if arc['freq'] > 0:
            if arc['to'] not in sinks:
                label = node_info[arc['to']]
                transitions[(arc['from'], label)] = (arc['to'], arc['freq'] / agg_outgoing_frequency[arc['from']])

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
    data = {}
    for (t_from, label), (t_to, prob) in transitions.items():
        g.add_edge(t_from, t_to, label=label+' - ' + str(round(prob,3)))

    # for start, edge in transitions.items():
    #     print(start, edge)

    # for node in g.nodes:
    #     if g.in_degree(node) == 0:
    #         print(f'{node} has no entries')
    #     if g.out_degree(node) == 0:
    #         print(f'{node} has no exits')
    #         for edge in g.in_edges(node):
    #             print(g.get_edge_data(edge[0], edge[1]))

    # dot = nx.drawing.nx_pydot.to_pydot(g)
    # file_name = './dfgs/temp3'
    # with open(file_name + '.dot', 'w') as file:
    #     file.write(str(dot))
    # check_call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png'])
    # os.remove(file_name + '.dot')

    tR = set()
    for source in sources:
        available = False
        for start, end in transitions.items():
            # print(start, end)
            if source in start or source in end:
                available = True
        if not available:
            tR.add(source)
    for tRe in tR:
        sources.remove(tRe)

    # return transitions, list(sources)[0], final_states, trans_table
    return transitions, sources, final_states, trans_table


# def calculate_entropic_relevance(dfg, log_truth):
#     transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg['nodes'], dfg['arcs'])
#
#     # Check if log_truth is a DataFrame (which it is in this case) and convert it to the expected format
#     if isinstance(log_truth, pd.DataFrame):
#         # Convert DataFrame to a list of traces
#         traces = []
#         for case_id, case_events in log_truth.groupby('case:concept:name'):
#             trace = []
#             for _, event in case_events.iterrows():
#                 trace.append({'concept:name': event['concept:name']})
#             traces.append(trace)
#         log_truth = traces
#
#     # Track non-fitting activities
#     non_fitting_activities = {}
#     # Track non-fitting traces and their activities
#     non_fitting_traces = {}
#     # Collect non-fitting activity logs
#     non_fitting_logs = []
#
#     ers = []
#     for source in sources:
#         info_gatherer = BackgroundModel()
#
#         initial_state = source
#         for t, trace in enumerate(log_truth):
#             curr = initial_state
#             non_fitting = False
#             non_fitting_activities_in_trace = []  # Track all non-fitting activities in trace
#             info_gatherer.open_trace()
#             len_trace = 0
#
#             # Create a trace representation for logging
#             trace_str = ""
#             for evt in trace:
#                 trace_str += evt['concept:name'] + " → "
#             trace_str = trace_str[:-3]  # Remove the last arrow
#
#             for event in trace:
#                 label = event['concept:name']
#
#                 if label in ['▶', '■']:
#                     continue
#                 len_trace += 1
#                 prob = 0
#                 if not non_fitting and (curr, label) in trans_table.keys():
#                     curr, prob = trans_table[(curr, label)]
#                 else:
#                     # Instead of printing, log the message
#                     log_message = f"Not fitting at {event['concept:name']} in trace: {trace_str}"
#                     non_fitting_logs.append(log_message)
#
#                     # Log the non-fitting activity
#                     activity_name = event['concept:name']
#                     non_fitting_activities[activity_name] = non_fitting_activities.get(activity_name, 0) + 1
#                     non_fitting_activities_in_trace.append(activity_name)
#
#                     non_fitting = True
#                 info_gatherer.process_event(label, prob)
#
#             if non_fitting and non_fitting_activities_in_trace:
#                 # Log this trace and its first non-fitting activity
#                 non_fitting_traces[trace_str] = non_fitting_activities_in_trace[0]
#
#             if not non_fitting and curr in final_states.keys():
#                 info_gatherer.close_trace(len_trace, True, final_states[curr])
#             else:
#                 info_gatherer.close_trace(len_trace, False, 0)
#
#         print('Non_fitting:', info_gatherer.total_number_non_fitting_traces)
#         print(info_gatherer.number_of_traces)
#
#         entropic_relevance = info_gatherer.compute_relevance()
#         ers.append(entropic_relevance)
#
#     entropic_relevance = min(ers)
#
#     # Sort non-fitting activities by frequency in descending order
#     sorted_non_fitting = dict(sorted(non_fitting_activities.items(), key=lambda x: x[1], reverse=True))
#
#     # Create a summary of logs
#     non_fitting_summary = {
#         "total_non_fitting_traces": info_gatherer.total_number_non_fitting_traces,
#         "total_traces": info_gatherer.number_of_traces,
#         "non_fitting_activities": sorted_non_fitting,
#         "non_fitting_logs": non_fitting_logs[:100],  # Limit to first 100 logs to avoid too much output
#         "non_fitting_log_count": len(non_fitting_logs)
#     }
#
#     return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces, sorted_non_fitting, non_fitting_traces, non_fitting_summary
#
# def calculate_entropic_relevance_corr(dfg, log_truth):
#     transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg['nodes'], dfg['arcs'])
#
#     if isinstance(log_truth, pd.DataFrame):
#         # Convert DataFrame to a list of traces
#         traces = []
#         for case_id, case_events in log_truth.groupby('case:concept:name'):
#             trace = []
#             for _, event in case_events.iterrows():
#                 trace.append({'concept:name': event['concept:name']})
#             traces.append(trace)
#         log_truth = traces
#
#
#     # assert len(sources) == 1
#     ers = []
#
#     fitting_traces = {}
#     non_fitting_traces = {}
#
#     # fitting_traces = {}
#     # fitting_trace_occurrences = {}
#     #
#     # non_fitting_traces = {}
#     # non_fitting_trace_occurrences = {}
#     #
#     # non_fitting_traces_key = {}
#     #
#     # num_non_fitting_traces = 0
#     # num_fitting_traces = 0
#     for source in sources:
#         info_gatherer = BackgroundModel()
#
#         initial_state = source
#         for t, trace in enumerate(log_truth):
#             curr = initial_state
#             non_fitting = False
#             info_gatherer.open_trace()
#             len_trace = 0
#             # print('Current state:', curr)
#
#             # trace_key = str(t) + "_" + "_".join([event['concept:name'] for event in trace])
#             trace_pattern = "_".join([event['concept:name'] for event in trace])
#             for event in trace:
#                 label = event['concept:name']
#
#                 # print(label)
#                 if label in ['▶', '■']:
#                     continue
#                 len_trace += 1
#                 prob = 0
#                 if not non_fitting and (curr, label) in trans_table.keys():
#                     curr, prob = trans_table[(curr, label)]
#
#                     # if trace_pattern not in non_fitting_traces:
#                     #     fitting_traces[trace_pattern] = event['concept:name']
#                     #
#                     # fitting_trace_occurrences[trace_pattern] = fitting_trace_occurrences.get(trace_pattern, 0) + 1
#                     #
#                     # num_fitting_traces += 1
#
#                 else:
#                     print('Not fitting at ', event['concept:name'])
#                     # non_fitting_traces_key[trace_key] = event['concept:name']
#                     # if trace_pattern not in non_fitting_traces:
#                     #     non_fitting_traces[trace_pattern] = event['concept:name']
#                     #
#                     # non_fitting_trace_occurrences[trace_pattern] = non_fitting_trace_occurrences.get(trace_pattern, 0) + 1
#                     #
#                     # num_non_fitting_traces += 1
#                     print('Trace:\n')
#                     string_p = ''
#                     for eve in trace:
#                         string_p += eve['concept:name'] + ' - '
#                     print(string_p)
#                     non_fitting = True
#                 info_gatherer.process_event(label, prob)
#
#             if not non_fitting and curr in final_states.keys():
#                 info_gatherer.close_trace(len_trace, True, final_states[curr])
#                 fitting_traces[trace_pattern] = fitting_traces.get(trace_pattern, 0) + 1
#
#             else:
#                 info_gatherer.close_trace(len_trace, False, 0)
#                 non_fitting_traces[trace_pattern] = non_fitting_traces.get(trace_pattern, 0) + 1
#
#         print('Non_fitting:', info_gatherer.total_number_non_fitting_traces)
#         print(info_gatherer.number_of_traces)
#
#         entropic_relevance = info_gatherer.compute_relevance()
#         ers.append(entropic_relevance)
#
#     entropic_relevance = min(ers)
#     # print('Entropic relevance:', entropic_relevance)
#     return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces, fitting_traces, non_fitting_traces
#
# pred_er, pred_no_non_fitting_traces, pred_no_traces, pred_fitting_traces, pred_non_fitting_traces = calculate_entropic_relevance_corr(dfg_pred_json, sub_ground_truth_log_start_end)


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

# # For ground truth DFG
# transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg_truth_json['nodes'], dfg_truth_json['arcs'], 'truth')

# For prediction DFG
pred_transitions, pred_sources, pred_final_states, pred_trans_table = convert_dfg_into_automaton_new(dfg_pred_json_new['nodes'], dfg_pred_json_new['arcs'], 'prediction')





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

pred_er, pred_no_non_fitting_traces, pred_no_traces, pred_fitting_traces, pred_non_fitting_traces = calculate_entropic_relevance_corr_new(dfg_pred_json_new, sub_ground_truth_log_start_end, 'prediction')
truth_er, truth_no_non_fitting_traces, truth_no_traces, truth_fitting_traces, truth_non_fitting_traces = calculate_entropic_relevance_corr_new(dfg_truth_json, sub_ground_truth_log_start_end, 'truth')








#
#
#
#
# truth_er, truth_no_non_fitting_traces, truth_no_traces, truth_non_fitting_activities, truth_non_fitting_traces, truth_non_fitting_summary = calculate_entropic_relevance(dfg_truth_json, sub_ground_truth_log_start_end)
#
# # Access the logs
# print(f"Found {truth_non_fitting_summary['non_fitting_log_count']} non-fitting activities")
# print("\nSample of non-fitting activities (first 5):")
# for log in truth_non_fitting_summary['non_fitting_logs'][:5]:
#     print(log)
#
# pred_er, pred_no_non_fitting_traces, pred_no_traces, pred_non_fitting_activities, pred_non_fitting_traces, pred_non_fitting_summary = calculate_entropic_relevance(dfg_pred_json, sub_ground_truth_log_start_end)
#
# print(f"Found {pred_non_fitting_summary['non_fitting_log_count']} non-fitting activities")
# print("\nSample of non-fitting activities (first 5):")
# for log in pred_non_fitting_summary['non_fitting_logs'][:5]:
#     print(log)
#
#
# # Save logs to a file if needed
# with open(f"{dataset}_{horizon}_non_fitting_logs.txt", "w") as file:
#     for log in pred_non_fitting_summary['non_fitting_logs']:
#         file.write(log + "\n")
#
#
#
# truth_er, truth_no_non_fitting_traces, truth_no_traces, truth_non_fitting_activities, truth_non_fitting_traces = calculate_entropic_relevance(dfg_truth_json, sub_ground_truth_log_start_end)
#
# pred_er, pred_no_non_fitting_traces, pred_no_traces, pred_non_fitting_activities, pred_non_fitting_traces = calculate_entropic_relevance(dfg_pred_json, sub_ground_truth_log_start_end)
#
#
# def calculate_entropic_relevance_stat(dfg, log_truth):
#     transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg['nodes'], dfg['arcs'])
#
#     # Check if log_truth is a DataFrame (which it is in this case) and convert it to the expected format
#     if isinstance(log_truth, pd.DataFrame):
#         # Convert DataFrame to a list of traces
#         traces = []
#         for case_id, case_events in log_truth.groupby('case:concept:name'):
#             trace = []
#             for _, event in case_events.iterrows():
#                 trace.append({'concept:name': event['concept:name']})
#             traces.append(trace)
#         log_truth = traces
#
#     # Track non-fitting activities and their counts
#     non_fitting_activities = {}
#     # Track non-fitting traces with their first failing activity
#     non_fitting_traces = {}
#     # Track all non-fitting traces
#     all_non_fitting_traces = []
#
#     ers = []
#     for source in sources:
#         info_gatherer = BackgroundModel()
#
#         initial_state = source
#         for t, trace in enumerate(log_truth):
#             curr = initial_state
#             non_fitting = False
#             first_non_fitting_activity = None
#             info_gatherer.open_trace()
#             len_trace = 0
#
#             # Create a trace representation for logging
#             trace_str = ""
#             for evt in trace:
#                 trace_str += evt['concept:name'] + " → "
#             trace_str = trace_str[:-3]  # Remove the last arrow
#
#             for event in trace:
#                 label = event['concept:name']
#
#                 if label in ['▶', '■']:
#                     continue
#                 len_trace += 1
#                 prob = 0
#                 if not non_fitting and (curr, label) in trans_table.keys():
#                     curr, prob = trans_table[(curr, label)]
#                 else:
#                     # Only record the first non-fitting activity in this trace
#                     if not non_fitting:
#                         first_non_fitting_activity = label
#                         non_fitting_activities[label] = non_fitting_activities.get(label, 0) + 1
#
#                     non_fitting = True
#                 info_gatherer.process_event(label, prob)
#
#             # Record non-fitting trace and its first non-fitting activity
#             if non_fitting:
#                 if first_non_fitting_activity:
#                     non_fitting_traces[trace_str] = first_non_fitting_activity
#                 all_non_fitting_traces.append(trace_str)
#
#             if not non_fitting and curr in final_states.keys():
#                 info_gatherer.close_trace(len_trace, True, final_states[curr])
#             else:
#                 info_gatherer.close_trace(len_trace, False, 0)
#
#         entropic_relevance = info_gatherer.compute_relevance()
#         ers.append(entropic_relevance)
#
#     entropic_relevance = min(ers)
#
#     # Sort non-fitting activities by frequency in descending order
#     sorted_non_fitting = dict(sorted(non_fitting_activities.items(), key=lambda x: x[1], reverse=True))
#
#     # Group non-fitting traces by activity
#     trace_by_activity = {}
#     for trace, activity in non_fitting_traces.items():
#         if activity not in trace_by_activity:
#             trace_by_activity[activity] = []
#         trace_by_activity[activity].append(trace)
#
#     # Create summary statistics
#     non_fitting_summary = {
#         "total_non_fitting_traces": info_gatherer.total_number_non_fitting_traces,
#         "total_traces": info_gatherer.number_of_traces,
#         "first_non_fitting_activities": sorted_non_fitting,
#         "first_non_fitting_activities_count": sum(sorted_non_fitting.values()),
#         "traces_by_activity": trace_by_activity,
#         "all_non_fitting_traces": all_non_fitting_traces,
#     }
#
#     return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces, sorted_non_fitting, non_fitting_traces, non_fitting_summary
#
# truth_er, truth_no_non_fitting_traces, truth_no_traces, truth_non_fitting_activities, truth_non_fitting_traces, truth_non_fitting_summary = calculate_entropic_relevance_stat(dfg_truth_json, sub_ground_truth_log_start_end)
#
# pred_er, pred_no_non_fitting_traces, pred_no_traces, pred_non_fitting_activities, pred_non_fitting_traces, pred_non_fitting_summary = calculate_entropic_relevance_stat(dfg_pred_json, sub_ground_truth_log_start_end)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ###################################
#
#
# dfg_truth, sa, ea = pm4py.discover_dfg(sub_ground_truth_log_start_end)
#
# # alternatively
# def create_df_relations(filtered_log):
#     """
#     Create directly-follows relations from filtered log
#
#     Args:
#         filtered_log: DataFrame containing events
#
#     Returns:
#         Dictionary with directly-follows relations and their frequencies
#     """
#     # Sort by case and timestamp
#     filtered_log = filtered_log.sort_values(['case:concept:name', 'time:timestamp'])
#
#     # Initialize dictionaries for DFG, start and end activities
#     dfg = {}
#     start_activities = {}
#     end_activities = {}
#
#     # Process each case
#     for case_id, case_events in filtered_log.groupby('case:concept:name'):
#         events = case_events['concept:name'].tolist()
#
#         # Record start activity
#         if len(events) > 0:
#             start_act = events[0]
#             if start_act in start_activities:
#                 start_activities[start_act] += 1
#             else:
#                 start_activities[start_act] = 1
#
#         # Record end activity
#         if len(events) > 0:
#             end_act = events[-1]
#             if end_act in end_activities:
#                 end_activities[end_act] += 1
#             else:
#                 end_activities[end_act] = 1
#
#         # Create directly-follows relations
#         for i in range(len(events) - 1):
#             source, target = events[i], events[i + 1]
#             if (source, target) in dfg:
#                 dfg[(source, target)] += 1
#             else:
#                 dfg[(source, target)] = 1
#
#     return dfg, start_activities, end_activities
#
# dfg_truth_log_manual, sa_m, ea_m = create_df_relations(sub_ground_truth_log_start_end)
#
#
# #### Predictions
#
# # load predictions
# predictions_df = pd.read_parquet(prediction_file)
#
# agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')
#
# agg_pred_round = agg_pred.round(0).astype(int)
#
# agg_pred_first = agg_pred_round.iloc[[0]]
#
# # generate prediction DFGs
# def create_dfg_from_predictions(predictions_df):
#     """
#     Create DFG structure from predictions dataframe
#
#     Args:
#         predictions_df: DataFrame with predictions
#
#     Returns:
#         DFG in JSON format
#     """
#     # Extract directly-follows relations
#     node_map = {}
#     reverse_map = {}
#     reverse_map['Start'] = 0
#     reverse_map['End'] = 1
#
#     # Process each column in predictions_df
#     # Each column represents a directly-follows relation
#     for df_relation in predictions_df.columns:
#         # Some columns might already be using '▶' or '■' as symbols for start/end
#         source, end = df_relation.replace('▶', 'Start').replace('■', 'End').split('->')
#         source = source.strip()
#         end = end.strip()
#
#         if source not in reverse_map:
#             reverse_map[source] = len(reverse_map)
#         if end not in reverse_map:
#             reverse_map[end] = len(reverse_map)
#
#     # Create arcs
#     arcs = []
#     node_freq = {node: 0 for node in reverse_map.keys()}
#
#     # Process each directly-follows relation
#     for df_relation in predictions_df.columns:
#         clean_relation = df_relation.replace('▶', 'Start').replace('■', 'End')
#         source, end = [part.strip() for part in clean_relation.split('->')]
#
#         # For each time step in the horizon, extract the prediction value
#         for _, row in predictions_df.iterrows():
#             freq = round(float(row[df_relation]))
#             if freq <= 0:
#                 continue
#
#             arcs.append({
#                 'from': reverse_map[source],
#                 'to': reverse_map[end],
#                 'freq': freq
#             })
#
#             if source == 'Start':
#                 node_freq[source] += freq
#             else:
#                 node_freq[end] += freq
#
#     # Create nodes
#     nodes = []
#     for node, freq in node_freq.items():
#         nodes.append({'label': node, 'id': reverse_map[node], 'freq': round(freq)})
#
#     return {'nodes': nodes, 'arcs': arcs}
#
#
# dfg_pred = create_dfg_from_predictions(agg_pred_first)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def add_start_end_symbols_to_log(log_df):
#     """
#     Add '▶' start and '■' end symbols to each trace in the log
#
#     Args:
#         log_df: DataFrame containing the event log
#
#     Returns:
#         Modified DataFrame with start and end symbols added to each trace
#     """
#     modified_log = log_df.copy()
#
#     # Replace any existing '▶' or '■' symbols with 'Start' and 'End'
#     modified_log['concept:name'] = modified_log['concept:name'].apply(
#         lambda x: x.replace('▶', 'Start').replace('■', 'End') if isinstance(x, str) else x
#     )
#
#     # Get the list of case IDs
#     case_ids = modified_log['case:concept:name'].unique()
#
#     # Create new rows to add
#     new_rows = []
#
#     for case_id in case_ids:
#         case_events = modified_log[modified_log['case:concept:name'] == case_id]
#
#         # Get the earliest and latest timestamp for the case
#         earliest_time = case_events['time:timestamp'].min()
#         latest_time = case_events['time:timestamp'].max()
#
#         # Create start event (1 second before earliest event)
#         start_event = case_events.iloc[0].copy()
#         start_event['concept:name'] = '▶'
#         start_event['time:timestamp'] = earliest_time - pd.Timedelta(seconds=1)
#
#         # Create end event (1 second after latest event)
#         end_event = case_events.iloc[-1].copy()
#         end_event['concept:name'] = '■'
#         end_event['time:timestamp'] = latest_time + pd.Timedelta(seconds=1)
#
#         new_rows.append(start_event)
#         new_rows.append(end_event)
#
#     # Add the new rows to the DataFrame
#     if new_rows:
#         modified_log = pd.concat([modified_log, pd.DataFrame(new_rows)], ignore_index=True)
#
#     # Sort by case ID and timestamp
#     modified_log = modified_log.sort_values(['case:concept:name', 'time:timestamp'])
#
#     return modified_log
#
# sub_ground_truth_log = add_start_end_symbols_to_log(sub_ground_truth_log)
#
# # dfg_truth_log_pm4py, sa, ea = pm4py.discover_dfg(sub_ground_truth_log)
#
#
# def create_dfg_from_truth(dfg_truth):
#     """
#     Create DFG structure from PM4Py's DFG output
#
#     Args:
#         dfg_truth: Dictionary containing the DFG from PM4Py (format: {(source, target): frequency})
#         sa: Start activities dictionary {activity: frequency}
#         ea: End activities dictionary {activity: frequency}
#
#     Returns:
#         DFG in JSON format
#     """
#     # Initialize node maps
#     reverse_map = {}
#     reverse_map['Start'] = 0
#     reverse_map['End'] = 1
#
#     # Map all activities to IDs
#     for (source, target), freq in dfg_truth.items():
#         # Handle special symbols for start/end
#         source = source.replace('▶', 'Start').replace('■', 'End') if isinstance(source, str) else source
#         target = target.replace('▶', 'Start').replace('■', 'End') if isinstance(target, str) else target
#
#         if source not in reverse_map:
#             reverse_map[source] = len(reverse_map)
#         if target not in reverse_map:
#             reverse_map[target] = len(reverse_map)
#
#     # Create arcs
#     arcs = []
#     node_freq = {node: 0 for node in reverse_map.keys()}
#
#     # # Add start activities if provided
#     # if sa:
#     #     for act, freq in sa.items():
#     #         # Handle special symbols for act
#     #         act = act.replace('▶', 'Start').replace('■', 'End') if isinstance(act, str) else act
#     #
#     #         if freq > 0:
#     #             arcs.append({
#     #                 'from': reverse_map['Start'],
#     #                 'to': reverse_map[act],
#     #                 'freq': freq
#     #             })
#     #             node_freq['Start'] += freq
#     #
#     # # Add end activities if provided
#     # if ea:
#     #     for act, freq in ea.items():
#     #         # Handle special symbols for act
#     #         act = act.replace('▶', 'Start').replace('■', 'End') if isinstance(act, str) else act
#     #
#     #         if freq > 0:
#     #             arcs.append({
#     #                 'from': reverse_map[act],
#     #                 'to': reverse_map['End'],
#     #                 'freq': freq
#     #             })
#     #             node_freq['End'] += freq
#
#     # Add directly-follows relations
#     for (source, target), freq in dfg_truth.items():
#         # Handle special symbols again for consistency
#         source = source.replace('▶', 'Start').replace('■', 'End') if isinstance(source, str) else source
#         target = target.replace('▶', 'Start').replace('■', 'End') if isinstance(target, str) else target
#
#         arcs.append({
#             'from': reverse_map[source],
#             'to': reverse_map[target],
#             'freq': freq
#         })
#         node_freq[target] += freq
#
#     # Create nodes
#     nodes = []
#     for node, freq in node_freq.items():
#         nodes.append({
#             'label': node,
#             'id': reverse_map[node],
#             'freq': int(freq)
#         })
#
#     return {'nodes': nodes, 'arcs': arcs}
#
# # def create_dfg_from_truth(dfg_truth):
# #     """
# #     Create DFG structure from PM4Py's DFG output
# #
# #     Args:
# #         dfg_truth: Dictionary containing the DFG from PM4Py (format: {(source, target): frequency})
# #         sa: Start activities dictionary {activity: frequency}
# #         ea: End activities dictionary {activity: frequency}
# #
# #     Returns:
# #         DFG in JSON format
# #     """
# #     # Initialize node maps
# #     reverse_map = {}
# #     reverse_map['Start'] = 0
# #     reverse_map['End'] = 1
# #
# #     # Map all activities to IDs
# #     for df_relation in predictions_df.columns:
# #         # Some columns might already be using '▶' or '■' as symbols for start/end
# #         source, end = df_relation.replace('▶', 'Start').replace('■', 'End').split('->')
# #         source = source.strip()
# #         end = end.strip()
# #     for (source, target), freq in dfg_truth.items():
# #         # Handle special symbols for start/end
# #         source = source.replace('▶', 'Start').replace('■', 'End') if isinstance(source, str) else source
# #         target = target.replace('▶', 'Start').replace('■', 'End') if isinstance(target, str) else target
# #
# #         if source not in reverse_map:
# #             reverse_map[source] = len(reverse_map)
# #         if target not in reverse_map:
# #             reverse_map[target] = len(reverse_map)
# #
# #     # Create arcs
# #     arcs = []
# #     node_freq = {node: 0 for node in reverse_map.keys()}
# #
# #     # Add start activities if provided
# #     if sa:
# #         for act, freq in sa.items():
# #             if freq > 0:
# #                 arcs.append({
# #                     'from': reverse_map['Start'],
# #                     'to': reverse_map[act],
# #                     'freq': freq
# #                 })
# #                 node_freq['Start'] += freq
# #
# #     # Add end activities if provided
# #     if ea:
# #         for act, freq in ea.items():
# #             if freq > 0:
# #                 arcs.append({
# #                     'from': reverse_map[act],
# #                     'to': reverse_map['End'],
# #                     'freq': freq
# #                 })
# #                 node_freq['End'] += freq
# #
# #     # Add directly-follows relations
# #     for (source, target), freq in dfg_truth.items():
# #         arcs.append({
# #             'from': reverse_map[source],
# #             'to': reverse_map[target],
# #             'freq': freq
# #         })
# #         node_freq[target] += freq
# #
# #     # Create nodes
# #     nodes = []
# #     for node, freq in node_freq.items():
# #         nodes.append({
# #             'label': node,
# #             'id': reverse_map[node],
# #             'freq': int(freq)
# #         })
# #
# #     return {'nodes': nodes, 'arcs': arcs}
#
# dfg_truth = create_dfg_from_truth(dfg_truth_log)
#
#
#
#
#
# #### Compare predictions with ground truth DFG
# def compare_dfgs(dfg_truth, dfg_pred):
#     # Create dictionaries for easier lookup
#     truth_arcs = {(arc['from'], arc['to']): arc['freq'] for arc in dfg_truth['arcs']}
#     pred_arcs = {(arc['from'], arc['to']): arc['freq'] for arc in dfg_pred['arcs']}
#
#     # Create lookup for node IDs to labels
#     truth_nodes = {node['id']: node['label'] for node in dfg_truth['nodes']}
#
#     # Find arcs in truth but not in predictions
#     missing_arcs = []
#     for (source, target), freq in truth_arcs.items():
#         if (source, target) not in pred_arcs:
#             missing_arcs.append({
#                 'from': truth_nodes[source],
#                 'to': truth_nodes[target],
#                 'freq': freq
#             })
#         elif pred_arcs[(source, target)] == 0 and freq > 0:
#             # Arc exists in pred but with zero frequency
#             missing_arcs.append({
#                 'from': truth_nodes[source],
#                 'to': truth_nodes[target],
#                 'freq': freq,
#                 'note': 'Zero frequency in predictions'
#             })
#
#     return missing_arcs
#
#
# # Compare the DFGs
# missing_arcs = compare_dfgs(dfg_truth, dfg_pred)
#
# # Print the missing arcs
# print(f"Found {len(missing_arcs)} arcs in truth but not in predictions:")
# for arc in missing_arcs:
#     print(f"  {arc['from']} -> {arc['to']} (freq: {arc['freq']})")
#
#
#
#
#
#
#
# #### Evaluation
# class BackgroundModel:
#
#     def __init__(self):
#         self.number_of_events = 0
#         self.number_of_traces = 0
#         self.trace_frequency = {}
#         self.labels = set()
#         self.large_string = ''
#         self.lprob = 0
#         self.trace_size = {}
#         self.log2_of_model_probability = {}
#         self.total_number_non_fitting_traces = 0
#         pass
#
#     def open_trace(self):
#         self.lprob = 0
#         self.large_string = ''
#
#     def process_event(self, event_label, probability):
#         self.large_string += event_label
#         self.number_of_events += 1
#         self.labels.add(event_label)
#         self.lprob += probability
#
#     def close_trace(self, trace_length, fitting, final_state_prob):
#         # print('Closing:', self.large_string)
#         self.trace_size[self.large_string] = trace_length
#         # print('Trace size:', trace_length)
#         self.number_of_traces += 1
#         if fitting:
#             self.log2_of_model_probability[self.large_string] = (self.lprob + final_state_prob) / math.log(2)
#         else:
#             self.total_number_non_fitting_traces += 1
#         tf = 0
#         if self.large_string in self.trace_frequency.keys():
#             tf = self.trace_frequency[self.large_string]
#         self.trace_frequency[self.large_string] = tf + 1
#
#     def h_0(self, accumulated_rho, total_number_of_traces):
#         if accumulated_rho == 0 or accumulated_rho == total_number_of_traces:
#             return 0
#         else:
#             p = (accumulated_rho / total_number_of_traces)
#             return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
#
#     def compute_relevance(self):
#         accumulated_rho = 0
#         accumulated_cost_bits = 0
#         accumulated_temp_cost_bits = 0
#         accumulated_prob_fitting_traces = 0
#
#         for trace_string, trace_freq in self.trace_frequency.items():
#             cost_bits = 0
#             nftrace_cost_bits = 0
#
#             if trace_string in self.log2_of_model_probability:
#                 cost_bits = - self.log2_of_model_probability[trace_string]
#                 accumulated_rho += trace_freq
#             else:
#                 cost_bits = (1 + self.trace_size[trace_string]) * math.log2(1 + len(self.labels))
#                 nftrace_cost_bits += trace_freq
#
#             accumulated_temp_cost_bits += nftrace_cost_bits * trace_freq
#             accumulated_cost_bits += (cost_bits * trace_freq) / self.number_of_traces
#
#             if trace_string in self.log2_of_model_probability:
#                 accumulated_prob_fitting_traces += trace_freq / self.number_of_traces
#
#         entropic_relevance = self.h_0(accumulated_rho, self.number_of_traces) + accumulated_cost_bits
#         return entropic_relevance
#
#
# def convert_dfg_into_automaton(nodes, arcs):
#     agg_outgoing_frequency = {}
#     node_info = {node['id']: node['label'] for node in nodes}
#
#     sinks = set(node_info.keys())
#     sources = list(node_info.keys())
#
#     for arc in arcs:
#         if arc['freq'] > 0:
#             arc_from = 0
#             if arc['from'] in agg_outgoing_frequency.keys():
#                 arc_from = agg_outgoing_frequency[arc['from']]
#             agg_outgoing_frequency[arc['from']] = arc_from + arc['freq']
#             sinks.discard(arc['from'])
#             # sources.discard(arc['to'])
#             if arc['to'] in sources:
#                 sources.remove(arc['to'])
#
#     # print('Outgoing frequencies:')
#     # print(agg_outgoing_frequency)
#
#     transitions = {}
#     for arc in arcs:
#         if arc['freq'] > 0:
#             if arc['to'] not in sinks:
#                 label = node_info[arc['to']]
#                 transitions[(arc['from'], label)] = (arc['to'], arc['freq'] / agg_outgoing_frequency[arc['from']])
#
#     for sink in sinks:
#         del node_info[sink]
#
#     states = set()
#     outgoing_prob = {}
#     trans_table = {}
#     for (t_from, label), (t_to, a_prob) in transitions.items():
#         trans_table[(t_from, label)] = (t_to, math.log(a_prob))
#         states.add(t_from)
#         states.add(t_to)
#         t_f = 0
#         if t_from in outgoing_prob.keys():
#             t_f = outgoing_prob[t_from]
#         outgoing_prob[t_from] = t_f + a_prob
#
#     final_states = {}
#     for state in states:
#         if not state in outgoing_prob.keys() or 1.0 - outgoing_prob[state] > 0.000006:
#             d_p = 0
#             if state in outgoing_prob.keys():
#                d_p = outgoing_prob[state]
#             final_states[state] = math.log(1 - d_p)
#
#     g = nx.DiGraph()
#     data = {}
#     for (t_from, label), (t_to, prob) in transitions.items():
#         g.add_edge(t_from, t_to, label=label+' - ' + str(round(prob,3)))
#
#     # for start, edge in transitions.items():
#     #     print(start, edge)
#
#     # for node in g.nodes:
#     #     if g.in_degree(node) == 0:
#     #         print(f'{node} has no entries')
#     #     if g.out_degree(node) == 0:
#     #         print(f'{node} has no exits')
#     #         for edge in g.in_edges(node):
#     #             print(g.get_edge_data(edge[0], edge[1]))
#
#     # dot = nx.drawing.nx_pydot.to_pydot(g)
#     # file_name = './dfgs/temp3'
#     # with open(file_name + '.dot', 'w') as file:
#     #     file.write(str(dot))
#     # check_call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png'])
#     # os.remove(file_name + '.dot')
#
#     tR = set()
#     for source in sources:
#         available = False
#         for start, end in transitions.items():
#             # print(start, end)
#             if source in start or source in end:
#                 available = True
#         if not available:
#             tR.add(source)
#     for tRe in tR:
#         sources.remove(tRe)
#
#     # return transitions, list(sources)[0], final_states, trans_table
#     return transitions, sources, final_states, trans_table
#
#
# def calculate_entropic_relevance(dfg, log_truth):
#     transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg['nodes'], dfg['arcs'])
#
#     # Check if log_truth is a DataFrame (which it is in this case) and convert it to the expected format
#     if isinstance(log_truth, pd.DataFrame):
#         # Convert DataFrame to a list of traces
#         traces = []
#         for case_id, case_events in log_truth.groupby('case:concept:name'):
#             trace = []
#             for _, event in case_events.iterrows():
#                 trace.append({'concept:name': event['concept:name']})
#             traces.append(trace)
#         log_truth = traces
#
#     # assert len(sources) == 1
#     ers = []
#     for source in sources:
#         info_gatherer = BackgroundModel()
#
#         initial_state = source
#         for t, trace in enumerate(log_truth):
#             curr = initial_state
#             non_fitting = False
#             info_gatherer.open_trace()
#             len_trace = 0
#             # print('Current state:', curr)
#             for event in trace:
#                 label = event['concept:name']
#
#                 # print(label)
#                 if label in ['Start', 'End']:
#                     continue
#                 len_trace += 1
#                 prob = 0
#                 if not non_fitting and (curr, label) in trans_table.keys():
#                     curr, prob = trans_table[(curr, label)]
#                 else:
#                     print('Not fitting at ', event['concept:name'])
#                     print('Trace:\n')
#                     string_p = ''
#                     for eve in trace:
#                         string_p += eve['concept:name'] + ' - '
#                     print(string_p)
#                     non_fitting = True
#                 info_gatherer.process_event(label, prob)
#
#             if not non_fitting and curr in final_states.keys():
#                 info_gatherer.close_trace(len_trace, True, final_states[curr])
#             else:
#                 info_gatherer.close_trace(len_trace, False, 0)
#
#         print('Non_fitting:', info_gatherer.total_number_non_fitting_traces)
#         print(info_gatherer.number_of_traces)
#
#         entropic_relevance = info_gatherer.compute_relevance()
#         ers.append(entropic_relevance)
#
#     entropic_relevance = min(ers)
#     # print('Entropic relevance:', entropic_relevance)
#     return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces
#
# truth_er, truth_no_non_fitting_traces, truth_no_traces = calculate_entropic_relevance(dfg_truth, sub_ground_truth_log)
#
# pred_er, pred_no_non_fitting_traces, pred_no_traces = calculate_entropic_relevance(dfg_pred, sub_ground_truth_log)
#
#
#
#
# #### New DFGs construction to include '▶'and '■'
#
# def create_dfg_from_truth_new(dfg_truth):
#     """
#     Create DFG structure from PM4Py's DFG output
#
#     Args:
#         dfg_truth: Dictionary containing the DFG from PM4Py (format: {(source, target): frequency})
#
#     Returns:
#         DFG in JSON format
#     """
#     # Initialize node maps with symbols
#     reverse_map = {}
#     reverse_map['▶'] = 0  # Start symbol
#     reverse_map['■'] = 1  # End symbol
#
#     # Map all activities to IDs
#     for (source, target), freq in dfg_truth.items():
#         if source not in reverse_map:
#             reverse_map[source] = len(reverse_map)
#         if target not in reverse_map:
#             reverse_map[target] = len(reverse_map)
#
#     # Create arcs
#     arcs = []
#     node_freq = {node: 0 for node in reverse_map.keys()}
#
#     # Add directly-follows relations
#     for (source, target), freq in dfg_truth.items():
#         arcs.append({
#             'from': reverse_map[source],
#             'to': reverse_map[target],
#             'freq': freq
#         })
#         node_freq[target] += freq
#
#     # Create nodes
#     nodes = []
#     for node, freq in node_freq.items():
#         nodes.append({
#             'label': node,
#             'id': reverse_map[node],
#             'freq': int(freq)
#         })
#
#     return {'nodes': nodes, 'arcs': arcs}
#
#
# def create_dfg_from_predictions_new(predictions_df):
#     """
#     Create DFG structure from predictions dataframe
#
#     Args:
#         predictions_df: DataFrame with predictions
#
#     Returns:
#         DFG in JSON format
#     """
#     # Extract directly-follows relations
#     node_map = {}
#     reverse_map = {}
#     reverse_map['▶'] = 0  # Start symbol
#     reverse_map['■'] = 1  # End symbol
#
#     # Process each column in predictions_df
#     # Each column represents a directly-follows relation
#     for df_relation in predictions_df.columns:
#         # Extract source and target, handling any format
#         if '->' in df_relation:
#             parts = df_relation.split('->')
#             source = parts[0].strip().replace('Start', '▶').replace('End', '■')
#             end = parts[1].strip().replace('Start', '▶').replace('End', '■')
#
#             if source not in reverse_map:
#                 reverse_map[source] = len(reverse_map)
#             if end not in reverse_map:
#                 reverse_map[end] = len(reverse_map)
#
#     # Create arcs
#     arcs = []
#     node_freq = {node: 0 for node in reverse_map.keys()}
#
#     # Process each directly-follows relation
#     for df_relation in predictions_df.columns:
#         if '->' in df_relation:
#             parts = df_relation.split('->')
#             source = parts[0].strip().replace('Start', '▶').replace('End', '■')
#             end = parts[1].strip().replace('Start', '▶').replace('End',
#             '■')
#             # For each time step in the horizon, extract the prediction value
#             for _, row in predictions_df.iterrows():
#                 freq = round(float(row[df_relation]))
#                 if freq <= 0:
#                     continue
#
#                 arcs.append({
#                     'from': reverse_map[source],
#                     'to': reverse_map[end],
#                     'freq': freq
#                 })
#
#                 if source == '▶':
#                     node_freq[source] += freq
#                 else:
#                     node_freq[end] += freq
#
#     # Create nodes
#     nodes = []
#     for node, freq in node_freq.items():
#         nodes.append({'label': node, 'id': reverse_map[node], 'freq': round(freq)})
#
#     return {'nodes': nodes, 'arcs': arcs}
#
#
# dfg_pred_json = create_dfg_
# from_predictions_new(agg_pred_first)
# dfg_truth_new = create_dfg_from_truth_new(dfg_truth_log)
#
#
# ####
# import json
#
# # Save prediction DFG to a JSON file
# with open(f'{dataset}_{horizon}_{model_group}_{model_name}_pred_dfg.json', 'w') as f:
#     json.dump(dfg_pred, f, indent=2)
#
# # Save ground truth DFG to a JSON file
# with open(f'{dataset}_{horizon}_truth_dfg.json', 'w') as f:
#     json.dump(dfg_truth, f, indent=2)
#
# # Save prediction DFG with special symbols to a JSON file
# with open(f'{dataset}_{horizon}_{model_group}_{model_name}_pred_dfg_new.json', 'w') as f:
#     json.dump(dfg_pred_json, f, indent=2)
#
# # Save ground truth DFG with special symbols to a JSON file
# with open(f'{dataset}_{horizon}_truth_dfg_new.json', 'w') as f:
#     json.dump(dfg_truth_new, f, indent=2)
#
# print(f"DFG dictionaries saved to JSON files in the current directory.")
#
#
#
# pm4py.write_xes(sub_ground_truth_log, f'{dataset}_{horizon}_sub_ground_truth.xes')