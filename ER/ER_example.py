import pm4py
import pandas as pd
import numpy as np
import os
import json
import math
import networkx as nx

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


    df = df.sort_values(by=[case_id_col, timestamp_col])
    cases_in_period = df[(df[timestamp_col] >= start_time) &
                         (df[timestamp_col] <= end_time)][case_id_col].unique()
    case_df = df[df[case_id_col].isin(cases_in_period)].copy()

    result_events = []

    for case_id in cases_in_period:
        case_events = case_df[case_df[case_id_col] == case_id].copy()
        case_events['next_timestamp'] = case_events[timestamp_col].shift(-1)
        case_events['is_within_period'] = (case_events[timestamp_col] >= start_time) & (
                    case_events[timestamp_col] <= end_time)
        case_events['next_within_period'] = (case_events['next_timestamp'] >= start_time) & (
                    case_events['next_timestamp'] <= end_time)


        relevant_events = case_events[case_events['is_within_period'] | case_events['next_within_period']]
        result_events.append(relevant_events)

    if not result_events:
        return pd.DataFrame(columns=df.columns)

    result_df = pd.concat(result_events)
    result_df[activity_col] = result_df[activity_col].replace({'▶': 'Start', '■': 'End'})

    return result_df[df.columns].sort_values(by=[case_id_col, timestamp_col])

start_time = '2016-10-22 00:00:00'
end_time = '2016-10-29 00:00:00'

sub_ground_truth_log = extract_time_period_sublog(log, 'case:concept:name', 'concept:name', 'time:timestamp', start_time, end_time)

sub_ground_truth_log_start_end = pm4py.insert_artificial_start_end(sub_ground_truth_log)

dfg_truth_log, sa, ea = pm4py.discover_dfg(sub_ground_truth_log_start_end)

pm4py.save_vis_dfg(dfg_truth_log, sa, ea, 'dfg_truth_log.png')



def create_dfg_from_truth(dfg_truth):

    reverse_map = {}
    reverse_map['▶'] = 0
    reverse_map['■'] = 1


    for (source, target), freq in dfg_truth.items():

        # source = source.replace('▶', 'Start').replace(, 'End') if isinstance(source, str) else source
        # target = target.replace('▶', 'Start').replace('■', 'End') if isinstance(target, str) else target

        if source not in reverse_map:
            reverse_map[source] = len(reverse_map)
        if target not in reverse_map:
            reverse_map[target] = len(reverse_map)

    # Create arcs
    arcs = []
    node_freq = {node: 0 for node in reverse_map.keys()}


    for (source, target), freq in dfg_truth.items():

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

    reverse_map = {}
    reverse_map['▶'] = 0
    reverse_map['■'] = 1
    reverse_map['Start'] = 2
    reverse_map['End'] = 3

    incoming_freq = {}
    outgoing_freq = {}

    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]

            source = source.replace('▶', 'Start').replace('■', 'End')
            target = target.replace('▶', 'Start').replace('■', 'End')

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

            source = source.replace('▶', 'Start').replace('■', 'End')
            target = target.replace('▶', 'Start').replace('■', 'End')

            total_freq = 0
            for _, row in predictions_df.iterrows():
                freq = round(float(row[df_relation]))
                if freq > 0:
                    total_freq += freq

            outgoing_freq[source] = outgoing_freq.get(source, 0) + total_freq
            incoming_freq[target] = incoming_freq.get(target, 0) + total_freq
            # if source not in ['Start', 'End']:
            #     outgoing_freq[source] = outgoing_freq.get(source, 0) + total_freq
            #
            # if target not in ['Start', 'End']:
            #     incoming_freq[target] = incoming_freq.get(target, 0) + total_freq

    # Create arcs
    arcs = []
    for df_relation in predictions_df.columns:
        if '->' in df_relation:
            source, target = [part.strip() for part in df_relation.split('->')]

            clean_source = source.replace('▶', 'Start').replace('■', 'End')
            clean_target = target.replace('▶', 'Start').replace('■', 'End')

            source_id = reverse_map.get(clean_source)
            target_id = reverse_map.get(clean_target)

            if source_id is not None and target_id is not None:
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

    node_freq = {node: 0 for node in reverse_map.keys()}

    for arc in arcs:
        source_id = arc['from']
        target_id = arc['to']

        source_label = None
        target_label = None
        for node, node_id in reverse_map.items():
            if node_id == source_id:
                source_label = node
            if node_id == target_id:
                target_label = node

        if source_label == '▶':
            node_freq['▶'] += arc['freq']

        if target_label:
            node_freq[target_label] += arc['freq']

    # Create nodes
    nodes = []
    for node, node_id in reverse_map.items():
        nodes.append({
            'label': node,
            'id': node_id,
            'freq': round(node_freq.get(node, 0))
        })

    return {'nodes': nodes, 'arcs': arcs}

dfg_pred_json = create_dfg_from_predictions(agg_pred_first)



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


def calculate_entropic_relevance_corr(dfg, log_truth):
    transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg['nodes'], dfg['arcs'])

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


truth_er, truth_no_non_fitting_traces, truth_no_traces, truth_fitting_traces, truth_non_fitting_traces = calculate_entropic_relevance_corr(dfg_truth_json, sub_ground_truth_log_start_end)
pred_er, pred_no_non_fitting_traces, pred_no_traces, pred_fitting_traces, pred_non_fitting_traces = calculate_entropic_relevance_corr(dfg_pred_json, sub_ground_truth_log_start_end)


