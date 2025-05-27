import json
import math
import networkx as nx
import os
from subprocess import check_call
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery


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


def convert_dfg_into_automaton(nodes, arcs, json_file):
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


def calculate_entropic_relevance(json_file, log, method):

    if method != 'bla':
        file = open(json_file)
        dfg = json.load(file)
    else:
        dfg = json_file

    if method == 'Actuals':
        dfg_graph = dfg_discovery.apply(log)

        node_map = {}
        reverse_map = {}
        reverse_map['Start'] = 0
        reverse_map['End'] = 1
        for (p1, p2), freq in dfg_graph.items():
            source, end = p1, p2
            if source not in reverse_map.keys():
                reverse_map[source] = len(reverse_map)
            if end not in reverse_map.keys():
                reverse_map[end] = len(reverse_map)

        # print(reverse_map)

        arcs = []
        node_freq = {node: 0 for node in reverse_map.keys()}
        for (p1, p2), freq in dfg_graph.items():
            source, end = p1, p2
            arcs.append({'from': reverse_map[source], 'to': reverse_map[end], 'freq': freq})
            if source == 'Start':
                node_freq[source] += freq
            else:
                node_freq[end] += freq

        nodes = []
        for node, freq in node_freq.items():
            nodes.append({'label': node, 'id': reverse_map[node], 'freq': int(freq)})

        dfg = {'nodes': nodes, 'arcs': arcs}

    transitions, sources, final_states, trans_table = convert_dfg_into_automaton(dfg['nodes'], dfg['arcs'], json_file)

    # assert len(sources) == 1
    ers = []
    for source in sources:
        info_gatherer = BackgroundModel()

        initial_state = source
        for t, trace in enumerate(log):
            curr = initial_state
            non_fitting = False
            info_gatherer.open_trace()
            len_trace = 0
            # print('Current state:', curr)
            for event in trace:
                label = event['concept:name']

                # print(label)
                if label in ['Start', 'End']:
                    continue
                len_trace += 1
                prob = 0
                if not non_fitting and (curr, label) in trans_table.keys():
                    curr, prob = trans_table[(curr, label)]
                else:
                    # print('Not fitting at ', event['concept:name'])
                    # print('Trace:\n')
                    # string_p = ''
                    # for eve in trace:
                    #     string_p += eve['concept:name'] + ' - '
                    # print(string_p)
                    non_fitting = True
                info_gatherer.process_event(label, prob)

            if not non_fitting and curr in final_states.keys():
                info_gatherer.close_trace(len_trace, True, final_states[curr])
            else:
                info_gatherer.close_trace(len_trace, False, 0)

        print('Non_fitting:', info_gatherer.total_number_non_fitting_traces)
        print(info_gatherer.number_of_traces)

        entropic_relevance = info_gatherer.compute_relevance()
        ers.append(entropic_relevance)

    entropic_relevance = min(ers)
    # print('Entropic relevance:', entropic_relevance)
    return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces