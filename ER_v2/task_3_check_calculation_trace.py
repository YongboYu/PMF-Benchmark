import pm4py
import pandas as pd
import numpy as np
import os
import json
import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Union, Set
from ER_v2.utils_ER_v2 import DFGConstructor, ERCalculator
import datetime

# Modified BackgroundModel class with detailed tracing
class BackgroundModelWithTrace:
    def __init__(self, trace_file_path, dfg_type, window_key, source_idx):
        self.number_of_events = 0
        self.number_of_traces = 0
        self.trace_frequency = {}
        self.labels = set()
        self.large_string = ''
        self.lprob = 0
        self.trace_size = {}
        self.log2_of_model_probability = {}
        self.total_number_non_fitting_traces = 0
        
        # Tracing variables
        self.trace_file = open(trace_file_path, 'w')
        self.current_trace_events = []
        self.current_trace_number = 0
        self.dfg_type = dfg_type
        self.window_key = window_key
        self.source_idx = source_idx
        
        # Write header with meaningful information
        self.trace_file.write("=== ER CALCULATION TRACE ===\n")
        self.trace_file.write(f"DFG Type: {dfg_type}\n")
        self.trace_file.write(f"Window: {window_key}\n")
        self.trace_file.write(f"Source Index: {source_idx}\n")
        self.trace_file.write(f"Analysis started at: {datetime.datetime.now()}\n\n")

    def open_trace(self):
        self.lprob = 0
        self.large_string = ''
        self.current_trace_events = []
        self.current_trace_number += 1
        
        self.trace_file.write(f"\n--- TRACE {self.current_trace_number} ---\n")
        self.trace_file.write("Opening new trace\n")
        self.trace_file.write(f"Initial lprob: {self.lprob}\n")
        self.trace_file.write(f"Initial large_string: '{self.large_string}'\n")

    def process_event(self, event_label, probability):
        self.trace_file.write(f"\nProcessing event: '{event_label}'\n")
        self.trace_file.write(f"  Event probability: {probability}\n")
        self.trace_file.write(f"  Before - lprob: {self.lprob}, large_string: '{self.large_string}'\n")
        
        self.large_string += event_label
        self.number_of_events += 1
        self.labels.add(event_label)
        self.lprob += probability
        
        self.trace_file.write(f"  After - lprob: {self.lprob}, large_string: '{self.large_string}'\n")
        self.trace_file.write(f"  Total events processed: {self.number_of_events}\n")
        
        self.current_trace_events.append({
            'event_label': event_label,
            'probability': probability,
            'cumulative_lprob': self.lprob
        })

    def close_trace(self, trace_length, fitting, final_state_prob):
        self.trace_file.write(f"\nClosing trace\n")
        self.trace_file.write(f"  Trace length: {trace_length}\n")
        self.trace_file.write(f"  Is fitting: {fitting}\n")
        self.trace_file.write(f"  Final state probability: {final_state_prob}\n")
        self.trace_file.write(f"  Final lprob: {self.lprob}\n")
        self.trace_file.write(f"  Final large_string: '{self.large_string}'\n")
        
        self.trace_size[self.large_string] = trace_length
        self.number_of_traces += 1
        
        if fitting:
            # log2_prob = (self.lprob + final_state_prob) / math.log(2)
            log2_prob = self.lprob
            self.log2_of_model_probability[self.large_string] = log2_prob
            self.trace_file.write(f"  FITTING TRACE - log2_probability: {log2_prob}\n")
        else:
            self.total_number_non_fitting_traces += 1
            self.trace_file.write(f"  NON-FITTING TRACE\n")
            
        # Update trace frequency
        tf = 0
        if self.large_string in self.trace_frequency.keys():
            tf = self.trace_frequency[self.large_string]
        self.trace_frequency[self.large_string] = tf + 1
        
        self.trace_file.write(f"  Trace frequency: {self.trace_frequency[self.large_string]}\n")
        self.trace_file.write(f"  Total traces processed: {self.number_of_traces}\n")
        self.trace_file.write(f"  Total non-fitting traces: {self.total_number_non_fitting_traces}\n")

    def h_0(self, accumulated_rho, total_number_of_traces):
        if accumulated_rho == 0 or accumulated_rho == total_number_of_traces:
            return 0
        else:
            p = (accumulated_rho / total_number_of_traces)
            return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def compute_relevance(self):
        self.trace_file.write(f"\n=== COMPUTING ENTROPIC RELEVANCE ===\n")
        
        accumulated_rho = 0
        accumulated_cost_bits = 0
        accumulated_temp_cost_bits = 0
        accumulated_prob_fitting_traces = 0

        self.trace_file.write(f"Total unique traces: {len(self.trace_frequency)}\n")
        self.trace_file.write(f"Total labels: {len(self.labels)}\n\n")
        
        for trace_string, trace_freq in self.trace_frequency.items():
            self.trace_file.write(f"\nProcessing trace pattern: '{trace_string}' (frequency: {trace_freq})\n")
            
            cost_bits = 0
            nftrace_cost_bits = 0

            if trace_string in self.log2_of_model_probability:
                cost_bits = - self.log2_of_model_probability[trace_string]
                accumulated_rho += trace_freq
                self.trace_file.write(f"  FITTING - log2_prob: {self.log2_of_model_probability[trace_string]}\n")
                self.trace_file.write(f"  Cost bits: {cost_bits}\n")
                self.trace_file.write(f"  Accumulated rho: {accumulated_rho}\n")
            else:
                cost_bits = (1 + self.trace_size[trace_string]) * math.log2(1 + len(self.labels))
                nftrace_cost_bits += trace_freq
                self.trace_file.write(f"  NON-FITTING - trace size: {self.trace_size[trace_string]}\n")
                self.trace_file.write(f"  Cost bits: {cost_bits}\n")

            accumulated_temp_cost_bits += nftrace_cost_bits * trace_freq
            accumulated_cost_bits += (cost_bits * trace_freq) / self.number_of_traces

            if trace_string in self.log2_of_model_probability:
                accumulated_prob_fitting_traces += trace_freq / self.number_of_traces
                
            self.trace_file.write(f"  Accumulated cost bits: {accumulated_cost_bits}\n")

        h0_value = self.h_0(accumulated_rho, self.number_of_traces)
        entropic_relevance = h0_value + accumulated_cost_bits
        
        self.trace_file.write(f"\n=== FINAL CALCULATION ===\n")
        self.trace_file.write(f"Accumulated rho: {accumulated_rho}\n")
        self.trace_file.write(f"Total traces: {self.number_of_traces}\n")
        self.trace_file.write(f"H0 value: {h0_value}\n")
        self.trace_file.write(f"Accumulated cost bits: {accumulated_cost_bits}\n")
        self.trace_file.write(f"Entropic Relevance: {entropic_relevance}\n")
        
        return entropic_relevance
    
    def close_file(self):
        self.trace_file.write(f"\n=== TRACE COMPLETED ===\n")
        self.trace_file.write(f"Analysis ended at: {datetime.datetime.now()}\n")
        self.trace_file.close()

# Function to save transition tables
def save_transition_table(transitions, sources, final_states, trans_table, output_path, dfg_type, window_key, node_info=None):
    """
    Save the transition table to JSON and CSV formats for analysis
    
    Parameters:
    - transitions: Raw transitions with probabilities
    - sources: Source states
    - final_states: Final states with probabilities
    - trans_table: Transition table with log probabilities
    - output_path: Base path for output files
    - dfg_type: Type of DFG (truth, training, prediction)
    - window_key: Window identifier
    - node_info: Mapping of node IDs to labels (optional)
    """
    
    # Prepare transition table data
    transition_data = {
        'window': window_key,
        'dfg_type': dfg_type,
        'sources': list(sources),
        'final_states': {str(state): prob for state, prob in final_states.items()},
        'transitions': [],
        'transition_table': []
    }
    
    # Convert transitions to serializable format
    for (from_state, label), (to_state, probability) in transitions.items():
        transition_data['transitions'].append({
            'from_state': from_state,
            'label': label,
            'to_state': to_state,
            'probability': probability,
            'log_probability': math.log2(probability)
        })
    
    # Convert trans_table to serializable format
    for (from_state, label), (to_state, log_prob) in trans_table.items():
        transition_data['transition_table'].append({
            'from_state': from_state,
            'label': label,
            'to_state': to_state,
            'log_probability': log_prob,
            'probability': math.exp(log_prob)
        })
    
    # Save as JSON
    json_file = f"{output_path}_{dfg_type}_transitions.json"
    with open(json_file, 'w') as f:
        json.dump(transition_data, f, indent=2)
    
    # Save as CSV for easy viewing
    csv_file = f"{output_path}_{dfg_type}_transitions.csv"
    
    # Create a flat DataFrame for CSV
    csv_data = []
    for item in transition_data['transition_table']:
        csv_data.append({
            'window': window_key,
            'dfg_type': dfg_type,
            'from_state': item['from_state'],
            'label': item['label'],
            'to_state': item['to_state'],
            'probability': item['probability'],
            'log_probability': item['log_probability']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    # Save final states as separate CSV
    final_states_file = f"{output_path}_{dfg_type}_final_states.csv"
    final_states_df = pd.DataFrame([
        {'window': window_key, 'dfg_type': dfg_type, 'state': state, 'log_probability': prob}
        for state, prob in final_states.items()
    ])
    final_states_df.to_csv(final_states_file, index=False)
    
    print(f"    Saved transition table: {json_file}, {csv_file}, {final_states_file}")

# Modified ERCalculator to use the tracing BackgroundModel
class ERCalculatorWithTrace:
    def convert_dfg_into_automaton(self, nodes: List[Dict], arcs: List[Dict]) -> Tuple[Dict, List, Dict, Dict]:
        """
        Convert DFG to automaton (same as original)
        """
        agg_outgoing_frequency = {}
        node_info = {node['id']: node['label'] for node in nodes}

        sinks = set(node_info.keys())
        sources = list(node_info.keys())

        # First pass: calculate outgoing frequencies and identify sinks/sources
        agg_outgoing_frequency = {}
        arc_logs = []
        arc_logs.append("=== FIRST PASS: CALCULATING OUTGOING FREQUENCIES ===")

        for arc in arcs:
            if arc['freq'] > 0:
                from_node = arc['from']
                to_node = arc['to']
                arc_log = f"Processing arc: {from_node}({node_info.get(from_node, 'unknown')}) -> {to_node}({node_info.get(to_node, 'unknown')}) [freq={arc['freq']}]"

                # Only count outgoing frequency if not going to end symbol
                if node_info[to_node] != '■':
                    before_freq = agg_outgoing_frequency.get(from_node, 0)
                    if from_node in agg_outgoing_frequency.keys():
                        agg_outgoing_frequency[from_node] += arc['freq']
                    else:
                        agg_outgoing_frequency[from_node] = arc['freq']
                    arc_log += f"\n  Updated outgoing frequency for {from_node}: {before_freq} -> {agg_outgoing_frequency[from_node]}"
                else:
                    arc_log += f"\n  Skipped outgoing frequency (end symbol)"

                sinks.discard(from_node)
                if to_node in sources:
                    sources.remove(to_node)

                arc_logs.append(arc_log)

        arc_logs.append(f"Final agg_outgoing_frequency: {agg_outgoing_frequency}")
        arc_logs.append(f"Final sinks: {sinks}")
        arc_logs.append(f"Final sources: {sources}\n")

        # Second pass: create transitions with adjusted probabilities
        arc_logs.append("=== SECOND PASS: CREATING TRANSITIONS ===")
        transitions = {}

        for arc in arcs:
            if arc['freq'] > 0:
                from_node = arc['from']
                to_node = arc['to']
                label = node_info[to_node]
                arc_log = f"Processing arc: {from_node}({node_info.get(from_node, 'unknown')}) -> {to_node}({node_info.get(to_node, 'unknown')}) [freq={arc['freq']}]"

                # Special handling for transitions to end symbol or from start symbol
                if node_info[from_node] == '▶' or label == '■':
                    # Always use probability 1.0 for start/end related transitions
                    transitions[(from_node, label)] = (to_node, 1.0)
                    arc_log += f"\n  Start/end symbol transition: ({from_node}, {label}) -> ({to_node}, 1.0)"
                # For normal transitions (not going to end)
                elif to_node not in sinks:
                    # Normal probability calculation
                    prob = arc['freq'] / agg_outgoing_frequency[from_node]
                    transitions[(from_node, label)] = (to_node, prob)
                    arc_log += f"\n  Regular transition: ({from_node}, {label}) -> ({to_node}, {prob:.4f})"
                    arc_log += f"\n  Calculation: {arc['freq']} / {agg_outgoing_frequency[from_node]} = {prob:.4f}"
                else:
                    arc_log += f"\n  Skipped non-end transition to sink: {to_node}"

                arc_logs.append(arc_log)

        arc_logs.append(f"Final transitions: {transitions}")

        # Write logs to file
        log_file_path = "dfg_transition_calculation_log.txt"
        with open(log_file_path, 'w') as f:
            for log_line in arc_logs:
                f.write(log_line + "\n")

        print(f"Transition calculation logs saved to {log_file_path}")

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

    def calculate_entropic_relevance_with_trace(self, dfg: Dict, log_truth: Union[pd.DataFrame, List], 
                                               trace_file_path: str, dfg_type: str, window_key: str) -> Tuple[float, int, int, Dict, Dict, Dict]:
        """
        Calculate entropic relevance with detailed tracing
        """
        transitions, sources, final_states, trans_table = self.convert_dfg_into_automaton(
            dfg['nodes'], dfg['arcs']
        )

        # Save transition table
        save_transition_table(transitions, sources, final_states, trans_table, 
                             trace_file_path.rsplit('_source_', 1)[0], dfg_type, window_key)

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

        for source_idx, source in enumerate(sources):
            source_trace_file = f"{trace_file_path}_{dfg_type}_source_{source_idx}.txt"
            info_gatherer = BackgroundModelWithTrace(source_trace_file, dfg_type, window_key, source_idx)
            
            info_gatherer.trace_file.write(f"=== DFG ANALYSIS DETAILS ===\n")
            info_gatherer.trace_file.write(f"Initial source state: {source}\n")
            info_gatherer.trace_file.write(f"Available transitions: {len(transitions)}\n")
            info_gatherer.trace_file.write(f"Total transitions: {list(transitions.keys())}\n")
            info_gatherer.trace_file.write(f"Final states: {final_states}\n")
            info_gatherer.trace_file.write(f"DFG nodes: {[node['label'] for node in dfg['nodes']]}\n")
            info_gatherer.trace_file.write(f"DFG arcs: {[(arc['from'], arc['to'], arc['freq']) for arc in dfg['arcs']]}\n\n")

            initial_state = source
            for t, trace in enumerate(log_truth):
                curr = initial_state
                non_fitting = False
                info_gatherer.open_trace()
                len_trace = 0

                trace_pattern = "_".join([event['concept:name'] for event in trace])
                info_gatherer.trace_file.write(f"Processing log trace {t + 1}: {trace_pattern}\n")
                
                for event_idx, event in enumerate(trace):
                    label = event['concept:name']
                    info_gatherer.trace_file.write(f"\n  Event {event_idx + 1}: '{label}'\n")

                    if label in ['▶', '■']:
                        info_gatherer.trace_file.write("  Skipping start/end symbol\n")
                        continue
                        
                    len_trace += 1
                    prob = 0
                    
                    info_gatherer.trace_file.write(f"  Current state: {curr}\n")
                    info_gatherer.trace_file.write(f"  Looking for transition: ({curr}, {label})\n")
                    
                    if not non_fitting and (curr, label) in trans_table.keys():
                        curr, prob = trans_table[(curr, label)]
                        info_gatherer.trace_file.write(f"  TRANSITION FOUND - Next state: {curr}, Log prob: {prob}\n")
                    else:
                        info_gatherer.trace_file.write(f"  TRANSITION NOT FOUND - Mark as non-fitting\n")
                        if (curr, label) not in trans_table.keys():
                            info_gatherer.trace_file.write(f"  Available transitions from state {curr}: {[key for key in trans_table.keys() if key[0] == curr]}\n")
                        non_fitting = True
                        
                    info_gatherer.process_event(label, prob)

                info_gatherer.trace_file.write(f"\nFinal state: {curr}\n")
                info_gatherer.trace_file.write(f"Checking for final state probability...\n")
                
                if not non_fitting and curr in final_states.keys():
                    final_prob = final_states[curr]
                    info_gatherer.trace_file.write(f"Final state probability: {final_prob}\n")
                    info_gatherer.close_trace(len_trace, True, final_prob)
                    fitting_traces[trace_pattern] = fitting_traces.get(trace_pattern, 0) + 1
                else:
                    info_gatherer.trace_file.write(f"No final state probability found - marking as non-fitting\n")
                    info_gatherer.close_trace(len_trace, False, 0)
                    non_fitting_traces[trace_pattern] = non_fitting_traces.get(trace_pattern, 0) + 1

            entropic_relevance = info_gatherer.compute_relevance()
            info_gatherer.close_file()
            ers.append(entropic_relevance)

        entropic_relevance = min(ers)
        return entropic_relevance, info_gatherer.total_number_non_fitting_traces, info_gatherer.number_of_traces, fitting_traces, non_fitting_traces, transitions

def run_er_trace_analysis():
    """
    Run complete ER trace analysis for truth, training, and prediction DFGs
    Following the exact workflow from utils_ER_v1.py
    """
    # Create output directory
    output_dir = "er_trace_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configuration - adjust these as needed
    dataset = "BPI2017"  # Change this to your dataset
    horizon = 7  # Change this to your horizon
    model_group = "regression"  # Change this to your model group
    model_name = "random_forest"  # Change this to your model name
    start_time = "2016-10-22 00:00:00"  # Change this to appropriate start time
    
    print(f"Running ER trace analysis for {dataset}")
    
    # 1. Initialize the classes
    dfg_constructor = DFGConstructor()
    er_calculator = ERCalculatorWithTrace()
    
    # 2. Load the event log
    log_file = f'data/interim/processed_logs/{dataset}.xes'
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found. Creating a simple example log for testing.")
        # Create a simple example log for testing
        data = []
        case_events = [
            ('case1', 'A', '2012-01-01 10:00:00'),
            ('case1', 'B', '2012-01-01 11:00:00'),
            ('case1', 'C', '2012-01-01 12:00:00'),
            ('case2', 'A', '2012-01-01 13:00:00'),
            ('case2', 'D', '2012-01-01 14:00:00'),
            ('case2', 'C', '2012-01-01 15:00:00'),
            ('case3', 'A', '2012-01-01 16:00:00'),
            ('case3', 'B', '2012-01-01 17:00:00'),
            ('case3', 'D', '2012-01-01 18:00:00'),
            ('case3', 'C', '2012-01-01 19:00:00'),
        ]
        
        for case_id, activity, timestamp in case_events:
            data.append({
                'case:concept:name': case_id,
                'concept:name': activity,
                'time:timestamp': pd.to_datetime(timestamp)
            })
        
        log_df = pd.DataFrame(data)
        log = pm4py.convert_to_event_log(log_df)
    else:
        log = pm4py.read_xes(log_file)
    
    # 3. Extract rolling window sublogs
    print("Extracting rolling window sublogs...")
    seq_test_log = dfg_constructor.extract_rolling_window_sublogs(
        log, 'case:concept:name', 'concept:name', 'time:timestamp',
        start_time, horizon
    )
    
    print(f"Found {len(seq_test_log)} rolling windows")
    
    # 4. Create ground truth DFGs from the sublogs
    print("Creating ground truth DFGs...")
    rolling_truth_dfgs = dfg_constructor.create_dfgs_from_rolling_window(seq_test_log)
    
    # 5. Create training baseline DFGs (using 80% of data by time)
    print("Creating training baseline DFGs...")
    rolling_training_dfgs = dfg_constructor.create_dfgs_from_rolling_training(
        seq_test_log,
        log,
        'case:concept:name',
        'concept:name',
        'time:timestamp',
        time_length=int(horizon)
    )
    
    # 6. Load and process predictions
    print(f"Loading predictions for {model_name}...")
    prediction_file = f'results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet'
    
    if os.path.exists(prediction_file):
        predictions_df = pd.read_parquet(prediction_file)
        agg_pred = predictions_df.groupby('sequence_start_time').sum().rename_axis('timestamp')
        agg_pred_round = agg_pred.round(0).astype(int)
    else:
        print(f"Prediction file not found: {prediction_file}")
        print("Creating synthetic predictions for testing...")
        # Create synthetic predictions based on the actual windows
        pred_data = []
        for window_key in seq_test_log.keys():
            start_date = window_key.split('_')[0]
            # Get actual activities from this window
            sublog = seq_test_log[window_key]
            activities = sublog['concept:name'].unique()
            activities = [act for act in activities if act not in ['▶', '■']]
            
            pred_row = {}
            # Create predictions for consecutive activity pairs
            for i in range(len(activities)):
                for j in range(len(activities)):
                    if i != j:  # Avoid self-loops
                        pred_row[f"{activities[i]}->{activities[j]}"] = 1.0
            
            pred_data.append(pred_row)
        
        agg_pred_round = pd.DataFrame(pred_data, index=[window.split('_')[0] for window in seq_test_log.keys()])
        agg_pred_round.index.name = 'timestamp'
    
    # 7. Create prediction DFGs
    print("Creating prediction DFGs...")
    rolling_pred_dfgs = dfg_constructor.create_dfgs_from_rolling_predictions(seq_test_log, agg_pred_round)
    
    # 8. Run ER trace analysis for each window and each DFG type
    print("\nRunning ER trace analysis for all windows...")
    
    for window_idx, (window_key, sublog) in enumerate(seq_test_log.items()):
        print(f"\nAnalyzing window {window_idx + 1}/{len(seq_test_log)}: {window_key}")
        print(f"  Sublog contains {len(sublog)} events from {sublog['case:concept:name'].nunique()} cases")
        
        # Clean window key for filename
        clean_window_key = window_key.replace('-', '_').replace(' ', '_').replace(':', '_')
        base_filename = f"{output_dir}/er_trace_{dataset}_window_{window_idx + 1}_{clean_window_key}"
        
        # Analyze truth DFG
        if window_key in rolling_truth_dfgs:
            print("  Tracing truth DFG...")
            truth_dfg = rolling_truth_dfgs[window_key]['dfg_json']
            print(f"    Truth DFG has {len(truth_dfg['nodes'])} nodes and {len(truth_dfg['arcs'])} arcs")
            er_calculator.calculate_entropic_relevance_with_trace(
                truth_dfg, sublog, base_filename, "truth", window_key
            )
        
        # Analyze training DFG
        if window_key in rolling_training_dfgs:
            print("  Tracing training DFG...")
            training_dfg = rolling_training_dfgs[window_key]['dfg_json']
            print(f"    Training DFG has {len(training_dfg['nodes'])} nodes and {len(training_dfg['arcs'])} arcs")
            er_calculator.calculate_entropic_relevance_with_trace(
                training_dfg, sublog, base_filename, "training", window_key
            )
        
        # Analyze prediction DFG
        if window_key in rolling_pred_dfgs:
            print("  Tracing prediction DFG...")
            pred_dfg = rolling_pred_dfgs[window_key]['dfg_json']
            print(f"    Prediction DFG has {len(pred_dfg['nodes'])} nodes and {len(pred_dfg['arcs'])} arcs")
            er_calculator.calculate_entropic_relevance_with_trace(
                pred_dfg, sublog, base_filename, "prediction", window_key
            )
    
    print(f"\nER trace analysis completed!")
    print(f"Check files in '{output_dir}' directory.")
    print("Each file contains detailed step-by-step ER calculation process for one DFG type and source state.")
    print("Transition tables are saved as JSON and CSV files for each DFG type and window.")

if __name__ == "__main__":
    run_er_trace_analysis()
