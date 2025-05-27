import pandas as pd
import pm4py
from pathlib import Path


# dataset_list = ['BPI2017', 'BPI2019_1', 'Hospital_Billing', 'Sepsis', 'RTFMP']
dataset = 'BPI2017'
raw_path = Path('data/raw')
processed_path = Path('data/interim/processed_logs')

raw_log = pm4py.read_xes(f'data/raw/{dataset}.xes')
processed_log = pm4py.read_xes(f'data/interim/processed_logs/{dataset}.xes')


# def get_log_stats(datasets):
#     log_stats = {}
#
#     for dataset in datasets:
#         log_stats[dataset] = {
#             'raw': {},
#             'processed': {}
#         }
#
#         raw_log = pm4py.read_xes(raw_path / f'{dataset}.xes')
#         processed_log = pm4py.read_xes(processed_path / f'{dataset}.xes')
#
#         for log_type, log in [('raw', raw_log), ('processed', processed_log)]:
#             timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
#             time_length = (max(timestamps) - min(timestamps)).days
#             case_count = sum(pm4py.get_variants(log).values())
#             variant_count = len(pm4py.get_variants(log))
#             event_count = len(log)
#             activity_count = len(pm4py.get_event_attribute_values(log, 'concept:name'))
#             dfg, _, _ = pm4py.discover_dfg(log)
#             df_distinct_count = len(dfg)
#             df_total_count = sum(dfg.values())
#
#             log_stats[dataset][log_type] = {
#                 'time length': time_length,
#                 '# cases': case_count,
#                 '# variants': variant_count,
#                 '# events': event_count,
#                 '# activities': activity_count,
#                 '# DFs': df_distinct_count,
#                 '# DFs occurrences': df_total_count
#             }
#
#     return log_stats


log_stats = get_log_stats(dataset_list)