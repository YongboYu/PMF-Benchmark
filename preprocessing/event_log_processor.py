import pm4py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from datetime import datetime


class EventLogProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_log(self, dataset: str) -> pm4py.objects.log.obj.EventLog:
        """Process event log with filtering and trimming"""
        try:
            # Load original log
            input_path = Path(self.config['paths']['raw']) / f'{dataset}.xes'
            log = pm4py.read_xes(str(input_path))

            # Get original timespan
            org_timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
            org_start_time = min(org_timestamps)
            org_end_time = max(org_timestamps)

            # Filter infrequent variants
            filtered_log = self._filter_variants(log)

            # Add artificial start/end events
            processed_log = self._add_artificial_events(filtered_log)

            # Trim timespan
            trimmed_log = self._trim_timespan(processed_log)

            # Save processed log
            self._save_processed_log(trimmed_log, dataset)

            # Save processing information
            self._save_processing_info(
                dataset=dataset,
                org_log=log,
                filter_log=filtered_log,
                trimmed_log=trimmed_log,
                org_start_time=org_start_time,
                org_end_time=org_end_time,
                trim_start=self.get_log_timespan(trimmed_log)[0],
                trim_end=self.get_log_timespan(trimmed_log)[1]
            )

            return trimmed_log

        except Exception as e:
            self.logger.error(f"Error processing log {dataset}: {e}")
            raise

    def _filter_variants(self, log) -> pm4py.objects.log.obj.EventLog:
        """Filter infrequent variants"""
        return pm4py.filter_variants_by_coverage_percentage(
            log,
            min_coverage_percentage=self.config['event_log']['filter_percentage']
        )


    def _trim_timespan(self, log) -> pm4py.objects.log.obj.EventLog:
        """Trim log timespan"""
        start_time, end_time = self.get_log_timespan(log)
        trim_days = int((end_time - start_time).days * self.config['event_log']['trim_percentage'])

        trim_start = start_time + pd.Timedelta(days=trim_days)
        trim_end = end_time - pd.Timedelta(days=trim_days)

        return pm4py.filter_time_range(log,
                                       trim_start.strftime('%Y-%m-%d %H:%M:%S'),
                                       trim_end.strftime('%Y-%m-%d %H:%M:%S'))

    def _add_artificial_events(self, log) -> pm4py.objects.log.obj.EventLog:
        """Add artificial start/end events"""
        return pm4py.insert_artificial_start_end(log)

    def _save_processed_log(self, log, dataset: str):
        """Save processed log"""
        output_path = Path(self.config['paths']['interim']['processed_logs'])
        output_path.mkdir(parents=True, exist_ok=True)

        pm4py.write_xes(log, str(output_path / f'{dataset}.xes'))

    def _save_processing_info(self, dataset: str, org_log, filter_log, trimmed_log,
                              org_start_time, org_end_time, trim_start, trim_end):
        """Save log processing information"""
        log_dir = Path('logs/data_preprocess')
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / f'{dataset}.txt', 'w') as f:
            f.write(f'Dataset: {dataset}\n')
            f.write(f'Number of traces (Original Log): {len(org_log)}\n')
            f.write(f'Start time (org): {org_start_time}\n')
            f.write(f'End time (org): {org_end_time}\n')
            f.write(f'Filter percentage: {self.config["event_log"]["filter_percentage"]}\n')
            f.write(
                f'Number of traces (Filtered Log): {len(filter_log)} '
                f'({round(len(filter_log) / len(org_log) * 100, 2)}%)\n'
            )
            f.write(f'Trim percentage: {self.config["event_log"]["trim_percentage"]}\n')
            f.write(f'Start time: {trim_start}\n')
            f.write(f'End time: {trim_end}\n')
            f.write(
                f'Number of traces (Trimmed Log): {len(trimmed_log)} '
                f'({round(len(trimmed_log) / len(org_log) * 100, 2)}%)\n\n'
            )

    def get_log_timespan(self, log) -> Tuple[datetime, datetime]:
        """Get log start and end times"""
        timestamps = pm4py.get_event_attribute_values(log, 'time:timestamp')
        return min(timestamps), max(timestamps)