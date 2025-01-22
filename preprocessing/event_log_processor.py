import pm4py
from pathlib import Path
from datetime import timedelta
import logging


class EventLogProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_log(self, dataset: str) -> pm4py.objects.log.obj.EventLog:
        """Process event log with filtering and trimming"""
        input_path = Path(self.config['paths']['raw_logs']) / f'{dataset}.xes'
        output_path = Path(self.config['paths']['processed_logs']) / f'{dataset}.xes'

        # Reuse existing preprocessing logic
        try:
            # Load original log
            org_log = pm4py.read_xes(input_path)

            # Filter infrequent variants
            filter_log = pm4py.filter_variants_by_coverage_percentage(
                org_log,
                self.config['event_log']['filter_percentage']
            )

            # Add artificial start/end
            log = pm4py.insert_artificial_start_end(
                filter_log,
                activity_key=self.config['event_log']['columns']['activity'],
                case_id_key=self.config['event_log']['columns']['case_id'],
                timestamp_key=self.config['event_log']['columns']['timestamp']
            )

            # Trim log
            trimmed_log = self._trim_log(log)

            # Save processed log
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pm4py.write_xes(trimmed_log, output_path)

            return trimmed_log

        except Exception as e:
            self.logger.error(f"Error processing log {dataset}: {e}")
            raise