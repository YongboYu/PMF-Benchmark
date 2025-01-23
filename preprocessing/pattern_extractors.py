import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging


class PatternExtractors:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_patterns(self, log, dataset: str):
        """Extract all patterns from log"""
        try:
            # Extract case attributes
            case_attrs = self._extract_case_attributes(log)
            self._save_patterns(case_attrs, 'case_attributes', dataset)

            # Extract activity patterns
            activity_patterns = self._extract_activity_patterns(log)
            self._save_patterns(activity_patterns, 'activity_patterns', dataset)

            # Extract resource patterns
            resource_patterns = self._extract_resource_patterns(log)
            self._save_patterns(resource_patterns, 'resource_patterns', dataset)

        except Exception as e:
            self.logger.error(f"Error extracting patterns for {dataset}: {e}")
            raise

    def _extract_case_attributes(self, log) -> pd.DataFrame:
        """Extract case-level attributes"""
        case_data = []

        for case in log:
            case_id = case.attributes[self.config['event_log']['columns']['case_id']]

            case_data.append({
                'case_id': case_id,
                'start_time': min(event[self.config['event_log']['columns']['timestamp']]
                                  for event in case),
                'end_time': max(event[self.config['event_log']['columns']['timestamp']]
                                for event in case),
                'n_activities': len(case),
                'variant': str([event[self.config['event_log']['columns']['activity']]
                                for event in case])
            })

        return pd.DataFrame(case_data)

    def _extract_activity_patterns(self, log) -> pd.DataFrame:
        """Extract activity-level patterns"""
        activities_data = []

        for case in log:
            for event in case:
                activities_data.append({
                    'timestamp': event[self.config['event_log']['columns']['timestamp']],
                    'activity': event[self.config['event_log']['columns']['activity']],
                    'case_id': case.attributes[self.config['event_log']['columns']['case_id']]
                })

        return pd.DataFrame(activities_data)

    def _extract_resource_patterns(self, log) -> pd.DataFrame:
        """Extract resource utilization patterns"""
        resource_data = []

        for case in log:
            for event in case:
                if self.config['event_log']['columns']['resource'] in event:
                    resource_data.append({
                        'timestamp': event[self.config['event_log']['columns']['timestamp']],
                        'resource': event[self.config['event_log']['columns']['resource']],
                        'activity': event[self.config['event_log']['columns']['activity']],
                        'case_id': case.attributes[self.config['event_log']['columns']['case_id']]
                    })

        return pd.DataFrame(resource_data)

    def _save_patterns(self, df: pd.DataFrame, pattern_type: str, dataset: str):
        """Save patterns to parquet file"""
        output_path = Path(self.config['paths']['interim'][pattern_type])
        output_path.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path / f'{dataset}.parquet', index=False)