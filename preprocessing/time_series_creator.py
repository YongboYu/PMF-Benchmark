import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


class TimeSeriesCreator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_time_series(self, log: pm4py.objects.log.obj.EventLog,
                           dataset: str, time_interval: str):
        """Create time series from event log"""
        try:
            # Get DF relations
            df_relations = self._extract_df_relations(log)

            # Create time series
            df_time_series = self._create_df_time_series(
                df_relations, time_interval
            )

            # Save outputs
            self._save_outputs(df_time_series, dataset, time_interval)

        except Exception as e:
            self.logger.error(f"Error creating time series: {e}")
            raise

    def _extract_df_relations(self, log):
        """Extract directly-follows relations"""
        # Reuse existing logic from dataset.py
        # Reference lines 20-95 from dataset.py