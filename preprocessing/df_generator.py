import pm4py
import json
from pathlib import Path
from typing import Dict, Any
import logging


class DFGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_df_dict(self, log: pm4py.objects.log.obj.EventLog) -> Dict[str, list]:
        """Generate directly-follows dictionary from event log"""
        try:
            df_dict = {}

            # Extract directly-follows relations
            for case in log:
                for i in range(len(case) - 1):
                    current_activity = case[i]["concept:name"]
                    next_activity = case[i + 1]["concept:name"]
                    df_relation = f"{current_activity}->{next_activity}"

                    if df_relation not in df_dict:
                        df_dict[df_relation] = []

                    df_dict[df_relation].append({
                        'start_time': case[i]["time:timestamp"],
                        'end_time': case[i + 1]["time:timestamp"]
                    })

            return df_dict

        except Exception as e:
            self.logger.error(f"Error generating DF dictionary: {e}")
            raise

    def save_df_dict(self, df_dict: Dict[str, list], dataset: str):
        """Save DF dictionary to file"""
        try:
            output_path = Path(self.config['paths']['df_dict']) / f"{dataset}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(df_dict, f, default=str)

        except Exception as e:
            self.logger.error(f"Error saving DF dictionary: {e}")
            raise