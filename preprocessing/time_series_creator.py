import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
import logging


class TimeSeriesCreator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    # def create_time_series(self, df_dict: Dict, patterns: Dict[str, pd.DataFrame],
    #                        start_time: pd.Timestamp, end_time: pd.Timestamp,
    #                        dataset: str):
    def create_time_series(self, df_dict: Dict,
                           start_time: pd.Timestamp, end_time: pd.Timestamp,
                           dataset: str):
        """Create and save time series data"""
        try:
            # Create univariate time series
            df_series = self._create_df_time_series(
                df_dict, start_time, end_time
            )
            self._save_univariate_series(df_series, dataset)

            # # Create multivariate time series
            # mv_series = self._create_multivariate_series(
            #     patterns, df_series, start_time, end_time
            # )
            # self._save_multivariate_series(mv_series, dataset)

            self.logger.info(f"Created time series for dataset: {dataset}")

        except Exception as e:
            self.logger.error(f"Error creating time series for {dataset}: {e}")
            raise

    def _create_df_time_series(self, df_dict: Dict,
                               start_time: pd.Timestamp,
                               end_time: pd.Timestamp) -> pd.DataFrame:
        """Create time series from DF relations"""
        # Create date range with daily frequency
        time_range = pd.date_range(
            start=start_time.normalize(),  # Start of day
            end=end_time.normalize(),  # End of day
            freq='D'
        )

        # Initialize DataFrame with zeros
        df_series = pd.DataFrame(0, index=time_range, columns=df_dict.keys())

        # Count occurrences for each DF relation per day
        for df_rel, occurrences in df_dict.items():
            for occ in occurrences:
                # Convert timestamp to datetime if string
                start_time = pd.to_datetime(occ['start_time'])
                # Get the day
                day = start_time.normalize()

                if day in df_series.index:
                    df_series.at[day, df_rel] += 1

        return df_series

    def _create_multivariate_series(self, patterns: Dict[str, pd.DataFrame],
                                    df_series: pd.DataFrame,
                                    start_time: pd.Timestamp,
                                    end_time: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Create multivariate time series"""
        mv_series = {}

        # Activity frequency series
        act_patterns = patterns['activity_patterns']
        mv_series['activity_freq'] = self._aggregate_by_time(
            act_patterns, 'activity', start_time, end_time
        )

        # Case attributes series
        case_attrs = patterns['case_attributes']
        mv_series['case_attrs'] = self._aggregate_case_attributes(
            case_attrs, start_time, end_time
        )

        # Resource utilization series
        resource_patterns = patterns['resource_patterns']
        mv_series['resource_util'] = self._aggregate_resource_patterns(
            resource_patterns, start_time, end_time
        )

        # Combined DF patterns
        mv_series['df_patterns'] = self._combine_patterns(
            df_series, mv_series, start_time, end_time
        )

        return mv_series

    def _save_univariate_series(self, df_series: pd.DataFrame, dataset: str):
        """Save univariate time series"""
        base_path = Path(self.config['paths']['processed']['univariate']) / dataset
        base_path.mkdir(parents=True, exist_ok=True)

        # Save time series
        df_series.to_csv(base_path / 'df_relations.csv')

        # Save metadata
        metadata = {
            'dataset': dataset,
            'start_date': df_series.index[0].strftime('%Y-%m-%d'),
            'end_date': df_series.index[-1].strftime('%Y-%m-%d'),
            'n_relations': len(df_series.columns),
            'frequency': 'D'  # Daily frequency
        }

        with open(base_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save stats
        stats = {
            'mean': df_series.mean().to_dict(),
            'std': df_series.std().to_dict(),
            'min': df_series.min().to_dict(),
            'max': df_series.max().to_dict()
        }

        with open(base_path / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=4)

    def _save_multivariate_series(self, mv_series: Dict[str, pd.DataFrame],
                                  dataset: str):
        """Save multivariate time series"""
        base_path = Path(self.config['paths']['processed']['multivariate']) / dataset
        base_path.mkdir(parents=True, exist_ok=True)

        # Save each feature set
        for feature_name, feature_df in mv_series.items():
            feature_df.to_csv(base_path / f'{feature_name}.csv')

        # Save metadata
        metadata = {
            'dataset': dataset,
            'features': list(mv_series.keys()),
            'feature_dimensions': {
                name: df.shape[1] for name, df in mv_series.items()
            },
            'frequency': self.config['time_series']['interval'],
            'start_date': mv_series['df_patterns'].index[0].strftime('%Y-%m-%d'),
            'end_date': mv_series['df_patterns'].index[-1].strftime('%Y-%m-%d')
        }

        with open(base_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)