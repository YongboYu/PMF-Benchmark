# utils/data_loader.py
from darts.dataprocessing.transformers import Scaler, BoxCox
from typing import Tuple, Dict, Any, Optional
from darts import TimeSeries
import pandas as pd


class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize transformers based on config
        self.transformers = {
            'deep_learning': Scaler(),
            'statistical': BoxCox(),
            'baseline': None,
            'regression': None
        }

    def load_data(self, dataset: str, time_interval: str) -> TimeSeries:
        """Load data from file"""
        data = pd.read_hdf(
            f'{self.config["data"]["path"]}/{time_interval}/{dataset}.h5',
            key=f'df_{dataset}'
        )
        return TimeSeries.from_dataframe(data)

    def split_data(self, series: TimeSeries) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Split data into train, validation and test sets"""
        train_val, test = series.split_after(self.config['data']['train_test_split'])
        train, val = train_val.split_after(self.config['data']['train_val_split'])
        return train, val, test

    def transform_data(
            self,
            train: TimeSeries,
            val: Optional[TimeSeries],
            test: TimeSeries,
            model_group: str
    ) -> Tuple[TimeSeries, Optional[TimeSeries], TimeSeries]:
        """Transform data based on model group"""
        transformer = self.transformers[model_group]
        if transformer is None:
            return train, val, test

        if model_group == 'statistical':
            # Add offset and epsilon for box-cox
            offset = self.config['transformation']['offset']
            epsilon = self.config['transformation']['epsilon']

            train = transformer.fit_transform(train + offset) + epsilon
            test = transformer.transform(test + offset) + epsilon
            return train, None, test

        # For deep learning models
        train = transformer.fit_transform(train)
        test = transformer.transform(test)
        val = transformer.transform(val) if val is not None else None
        return train, val, test

    def get_transformer(self, model_group: str):
        """Get transformer for a model group"""
        return self.transformers[model_group]