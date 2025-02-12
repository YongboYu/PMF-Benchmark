from darts.dataprocessing.transformers import Scaler
from typing import Tuple, Dict, Any, Optional, NamedTuple
from darts import TimeSeries
from pathlib import Path
import pandas as pd
import logging
import numpy as np

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def load_data(self, dataset: str) -> TimeSeries:
        """Load data from file"""
        data_path = Path(self.config["paths"]["data_dir"]) / 'processed' / 'time_series_df.h5'
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        data = pd.read_hdf(data_path, key=dataset)
        return TimeSeries.from_dataframe(data)

    def split_data(self, series: TimeSeries, model_group: str) -> Tuple[TimeSeries, Optional[TimeSeries], TimeSeries]:
        """Split data into train, validation, and test sets"""
        if model_group in ['baseline', 'statistical']:
            train_size = self.config['data']['simple_split']
            train, test = series.split_after(train_size)
            return train, None, test
        
        test_size = self.config['data']['test_split']
        val_size = self.config['data']['val_split']
        
        train_val, test = series.split_after(1 - test_size)
        train, val = train_val.split_after(1 - val_size / (1 - test_size))
        return train, val, test

    def transform_data(
            self,
            train: TimeSeries,
            val: Optional[TimeSeries],
            test: TimeSeries,
            model_group: str
    ) -> Tuple[TimeSeries, Optional[TimeSeries], TimeSeries, Optional[Any]]:
        """Transform data based on model group"""
        try:
            # Initialize transformer based on model group
            transformer = None
            transform_configs = self.config.get('transformations', {})
            
            # For baseline and regression models, return untransformed data
            if model_group in ['baseline', 'regression']:
                return train, val, test, None
            
            elif model_group == 'deep_learning':
                transformer = Scaler()
                train_t = transformer.fit_transform(train)
                val_t = transformer.transform(val)
                test_t = transformer.transform(test)
                return train_t, val_t, test_t, transformer
            
            elif model_group == 'statistical':
                transform_config = transform_configs[model_group]
                offset = transform_config['offset']
                epsilon = transform_config['epsilon']
                
                # Step 1: Add offset to raw data
                train_offset = train + offset
                test_offset = test + offset
                
                # Step 2: Apply log transformation
                train_t = train_offset.map(lambda x: np.log(x))
                test_t = test_offset.map(lambda x: np.log(x))
                
                # Step 3: Add epsilon in log space
                train_t = train_t + epsilon
                test_t = test_t + epsilon
                
                return train_t, None, test_t, None
                
            else:
                raise ValueError(f"Unknown model group: {model_group}")
            
        except Exception as e:
            self.logger.error(f"Error in {model_group} transformation: {str(e)}")
            raise

    def prepare_data(self, dataset: str, model_group: str) -> Tuple[TimeSeries, Optional[TimeSeries], TimeSeries, Any, Optional[Dict[str, Any]]]:
        """Main function to prepare data for training
        
        Args:
            dataset: Name of the dataset
            model_group: Type of model to prepare data for
            
        Returns:
            Tuple containing:
            - train: Training data
            - val: Validation data (or None)
            - test: Test data
            - transformer: The fitted transformer (or None)
            - transform_params: Dictionary of transformation parameters (or None)
        """
        try:
            # Load data
            series = self.load_data(dataset)
            
            # Split data
            train, val, test = self.split_data(series, model_group)
            
            # Transform data
            return self.transform_data(
                train, val, test, model_group
            )
            
        except Exception as e:
            raise Exception(f"Error preparing data for {dataset}: {str(e)}")

    def inverse_transform(self, series: TimeSeries, transformer: Any, model_group: str) -> TimeSeries:
        """Inverse transform data"""
        if model_group in ['baseline', 'regression']:
            return series
        
        try:
            if model_group == 'statistical':
                # Get transformation parameters
                transform_params = self.config['transformations'][model_group]
                epsilon = transform_params.get('epsilon')
                offset = transform_params.get('offset')
                transform_type = transform_params.get('type')
                
                # Remove epsilon
                series_no_epsilon = series - epsilon
                
                # Apply inverse transform
                if transform_type == 'log':
                    series_no_transform = series_no_epsilon.map(lambda x: np.exp(x))
                else:  # box-cox
                    series_no_transform = transformer.inverse_transform(series_no_epsilon)
                
                # Remove offset
                return series_no_transform - offset
            
            else:  # deep_learning
                return transformer.inverse_transform(series)
            
        except Exception as e:
            self.logger.error(f"Error in inverse transformation: {str(e)}")
            raise