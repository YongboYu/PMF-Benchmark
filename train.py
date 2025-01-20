# train.py
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
from darts import TimeSeries
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from models.baseline_models import BaselineModels
from models.statistical_models import StatisticalModels
from models.regression_models import RegressionModels
from models.deep_learning_models import DeepLearningModels


class ModelTrainer:
    def __init__(self, config_path: str = "config/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_loader = DataLoader(self.config)
        self.evaluator = Evaluator(self.config)

    def train_and_evaluate(self, dataset: str, time_interval: str) -> Dict[str, Any]:
        # Load and split data
        series = self.data_loader.load_data(dataset, time_interval)
        train, val, test = self.data_loader.split_data(series)

        results = {}

        # Train and evaluate each model group
        for model_group in ['baseline', 'statistical', 'regression', 'deep_learning']:
            # Transform data and get transformer
            train_t, val_t, test_t, transformer = self.data_loader.transform_data(
                train, val, test, model_group
            )

            # Get model class
            model_class = {
                'deep_learning': DeepLearningModels,
                'statistical': StatisticalModels,
                'baseline': BaselineModels,
                'regression': RegressionModels
            }[model_group]

            # Initialize model with config and transformer
            model = model_class(self.config)

            # Train models and get predictions (transformer passed for validation scoring)
            predictions = model.train_and_predict(train_t, val_t, test_t, transformer)

            # Evaluate final test predictions using evaluation.py
            group_results = self.evaluator.evaluate_model_group(
                predictions=predictions,
                test=test,  # Original test data
                transformer=transformer
            )

            # Update results
            results.update(group_results)

        return results

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate time series models')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--time_interval', type=str, required=True, help='Time interval')
    parser.add_argument('--model_group', type=str, required=True,
                        choices=['baseline', 'statistical', 'regression', 'deep_learning'],
                        help='Model group to train')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Initialize trainer
    trainer = ModelTrainer(config_path=args.config)

    # Train and evaluate single model group
    series = trainer.data_loader.load_data(args.dataset, args.time_interval)
    train, val, test = trainer.data_loader.split_data(series)

    # Transform data and get transformer
    train_t, val_t, test_t, transformer = trainer.data_loader.transform_data(
        train, val, test, args.model_group
    )

    # Get model class
    model_class = {
        'deep_learning': DeepLearningModels,
        'statistical': StatisticalModels,
        'baseline': BaselineModels,
        'regression': RegressionModels
    }[args.model_group]

    # Train models and get predictions
    model = model_class(trainer.config)
    predictions = model.train_and_predict(train_t, val_t, test_t, transformer)

    # Evaluate predictions
    results = trainer.evaluator.evaluate_model_group(
        predictions=predictions,
        test=test,
        model_group=args.model_group,
        transformer=transformer
    )

    # Save results
    trainer.evaluator.save_results(results, args.dataset, args.time_interval)

if __name__ == '__main__':
    main()

