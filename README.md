# PMF Benchmark
**Process Model Forecasting Benchmark Framework**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for preprocessing event logs, extracting directly-follows (DF) relations, 
converting them to time series, and benchmarking various forecasting models with statistical and 
entropic relevance evaluation.

## Overview

This framework enables researchers and practitioners to:
- **Process Event Logs**: Automated preprocessing of process event logs (XES format)
- **Generate Time Series**: Extract and transform directly-follows relations into time series data
- **Benchmark Models**: Comprehensive evaluation across multiple forecasting approaches
- **Evaluate with Process-Aware Metrics**: Advanced entropic relevance (ER) evaluation
- **Track Experiments**: Integration with Weights & Biases for monitoring and visualization

## Features

### Model Categories
- **Baseline Models**: Persistence, Naive Mean, Naive Seasonal, Naive Drift, Naive Moving Average
- **Statistical Models**: AR, ARIMA, SES, Prophet
- **Machine Learning Models**: Linear Regression, Random Forest, XGBoost, LightGBM
- **Deep Learning Models**: RNN, LSTM, GRU, DeepAR, N-HiTS, Transformer, TCN, DLinear, NLinear
- **Foundation Models**: TimeGPT

### Supported Datasets
- **[BPI2017](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)**:  BPI Challenge 2017
- **[BPI2019_1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1)**: BPI Challenge 2019 (3-way matching, invoice before GR)
- **[Sepsis](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)**: Sepsis event log
- **[Hospital_Billing](https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741)**: Hospital billing process data


### Evaluation Metrics
- **Traditional**: MAE, RMSE
- **Process-Aware**: Entropic Relevance (ER)

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/PMF_Benchmark.git
cd PMF_Benchmark

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Optional: Initialize Weights & Biases
wandb login
```

### Basic Usage

1. **Preprocess Event Logs**
```bash
python scripts/preprocess_logs.py --dataset BPI2017
```

2. **Train Models**
```bash
# Train a specific model
python train.py --dataset BPI2017 --model_group statistical --model prophet --horizon 7

# Train all models in a category
python train.py --dataset BPI2017 --model_group deep_learning --horizon 7

# Train all models
python train.py --dataset BPI2017 --model_group all --horizon 7
```

3. **Calculate Entropic Relevance**
```bash
python run_er_evaluation.py --dataset BPI2017 --horizon 7
```

### Available Options

- **Datasets**: `BPI2017`, `BPI2019_1`, `Hospital_Billing`, `sepsis`
- **Model Groups**: `baseline`, `statistical`, `regression`, `deep_learning`, `foundation`
- **Horizons**: `7`, `28` (days)

## Project Structure

```
PMF_Benchmark/
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration
│   ├── dataset/               # Dataset-specific configs
│   └── model_configs/         # Model-specific configurations
│
├── models/                     # Model implementations
│   ├── baseline_models.py     # Naive forecasting models
│   ├── statistical_models.py  # ARIMA, Prophet, etc.
│   ├── regression_models.py   # Traditional ML models
│   ├── deep_learning_models.py # Neural networks
│   └── foundation_models.py   # Pre-trained models
│
├── preprocessing/              # Data preprocessing
│   ├── event_log_processor.py # Event log preprocessing
│   ├── df_generator.py        # DF relation extraction
│   └── time_series_creator.py # Time series generation
│
├── utils/                      # Utility functions
│   ├── data_loader.py         # Data loading utilities
│   ├── evaluation.py          # Model evaluation
│   ├── logging_manager.py     # Logging configuration
│   ├── optuna_manager.py      # Hyperparameter optimization
│   └── wandb_logger.py        # Experiment tracking
│
├── data/                       # Data organization
│   ├── raw/                   # Original event logs
│   ├── interim/               # Intermediate processing
│   ├── processed/             # Final time series
│   └── ground_truth/          # Ground truth data
│
├── results/                    # Experiment results
│   ├── evaluation/            # Model evaluation metrics
│   ├── predictions/           # Model predictions
│   └── models/                # Saved models
│
├── scripts/                    # Execution scripts
├── tests/                      # Test scripts
└── notebooks/                  # Analysis notebooks
```

## Data Pipeline

The framework follows a structured data processing pipeline:

1. **Event Log Preprocessing**: Filter infrequent variants, trim time periods, add artificial start/end events
2. **DF Relation Extraction**: Extract directly-follows relations with timestamps from processed logs
3. **Time Series Generation**: Aggregate relations by time windows (daily frequency)
4. **Model Training**: Train models with hyperparameter optimization using Optuna
5. **Evaluation**: Calculate traditional and process-aware metrics including Entropic Relevance

## Configuration

The framework uses YAML-based configuration with hierarchical organization:

- `configs/base_config.yaml`: Core settings (paths, data splits, transformations)
- `configs/dataset/`: Dataset-specific parameters
- `configs/model_configs/`: Model-specific hyperparameters

Example model training with custom config:
```bash
python train.py --config configs/custom_config.yaml --dataset BPI2017 --horizon 7
```

## Experiment Tracking

### Weights & Biases Integration
- Automatic logging of metrics, hyperparameters, and predictions
- Real-time training monitoring
- Experiment comparison and visualization
- Model artifact storage

### Local Tracking
- Results stored in `results/results_log.json`
- Detailed metrics in `results/evaluation/`
- Model predictions in `results/predictions/`

## Key Dependencies

- **[Darts](https://unit8co.github.io/darts/)**: Core forecasting library with unified interface for multiple model types
- **[PM4Py](https://pm4py.fit.fraunhofer.de/)**: Process mining and event log processing
- **[Optuna](https://optuna.org/)**: Hyperparameter optimization with automated parameter space search
- **[Weights & Biases](https://wandb.ai/)**: Experiment tracking and visualization
- **[PyTorch](https://pytorch.org/)**: Deep learning backend

## Citation

If you find this repository helpful for your work, please consider citing our paper:

```bibtex

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For detailed documentation, tutorials, and API reference, visit our [documentation](docs/) or explore the `notebooks/` directory for usage examples.