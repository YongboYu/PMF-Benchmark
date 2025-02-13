# PMF_Benchmark
Directly-Follows Time Series Forecast Pipeline and Benchmark

## Project Overview
A comprehensive framework for preprocessing event logs, generating time series data from directly-follows relations, and benchmarking various forecasting models. The project supports multiple forecasting approaches:

- **Baseline Models**: Simple forecasting baselines
- **Statistical Models**: Traditional statistical approaches (ARIMA, Prophet, etc.)
- **Regression Models**: Machine learning models (Random Forest, XGBoost, LightGBM)
- **Deep Learning Models**: Neural network architectures (RNN, LSTM, GRU, NBEATS, etc.)

## Key Features
- Automated preprocessing pipeline for event logs
- Extraction and transformation of directly-follows relations into time series
- Comprehensive model benchmarking with hyperparameter optimization
- Experiment tracking with Weights & Biases
- Support for both univariate and multivariate time series
- Scalable architecture supporting multiple datasets and prediction horizons

## Project Structure
```
time_series_benchmarks/
├── train.py                          # Main training script
├── config/                           # Configuration files
│   ├── base_config.yaml             # Base configuration for models
│   ├── preprocessing_config.yaml     # Configuration for preprocessing
│   └── model_configs/               # Model-specific configurations
│       ├── baseline_models.yaml
│       ├── statistical_models.yaml
│       ├── regression_models.yaml
│       └── deep_learning_models.yaml
│
├── preprocessing/                    # Preprocessing modules
│   ├── __init__.py
│   ├── event_log_processor.py       # Event log preprocessing
│   ├── df_generator.py              # Directly-follows relation extraction
│   └── time_series_creator.py       # Time series generation
│
├── models/                          # Model implementations
│   ├── __init__.py
│   ├── baseline_models.py           # Baseline forecasting models
│   ├── statistical_models.py        # Statistical forecasting models
│   ├── regression_models.py         # Regression models
│   └── deep_learning_models.py      # Deep learning models
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities
│   ├── evaluation.py               # Model evaluation
│   ├── logging_manager.py          # Centralized logging configuration
│   ├── optuna_manager.py           # Hyperparameter optimization
│   └── wandb_logger.py             # Experiment tracking
│
├── scripts/                         # Execution scripts
│   ├── preprocess_logs.py          # Main preprocessing script
│   └── run_experiments.sh          # Experiment execution script
│
├── notebooks/                       # Jupyter notebooks
│   └── check_intermittency.ipynb   # Check for intermittent DF time series
│
├── data/                            # Data directory
│   ├── raw/                        # Raw event logs
│   │   └── *.xes                   # Original XES files
│   │
│   ├── interim/                    # Intermediate processing data
│   │   ├── processed_logs/        # Filtered and trimmed logs
│   │   │   └── *.xes              # Processed XES files
│   │   └── df_relations/          # Extracted DF relations
│   │       └── *.json             # DF relations in JSON format
│   │
│   └── processed/                  # Final time series data
│       ├── time_series_df.h5      # Combined HDF5 file for all datasets
│       ├── BPI2019_1/              # Dataset-specific directories
│       │   ├── df_relations.csv    # Daily frequencies
│       │   ├── metadata.json       # Dataset information
│       │   └── stats.json          # Statistical properties
│       ├── BPI2017/                # Similar structure for other datasets
│       └── RTFMP/                  # Similar structure for other datasets
│
├── results/                         # Results and outputs
│   ├── results_log.json            # Summary of experiment results
│   ├── models/                     # Saved model files
│   │   ├── BPI2019_1/              # Dataset-specific directories
│   │   │   ├── horizon_1/           # Prediction horizon-specific directories
│   │   │   │   ├── baseline/         # Baseline models group
│   │   │   │   ├── statistical/      # Statistical models group
│   │   │   │   ├── regression/       # Regression models group
│   │   │   │   └── deep_learning/    # Deep learning models group
│   │   │   ├── horizon_3/
│   │   │   └── horizon_7/
│   │   ├── BPI2017/
│   │   └── RTFMP/
│   ├── predictions/                # Model predictions
│   │   ├── BPI2019_1/
│   │   ├── BPI2017/
│   │   └── RTFMP/
│   ├── metrics/                    # Evaluation metrics
│   │   ├── BPI2019_1/
│   │   ├── BPI2017/
│   │   └── RTFMP/
│   └── time_series_analysis/       # Time series analysis tables
│
├── logs/                          # Log files
│   ├── preprocessing.txt         # Preprocessing logs
│   ├── training.txt              # Model training logs
│   ├── data_preprocess/          # Preprocessed data stats
│   │   ├── *.txt                # Dataset-specific 
│   ├── optuna/                   # Optuna optimization logs
│   │   ├── studies/             # Study files
│   └── wandb/                   # Weights & Biases logs
│       ├── runs/               # Individual run logs
│       └── artifacts/          # Model artifacts
│
├── tests/                         # Test files
│   ├── setup_environment.sh
│   ├── run_statistical_rtfmp.sh
│   ├── run_rergession_rtfmp.sh
│   └── run_deep_learning_rtfmp.sh
│
├── pyproject.toml                 # Poetry configuration
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Data Directory Specifications

### Raw Data (`data/raw/`)
- Original event logs in XES format
- Naming convention: `dataset_name.xes`

Download the required event logs and place them in this directory:
- BPI2017: [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)
- BPI2019: [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853)
- RTFMP: [4TU.ResearchData](https://data.4tu.nl/articles/_/12683249/1)

Note: For BPI2019, use only the event log for "3-way matching, invoice before GR" category.

### Intermediate Data (`data/interim/`)
1. **Processed Logs** (`processed_logs/`)
   - Filtered and trimmed event logs
   - Format: XES
   - Contents:
     - Removed infrequent variants
     - Trimmed time period
     - Added artificial start/end events

2. **DF Relations** (`df_relations/`)
   - Directly-follows relations extracted from logs
   - Format: JSON
   - Structure:
     ```json
     {
       "A->B": [
         {
           "start_time": "2024-01-01T10:00:00",
           "end_time": "2024-01-01T11:30:00",
           "case_id": "case_1",
           "frequency": 1
         }
       ]
     }
     ```

### Processed Data (`data/processed/`)
- Combined time series data file: `time_series_df.h5`
  - Single HDF5 file containing all datasets
  - Each dataset stored as a separate key
  - Format: HDF5 Table format

- Dataset-specific directories (e.g., `BPI2019_1/`, `BPI2017/`, `RTFMP/`)
  - `df_relations.csv`: Daily frequencies for directly-follows relations
  - `metadata.json`: Dataset information including:
    - Dataset name
    - Start/end dates
    - Number of relations
    - Time frequency
  - `stats.json`: Statistical properties including:
    - Mean values
    - Standard deviations
    - Minimum values
    - Maximum values

## Setup and Installation

**Requirements**
- Python 3.11 or higher

```bash
# Clone the repository
git clone https://github.com/your-username/PMF_Benchmark.git
cd PMF_Benchmark

# Create virtual environment with Python 3.11+
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Initialize wandb
wandb login
```

## Usage

### Available Parameters

**Datasets**
- `RTFMP`: Road Traffic Fine Management Process dataset
- `BPI2017`: Business Process Intelligence Challenge 2017 dataset
- `BPI2019_1`: Business Process Intelligence Challenge 2019 dataset (filtered for the Item Category of "3-way matching, invoice before GR")

**Prediction Horizons**
- `1`: One-day ahead prediction
- `3`: Three-days ahead prediction
- `7`: Seven-days ahead prediction

### Data Preprocessing
```bash
# Preprocess a single event log
python scripts/preprocess_logs.py --dataset RTFMP
```

### Model Training
```bash
# Train all models for a specific dataset and horizon
python train.py \
    --dataset RTFMP \
    --model_group all \
    --horizon 1

# Train all models in a specific group
python train.py \
    --dataset BPI2019 \
    --model_group deep_learning \
    --horizon 3

# Train a specific model
python train.py \
    --dataset BPI2020 \
    --model_group statistical \
    --model prophet \
    --horizon 7

# Run full benchmark (all datasets × all horizons)
./scripts/run_experiments.sh
```

### Running Test Scripts on VSC (Temporary)
```bash
# Run statistical models on RTFMP dataset
./tests/run_statistical_rtfmp.sh

# Run regression models on RTFMP dataset
./tests/run_regression_rtfmp.sh

# Run deep learning models on RTFMP dataset
./tests/run_deep_learning_rtfmp.sh

# Setup environment (if not already done)
./tests/setup_environment.sh
```

These test scripts are configured to run on a SLURM cluster with specific resource requirements:
- GPU: NVIDIA P100
- CPU cores: 4
- Memory: 32GB
- Maximum runtime: varies by model type (2-4 hours)

Each script runs a specific group of models:
- Statistical: 6 models (Prophet, Exponential Smoothing, Auto ARIMA, Theta, TBATS, Four Theta)
- Regression: 4 models (Linear, Random Forest, XGBoost, LightGBM)
- Deep Learning: 11 models (RNN, LSTM, GRU, DeepAR, NBEATS, NHITS, TCN, Transformer, TFT, DLinear, NLinear)

## Dependencies
- Python 3.11+
- PM4Py
- Pandas
- NumPy
- Optuna
- Weights & Biases
- Darts
- LightGBM
- PyTorch
- Scikit-learn

## Key Dependencies Overview

### Darts
Darts is the core forecasting library used for time series modeling. Key features utilized:
- Unified interface for multiple model types (statistical, ML, deep learning)
- Built-in time series transformations and preprocessing
- Native support for both univariate and multivariate forecasting
- Integrated cross-validation and backtesting

Example usage in deep learning models:
```python
from darts.models import RNNModel, NBEATSModel, TransformerModel
model = RNNModel(
    input_chunk_length=24,
    output_chunk_length=horizon,
    model="LSTM",
    hidden_dim=32
)
model.fit(train)
predictions = model.predict(n=len(test))
```
For detailed configuration options and usage examples, refer to the model-specific configuration files in the `config/model_configs/` directory.


### Weights & Biases (wandb)
Used for experiment tracking and visualization:
- Automated logging of model metrics and parameters
- Real-time training monitoring
- Hyperparameter importance visualization
- Experiment comparison and analysis


### Optuna
Handles hyperparameter optimization with:
- Automated parameter space search
- Integration with wandb for visualization
- Support for different optimization algorithms
- Trial pruning for efficient search


### PM4Py
Used for event log processing:
- XES file parsing and manipulation
- Event log filtering and transformation
- Directly-follows relation extraction
- Process mining analytics

## Contributing
[Your contribution guidelines]

## License
[Your chosen license]

## Citation
```bibtex
[Your citation information]
```

---

_**❗Note: The following sections are for internal sharing purposes and will be removed in the future.**_

---

## Work in Progress

### Known Issues and Planned Improvements

1. **Inference Framework 📍**
   - Change sequence forecasting **inputs** from previous predictions to **ground truth values** on the validation and test set

2. **Multi-step Forecasting Evaluation 📍**
   - Implement dual evaluation strategy for horizons > 1:
     - Evaluate all points in prediction sequences
     - Evaluate only terminal points in sequences
   - Example: For forecasting 3 sequences with horizon=3 , evaluate both all 9 points and only the 3 terminal points
   - Modify evaluation metrics calculation in`utils/evaluation.py`
   - Update prediction visualization in `utils/wandb_logger.py`

3. **Model Implementation Issues**
   - Fix incorrect implementations of:
     - Exponential Smoothing
     - LightGBM
     - N-BEATS
     - N-HiTS
     - TFT

4. **Weights & Biases Integration**
   - Current wandb logging implementation needs fixes for proper metric tracking
   - Handle the error in Optuna callback
   - Enhance visualization logging of model training


### Upcoming Features

1. **Model Enhancements 📍**
   - Add TimeGPT and other foundation models to benchmark suite
   - Implement univariate learning/prediction options for:
     - Regression models
     - Deep learning models

2. **Dataset Implementation**
   - Next priority: Implement BPI2017 dataset with 7-day horizon forecasting 📍
   - Investigate time series intermittency
   - Include example notebooks for data analysis


## Example Results

### Results Structure
The experiment results are stored in `results/results_log.json` in a hierarchical structure organized by dataset, prediction horizon, and model groups. Here's an example of results for the RTFMP dataset:

```json
{
    "dataset": "RTFMP",
    "last_updated": "2025-02-11 16:43:50",
    "horizons": {
        "1": {
            "model_groups": {
                "baseline": {
                    "models": {
                        "naive_mean": {
                            "metrics": {
                                "mae": 5.301,
                                "rmse": 16.081
                            },
                            "training_time": 0.001,
                            "timestamp": "2025-02-04 13:20:27"
                        }
                    }
                },
                "statistical": {
                    "models": {
                        "prophet": {
                            "metrics": {
                                "mae": 11.859,
                                "rmse": 22.108
                            },
                            "training_time": 14.629,
                            "timestamp": "2025-02-06 16:30:55"
                        }
                    }
                }
            }
        }
    }
}
```

### Converting Results to CSV
For easier analysis, you can convert the JSON results to a flattened CSV format using the provided utility function:

```python
from pathlib import Path
from utils.evaluation import generate_results_csv

# Convert results to CSV
json_path = Path('results/results_log.json')
output_path = Path('results/summary.csv')
df = generate_results_csv(json_path, output_path)
```

The generated CSV will have the following columns:
- dataset
- horizon
- model_group
- model
- timestamp
- training_time
- mae
- rmse

Example CSV output:
```csv
dataset,horizon,model_group,model,timestamp,training_time,mae,rmse
RTFMP,1,baseline,naive_mean,2025-02-04 13:20:27,0.001,5.301,16.081
RTFMP,1,statistical,prophet,2025-02-06 16:30:55,14.629,11.859,22.108
RTFMP,3,statistical,theta,2025-02-09 17:34:01,0.323,0.301,0.473
```




