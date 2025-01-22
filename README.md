# PMF_Benchmark
Directly-Follows Time Series Forecast Pipeline and Benchmark

## Project Overview
A comprehensive framework for preprocessing event logs, generating time series data from directly-follows relations, and benchmarking various forecasting models.

## Project Structure
```
time_series_benchmarks/
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
│   ├── feature_engineering.py       # Feature creation
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
│   ├── lag_utils.py                # Lag feature creation
│   ├── optuna_manager.py           # Hyperparameter optimization
│   └── wandb_logger.py             # Experiment tracking
│
├── scripts/                         # Execution scripts
│   ├── preprocess_logs.py          # Main preprocessing script
│   └── run_experiments.sh          # Experiment execution script
│
├── notebooks/                       # Jupyter notebooks
│   ├── experiment_analysis.ipynb    # Analysis of results
│   └── visualization.ipynb         # Visualization notebooks
│
├── data/                           # Data directory
│   ├── raw/                        # Raw event logs
│   │   └── *.xes                   # Original XES files
│   │
│   ├── interim/                    # Intermediate processing data
│   │   ├── processed_logs/         # Filtered and trimmed logs
│   │   │   └── *.xes              # Processed XES files
│   │   ├── df_relations/           # Extracted DF relations
│   │   │   └── *.json             # DF relations in JSON format
│   │   ├── case_attributes/        # Extracted case attributes
│   │   │   └── *.parquet          # Case level features
│   │   ├── activity_patterns/      # Activity patterns
│   │   │   └── *.parquet          # Activity level features
│   │   └── resource_patterns/      # Resource patterns
│   │       └── *.parquet          # Resource utilization features
│   │
│   └── processed/                  # Final time series data
│       ├── univariate/             # Univariate time series
│       │   ├── dataset1/           # Dataset-specific files
│       │   │   ├── df_relations.csv    # DF relation frequencies
│       │   │   ├── metadata.json       # Dataset metadata
│       │   │   └── stats.json          # Statistical properties
│       │   ├── dataset2/
│       │   └── dataset3/
│       └── multivariate/           # Multivariate time series
│           ├── dataset1/           # Dataset-specific files
│           │   ├── activity_freq.csv    # Activity frequencies
│           │   ├── case_attrs.csv       # Case attributes
│           │   ├── df_patterns.csv      # Combined DF patterns
│           │   ├── resource_util.csv    # Resource utilization
│           │   └── metadata.json        # Feature descriptions
│           ├── dataset2/
│           └── dataset3/
│
├── results/                        # Results and outputs
│   ├── models/                     # Saved model files
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── dataset3/
│   ├── predictions/                # Model predictions
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── dataset3/
│   ├── metrics/                    # Evaluation metrics
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── dataset3/
│   └── optuna_studies/            # Optimization results
│       └── snapshots/             # Study snapshots
│
├── logs/                          # Log files
│   ├── wandb/                     # WandB logs
│   └── error_logs/                # Error logs
│
├── tests/                         # Test files
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── requirements.txt               # Project dependencies
├── setup.py                      # Package setup file
├── README.md                     # Project documentation
└── train.py                      # Main training script
```

## Data Directory Specifications

### Raw Data (`data/raw/`)
- Original event logs in XES format
- Naming convention: `dataset_name.xes`

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

3. **Case Attributes** (`case_attributes/`)
   - Case-level features and patterns
   - Format: Parquet
   - Features:
     - Case duration
     - Number of activities
     - Start/end timestamps
     - Variant information

4. **Activity Patterns** (`activity_patterns/`)
   - Activity-level aggregated features
   - Format: Parquet
   - Features:
     - Activity frequencies
     - Average duration per activity
     - Activity transitions
     - Concurrent activities

5. **Resource Patterns** (`resource_patterns/`)
   - Resource utilization features
   - Format: Parquet
   - Features:
     - Resource workload
     - Resource availability
     - Activity-resource mappings
     - Resource performance metrics

### Processed Data (`data/processed/`)
1. **Univariate Time Series** (`univariate/`)
   - Single-variable time series for each DF relation
   - Files per dataset:
     - `df_relations.csv`: Daily frequencies
     - `metadata.json`: Dataset information
     - `stats.json`: Statistical properties

2. **Multivariate Time Series** (`multivariate/`)
   - Multi-variable time series combining different features
   - Files per dataset:
     - `activity_freq.csv`: Activity frequencies
     - `case_attrs.csv`: Aggregated case attributes
     - `df_patterns.csv`: Combined DF patterns
     - `resource_util.csv`: Resource utilization
     - `metadata.json`: Feature descriptions

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/your-username/PMF_Benchmark.git
cd PMF_Benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Initialize wandb
wandb login
```

## Usage

### Data Preprocessing
```bash
# Preprocess a single event log
python scripts/preprocess_logs.py --dataset dataset_name

# Preprocess multiple event logs
./scripts/run_experiments.sh
```

### Model Training
```bash
# Train a specific model
python train.py \
    --dataset dataset_name \
    --model_group deep_learning

# Run full benchmark
./scripts/run_experiments.sh
```

### Analysis
```bash
# Open Jupyter notebook for analysis
jupyter notebook notebooks/experiment_analysis.ipynb
```

## Dependencies
- Python 3.8+
- PM4Py
- Pandas
- NumPy
- Optuna
- Weights & Biases
- PyTorch
- Scikit-learn

## Contributing
[Your contribution guidelines]

## License
[Your chosen license]

## Citation
```bibtex
[Your citation information]
```