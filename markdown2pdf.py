from mdpdf import MarkdownPdf

# Create the markdown content
markdown_content = """
PMF_Benchmark/
├── config/
│   ├── base_config.yaml
│   ├── preprocessing_config.yaml
│   └── model_configs/
│       ├── baseline_models.yaml
│       ├── statistical_models.yaml
│       ├── regression_models.yaml
│       └── deep_learning_models.yaml
│
├── preprocessing/
│   ├── __init__.py
│   ├── event_log_processor.py
│   ├── df_generator.py
│   ├── feature_engineering.py
│   └── time_series_creator.py
│
├── models/
│   ├── __init__.py
│   ├── baseline_models.py
│   ├── statistical_models.py
│   ├── regression_models.py
│   └── deep_learning_models.py
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── lag_utils.py
│   ├── optuna_manager.py
│   └── wandb_logger.py
│
├── scripts/
│   ├── preprocess_logs.py
│   └── run_experiments.sh
│
├── notebooks/
│   ├── experiment_analysis.ipynb
│   └── visualization.ipynb
│
├── data/
│   ├── raw/
│   │   └── *.xes
│   │
│   ├── interim/
│   │   ├── processed_logs/
│   │   │   └── *.xes
│   │   ├── df_relations/
│   │   │   └── *.json
│   │   ├── case_attributes/
│   │   │   └── *.parquet
│   │   ├── activity_patterns/
│   │   │   └── *.parquet
│   │   └── resource_patterns/
│   │       └── *.parquet
│   │
│   └── processed/
│       ├── univariate/
│       │   ├── dataset1/
│       │   │   ├── df_relations.csv
│       │   │   ├── metadata.json
│       │   │   └── stats.json
│       │   ├── dataset2/
│       │   └── dataset3/
│       └── multivariate/
│           ├── dataset1/
│           │   ├── activity_freq.csv
│           │   ├── case_attrs.csv
│           │   ├── df_patterns.csv
│           │   ├── resource_util.csv
│           │   └── metadata.json
│           ├── dataset2/
│           └── dataset3/
│
├── results/
│   ├── models/
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── dataset3/
│   ├── predictions/
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── dataset3/
│   ├── metrics/
│   │   ├── dataset1/
│   │   ├── dataset2/
│   │   └── dataset3/
│   └── optuna_studies/
│       └── snapshots/
│
├── logs/
│   ├── wandb/
│   └── error_logs/
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── requirements.txt
├── setup.py
├── README.md
└── train.py
"""

# Save to markdown file
with open('project_structure.md', 'w') as f:
    f.write(markdown_content)

# Convert to PDF
md_pdf = MarkdownPdf()
md_pdf.convert('project_structure.md', 'project_structure.pdf')