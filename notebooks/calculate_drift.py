# python notebooks/calculate_drift.py --datasets BPI2017 BPI2019_1 Hospital_Billing sepsis RTFMP

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_dataset_key(store: pd.HDFStore, dataset: str) -> str:
    """Get the correct key for a dataset in the HDF5 store"""
    keys = store.keys()
    logger = logging.getLogger(__name__)
    logger.info(f"Available keys in HDF5 store: {keys}")

    # Find matching key for the dataset
    matching_keys = [key for key in keys if dataset.lower() in key.lower()]

    if not matching_keys:
        raise KeyError(f"No matching key found for dataset {dataset}")

    if len(matching_keys) > 1:
        logger.warning(f"Multiple keys found for dataset {dataset}: {matching_keys}")
        logger.warning(f"Using first matching key: {matching_keys[0]}")

    return matching_keys[0]


def calculate_drift_scores(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[
    Dict[str, float], Dict[str, Dict[str, float]]]:
    """Calculate drift scores between train and test sets for each DF relation."""

    # Split the time series data
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Calculate average occurrences for train and test
    train_avg = train_df.mean()
    test_avg = test_df.mean()

    drift_scores = {}
    avg_occurrences = {}

    # Calculate drift score for each DF relation
    for df_relation in df.columns:
        train_val = train_avg[df_relation]
        test_val = test_avg[df_relation]

        # Calculate drift score
        if train_val + test_val > 0:  # Avoid division by zero
            drift_score = abs(train_val - test_val) / (train_val + test_val)
        else:
            drift_score = 0.0

        # Store results
        drift_scores[df_relation] = drift_score
        avg_occurrences[df_relation] = {
            'train_avg': train_val,
            'test_avg': test_val
        }

    return drift_scores, avg_occurrences


def process_dataset(dataset: str, time_series_data: pd.HDFStore, config: dict) -> pd.DataFrame:
    """Process a single dataset and return results DataFrame"""

    logger = logging.getLogger(__name__)
    logger.info(f"Processing dataset: {dataset}")

    try:
        # Get correct key for the dataset
        dataset_key = get_dataset_key(time_series_data, dataset)
        logger.info(f"Using key: {dataset_key}")

        # Read dataset from HDF5 store
        df = time_series_data.get(dataset_key)

        # Calculate drift scores
        drift_scores, avg_occurrences = calculate_drift_scores(
            df,
            config['data']['simple_split']
        )

        # Create DataFrame
        df_results = pd.DataFrame.from_dict(drift_scores, orient='index',
                                            columns=['drift_score'])

        # Add average occurrences
        df_results['train_avg'] = pd.Series({k: v['train_avg'] for k, v in avg_occurrences.items()})
        df_results['test_avg'] = pd.Series({k: v['test_avg'] for k, v in avg_occurrences.items()})

        # Add percentage difference column
        df_results['pct_difference'] = ((df_results['test_avg'] - df_results['train_avg']) /
                                        df_results['train_avg'] * 100)

        df_results.index.name = 'df_relation'

        return df_results

    except Exception as e:
        logger.error(f"Error processing dataset {dataset}: {str(e)}")
        raise


def create_drift_plots(df_results: pd.DataFrame, dataset: str, output_dir: Path):
    """Create visualizations for drift analysis"""
    # Use a more basic style that's guaranteed to work
    plt.style.use('default')
    
    # Create figure directory
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(df_results['drift_score'], bins=50, alpha=0.7, color='skyblue')
    plt.title(f'Distribution of Drift Scores - {dataset}')
    plt.xlabel('Drift Score')
    plt.ylabel('Count')
    plt.axvline(df_results['drift_score'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df_results["drift_score"].mean():.3f}')
    plt.axvline(df_results['drift_score'].median(), color='g', linestyle='--', 
                label=f'Median: {df_results["drift_score"].median():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f'{dataset}_drift_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top 20 Highest Drift Relations
    plt.figure(figsize=(15, 8))
    top_20 = df_results.nlargest(20, 'drift_score')
    plt.barh(range(len(top_20)), top_20['drift_score'], alpha=0.7, color='skyblue')
    plt.yticks(range(len(top_20)), top_20.index, fontsize=8)
    plt.title(f'Top 20 Highest Drift Relations - {dataset}')
    plt.xlabel('Drift Score')
    plt.ylabel('DF Relation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f'{dataset}_top20_drift.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot of Train vs Test averages
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(df_results['train_avg'], df_results['test_avg'], 
                         alpha=0.5, c=df_results['drift_score'], cmap='viridis')
    plt.colorbar(scatter, label='Drift Score')
    
    # Add diagonal line
    max_val = max(df_results['train_avg'].max(), df_results['test_avg'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='No Drift Line')
    
    plt.title(f'Train vs Test Average Occurrences - {dataset}')
    plt.xlabel('Train Average')
    plt.ylabel('Test Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f'{dataset}_train_test_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot of drift scores with individual points
    plt.figure(figsize=(10, 6))
    plt.boxplot(df_results['drift_score'])
    plt.scatter([1] * len(df_results), df_results['drift_score'], 
                alpha=0.3, color='red', s=20)
    plt.title(f'Drift Score Distribution - {dataset}')
    plt.ylabel('Drift Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f'{dataset}_drift_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Sorted Drift Scores Plot with Quartile Lines
    plt.figure(figsize=(10, 6))
    
    # Sort drift scores in descending order
    sorted_scores = df_results['drift_score'].sort_values(ascending=False)
    x_values = np.arange(len(sorted_scores))  # Use integers for x-axis
    
    # Plot the main line
    plt.plot(x_values, sorted_scores, 'b-', alpha=0.7)
    
    # Calculate quartile positions
    q1_pos = int(len(sorted_scores) * 0.25)
    q2_pos = int(len(sorted_scores) * 0.50)
    q3_pos = int(len(sorted_scores) * 0.75)

    # Add horizontal lines for mean and median
    plt.axhline(y=sorted_scores.mean(), color='r', linestyle='--',
                label='Mean')
    plt.axhline(y=sorted_scores.median(), color='g', linestyle='--',
                label='Median')

    # Add vertical lines for quartiles
    plt.axvline(x=q1_pos, color='orange', linestyle='--', 
                label='25%')
    plt.axvline(x=q2_pos, color='black', linestyle='--',
                label='50%')
    plt.axvline(x=q3_pos, color='purple', linestyle='--', 
                label='75%')

    plt.title(f'Sorted Drift Scores - {dataset}')
    plt.xlabel('DF Relations (sorted by drift score)')
    plt.ylabel('Drift Score')
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Format x-axis with integers
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f'{dataset}_sorted_drift_scores.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_plot(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create comparison plots across all datasets"""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for dataset, df in results.items():
        plot_data.append({
            'dataset': dataset,
            'mean_drift': df['drift_score'].mean(),
            'median_drift': df['drift_score'].median(),
            'std_drift': df['drift_score'].std()
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.35
    
    plt.bar(x - width/2, plot_df['mean_drift'], width, label='Mean Drift',
            yerr=plot_df['std_drift'], capsize=5)
    plt.bar(x + width/2, plot_df['median_drift'], width, label='Median Drift')
    
    plt.xlabel('Dataset')
    plt.ylabel('Drift Score')
    plt.title('Drift Score Comparison Across Datasets')
    plt.xticks(x, plot_df['dataset'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate drift scores for time series data")
    parser.add_argument("--datasets", nargs='+', required=True,
                        help="List of dataset names to process")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                        help="Path to configuration file")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(config['paths']['results_dir']) / "drift_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open HDF5 file
    data_path = Path(config['paths']['data_dir']) / "processed" / "time_series_df.h5"

    if not data_path.exists():
        logger.error(f"Time series data file not found: {data_path}")
        return

    results = {}
    summary_stats = []

    with pd.HDFStore(data_path, mode='r') as store:
        # Log available keys
        logger.info(f"Available keys in HDF5 store: {store.keys()}")

        # Process each dataset
        for dataset in args.datasets:
            try:
                # Calculate drift scores
                df_results = process_dataset(dataset, store, config)

                if df_results is not None:
                    # Save individual results
                    df_results.to_csv(output_dir / f"{dataset}_drift_scores.csv")
                    results[dataset] = df_results

                    # Create plots for individual dataset
                    create_drift_plots(df_results, dataset, output_dir)

                    # Calculate summary statistics
                    summary_stats.append({
                        'dataset': dataset,
                        'avg_drift_score': df_results['drift_score'].mean(),
                        'median_drift_score': df_results['drift_score'].median(),
                        'std_drift_score': df_results['drift_score'].std(),
                        'max_drift_score': df_results['drift_score'].max(),
                        'min_drift_score': df_results['drift_score'].min(),
                        'num_df_relations': len(df_results),
                        'high_drift_relations': len(df_results[df_results['drift_score'] > 0.5]),
                        'medium_drift_relations': len(df_results[
                                                          (df_results['drift_score'] >= 0.1) &
                                                          (df_results['drift_score'] <= 0.5)
                                                          ]),
                        'low_drift_relations': len(df_results[df_results['drift_score'] < 0.1]),
                        'avg_pct_difference': df_results['pct_difference'].mean()
                    })

                    logger.info(f"Successfully processed {dataset}")
                    logger.info(f"Average drift score: {df_results['drift_score'].mean():.4f}")
                    logger.info(f"Number of high drift relations (>0.5): {summary_stats[-1]['high_drift_relations']}")

            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {str(e)}")
                continue

    # Create comparison plot across datasets
    if len(results) > 1:
        create_comparison_plot(results, output_dir)

    # Save summary statistics
    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        df_summary.to_csv(output_dir / "drift_summary.csv", index=False)

        # Print summary table
        print("\nSummary Statistics:")
        print(df_summary.to_string(index=False))

        logger.info(f"Summary statistics saved to {output_dir / 'drift_summary.csv'}")

    logger.info("Processing completed")


if __name__ == "__main__":
    main()