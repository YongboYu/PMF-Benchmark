import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime


def visualize_er_over_time(combined_metrics_csv, dataset, horizon, output_file=None):
    """
    Visualize Entropic Relevance over time for truth, training and predictions by models.
    Uses a predefined list of models to locate columns and simplify labels.
    Drops the first two rows from the dataset before plotting.
    Extracts the first date from window time spans (e.g., "2016-10-22_2016-10-28").

    Args:
        combined_metrics_csv: Path to the combined CSV file with ER metrics
        output_file: Path to save the visualization (if None, will display plot)
    """
    # Define models to evaluate
    models_to_evaluate = [
        ('baseline', 'persistence'),
        ('baseline', 'naive_seasonal'),
        ('statistical', 'ar2'),
        ('regression', 'random_forest'),
        ('regression', 'xgboost'),
        ('deep_learning', 'rnn'),
        ('deep_learning', 'deepar')
    ]

    # Load the data
    df = pd.read_csv(combined_metrics_csv)

    # Drop the first two rows
    df = df.iloc[2:].copy()

    # Extract first date from time span format (e.g., "2016-10-22_2016-10-28")
    if 'window' in df.columns:
        try:
            # Extract the first date from the time span format
            df['window_date'] = df['window'].apply(
                lambda x: datetime.strptime(x.split('_')[0], '%Y-%m-%d')
                if isinstance(x, str) and '_' in x
                else pd.to_datetime(x)
            )
            # Set the extracted date as index
            df.set_index('window_date', inplace=True)
        except Exception as e:
            print(f"Warning: Could not parse window dates: {e}")
            # Fall back to using window as string
            df.set_index('window', inplace=True)
    else:
        print("Warning: 'window' column not found")
        # If there is no window column, use the existing index
        pass

    # Define base columns
    truth_col = 'truth_er'
    training_col = 'training_er'

    # Identify model columns based on the models_to_evaluate list
    model_er_cols = []
    model_names = {}

    for group, name in models_to_evaluate:
        col = f"{group}_{name}_er"
        if col in df.columns:
            model_er_cols.append(col)
            # Store just the model name (not the group) for the label
            model_names[col] = name

    # Prepare the plot
    plt.figure(figsize=(12, 7))

    # Plot ground truth
    plt.plot(df.index, df[truth_col], 'k-', linewidth=2, label='Ground Truth')

    # Plot training baseline
    plt.plot(df.index, df[training_col], 'k--', linewidth=2, label='Training Set')

    # Plot models with different colors
    colors = sns.color_palette('colorblind', n_colors=len(model_er_cols))
    for i, col in enumerate(model_er_cols):
        # Use the simplified model name from our dictionary
        model_name = model_names[col]
        plt.plot(df.index, df[col], color=colors[i], linewidth=1.5,
                  markersize=4, label=model_name)

    # Customize plot
    plt.xlabel('Time Window (Start Date)')
    plt.ylabel('Entropic Relevance (ER)')
    plt.title(f'Entropic Relevance Over Time: Ground Truth vs Models ({dataset}, horizon {horizon})')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Format x-axis ticks for better readability
    # plt.xticks(rotation=45)
    plt.xticks()
    plt.tight_layout()

    # Add legend with smaller font and outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

    plt.close()

# Example usage
# Example usage
# visualize_er_over_time(
#     combined_metrics_csv=f'results/er_metrics/{dataset}_horizon_{horizon}_combined_er_metrics.csv',
#     dataset=dataset,
#     horizon=horizon,
#     output_file=f'results/er_metrics/{dataset}_horizon_{horizon}_er_visualization.png'
# )

def run_visualizations_for_datasets(datasets, horizons):
    """
    Run visualize_er_over_time for multiple datasets and horizons.

    Args:
        datasets: List of dataset names
        horizons: List of horizon values
    """
    for dataset in datasets:
        for horizon in horizons:
            try:
                # Path to CSV file
                csv_path = f'results/er_metrics_v2/{dataset}_horizon_{horizon}_combined_er_metrics.csv'
                # Output file path
                output_path = f'results/er_metrics_v2/{dataset}_horizon_{horizon}_er_visualization.png'

                print(f"Processing {dataset} with horizon {horizon}...")
                visualize_er_over_time(
                    combined_metrics_csv=csv_path,
                    dataset=dataset,
                    horizon=horizon,
                    output_file=output_path
                )
                print(f"Visualization saved to {output_path}")
            except Exception as e:
                print(f"Error processing {dataset} with horizon {horizon}: {str(e)}")


# Example usage with 5 datasets and 2 horizons
# datasets = ["BPI2017", "BPI2019_1", "sepsis", "Hospital_Billing", "RTFMP"]
# datasets = ["sepsis"]
datasets = ["BPI2017", "sepsis", "Hospital_Billing"]
# horizons = [7, 28]
horizons = [7]
run_visualizations_for_datasets(datasets, horizons)