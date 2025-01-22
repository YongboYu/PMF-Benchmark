import argparse
import yaml
from pathlib import Path
from preprocessing.event_log_processor import EventLogProcessor
from preprocessing.df_generator import DFGenerator
from preprocessing.time_series_creator import TimeSeriesCreator


def main():
    parser = argparse.ArgumentParser(description="Preprocess event logs to time series")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the event log file (without extension)")
    parser.add_argument("--config", type=str,
                        default="config/preprocessing_config.yaml",
                        help="Path to preprocessing configuration")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        # Initialize components
        log_processor = EventLogProcessor(config)
        df_generator = DFGenerator(config)
        ts_creator = TimeSeriesCreator(config)

        # Process event log
        processed_log = log_processor.process_log(args.dataset)

        # Generate DF relations
        df_dict = df_generator.generate_df_dict(processed_log)

        # Get log timespan
        start_time, end_time = log_processor.get_log_timespan(processed_log)

        # Create and save time series
        ts_creator.create_and_save_time_series(
            args.dataset,
            df_dict,
            processed_log,
            start_time,
            end_time
        )

        print(f"Successfully processed {args.dataset}")

    except Exception as e:
        print(f"Error processing {args.dataset}: {e}")
        raise


if __name__ == "__main__":
    main()