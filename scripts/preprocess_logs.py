import sys
from pathlib import Path

# Add the parent directory of the preprocessing module to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

###########

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from preprocessing.event_log_processor import EventLogProcessor
from preprocessing.df_generator import DFGenerator
from preprocessing.pattern_extractors import PatternExtractors
from preprocessing.time_series_creator import TimeSeriesCreator


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )


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

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        log_processor = EventLogProcessor(config)
        df_generator = DFGenerator(config)
        pattern_extractors = PatternExtractors(config)
        ts_creator = TimeSeriesCreator(config)

        logger.info(f"Processing dataset: {args.dataset}")

        # Process event log
        processed_log = log_processor.process_log(args.dataset)

        # Generate DF relations
        df_dict = df_generator.generate_df_relations(processed_log, args.dataset)

        # # Extract patterns
        # pattern_extractors.extract_patterns(processed_log, args.dataset)

        # Get log timespan
        start_time, end_time = log_processor.get_log_timespan(processed_log)

        # Create time series
        ts_creator.create_time_series(
            df_dict=df_dict,
            # patterns={
            #     'activity_patterns': pd.read_parquet(
            #         f"{config['paths']['interim']['activity_patterns']}/{args.dataset}.parquet"
            #     ),
            #     'case_attributes': pd.read_parquet(
            #         f"{config['paths']['interim']['case_attributes']}/{args.dataset}.parquet"
            #     ),
            #     'resource_patterns': pd.read_parquet(
            #         f"{config['paths']['interim']['resource_patterns']}/{args.dataset}.parquet"
            #     )
            # },
            start_time=start_time,
            end_time=end_time,
            dataset=args.dataset
        )

        logger.info(f"Successfully processed {args.dataset}")

    except Exception as e:
        logger.error(f"Error processing {args.dataset}: {e}")
        raise


if __name__ == "__main__":
    main()