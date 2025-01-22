import optuna
from typing import Dict, Any, Callable
import logging
from pathlib import Path
import joblib


class OptunaManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create directory for storing studies
        self.study_dir = Path(config['paths']['optuna_studies_dir'])
        self.study_dir.mkdir(parents=True, exist_ok=True)

    def create_study(self, model_name: str, horizon: int) -> optuna.Study:
        """Create or load an Optuna study"""
        study_name = f"{model_name}_h{horizon}"
        storage_name = f"sqlite:///{self.study_dir}/{study_name}.db"

        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction="minimize",
                load_if_exists=True
            )
            return study
        except Exception as e:
            self.logger.error(f"Error creating/loading study: {e}")
            raise

    def optimize(self,
                 study: optuna.Study,
                 objective: Callable,
                 n_trials: int,
                 timeout: Optional[int] = None) -> optuna.Study:
        """Run optimization"""
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            return study
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

    def save_study(self, study: optuna.Study, model_name: str, horizon: int):
        """Save study results"""
        study_path = self.study_dir / f"{model_name}_h{horizon}_study.pkl"
        joblib.dump(study, study_path)

    def load_study(self, model_name: str, horizon: int) -> Optional[optuna.Study]:
        """Load existing study"""
        study_path = self.study_dir / f"{model_name}_h{horizon}_study.pkl"
        if study_path.exists():
            return joblib.load(study_path)
        return None