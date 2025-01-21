# utils/lag_utils.py
from typing import List


def get_lags_for_horizon(horizon: int, config: dict) -> List[int]:
    """Get appropriate lags based on forecast horizon"""
    if horizon == 1:
        min_lag = config['lags']['horizon_1']['min_lag']
        max_lag = config['lags']['horizon_1']['max_lag']
    elif horizon == 3:
        min_lag = config['lags']['horizon_3']['min_lag']
        max_lag = config['lags']['horizon_3']['max_lag']
    elif horizon == 7:
        min_lag = config['lags']['horizon_7']['min_lag']
        max_lag = config['lags']['horizon_7']['max_lag']
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")

    return list(range(horizon, max_lag + 1))