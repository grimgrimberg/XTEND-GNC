"""Shared utility functions for the XTEND submission helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy/pandas types to Python built-ins for JSON serialization."""

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
