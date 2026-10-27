"""Loading helpers shared by the notebooks and the training script."""
from pathlib import Path

import pandas as pd

# Repo root = one folder above src/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_train() -> pd.DataFrame:
    """Jan-Oct 2025 loads with the target column posted_rate."""
    return pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    """Nov-Dec 2025 loads we must predict (no posted_rate)."""
    return pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])


def load_december_inputs() -> pd.DataFrame:
    """31 fixed Lexington -> Fort Wayne rows, one per day of December 2025."""
    return pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"])
