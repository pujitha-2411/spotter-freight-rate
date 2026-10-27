"""Step 2: fix the data-quality problems found in the EDA.

The same fixes run on training AND validation rows, because the
validation file has the same missing / negative values.
"""
import pandas as pd

# Rate-per-mile limits for "believable" loads. The healthy middle of the
# data is $1.67-$3.18 per mile (1st-99th percentile); the two error groups
# sit clearly outside $1.30 and $4.00, with an empty gap in between.
MIN_RATE_PER_MILE = 1.30
MAX_RATE_PER_MILE = 4.00


def fix_weight(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Negative weights are sign errors, missing weights get a typical value."""
    df = df.copy()
    # -31,000 lb has exactly the same shape as +31,000 lb loads -> flip the sign
    df["weight"] = df["weight"].abs()
    # Missing weight -> the median weight for that equipment type (learned on training)
    df["weight"] = df["weight"].fillna(df["equipment"].map(fill_values))
    return df


def fix_market_index(df: pd.DataFrame) -> pd.DataFrame:
    """market_index is basically one number per day (it varies only ~0.02
    between loads on the same day), so a missing value is filled with the
    median of the other loads on that date."""
    df = df.copy()
    same_day_median = df.groupby("date")["market_index"].transform("median")
    df["market_index"] = df["market_index"].fillna(same_day_median)
    # Safety net (never needed on this data): a day with no values at all
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())
    return df


def weight_fill_values(train: pd.DataFrame) -> dict:
    """Median (absolute) weight per equipment type, learned on training only."""
    return train.assign(weight=train["weight"].abs()).groupby("equipment")["weight"].median().to_dict()


def clean(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Apply every feature fix. Safe for training and validation rows."""
    return fix_market_index(fix_weight(df, fill_values))


def is_rate_outlier(df: pd.DataFrame) -> pd.Series:
    """True for loads whose price per mile is not believable (~1.4% of rows).
    We only drop these from TRAINING: we can't drop rows we must predict."""
    rate_per_mile = df["posted_rate"] / df["distance"]
    return (rate_per_mile < MIN_RATE_PER_MILE) | (rate_per_mile > MAX_RATE_PER_MILE)
