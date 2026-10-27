"""Step 3: turn a cleaned load into model inputs.

Design choices:
- No city names: 8 validation cities never appear in training, so the
  model describes a lane by coordinates and distance instead.
- No month / day-of-year: training has no Nov-Dec, so a tree model could
  not use those values anyway. The market signals carry the seasonality.
"""
import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2025-01-01")

FEATURES = [
    "distance",            # road miles (the biggest driver of price)
    "straight_miles",      # straight-line miles between the two cities
    "circuity",            # road miles / straight miles (twisty routes cost more per mile)
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",  # where the lane is
    "equipment",           # Dry Van / Reefer / Flatbed
    "weight",
    "market_index",        # daily market level
    "quote_signal",        # this load's quote signal
    "quote_day_mean",      # average quote signal across all loads that day
    "quote_vs_day",        # this load's quote minus the day average
    "day_of_week",
]


def straight_line_miles(lat1, lon1, lat2, lon2):
    """Haversine formula: distance on the earth's surface, in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["straight_miles"] = straight_line_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["circuity"] = df["distance"] / df["straight_miles"]
    # Quote signal is much more useful compared with the same day's quotes
    df["quote_day_mean"] = df.groupby("date")["quote_signal"].transform("mean")
    df["quote_vs_day"] = df["quote_signal"] - df["quote_day_mean"]
    df["day_of_week"] = df["date"].dt.dayofweek
    # Days since 1 Jan 2025: used by the linear trend, NOT given to the trees
    df["days"] = (df["date"] - START_DATE).dt.days
    df["equipment"] = pd.Categorical(df["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
    return df
