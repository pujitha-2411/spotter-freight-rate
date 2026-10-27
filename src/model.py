"""Step 4: the model = slow upward price trend + LightGBM.

Target: log(rate per mile). Dividing by distance takes out the biggest
effect; the log makes errors relative (5% on a $500 load counts like 5%
on a $5,000 load) and keeps predictions positive.

Why a separate trend? After removing lane and market effects, rates
still creep up about 0.6% a month. Tree models can't extrapolate past
the last date they saw, so on Nov-Dec they would freeze at October's
level. A one-number linear trend handles the drift; LightGBM learns
everything else on the de-trended target.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import FEATURES

LGBM_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
)


def estimate_daily_trend(train: pd.DataFrame) -> float:
    """Slope of log(rate per mile) per day, controlling for market level,
    distance and equipment so the trend isn't just the market cycle."""
    X = pd.DataFrame({
        "days": train["days"],
        "market_index": train["market_index"],
        "quote_day_mean": train["quote_day_mean"],
        "log_distance": np.log(train["distance"]),
    })
    X = X.join(pd.get_dummies(train["equipment"].astype(str), dtype=float))  # one 0/1 column per equipment
    y = np.log(train["posted_rate"] / train["distance"])
    return float(LinearRegression().fit(X, y).coef_[0])


class FreightRateModel:
    def fit(self, train: pd.DataFrame):
        self.trend_per_day = estimate_daily_trend(train)
        log_rate_per_mile = np.log(train["posted_rate"] / train["distance"])
        detrended = log_rate_per_mile - self.trend_per_day * train["days"]
        self.trees = lgb.LGBMRegressor(**LGBM_PARAMS).fit(train[FEATURES], detrended)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        log_rate_per_mile = self.trees.predict(df[FEATURES]) + self.trend_per_day * df["days"]
        return np.exp(log_rate_per_mile) * df["distance"].to_numpy()   # back to dollars


class DistanceBaseline:
    """Benchmark: posted_rate = a + b * distance (one line for everything)."""
    def fit(self, train):
        self.lr = LinearRegression().fit(train[["distance"]], train["posted_rate"])
        return self

    def predict(self, df):
        return np.clip(self.lr.predict(df[["distance"]]), 1, None)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y) * 100)
