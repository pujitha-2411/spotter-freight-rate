"""Run the whole pipeline:  python -m src.train

1. clean + feature-engineer the data
2. validate on a time-based split (train Jan-Aug, test Sep-Oct)
3. rolling monthly backtest (train on everything before month M, test on M)
4. refit on all of Jan-Oct and predict the 12,000 validation loads
5. predict the 31 fixed December loads for score.py's chart
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cleaning import clean, is_rate_outlier, weight_fill_values
from src.data import DATA_DIR, ROOT, load_december_inputs, load_train, load_validation
from src.features import FEATURES, add_features
from src.model import DistanceBaseline, FreightRateModel, mae, mape

FIG_DIR = ROOT / "reports" / "figures"
HOLDOUT_START = "2025-09-01"   # last two months of training data = our test set


def evaluate(model, train, test):
    """Fit on believable training rows; score on ALL test rows and on believable ones."""
    model.fit(train[~is_rate_outlier(train)])
    pred = model.predict(test)
    good = ~is_rate_outlier(test)
    y = test["posted_rate"]
    return {
        "MAE_all": round(mae(y, pred), 1), "MAPE_all": round(mape(y, pred), 2),
        "MAE_believable": round(mae(y[good], pred[good]), 1), "MAPE_believable": round(mape(y[good], pred[good]), 2),
    }, pred


def december_frame(december, train, validation):
    """The December file only has pickup, delivery, distance, equipment,
    weight and date. Fill in what the model also needs:
    - coordinates: looked up from the city names in the training data
    - market_index and quote signal: the average over the real validation
      loads on that same December day (the best market estimate we have)."""
    coords = {}
    for side in ["pickup", "delivery"]:
        c = train[[side, f"{side}_lat", f"{side}_lon"]].drop_duplicates(side)
        coords.update({row[side]: (row[f"{side}_lat"], row[f"{side}_lon"]) for _, row in c.iterrows()})
    df = december.copy()
    df["pickup_lat"], df["pickup_lon"] = zip(*df["pickup"].map(coords))
    df["delivery_lat"], df["delivery_lon"] = zip(*df["delivery"].map(coords))
    daily = validation.groupby("date")[["market_index", "quote_signal"]].mean()
    df = df.join(daily, on="date")
    return df


def main():
    # ---------- 1. load, clean, features ----------
    raw_train, raw_val, december = load_train(), load_validation(), load_december_inputs()
    fill = weight_fill_values(raw_train)
    train = add_features(clean(raw_train, fill))
    val = add_features(clean(raw_val, fill))
    print(f"Training rows: {len(train):,}  (dropping {is_rate_outlier(train).sum()} unbelievable rates)")

    # ---------- 2. time-based holdout ----------
    dev, holdout = train[train.date < HOLDOUT_START], train[train.date >= HOLDOUT_START]
    results = {}
    results["baseline_distance_only"], _ = evaluate(DistanceBaseline(), dev, holdout)
    results["trend_plus_lightgbm"], holdout_pred = evaluate(FreightRateModel(), dev, holdout)
    for name, r in results.items():
        print(f"Holdout Sep-Oct | {name:24s} {r}")

    # ---------- 3. rolling monthly backtest ----------
    backtest = []
    for month in ["2025-07", "2025-08", "2025-09", "2025-10"]:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        r, _ = evaluate(FreightRateModel(), train[train.date < start], train[(train.date >= start) & (train.date < end)])
        backtest.append({"test_month": month, **r})
        print(f"Backtest {month}: {r}")
    results["rolling_backtest"] = backtest

    # ---------- 4. refit on all of Jan-Oct, predict validation ----------
    final = FreightRateModel().fit(train[~is_rate_outlier(train)])
    results["trend_percent_per_month"] = round(final.trend_per_day * 30 * 100, 2)
    val["predicted_rate"] = final.predict(val)
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")[["load_id"]]
    out = template.merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    out["predicted_rate"] = out["predicted_rate"].round(2)
    assert len(out) == 12_000 and out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("Saved validation_predictions.csv")

    # ---------- 5. December chart inputs ----------
    dec = add_features(december_frame(december, raw_train, val))
    december["predicted_rate"] = final.predict(dec).round(2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)
    print("Filled data/december_chart_inputs.csv")

    # ---------- figures + metrics for the report ----------
    importance = pd.Series(final.trees.booster_.feature_importance("gain"), index=FEATURES)
    results["feature_importance_share"] = (importance / importance.sum()).round(3).sort_values(ascending=False).to_dict()
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))
    make_figures(holdout, holdout_pred, importance, backtest)


def make_figures(holdout, pred, importance, backtest):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    good = ~is_rate_outlier(holdout).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(holdout.posted_rate[good], pred[good], s=3, alpha=0.3, label="believable rates")
    ax.scatter(holdout.posted_rate[~good], pred[~good], s=6, color="red", alpha=0.6, label="unbelievable rates (data errors)")
    lim = [0, 8000]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Actual posted rate ($)", ylabel="Predicted rate ($)", title="Holdout Sep-Oct 2025: predicted vs actual")
    ax.legend(loc="upper left")
    fig.savefig(FIG_DIR / "05_holdout_pred_vs_actual.png", dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    importance.sort_values().plot.barh(ax=ax, color="#064A56")
    ax.set(title="What the LightGBM part relies on (gain)", xlabel="Importance")
    fig.savefig(FIG_DIR / "06_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
