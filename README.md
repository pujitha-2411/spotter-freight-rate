# Freight Rate Prediction (Spotter AI ML assessment)

Predict the posted rate ($) of a truckload from its lane, equipment, weight, date and market signals.

**Result on a time-based holdout (train Jan-Aug 2025, test Sep-Oct 2025):**

| Model | MAE (all rows) | MAPE (all rows) | MAE (believable rows) | MAPE (believable rows) |
|---|---|---|---|---|
| Baseline: rate = a + b * distance | $194.9 | 9.51% | $140.4 | 7.03% |
| **Trend + LightGBM (final)** | **$90.5** | **4.06%** | **$34.5** | **1.51%** |

"Believable" excludes the ~1.4% of loads whose rate per mile is below $1.30 or above $4.00, which look like data-entry errors (see the report).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python -m src.train          # cleans data, validates, trains, writes predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`python -m src.train` prints the holdout and backtest scores and writes:

- `validation_predictions.csv`: 12,000 rows, `load_id,predicted_rate`
- `data/december_chart_inputs.csv`: the 31 December rows with `predicted_rate` filled in
- `reports/metrics.json` and `reports/figures/05-06_*.png`

`score.py` (Spotter's, unchanged) validates both files and draws `scorer_results/candidate_december.png`.

## Repo layout

```
data/                 Spotter's CSVs
notebooks/01_eda.ipynb  exploration and the four EDA charts
src/data.py           load the CSVs
src/cleaning.py       fix negative/missing weight, missing market_index, flag rate outliers
src/features.py       lane geometry, quote-signal features, day of week
src/model.py          baseline + the final model (linear trend + LightGBM)
src/train.py          the full pipeline: validate -> refit -> predict
reports/              report PDF, figures, metrics.json
score.py              Spotter's scorer
```

## Approach in one paragraph

Distance explains most of the price, so the model predicts **log(rate per mile)** and multiplies back by distance. Negative weights are flipped (they are sign errors), missing weights get the equipment median, and missing `market_index` gets the same-day median (it is a daily value). Rate-per-mile outliers are dropped from training only. Because the validation loads are the future (Nov-Dec), every check is **time-based**: a Sep-Oct holdout plus a rolling month-by-month backtest. Rates drift up ~0.67% per month on top of the market cycle, which trees can't extrapolate, so a one-number linear trend handles the drift and LightGBM learns the rest. Cities are described by coordinates, not names, because 8 validation cities never appear in training.

## Report

`reports/freight_rate_report.pdf` (rebuild with `python reports/make_report.py` after training and scoring).
