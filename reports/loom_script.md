# Loom script (2-3 min, camera on)

Screen order: README results table -> notebooks/01_eda.ipynb charts -> src/cleaning.py -> src/model.py -> src/train.py -> December chart.

**0:00-0:15 Intro (camera, README open)**
"Hi, I'm Pujitha. This is my freight-rate model for Spotter. On a Sep-Oct holdout it gets about 4% MAPE on all loads and 1.5% on loads with believable prices, versus 9.5% for a distance-only baseline."

**0:15-0:50 EDA findings (01_eda.ipynb, charts 1-3)**
- "Distance explains most of the price, correlation 0.91, so I predict rate per mile, around $2.15 a mile."
- "Reefer costs most per mile, then Flatbed, then Dry Van."
- "Prices follow the market index up to a June peak and back down."
- "Validation is November-December, all after training, so everything is validated by time, never randomly."

**0:50-1:20 Data quality (chart 4, src/cleaning.py)**
- "Negative weights have the same shape as normal ones, so they're sign errors: I flip them."
- "Missing weights get the median for that equipment. Missing market_index gets the same day's median, because it's basically one value per day."
- "About 1.4% of loads cost under $1.30 or over $4 a mile with gaps around them, and no pattern. I treat them as entry errors and drop them from training only."
- "Validation has the same problems, so the same fixes run on it."

**1:20-1:55 Model choice (src/model.py)**
- "Target is log of rate per mile: errors become relative and predictions are always positive."
- "I describe lanes by coordinates, not city names, because 8 validation cities never appear in training."
- "Rates also drift up about 0.67% a month. Trees can't extrapolate past October, so a one-number linear trend handles the drift and LightGBM learns the rest."
- "Comparing each quote signal with that day's average was the other big gain."

**1:55-2:25 Split + code walkthrough (src/train.py)**
- "Main check: train January-August, test September-October. Then a rolling backtest, month by month, July to October."
- "August is hardest, 5% on clean loads, because the market index dropped faster than prices. Once it's in training, Sep-Oct are back to 1-2%."
- "Then I refit on all ten months and write the 12,000 predictions."

**2:25-2:50 December chart + close**
- "For December, coordinates come from the city names and the market signals are the real December daily averages, so the line follows the weekly market cycle, about $830-850. One limit: no November-December training data, so holiday premiums only come through the market index. Thanks!"

Likely follow-up questions: why not a random split (leakage from neighbouring days); why LightGBM (non-linear interactions, handles categories, fast); why log target (relative error, positivity); what would you do next (holiday seasonality with more history, quantile ranges, outlier classifier).
