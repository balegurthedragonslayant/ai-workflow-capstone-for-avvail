# AAVAIL – 30-day revenue forecast (IBM AI Enterprise Workflow capstone)

Predicts the revenue of the **next 30 days** from any given date, for all countries combined or for one of
the top-10 countries, and serves it through a Flask API in Docker. (Original course instructions: `INSTRUCTIONS.md`.)

## Quick start

```bash
pip install -r requirements.txt
python run-tests.py                 # ALL unit tests (logger, ingestion/features, model, API) - single script
python -c "import model; model.model_train()"   # train production models -> ./models
python app.py -p 8080               # serve the API

docker build -t aavail-capstone .   # build runs the unit tests; fails if any test fails
docker run -p 8080:8080 aavail-capstone
```

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{"country":"all","date":"2019-06-15"}'
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{"country":"united_kingdom","date":"2019-06-15"}'
curl -X POST localhost:8080/train   -H 'Content-Type: application/json' -d '{"test":false}'
curl localhost:8080/logs ; curl localhost:8080/logs/predict-2026-10.log ; curl localhost:8080/monitor
```

New data: drop the new monthly `invoices-YYYY-MM.json` files into `cs-train/`. The API immediately predicts for
the new dates; `POST /train` re-trains on every file in the directory.

## Layout

| file | role |
|---|---|
| `cslib.py` | **data ingestion** (`fetch_data`, `fetch_ts`) + feature engineering – handles the non-uniform column names, strips letters from invoice ids, fills missing days |
| `eda.py` | EDA with visualizations → `figures/eda-*.png` |
| `model.py` | train (compares baseline / ridge / random forest / gradient boosting), load, predict |
| `model_comparison.py` | hold-out comparison → `reports/model_comparison.csv`, `figures/model-*.png` |
| `logger.py` | train + predict logging (csv), separate `*-test.log` files for tests |
| `app.py` | Flask API: `/train`, `/predict`, `/logs`, `/monitor` |
| `monitoring.py` | production replay, model-vs-baseline performance, data drift → `reports/monitoring.json`, `figures/monitoring-performance.png` |
| `unittests/`, `run-tests.py` | unit tests (39) and the single runner |
| `Dockerfile`, `requirements.txt` | container |

## Part 1 – business opportunity, hypotheses, data

**Opportunity.** AAVAIL management wants a forecast of the revenue for the next 30 days (for any start date, overall
or per country) so that it can plan ahead, instead of waiting for month-end numbers.

**Testable hypotheses**
1. Revenue over the next 30 days can be predicted from recent revenue (previous 7/14/28/70 days) better than by a naive "same as last month" rule.
2. Activity measures (unique invoices, views) add information beyond revenue itself.
3. Revenue has seasonality (autumn peak) – needs ≥ 2 years of history to be exploited; with 19 months it can't be learned reliably.
4. Each country behaves differently, so a model per country (top 10) beats one global model for that country.

**Ideal data** (stated before looking at the data): one record per purchase with date, country, price/revenue,
customer id, product (stream) id, and engagement (times viewed) – at least 2 years, so seasonality can be learned.
The target is revenue summed over the 30 days after a date; features only use the days before it.

**Findings (EDA, `figures/eda-*.png`)**
- The UK is ~90 % of revenue (`eda-revenue-by-country`), so UK ≈ all-country behaviour; the other nine countries are small and noisy.
- Revenue has a strong Oct–Nov peak (`eda-monthly-revenue`) and a weekday pattern (`eda-weekday-correlation`).
- Data quality: column names differ between files (`StreamID`/`stream_id`, `total_price`/`price`), invoice ids contain letters,
  ~190k lines have no `customer_id`, a few large negative prices (returns, e.g. −45k on one day) – handled/visible in `eda-daily-revenue`, `eda-distributions`.

## Part 2 – modelling

Four approaches are compared per data set on a chronological hold-out (last 20 % of days, with a 30-day gap so
targets cannot overlap) using time-series cross-validation for tuning – no shuffling:

| approach | |
|---|---|
| baseline | previous 28 days revenue × 30/28 |
| ridge, random forest, gradient boosting | on features `previous_7/14/28/70`, `recent_invoices`, `recent_views` |

The best by hold-out RMSE is re-trained on all data and saved. Result (`figures/model-comparison-rmse.png`,
`figures/model-vs-baseline.png`, `reports/model_comparison.csv`): the selected model beats the baseline on 9 of 11 data
sets (e.g. all: RMSE 19.1k vs 28.3k, −32 %; UK 17.7k vs 25.1k, −30 %); for Spain and Hong Kong the baseline wins and is kept.

Lesson learned: an initial `previous_year` feature (revenue a year earlier) made every model 3-5× *worse* than the baseline –
with 19 months of data it is zero for the first year of training and populated afterwards (covariate shift). It is now
optional and off by default (`engineer_features(include_previous_year=True)`).

## Part 3 – deployment, tests, monitoring

- **Unit tests** (`python run-tests.py`, 39 tests, all pass): logger (7), ingestion/features (9, incl. a no-leakage check), model (13), API (10).
- **Isolation from production**: `unittests/__init__.py` redirects `AAVAIL_MODEL_DIR` / `AAVAIL_LOG_DIR` to a temp directory
  *before* anything is imported, and every write also uses `test=True` (`test-*.joblib` models, `*-test.log` logs, reduced grids).
  Verified: `./models` and `./logs` are unchanged after a full test run.
- **Monitoring**: `python monitoring.py` replays Aug–Dec 2019 (`cs-production`) day by day – each prediction only uses earlier data and
  goes through `model_predict` (so it is in the predict log) – then compares with the revenue that happened (`figures/monitoring-performance.png`):

  | | model RMSE | baseline RMSE | model MAPE | baseline MAPE |
  |---|---|---|---|---|
  | all | 65.5k | 64.2k | 22.2 % | 25.1 % |
  | united_kingdom | 58.2k | 63.6k | 21.1 % | 25.5 % |

  It also computes a drift z-score per feature (training vs production). All revenue and activity features are flagged
  (|z| > 3): the Q4 seasonal peak was never seen in a full year of training. **Recommendation:** re-train with the new
  months (now ≥ 2 years of data, so `previous_year` can be enabled), keep monitoring against the baseline, and alert on drift.
  `GET /monitor` exposes the training metrics and the drift report.

## Honest limitations

- The Docker image could not be built in the authoring sandbox (Docker Hub was unreachable). The `Dockerfile` steps were reproduced instead in a clean virtualenv
  from `requirements.txt` (tests 39/39 OK, gunicorn serving, API queried for `all`, a country, and bad input). Please run `docker build` once on your machine.
- Hold-out and production RMSE are dominated by a few outliers (returns, the 4 Dec 2018 spike); with only 19 months of data the seasonal pattern isn't learnable.
