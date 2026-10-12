#!/usr/bin/env python
"""
model.py - train / load / predict for the 30 day revenue forecast.

One model is trained per data set ('all' + the top-10 countries by revenue).
For each data set several approaches are compared with time-series aware validation
(no shuffling) and the best one (lowest RMSE on a chronological hold-out) is re-trained
on all of the data and saved:

    baseline   naive forecast: revenue of the previous 28 days scaled to 30 days
    ridge      linear model on the engineered features
    rf         random forest
    gb         gradient boosting

Isolation of tests: every function that writes (models, logs) accepts test=True which
  - subsets the countries / hyper-parameter grid so it is fast
  - writes models with a 'test-' prefix and logs to the '*-test.log' files
Directories can be redirected with AAVAIL_MODEL_DIR / AAVAIL_LOG_DIR / AAVAIL_DATA_DIR.
"""

import os
import re
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cslib import fetch_ts, engineer_features, country_key, HORIZON
import logger
from logger import update_predict_log, update_train_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("AAVAIL_MODEL_DIR", os.path.join(BASE_DIR, "models"))
DATA_DIR = os.environ.get("AAVAIL_DATA_DIR", os.path.join(BASE_DIR, "cs-train"))
REPORT_DIR = os.environ.get("AAVAIL_REPORT_DIR", os.path.join(BASE_DIR, "reports"))

## iterate the version and note with each change
MODEL_VERSION = 1.0
MODEL_VERSION_NOTE = "supervised time-series model: best of baseline/ridge/random forest/gradient boosting"

TEST_COUNTRIES = ['all', 'united_kingdom']

_TS_CACHE = {}   # (data_dir, signature) -> dict of time-series data frames


class BaselineModel(BaseEstimator, RegressorMixin):
    """naive benchmark: the next 30 days look like the last 28 days (feature 'previous_28')"""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X['previous_28'], dtype=float) * (HORIZON / 28.0)


def _candidates(test=False):
    """candidate models with their (small) hyper-parameter grids"""
    rf_grid = {'rf__n_estimators': [20] if test else [50, 150],
               'rf__max_depth': [None] if test else [None, 8]}
    gb_grid = {'gb__n_estimators': [30] if test else [100, 200],
               'gb__learning_rate': [0.05, 0.1],
               'gb__max_depth': [2] if test else [2, 3]}
    cands = {
        'baseline': (BaselineModel(), {}),
        'ridge': (Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]),
                  {'ridge__alpha': [0.1, 1.0, 10.0, 100.0]}),
        'rf': (Pipeline([('scaler', StandardScaler()),
                         ('rf', RandomForestRegressor(random_state=42))]), rf_grid),
        'gb': (Pipeline([('scaler', StandardScaler()),
                         ('gb', GradientBoostingRegressor(random_state=42))]), gb_grid),
    }
    return cands


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _split(n, holdout_frac=0.2):
    """chronological split with a gap of HORIZON rows so train targets never overlap the hold-out"""
    n_hold = max(int(round(n * holdout_frac)), 1)
    train_end = n - n_hold - HORIZON
    return train_end, n - n_hold


def evaluate_models(X, y, test=False):
    """
    compare the candidate models on a chronological hold-out (the last 20% of days).
    returns {name: {'rmse','mae','cv_rmse','best_params','y_pred'}} and the hold-out slice
    """
    train_end, hold_start = _split(len(y))
    if train_end < 60:
        raise ValueError("not enough data to train: {} usable rows".format(len(y)))
    X_tr, y_tr = X.iloc[:train_end], y[:train_end]
    X_ho, y_ho = X.iloc[hold_start:], y[hold_start:]

    results = {}
    for name, (est, grid) in _candidates(test).items():
        if grid:
            search = GridSearchCV(est, grid, cv=TimeSeriesSplit(n_splits=3),
                                  scoring='neg_root_mean_squared_error', n_jobs=1)
            search.fit(X_tr, y_tr)
            fitted, cv_rmse, params = search.best_estimator_, -search.best_score_, search.best_params_
        else:
            fitted = est.fit(X_tr, y_tr)
            cv_rmse = _rmse(y_tr, fitted.predict(X_tr))
            params = {}
        pred = fitted.predict(X_ho)
        results[name] = {'rmse': _rmse(y_ho, pred), 'mae': float(mean_absolute_error(y_ho, pred)),
                         'cv_rmse': float(cv_rmse), 'best_params': params, 'y_pred': pred}
    return results, (hold_start, y_ho)


def _model_train(df, tag, test=False):
    """
    train (compare + select + re-train on all data) and persist the model for one data set.
    """
    time_start = time.time()
    X, y, dates = engineer_features(df)

    results, (hold_start, y_ho) = evaluate_models(X, y, test=test)

    ## choose the best approach by hold-out RMSE (the baseline is eligible too)
    best_name = min(results, key=lambda k: results[k]['rmse'])
    est, grid = _candidates(test)[best_name]
    if grid:
        search = GridSearchCV(est, grid, cv=TimeSeriesSplit(n_splits=3),
                              scoring='neg_root_mean_squared_error', n_jobs=1)
        search.fit(X, y)
        final = search.best_estimator_
    else:
        final = est.fit(X, y)

    comparison = {k: {m: v[m] for m in ('rmse', 'mae', 'cv_rmse')} for k, v in results.items()}
    bundle = {'model': final, 'name': best_name, 'version': MODEL_VERSION, 'tag': tag,
              'comparison': comparison, 'trained_on': (str(dates[0]), str(dates[-1])),
              'train_feature_means': X.mean().to_dict(), 'train_feature_stds': X.std().to_dict(),
              'holdout_rmse': results[best_name]['rmse']}

    os.makedirs(MODEL_DIR, exist_ok=True)
    prefix = "test" if test else "sl"
    saved_model = os.path.join(MODEL_DIR, "{}-{}-{}.joblib".format(prefix, tag, re.sub(r"\.", "_", str(MODEL_VERSION))))
    joblib.dump(bundle, saved_model)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    update_train_log(tag, (str(dates[0]), str(dates[-1])),
                     {'rmse': round(results[best_name]['rmse'], 2),
                      'baseline_rmse': round(results['baseline']['rmse'], 2)},
                     runtime, MODEL_VERSION, MODEL_VERSION_NOTE, model_name=best_name, test=test)
    return bundle


def _data_signature(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".json"))
    return tuple((f, os.path.getsize(os.path.join(data_dir, f)),
                  int(os.path.getmtime(os.path.join(data_dir, f)))) for f in files)


def load_ts(data_dir=None):
    """time-series data for every data set (cached in memory until the files change)"""
    data_dir = data_dir or DATA_DIR
    key = (os.path.abspath(data_dir), _data_signature(data_dir) if os.path.isdir(data_dir) else None)
    if key not in _TS_CACHE:
        _TS_CACHE.clear()
        _TS_CACHE[key] = fetch_ts(data_dir)
    return _TS_CACHE[key]


def model_train(data_dir=None, test=False):
    """
    train a model for every data set found in data_dir (all json files are used, so
    re-training on new data only requires adding files to the directory).
    test=True: subset of the countries, small grids, 'test' prefixed model + test log.
    returns the list of data sets that were trained.
    """
    data_dir = data_dir or DATA_DIR
    ts_data = load_ts(data_dir)

    trained = []
    for country, df in ts_data.items():
        if test and country not in TEST_COUNTRIES:
            continue
        _model_train(df, country, test=test)
        trained.append(country)
    return trained


def model_load(prefix='sl', data_dir=None, training=True):
    """
    load the saved models (files starting with `prefix`) and the feature data.
    training=False returns features for every day (needed to predict recent / future dates)
    """
    data_dir = data_dir or DATA_DIR
    if not os.path.isdir(MODEL_DIR):
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))
    models = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix + "-") and f.endswith(".joblib")]
    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for name in models:
        all_models[name.split("-")[1]] = joblib.load(os.path.join(MODEL_DIR, name))

    all_data = {}
    for country, df in load_ts(data_dir).items():
        X, y, dates = engineer_features(df, training=training)
        all_data[country] = {"X": X, "y": y, "dates": np.array([str(d) for d in dates]),
                            "ts": df}
    return all_data, all_models


def _parse_date(year, month, day):
    for d in (year, month, day):
        if re.search(r"\D", str(d)):
            raise ValueError("invalid year, month or day")
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=int(day)).strftime("%Y-%m-%d")
    except Exception:
        raise ValueError("invalid year, month or day")


def model_predict(country, year, month, day, all_models=None, all_data=None, test=False):
    """
    predict the revenue of the 30 days starting on year-month-day for `country`
    ('all' for every country combined). Only data before that day is used as input.
    """
    time_start = time.time()

    if all_models is None or all_data is None:
        all_data, all_models = model_load(prefix="test" if test else "sl", training=False)

    country = country_key(country)
    if country not in all_models:
        raise ValueError("model for country '{}' could not be found".format(country))
    target_date = _parse_date(year, month, day)

    data = all_data[country]
    dates = data['dates']
    last_known = str(np.datetime64(dates[-1], 'D') + np.timedelta64(1, 'D'))
    if target_date not in dates and target_date != last_known:
        raise ValueError("date {} not in range {} - {}".format(target_date, dates[0], last_known))
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise ValueError("dimensions mismatch")

    if target_date == last_known:
        query = _query_for_next_day(data['ts'])
    else:
        query = data['X'].iloc[[int(np.where(dates == target_date)[0][0])]]

    y_pred = float(all_models[country]['model'].predict(query)[0])
    y_pred = max(y_pred, 0.0)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)
    update_predict_log(country, y_pred, None, target_date, runtime, all_models[country]['version'], test=test)
    return {'y_pred': y_pred, 'y_proba': None, 'country': country, 'target_date': target_date}


def _query_for_next_day(ts):
    """features for the day after the last observed day: extend the series with an empty day"""
    last = pd.to_datetime(ts['date'].iloc[-1]) + pd.Timedelta(days=1)
    extra = pd.DataFrame({'date': [last], 'purchases': [0], 'unique_invoices': [0], 'unique_streams': [0],
                          'total_views': [0], 'revenue': [0.0], 'year_month': [last.strftime("%Y-%m")]})
    X, _, d = engineer_features(pd.concat([ts, extra], ignore_index=True), training=False)
    return X.iloc[[-1]]


if __name__ == "__main__":
    print("TRAINING MODELS (test mode)")
    print(model_train(test=True))
    all_data, all_models = model_load(prefix="test", training=False)
    print("... models loaded:", ",".join(all_models))
    print(model_predict('all', '2018', '01', '05', all_models=all_models, all_data=all_data, test=True))
