#!/usr/bin/env python
"""
monitoring.py - post-production performance monitoring.

1. simulate_production(): replay the production period day by day. For every day the models only see
   data BEFORE that day, predict the next 30 days through model_predict (so predictions land in the
   predict log) and the result is compared to the revenue that was actually observed later.
2. performance_report(): RMSE / MAPE of the model vs the baseline per data set.
3. drift_report(): compares the distribution of the features seen in production with the training
   distribution (z-score of the mean shift) - an early warning that the model needs re-training.
4. plots written to ./figures/monitoring-*.png

usage:  python monitoring.py            (uses cs-train + cs-production, writes reports/monitoring.json)

The functions are also exposed by the API (GET /monitor).
"""
import os
import json
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model
from cslib import fetch_ts, engineer_features

BASE_DIR = model.BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, "figures")
PROD_DIR = os.environ.get("AAVAIL_PROD_DIR", os.path.join(BASE_DIR, "cs-production"))
DRIFT_Z_THRESHOLD = 3.0


def _combined_dir(*dirs):
    """a temp directory with links to every json file of the given dirs (train + new production data)"""
    tmp = tempfile.mkdtemp(prefix="aavail-data-")
    for d in dirs:
        for f in os.listdir(d):
            if f.lower().endswith(".json"):
                os.symlink(os.path.abspath(os.path.join(d, f)), os.path.join(tmp, f))
    return tmp


def simulate_production(train_dir=None, prod_dir=None, countries=('all', 'united_kingdom'),
                        model_dir=None, test=False, max_days=None):
    """
    replay production. Returns a DataFrame [country, date, y_pred, y_true, y_base].
    y_true is NaN for the last 29 days of the data (target not fully observed yet).
    """
    train_dir = train_dir or model.DATA_DIR
    prod_dir = prod_dir or PROD_DIR
    if model_dir:
        model.MODEL_DIR = model_dir

    all_models_cache = None
    combo = _combined_dir(train_dir, prod_dir)
    try:
        all_data, all_models = model.model_load(prefix="test" if test else "sl", data_dir=combo, training=False)
        train_ts = model.load_ts(train_dir)
        rows = []
        for country in countries:
            ts_end = pd.to_datetime(train_ts[country]['date'].iloc[-1])
            data = all_data[country]
            dates = pd.to_datetime(data['dates'])
            ts = data['ts'].set_index('date')['revenue']
            days = [d for d in dates if d > ts_end]
            if max_days:
                days = days[:max_days]
            for d in days:
                res = model.model_predict(country, d.year, d.month, d.day, all_models=all_models,
                                          all_data=all_data, test=test)
                window = ts[d:d + pd.Timedelta(days=29)]
                y_true = float(window.sum()) if len(window) == 30 else np.nan
                base = float(ts[d - pd.Timedelta(days=28):d - pd.Timedelta(days=1)].sum()) * 30 / 28
                rows.append({'country': country, 'date': d, 'y_pred': res['y_pred'],
                             'y_true': y_true, 'y_base': base})
        return pd.DataFrame(rows)
    finally:
        shutil.rmtree(combo, ignore_errors=True)


def performance_report(sim):
    """RMSE / MAPE for model and baseline (only days where the actual 30 day revenue is known)"""
    out = {}
    for country, g in sim.dropna(subset=['y_true']).groupby('country'):
        rm = lambda p: float(np.sqrt(np.mean((g['y_true'] - p) ** 2)))
        mape = lambda p: float(np.mean(np.abs(g['y_true'] - p) / np.maximum(np.abs(g['y_true']), 1)) * 100)
        out[country] = {'n_days': int(len(g)),
                        'model_rmse': rm(g['y_pred']), 'baseline_rmse': rm(g['y_base']),
                        'model_mape_pct': mape(g['y_pred']), 'baseline_mape_pct': mape(g['y_base']),
                        'beats_baseline': bool(rm(g['y_pred']) < rm(g['y_base']))}
    return out


def drift_report(train_dir=None, prod_dir=None, countries=('all', 'united_kingdom')):
    """
    z-score of the mean shift of every engineered feature between training and production
    (|z| above DRIFT_Z_THRESHOLD flags the feature as drifting).
    """
    train_dir = train_dir or model.DATA_DIR
    prod_dir = prod_dir or PROD_DIR
    combo = _combined_dir(train_dir, prod_dir)
    try:
        train_ts, all_ts = model.load_ts(train_dir), fetch_ts(combo)
    finally:
        shutil.rmtree(combo, ignore_errors=True)

    out = {}
    for country in countries:
        Xt, _, dt = engineer_features(train_ts[country], training=False)
        Xa, _, da = engineer_features(all_ts[country], training=False)
        Xp = Xa[da > dt[-1]]
        feats = {}
        for c in Xt.columns:
            sd = Xt[c].std() or 1.0
            z = float((Xp[c].mean() - Xt[c].mean()) / (sd / np.sqrt(max(len(Xp), 1))))
            feats[c] = {'train_mean': float(Xt[c].mean()), 'prod_mean': float(Xp[c].mean()),
                        'z': z, 'drift': bool(abs(z) > DRIFT_Z_THRESHOLD)}
        out[country] = feats
    return out


def plot_performance(sim, report):
    os.makedirs(FIG_DIR, exist_ok=True)
    countries = list(sim['country'].unique())
    fig, axes = plt.subplots(len(countries), 1, figsize=(10, 3.8 * len(countries)))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, countries):
        g = sim[sim.country == c]
        ax.plot(g['date'], g['y_true'], color="black", lw=2, label="actual 30-day revenue")
        ax.plot(g['date'], g['y_pred'], color="#0072B2", lw=1.8,
                label="model (RMSE {:,.0f})".format(report[c]['model_rmse']))
        ax.plot(g['date'], g['y_base'], color="#6b7280", lw=1.4, ls="--",
                label="baseline (RMSE {:,.0f})".format(report[c]['baseline_rmse']))
        ax.set_title("Production monitoring - " + c, loc="left", fontweight="bold")
        ax.set_ylabel("30-day revenue")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", color="#e5e7eb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "monitoring-performance.png"), dpi=130)
    plt.close(fig)


def main():
    sim = simulate_production()
    perf = performance_report(sim)
    drift = drift_report()
    plot_performance(sim, perf)
    os.makedirs(model.REPORT_DIR, exist_ok=True)
    sim.to_csv(os.path.join(model.REPORT_DIR, "production_predictions.csv"), index=False)
    with open(os.path.join(model.REPORT_DIR, "monitoring.json"), "w") as f:
        json.dump({'performance': perf, 'drift': drift}, f, indent=2)
    print(json.dumps(perf, indent=2))
    flagged = {c: [k for k, v in d.items() if v['drift']] for c, d in drift.items()}
    print("drifting features:", flagged)
    return perf, drift


if __name__ == "__main__":
    main()
