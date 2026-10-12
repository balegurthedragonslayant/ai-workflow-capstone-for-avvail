#!/usr/bin/env python
"""
model_comparison.py - compare the candidate models to the baseline on a chronological hold-out
and write
    reports/model_comparison.csv
    figures/model-comparison-rmse.png     (RMSE per model for every data set)
    figures/model-vs-baseline.png         (hold-out time-series: actual vs model vs baseline)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model
from cslib import engineer_features

FIG_DIR = os.path.join(model.BASE_DIR, "figures")
COLORS = {"baseline": "#6b7280", "ridge": "#0072B2", "rf": "#009E73", "gb": "#D55E00"}


def run(data_dir=None):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(model.REPORT_DIR, exist_ok=True)
    ts = model.load_ts(data_dir)

    rows, holdouts = [], {}
    for country, df in ts.items():
        X, y, dates = engineer_features(df)
        results, (start, y_ho) = model.evaluate_models(X, y)
        best = min(results, key=lambda k: results[k]['rmse'])
        for name, r in results.items():
            rows.append({'data_set': country, 'model': name, 'holdout_rmse': r['rmse'],
                         'holdout_mae': r['mae'], 'cv_rmse': r['cv_rmse'], 'selected': name == best,
                         'improvement_vs_baseline_pct': 100 * (1 - r['rmse'] / results['baseline']['rmse'])})
        holdouts[country] = (dates[start:], y_ho, {k: v['y_pred'] for k, v in results.items()}, best)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(model.REPORT_DIR, "model_comparison.csv"), index=False)

    ## figure 1: RMSE relative to baseline for every data set (1.0 = baseline) - log scale
    pivot = table.pivot(index='data_set', columns='model', values='holdout_rmse')
    rel = pivot.div(pivot['baseline'], axis=0)[['ridge', 'rf', 'gb']]
    rel = rel.loc[list(ts.keys())]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.26
    x = np.arange(len(rel))
    for i, name in enumerate(rel.columns):
        ax.bar(x + (i - 1) * w, rel[name].clip(upper=10), w, label=name, color=COLORS[name])
    ax.axhline(1.0, color=COLORS['baseline'], lw=2)
    ax.text(len(rel) - 0.5, 1.05, "baseline = 1.0", color=COLORS['baseline'], ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(rel.index, rotation=30, ha="right")
    ax.set_ylabel("hold-out RMSE / baseline RMSE  (lower is better)")
    ax.set_title("Models vs baseline - bars below the line beat the baseline (capped at 10x)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model-comparison-rmse.png"), dpi=130)
    plt.close(fig)

    ## figure 2: hold-out predictions vs actual vs baseline for 'all' and 'united_kingdom'
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, key in zip(axes, ['all', 'united_kingdom']):
        d, y_ho, preds, best = holdouts[key]
        ax.plot(d, y_ho, color="black", lw=2, label="actual 30-day revenue")
        ax.plot(d, preds['baseline'], color=COLORS['baseline'], lw=1.6, ls="--",
                label="baseline (RMSE {:,.0f})".format(table.query("data_set==@key and model=='baseline'").holdout_rmse.iloc[0]))
        ax.plot(d, preds[best], color=COLORS[best], lw=1.8,
                label="{} - selected (RMSE {:,.0f})".format(best, table.query("data_set==@key and model==@best").holdout_rmse.iloc[0]))
        ax.set_title("Hold-out: " + key, loc="left", fontweight="bold")
        ax.set_ylabel("30-day revenue")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", color="#e5e7eb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model-vs-baseline.png"), dpi=130)
    plt.close(fig)
    return table


if __name__ == "__main__":
    t = run()
    print(t.pivot(index='data_set', columns='model', values='holdout_rmse').round(0))
    print(t[t.selected][['data_set', 'model', 'holdout_rmse', 'improvement_vs_baseline_pct']].round(1))
