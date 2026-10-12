#!/usr/bin/env python
"""
eda.py - exploratory data analysis. Reads the raw data through cslib (the ingestion function)
and writes the figures used in the report to ./figures/eda-*.png

usage: python eda.py [data_dir]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cslib import fetch_data, convert_to_ts, country_key

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
C1, C2, C3, GRID = "#0072B2", "#D55E00", "#6b7280", "#e5e7eb"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def run(data_dir):
    os.makedirs(FIG_DIR, exist_ok=True)
    df = fetch_data(data_dir)
    out = {}

    ## 1. revenue by country (top 10) - shows how dominant the UK is
    by_country = df.groupby('country')['price'].sum().sort_values(ascending=False)
    top = by_country.head(10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(top.index[::-1], top.values[::-1], color=C1)
    ax.set_xscale("log")
    _style(ax, "Total revenue by country (top 10, log scale)", "revenue (log)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    for i, v in enumerate(top.values[::-1]):
        ax.text(v * 1.05, i, "{:,.0f}".format(v), va="center", fontsize=8, color=C3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda-revenue-by-country.png"), dpi=130)
    plt.close(fig)
    out['uk_share'] = float(by_country['United Kingdom'] / by_country.sum())

    ## 2. daily revenue, all countries + UK, with 28-day rolling mean
    ts_all = convert_to_ts(df)
    ts_uk = convert_to_ts(df, country="United Kingdom")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, ts, name in ((axes[0], ts_all, "All countries"), (axes[1], ts_uk, "United Kingdom")):
        ax.plot(ts['date'], ts['revenue'], color=C1, lw=0.7, alpha=0.5, label="daily revenue")
        ax.plot(ts['date'], ts['revenue'].rolling(28, min_periods=1).mean(), color=C2, lw=2,
                label="28-day mean")
        _style(ax, "Daily revenue - " + name, ylabel="revenue")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda-daily-revenue.png"), dpi=130)
    plt.close(fig)

    ## 3. monthly revenue (seasonality / trend)
    monthly = ts_all.groupby('year_month')['revenue'].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.index, monthly.values, color=C1)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=70, fontsize=8)
    _style(ax, "Monthly revenue - strong autumn peak (Oct-Nov)", ylabel="revenue")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda-monthly-revenue.png"), dpi=130)
    plt.close(fig)

    ## 4. data quality: distribution of invoice-line prices (returns / outliers) and views
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    p = df['price'].clip(-50, 100)
    axes[0].hist(p, bins=100, color=C1)
    axes[0].set_yscale("log")
    _style(axes[0], "Invoice line price (clipped -50..100)", "price", "lines (log)")
    v = df['times_viewed'].clip(upper=60)
    axes[1].hist(v, bins=60, color=C2)
    axes[1].set_yscale("log")
    _style(axes[1], "Times viewed per line (clipped at 60)", "times viewed", "lines (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda-distributions.png"), dpi=130)
    plt.close(fig)
    out['negative_price_lines'] = int((df['price'] < 0).sum())
    out['missing_customer_id'] = int(df['customer_id'].isna().sum())

    ## 5. day-of-week pattern and correlation of daily features with revenue
    dow = ts_all.assign(dow=ts_all['date'].dt.day_name()).groupby('dow')['revenue'].mean()
    dow = dow.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([d[:3] for d in dow.index], dow.values, color=C1)
    _style(axes[0], "Mean daily revenue by weekday", ylabel="revenue")
    cols = ['purchases', 'unique_invoices', 'unique_streams', 'total_views', 'revenue']
    corr = ts_all[cols].corr()
    im = axes[1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(cols)))
    axes[1].set_yticks(range(len(cols)))
    axes[1].set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    axes[1].set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            axes[1].text(j, i, "{:.2f}".format(corr.values[i, j]), ha="center", va="center", fontsize=7)
    axes[1].set_title("Correlation of daily metrics", loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda-weekday-correlation.png"), dpi=130)
    plt.close(fig)

    ## 6. top-10 countries as share of monthly revenue over time (small multiples)
    names = list(top.index[1:7])
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        t = convert_to_ts(df, country=name)
        ax.plot(t['date'], t['revenue'].rolling(28, min_periods=1).mean(), color=C1, lw=1.5)
        _style(ax, name)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.suptitle("28-day mean daily revenue, next largest countries", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda-other-countries.png"), dpi=130)
    plt.close(fig)

    return out


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "cs-train")
    print(run(data_dir))
