#!/usr/bin/env python
"""
cslib.py - data ingestion and feature engineering for the AAVAIL capstone.

Public functions
    fetch_data(data_dir)           -> raw invoice-level DataFrame (all json files, cleaned)
    convert_to_ts(df, country)     -> daily time-series DataFrame (one row per calendar day)
    fetch_ts(data_dir)             -> dict {'all': ts, 'united_kingdom': ts, ...} (top-N countries + all)
    engineer_features(ts, training)-> X, y, dates  (y = revenue of the next 30 days)

The ingestion step is designed to be automated: drop new monthly json files in the
data directory and call fetch_ts() again (this is exactly what the /train endpoint does).
"""

import os
import re
import json
import time

import numpy as np
import pandas as pd

TOP_N_COUNTRIES = 10
HORIZON = 30                      # days ahead that we forecast (30 day revenue)
LOOKBACK_WINDOWS = [7, 14, 28, 70]  # days of revenue history used as features

FINAL_COLUMNS = ['country', 'customer_id', 'day', 'invoice', 'month',
                 'price', 'stream_id', 'times_viewed', 'year']

## the different spellings used across the monthly json files
COLUMN_ALIASES = {'StreamID': 'stream_id',
                  'TimesViewed': 'times_viewed',
                  'total_price': 'price'}


def country_key(name):
    """'United Kingdom' -> 'united_kingdom' (the key used for models / api)"""
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _read_one(file_name):
    """read one monthly json file and harmonise its column names"""
    with open(file_name) as f:
        try:
            records = json.load(f)
        except ValueError:
            raise ValueError("{}: not a valid json file".format(os.path.basename(file_name)))
    df = pd.DataFrame.from_records(records)
    for col in ('invoice', 'StreamID', 'stream_id'):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    missing = set(FINAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError("{}: could not match columns, missing {}".format(
            os.path.basename(file_name), sorted(missing)))
    return df[FINAL_COLUMNS]


def fetch_data(data_dir):
    """
    load all json formatted files in data_dir into a single DataFrame

    - handles the non-uniform column names (StreamID / stream_id, total_price / price ...)
    - strips letters from invoice ids (they are used for matching)
    - builds an 'invoice_date' column and drops rows that cannot be dated
    - raises a clear Exception for the common input errors (missing dir, no json, bad columns)
    """
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist: {}".format(data_dir))

    file_list = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                       if f.lower().endswith(".json"))
    if len(file_list) == 0:
        raise Exception("specified data dir does not contain any json files: {}".format(data_dir))

    df = pd.concat([_read_one(f) for f in file_list], ignore_index=True)

    ## dates
    date_str = (df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2) + "-" +
                df['day'].astype(str).str.zfill(2))
    df['invoice_date'] = pd.to_datetime(date_str, errors='coerce')
    df = df[df['invoice_date'].notna()].copy()

    ## numeric clean-up
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['times_viewed'] = pd.to_numeric(df['times_viewed'], errors='coerce')
    df = df[df['price'].notna()].copy()

    ## invoice ids: remove the letters (e.g. 'C489449' -> '489449')
    df['invoice'] = df['invoice'].astype(str).str.replace(r"\D+", "", regex=True)

    df = df.sort_values('invoice_date', kind='stable').reset_index(drop=True)
    return df


def convert_to_ts(df_orig, country=None):
    """
    aggregate the invoice level DataFrame to one row per calendar day.
    Days without any purchase are kept (with zeros) so the series has no gaps.
    """
    if country:
        key = country_key(country)
        names = [c for c in df_orig['country'].unique() if country_key(c) == key]
        if len(names) == 0:
            raise Exception("country not found: {}".format(country))
        df = df_orig[df_orig['country'].isin(names)]
    else:
        df = df_orig

    grp = df.groupby('invoice_date')
    daily = pd.DataFrame({
        'purchases': grp.size(),
        'unique_invoices': grp['invoice'].nunique(),
        'unique_streams': grp['stream_id'].nunique(),
        'total_views': grp['times_viewed'].sum(),
        'revenue': grp['price'].sum(),
    })

    ## use the full range of the whole data set so every country has the same calendar
    full_range = pd.date_range(df_orig['invoice_date'].min(), df_orig['invoice_date'].max(), freq='D')
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = 'date'
    daily = daily.reset_index()
    daily['year_month'] = daily['date'].dt.strftime("%Y-%m")
    return daily


def fetch_ts(data_dir, n_countries=TOP_N_COUNTRIES):
    """
    returns a dictionary of daily time-series data frames:
    'all' plus one entry for each of the top-N countries (by revenue)
    """
    df = fetch_data(data_dir)
    revenue = df.groupby('country')['price'].sum().sort_values(ascending=False)
    top = list(revenue.index[:n_countries])

    dfs = {'all': convert_to_ts(df)}
    for country in top:
        dfs[country_key(country)] = convert_to_ts(df, country=country)
    return dfs


def engineer_features(df, training=True, include_previous_year=False):
    """
    for any given day the target is the revenue summed over that day and the following 29 days.
    Features only use information from BEFORE that day:
        previous_7/14/28/70   revenue summed over the previous N days
        previous_year         (optional, off by default) revenue over the same 30 days one year earlier.
                              Off because the data only spans ~19 months: the feature is 0 for the first
                              year of training and populated afterwards, which shifts its distribution and
                              made every model much worse than the baseline on the hold-out (see README).
        recent_invoices       mean daily unique invoices over the previous 30 days
        recent_views          mean daily views over the previous 30 days

    training=True drops the last 30 days (target not fully observed);
    training=False returns every day (used to build the query for prediction)
    """
    ts = df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(ts['date']).values.astype('datetime64[D]')
    n = len(ts)

    rev = ts['revenue'].values.astype(float)
    inv = ts['unique_invoices'].values.astype(float)
    views = ts['total_views'].values.astype(float)

    csum = lambda a: np.concatenate([[0.0], np.cumsum(a)])   # csum[i] = sum(a[:i])
    c_rev, c_inv, c_views = csum(rev), csum(inv), csum(views)
    idx = np.arange(n)

    def window_sum(c, lo, hi):
        """sum over positions [lo, hi) with clipping at the edges"""
        lo = np.clip(lo, 0, n)
        hi = np.clip(hi, 0, n)
        return np.where(hi > lo, c[hi] - c[lo], 0.0)

    feats = {}
    for w in LOOKBACK_WINDOWS:
        feats["previous_{}".format(w)] = window_sum(c_rev, idx - w, idx)

    if include_previous_year:
        feats['previous_year'] = window_sum(c_rev, idx - 365, idx - 365 + HORIZON)

    ## mean over the 30 previous days (divide by days that exist, like pandas .mean() on the slice)
    n_days = np.clip(np.minimum(idx, 30), 1, None)
    feats['recent_invoices'] = window_sum(c_inv, idx - 30, idx) / n_days
    feats['recent_views'] = window_sum(c_views, idx - 30, idx) / n_days

    X = pd.DataFrame(feats)
    y = window_sum(c_rev, idx, idx + HORIZON)

    ## remove the rows without any history at all
    mask = (X.sum(axis=1) > 0).values
    X, y, dates = X[mask].reset_index(drop=True), y[mask], dates[mask]

    if training:
        keep = np.arange(X.shape[0]) < X.shape[0] - HORIZON
        X, y, dates = X[keep].reset_index(drop=True), y[keep], dates[keep]

    return X, y, dates


if __name__ == "__main__":
    run_start = time.time()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs-train")
    ts_all = fetch_ts(data_dir)
    print("load time: {:.1f}s".format(time.time() - run_start))
    for key, item in ts_all.items():
        print(key, item.shape)
