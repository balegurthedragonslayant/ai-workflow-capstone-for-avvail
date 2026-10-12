#!/usr/bin/env python
"""
logger.py - train / predict logging.

Log files are csv files in LOG_DIR (default ./logs, override with the AAVAIL_LOG_DIR env var).
Every function takes test=True to write to separate 'test' log files so that unit tests
never touch the production logs.
"""

import os
import csv
import time
import uuid
from datetime import date

LOG_DIR = os.environ.get("AAVAIL_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

TRAIN_HEADER = ['unique_id', 'timestamp', 'tag', 'date_range', 'eval_metrics', 'model_name',
                'runtime', 'model_version', 'model_version_note']
PREDICT_HEADER = ['unique_id', 'timestamp', 'country', 'y_pred', 'y_proba', 'target_date',
                  'runtime', 'model_version']


def _log_path(kind, test):
    """kind is 'train' or 'predict'; one file per month for production, a single file for tests"""
    os.makedirs(LOG_DIR, exist_ok=True)
    if test:
        return os.path.join(LOG_DIR, "{}-test.log".format(kind))
    today = date.today()
    return os.path.join(LOG_DIR, "{}-{}-{}.log".format(kind, today.year, today.month))


def _append(path, header, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def update_train_log(tag, date_range, eval_metrics, runtime, model_version, model_version_note,
                     model_name="", test=False):
    """update the train log file"""
    row = [uuid.uuid4(), time.time(), tag, date_range, eval_metrics, model_name,
           runtime, model_version, model_version_note]
    _append(_log_path('train', test), TRAIN_HEADER, row)


def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    """update the predict log file"""
    row = [uuid.uuid4(), time.time(), country, y_pred, y_proba, target_date, runtime, model_version]
    _append(_log_path('predict', test), PREDICT_HEADER, row)


def list_logs(test=False):
    """names of the available log files"""
    if not os.path.isdir(LOG_DIR):
        return []
    return sorted(f for f in os.listdir(LOG_DIR)
                  if f.endswith(".log") and (("-test" in f) == bool(test)))


def read_log(file_name):
    """return a log file as a list of dicts (raises FileNotFoundError if it does not exist)"""
    path = os.path.join(LOG_DIR, os.path.basename(file_name))
    if not os.path.exists(path):
        raise FileNotFoundError(file_name)
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=',', quotechar='|'))


if __name__ == "__main__":
    update_train_log('demo', ('2018-01-01', '2018-12-31'), {'rmse': 0.5}, "00:00:01", 0.1, "demo", test=True)
    update_predict_log('all', [1.0], None, '2018-01-05', "00:00:01", 0.1, test=True)
    print(list_logs(test=True))
