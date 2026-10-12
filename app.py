#!/usr/bin/env python
"""
app.py - Flask API

  GET  /                      health check
  POST /train                 json {"test": false}  re-train all models on every file in the data dir
  POST /predict               json {"country": "all", "date": "2019-08-15"}   (or year/month/day)
                              -> {"y_pred": ..., "country": ..., "target_date": ...}
  GET  /logs                  list of log files  (?test=true for the test logs)
  GET  /logs/<file_name>      content of a log file as json
  GET  /monitor               performance of the deployed models (train log) + data drift

country is 'all' (every country combined) or one of the top-10 countries, e.g. 'united_kingdom'.
"""
import os
import argparse

from flask import Flask, jsonify, request

import model
import logger

app = Flask(__name__)

_CACHE = {}   # loaded models + feature data, refreshed after /train


def _loaded(test=False):
    """models + feature data; reloaded after /train or when new data files appear in the data dir
    (so new months of data can be predicted on without re-training)"""
    key = 'test' if test else 'sl'
    sig = model._data_signature(model.DATA_DIR)
    if key not in _CACHE or _CACHE[key][2] != sig:
        data, models = model.model_load(prefix=key, training=False)
        _CACHE[key] = (data, models, sig)
    return _CACHE[key][:2]


def _error(msg, code=400):
    return jsonify({"error": str(msg)}), code


@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "aavail revenue forecast", "status": "ok",
                    "endpoints": ["/train", "/predict", "/logs", "/monitor"]})


@app.route("/train", methods=["POST"])
def train():
    body = request.get_json(silent=True) or {}
    test = bool(body.get("test", False))
    try:
        trained = model.model_train(data_dir=body.get("data_dir"), test=test)
    except Exception as e:
        return _error(e, 500)
    _CACHE.pop('test' if test else 'sl', None)
    return jsonify({"trained": trained, "test": test, "model_version": model.MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or 'country' not in body:
        return _error("request must be json with a 'country' and a 'date' (YYYY-MM-DD)")
    test = bool(body.get("test", False))

    try:
        if 'date' in body:
            parts = str(body['date']).split("-")
            if len(parts) != 3:
                raise ValueError("date must have the format YYYY-MM-DD")
            year, month, day = parts
        elif all(k in body for k in ('year', 'month', 'day')):
            year, month, day = body['year'], body['month'], body['day']
        else:
            raise ValueError("provide 'date' (YYYY-MM-DD) or 'year', 'month' and 'day'")

        data, models = _loaded(test)
        result = model.model_predict(body['country'], year, month, day,
                                     all_models=models, all_data=data, test=test)
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:
        return _error(e, 500)
    return jsonify(result)


@app.route("/logs", methods=["GET"])
def logs():
    test = request.args.get("test", "false").lower() == "true"
    return jsonify(logger.list_logs(test=test))


@app.route("/logs/<file_name>", methods=["GET"])
def log_file(file_name):
    try:
        return jsonify(logger.read_log(file_name))
    except FileNotFoundError:
        return _error("log file '{}' does not exist".format(file_name), 404)


@app.route("/monitor", methods=["GET"])
def monitor():
    """latest training metrics per data set + data drift check (needs ./cs-production data)"""
    out = {"train_log": []}
    for name in logger.list_logs():
        if name.startswith("train"):
            out["train_log"] += logger.read_log(name)[-20:]
    try:
        import monitoring
        out["drift"] = monitoring.drift_report()
    except Exception as e:
        out["drift"] = {"unavailable": str(e)}
    return jsonify(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug mode")
    ap.add_argument("-p", "--port", type=int, default=8080)
    args = ap.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
