# -*- coding: utf-8 -*-
"""
Fair 5-model forecast for the 9000-SKU, 2-year sales data.

This is the apples-to-apples version of the earlier model selection idea:
  - composite sku_id key, so all 9000 SKUs are kept
  - real calendar features for regression models
  - validation model selection and separate future forecast

Outputs:
  - data/forecast_9000_5model_result.json
  - logs/forecast_9000_5model_summary.json
  - logs/forecast_9000_5model.log
"""
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
INPUT_CSV = os.path.join(DATA_DIR, "sales_data_9000_2year.csv")
OUT_JSON = os.path.join(DATA_DIR, "forecast_9000_5model_result.json")
SUMMARY_JSON = os.path.join(LOGS_DIR, "forecast_9000_5model_summary.json")
LOG_PATH = os.path.join(LOGS_DIR, "forecast_9000_5model.log")
FORECAST_DAYS = 30


def calc_mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100)


def calc_rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def make_calendar_features(dates, start_idx=0):
    rows = []
    for pos, d in enumerate(pd.to_datetime(dates)):
        dow = d.weekday()
        month = d.month
        row = [start_idx + pos]
        row.extend(1 if j == dow else 0 for j in range(7))
        row.extend(1 if j == month else 0 for j in range(1, 13))
        row.append(1 if dow >= 5 else 0)
        row.append(1 if month in (1, 2, 3, 11, 12) else 0)
        row.append(1 if month in (6, 7, 8) else 0)
        row.append(
            1
            if (
                (month == 1 and d.day <= 7)
                or (month == 2 and d.day <= 7)
                or (month == 10 and 1 <= d.day <= 7)
                or (month == 12 and d.day >= 25)
            )
            else 0
        )
        rows.append(row)
    return np.asarray(rows, dtype=float)


def predict_ets(train_values, target_len=FORECAST_DAYS):
    try:
        model = ExponentialSmoothing(
            train_values,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        return np.maximum(np.asarray(fit.forecast(target_len), dtype=float), 0)
    except Exception:
        return None


def predict_seasonal_naive(train_values, target_len=FORECAST_DAYS):
    if len(train_values) < 7:
        return None
    return np.asarray([train_values[len(train_values) - 7 + (i % 7)] for i in range(target_len)], dtype=float)


def predict_moving_average(train_values, target_len=FORECAST_DAYS, window=28):
    if len(train_values) == 0:
        return None
    avg = float(np.mean(train_values[-min(window, len(train_values)):]))
    return np.full(target_len, avg)


def predict_date_regression(train_values, train_dates, target_dates, ridge=False):
    try:
        model = Ridge(alpha=1.0) if ridge else LinearRegression()
        x_train = make_calendar_features(train_dates)
        y_train = np.asarray(train_values, dtype=float)
        model.fit(x_train, y_train)
        x_target = make_calendar_features(target_dates, start_idx=len(train_values))
        return np.maximum(model.predict(x_target), 0)
    except Exception:
        return None


def predict_model(name, values, dates, target_dates):
    if name == "ExponentialSmoothing":
        return predict_ets(values, len(target_dates))
    if name == "SeasonalNaive":
        return predict_seasonal_naive(values, len(target_dates))
    if name == "MovingAverage":
        return predict_moving_average(values, len(target_dates))
    if name == "DateLinearRegression":
        return predict_date_regression(values, dates, target_dates, ridge=False)
    if name == "DateRidge":
        return predict_date_regression(values, dates, target_dates, ridge=True)
    return None


def ensemble_from_scores(predictions, scores):
    valid = {k: v for k, v in scores.items() if k in predictions and v < 999}
    if len(valid) < 2:
        return None
    weights = {k: 1.0 / max(v, 1e-6) ** 2 for k, v in valid.items()}
    total = sum(weights.values())
    pred = np.zeros(len(next(iter(predictions.values()))), dtype=float)
    for name, weight in weights.items():
        pred += (weight / total) * predictions[name]
    return np.maximum(pred, 0)


def detect_drift(sales):
    recent = np.asarray(sales[-30:], dtype=float)
    baseline = np.asarray(sales[-150:-30], dtype=float)
    drift_ratio = (recent.mean() - baseline.mean()) / max(baseline.mean(), 1) * 100
    try:
        from scipy import stats

        _, p_value = stats.ttest_ind(recent, baseline, equal_var=False)
    except Exception:
        p_value = 1.0
    return round(float(drift_ratio), 1), bool(abs(drift_ratio) > 20 and p_value < 0.05)


def forecast_one(sku_id, group):
    group = group.sort_values("date")
    sales = group["sales"].to_numpy(dtype=float)
    dates = group["date"].reset_index(drop=True)

    test = sales[-FORECAST_DAYS:]
    train_all = sales[:-FORECAST_DAYS]
    dates_train_all = dates.iloc[:-FORECAST_DAYS]
    dates_test = dates.iloc[-FORECAST_DAYS:]

    fit = train_all[:-FORECAST_DAYS]
    val = train_all[-FORECAST_DAYS:]
    dates_fit = dates_train_all.iloc[:-FORECAST_DAYS]
    dates_val = dates_train_all.iloc[-FORECAST_DAYS:]

    base_models = [
        "ExponentialSmoothing",
        "SeasonalNaive",
        "MovingAverage",
        "DateLinearRegression",
        "DateRidge",
    ]

    val_preds = {}
    val_scores = {}
    for name in base_models:
        pred = predict_model(name, fit, dates_fit, dates_val)
        if pred is not None and len(pred) == len(val):
            val_preds[name] = pred
            val_scores[name] = calc_mape(val, pred)

    ensemble_val = ensemble_from_scores(val_preds, val_scores)
    if ensemble_val is not None:
        val_preds["Ensemble"] = ensemble_val
        val_scores["Ensemble"] = calc_mape(val, ensemble_val)

    if not val_scores:
        best_model = "MovingAverage"
    else:
        best_model = min(val_scores, key=val_scores.get)

    full_preds = {}
    for name in base_models:
        pred = predict_model(name, train_all, dates_train_all, dates_test)
        if pred is not None and len(pred) == len(test):
            full_preds[name] = pred
    if best_model == "Ensemble":
        backtest_pred = ensemble_from_scores(full_preds, val_scores)
        if backtest_pred is None:
            best_model = min(full_preds, key=lambda n: val_scores.get(n, 999))
            backtest_pred = full_preds[best_model]
    else:
        backtest_pred = full_preds.get(best_model)
    if backtest_pred is None:
        best_model = "MovingAverage"
        backtest_pred = predict_moving_average(train_all, len(test))

    future_dates = pd.date_range(dates.iloc[-1] + timedelta(days=1), periods=FORECAST_DAYS)
    future_base_preds = {}
    for name in base_models:
        pred = predict_model(name, sales, dates, future_dates)
        if pred is not None and len(pred) == FORECAST_DAYS:
            future_base_preds[name] = pred
    if best_model == "Ensemble":
        future_pred = ensemble_from_scores(future_base_preds, val_scores)
        if future_pred is None:
            future_pred = future_base_preds.get("DateLinearRegression")
    else:
        future_pred = future_base_preds.get(best_model)
    if future_pred is None:
        future_pred = predict_moving_average(sales, FORECAST_DAYS)

    future_pred = np.maximum(future_pred, 0)
    total_forecast = int(round(float(future_pred.sum())))
    avg_daily = float(future_pred.mean())
    safety_stock = int(round(avg_daily * 7))
    recommended_stock = int(total_forecast + safety_stock)
    drift_ratio, is_drift = detect_drift(sales)

    drug_name = str(group["drug_name"].iloc[0])
    category = str(group["category"].iloc[0])
    sub_category = str(group["sub_category"].iloc[0])

    return {
        "sku_id": sku_id,
        "drug_name": drug_name,
        "display_name": f"{drug_name} ({sub_category})",
        "category": category,
        "sub_category": sub_category,
        "model_used": best_model,
        "model_scores": {k: round(float(v), 1) for k, v in val_scores.items()},
        "mape": round(calc_mape(test, backtest_pred), 1),
        "rmse": round(calc_rmse(test, backtest_pred), 2),
        "total_forecast_30days": total_forecast,
        "avg_daily": round(avg_daily, 1),
        "safety_stock": safety_stock,
        "recommended_stock": recommended_stock,
        "is_drift": is_drift,
        "drift_ratio": drift_ratio,
        "history_dates": dates.dt.strftime("%Y-%m-%d").tolist()[-60:],
        "history_sales": [int(round(x)) for x in sales[-60:]],
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_values": [int(round(x)) for x in future_pred],
        "test_actual": [int(round(x)) for x in test],
        "backtest_forecast_values": [int(round(x)) for x in backtest_pred],
        "backtest_dates": dates_test.dt.strftime("%Y-%m-%d").tolist(),
    }


def summarize(results, elapsed):
    valid = [v for v in results.values() if v.get("mape", 999) < 999]
    mapes = np.asarray([v["mape"] for v in valid], dtype=float)
    model_dist = Counter(v.get("model_used", "unknown") for v in valid)
    by_category = defaultdict(list)
    for item in valid:
        by_category[item["category"]].append(item["mape"])
    category_stats = {
        cat: {
            "count": len(vals),
            "avg_mape": round(float(np.mean(vals)), 2),
            "median_mape": round(float(np.median(vals)), 2),
        }
        for cat, vals in sorted(by_category.items())
    }

    return {
        "input_csv": INPUT_CSV,
        "output_json": OUT_JSON,
        "total_products": len(results),
        "valid_products": len(valid),
        "elapsed_seconds": round(float(elapsed), 1),
        "avg_mape": round(float(mapes.mean()), 3),
        "median_mape": round(float(np.median(mapes)), 3),
        "p90_mape": round(float(np.percentile(mapes, 90)), 3),
        "mape_dist": {
            "lt_10": int((mapes < 10).sum()),
            "10_15": int(((mapes >= 10) & (mapes < 15)).sum()),
            "15_25": int(((mapes >= 15) & (mapes < 25)).sum()),
            "25_35": int(((mapes >= 25) & (mapes < 35)).sum()),
            "gte_35": int((mapes >= 35).sum()),
        },
        "model_dist": dict(model_dist),
        "category_stats": category_stats,
    }


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    log = open(LOG_PATH, "w", encoding="utf-8")

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            log.write(text)
            log.flush()

        def flush(self):
            sys.__stdout__.flush()
            log.flush()

    sys.stdout = Tee()
    sys.stderr = Tee()
    t0 = time.time()

    print("=" * 60)
    print("  9000-SKU fair 5-model forecast")
    print("=" * 60)
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df["sku_id"] = (
        df["category"].astype(str)
        + "|"
        + df["sub_category"].astype(str)
        + "|"
        + df["drug_name"].astype(str)
    )
    print(
        f"Loaded: {len(df):,} rows, {df['sku_id'].nunique():,} sku_id, "
        f"{df['drug_name'].nunique():,} unique drug_name"
    )

    results = {}
    grouped = df.groupby("sku_id", sort=True)
    total = grouped.ngroups
    for i, (sku_id, group) in enumerate(grouped, 1):
        try:
            results[sku_id] = forecast_one(sku_id, group)
        except Exception as exc:
            results[sku_id] = {"sku_id": sku_id, "error": str(exc)[:200], "mape": 999}
        if i % 500 == 0 or i == total:
            elapsed = time.time() - t0
            done = [v for v in results.values() if v.get("mape", 999) < 999]
            avg = np.mean([v["mape"] for v in done]) if done else 0
            print(f"Progress: {i:,}/{total:,}, avg_mape_so_far={avg:.2f}%, elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    summary = summarize(results, elapsed)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  5-model forecast done")
    print("=" * 60)
    print(f"Products: {summary['valid_products']}/{summary['total_products']}")
    print(
        f"MAPE avg={summary['avg_mape']}%, "
        f"median={summary['median_mape']}%, "
        f"p90={summary['p90_mape']}%"
    )
    print(f"MAPE dist: {summary['mape_dist']}")
    print(f"Model dist: {summary['model_dist']}")
    print(f"Saved result: {OUT_JSON}")
    print(f"Saved summary: {SUMMARY_JSON}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log.close()


if __name__ == "__main__":
    main()
