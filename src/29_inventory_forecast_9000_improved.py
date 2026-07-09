# -*- coding: utf-8 -*-
"""
Improved 9000-SKU inventory forecast.

This script keeps the old forecast files intact and writes:
  - data/forecast_9000_improved_result.json
  - logs/forecast_9000_improved_summary.json
  - logs/forecast_9000_improved.log

Main fixes compared with the earlier 9000 result:
  1. Use a composite sku_id instead of drug_name as the unique key.
  2. Use the real calendar dates for regression features.
  3. Separate backtest accuracy from the true future 30-day forecast.
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
OUT_JSON = os.path.join(DATA_DIR, "forecast_9000_improved_result.json")
SUMMARY_JSON = os.path.join(LOGS_DIR, "forecast_9000_improved_summary.json")
LOG_PATH = os.path.join(LOGS_DIR, "forecast_9000_improved.log")
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


def fit_predict_linear(train_values, train_dates, target_dates, ridge=False):
    model = Ridge(alpha=1.0) if ridge else LinearRegression()
    x_train = make_calendar_features(train_dates)
    y_train = np.asarray(train_values, dtype=float)
    model.fit(x_train, y_train)
    x_target = make_calendar_features(target_dates, start_idx=len(train_values))
    return np.maximum(model.predict(x_target), 0)


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
    train_for_backtest = sales[:-FORECAST_DAYS]
    dates_train_backtest = dates.iloc[:-FORECAST_DAYS]
    dates_test = dates.iloc[-FORECAST_DAYS:]

    pred_lr_backtest = fit_predict_linear(
        train_for_backtest, dates_train_backtest, dates_test, ridge=False
    )
    pred_ridge_backtest = fit_predict_linear(
        train_for_backtest, dates_train_backtest, dates_test, ridge=True
    )

    mape_lr = calc_mape(test, pred_lr_backtest)
    mape_ridge = calc_mape(test, pred_ridge_backtest)
    if mape_ridge + 0.05 < mape_lr:
        model_used = "DateRidge"
        backtest_pred = pred_ridge_backtest
        mape_value = mape_ridge
    else:
        model_used = "DateLinearRegression"
        backtest_pred = pred_lr_backtest
        mape_value = mape_lr

    future_dates = pd.date_range(dates.iloc[-1] + timedelta(days=1), periods=FORECAST_DAYS)
    future_pred = fit_predict_linear(sales, dates, future_dates, ridge=(model_used == "DateRidge"))
    future_pred = np.maximum(future_pred, 0)

    total_forecast = int(round(float(future_pred.sum())))
    avg_daily = float(future_pred.mean())
    safety_stock = int(round(avg_daily * 7))
    recommended_stock = int(total_forecast + safety_stock)
    drift_ratio, is_drift = detect_drift(sales)

    drug_name = str(group["drug_name"].iloc[0])
    category = str(group["category"].iloc[0])
    sub_category = str(group["sub_category"].iloc[0])
    display_name = f"{drug_name} ({sub_category})"

    return {
        "sku_id": sku_id,
        "drug_name": drug_name,
        "display_name": display_name,
        "category": category,
        "sub_category": sub_category,
        "model_used": model_used,
        "model_scores": {
            "DateLinearRegression": round(float(mape_lr), 1),
            "DateRidge": round(float(mape_ridge), 1),
        },
        "mape": round(float(mape_value), 1),
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

    category_stats = {}
    by_category = defaultdict(list)
    for item in valid:
        by_category[item["category"]].append(item["mape"])
    for category, vals in by_category.items():
        category_stats[category] = {
            "count": len(vals),
            "avg_mape": round(float(np.mean(vals)), 2),
            "median_mape": round(float(np.median(vals)), 2),
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
        "category_stats": dict(sorted(category_stats.items())),
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

    print("=" * 60)
    print("  Improved 9000-SKU inventory forecast")
    print("=" * 60)
    t0 = time.time()

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df["sku_id"] = (
        df["category"].astype(str)
        + "|"
        + df["sub_category"].astype(str)
        + "|"
        + df["drug_name"].astype(str)
    )
    print(
        f"Loaded: {len(df):,} rows, "
        f"{df['sku_id'].nunique():,} sku_id, "
        f"{df['drug_name'].nunique():,} unique drug_name"
    )
    print(f"Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

    results = {}
    grouped = df.groupby("sku_id", sort=True)
    total = grouped.ngroups
    for i, (sku_id, group) in enumerate(grouped, 1):
        try:
            results[sku_id] = forecast_one(sku_id, group)
        except Exception as exc:
            results[sku_id] = {"sku_id": sku_id, "error": str(exc)[:200], "mape": 999}
        if i % 1000 == 0 or i == total:
            print(f"Progress: {i:,}/{total:,}, elapsed={time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    summary = summarize(results, elapsed)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  Improved forecast done")
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
