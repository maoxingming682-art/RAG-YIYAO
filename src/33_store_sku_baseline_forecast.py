# -*- coding: utf-8 -*-
"""
Store-SKU baseline forecasting for the 140-store x 5000-SKU dataset.

Goals:
  - Use all active store-SKU series, not a tiny sample.
  - Treat stockout days as censored demand.
  - Select the best baseline model on a validation window.
  - Report both MAPE and WMAPE on the final holdout window.

Output files are written to data/store_sku_140x5000_2y:
  - baseline_forecast_30d_uint16.npy
  - baseline_active_index.parquet
  - baseline_forecast_summary.json
  - baseline_model_category_summary.csv
  - baseline_model_store_grade_summary.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR


DATA_DIR = Path(BASE_DIR) / "data" / "store_sku_140x5000_2y"
FORECAST_LEN = 30
VAL_LEN = 30
TEST_LEN = 30
MODELS = [
    "SeasonalNaive7",
    "DowMean56",
    "TrendAdjustedDow56",
    "MovingAverage28",
    "MovingAverage56",
    "DateRidge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run store-SKU baseline forecast.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all active store-SKU pairs.")
    parser.add_argument(
        "--selection-metric",
        choices=["wmape", "mape", "blend"],
        default="wmape",
        help="Metric used for validation model selection.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_mape(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    actual = actual.astype(np.float32, copy=False)
    pred = pred.astype(np.float32, copy=False)
    mask = mask.astype(bool, copy=False)
    abs_pct = np.abs(actual - pred) / np.maximum(actual, 1.0) * 100.0
    counts = mask.sum(axis=1)
    vals = (abs_pct * mask).sum(axis=1) / np.maximum(counts, 1)
    return np.where(counts > 0, vals, 999.0).astype(np.float32)


def safe_wmape(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    actual = actual.astype(np.float32, copy=False)
    pred = pred.astype(np.float32, copy=False)
    mask = mask.astype(bool, copy=False)
    abs_err = np.abs(actual - pred) * mask
    denom = (actual * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    vals = abs_err.sum(axis=1) / np.maximum(denom, 1.0) * 100.0
    return np.where(counts > 0, vals, 999.0).astype(np.float32)


def safe_rmse(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    actual = actual.astype(np.float32, copy=False)
    pred = pred.astype(np.float32, copy=False)
    mask = mask.astype(bool, copy=False)
    counts = mask.sum(axis=1)
    sq = ((actual - pred) ** 2) * mask
    vals = np.sqrt(sq.sum(axis=1) / np.maximum(counts, 1))
    return np.where(counts > 0, vals, 999.0).astype(np.float32)


def build_features(dates: pd.Series | pd.DatetimeIndex, start_index: int = 0) -> np.ndarray:
    dt = pd.to_datetime(dates)
    n = len(dt)
    t = (np.arange(start_index, start_index + n, dtype=np.float32) / 365.0).reshape(-1, 1)
    if isinstance(dt, pd.Series):
        dow = dt.dt.dayofweek.to_numpy()
        month = dt.dt.month.to_numpy()
        day = dt.dt.day.to_numpy()
    else:
        dow = dt.dayofweek.to_numpy()
        month = dt.month.to_numpy()
        day = dt.day.to_numpy()
    rows = [np.ones((n, 1), dtype=np.float32), t]
    rows.append(np.eye(7, dtype=np.float32)[dow])
    rows.append(np.eye(12, dtype=np.float32)[month - 1])
    weekend = (dow >= 5).astype(np.float32).reshape(-1, 1)
    winter = np.isin(month, [1, 2, 3, 11, 12]).astype(np.float32).reshape(-1, 1)
    summer = np.isin(month, [6, 7, 8]).astype(np.float32).reshape(-1, 1)
    holiday = (
        ((month == 1) & (day <= 3))
        | ((month == 1) & (day >= 24))
        | ((month == 2) & (day <= 12))
        | ((month == 10) & (day <= 7))
        | ((month == 12) & (day >= 20))
    ).astype(np.float32).reshape(-1, 1)
    rows.extend([weekend, winter, summer, holiday])
    return np.concatenate(rows, axis=1).astype(np.float32)


def ridge_projector(x_train: np.ndarray, alpha: float) -> np.ndarray:
    p = x_train.shape[1]
    penalty = np.eye(p, dtype=np.float32) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_train.T @ x_train + penalty, x_train.T).T.astype(np.float32)


def repair_stockouts(sales: np.ndarray, stockout: np.ndarray) -> np.ndarray:
    """Use a conservative demand proxy for stockout days."""
    y = sales.astype(np.float32, copy=True)
    stockout = stockout.astype(bool, copy=False)
    non_stock = ~stockout
    non_count = non_stock.sum(axis=1)
    non_sum = (y * non_stock).sum(axis=1)
    mean_non = non_sum / np.maximum(non_count, 1)
    mean_non = np.where(non_count > 0, mean_non, y.mean(axis=1))

    lag7 = np.empty_like(y)
    lag7[:, :7] = mean_non[:, None]
    lag7[:, 7:] = y[:, :-7]
    imputed = np.maximum(y, np.maximum(lag7, mean_non[:, None]) * 1.05)
    y[stockout] = imputed[stockout]
    return y


def predict_ma(train: np.ndarray, target_len: int, window: int) -> np.ndarray:
    avg = train[:, -min(window, train.shape[1]):].mean(axis=1)
    return np.repeat(avg[:, None], target_len, axis=1).astype(np.float32)


def predict_seasonal_naive(train: np.ndarray, target_len: int) -> np.ndarray:
    if train.shape[1] < 7:
        return predict_ma(train, target_len, min(28, train.shape[1]))
    idx = train.shape[1] - 7 + (np.arange(target_len) % 7)
    return train[:, idx].astype(np.float32)


def predict_dow_mean(train: np.ndarray, train_dow: np.ndarray, target_dow: np.ndarray, window: int = 56) -> np.ndarray:
    win = min(window, train.shape[1])
    y = train[:, -win:]
    dows = train_dow[-win:]
    pred = np.zeros((train.shape[0], len(target_dow)), dtype=np.float32)
    global_avg = y.mean(axis=1)
    dow_means = {}
    for dow in range(7):
        cols = dows == dow
        if cols.any():
            dow_means[dow] = y[:, cols].mean(axis=1)
        else:
            dow_means[dow] = global_avg
    for h, dow in enumerate(target_dow):
        pred[:, h] = dow_means[int(dow)]
    return pred


def predict_trend_adjusted_dow(train: np.ndarray, train_dow: np.ndarray, target_dow: np.ndarray) -> np.ndarray:
    pred = predict_dow_mean(train, train_dow, target_dow, window=56)
    recent = train[:, -28:].mean(axis=1)
    baseline_start = max(0, train.shape[1] - 84)
    baseline_end = max(1, train.shape[1] - 28)
    baseline = train[:, baseline_start:baseline_end].mean(axis=1)
    ratio = np.clip(recent / np.maximum(baseline, 1.0), 0.70, 1.35)
    return (pred * ratio[:, None]).astype(np.float32)


def predict_date_ridge(train: np.ndarray, projector: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    beta = train @ projector
    pred = beta @ x_target.T
    return np.maximum(pred, 0).astype(np.float32)


def make_predictions(
    train: np.ndarray,
    train_dow: np.ndarray,
    target_dow: np.ndarray,
    ridge_projector_for_train: np.ndarray,
    x_target: np.ndarray,
) -> list[np.ndarray]:
    return [
        predict_seasonal_naive(train, len(target_dow)),
        predict_dow_mean(train, train_dow, target_dow, window=56),
        predict_trend_adjusted_dow(train, train_dow, target_dow),
        predict_ma(train, len(target_dow), window=28),
        predict_ma(train, len(target_dow), window=56),
        predict_date_ridge(train, ridge_projector_for_train, x_target),
    ]


def aggregate_overall(actual_sum: float, abs_err_sum: float, sq_err_sum: float, count: int) -> dict:
    return {
        "overall_wmape": round(abs_err_sum / max(actual_sum, 1.0) * 100.0, 4),
        "overall_rmse": round(math.sqrt(sq_err_sum / max(count, 1)), 4),
        "eval_points": int(count),
        "actual_sum": int(round(actual_sum)),
        "absolute_error_sum": round(float(abs_err_sum), 3),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    data_dir = args.data_dir
    out_pred = data_dir / "baseline_forecast_30d_uint16.npy"
    out_index = data_dir / "baseline_active_index.parquet"
    out_summary = data_dir / "baseline_forecast_summary.json"

    if out_summary.exists() and not args.force:
        raise SystemExit(f"Baseline outputs already exist: {out_summary}. Use --force to overwrite.")

    print("=" * 72)
    print("Store-SKU baseline forecast")
    print("=" * 72)
    print(f"Data: {data_dir}")
    print(f"Selection metric: {args.selection_metric}")

    sales = np.load(data_dir / "sales_qty_uint16.npy", mmap_mode="r")
    stockout = np.load(data_dir / "stockout_uint8.npy", mmap_mode="r")
    assortment = np.load(data_dir / "assortment_uint8.npy", mmap_mode="r")
    products = pd.read_csv(data_dir / "product_master.csv")
    stores = pd.read_csv(data_dir / "store_master.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    dates = pd.to_datetime(calendar["date"])
    n_stores, n_skus, n_days = sales.shape
    if n_days < VAL_LEN + TEST_LEN + 120:
        raise SystemExit("Not enough history for validation/test split.")

    val_start = n_days - TEST_LEN - VAL_LEN
    test_start = n_days - TEST_LEN
    fit_len = val_start
    train_all_len = test_start

    active_store_idx, active_sku_idx = np.where(assortment > 0)
    n_active_total = len(active_store_idx)
    if args.max_pairs and args.max_pairs > 0:
        n_active = min(args.max_pairs, n_active_total)
        active_store_idx = active_store_idx[:n_active]
        active_sku_idx = active_sku_idx[:n_active]
    else:
        n_active = n_active_total

    print(f"Tensor shape: stores={n_stores}, skus={n_skus}, days={n_days}")
    print(f"Active store-SKU pairs: {n_active:,}/{n_active_total:,}")
    print(f"Split: fit=0:{fit_len}, val={val_start}:{test_start}, test={test_start}:{n_days}")

    hist_dow = calendar["day_of_week"].to_numpy(dtype=np.int16)
    future_dates = pd.date_range(dates.iloc[-1] + timedelta(days=1), periods=FORECAST_LEN)
    future_dow = future_dates.dayofweek.to_numpy(dtype=np.int16)
    val_dow = hist_dow[val_start:test_start]
    test_dow = hist_dow[test_start:n_days]

    x_fit = build_features(dates.iloc[:fit_len], start_index=0)
    x_val = build_features(dates.iloc[val_start:test_start], start_index=val_start)
    x_train_all = build_features(dates.iloc[:train_all_len], start_index=0)
    x_test = build_features(dates.iloc[test_start:n_days], start_index=test_start)
    x_all = build_features(dates, start_index=0)
    x_future = build_features(future_dates, start_index=n_days)
    projector_fit = ridge_projector(x_fit, args.ridge_alpha)
    projector_train_all = ridge_projector(x_train_all, args.ridge_alpha)
    projector_all = ridge_projector(x_all, args.ridge_alpha)

    forecast_mm = np.lib.format.open_memmap(
        out_pred,
        mode="w+",
        dtype=np.uint16,
        shape=(n_active, FORECAST_LEN),
    )

    rows = []
    model_counts = {m: 0 for m in MODELS}
    model_test_mapes = {m: [] for m in MODELS}
    model_test_wmapes = {m: [] for m in MODELS}
    all_test_mapes = []
    all_test_wmapes = []
    all_test_rmses = []
    all_val_scores = []
    overall_actual_sum = 0.0
    overall_abs_err_sum = 0.0
    overall_sq_err_sum = 0.0
    overall_eval_count = 0

    n_chunks = math.ceil(n_active / args.chunk_size)
    for chunk_idx, start in enumerate(range(0, n_active, args.chunk_size), 1):
        end = min(start + args.chunk_size, n_active)
        s_idx = active_store_idx[start:end]
        p_idx = active_sku_idx[start:end]

        y_raw = sales[s_idx, p_idx, :].astype(np.float32)
        so = stockout[s_idx, p_idx, :].astype(bool)
        y = repair_stockouts(y_raw, so)

        y_fit = y[:, :fit_len]
        y_train_all = y[:, :train_all_len]
        y_all = y
        val_actual = y_raw[:, val_start:test_start]
        test_actual = y_raw[:, test_start:n_days]
        val_mask = ~so[:, val_start:test_start]
        test_mask = ~so[:, test_start:n_days]

        val_preds = make_predictions(y_fit, hist_dow[:fit_len], val_dow, projector_fit, x_val)
        val_mape_by_model = np.vstack([safe_mape(val_actual, pred, val_mask) for pred in val_preds])
        val_wmape_by_model = np.vstack([safe_wmape(val_actual, pred, val_mask) for pred in val_preds])
        if args.selection_metric == "mape":
            val_scores = val_mape_by_model
        elif args.selection_metric == "blend":
            val_scores = 0.5 * val_mape_by_model + 0.5 * val_wmape_by_model
        else:
            val_scores = val_wmape_by_model
        best_idx = np.argmin(val_scores, axis=0).astype(np.int16)
        best_val_score = val_scores[best_idx, np.arange(end - start)]

        test_preds = make_predictions(y_train_all, hist_dow[:train_all_len], test_dow, projector_train_all, x_test)
        future_preds = make_predictions(y_all, hist_dow, future_dow, projector_all, x_future)

        selected_test_pred = np.zeros((end - start, TEST_LEN), dtype=np.float32)
        selected_future_pred = np.zeros((end - start, FORECAST_LEN), dtype=np.float32)
        for model_idx, model_name in enumerate(MODELS):
            pick = best_idx == model_idx
            if not pick.any():
                continue
            selected_test_pred[pick] = test_preds[model_idx][pick]
            selected_future_pred[pick] = future_preds[model_idx][pick]
            model_counts[model_name] += int(pick.sum())

        test_mape = safe_mape(test_actual, selected_test_pred, test_mask)
        test_wmape = safe_wmape(test_actual, selected_test_pred, test_mask)
        test_rmse = safe_rmse(test_actual, selected_test_pred, test_mask)
        all_test_mapes.extend(test_mape.tolist())
        all_test_wmapes.extend(test_wmape.tolist())
        all_test_rmses.extend(test_rmse.tolist())
        all_val_scores.extend(best_val_score.tolist())

        for model_idx, model_name in enumerate(MODELS):
            pick = best_idx == model_idx
            if pick.any():
                model_test_mapes[model_name].extend(test_mape[pick].tolist())
                model_test_wmapes[model_name].extend(test_wmape[pick].tolist())

        abs_err = np.abs(test_actual - selected_test_pred) * test_mask
        sq_err = ((test_actual - selected_test_pred) ** 2) * test_mask
        overall_actual_sum += float((test_actual * test_mask).sum())
        overall_abs_err_sum += float(abs_err.sum())
        overall_sq_err_sum += float(sq_err.sum())
        overall_eval_count += int(test_mask.sum())

        future_u16 = np.clip(np.rint(selected_future_pred), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        forecast_mm[start:end] = future_u16

        rows.append(pd.DataFrame({
            "pair_idx": np.arange(start, end, dtype=np.int32),
            "store_idx": s_idx.astype(np.int16),
            "sku_idx": p_idx.astype(np.int32),
            "store_id": stores["store_id"].iloc[s_idx].to_numpy(),
            "sku_id": products["sku_id"].iloc[p_idx].to_numpy(),
            "model_used": np.array(MODELS, dtype=object)[best_idx],
            "val_score": np.round(best_val_score, 4),
            "test_mape": np.round(test_mape, 4),
            "test_wmape": np.round(test_wmape, 4),
            "test_rmse": np.round(test_rmse, 4),
            "forecast_30d": future_u16.sum(axis=1).astype(np.int32),
            "avg_daily_forecast": np.round(future_u16.mean(axis=1), 3),
            "safety_stock": np.rint(future_u16.mean(axis=1) * 7).astype(np.int32),
            "recommended_stock": (future_u16.sum(axis=1) + np.rint(future_u16.mean(axis=1) * 7)).astype(np.int32),
            "test_actual_30d": np.rint((test_actual * test_mask).sum(axis=1)).astype(np.int32),
            "test_eval_days": test_mask.sum(axis=1).astype(np.int16),
            "test_stockout_days": so[:, test_start:n_days].sum(axis=1).astype(np.int16),
        }))

        forecast_mm.flush()
        elapsed = time.time() - t0
        chunk_wmape = overall_abs_err_sum / max(overall_actual_sum, 1.0) * 100.0
        print(
            f"Chunk {chunk_idx:>3}/{n_chunks}: pairs {start:,}-{end:,}, "
            f"overall_wmape={chunk_wmape:.3f}%, "
            f"avg_pair_mape={np.mean(all_test_mapes):.3f}%, elapsed={elapsed:.1f}s",
            flush=True,
        )

    forecast_mm.flush()
    index_df = pd.concat(rows, ignore_index=True)
    index_df = index_df.merge(
        stores[["store_id", "city", "store_grade", "business_district", "warehouse_id"]],
        on="store_id",
        how="left",
    ).merge(
        products[["sku_id", "display_name", "category", "sub_category", "rx_type", "base_price", "season_type"]],
        on="sku_id",
        how="left",
    )
    index_df.to_parquet(out_index, index=False)

    model_summary = []
    for model in MODELS:
        count = model_counts[model]
        model_summary.append({
            "model": model,
            "count": int(count),
            "pct": round(count / max(n_active, 1) * 100.0, 3),
            "avg_test_mape": round(float(np.mean(model_test_mapes[model])), 4) if model_test_mapes[model] else None,
            "avg_test_wmape": round(float(np.mean(model_test_wmapes[model])), 4) if model_test_wmapes[model] else None,
        })

    category_summary = (
        index_df.groupby("category")
        .agg(
            pairs=("pair_idx", "count"),
            forecast_30d=("forecast_30d", "sum"),
            actual_30d=("test_actual_30d", "sum"),
            avg_test_mape=("test_mape", "mean"),
            avg_test_wmape=("test_wmape", "mean"),
            stockout_days=("test_stockout_days", "sum"),
        )
        .reset_index()
    )
    category_summary["avg_test_mape"] = category_summary["avg_test_mape"].round(4)
    category_summary["avg_test_wmape"] = category_summary["avg_test_wmape"].round(4)
    category_summary.to_csv(data_dir / "baseline_model_category_summary.csv", index=False, encoding="utf-8-sig")

    grade_summary = (
        index_df.groupby("store_grade")
        .agg(
            pairs=("pair_idx", "count"),
            forecast_30d=("forecast_30d", "sum"),
            actual_30d=("test_actual_30d", "sum"),
            avg_test_mape=("test_mape", "mean"),
            avg_test_wmape=("test_wmape", "mean"),
            stockout_days=("test_stockout_days", "sum"),
        )
        .reset_index()
        .sort_values("store_grade")
    )
    grade_summary["avg_test_mape"] = grade_summary["avg_test_mape"].round(4)
    grade_summary["avg_test_wmape"] = grade_summary["avg_test_wmape"].round(4)
    grade_summary.to_csv(data_dir / "baseline_model_store_grade_summary.csv", index=False, encoding="utf-8-sig")

    valid_mape = np.asarray([x for x in all_test_mapes if x < 999], dtype=np.float32)
    valid_wmape = np.asarray([x for x in all_test_wmapes if x < 999], dtype=np.float32)
    valid_rmse = np.asarray([x for x in all_test_rmses if x < 999], dtype=np.float32)
    summary = {
        "dataset": str(data_dir),
        "forecast_horizon_days": FORECAST_LEN,
        "validation_days": VAL_LEN,
        "test_days": TEST_LEN,
        "selection_metric": args.selection_metric,
        "active_pairs": int(n_active),
        "active_pairs_total": int(n_active_total),
        "valid_pairs": int(len(valid_mape)),
        "avg_pair_mape": round(float(valid_mape.mean()), 4),
        "median_pair_mape": round(float(np.median(valid_mape)), 4),
        "p90_pair_mape": round(float(np.percentile(valid_mape, 90)), 4),
        "avg_pair_wmape": round(float(valid_wmape.mean()), 4),
        "median_pair_wmape": round(float(np.median(valid_wmape)), 4),
        "p90_pair_wmape": round(float(np.percentile(valid_wmape, 90)), 4),
        "avg_pair_rmse": round(float(valid_rmse.mean()), 4),
        "overall": aggregate_overall(overall_actual_sum, overall_abs_err_sum, overall_sq_err_sum, overall_eval_count),
        "model_distribution": model_summary,
        "mape_dist": {
            "lt_10": int((valid_mape < 10).sum()),
            "10_15": int(((valid_mape >= 10) & (valid_mape < 15)).sum()),
            "15_25": int(((valid_mape >= 15) & (valid_mape < 25)).sum()),
            "25_35": int(((valid_mape >= 25) & (valid_mape < 35)).sum()),
            "gte_35": int((valid_mape >= 35).sum()),
        },
        "top_restock": index_df.sort_values("recommended_stock", ascending=False)
        .head(20)[["store_id", "store_grade", "sku_id", "display_name", "category", "forecast_30d", "recommended_stock", "test_mape"]]
        .to_dict(orient="records"),
        "worst_mape": index_df.sort_values("test_mape", ascending=False)
        .head(20)[["store_id", "store_grade", "sku_id", "display_name", "category", "test_mape", "test_wmape", "test_actual_30d", "forecast_30d"]]
        .to_dict(orient="records"),
        "outputs": {
            "forecast_30d": str(out_pred),
            "active_index": str(out_index),
            "category_summary": str(data_dir / "baseline_model_category_summary.csv"),
            "store_grade_summary": str(data_dir / "baseline_model_store_grade_summary.csv"),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Baseline forecast complete")
    print("=" * 72)
    print(f"Active pairs: {n_active:,}")
    print(f"Overall WMAPE: {summary['overall']['overall_wmape']}%")
    print(f"Avg pair MAPE: {summary['avg_pair_mape']}%")
    print(f"Median pair MAPE: {summary['median_pair_mape']}%")
    print(f"P90 pair MAPE: {summary['p90_pair_mape']}%")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
