# -*- coding: utf-8 -*-
"""
Global Transformer baseline for store-SKU forecasting.

This is the first deep-learning comparison against the store-SKU baseline:
  - input: 90 days of sales, stock, inbound, stockout + calendar features
  - embeddings: store, SKU, category, store grade
  - output: next 30 days sales
  - evaluation: same final 30-day holdout as the baseline script

The model is trained with sampled active store-SKU pairs to keep iteration
time reasonable, then evaluated on a large deterministic holdout sample.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR


DATA_DIR = Path(BASE_DIR) / "data" / "store_sku_140x5000_2y"
INPUT_LEN = 90
FORECAST_LEN = 30
TEST_LEN = 30
VAL_LEN = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a store-SKU Transformer forecast model.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-pairs", type=int, default=80000)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_calendar_features(calendar: pd.DataFrame) -> np.ndarray:
    dow = calendar["day_of_week"].to_numpy()
    month = calendar["month"].to_numpy()
    rows = [
        np.eye(7, dtype=np.float32)[dow],
        np.eye(12, dtype=np.float32)[month - 1],
        calendar[["is_weekend", "is_new_year", "is_spring_festival_window", "is_national_day_window", "is_year_end_window"]]
        .to_numpy(dtype=np.float32),
    ]
    return np.concatenate(rows, axis=1).astype(np.float32)


def metrics(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    actual = actual.astype(np.float32)
    pred = pred.astype(np.float32)
    mask = mask.astype(bool)
    abs_err = np.abs(actual - pred) * mask
    sq_err = ((actual - pred) ** 2) * mask
    denom = (actual * mask).sum()
    eval_points = mask.sum()
    wmape = abs_err.sum() / max(float(denom), 1.0) * 100.0
    rmse = math.sqrt(float(sq_err.sum()) / max(int(eval_points), 1))
    per_pair_mape = np.where(
        mask.sum(axis=1) > 0,
        (np.abs(actual - pred) / np.maximum(actual, 1.0) * 100.0 * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1),
        999.0,
    )
    valid = per_pair_mape[per_pair_mape < 999]
    return {
        "overall_wmape": round(float(wmape), 4),
        "overall_rmse": round(float(rmse), 4),
        "avg_pair_mape": round(float(valid.mean()), 4),
        "median_pair_mape": round(float(np.median(valid)), 4),
        "p90_pair_mape": round(float(np.percentile(valid, 90)), 4),
        "eval_points": int(eval_points),
        "actual_sum": int(round(float((actual * mask).sum()))),
    }


class StoreSkuWindowDataset(Dataset):
    def __init__(
        self,
        sales,
        stock,
        inbound,
        stockout,
        active_store_idx: np.ndarray,
        active_sku_idx: np.ndarray,
        store_codes: np.ndarray,
        sku_codes: np.ndarray,
        category_codes: np.ndarray,
        grade_codes: np.ndarray,
        cal_features: np.ndarray,
        start_low: int,
        start_high: int,
        size: int,
        seed: int,
    ):
        self.sales = sales
        self.stock = stock
        self.inbound = inbound
        self.stockout = stockout
        self.active_store_idx = active_store_idx
        self.active_sku_idx = active_sku_idx
        self.store_codes = store_codes
        self.sku_codes = sku_codes
        self.category_codes = category_codes
        self.grade_codes = grade_codes
        self.cal_features = cal_features
        self.start_low = start_low
        self.start_high = start_high
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _: int):
        pair_pos = int(self.rng.integers(0, len(self.active_store_idx)))
        s_idx = int(self.active_store_idx[pair_pos])
        p_idx = int(self.active_sku_idx[pair_pos])
        start = int(self.rng.integers(self.start_low, self.start_high + 1))
        x_slice = slice(start, start + INPUT_LEN)
        y_slice = slice(start + INPUT_LEN, start + INPUT_LEN + FORECAST_LEN)

        y_hist = self.sales[s_idx, p_idx, x_slice].astype(np.float32)
        stock_hist = self.stock[s_idx, p_idx, x_slice].astype(np.float32)
        inbound_hist = self.inbound[s_idx, p_idx, x_slice].astype(np.float32)
        stockout_hist = self.stockout[s_idx, p_idx, x_slice].astype(np.float32)
        target = self.sales[s_idx, p_idx, y_slice].astype(np.float32)
        target_stockout = self.stockout[s_idx, p_idx, y_slice].astype(np.float32)

        scale = max(float(y_hist[stockout_hist < 0.5].mean()) if np.any(stockout_hist < 0.5) else float(y_hist.mean()), 1.0)
        x = np.concatenate(
            [
                (y_hist / scale).reshape(-1, 1),
                (stock_hist / max(scale * 14.0, 1.0)).reshape(-1, 1),
                (inbound_hist / max(scale * 14.0, 1.0)).reshape(-1, 1),
                stockout_hist.reshape(-1, 1),
                self.cal_features[x_slice],
            ],
            axis=1,
        ).astype(np.float32)
        target_norm = (target / scale).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "target": torch.from_numpy(target_norm),
            "target_raw": torch.from_numpy(target),
            "target_mask": torch.from_numpy((1.0 - target_stockout).astype(np.float32)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "store": torch.tensor(self.store_codes[s_idx], dtype=torch.long),
            "sku": torch.tensor(self.sku_codes[p_idx], dtype=torch.long),
            "category": torch.tensor(self.category_codes[p_idx], dtype=torch.long),
            "grade": torch.tensor(self.grade_codes[s_idx], dtype=torch.long),
        }


class StoreSkuTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_stores: int,
        n_skus: int,
        n_categories: int,
        n_grades: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, INPUT_LEN, d_model))
        self.store_emb = nn.Embedding(n_stores, d_model)
        self.sku_emb = nn.Embedding(n_skus, d_model)
        self.category_emb = nn.Embedding(n_categories, d_model)
        self.grade_emb = nn.Embedding(n_grades, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, FORECAST_LEN),
            nn.Softplus(),
        )

    def forward(self, x, store, sku, category, grade):
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        context = (
            self.store_emb(store)
            + self.sku_emb(sku)
            + self.category_emb(category)
            + self.grade_emb(grade)
        ).unsqueeze(1)
        h = h + context
        h = self.encoder(h)
        pooled = self.norm(h[:, -1, :])
        return self.head(pooled)


def evaluate_model(model, eval_loader, device) -> dict:
    model.eval()
    actuals = []
    preds = []
    masks = []
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            store = batch["store"].to(device)
            sku = batch["sku"].to(device)
            category = batch["category"].to(device)
            grade = batch["grade"].to(device)
            pred_norm = model(x, store, sku, category, grade)
            pred = pred_norm.cpu().numpy() * batch["scale"].numpy()[:, None]
            actuals.append(batch["target_raw"].numpy())
            preds.append(pred)
            masks.append(batch["target_mask"].numpy())
    return metrics(np.vstack(actuals), np.vstack(preds), np.vstack(masks))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    t0 = time.time()
    data_dir = args.data_dir
    out_summary = data_dir / "transformer_forecast_summary.json"
    out_model = data_dir / "transformer_store_sku_model.pt"
    if out_summary.exists() and not args.force:
        raise SystemExit(f"Transformer output already exists: {out_summary}. Use --force to overwrite.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("Store-SKU Transformer forecast")
    print("=" * 72)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sales = np.load(data_dir / "sales_qty_uint16.npy", mmap_mode="r")
    stock = np.load(data_dir / "stock_qty_uint16.npy", mmap_mode="r")
    inbound = np.load(data_dir / "inbound_qty_uint16.npy", mmap_mode="r")
    stockout = np.load(data_dir / "stockout_uint8.npy", mmap_mode="r")
    assortment = np.load(data_dir / "assortment_uint8.npy", mmap_mode="r")
    products = pd.read_csv(data_dir / "product_master.csv")
    stores = pd.read_csv(data_dir / "store_master.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    baseline_summary = json.load(open(data_dir / "baseline_forecast_summary.json", encoding="utf-8"))

    active_store_idx, active_sku_idx = np.where(assortment > 0)
    n_active = len(active_store_idx)
    n_stores, n_skus, n_days = sales.shape
    fit_end = n_days - TEST_LEN - VAL_LEN
    val_start = n_days - TEST_LEN - VAL_LEN - INPUT_LEN
    test_start = n_days - TEST_LEN - INPUT_LEN
    train_start_low = 0
    train_start_high = fit_end - INPUT_LEN - FORECAST_LEN
    if train_start_high <= train_start_low:
        raise SystemExit("Not enough history.")

    store_codes = np.arange(n_stores, dtype=np.int64)
    sku_codes = np.arange(n_skus, dtype=np.int64)
    category_codes = pd.Categorical(products["category"]).codes.astype(np.int64)
    grade_codes = pd.Categorical(stores["store_grade"], categories=["A", "B", "C", "D"]).codes.astype(np.int64)
    n_categories = int(category_codes.max() + 1)
    n_grades = int(grade_codes.max() + 1)
    cal_features = build_calendar_features(calendar)
    input_dim = 4 + cal_features.shape[1]

    train_ds = StoreSkuWindowDataset(
        sales, stock, inbound, stockout,
        active_store_idx, active_sku_idx,
        store_codes, sku_codes, category_codes, grade_codes,
        cal_features,
        train_start_low, train_start_high,
        size=args.steps_per_epoch * args.batch_size,
        seed=args.seed,
    )
    val_ds = StoreSkuWindowDataset(
        sales, stock, inbound, stockout,
        active_store_idx, active_sku_idx,
        store_codes, sku_codes, category_codes, grade_codes,
        cal_features,
        val_start, val_start,
        size=min(args.eval_pairs, n_active),
        seed=args.seed + 100,
    )
    test_ds = StoreSkuWindowDataset(
        sales, stock, inbound, stockout,
        active_store_idx, active_sku_idx,
        store_codes, sku_codes, category_codes, grade_codes,
        cal_features,
        test_start, test_start,
        size=min(args.eval_pairs, n_active),
        seed=args.seed + 200,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    model = StoreSkuTransformer(
        input_dim=input_dim,
        n_stores=n_stores,
        n_skus=n_skus,
        n_categories=n_categories,
        n_grades=n_grades,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * args.steps_per_epoch, 1))
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = 999.0
    best_epoch = 0
    history = []
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Active pairs: {n_active:,}")
    print(f"Train windows/epoch: {len(train_ds):,}; eval pairs: {len(test_ds):,}")
    print(f"Model params: {total_params:,}")
    print(f"Input dim: {input_dim}, d_model={args.d_model}, layers={args.layers}, heads={args.heads}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["target_mask"].to(device, non_blocking=True)
            store = batch["store"].to(device, non_blocking=True)
            sku = batch["sku"].to(device, non_blocking=True)
            category = batch["category"].to(device, non_blocking=True)
            grade = batch["grade"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pred = model(x, store, sku, category, grade)
                loss_mat = loss_fn(pred, target)
                loss = (loss_mat * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step >= args.steps_per_epoch:
                break

        val_metrics = evaluate_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6), "val": val_metrics})
        print(
            f"Epoch {epoch}/{args.epochs}: loss={np.mean(losses):.5f}, "
            f"val_wmape={val_metrics['overall_wmape']}%, "
            f"val_mape={val_metrics['avg_pair_mape']}%, elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        if val_metrics["overall_wmape"] < best_val:
            best_val = val_metrics["overall_wmape"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "input_dim": input_dim}, out_model)

    checkpoint = torch.load(out_model, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_model(model, test_loader, device)
    baseline = {
        "overall_wmape": baseline_summary["overall"]["overall_wmape"],
        "avg_pair_mape": baseline_summary["avg_pair_mape"],
        "median_pair_mape": baseline_summary["median_pair_mape"],
        "p90_pair_mape": baseline_summary["p90_pair_mape"],
    }
    summary = {
        "dataset": str(data_dir),
        "active_pairs_total": int(n_active),
        "eval_pairs": int(len(test_ds)),
        "input_len": INPUT_LEN,
        "forecast_len": FORECAST_LEN,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "model": {
            "type": "StoreSkuTransformer",
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "params": int(total_params),
        },
        "best_epoch": int(best_epoch),
        "history": history,
        "transformer": test_metrics,
        "baseline": baseline,
        "delta_vs_baseline": {
            "overall_wmape_points": round(test_metrics["overall_wmape"] - baseline["overall_wmape"], 4),
            "avg_pair_mape_points": round(test_metrics["avg_pair_mape"] - baseline["avg_pair_mape"], 4),
        },
        "winner": "transformer" if test_metrics["overall_wmape"] < baseline["overall_wmape"] else "baseline",
        "elapsed_seconds": round(time.time() - t0, 1),
        "outputs": {"model": str(out_model), "summary": str(out_summary)},
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Transformer forecast complete")
    print("=" * 72)
    print(f"Transformer WMAPE: {test_metrics['overall_wmape']}%")
    print(f"Baseline WMAPE: {baseline['overall_wmape']}%")
    print(f"Winner: {summary['winner']}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
