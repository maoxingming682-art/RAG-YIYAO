# -*- coding: utf-8 -*-
"""Parallel runner for the fair 9000-SKU 5-model forecast."""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR
from importlib import import_module

forecast_mod = import_module("30_inventory_forecast_9000_5model")

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
INPUT_CSV = os.path.join(DATA_DIR, "sales_data_9000_2year.csv")
OUT_JSON = os.path.join(DATA_DIR, "forecast_9000_5model_result.json")
SUMMARY_JSON = os.path.join(LOGS_DIR, "forecast_9000_5model_summary.json")
CHECKPOINT_JSON = os.path.join(DATA_DIR, "forecast_9000_5model_checkpoint.json")
LOG_PATH = os.path.join(LOGS_DIR, "forecast_9000_5model_parallel.log")


def run_one(task):
    sku_id, group = task
    return sku_id, forecast_mod.forecast_one(sku_id, group)


def save_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


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
    print("  9000-SKU fair 5-model forecast - parallel")
    print("=" * 60)

    done = {}
    if os.path.exists(CHECKPOINT_JSON):
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
            done = json.load(f)
        print(f"Loaded checkpoint: {len(done):,} finished")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df["sku_id"] = (
        df["category"].astype(str)
        + "|"
        + df["sub_category"].astype(str)
        + "|"
        + df["drug_name"].astype(str)
    )
    grouped = [(sku_id, group) for sku_id, group in df.groupby("sku_id", sort=True) if sku_id not in done]
    total = df["sku_id"].nunique()
    print(f"Loaded: {len(df):,} rows, {total:,} sku_id")
    print(f"Remaining: {len(grouped):,}")

    workers = min(6, max(1, (os.cpu_count() or 4) - 1))
    print(f"Workers: {workers}")
    completed_since_save = 0

    if grouped:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_one, task): task[0] for task in grouped}
            for future in as_completed(future_map):
                sku_id = future_map[future]
                try:
                    result_sku, result = future.result()
                    done[result_sku] = result
                except Exception as exc:
                    done[sku_id] = {"sku_id": sku_id, "error": str(exc)[:200], "mape": 999}

                completed_since_save += 1
                n_done = len(done)
                if completed_since_save >= 500 or n_done == total:
                    completed_since_save = 0
                    save_json(CHECKPOINT_JSON, done)
                    valid = [v for v in done.values() if v.get("mape", 999) < 999]
                    avg = np.mean([v["mape"] for v in valid]) if valid else 0
                    print(
                        f"Progress: {n_done:,}/{total:,}, avg_mape_so_far={avg:.2f}%, "
                        f"elapsed={time.time() - t0:.1f}s",
                        flush=True,
                    )

    elapsed = time.time() - t0
    save_json(OUT_JSON, done)
    summary = forecast_mod.summarize(done, elapsed)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  Parallel 5-model forecast done")
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
