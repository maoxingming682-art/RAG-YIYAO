# -*- coding: utf-8 -*-
"""
25_multi_model_forecast.py
多模型自动选优：每个商品跑4个模型，验证集PK，选MAPE最低的预测

模型：
  1. ExponentialSmoothing  趋势+季节性
  2. SeasonalNaive          上周同一天（强周期）
  3. MovingAverage          近28天均值（平稳序列）
  4. LinearRegression       趋势+星期+月份特征（有趋势的）

选优逻辑：
  - train(前305天) 再 split 成 fit(前275天) + val(后30天)
  - 4个模型在val上算MAPE
  - 选MAPE最低的，用全部train(335天)重训，预测未来30天
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from datetime import timedelta
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# 模型1：ExponentialSmoothing
# ============================================================
def model_ets(train, forecast_days=30):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        m = ExponentialSmoothing(train, trend='add', seasonal='add',
                                  seasonal_periods=7, initialization_method='estimated')
        fit = m.fit(optimized=True)
        pred = np.maximum(fit.forecast(forecast_days), 0)
        return pred
    except Exception:
        return None


# ============================================================
# 模型2：Seasonal Naive（上周同一天）
# ============================================================
def model_seasonal_naive(train, forecast_days=30):
    """预测第t天 = train里第t-7天的值（上周同一天）"""
    if len(train) < 7:
        return None
    pred = []
    for i in range(forecast_days):
        idx = len(train) - 7 + (i % 7)
        # 循环取上周同一天
        lookback = len(train) - 7 + (i % 7)
        if lookback < 0:
            lookback = len(train) - 7
        pred.append(train[lookback])
    return np.array(pred, dtype=float)


# ============================================================
# 模型3：Moving Average（近28天均值）
# ============================================================
def model_moving_average(train, forecast_days=30, window=28):
    """预测 = 最近window天的均值，每天相同"""
    if len(train) < window:
        window = len(train)
    avg = np.mean(train[-window:])
    return np.full(forecast_days, avg)


# ============================================================
# 模型4：Linear Regression with features
# ============================================================
def model_linear_regression(train, forecast_days=30):
    """用趋势+星期+月份做线性回归"""
    from sklearn.linear_model import LinearRegression
    try:
        n = len(train)
        # 构建特征：[序号, day_of_week(7个one-hot), month(12个one-hot)]
        dates = pd.date_range('2025-01-01', periods=n, freq='D')
        features = []
        for i, d in enumerate(dates):
            feat = [i]  # 趋势项
            # 星期 one-hot
            dow = d.weekday()
            feat.extend([1 if j == dow else 0 for j in range(7)])
            # 月份 one-hot
            month = d.month
            feat.extend([1 if j == month else 0 for j in range(1, 13)])
            features.append(feat)
        X = np.array(features)
        y = np.array(train, dtype=float)

        reg = LinearRegression()
        reg.fit(X, y)

        # 预测未来
        future_dates = pd.date_range(dates[-1] + timedelta(days=1), periods=forecast_days, freq='D')
        future_features = []
        for i, d in enumerate(future_dates):
            feat = [n + i]  # 趋势延续
            dow = d.weekday()
            feat.extend([1 if j == dow else 0 for j in range(7)])
            month = d.month
            feat.extend([1 if j == month else 0 for j in range(1, 13)])
            future_features.append(feat)
        X_future = np.array(future_features)
        pred = np.maximum(reg.predict(X_future), 0)
        return pred
    except Exception:
        return None


# ============================================================
# MAPE计算
# ============================================================
def calc_mape(actual, pred):
    return np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100


# ============================================================
# 单商品多模型选优
# ============================================================
def forecast_product_multi_model(drug_name, drug_df, forecast_days=30):
    """对单个商品跑4个模型，验证集选优，返回最优预测+模型名"""
    sales = drug_df.sort_values('date')['sales'].values.astype(float)
    dates = drug_df.sort_values('date')['date'].dt.strftime('%Y-%m-%d').tolist()
    category = drug_df['category'].iloc[0]
    sub_category = drug_df['sub_category'].iloc[0]

    # 全量split: train(335) + test(30)
    train_all = sales[:-forecast_days]
    test = sales[-forecast_days:]

    # 验证集split: fit(275) + val(30) —— 从train_all里再切30天做验证
    val_size = 30
    fit_data = train_all[:-val_size]
    val_data = train_all[-val_size:]

    # 4个模型在验证集上PK
    models = {
        "ExponentialSmoothing": lambda: model_ets(fit_data, val_size),
        "SeasonalNaive": lambda: model_seasonal_naive(fit_data, val_size),
        "MovingAverage": lambda: model_moving_average(fit_data, val_size),
        "LinearRegression": lambda: model_linear_regression(fit_data, val_size),
    }

    model_preds = {}   # 存验证集预测，给集成用
    model_scores = {}  # 验证集MAPE
    for name, fn in models.items():
        try:
            pred_val = fn()
            if pred_val is not None and len(pred_val) == val_size:
                mape = calc_mape(val_data, pred_val)
                model_scores[name] = mape
                model_preds[name] = pred_val
        except Exception:
            model_scores[name] = 999

    # 第5个选项：加权集成（权重=1/mape²，越准的模型权重越大）
    ensemble_pred = None
    if len(model_preds) >= 2:
        valid_models = {k: v for k, v in model_scores.items() if v < 999 and k in model_preds}
        if len(valid_models) >= 2:
            weights = {k: 1.0 / (v ** 2) for k, v in valid_models.items()}
            total_w = sum(weights.values())
            weights = {k: v / total_w for k, v in weights.items()}
            ensemble_pred = np.zeros(val_size)
            for name, w in weights.items():
                ensemble_pred += w * model_preds[name]
            ensemble_mape = calc_mape(val_data, ensemble_pred)
            model_scores["Ensemble"] = ensemble_mape
            model_preds["Ensemble"] = ensemble_pred

    # 选MAPE最低的模型
    if not model_scores:
        return {"drug_name": drug_name, "error": "all models failed", "mape": 999}

    best_model = min(model_scores, key=model_scores.get)

    # 用最优模型在全量train上重训，预测未来30天
    model_fns = {
        "ExponentialSmoothing": lambda: model_ets(train_all, forecast_days),
        "SeasonalNaive": lambda: model_seasonal_naive(train_all, forecast_days),
        "MovingAverage": lambda: model_moving_average(train_all, forecast_days),
        "LinearRegression": lambda: model_linear_regression(train_all, forecast_days),
    }

    try:
        if best_model == "Ensemble":
            # 集成：在全量train上重跑各模型，用验证集权重加权
            full_preds = {}
            full_scores = {k: v for k, v in model_scores.items() if k != "Ensemble" and v < 999}
            for name in full_scores:
                p = model_fns[name]()
                if p is not None:
                    full_preds[name] = p
            if len(full_preds) >= 2:
                weights = {k: 1.0 / (full_scores[k] ** 2) for k in full_preds if k in full_preds}
                total_w = sum(weights.values())
                weights = {k: v / total_w for k, v in weights.items()}
                pred = np.zeros(forecast_days)
                for name, w in weights.items():
                    pred += w * full_preds[name]
            else:
                # 集成降级：取唯一可用的
                pred = list(full_preds.values())[0] if full_preds else model_moving_average(train_all, forecast_days)
        else:
            pred = model_fns[best_model]()
            if pred is None:
                best_model = "MovingAverage"
                pred = model_moving_average(train_all, forecast_days)
    except Exception:
        best_model = "MovingAverage"
        pred = model_moving_average(train_all, forecast_days)

    pred = np.maximum(pred, 0)
    test_mape = calc_mape(test, pred)
    rmse = np.sqrt(np.mean((test - pred) ** 2))

    # 库存建议
    total_forecast = int(sum(pred))
    avg_daily = float(np.mean(pred))
    safety_stock = int(7 * avg_daily)
    recommended = int(total_forecast + safety_stock)

    # 漂移检测
    from scipy import stats
    recent = sales[-30:]
    baseline = sales[-120:-30]
    drift_ratio = (recent.mean() - baseline.mean()) / max(baseline.mean(), 1) * 100
    try:
        _, p_value = stats.ttest_ind(recent, baseline)
    except Exception:
        p_value = 0.0
    is_drift = abs(drift_ratio) > 20 and p_value < 0.05

    return {
        "drug_name": drug_name,
        "category": category,
        "sub_category": sub_category,
        "model_used": best_model,           # ★ 关键：记录用了哪个模型
        "model_scores": {k: round(v, 1) for k, v in model_scores.items()},  # 4个模型的验证集MAPE
        "mape": round(float(test_mape), 1),  # 测试集MAPE（最终精度）
        "rmse": round(float(rmse), 2),
        "total_forecast_30days": total_forecast,
        "avg_daily": round(avg_daily, 1),
        "safety_stock": safety_stock,
        "recommended_stock": recommended,
        "is_drift": bool(is_drift),
        "drift_ratio": round(float(drift_ratio), 1),
        "history_dates": dates[-60:],
        "history_sales": [int(x) for x in sales[-60:]],
        "forecast_dates": [d.strftime('%Y-%m-%d') for d in pd.date_range(
            pd.to_datetime(dates[-1]) + timedelta(days=1), periods=forecast_days)],
        "forecast_values": [int(x) for x in pred],
        "test_actual": [int(x) for x in test],
    }


# ============================================================
# 全量5000商品多模型预测
# ============================================================
def main():
    log_path = os.path.join(BASE_DIR, "logs", "multi_model_forecast.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, 'w', encoding='utf-8')
    class T:
        def write(self, s): log.write(s); log.flush()
        def flush(self): log.flush()
    sys.stdout = T(); sys.stderr = T()

    df = pd.read_csv(os.path.join(DATA_DIR, 'sales_data_5000.csv'), parse_dates=['date'])
    print(f"数据: {len(df)}行, {df['drug_name'].nunique()}商品")

    drugs = list(df['drug_name'].unique())
    print(f"开始多模型预测 {len(drugs)} 个商品（每商品跑4模型选优）...")

    all_results = {}
    model_win_count = {"ExponentialSmoothing": 0, "SeasonalNaive": 0,
                       "MovingAverage": 0, "LinearRegression": 0, "Ensemble": 0}
    t_start = time.time()

    for i, drug in enumerate(drugs):
        try:
            result = forecast_product_multi_model(drug, df[df['drug_name'] == drug])
            all_results[drug] = result
            if result.get("model_used"):
                model_win_count[result["model_used"]] = model_win_count.get(result["model_used"], 0) + 1
        except Exception as e:
            all_results[drug] = {"drug_name": drug, "error": str(e)[:60], "mape": 999}

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            ok = sum(1 for v in all_results.values() if v.get("mape", 999) < 999)
            print(f"  进度 {i+1}/{len(drugs)} 成功{ok} 耗时{elapsed:.0f}秒", flush=True)

    total_time = time.time() - t_start
    valid = [v for v in all_results.values() if v.get("mape", 999) < 999]
    mapes = [v['mape'] for v in valid]

    print(f"\n{'='*60}")
    print(f"  多模型选优预测完成")
    print(f"{'='*60}")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"成功: {len(valid)} 失败: {5000-len(valid)}")
    print(f"平均MAPE: {np.mean(mapes):.1f}%")
    print(f"\n各模型胜出次数:")
    for name, cnt in sorted(model_win_count.items(), key=lambda x: -x[1]):
        pct = cnt / len(valid) * 100 if valid else 0
        print(f"  {name}: {cnt}次 ({pct:.1f}%)")

    # MAPE分布
    exc = sum(1 for m in mapes if m < 15)
    good = sum(1 for m in mapes if 15 <= m < 25)
    fair = sum(1 for m in mapes if 25 <= m < 35)
    poor = sum(1 for m in mapes if m >= 35)
    print(f"\nMAPE分布:")
    print(f"  优秀(<15%): {exc} | 良好(15-25%): {good} | 一般(25-35%): {fair} | 需改进(>35%): {poor}")

    # 按模型分组的MAPE
    model_mapes = {}
    for v in valid:
        m = v.get("model_used", "unknown")
        if m not in model_mapes:
            model_mapes[m] = []
        model_mapes[m].append(v["mape"])
    print(f"\n各模型平均MAPE（只在胜出的商品上）:")
    for m, ms in model_mapes.items():
        print(f"  {m}: {np.mean(ms):.1f}% (n={len(ms)})")

    # 对比单模型ExponentialSmoothing
    print(f"\n对比:")
    print(f"  单模型(ExponentialSmoothing) MAPE: 14.2%")
    print(f"  多模型选优 MAPE: {np.mean(mapes):.1f}%")
    improvement = 14.2 - np.mean(mapes)
    print(f"  提升: {improvement:.1f}个百分点")

    # 保存
    out_path = os.path.join(DATA_DIR, 'forecast_5000_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False)
    print(f"\n保存到: {out_path}")
    print(f"文件大小: {os.path.getsize(out_path)/1024/1024:.1f}MB")
    print("DONE")
    log.close()


if __name__ == "__main__":
    main()
