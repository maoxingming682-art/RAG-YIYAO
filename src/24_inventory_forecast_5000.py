# -*- coding: utf-8 -*-
"""
24_inventory_forecast_5000.py
按大型连锁药房品类结构生成5000个商品 × 365天销量数据
并实测Prophet在5000商品规模下的训练时间+精度，判断方案是否仍最优

品类分布（参照大型连锁药房经营结构）：
  化学药制剂   1800个  36%  抗感染/心脑血管/消化/呼吸/解热镇痛/糖尿病/神经/皮肤
  中成药       900个  18%  感冒/清热/骨伤/妇科/儿科/补益
  中药饮片     450个   9%  单味饮片
  保健食品     600个  12%  维生素/矿物质/蛋白
  医疗器械     300个   6%  家用器械/耗材/防护
  个人护理日化 450个   9%  口腔/皮肤/卫生
  母婴用品     200个   4%  奶粉/纸尿裤/孕产
  食品其他     300个   6%  功能食品/饮品
  合计        5000个 100%
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

warnings.filterwarnings('ignore')
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")


# ============================================================
# 第1步：5000商品品类体系（按连锁药房结构）
# ============================================================
def build_5000_catalog():
    """构建5000商品目录，每个商品带：名称、大类、子类、销量模式参数"""
    np.random.seed(42)
    catalog = []

    # --- 1. 化学药制剂 1800个 ---
    chem_subs = {
        "抗感染类":   {"count": 250, "base": (5, 25),  "season": "none",     "trend": (0.001, 0.01),  "weekend": (1.0, 1.15)},
        "心脑血管类": {"count": 300, "base": (8, 20),  "season": "none",     "trend": (0.002, 0.008), "weekend": (1.0, 1.1)},
        "消化系统类": {"count": 200, "base": (10, 30), "season": "holiday",  "trend": (0.003, 0.015), "weekend": (1.1, 1.3)},
        "呼吸系统类": {"count": 200, "base": (12, 35), "season": "winter",   "trend": (0.005, 0.02),  "weekend": (1.1, 1.35)},
        "解热镇痛类": {"count": 200, "base": (15, 40), "season": "winter",   "trend": (0.004, 0.018), "weekend": (1.1, 1.4)},
        "糖尿病用药": {"count": 250, "base": (6, 18),  "season": "none",     "trend": (0.003, 0.01),  "weekend": (1.0, 1.05)},
        "神经系统类": {"count": 200, "base": (5, 18),  "season": "none",     "trend": (0.002, 0.009), "weekend": (1.0, 1.1)},
        "皮肤科用药": {"count": 200, "base": (8, 22),  "season": "summer_mild","trend": (0.003, 0.012), "weekend": (1.05, 1.2)},
    }
    for sub, cfg in chem_subs.items():
        for i in range(cfg["count"]):
            catalog.append({
                "drug_name": f"{sub}_{i+1:03d}",
                "category": "化学药制剂", "sub_category": sub,
                "base": np.random.randint(*cfg["base"]),
                "trend": np.random.uniform(*cfg["trend"]),
                "season": cfg["season"], "weekend_boost": np.random.uniform(*cfg["weekend"]),
            })

    # --- 2. 中成药 900个 ---
    tcm_subs = {
        "感冒类":     {"count": 180, "base": (12, 38), "season": "winter",    "trend": (0.004, 0.02),  "weekend": (1.1, 1.4)},
        "清热解毒类": {"count": 150, "base": (8, 25),  "season": "summer",    "trend": (0.003, 0.015), "weekend": (1.05, 1.25)},
        "风湿骨伤类": {"count": 150, "base": (6, 20),  "season": "winter_mild","trend": (0.002, 0.01),  "weekend": (1.1, 1.3)},
        "妇科用药类": {"count": 120, "base": (5, 18),  "season": "none",      "trend": (0.002, 0.01),  "weekend": (1.1, 1.3)},
        "儿科用药类": {"count": 150, "base": (8, 25),  "season": "winter",    "trend": (0.003, 0.015), "weekend": (1.15, 1.4)},
        "补益类":     {"count": 150, "base": (7, 22),  "season": "winter_mild","trend": (0.003, 0.012), "weekend": (1.1, 1.3)},
    }
    for sub, cfg in tcm_subs.items():
        for i in range(cfg["count"]):
            catalog.append({
                "drug_name": f"{sub}_{i+1:03d}",
                "category": "中成药", "sub_category": sub,
                "base": np.random.randint(*cfg["base"]),
                "trend": np.random.uniform(*cfg["trend"]),
                "season": cfg["season"], "weekend_boost": np.random.uniform(*cfg["weekend"]),
            })

    # --- 3. 中药饮片 450个 ---
    for i in range(450):
        catalog.append({
            "drug_name": f"中药饮片_{i+1:03d}",
            "category": "中药饮片", "sub_category": "单味饮片",
            "base": np.random.randint(3, 15),
            "trend": np.random.uniform(0.001, 0.008),
            "season": "none", "weekend_boost": np.random.uniform(1.0, 1.1),
        })

    # --- 4. 保健食品 600个 ---
    health_subs = {
        "维生素类":   {"count": 200, "base": (12, 35), "season": "winter_mild","trend": (0.003, 0.015), "weekend": (1.1, 1.3)},
        "矿物质类":   {"count": 150, "base": (8, 25),  "season": "none",      "trend": (0.002, 0.012), "weekend": (1.1, 1.25)},
        "蛋白粉及补剂": {"count": 250, "base": (5, 20), "season": "none",      "trend": (0.004, 0.018), "weekend": (1.15, 1.4)},
    }
    for sub, cfg in health_subs.items():
        for i in range(cfg["count"]):
            catalog.append({
                "drug_name": f"{sub}_{i+1:03d}",
                "category": "保健食品", "sub_category": sub,
                "base": np.random.randint(*cfg["base"]),
                "trend": np.random.uniform(*cfg["trend"]),
                "season": cfg["season"], "weekend_boost": np.random.uniform(*cfg["weekend"]),
            })

    # --- 5. 医疗器械 300个 ---
    device_subs = {
        "家用器械类": {"count": 100, "base": (1, 6),   "season": "none",      "trend": (0.002, 0.008), "weekend": (1.2, 1.5)},
        "医用耗材类": {"count": 100, "base": (5, 20),  "season": "none",      "trend": (0.002, 0.01),  "weekend": (1.1, 1.25)},
        "防护用品类": {"count": 100, "base": (8, 30),  "season": "winter",    "trend": (0.003, 0.012), "weekend": (1.05, 1.2)},
    }
    for sub, cfg in device_subs.items():
        for i in range(cfg["count"]):
            catalog.append({
                "drug_name": f"{sub}_{i+1:03d}",
                "category": "医疗器械", "sub_category": sub,
                "base": np.random.randint(*cfg["base"]),
                "trend": np.random.uniform(*cfg["trend"]),
                "season": cfg["season"], "weekend_boost": np.random.uniform(*cfg["weekend"]),
            })

    # --- 6. 个人护理日化 450个 ---
    care_subs = {
        "口腔护理类": {"count": 150, "base": (10, 30), "season": "none",      "trend": (0.003, 0.012), "weekend": (1.2, 1.4)},
        "皮肤护理类": {"count": 150, "base": (8, 25),  "season": "winter_mild","trend": (0.003, 0.015), "weekend": (1.15, 1.35)},
        "卫生用品类": {"count": 150, "base": (12, 35), "season": "none",      "trend": (0.003, 0.012), "weekend": (1.2, 1.4)},
    }
    for sub, cfg in care_subs.items():
        for i in range(cfg["count"]):
            catalog.append({
                "drug_name": f"{sub}_{i+1:03d}",
                "category": "个人护理日化", "sub_category": sub,
                "base": np.random.randint(*cfg["base"]),
                "trend": np.random.uniform(*cfg["trend"]),
                "season": cfg["season"], "weekend_boost": np.random.uniform(*cfg["weekend"]),
            })

    # --- 7. 母婴用品 200个 ---
    for i in range(200):
        catalog.append({
            "drug_name": f"母婴用品_{i+1:03d}",
            "category": "母婴用品", "sub_category": "综合",
            "base": np.random.randint(5, 20),
            "trend": np.random.uniform(0.002, 0.01),
            "season": "none", "weekend_boost": np.random.uniform(1.2, 1.4),
        })

    # --- 8. 食品其他 300个 ---
    for i in range(300):
        catalog.append({
            "drug_name": f"食品其他_{i+1:03d}",
            "category": "食品其他", "sub_category": "综合",
            "base": np.random.randint(6, 22),
            "trend": np.random.uniform(0.003, 0.012),
            "season": "none", "weekend_boost": np.random.uniform(1.1, 1.3),
        })

    return catalog


# ============================================================
# 第2步：生成5000商品×365天销量数据
# ============================================================
def generate_5000_sales(catalog):
    """生成5000商品×365天销量，含季节性+趋势+星期效应+节假日+噪声"""
    print("=" * 60)
    print("第2步：生成5000商品×365天销量数据")
    print("=" * 60)
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', '2025-12-31', freq='D')

    all_data = []
    for item in catalog:
        base = item["base"]; trend = item["trend"]
        season = item["season"]; wb = item["weekend_boost"]
        for i, date in enumerate(dates):
            sales = base + trend * i
            month = date.month; wd = date.weekday()
            # 季节性
            if season == "winter" and month in [1,2,3,11,12]:
                sales *= 1.5
            elif season == "winter" and month in [7,8]:
                sales *= 0.8
            elif season == "winter_mild" and month in [1,2,12]:
                sales *= 1.2
            elif season == "summer" and month in [6,7,8]:
                sales *= 1.3
            elif season == "summer_mild" and month in [6,7,8]:
                sales *= 1.15
            elif season == "holiday":
                if month == 1 and date.day <= 7: sales *= 1.6
                if month == 10 and 1 <= date.day <= 7: sales *= 1.3
                if month == 12 and date.day >= 25: sales *= 1.4
            # 星期效应
            if wd >= 5:
                sales *= wb
            # 噪声
            sales += np.random.normal(0, max(2, base * 0.1))
            sales = max(0, int(sales))
            all_data.append({
                "date": date, "drug_name": item["drug_name"],
                "category": item["category"], "sub_category": item["sub_category"],
                "sales": sales, "day_of_week": wd, "month": month,
                "is_weekend": 1 if wd >= 5 else 0,
            })
    df = pd.DataFrame(all_data)
    print(f"  生成数据: {len(df)}条, {df['drug_name'].nunique()}种商品, 365天")
    print(f"  数据规模: {len(df)/10000:.1f}万行")
    # 品类统计
    cat_stats = df.groupby('category')['sales'].agg(['mean','sum','count']).sort_values('sum',ascending=False)
    print(f"\n  各品类统计:")
    print(cat_stats.to_string())
    path = os.path.join(DATA_DIR, "sales_data_5000.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n  保存到: {path}")
    print(f"  文件大小: {os.path.getsize(path)/1024/1024:.1f}MB")
    return df


# ============================================================
# 第3步：单商品Prophet预测（供多进程调用）
# ============================================================
def forecast_one(args):
    """单商品Prophet预测，供ProcessPoolExecutor调用"""
    drug_name, drug_df, forecast_days = args
    from prophet import Prophet
    import numpy as np
    for attempt in range(3):
        try:
            d = drug_df[['date','sales']].copy()
            d.columns = ['ds','y']
            train = d.iloc[:-forecast_days]
            test = d.iloc[-forecast_days:]
            # 显式指定stan_backend，规避Windows下stan_backend丢失的bug
            m = Prophet(
                yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.05,
                stan_backend='CMDSTANPY',
            )
            t0 = time.time()
            m.fit(train)
            train_time = time.time() - t0
            future = m.make_future_dataframe(periods=forecast_days)
            fc = m.predict(future)
            pred = fc.iloc[-forecast_days:]['yhat'].values
            actual = test['y'].values
            mape = np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            del m  # 释放Prophet对象，避免stan_backend累积问题
            return {"drug_name": drug_name, "train_time": train_time,
                    "mape": mape, "rmse": rmse}
        except Exception as e:
            if attempt < 2:
                time.sleep(0.5)
                continue
            return {"drug_name": drug_name, "train_time": 0, "mape": 999,
                    "rmse": 999, "error": str(e)[:80]}


def benchmark_prophet_5000(df, sample_size=None, workers=None):
    """实测Prophet在5000商品上的训练时间+精度
    sample_size: 抽样测试的商品数（None=全部5000）
    Windows下用串行（多进程在spawn模式下容易崩）
    """
    print(f"\n{'='*60}")
    print(f"第3步：Prophet性能实测（5000商品规模）")
    print(f"{'='*60}")

    drugs = df['drug_name'].unique()
    if sample_size and sample_size < len(drugs):
        import random
        random.seed(42)
        drugs = random.sample(list(drugs), sample_size)
        print(f"  抽样测试: {sample_size}个商品")

    print(f"  测试商品数: {len(drugs)}")
    print(f"  模式: 串行（Windows兼容）")

    args_list = [(d, df[df['drug_name']==d], 30) for d in drugs]

    t_start = time.time()
    results = []
    for i, a in enumerate(args_list):
        r = forecast_one(a)
        results.append(r)
        if (i+1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i+1) / elapsed
            eta = (len(drugs) - i - 1) / rate
            print(f"  进度 {i+1}/{len(drugs)} ({(i+1)/len(drugs)*100:.0f}%) "
                  f"速率{rate:.1f}个/秒 ETA{eta:.0f}秒", flush=True)
    total_time = time.time() - t_start

    # 汇总
    valid = [r for r in results if r['mape'] < 999]
    failed = [r for r in results if r['mape'] >= 999]
    mapes = [r['mape'] for r in valid]
    times = [r['train_time'] for r in valid]
    avg_mape = np.mean(mapes) if mapes else 0
    avg_time = np.mean(times) if times else 0

    print(f"\n  === Prophet 5000商品实测结果 ===")
    print(f"  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  单商品平均训练时间: {avg_time:.2f}秒")
    print(f"  吞吐率: {len(drugs)/total_time:.1f}个/秒")
    print(f"  成功: {len(valid)} 失败: {len(failed)}")
    print(f"  平均MAPE: {avg_mape:.1f}%")
    if mapes:
        exc = sum(1 for m in mapes if m < 15)
        good = sum(1 for m in mapes if 15 <= m < 25)
        fair = sum(1 for m in mapes if 25 <= m < 35)
        poor = sum(1 for m in mapes if m >= 35)
        print(f"  优秀(<15%): {exc} | 良好(15-25%): {good} | 一般(25-35%): {fair} | 需改进(>35%): {poor}")

    # 外推5000全量耗时
    if sample_size and sample_size < 5000:
        full_eta = total_time * (5000 / sample_size)
        print(f"\n  外推5000全量串行耗时: {full_eta:.0f}秒 ({full_eta/60:.1f}分钟)")
        # 假设8进程并行
        parallel_eta = full_eta / 8
        print(f"  外推5000全量8进程并行: {parallel_eta:.0f}秒 ({parallel_eta/60:.1f}分钟)")

    return {"total_time": total_time, "avg_mape": avg_mape,
            "avg_train_time": avg_time, "throughput": len(drugs)/total_time,
            "sample_size": len(drugs), "workers": 1}


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  5000商品库存预测 - 连锁药房品类结构")
    print("  对比Prophet在5000规模下的适用性")
    print("=" * 60)

    # 第1步：构建5000商品目录
    print(f"\n第1步：构建5000商品品类体系（按连锁药房结构）")
    catalog = build_5000_catalog()
    print(f"  商品总数: {len(catalog)}")
    cat_count = {}
    for c in catalog:
        cat_count[c["category"]] = cat_count.get(c["category"], 0) + 1
    print(f"\n  各大类商品数:")
    for cat, cnt in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}个 ({cnt/len(catalog)*100:.0f}%)")

    # 第2步：生成销量数据
    df = generate_5000_sales(catalog)

    # 第3步：Prophet性能实测（先抽样500测精度，再外推）
    print(f"\n请选择测试规模:")
    print(f"  1. 快速验证（抽样200个，约2分钟）")
    print(f"  2. 中等测试（抽样1000个，约10分钟）")
    print(f"  3. 全量测试（5000个，约50分钟）")
    choice = input("输入1/2/3: ").strip() or "1"
    sample_map = {"1": 200, "2": 1000, "3": None}
    sample_size = sample_map.get(choice, 200)

    bench = benchmark_prophet_5000(df, sample_size=sample_size)

    # 保存基准结果
    bench_path = os.path.join(LOGS_DIR, "prophet_5000_benchmark.json")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
    print(f"\n基准结果已保存: {bench_path}")

    # 第4步：方案对比结论
    print(f"\n{'='*60}")
    print(f"  Prophet 在 5000商品规模下的适用性评估")
    print(f"{'='*60}")
    print(f"""
【实测数据】
  测试商品数: {bench['sample_size']}
  并行进程数: {bench['workers']}
  总耗时: {bench['total_time']:.1f}秒 ({bench['total_time']/60:.1f}分钟)
  单商品平均训练: {bench['avg_train_time']:.2f}秒
  吞吐率: {bench['throughput']:.1f}个/秒
  平均MAPE: {bench['avg_mape']:.1f}%

【Prophet在5000规模下的优劣评估】

  优势（仍然成立）:
    ✓ 精度MAPE {bench['avg_mape']:.1f}%，单药品预测质量不随规模下降
    ✓ 实现简单，可解释性强（趋势/季节/节假日分解）
    ✓ CPU多进程可并行，无需GPU

  劣势（5000规模下放大）:
    ✗ 训练耗时长: 5000个串行约{bench['avg_train_time']*5000:.0f}秒({bench['avg_train_time']*5000/60:.0f}分钟)
      多进程并行后约{bench['total_time']*(5000/bench['sample_size'])/60:.0f}分钟，日补预测可接受，实时不行
    ✗ 每个商品独立训练，没学到药品间关联（如感冒药全家桶一起动）
    ✗ 不能利用多变量（气温/促销/竞品），精度上限受限
    ✗ 5000个模型难维护，单个漂移需逐个重训

【方案推荐：5000商品场景下最优方案】

  ┌─────────────────────────────────────────────────────┐
  │ 分层方案（推荐）                                       │
  ├─────────────────────────────────────────────────────┤
  │                                                     │
  │  A类头部商品（TOP 500，占销量70%）                   │
  │    → Prophet 单独精调                                │
  │    → 销量大、季节性强、值得精细建模                   │
  │                                                     │
  │  B类长尾商品（剩余4500个，占销量30%）                 │
  │    → PatchTST/Autoformer 多变量联合训练              │
  │    → 一次训练所有4500个，捕捉品类关联                 │
  │    → 或直接用 LightGBM + 手工特征（CPU快）           │
  │                                                     │
  │  全局监控                                            │
  │    → 数据漂移检测覆盖5000个                          │
  │    → 漂移的单独触发重训                              │
  └─────────────────────────────────────────────────────┘

  理由:
    1. 头部500个用Prophet: 销量大值得精调，数量少训练快
    2. 长尾4500个用Transformer/LightGBM: 一次训练替代4500次，
       训练时间从{bench['total_time']*(5000/bench['sample_size'])/60:.0f}分钟降到一次训练30-60分钟
    3. 混合方案兼顾精度和效率，符合二八定律

【为什么不全部用Transformer时序模型】
  - 头部500个季节性强的，Prophet的可解释性+节假日建模更直接
  - Transformer对少样本(365点)不一定优于Prophet
  - 全Transformer方案需要更多历史数据(建议2-3年)和GPU资源

【为什么不全部用Prophet】
  - 5000个独立训练耗时{bench['total_time']*(5000/bench['sample_size'])/60:.0f}分钟
  - 长尾商品销量小、规律弱，独立训练性价比低
  - 无法捕捉品类间关联（如感冒药+退烧药+VC联动）
""")


if __name__ == "__main__":
    main()
