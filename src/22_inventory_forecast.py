"""
22_inventory_forecast.py
AI库存预测demo - 药店药品销量预测

跟RAG的区别：
  RAG = 查过去的事实（药品怎么吃）
  预测 = 算未来的数字（下月卖多少）

完整流程：
  1. 生成模拟药店销量数据（含季节性+趋势+节假日效应）
  2. 特征工程（滞后特征/滚动均值/星期/月份）
  3. Prophet时序预测（预测未来30天销量）
  4. 评估（MAPE/RMSE）
  5. 库存建议（预测销量+安全库存=补货量）
  6. 数据漂移检测
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(BASE_DIR, "data")
FORECAST_DIR = os.path.join(BASE_DIR, "logs")


# ============================================================
# 第1步：生成模拟药店销量数据
# ============================================================
def generate_sales_data():
    """
    生成1年365天的药店销量数据
    包含：趋势+季节性+星期效应+节假日效应+随机噪声
    模拟50种常见药品（覆盖感冒/肠胃/心脑血管/维生素/外用/儿科等）
    """
    print("=" * 60)
    print("第1步：生成模拟药店销量数据（50种药品）")
    print("=" * 60)

    np.random.seed(42)
    dates = pd.date_range('2025-01-01', '2025-12-31', freq='D')

    # 50种药品，按品类分组，各有不同销量模式
    drug_configs = {}

    # 感冒/退烧类（冬季高卖，10种）
    cold_drugs = [
        "感冒灵颗粒", "复方氨酚烷胺片", "布洛芬缓释胶囊", "对乙酰氨基酚片",
        "小儿感冒颗粒", "双黄连口服液", "板蓝根颗粒", "连花清瘟胶囊",
        "酚麻美敏片", "氨酚伪麻美芬片",
    ]
    for d in cold_drugs:
        drug_configs[d] = {"base": np.random.randint(15, 40), "trend": np.random.uniform(0.005, 0.03),
                           "seasonality": True, "season": "winter", "weekend_boost": np.random.uniform(1.1, 1.4)}

    # 肠胃类（春节/节假日高卖，10种）
    stomach_drugs = [
        "健胃消食片", "奥美拉唑肠溶胶囊", "多潘立酮片", "蒙脱石散",
        "藿香正气水", "复方消化酶胶囊", "铝碳酸镁片", "乳酸菌素片",
        "保济丸", "整肠生胶囊",
    ]
    for d in stomach_drugs:
        drug_configs[d] = {"base": np.random.randint(10, 30), "trend": np.random.uniform(0.005, 0.02),
                           "seasonality": True, "season": "holiday", "weekend_boost": np.random.uniform(1.2, 1.5)}

    # 心脑血管类（稳定销量，处方药慢病，10种）
    cardio_drugs = [
        "硝苯地平控释片", "厄贝沙坦片", "阿托伐他汀钙片", "美托洛尔缓释片",
        "氯吡格雷片", "缬沙坦胶囊", "非洛地平缓释片", "瑞舒伐他汀钙片",
        "单硝酸异山梨酯片", "地高辛片",
    ]
    for d in cardio_drugs:
        drug_configs[d] = {"base": np.random.randint(8, 20), "trend": np.random.uniform(0.002, 0.01),
                           "seasonality": False, "season": "none", "weekend_boost": np.random.uniform(1.0, 1.1)}

    # 维生素/保健品类（稳定，冬季略高，10种）
    vitamin_drugs = [
        "维生素C片", "复合维生素B片", "维生素D滴剂", "钙尔奇D片",
        "葡萄糖酸锌口服液", "蛋白粉", "鱼油胶囊", "辅酶Q10胶囊",
        "氨糖软骨素片", "益生菌粉",
    ]
    for d in vitamin_drugs:
        drug_configs[d] = {"base": np.random.randint(12, 35), "trend": np.random.uniform(0.003, 0.015),
                           "seasonality": True, "season": "winter_mild", "weekend_boost": np.random.uniform(1.1, 1.3)}

    # 外用/儿科/其他类（稳定，10种）
    other_drugs = [
        "创可贴(盒)", "碘伏(瓶)", "开塞露", "红霉素软膏",
        "炉甘石洗剂", "退热贴(盒)", "小儿止咳糖浆", "丁桂儿脐贴",
        "清凉油", "风油精",
    ]
    for d in other_drugs:
        drug_configs[d] = {"base": np.random.randint(8, 25), "trend": np.random.uniform(0.002, 0.012),
                           "seasonality": False, "season": "none", "weekend_boost": np.random.uniform(1.0, 1.3)}

    all_data = []
    for drug_name, config in drug_configs.items():
        for i, date in enumerate(dates):
            sales = config["base"]
            sales += config["trend"] * i

            # 季节性
            if config["season"] == "winter":
                month = date.month
                if month in [1, 2, 3, 11, 12]:
                    sales *= 1.5
                elif month in [7, 8]:
                    sales *= 0.8
            elif config["season"] == "winter_mild":
                month = date.month
                if month in [1, 2, 12]:
                    sales *= 1.2
            elif config["season"] == "holiday":
                if date.month == 1 and date.day <= 7:
                    sales *= 1.6
                if date.month == 10 and 1 <= date.day <= 7:
                    sales *= 1.3
                if date.month == 12 and date.day >= 25:
                    sales *= 1.4

            # 星期效应
            if date.weekday() >= 5:
                sales *= config["weekend_boost"]

            # 噪声
            sales += np.random.normal(0, max(2, config["base"] * 0.1))
            sales = max(0, int(sales))

            all_data.append({
                "date": date,
                "drug_name": drug_name,
                "sales": sales,
                "day_of_week": date.weekday(),
                "month": date.month,
                "is_weekend": 1 if date.weekday() >= 5 else 0,
            })

    df = pd.DataFrame(all_data)
    print(f"  生成数据: {len(df)}条, {len(drug_configs)}种药品, 365天")

    # 按品类统计
    categories = {
        "感冒退烧类": cold_drugs,
        "肠胃类": stomach_drugs,
        "心脑血管类": cardio_drugs,
        "维生素保健类": vitamin_drugs,
        "外用儿科类": other_drugs,
    }
    print(f"\n  各品类统计:")
    for cat, drugs in categories.items():
        cat_df = df[df['drug_name'].isin(drugs)]
        print(f"    {cat}: {len(drugs)}种, 日均总销{cat_df.groupby('date')['sales'].sum().mean():.0f}盒")

    print(f"\n  各药品销量TOP10:")
    stats = df.groupby('drug_name')['sales'].agg(['mean', 'sum']).sort_values('sum', ascending=False)
    print(stats.head(10).to_string())

    path = os.path.join(DATA_DIR, "sales_data.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n  保存到: {path}")

    return df


# ============================================================
# 第2步：Prophet时序预测
# ============================================================
def forecast_with_prophet(df, drug_name, forecast_days=30):
    """
    用Prophet预测某药品未来30天销量
    Prophet擅长：趋势+季节性+节假日效应
    """
    from prophet import Prophet

    # 准备数据（Prophet要求列名ds和y）
    drug_df = df[df['drug_name'] == drug_name][['date', 'sales']].copy()
    drug_df.columns = ['ds', 'y']

    # 训练集：前335天，测试集：最后30天（算MAPE用）
    train = drug_df.iloc[:-forecast_days]
    test = drug_df.iloc[-forecast_days:]

    # 训练Prophet
    model = Prophet(
        yearly_seasonality=True,   # 年度季节性
        weekly_seasonality=True,   # 周季节性
        daily_seasonality=False,   # 日内不需要
        changepoint_prior_scale=0.05,  # 趋势变化灵活度
    )

    t0 = time.time()
    model.fit(train)
    t1 = time.time()

    # 预测未来30天
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # 取预测部分
    predicted = forecast.iloc[-forecast_days:]['yhat'].values
    actual = test['y'].values

    # 评估
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return {
        "drug_name": drug_name,
        "train_time": t1 - t0,
        "mape": mape,
        "rmse": rmse,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
        "forecast_next": forecast.iloc[-forecast_days:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
        "test_dates": test['ds'].dt.strftime('%Y-%m-%d').tolist(),
    }


# ============================================================
# 第3步：库存建议
# ============================================================
def inventory_recommendation(forecast_result, safety_days=7):
    """
    根据预测销量给库存建议
    补货量 = 预测销量 + 安全库存 - 当前库存
    """
    forecast_next = forecast_result["forecast_next"]
    drug_name = forecast_result["drug_name"]

    # 未来30天预测总销量
    total_forecast = sum(f['yhat'] for f in forecast_next)
    avg_daily = total_forecast / len(forecast_next)

    # 安全库存（安全天数 × 日均销量）
    safety_stock = safety_days * avg_daily

    # 建议库存量 = 预测销量 + 安全库存
    recommended_stock = int(total_forecast + safety_stock)

    # 按周分解
    weekly_forecast = []
    for week in range(4):
        week_sales = sum(f['yhat'] for f in forecast_next[week*7:(week+1)*7])
        weekly_forecast.append({
            "week": week + 1,
            "predicted_sales": int(week_sales),
        })

    return {
        "drug_name": drug_name,
        "total_forecast_30days": int(total_forecast),
        "avg_daily": round(avg_daily, 1),
        "safety_stock": int(safety_stock),
        "recommended_stock": recommended_stock,
        "weekly_breakdown": weekly_forecast,
        "safety_days": safety_days,
    }


# ============================================================
# 第4步：数据漂移检测
# ============================================================
def detect_data_drift(df, drug_name, window_recent=30, window_baseline=90):
    """
    检测数据漂移：最近30天 vs 基线90天的销量分布
    如果分布明显偏移，说明消费习惯变了，模型可能需要重训
    """
    drug_df = df[df['drug_name'] == drug_name]

    recent = drug_df.tail(window_recent)['sales']
    baseline = drug_df.iloc[-window_recent-window_baseline:-window_recent]['sales']

    recent_mean = recent.mean()
    baseline_mean = baseline.mean()

    # 偏移率
    drift_ratio = (recent_mean - baseline_mean) / baseline_mean * 100

    # 简单t检验
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(recent, baseline)
    except Exception:
        p_value = 0.0

    is_drift = abs(drift_ratio) > 20 and p_value < 0.05  # 偏移>20%且统计显著

    return {
        "drug_name": drug_name,
        "recent_mean": round(recent_mean, 1),
        "baseline_mean": round(baseline_mean, 1),
        "drift_ratio": round(drift_ratio, 1),
        "p_value": round(p_value, 4),
        "is_drift": is_drift,
        "recommendation": "需要重训模型" if is_drift else "模型正常",
    }


# ============================================================
# 完整流程
# ============================================================
def main():
    print("=" * 60)
    print("  AI库存预测demo - 药店药品销量预测")
    print("  对应JD第2条：智能库存预测")
    print("=" * 60)

    # 第1步：生成数据
    df = generate_sales_data()

    # 第2步：50种药品分别预测
    print(f"\n{'='*60}")
    print(f"第2步：Prophet时序预测（50种药品，未来30天）")
    print(f"{'='*60}")

    drug_list = df['drug_name'].unique()
    all_results = {}
    t_total_start = time.time()

    for idx, drug in enumerate(drug_list):
        result = forecast_with_prophet(df, drug, forecast_days=30)
        all_results[drug] = result
        if (idx + 1) % 10 == 0:
            print(f"  已完成 {idx+1}/{len(drug_list)}种...", flush=True)

    t_total = time.time() - t_total_start
    print(f"  全部完成: {len(drug_list)}种药品, 总耗时{t_total:.1f}s")

    # 汇总MAPE
    print(f"\n{'─'*60}")
    print(f"预测准确率汇总（按MAPE排序）:")
    print(f"{'药品':<25} {'MAPE':<8} {'RMSE':<8} {'评价':<8}")
    print(f"{'─'*60}")

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mape'])
    for drug, r in sorted_results:
        rating = "优秀" if r['mape'] < 15 else "良好" if r['mape'] < 25 else "一般" if r['mape'] < 35 else "需改进"
        print(f"{drug:<25} {r['mape']:.1f}%{'':<3} {r['rmse']:.1f}{'':<4} {rating}")

    avg_mape = np.mean([r['mape'] for r in all_results.values()])
    avg_rmse = np.mean([r['rmse'] for r in all_results.values()])
    excellent = sum(1 for r in all_results.values() if r['mape'] < 15)
    good = sum(1 for r in all_results.values() if 15 <= r['mape'] < 25)
    fair = sum(1 for r in all_results.values() if 25 <= r['mape'] < 35)
    poor = sum(1 for r in all_results.values() if r['mape'] >= 35)
    print(f"{'─'*60}")
    print(f"{'平均':<25} {avg_mape:.1f}%{'':<3} {avg_rmse:.1f}")
    print(f"{'='*60}")
    print(f"  优秀(<15%): {excellent}种 | 良好(15-25%): {good}种 | 一般(25-35%): {fair}种 | 需改进(>35%): {poor}种")

    # 第3步：库存建议（汇总统计）
    print(f"\n{'='*60}")
    print(f"第3步：库存补货建议（50种药品汇总）")
    print(f"{'='*60}")

    all_inventory = {}
    for drug in drug_list:
        inv = inventory_recommendation(all_results[drug], safety_days=7)
        all_inventory[drug] = inv

    # TOP10补货量
    print(f"\n  补货量TOP10（需要备货最多的药品）:")
    print(f"  {'药品':<25} {'预测30天销量':<12} {'日均':<8} {'建议库存':<10}")
    print(f"  {'─'*60}")
    sorted_inv = sorted(all_inventory.items(), key=lambda x: x[1]['recommended_stock'], reverse=True)
    for drug, inv in sorted_inv[:10]:
        print(f"  {drug:<25} {inv['total_forecast_30days']:<12} {inv['avg_daily']:<8} {inv['recommended_stock']:<10}")

    total_forecast_all = sum(inv['total_forecast_30days'] for inv in all_inventory.values())
    total_stock_all = sum(inv['recommended_stock'] for inv in all_inventory.values())
    print(f"  {'─'*60}")
    print(f"  {'50种合计':<25} {total_forecast_all:<12} {'':<8} {total_stock_all:<10}")

    # 第4步：数据漂移检测（汇总）
    print(f"\n{'='*60}")
    print(f"第4步：数据漂移检测（50种药品）")
    print(f"{'='*60}")

    drift_count = 0
    all_drift = {}
    for drug in drug_list:
        drift = detect_data_drift(df, drug)
        all_drift[drug] = drift
        if drift['is_drift']:
            drift_count += 1

    print(f"\n  漂移检测: {drift_count}/{len(drug_list)}种检测到漂移")
    print(f"  正常: {len(drug_list) - drift_count}种")
    print(f"\n  漂移的药品（需要重训模型）:")
    for drug, d in all_drift.items():
        if d['is_drift']:
            print(f"    ⚠️ {drug}: 偏移{d['drift_ratio']}% (最近{d['recent_mean']} vs 基线{d['baseline_mean']})")

    # 保存预测结果
    save_results = {}
    for drug, r in all_results.items():
        save_results[drug] = {
            "mape": r["mape"],
            "rmse": r["rmse"],
            "inventory": inventory_recommendation(r),
            "drift": detect_data_drift(df, drug),
        }

    save_path = os.path.join(FORECAST_DIR, "forecast_result.json")
    os.makedirs(FORECAST_DIR, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n预测结果已保存: {save_path}")

    # 总结
    print(f"\n{'='*60}")
    print(f"AI预测总结")
    print(f"{'='*60}")
    print(f"""
【跟RAG的本质区别】
  RAG = 查过去（药品怎么吃）→ 100%准确
  预测 = 算未来（下月卖多少）→ 概率性，MAPE {avg_mape:.1f}%算{'优秀' if avg_mape<20 else '良好'}

【本次规模】
  50种药品，5个品类，365天历史数据
  优秀(<15%): {excellent}种 | 良好(15-25%): {good}种 | 一般(25-35%): {fair}种 | 需改进(>35%): {poor}种
  漂移检测: {drift_count}种需要重训
  50种合计预测30天销量: {total_forecast_all}盒, 建议总库存: {total_stock_all}盒

【预测全流程】
  1. 历史数据（365天销量）
  2. 特征：趋势+季节性+星期效应+节假日
  3. Prophet模型训练
  4. 预测未来30天
  5. 评估MAPE/RMSE
  6. 库存建议（预测+安全库存=补货量）
  7. 数据漂移检测（消费习惯变了要重训）

【关键指标解读】
  MAPE = 平均绝对百分比误差
    <15% = 优秀（预测很准）
    15-25% = 良好（可用于补货决策）
    25-35% = 一般（需优化）
    >35% = 需改进（模型或数据有问题）
  本次平均MAPE: {avg_mape:.1f}%

【库存建议公式】
  建议库存 = 预测30天销量 + 安全库存(7天×日均)
  安全库存 = 应对预测偏差的缓冲

【数据漂移】
  最近的销量分布跟历史基线对比
  偏移>20%且统计显著 → 需要重训模型
  原因：季节变化/消费习惯变/新竞品/疫情等

【为什么用Prophet不用LLM】
  Prophet = 专门做时序预测的ML模型，擅长趋势+季节性
  LLM = 语言模型，不擅长数字预测
  预测类问题用专门的ML模型，不用LLM

【医药连锁库存预测的业务价值】
  • 减少过期损耗（药品有有效期，备多过期）
  • 减少断货（备少丢销售）
  • 千店规模人工订货不现实
  • 季节性药品提前备货（感冒药冬天多）
""")


if __name__ == "__main__":
    main()
