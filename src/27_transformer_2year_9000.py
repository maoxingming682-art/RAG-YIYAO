# -*- coding: utf-8 -*-
"""
27_transformer_2year_9000.py
9000商品 × 730天(2年) → Transformer vs ETS 对比

关键改进（vs 26号脚本）：
  1. 商品5000→9000（大型连锁药房实际规模）
  2. 数据365天→730天（2年，Transformer能看到2个完整年度周期）
  3. 输入窗口60天→90天（更多上下文）
  4. 滑动窗口增广：每个商品切多个训练样本（180K总样本 vs 之前5K）
  5. Transformer能学习跨年度的周期模式

假设：2年数据下Transformer应该能反超ETS
"""
import sys, os, json, time, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 日志重定向
log_path = os.path.join(LOGS_DIR, "transformer_2year.log")
os.makedirs(LOGS_DIR, exist_ok=True)
log = open(log_path, 'w', encoding='utf-8')
class T:
    def write(self, s): log.write(s); log.flush()
    def flush(self): log.flush()
sys.stdout = T(); sys.stderr = T()


# ============================================================
# 第1步：生成9000商品×730天数据
# ============================================================
def generate_9000_data():
    """按连锁药房品类×1.8生成9000商品730天(2年)销量数据"""
    print("=" * 60)
    print("  第1步：生成9000商品×730天(2年)销量数据")
    print("=" * 60)
    np.random.seed(42)

    # 品类配置（×1.8 扩展到9000）
    categories = {
        "化学药制剂": {"count": 3240, "subs": {
            "抗感染类": 450, "心脑血管类": 540, "消化系统类": 360,
            "呼吸系统类": 360, "解热镇痛类": 360, "糖尿病用药": 450,
            "神经系统类": 360, "皮肤科用药": 360,
        }, "base_range": (5, 40), "seasons": {
            "抗感染类": "none", "心脑血管类": "none", "消化系统类": "holiday",
            "呼吸系统类": "winter", "解热镇痛类": "winter", "糖尿病用药": "none",
            "神经系统类": "none", "皮肤科用药": "summer_mild",
        }},
        "中成药": {"count": 1620, "subs": {
            "感冒类": 324, "清热解毒类": 270, "风湿骨伤类": 270,
            "妇科用药类": 216, "儿科用药类": 270, "补益类": 270,
        }, "base_range": (5, 38), "seasons": {
            "感冒类": "winter", "清热解毒类": "summer", "风湿骨伤类": "winter_mild",
            "妇科用药类": "none", "儿科用药类": "winter", "补益类": "winter_mild",
        }},
        "保健食品": {"count": 1080, "subs": {
            "维生素类": 360, "矿物质类": 270, "蛋白粉及补剂": 450,
        }, "base_range": (5, 35), "seasons": {
            "维生素类": "winter_mild", "矿物质类": "none", "蛋白粉及补剂": "none",
        }},
        "中药饮片": {"count": 810, "subs": {"单味饮片": 810},
                     "base_range": (3, 15), "seasons": {"单味饮片": "none"}},
        "个人护理日化": {"count": 810, "subs": {
            "口腔护理类": 270, "皮肤护理类": 270, "卫生用品类": 270,
        }, "base_range": (8, 35), "seasons": {
            "口腔护理类": "none", "皮肤护理类": "winter_mild", "卫生用品类": "none",
        }},
        "医疗器械": {"count": 540, "subs": {
            "家用器械类": 180, "医用耗材类": 180, "防护用品类": 180,
        }, "base_range": (1, 30), "seasons": {
            "家用器械类": "none", "医用耗材类": "none", "防护用品类": "winter",
        }},
        "食品其他": {"count": 540, "subs": {"综合": 540},
                     "base_range": (6, 22), "seasons": {"综合": "none"}},
        "母婴用品": {"count": 360, "subs": {"综合": 360},
                     "base_range": (5, 20), "seasons": {"综合": "none"}},
    }

    # 2年日期
    dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')  # 730天
    print(f"  日期范围: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}天)")

    catalog = []
    for cat_name, cat_cfg in categories.items():
        for sub_name, count in cat_cfg["subs"].items():
            season = cat_cfg["seasons"].get(sub_name, "none")
            for i in range(count):
                catalog.append({
                    "drug_name": f"{sub_name}_{i+1:04d}",
                    "category": cat_name, "sub_category": sub_name,
                    "base": np.random.randint(*cat_cfg["base_range"]),
                    "trend": np.random.uniform(0.001, 0.02),
                    "season": season,
                    "weekend_boost": np.random.uniform(1.0, 1.4),
                })

    print(f"  商品总数: {len(catalog)}")

    # 生成销量
    all_data = []
    for item in catalog:
        base = item["base"]; trend = item["trend"]
        season = item["season"]; wb = item["weekend_boost"]
        for i, date in enumerate(dates):
            sales = base + trend * i
            month = date.month; wd = date.weekday()
            # 季节性（2年重复）
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
                if month == 2 and date.day <= 7: sales *= 1.5  # 2025春节
                if month == 10 and 1 <= date.day <= 7: sales *= 1.3
                if month == 12 and date.day >= 25: sales *= 1.4
            if wd >= 5:
                sales *= wb
            sales += np.random.normal(0, max(2, base * 0.1))
            sales = max(0, int(sales))
            all_data.append({"date": date, "drug_name": item["drug_name"],
                             "category": item["category"], "sub_category": item["sub_category"],
                             "sales": sales, "day_of_week": wd, "month": month,
                             "is_weekend": 1 if wd >= 5 else 0})

    df = pd.DataFrame(all_data)
    print(f"  生成数据: {len(df)}行, {df['drug_name'].nunique()}商品, {len(dates)}天")
    print(f"  数据规模: {len(df)/10000:.1f}万行")

    # 品类统计
    cat_stats = df.groupby('category')['sales'].agg(['mean','sum']).sort_values('sum',ascending=False)
    print(f"\n  各品类统计:")
    print(cat_stats.to_string())

    path = os.path.join(DATA_DIR, "sales_data_9000_2year.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n  保存到: {path}")
    print(f"  文件大小: {os.path.getsize(path)/1024/1024:.1f}MB")
    return df


# ============================================================
# 第2步：Transformer模型（滑动窗口增广）
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=800):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=3,
                 forecast_len=30, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.decoder(x[:, -1, :])


def prepare_data_sliding(df, input_len=90, forecast_len=30, step=30):
    """滑动窗口构建训练数据（每个商品切多个样本）
    730天: train前700天，test最后30天
    滑动窗口从train里切多个(input_len, forecast_len)样本
    """
    print(f"\n  准备数据: input={input_len}天, forecast={forecast_len}天, step={step}")
    drugs = sorted(df['drug_name'].unique())
    all_sales = []
    for drug in drugs:
        s = df[df['drug_name'] == drug].sort_values('date')['sales'].values.astype(np.float32)
        all_sales.append(s)
    # 统一长度（有些商品可能少1天，截断到最短长度）
    min_len = min(len(s) for s in all_sales)
    all_sales = np.array([s[:min_len] for s in all_sales])  # (8640, min_len)
    n_products, n_days = all_sales.shape
    print(f"  原始数据: {n_products}商品 × {n_days}天")

    # 归一化（按商品，用train部分算均值标准差）
    means = all_sales[:, :-30].mean(axis=1, keepdims=True)
    stds = all_sales[:, :-30].std(axis=1, keepdims=True) + 1e-6
    normalized = (all_sales - means) / stds

    # 滑动窗口切训练样本
    train_end = n_days - 30  # 700天用于train+val
    val_start = train_end - 30  # 最后30天做val

    train_inputs = []; train_targets = []
    val_inputs = []; val_targets = []

    for p in range(n_products):
        # 训练样本：从0滑到val_start-input_len-forecast_len
        for start in range(0, val_start - input_len - forecast_len + 1, step):
            inp = normalized[p, start:start+input_len]
            tgt = normalized[p, start+input_len:start+input_len+forecast_len]
            if len(tgt) == forecast_len:
                train_inputs.append(inp)
                train_targets.append(tgt)
        # 验证样本：最后一个窗口
        v_inp = normalized[p, val_start-input_len:val_start]
        v_tgt = normalized[p, val_start:val_start+forecast_len]
        if len(v_tgt) == forecast_len:
            val_inputs.append(v_inp)
            val_targets.append(v_tgt)

    train_inputs = np.array(train_inputs)  # (N, 90)
    train_targets = np.array(train_targets)  # (N, 30)
    val_inputs = np.array(val_inputs)  # (9000, 90)
    val_targets = np.array(val_targets)  # (9000, 30)

    # 测试input：最后90天 → 预测最后30天
    test_input = normalized[:, -30-input_len:-30]  # (9000, 90)
    test_target = all_sales[:, -30:]  # (9000, 30) 原始值

    print(f"  训练样本: {train_inputs.shape[0]} (滑动窗口增广)")
    print(f"  验证样本: {val_inputs.shape[0]}")
    print(f"  测试: {test_input.shape} → {test_target.shape}")

    return {
        'drugs': drugs,
        'train_input': torch.FloatTensor(train_inputs).unsqueeze(-1),
        'train_target': torch.FloatTensor(train_targets),
        'val_input': torch.FloatTensor(val_inputs).unsqueeze(-1),
        'val_target': torch.FloatTensor(val_targets),
        'test_input': torch.FloatTensor(test_input).unsqueeze(-1),
        'test_target': test_target,
        'means': means, 'stds': stds,
    }


def train_transformer(data, epochs=30, batch_size=512, lr=1e-3):
    model = TimeSeriesTransformer(
        input_dim=1, d_model=128, nhead=8, num_layers=3,
        forecast_len=30, dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_input = data['train_input'].to(DEVICE)
    train_target = data['train_target'].to(DEVICE)
    val_input = data['val_input'].to(DEVICE)
    val_target = data['val_target'].to(DEVICE)
    n = train_input.size(0)

    print(f"\n  训练Transformer: {n}样本, {epochs}轮, batch={batch_size}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  d_model=128, nhead=8, layers=3 (比之前64/4/2更大)")

    best_val = float('inf'); best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(train_input[idx])
            loss = criterion(pred, train_target[idx])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_pred = model(val_input)
            val_loss = criterion(val_pred, val_target).item()
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} train={train_loss:.4f} val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

    if best_state:
        model.load_state_dict(best_state); model = model.to(DEVICE)
    print(f"  训练完成, 最优val_loss={best_val:.4f}")
    return model


def predict_eval(model, data):
    model.eval()
    with torch.no_grad():
        pred_norm = model(data['test_input'].to(DEVICE)).cpu().numpy()
    pred = pred_norm * data['stds'] + data['means']
    pred = np.maximum(pred, 0)
    actual = data['test_target']
    mapes = np.array([np.mean(np.abs((actual[i]-pred[i])/np.maximum(actual[i],1)))*100 for i in range(len(actual))])
    return pred, actual, mapes


# ============================================================
# 第3步：ETS对比（抽样1000个，外推9000）
# ============================================================
def benchmark_ets(df, sample=1000):
    """ETS抽样测试，外推9000"""
    import random
    random.seed(42)
    drugs = list(df['drug_name'].unique())
    sample_drugs = random.sample(drugs, min(sample, len(drugs)))
    print(f"\n  ETS抽样测试: {len(sample_drugs)}个商品")

    mapes = []
    t0 = time.time()
    for drug in sample_drugs:
        try:
            s = df[df['drug_name']==drug].sort_values('date')['sales'].values.astype(float)
            train = s[:-30]; test = s[-30:]
            m = ExponentialSmoothing(train, trend='add', seasonal='add',
                                     seasonal_periods=7, initialization_method='estimated')
            fit = m.fit(optimized=True)
            pred = np.maximum(fit.forecast(30), 0)
            mape = np.mean(np.abs((test-pred)/np.maximum(test,1)))*100
            mapes.append(mape)
        except:
            mapes.append(999)
    elapsed = time.time() - t0
    valid = [m for m in mapes if m < 999]
    avg = np.mean(valid)
    print(f"  ETS: 成功{len(valid)}/{len(sample_drugs)}, 平均MAPE={avg:.1f}%, 耗时{elapsed:.0f}秒")
    return avg, elapsed, len(valid)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  9000商品 × 2年数据 → Transformer vs ETS 对比")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 生成数据
    csv_path = os.path.join(DATA_DIR, "sales_data_9000_2year.csv")
    if os.path.exists(csv_path):
        print("\n数据已存在，直接加载...")
        df = pd.read_csv(csv_path, parse_dates=['date'])
        print(f"  数据: {len(df)}行, {df['drug_name'].nunique()}商品")
    else:
        df = generate_9000_data()

    # 2. Transformer
    print(f"\n{'='*60}")
    print(f"  Transformer训练（2年数据，滑动窗口增广）")
    print(f"{'='*60}")
    t0 = time.time()
    data = prepare_data_sliding(df, input_len=90, forecast_len=30, step=30)
    model = train_transformer(data, epochs=30, batch_size=512, lr=1e-3)
    train_time = time.time() - t0
    if DEVICE == "cuda":
        print(f"  GPU显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

    pred, actual, tf_mapes = predict_eval(model, data)
    print(f"\n  Transformer: MAPE={tf_mapes.mean():.1f}%, 训练{train_time:.1f}秒")
    exc = (tf_mapes<15).sum(); good = ((tf_mapes>=15)&(tf_mapes<25)).sum()
    fair = ((tf_mapes>=25)&(tf_mapes<35)).sum(); poor = (tf_mapes>=35).sum()
    print(f"  优秀{exc} 良好{good} 一般{fair} 需改进{poor}")

    # 3. ETS对比
    print(f"\n{'='*60}")
    print(f"  ETS对比（抽样1000外推）")
    print(f"{'='*60}")
    ets_mape, ets_time, ets_ok = benchmark_ets(df, sample=1000)

    # 4. 全量对比
    print(f"\n{'='*60}")
    print(f"  全量对比结果")
    print(f"{'='*60}")
    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │              365天/5000商品    730天/9000商品    变化             │
  ├──────────────────────────────────────────────────────────────────┤
  │ ETS         14.2%             {ets_mape:.1f}%             {'改善' if ets_mape<14.2 else '持平'}             │
  │ Transformer 17.4%             {tf_mapes.mean():.1f}%             {'✅反超ETS' if tf_mapes.mean()<ets_mape else '❌仍落后'}             │
  ├──────────────────────────────────────────────────────────────────┤
  │ 训练样本    5,000             {data['train_input'].shape[0]:,}           {data['train_input'].shape[0]/5000:.0f}x增广      │
  │ TF训练时间  10秒              {train_time:.0f}秒              {train_time/10:.1f}x             │
  │ TF参数量    102K              {sum(p.numel() for p in model.parameters()):,}        更大模型         │
  │ 输入窗口    60天              90天              更长上下文       │
  └──────────────────────────────────────────────────────────────────┘

  关键发现：
    {'✅ Transformer反超ETS！' if tf_mapes.mean()<ets_mape else '❌ Transformer仍落后ETS'}
    - 365天时: TF 17.4% > ETS 14.2% (TF落后3.2个百分点)
    - 730天时: TF {tf_mapes.mean():.1f}% vs ETS {ets_mape:.1f}% ({'TF领先' if tf_mapes.mean()<ets_mape else 'TF仍落后'}{abs(tf_mapes.mean()-ets_mape):.1f}个百分点)
    - 滑动窗口增广: 训练样本从5K→{data['train_input'].shape[0]:,}，让TF学到更多模式
    - 2年数据: TF能看到2个完整年度周期，学到跨年季节性
""")

    # 保存结果摘要
    result = {
        "data_scale": {"products": 9000, "days": 730, "train_samples": data['train_input'].shape[0]},
        "transformer": {"mape": float(tf_mapes.mean()), "train_time": train_time,
                        "params": sum(p.numel() for p in model.parameters()),
                        "input_len": 90, "d_model": 128, "epochs": 30},
        "ets": {"mape": float(ets_mape), "sample_size": 1000},
        "previous_365day": {"transformer": 17.4, "ets": 14.2},
    }
    out = os.path.join(LOGS_DIR, "transformer_2year_result.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果保存: {out}")
    print("DONE")
    log.close()


if __name__ == "__main__":
    main()
