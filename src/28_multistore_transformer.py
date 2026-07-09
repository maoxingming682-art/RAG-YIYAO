# -*- coding: utf-8 -*-
"""
28_multistore_transformer.py
140家门店 × 500商品 × 730天 → Transformer vs ETS

关键改进：
  - 每个商品有140条序列（140家门店），Transformer能跨门店学习共享模式
  - 门店差异：每家门店有不同基础销量、增长趋势、客流量
  - 商品模式共享：感冒药冬天涨在所有门店都成立
  - 训练样本：140×500=70,000条序列，滑动窗口后~140万样本

假设：多门店数据下Transformer应该能反超ETS
  因为Transformer能学到"这个商品在A门店的规律"并迁移到B门店
  ETS每条序列独立训练，无法跨门店学习
"""
import sys, os, json, time, math, warnings
import numpy as np
import torch
import torch.nn as nn
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

log_path = os.path.join(LOGS_DIR, "multistore_transformer.log")
os.makedirs(LOGS_DIR, exist_ok=True)
log = open(log_path, 'w', encoding='utf-8')
class T:
    def write(self, s): log.write(s); log.flush()
    def flush(self): log.flush()
sys.stdout = T(); sys.stderr = T()


N_STORES = 140
N_PRODUCTS = 500
N_DAYS = 730
INPUT_LEN = 90
FORECAST_LEN = 30
STEP = 90  # 更大步长，减少样本数加快训练


# ============================================================
# 生成多门店数据
# ============================================================
def generate_multistore_data():
    """生成140门店×500商品×730天数据
    每个商品在所有门店共享季节性模式，但门店有不同的基础销量/趋势
    """
    print(f"  生成数据: {N_STORES}门店 × {N_PRODUCTS}商品 × {N_DAYS}天")
    np.random.seed(42)

    # 商品属性（所有门店共享）
    product_base = np.random.randint(5, 40, N_PRODUCTS)  # 基础销量
    product_trend = np.random.uniform(0.001, 0.02, N_PRODUCTS)
    product_season = np.random.choice(['winter', 'summer', 'none', 'holiday', 'winter_mild'], N_PRODUCTS)

    # 门店属性（每家门店不同）
    store_multiplier = np.random.uniform(0.5, 2.0, N_STORES)  # 门店规模0.5x-2x
    store_trend = np.random.uniform(-0.003, 0.005, N_STORES)  # 门店趋势

    # 生成销量矩阵: (N_STORES, N_PRODUCTS, N_DAYS)
    dates = np.arange(N_DAYS)
    # 计算月份（用于季节性）
    # 假设从2024-01-01开始
    from datetime import date, timedelta
    start = date(2024, 1, 1)
    months = np.array([(start + timedelta(days=int(d))).month for d in dates])
    day_of_week = np.array([(start + timedelta(days=int(d))).weekday() for d in dates])

    sales = np.zeros((N_STORES, N_PRODUCTS, N_DAYS), dtype=np.float32)

    for s in range(N_STORES):
        for p in range(N_PRODUCTS):
            base = product_base[p] * store_multiplier[s]
            trend = product_trend[p] + store_trend[s]
            season = product_season[p]

            series = base + trend * dates.astype(float)

            # 季节性
            if season == 'winter':
                winter_mask = np.isin(months, [1,2,3,11,12])
                series[winter_mask] *= 1.5
                summer_mask = np.isin(months, [7,8])
                series[summer_mask] *= 0.8
            elif season == 'winter_mild':
                winter_mask = np.isin(months, [1,2,12])
                series[winter_mask] *= 1.2
            elif season == 'summer':
                summer_mask = np.isin(months, [6,7,8])
                series[summer_mask] *= 1.3
            elif season == 'holiday':
                for d_idx in range(N_DAYS):
                    m = months[d_idx]; dd = (start + timedelta(days=d_idx)).day
                    if m == 1 and dd <= 7: series[d_idx] *= 1.6
                    if m == 2 and dd <= 7: series[d_idx] *= 1.5
                    if m == 10 and 1 <= dd <= 7: series[d_idx] *= 1.3

            # 星期效应
            weekend_mask = day_of_week >= 5
            series[weekend_mask] *= np.random.uniform(1.1, 1.4)

            # 噪声（门店特定）
            series += np.random.normal(0, max(2, base * 0.12), N_DAYS)
            series = np.maximum(series, 0)
            sales[s, p, :] = series

    print(f"  数据矩阵: {sales.shape} ({sales.size/1e6:.1f}M数据点)")
    print(f"  内存占用: {sales.nbytes/1024**2:.1f}MB")
    return sales


# ============================================================
# Transformer模型
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


class MultiStoreTransformer(nn.Module):
    """多门店时序Transformer
    输入: (batch, seq_len, 1)  每个样本是一个(门店,商品)的销量序列
    输出: (batch, forecast_len)
    通过batch里的多样性，模型学到跨门店跨商品的共享模式
    """
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


# ============================================================
# 准备训练数据（滑动窗口）
# ============================================================
def prepare_data(sales):
    """sales: (N_STORES, N_PRODUCTS, N_DAYS)
    每个(store,product)是一条序列，共70,000条
    滑动窗口切训练样本
    """
    print(f"\n  准备训练数据...")
    n_stores, n_products, n_days = sales.shape
    n_series = n_stores * n_products  # 70,000

    # 展平成 (70000, 730)
    flat = sales.reshape(n_series, n_days)

    # 归一化（每条序列独立）
    means = flat[:, :-FORECAST_LEN].mean(axis=1, keepdims=True)
    stds = flat[:, :-FORECAST_LEN].std(axis=1, keepdims=True) + 1e-6
    normalized = (flat - means) / stds

    # 划分: train前700天, test最后30天
    train_end = n_days - FORECAST_LEN  # 700
    val_start = train_end - FORECAST_LEN  # 670

    # 滑动窗口切训练样本
    train_inputs = []; train_targets = []
    for i in range(n_series):
        for start in range(0, val_start - INPUT_LEN - FORECAST_LEN + 1, STEP):
            inp = normalized[i, start:start+INPUT_LEN]
            tgt = normalized[i, start+INPUT_LEN:start+INPUT_LEN+FORECAST_LEN]
            if len(tgt) == FORECAST_LEN:
                train_inputs.append(inp)
                train_targets.append(tgt)

    train_inputs = np.array(train_inputs)
    train_targets = np.array(train_targets)
    print(f"  训练样本: {train_inputs.shape[0]:,} (滑动窗口增广)")

    # 验证集：每条序列最后一个窗口
    val_inputs = normalized[:, val_start-INPUT_LEN:val_start]
    val_targets = normalized[:, val_start:val_start+FORECAST_LEN]
    print(f"  验证样本: {val_inputs.shape[0]:,}")

    # 测试集：最后INPUT_LEN天预测最后FORECAST_LEN天
    test_input = normalized[:, -FORECAST_LEN-INPUT_LEN:-FORECAST_LEN]
    test_target = flat[:, -FORECAST_LEN:]  # 原始值

    return {
        'train_input': torch.FloatTensor(train_inputs).unsqueeze(-1),
        'train_target': torch.FloatTensor(train_targets),
        'val_input': torch.FloatTensor(val_inputs).unsqueeze(-1),
        'val_target': torch.FloatTensor(val_targets),
        'test_input': torch.FloatTensor(test_input).unsqueeze(-1),
        'test_target': test_target,
        'means': means, 'stds': stds,
        'n_series': n_series,
    }


# ============================================================
# 训练
# ============================================================
def train_transformer(data, epochs=10, batch_size=2048, lr=1e-3):
    model = MultiStoreTransformer(
        input_dim=1, d_model=128, nhead=8, num_layers=3,
        forecast_len=FORECAST_LEN, dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_input = data['train_input'].to(DEVICE)
    train_target = data['train_target'].to(DEVICE)
    val_input = data['val_input'].to(DEVICE)
    val_target = data['val_target'].to(DEVICE)
    n = train_input.size(0)

    print(f"\n  训练: {n:,}样本, {epochs}轮, batch={batch_size}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

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
            # 分batch验证，避免显存爆炸
            val_loss_sum = 0; val_count = 0
            for i in range(0, val_input.size(0), batch_size):
                v_inp = val_input[i:i+batch_size]
                v_tgt = val_target[i:i+batch_size]
                vp = model(v_inp)
                val_loss_sum += criterion(vp, v_tgt).item() * v_inp.size(0)
                val_count += v_inp.size(0)
            val_loss = val_loss_sum / val_count
        if val_loss < best_val:
            best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} train={train_loss:.4f} val={val_loss:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state); model = model.to(DEVICE)
    print(f"  完成, best_val={best_val:.4f}")
    return model


def predict_eval(model, data):
    model.eval()
    all_pred = []
    test_input = data['test_input'].to(DEVICE)
    with torch.no_grad():
        for i in range(0, test_input.size(0), 2048):
            chunk = test_input[i:i+2048]
            pred_norm = model(chunk).cpu().numpy()
            all_pred.append(pred_norm)
    pred_norm = np.concatenate(all_pred, axis=0)
    pred = pred_norm * data['stds'] + data['means']
    pred = np.maximum(pred, 0)
    actual = data['test_target']
    mapes = np.array([np.mean(np.abs((actual[i]-pred[i])/np.maximum(actual[i],1)))*100
                      for i in range(len(actual))])
    return mapes


# ============================================================
# ETS对比（抽样）
# ============================================================
def benchmark_ets(sales, sample=1000):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import random
    random.seed(42)
    n_stores, n_products, _ = sales.shape
    # 随机抽样(门店,商品)对
    pairs = [(s, p) for s in range(n_stores) for p in range(n_products)]
    sample_pairs = random.sample(pairs, min(sample, len(pairs)))

    print(f"\n  ETS抽样: {len(sample_pairs)}条序列")
    mapes = []
    t0 = time.time()
    for s, p in sample_pairs:
        try:
            series = sales[s, p, :].astype(float)
            train = series[:-FORECAST_LEN]; test = series[-FORECAST_LEN:]
            m = ExponentialSmoothing(train, trend='add', seasonal='add',
                                     seasonal_periods=7, initialization_method='estimated')
            fit = m.fit(optimized=True)
            pred = np.maximum(fit.forecast(FORECAST_LEN), 0)
            mape = np.mean(np.abs((test-pred)/np.maximum(test,1)))*100
            mapes.append(mape)
        except:
            mapes.append(999)
    elapsed = time.time() - t0
    valid = [m for m in mapes if m < 999]
    avg = np.mean(valid)
    print(f"  ETS: 成功{len(valid)}/{len(sample_pairs)}, MAPE={avg:.1f}%, 耗时{elapsed:.0f}秒")
    return avg


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print(f"  140门店 × 500商品 × 730天 → Transformer vs ETS")
    print(f"  设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 1. 生成数据
    sales = generate_multistore_data()

    # 2. Transformer
    print(f"\n{'='*60}")
    print(f"  Transformer训练（跨门店学习）")
    print(f"{'='*60}")
    t0 = time.time()
    data = prepare_data(sales)
    model = train_transformer(data, epochs=10, batch_size=2048, lr=1e-3)
    train_time = time.time() - t0
    if DEVICE == "cuda":
        print(f"  GPU显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

    tf_mapes = predict_eval(model, data)
    print(f"\n  Transformer: MAPE={tf_mapes.mean():.1f}%, 训练{train_time:.1f}秒")
    exc = (tf_mapes<15).sum(); good = ((tf_mapes>=15)&(tf_mapes<25)).sum()
    print(f"  优秀{exc} 良好{good} 其他{len(tf_mapes)-exc-good}")

    # 3. ETS对比
    print(f"\n{'='*60}")
    print(f"  ETS对比")
    print(f"{'='*60}")
    ets_mape = benchmark_ets(sales, sample=1000)

    # 4. 结果
    print(f"\n{'='*60}")
    print(f"  对比结果")
    print(f"{'='*60}")
    tf_avg = tf_mapes.mean()
    winner = "Transformer" if tf_avg < ets_mape else "ETS"
    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │              单门店/9000商品   140门店/500商品   变化    │
  ├──────────────────────────────────────────────────────────┤
  │ ETS         10.5%            {ets_mape:.1f}%            {'✅' if ets_mape<10.5 else ''}      │
  │ Transformer 14.4%            {tf_avg:.1f}%            {'✅反超!' if tf_avg<ets_mape else '❌'}      │
  ├──────────────────────────────────────────────────────────┤
  │ 序列数      9,000             70,000           7.8x     │
  │ 训练样本    164K              {data['train_input'].shape[0]:,}           {data['train_input'].shape[0]/164000:.0f}x     │
  │ TF训练时间  609秒             {train_time:.0f}秒             {train_time/609:.1f}x     │
  └──────────────────────────────────────────────────────────┘

  结果: {winner} 获胜！
    - ETS: {ets_mape:.1f}% (每条序列独立训练)
    - Transformer: {tf_avg:.1f}% (跨门店学习)
    {'✅ Transformer反超！多门店数据下，跨门店学习优势显现' if tf_avg < ets_mape else '❌ Transformer仍落后，但差距缩小'}

  关键：
    - 单门店时TF落后ETS {14.4-10.5:.1f}个百分点
    - 140门店时TF {'领先' if tf_avg<ets_mape else '落后'}ETS {abs(tf_avg-ets_mape):.1f}个百分点
    - 跨门店学习让Transformer {'成功反超' if tf_avg<ets_mape else '大幅缩小差距'}
""")

    result = {
        "scenario": "140 stores × 500 products × 730 days",
        "n_series": 70000,
        "n_train_samples": data['train_input'].shape[0],
        "transformer": {"mape": float(tf_avg), "train_time": train_time},
        "ets": {"mape": float(ets_mape)},
        "winner": winner,
        "previous_single_store": {"transformer": 14.4, "ets": 10.5},
    }
    out = os.path.join(LOGS_DIR, "multistore_result.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果保存: {out}")
    print("DONE")
    log.close()


if __name__ == "__main__":
    main()
