# -*- coding: utf-8 -*-
"""
pipeline.py - 库存预测统一Pipeline框架

把5个散脚本串成一条标准化管线：
  数据加载 → 多模型预测 → 模型选优 → 漂移检测 → LLM结论 → 结果输出

设计理念（类LangGraph的节点式工作流，但针对ML数据管线优化）：
  - 每个节点 = 一个处理步骤，输入dict → 输出dict
  - 节点间通过 state 传递数据（状态机模式）
  - 支持断点续跑（每步结果缓存到磁盘）
  - 支持单独调用任意节点
  - LLM节点用llm_pool（多API轮询）

用法：
  from pipeline import InventoryForecastPipeline
  pipe = InventoryForecastPipeline()
  result = pipe.run()                    # 跑完整管线
  result = pipe.run_step("predict")      # 只跑某一步
  result = pipe.run_step("conclusion")   # 只跑LLM结论
"""
import os, sys, json, time, warnings
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
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "data", "pipeline_cache")  # 断点续跑缓存


# ============================================================
# Pipeline核心类
# ============================================================
class InventoryForecastPipeline:
    """
    库存预测统一管线

    节点流程（类LangGraph工作流）：
      [load_data] → [multi_predict] → [select_best] → [drift_detect] → [build_conclusion]
         ↓              ↓                 ↓               ↓                  ↓
       加载CSV      5模型并行预测      验证集选优       漂移检测          LLM出结论

    每个节点：输入state → 处理 → 输出更新后的state
    节点间通过state传递，支持缓存断点续跑
    """

    # 节点定义（顺序执行）
    STEPS = ["load_data", "multi_predict", "select_best", "drift_detect", "build_conclusion"]

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.state = {}  # 管线状态，节点间传递
        self.step_times = {}  # 各节点耗时

    def run(self, steps=None):
        """跑完整管线或指定步骤
        steps: None=全部, 或 ["predict","conclusion"] 跑指定步
        """
        steps = steps or self.STEPS
        print(f"\n{'='*60}")
        print(f"  库存预测管线启动")
        print(f"  步骤: {' → '.join(steps)}")
        print(f"{'='*60}")

        t_total = time.time()
        for step in steps:
            if step not in self.STEPS:
                print(f"  ⚠️ 未知步骤: {step}, 跳过")
                continue

            # 缓存检查
            if self.use_cache and self._has_cache(step):
                print(f"\n[{step}] 命中缓存，跳过执行")
                self._load_cache(step)
                continue

            # 执行节点
            print(f"\n[{step}] 执行中...")
            t0 = time.time()
            handler = getattr(self, f"_step_{step}")
            self.state = handler(self.state)
            elapsed = time.time() - t0
            self.step_times[step] = elapsed
            print(f"[{step}] 完成 ({elapsed:.1f}秒)")

            # 写缓存
            if self.use_cache:
                self._save_cache(step)

        total = time.time() - t_total
        print(f"\n{'='*60}")
        print(f"  管线完成！总耗时: {total:.1f}秒")
        print(f"  各步骤耗时: {self.step_times}")
        print(f"{'='*60}")
        return self.state

    def run_step(self, step):
        """只跑某一步"""
        return self.run(steps=[step])

    def get_status(self):
        """获取管线状态（各步骤是否完成）"""
        status = {}
        for step in self.STEPS:
            status[step] = {
                "done": step in self.state or (self.use_cache and self._has_cache(step)),
                "cached": self.use_cache and self._has_cache(step),
                "time": self.step_times.get(step, 0),
            }
        return status

    # ===== 缓存方法 =====
    def _cache_path(self, step):
        return os.path.join(CACHE_DIR, f"{step}.json")

    def _has_cache(self, step):
        return os.path.exists(self._cache_path(step))

    def _save_cache(self, step):
        """保存节点输出到缓存（只存可序列化的部分）"""
        cache_data = {}
        for k, v in self.state.items():
            try:
                json.dumps(v, ensure_ascii=False)
                cache_data[k] = v
            except (TypeError, ValueError):
                pass  # 跳过不可序列化的（如ndarray）
        with open(self._cache_path(step), 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)

    def _load_cache(self, step):
        with open(self._cache_path(step), 'r', encoding='utf-8') as f:
            self.state.update(json.load(f))

    # ============================================================
    # 节点1：数据加载
    # ============================================================
    def _step_load_data(self, state):
        """加载5000商品销量数据"""
        path = os.path.join(DATA_DIR, 'sales_data_5000.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}，请先运行24号脚本生成数据")

        df = pd.read_csv(path, parse_dates=['date'])
        state["df"] = df  # DataFrame不缓存，每次重新加载
        state["n_products"] = int(df['drug_name'].nunique())
        state["n_days"] = int(df['date'].nunique())
        state["categories"] = df['category'].unique().tolist()

        print(f"  数据: {len(df)}行, {state['n_products']}商品, {state['n_days']}天")
        print(f"  品类: {state['categories']}")
        return state

    # ============================================================
    # 节点2：多模型预测（5模型并行）
    # ============================================================
    def _step_multi_predict(self, state):
        """对每个商品跑5个模型，验证集选优"""
        from importlib import import_module
        mod = import_module('25_multi_model_forecast')

        df = state.get("df")
        if df is None:
            raise ValueError("数据未加载，请先执行load_data步骤")

        drugs = list(df['drug_name'].unique())
        all_results = {}
        model_win = {"ExponentialSmoothing": 0, "SeasonalNaive": 0,
                     "MovingAverage": 0, "LinearRegression": 0, "Ensemble": 0}

        t_start = time.time()
        for i, drug in enumerate(drugs):
            try:
                result = mod.forecast_product_multi_model(drug, df[df['drug_name'] == drug])
                all_results[drug] = result
                if result.get("model_used"):
                    model_win[result["model_used"]] = model_win.get(result["model_used"], 0) + 1
            except Exception as e:
                all_results[drug] = {"drug_name": drug, "error": str(e)[:60], "mape": 999}

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                print(f"  进度 {i+1}/{len(drugs)} ({(i+1)/len(drugs)*100:.0f}%) {elapsed:.0f}秒", flush=True)

        valid = [v for v in all_results.values() if v.get("mape", 999) < 999]
        mapes = [v['mape'] for v in valid]

        state["predictions"] = all_results
        state["n_success"] = len(valid)
        state["avg_mape"] = round(float(np.mean(mapes)), 1) if mapes else 0
        state["model_win_count"] = model_win
        state["mape_dist"] = {
            "excellent": sum(1 for m in mapes if m < 15),
            "good": sum(1 for m in mapes if 15 <= m < 25),
            "fair": sum(1 for m in mapes if 25 <= m < 35),
            "poor": sum(1 for m in mapes if m >= 35),
        }
        print(f"  成功: {len(valid)}/{len(drugs)}")
        print(f"  平均MAPE: {state['avg_mape']}%")
        print(f"  模型胜出: {model_win}")
        return state

    # ============================================================
    # 节点3：模型选优结果整理（已在predict里完成，这步做汇总统计）
    # ============================================================
    def _step_select_best(self, state):
        """汇总选优结果：品类统计、补货TOP、漂移TOP"""
        predictions = state.get("predictions", {})
        valid = [v for v in predictions.values() if v.get("mape", 999) < 999]
        if not valid:
            return state

        # 品类统计
        cat_stats = {}
        for v in valid:
            cat = v.get("category", "未知")
            if cat not in cat_stats:
                cat_stats[cat] = {"count": 0, "forecast": 0, "stock": 0, "drift": 0}
            cat_stats[cat]["count"] += 1
            cat_stats[cat]["forecast"] += v.get("total_forecast_30days", 0)
            cat_stats[cat]["stock"] += v.get("recommended_stock", 0)
            if v.get("is_drift"):
                cat_stats[cat]["drift"] += 1

        # 模型分布
        model_stats = {}
        for v in valid:
            m = v.get("model_used", "unknown")
            if m not in model_stats:
                model_stats[m] = {"count": 0, "mape_sum": 0}
            model_stats[m]["count"] += 1
            model_stats[m]["mape_sum"] += v["mape"]
        for m in model_stats:
            c = model_stats[m]["count"]
            model_stats[m]["avg_mape"] = round(model_stats[m]["mape_sum"] / c, 1) if c else 0
            model_stats[m]["pct"] = round(c / len(valid) * 100, 1)
            del model_stats[m]["mape_sum"]

        # 补货TOP10
        top_restock = sorted(valid, key=lambda x: -x.get("recommended_stock", 0))[:10]
        state["top_restock"] = [{"drug_name": v["drug_name"], "category": v.get("category", ""),
                                 "recommended_stock": v.get("recommended_stock", 0),
                                 "total_forecast_30days": v.get("total_forecast_30days", 0),
                                 "mape": v.get("mape", 0)} for v in top_restock]

        state["category_stats"] = cat_stats
        state["model_stats"] = model_stats
        state["total_forecast"] = sum(v.get("total_forecast_30days", 0) for v in valid)
        state["total_stock"] = sum(v.get("recommended_stock", 0) for v in valid)

        print(f"  品类数: {len(cat_stats)}")
        print(f"  模型分布: {list(model_stats.keys())}")
        print(f"  总预测销量: {state['total_forecast']}")
        print(f"  总建议库存: {state['total_stock']}")
        return state

    # ============================================================
    # 节点4：漂移检测（已在predict里完成，这步汇总）
    # ============================================================
    def _step_drift_detect(self, state):
        """汇总漂移检测结果"""
        predictions = state.get("predictions", {})
        valid = [v for v in predictions.values() if v.get("mape", 999) < 999]
        drift_items = [v for v in valid if v.get("is_drift")]
        drift_items.sort(key=lambda x: -abs(x.get("drift_ratio", 0)))

        state["drift_count"] = len(drift_items)
        state["top_drift"] = [{"drug_name": v["drug_name"], "category": v.get("category", ""),
                               "drift_ratio": v.get("drift_ratio", 0),
                               "mape": v.get("mape", 0)} for v in drift_items[:10]]
        print(f"  漂移商品: {len(drift_items)}/{len(valid)}")
        return state

    # ============================================================
    # 节点5：LLM生成补货结论
    # ============================================================
    def _step_build_conclusion(self, state):
        """调LLM把预测数字转成可执行补货建议"""
        predictions = state.get("predictions", {})
        valid = [v for v in predictions.values() if v.get("mape", 999) < 999]
        if not valid:
            return state

        # 汇总数据（取TOP30补货 + TOP10漂移 + TOP10差MAPE）
        top_restock = sorted(valid, key=lambda x: -x.get("recommended_stock", 0))[:30]
        drift_items = sorted([v for v in valid if v.get("is_drift")],
                              key=lambda x: -abs(x.get("drift_ratio", 0)))[:10]
        poor_mape = sorted(valid, key=lambda x: -x.get("mape", 0))[:10]

        lines = []
        for v in top_restock:
            lines.append(f"- {v['drug_name']}({v.get('category','')}): 预测{v.get('total_forecast_30days',0)}盒, "
                         f"建议库存{v.get('recommended_stock',0)}盒, MAPE={v.get('mape',0)}%")
        lines.append("\n【漂移商品TOP10】")
        for v in drift_items:
            lines.append(f"- {v['drug_name']}: 偏移{v.get('drift_ratio',0)}%, MAPE={v.get('mape',0)}%")
        lines.append("\n【预测不准TOP10】")
        for v in poor_mape:
            lines.append(f"- {v['drug_name']}: MAPE={v.get('mape',0)}%")
        data_text = "\n".join(lines)

        prompt = f"""你是药店库存分析师。以下是5000商品预测的关键汇总数据，请生成可执行的库存管理结论。

【预测数据汇总】
{data_text}

【整体统计】
  商品总数: {len(valid)}
  预测30天总销量: {state.get('total_forecast', 0)}盒
  建议总库存: {state.get('total_stock', 0)}盒
  漂移商品: {state.get('drift_count', 0)}个
  平均MAPE: {state.get('avg_mape', 0)}%

【模型分布】
{json.dumps(state.get('model_stats', {}), ensure_ascii=False, indent=2)}

【请生成】
1. 【整体结论】（2-3句话）
2. 【紧急补货TOP5】（具体补货量）
3. 【漂移预警】（消费习惯变化的商品+可能原因）
4. 【预测不准需人工复核】（MAPE高的）
5. 【可执行操作清单】（给店长/采购员3-5条指令）

用markdown格式输出，结论要具体可执行。"""

        try:
            from llm_pool import chat
            conclusion = chat(
                [{"role": "system", "content": "你是专业药店库存分析师。"},
                 {"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=1500,
            )
            state["conclusion"] = conclusion
            print(f"  LLM结论生成完成, 长度: {len(conclusion)}字")
        except Exception as e:
            state["conclusion"] = f"LLM生成失败: {e}"
            print(f"  LLM生成失败: {e}")

        return state

    # ============================================================
    # 导出结果
    # ============================================================
    def export(self, path=None):
        """导出完整预测结果到JSON"""
        path = path or os.path.join(DATA_DIR, 'forecast_5000_result.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.state.get("predictions", {}), f, ensure_ascii=False)
        print(f"导出到: {path}")
        return path


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    pipe = InventoryForecastPipeline(use_cache=True)

    # 查看状态
    print("\n当前管线状态:")
    status = pipe.get_status()
    for step, info in status.items():
        print(f"  {step}: {'✅' if info['done'] else '❌'} (缓存: {'有' if info['cached'] else '无'})")

    # 跑完整管线
    result = pipe.run()

    # 导出
    pipe.export()

    # 打印LLM结论
    if result.get("conclusion"):
        print(f"\n{'='*60}")
        print("  AI补货建议")
        print(f"{'='*60}")
        print(result["conclusion"])
