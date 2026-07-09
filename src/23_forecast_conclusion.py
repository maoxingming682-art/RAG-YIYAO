"""
23_forecast_conclusion.py
把Prophet的数字预测转成LLM的文字结论

Prophet只产出数字（下月卖多少盒）
LLM把数字解读成可执行结论（该备什么货/该清什么库存/有什么风险）
"""
import os, sys, json, time, numpy as np, pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR
from src.config import DATA_DIR
from llm_pool import chat

FORECAST_DIR = os.path.join(BASE_DIR, "logs")


def call_llm(prompt, system="你是药店库存分析助手", temperature=0.4, max_tokens=1200):
    """通过多API轮询调用LLM"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return chat(messages, temperature=temperature, max_tokens=max_tokens)


def generate_conclusion():
    """
    读取Prophet预测结果，用LLM生成可执行结论
    """
    print("=" * 60)
    print("  预测结论生成：Prophet数字 → LLM文字结论")
    print("=" * 60)

    # 1. 读取预测结果
    forecast_path = os.path.join(FORECAST_DIR, "forecast_result.json")
    if not os.path.exists(forecast_path):
        print("错误：请先运行 22_inventory_forecast.py")
        return

    with open(forecast_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"读取 {len(results)} 种药品预测结果")

    # 2. 整理数据给LLM
    # 按品类分组汇总
    drug_list = list(results.keys())
    summary_lines = []
    for drug in drug_list:
        r = results[drug]
        inv = r.get("inventory", {})
        drift = r.get("drift", {})
        mape = r.get("mape", 0)
        forecast_30 = inv.get("total_forecast_30days", 0)
        avg_daily = inv.get("avg_daily", 0)
        recommended = inv.get("recommended_stock", 0)
        is_drift = drift.get("is_drift", False)
        drift_ratio = drift.get("drift_ratio", 0)

        summary_lines.append(
            f"- {drug}: 预测30天{forecast_30}盒, 日均{avg_daily}盒, "
            f"建议库存{recommended}盒, MAPE={mape:.1f}%, "
            f"漂移={'是(偏移'+str(drift_ratio)+'%)' if is_drift else '否'}"
        )

    data_text = "\n".join(summary_lines)

    # 3. 调LLM生成结论
    print(f"\n调用LLM生成结论...")

    prompt = f"""你是药店库存分析师。以下是50种药品未来30天的销量预测数据，请生成一份可执行的库存管理结论报告。

【预测数据】
{data_text}

【MAPE说明】
  <15% = 优秀（预测很准，可放心执行）
  15-25% = 良好（可用，但多留安全库存）
  25-35% = 一般（仅供参考，建议人工复核）
  >35% = 不准（需人工判断，模型需优化）

【漂移说明】
  漂移=最近销量跟历史基线偏移>20%，说明消费习惯变了，模型需要重训

【请生成以下内容】

1. 【整体结论】（2-3句话总结）
   - 下月整体销量趋势（上升/下降/平稳）
   - 主要增长品类和下降品类

2. 【补货建议】（分类给建议）
   - 紧急补货（需求大幅增长的药品）
   - 正常补货（稳定增长的药品）
   - 维持不变（销量平稳的药品）
   - 减少库存（销量下降或预测不准的药品）

3. 【风险提醒】
   - 预测不准的药品（MAPE>30%），需人工复核
   - 检测到漂移的药品，模型需要重训
   - 季节性药品的备货窗口

4. 【可执行操作清单】（给店长/采购员的明确指令）
   - 该订什么货、订多少
   - 该清什么库存
   - 什么药品需要关注

请用清晰的markdown格式输出，结论要具体可执行，不要泛泛而谈。"""

    t0 = time.time()
    conclusion = call_llm(prompt,
        system="你是专业药店库存分析师，擅长把预测数据转成可执行的补货建议。",
        temperature=0.4, max_tokens=1500)
    t1 = time.time()

    print(f"  生成完成 (耗时{t1-t0:.1f}s)\n")

    print("=" * 60)
    print("  📊 库存预测结论报告")
    print("=" * 60)
    print(conclusion)

    # 保存
    conclusion_path = os.path.join(FORECAST_DIR, "forecast_conclusion.md")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write(conclusion)
    print(f"\n结论已保存: {conclusion_path}")

    # 关键认知总结
    print(f"""
{'='*60}
关键认知
{'='*60}

【预测类AI产品 = 两层组合】

  第1层：时序模型（Prophet）
    输入：历史销量数字
    输出：未来销量数字
    作用：算出"下月卖多少盒"
    局限：只给数字，不给结论，不懂业务

  第2层：LLM（语言模型）
    输入：Prophet的数字 + 业务上下文
    输出：可执行的文字结论
    作用：解读数字 → 给补货建议/风险提醒/操作清单
    优势：懂业务语义，能把数字变成"该做什么"

【为什么Prophet自己不能给结论】
  1. Prophet是数学模型，不懂"冬季感冒药要提前备"这种业务知识
  2. Prophet不知道MAPE 38.9%意味着"这个预测不准要人工复核"
  3. Prophet不会说"17种漂移了要重训模型"——它只算数字
  4. 结论需要业务理解力，这是LLM的强项

【完整的预测产品链路】
  历史数据 → Prophet预测数字 → LLM解读结论 → 店长执行
  （算数字）   （给数字）        （给建议）       （行动）

【跟RAG的对比】
  RAG：     文本→检索→LLM→答案（查过去）
  预测：    数字→Prophet→LLM→结论（算未来）
  共同点：  都需要LLM做最后一步"把原始结果变成人能用的答案"
""")

    return conclusion


if __name__ == "__main__":
    generate_conclusion()
