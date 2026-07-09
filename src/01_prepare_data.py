"""
01_prepare_data.py
从 huatuo_encyclopedia_qa 数据集下载并筛选 2000 条药品知识
产出: data/drug_knowledge.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MAX_KNOWLEDGE_ITEMS

DRUG_KEYWORDS = [
    "用法用量", "说明书", "片", "胶囊", "注射液", "颗粒", "口服液",
    "丸", "软膏", "滴眼液", "喷雾剂", "分散片", "缓释片", "肠溶片",
    "用法", "用量", "服药", "剂量", "不良反应", "禁忌", "注意事项",
    "孕妇", "儿童用药", "老年用药", "药物相互作用",
]


def prepare_data():
    print("=" * 60)
    print("第1步：下载并筛选药品知识数据")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "drug_knowledge.json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"已有缓存数据: {len(existing)} 条")
        return existing

    print("正在从 HuggingFace 下载 FreedomIntelligence/huatuo_encyclopedia_qa ...")

    from datasets import load_dataset
    ds = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", split="train")
    print(f"数据集加载成功，共 {len(ds)} 条，开始筛选药品相关条目...")

    knowledge_items = []
    seen_instructions = set()

    count = 0
    for item in ds:
        questions = item.get("questions") or []
        answers = item.get("answers") or []
        if not questions or not answers:
            continue
        q_list = questions[0] if isinstance(questions[0], list) else questions
        instruction = q_list[0].strip() if q_list else ""
        output = answers[0].strip() if isinstance(answers[0], str) else str(answers[0]).strip()
        if not instruction or not output:
            continue
        if len(output) < 20:
            continue
        if instruction in seen_instructions:
            continue

        is_drug_related = any(kw in instruction for kw in DRUG_KEYWORDS) or \
                          any(kw in output[:200] for kw in DRUG_KEYWORDS[:10])
        if not is_drug_related:
            continue

        seen_instructions.add(instruction)
        knowledge_items.append({
            "id": f"drug_{len(knowledge_items):05d}",
            "question": instruction,
            "answer": output,
            "text": f"问题：{instruction}\n答案：{output}",
        })

        count += 1
        if count % 2000 == 0:
            print(f"  已扫描 {count} 条，筛选出 {len(knowledge_items)} 条药品知识...")

        if len(knowledge_items) >= MAX_KNOWLEDGE_ITEMS:
            print(f"  达到目标数量 {MAX_KNOWLEDGE_ITEMS}，停止筛选")
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_items, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(item["text"]) for item in knowledge_items)
    print(f"\n筛选完成：")
    print(f"  条目数: {len(knowledge_items)}")
    print(f"  总字数: {total_chars:,}")
    print(f"  平均每条: {total_chars // len(knowledge_items)} 字")
    print(f"  保存到: {output_path}")
    return knowledge_items


if __name__ == "__main__":
    data = prepare_data()
    print(f"\n第1步完成，共 {len(data)} 条药品知识")
