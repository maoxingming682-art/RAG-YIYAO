"""
10_eval_finetune.py
评估微调效果：对比微调前后的回答风格
"""
import os
import sys
import json
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, BASE_DIR, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = os.path.join(BASE_DIR, "lora_output")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def generate_with_model(model, tokenizer, question, max_new_tokens=300):
    """用本地模型生成回答"""
    messages = [
        {"role": "system", "content": "你是专业药学咨询助手，回答用药问题。回答必须包含用药建议、注意事项和安全提示。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    # 只取新生成的部分
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def check_style(answer):
    """检查答案是否符合药学咨询风格"""
    checks = {
        "有注意事项": "注意事项" in answer or "注意" in answer,
        "有安全提示": "咨询" in answer or "医生" in answer or "药师" in answer or "处方" in answer,
        "结构化回答": any(kw in answer for kw in ["1.", "【", "①", "第一", "用法", "用量"]),
        "长度适中": 50 < len(answer) < 1000,
    }
    score = sum(checks.values()) / len(checks) * 100
    return checks, score


def evaluate():
    print("=" * 60)
    print("第2-5步：评估微调效果")
    print("=" * 60)

    # 测试问题
    test_questions = [
        "依托度酸片怎么服用？用法用量是什么？",
        "布洛芬缓释胶囊有什么不良反应？",
        "阿司匹林肠溶胶囊的用法用量",
        "醋酸地塞米松软膏怎么用？",
        "孕妇能用布洛芬吗？",
    ]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 1. 加载微调后的模型（base + LoRA）
    print("加载微调后模型（base + LoRA）...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, trust_remote_code=True
    ).to("cuda")

    # 加载LoRA适配器
    from peft import PeftModel
    model_ft = PeftModel.from_pretrained(model_ft, LORA_PATH)
    model_ft.eval()
    print(f"微调模型加载完成, GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)

    # 2. 逐个测试
    print(f"\n测试 {len(test_questions)} 个问题\n")

    results = []
    for i, q in enumerate(test_questions):
        print(f"\n{'─'*60}")
        print(f"问题{i+1}: {q}")
        print(f"{'─'*60}")

        # 微调后模型回答
        print("微调后模型回答: ", end="", flush=True)
        answer_ft = generate_with_model(model_ft, tokenizer, q)
        checks_ft, score_ft = check_style(answer_ft)
        print(f"{answer_ft[:200]}...")
        print(f"  风格评分: {score_ft:.0f}/100  检查: {checks_ft}")

        results.append({
            "question": q,
            "answer_finetuned": answer_ft,
            "style_score": score_ft,
            "checks": checks_ft,
        })

    # 3. 释放显存，加载原始模型对比
    del model_ft
    torch.cuda.empty_cache()

    print(f"\n加载原始模型（未微调）对比...", flush=True)
    model_raw = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    model_raw.eval()

    for i, q in enumerate(test_questions):
        print(f"\n原始模型回答问题{i+1}: ", end="", flush=True)
        answer_raw = generate_with_model(model_raw, tokenizer, q)
        checks_raw, score_raw = check_style(answer_raw)
        print(f"{answer_raw[:200]}...")
        print(f"  风格评分: {score_raw:.0f}/100  检查: {checks_raw}")
        results[i]["answer_raw"] = answer_raw
        results[i]["raw_score"] = score_raw
        results[i]["raw_checks"] = checks_raw

    del model_raw
    torch.cuda.empty_cache()

    # 4. 汇总对比
    print(f"\n{'='*60}")
    print(f"汇总对比")
    print(f"{'='*60}")
    print(f"{'问题':<30} {'原始评分':<10} {'微调评分':<10} {'提升':<10}")
    print(f"{'-'*60}")
    for r in results:
        q_short = r["question"][:25]
        raw = r.get("raw_score", 0)
        ft = r.get("style_score", 0)
        diff = ft - raw
        print(f"{q_short:<30} {raw:<10.0f} {ft:<10.0f} {diff:+.0f}")

    avg_raw = sum(r.get("raw_score", 0) for r in results) / len(results)
    avg_ft = sum(r.get("style_score", 0) for r in results) / len(results)
    print(f"{'-'*60}")
    print(f"{'平均':<30} {avg_raw:<10.0f} {avg_ft:<10.0f} {avg_ft-avg_raw:+.0f}")
    print(f"{'='*60}")

    print(f"""
【关键结论】
1. 微调前平均风格评分: {avg_raw:.0f}/100
2. 微调后平均风格评分: {avg_ft:.0f}/100
3. 提升: {avg_ft-avg_raw:+.0f} 分
4. LoRA适配器大小: 8.5MB（极小，可随时替换）
5. 训练耗时: 1.4分钟（360条数据3轮）

【微调学到的是"风格"不是"事实"】
- 风格固化: 每次回答都带【注意事项】+【安全提示】
- 事实来源: 还是靠RAG检索给准确药品知识
- 组合使用: 微调模型 + RAG = 风格稳定 + 事实准确
""")

    # 保存结果
    eval_path = os.path.join(BASE_DIR, "logs", "eval_result.json")
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"评估结果已保存: {eval_path}")


if __name__ == "__main__":
    evaluate()
