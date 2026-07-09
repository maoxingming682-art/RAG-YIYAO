"""
10_eval_finetune_v2.py
升级版评估：测试5类问题，验证机械化问题是否解决
"""
import os, sys, json, torch, time

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = os.path.join(BASE_DIR, "lora_output_7b")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 5类测试问题
TEST_QUESTIONS = [
    # 1.用法用量（应该简洁直接）
    {"type": "用法用量", "q": "依托度酸片怎么服用？", "expect": "简洁给剂量+简短提醒"},
    # 2.不良反应（应该列举+提醒）
    {"type": "不良反应", "q": "酚咖片有什么不良反应？", "expect": "列举反应+就医提醒"},
    # 3.禁忌人群（应该重点警告）
    {"type": "禁忌", "q": "复方泛影葡胺注射液有什么禁忌？", "expect": "重点警告+遵医嘱"},
    # 4.闲聊（不应该套模板）
    {"type": "闲聊", "q": "你好", "expect": "正常打招呼，不带药学模板"},
    # 5.超纲（知识库可能没有）
    {"type": "超纲", "q": "阿莫西林胶囊怎么吃？", "expect": "诚实说没有+建议咨询"},
    # 6.错字（应该提示正确药名）
    {"type": "错字", "q": "阿斯匹林肠溶胶囊的用法用量", "expect": "提示可能是阿司匹林"},
]


def generate_local(model, tokenizer, question, max_new_tokens=300):
    messages = [
        {"role": "system", "content": "你是专业药学咨询助手。根据问题类型用不同风格回答：简单问题简洁答，复杂问题详细答，闲聊正常对话。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate():
    print("=" * 60)
    print("第5步：评估微调效果（升级版-5类问题）")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("加载微调模型...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, trust_remote_code=True).to("cuda")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    print(f"模型加载完成, GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB\n", flush=True)

    results = []
    for i, test in enumerate(TEST_QUESTIONS):
        print(f"\n{'─'*60}")
        print(f"测试{i+1} [{test['type']}]: {test['q']}")
        print(f"期望: {test['expect']}")
        print(f"{'─'*60}")

        t0 = time.time()
        answer = generate_local(model, tokenizer, test["q"])
        t1 = time.time()

        print(f"耗时: {t1-t0:.1f}s")
        print(f"回答: {answer[:300]}")

        # 检查是否机械化（每条都有一样的模板）
        has_template = "【注意事项】" in answer and "【安全提示】" in answer
        is_chitchat = test["type"] == "闲聊"
        is_out_of_scope = test["type"] == "超纲"
        has_typo_fix = test["type"] == "错字" and ("可能是" in answer or "即" in answer)

        # 评分
        if is_chitchat:
            score = 100 if not has_template else 40
            issue = "闲聊不应套模板" if has_template else "正常对话✓"
        elif is_out_of_scope:
            score = 100 if ("没有" in answer or "暂未" in answer or "建议" in answer) else 60
            issue = "诚实兜底✓" if score == 100 else "可能编造了信息"
        elif has_typo_fix:
            score = 100 if has_typo_fix else 70
            issue = "纠错提示✓" if has_typo_fix else "未提示正确药名"
        else:
            score = 100
            issue = "正常回答✓"

        print(f"\n评分: {score}/100  分析: {issue}")
        results.append({"type": test["type"], "question": test["q"], "answer": answer, "score": score, "issue": issue})

    del model
    torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总评估")
    print(f"{'='*60}")
    print(f"{'类型':<12} {'评分':<8} {'问题':<30} {'分析':<20}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['type']:<12} {r['score']:<8} {r['question'][:28]:<30} {r['issue'][:18]:<20}")
    avg = sum(r["score"] for r in results) / len(results)
    print(f"{'-'*60}")
    print(f"{'平均':<12} {avg:.0f}")
    print(f"{'='*60}")

    # 关键结论
    chitchat_ok = all(r["score"] >= 80 for r in results if r["type"] == "闲聊")
    outscope_ok = all(r["score"] >= 80 for r in results if r["type"] == "超纲")
    no_mechanical = all(not ("【注意事项】" in r["answer"] and "【安全提示】" in r["answer"])
                       for r in results if r["type"] not in ["用法用量"])

    print(f"""
【关键结论】
1. 闲聊不套模板: {'✓ 是' if chitchat_ok else '✗ 否'}
2. 超纲诚实兜底: {'✓ 是' if outscope_ok else '✗ 否'}
3. 回答不再死板: {'✓ 是' if no_mechanical else '✗ 部分仍套模板'}
4. 平均评分: {avg:.0f}/100
""")

    eval_path = os.path.join(BASE_DIR, "logs", "eval_v2_result.json")
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {eval_path}")


if __name__ == "__main__":
    evaluate()
