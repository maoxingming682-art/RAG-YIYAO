"""05_baseline_full.py - 全量注入对比版"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


def build_full_prompt(question, all_knowledge):
    knowledge_text = "\n\n".join(f"【知识{i+1}】{item['text']}" for i, item in enumerate(all_knowledge))
    return f"""你是专业药学咨询助手。请基于以下全部药品知识回答用户问题。
【全部药品知识（共{len(all_knowledge)}条）】
{knowledge_text}
【用户问题】
{question}
请回答："""


def baseline_full_answer(question, max_items=None):
    print("=" * 60)
    print("第5步：全量注入对比版（Baseline）")
    print("=" * 60)
    print(f"问题: {question}\n")
    with open(os.path.join(DATA_DIR, "drug_knowledge.json"), encoding="utf-8") as f:
        all_knowledge = json.load(f)
    if max_items: all_knowledge = all_knowledge[:max_items]
    prompt = build_full_prompt(question, all_knowledge)
    est = int(len(prompt) * 1.5)
    print(f"全量注入: 知识{len(all_knowledge)}条, prompt{len(prompt):,}字符, token~{est:,}")
    print(f"\n对比: 全量~{est:,} vs 检索~{int(5*300*1.5):,}, 节省{100-int(5*300*1.5/max(1,est)*100)}%")
    if est > 128000:
        print(f"\n  ⚠️ 超过128K上下文限制！无法调用——必须用向量检索")
        return {"est_prompt_tokens": est, "status": "exceeded_context", "answer": None}
    elif est > 32000:
        print(f"\n  ⚠️ 超过32K，跳过调用省钱")
        return {"est_prompt_tokens": est, "status": "skipped_to_save_cost", "answer": None}
    else:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120)
        t0 = time.time()
        stream = client.chat.completions.create(model=LLM_MODEL, messages=[
            {"role":"system","content":"你是专业药学咨询助手。"},
            {"role":"user","content":prompt}], temperature=0.3, max_tokens=1024, stream=True)
        chunks = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
        answer = "".join(chunks).strip()
        print(f"  生成耗时: {time.time()-t0:.1f}s\n【回答】\n{answer}")
        client.close()
        return {"est_prompt_tokens": est, "status": "ok", "answer": answer}

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "依托度酸片怎么服用？"
    baseline_full_answer(q)
