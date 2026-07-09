"""
04_generate.py
拼 prompt + 调 LLM 生成答案（向量检索版 RAG）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K


def build_rag_prompt(question, retrieved_chunks):
    knowledge_text = "\n\n".join(
        f"【知识块{i+1}】(来源: {c['source'][:40]}, 相似度: {c.get('similarity',0):.2f})\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    # 检测最高相似度
    max_sim = max((c.get("similarity", 0) for c in retrieved_chunks), default=0)
    # 提取检索到的药品名（从source字段）
    retrieved_drug_names = [c.get("source", "")[:20] for c in retrieved_chunks[:3]]

    return f"""你是专业药学咨询助手。请基于检索到的药品知识回答用户问题。

【回答风格规则（重要）】
根据问题类型调整回答风格，不要每条都用同一个模板：
- 用法用量查询 → 简洁直接给剂量，末尾简短提醒"遵医嘱"
- 不良反应咨询 → 列举不良反应，提醒"严重时就医"
- 禁忌/孕妇/儿童 → 重点警告，强调"遵医嘱"
- 一般咨询 → 自然对话语气，像药师面对面交流
- 非用药问题（如打招呼/闲聊）→ 正常对话，不套药学模板

【超纲处理规则】
- 如果检索到的知识与问题不完全匹配（最高相似度<0.6），诚实说"我的知识库暂未收录该药品的详细信息，建议查看说明书或咨询药师"
- 不要编造知识库里没有的药品信息
- 如果用户提到的药名与检索到的来源药名不完全一致（可能有错字），提示"您提到的[用户药名]可能是[正确药名]"

【安全规则】
- 涉及处方药提示"需凭处方购买"
- 涉及用药建议附加"具体用药请咨询药师或医生"
- 但安全提示要自然融入回答，不要每条都用一样的格式

【检索到的药品知识】（最高相似度: {max_sim:.2f}）
{knowledge_text}

【用户问题】
{question}

请根据以上规则自然地回答："""


def generate_answer(question, retrieved_chunks):
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60)
    prompt = build_rag_prompt(question, retrieved_chunks)
    messages = [
        {"role": "system", "content": "你是专业药学咨询助手。根据问题类型用不同风格回答：简单问题简洁答，复杂问题详细答，闲聊正常对话。基于知识库回答不编造，注意药名纠错和超纲兜底。"},
        {"role": "user", "content": prompt},
    ]
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            temperature=0.3, max_tokens=1024, stream=True,
        )
        chunks = []
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    chunks.append(delta.content)
        result = "".join(chunks).strip()
        if not result:
            raise Exception("LLM 返回空回复")
        return result
    finally:
        client.close()


def rag_answer(question):
    print("=" * 60)
    print("第4步：向量检索 RAG 生成答案")
    print("=" * 60)
    print(f"问题: {question}\n")

    import importlib.util
    spec = importlib.util.spec_from_file_location("retrieve_mod", os.path.join(os.path.dirname(__file__), "03_retrieve.py"))
    retrieve_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrieve_mod)

    retrieved = retrieve_mod.retrieve(question, top_k=TOP_K)
    prompt = build_rag_prompt(question, retrieved)
    prompt_chars = len(prompt)
    est_prompt_tokens = int(prompt_chars * 1.5)

    print(f"\n拼装 prompt: 字符数 {prompt_chars:,}, 估算 token ~{est_prompt_tokens:,}")
    print(f"\n调用 LLM ({LLM_MODEL}) 生成答案...")
    t0 = time.time()
    answer = generate_answer(question, retrieved)
    t1 = time.time()
    print(f"\n生成耗时: {t1-t0:.1f}s")
    print(f"\n{'='*60}\n【回答】\n{'='*60}")
    print(answer)
    return {"question": question, "answer": answer, "retrieved_chunks": retrieved,
            "prompt_chars": prompt_chars, "est_prompt_tokens": est_prompt_tokens, "gen_time": t1-t0}


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "依托度酸片怎么服用？"
    result = rag_answer(q)
