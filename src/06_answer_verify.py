"""06_answer_verify.py - 答案校验回调"""
import os, sys, time, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K


def call_llm(messages, temperature=0.2, max_tokens=1024):
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60)
    try:
        stream = client.chat.completions.create(model=LLM_MODEL, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=True)
        chunks = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
        return "".join(chunks).strip()
    finally:
        client.close()


def verify_answer(question, answer, retrieved_chunks):
    print("=" * 60)
    print("第6步：答案校验回调")
    print("=" * 60)
    knowledge_text = "\n\n".join(f"【块{i+1}】{c['text']}" for i, c in enumerate(retrieved_chunks))
    verify_prompt = f"""你是答案质量审核员。请审核以下药学咨询答案的质量。
【用户问题】{question}
【检索到的知识块】{knowledge_text}
【待审核的答案】{answer}
请从3个维度审核，输出严格JSON：
```json
{{"faithfulness":"通过/不通过","faithfulness_reason":"...",
"relevance":"通过/不通过","relevance_reason":"...",
"safety_notice":"通过/不通过","safety_reason":"...",
"overall_pass":true,"score":85,"issues":[],"suggestion":""}}
```
只输出JSON。"""
    t0 = time.time()
    reply = call_llm([
        {"role":"system","content":"你是答案质量审核员，严格审核药学咨询答案。只输出JSON。"},
        {"role":"user","content":verify_prompt}], temperature=0.1, max_tokens=800)
    result = None
    m = re.search(r'```json\s*\n?(.*?)```', reply, re.DOTALL)
    if m:
        try: result = json.loads(m.group(1).strip())
        except: pass
    if not result:
        try:
            s, e = reply.find('{'), reply.rfind('}')
            if s != -1: result = json.loads(reply[s:e+1])
        except: pass
    if not result:
        return {"overall_pass": False, "score": 0, "issues": ["校验解析失败"], "raw": reply}
    print(f"\n校验结果(耗时{time.time()-t0:.1f}s):")
    print(f"  通过: {'✓' if result.get('overall_pass') else '✗'} 评分: {result.get('score','N/A')}")
    print(f"  忠实度: {result.get('faithfulness','')} - {result.get('faithfulness_reason','')[:60]}")
    print(f"  相关性: {result.get('relevance','')} - {result.get('relevance_reason','')[:60]}")
    print(f"  安全提示: {result.get('safety_notice','')} - {result.get('safety_reason','')[:60]}")
    return result


def rag_with_verification(question, max_retries=2):
    import importlib.util
    def load(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    retrieve_mod = load("03_retrieve")
    generate_mod = load("04_generate")
    current_q = question
    for attempt in range(max_retries + 1):
        print(f"\n--- 第{attempt+1}次尝试 ---")
        retrieved = retrieve_mod.retrieve(current_q, top_k=TOP_K)
        answer = generate_mod.generate_answer(current_q, retrieved)
        verify_result = verify_answer(current_q, answer, retrieved)
        if verify_result.get("overall_pass"):
            print(f"\n{'✓'*60}\n校验通过！最终答案:\n{'✓'*60}\n\n{answer}")
            return {"question": question, "answer": answer, "verify": verify_result, "attempts": attempt+1, "success": True}
        if attempt < max_retries:
            suggestion = verify_result.get("suggestion", "")
            rewrite = call_llm([{"role":"system","content":"你擅长改写问题以提升检索效果。只输出改写后的问题。"},
                {"role":"user","content":f"用户问题:{question}\n校验建议:{suggestion}\n请改写问题。"}], temperature=0.3, max_tokens=100)
            current_q = rewrite.strip()
            print(f"改写问题: {current_q}")
        else:
            return {"question": question, "answer": answer, "verify": verify_result, "attempts": attempt+1, "success": False}

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "依托度酸片怎么服用？"
    result = rag_with_verification(q)
    print(f"\n第6步完成，共尝试{result['attempts']}次，{'通过' if result['success'] else '未通过'}校验")
