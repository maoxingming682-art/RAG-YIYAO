"""
16_symptom_handler.py
症状描述处理：用户描述病情 → LLM提取可能药品 → RAG查药品详情 → 生成建议

解决：知识库只有药品知识没有症状诊断知识时的瘫痪问题
核心思路：用LLM自带的医学知识做"症状→药品"桥梁，用RAG给准确的药品用法
"""
import os, sys, json, time, numpy as np, torch, torch.nn.functional as F, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_embed_tok = None
_embed_mdl = None
_vectors = None
_chunks = None


def _load():
    global _embed_tok, _embed_mdl, _vectors, _chunks
    if _vectors is not None:
        return
    from transformers import AutoTokenizer, AutoModel
    _embed_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    _embed_mdl = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
    _embed_mdl.eval()
    _vectors = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    _chunks = json.load(open(os.path.join(DATA_DIR, "chunks.json"), encoding="utf-8"))
    print(f"向量库加载: {len(_chunks)}块", flush=True)


def embed(text):
    _load()
    inputs = _embed_tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = _embed_mdl(**inputs)
    am = inputs["attention_mask"]
    te = out.last_hidden_state
    ime = am.unsqueeze(-1).expand(te.size()).float()
    emb = torch.sum(te * ime, 1) / torch.clamp(ime.sum(1), min=1e-9)
    return F.normalize(emb, p=2, dim=1)[0].numpy()


def retrieve(question, top_k=TOP_K):
    _load()
    qv = embed(question)
    norms = _vectors / (np.linalg.norm(_vectors, axis=1, keepdims=True) + 1e-10)
    sims = norms @ qv
    idx = np.argsort(sims)[::-1][:top_k]
    return [{"text": _chunks[i]["text"], "source": _chunks[i]["source"],
             "similarity": float(sims[i])} for i in idx]


def call_llm(prompt, system="你是专业药学咨询助手", temperature=0.3, max_tokens=800):
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=90)
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, stream=True)
        chunks = [c.choices[0].delta.content for c in stream if c.choices and c.choices[0].delta.content]
        return "".join(chunks).strip()
    finally:
        client.close()


# ===== 核心：症状→药品提取 =====
def extract_drugs_from_symptoms(user_input):
    """
    从症状描述中提取可能的药品/药品类别
    用LLM的医学知识做"症状→药品"桥梁

    返回: {
        "is_symptom": bool,        # 是否是症状描述
        "symptoms": ["头痛","发烧"],# 提取的症状
        "possible_drugs": ["对乙酰氨基酚","布洛芬"], # 推荐药品
        "possible_categories": ["感冒药"], # 推荐药品类别
        "severity": "轻/中/重",    # 严重程度
        "needs_doctor": bool,      # 是否建议就医
    }
    """
    prompt = f"""你是药学咨询助手。用户描述了自己的症状，请分析并提取可能的药品信息。

【用户描述】
{user_input}

【分析任务】
1. 判断用户是在描述症状/病情（而非直接问药品用法）
2. 提取用户提到的所有症状
3. 根据症状推荐可能的非处方药品名称（具体药品名，便于知识库检索）
4. 推荐药品类别（如"感冒药""退烧药""止痛药"）
5. 判断严重程度
6. 判断是否需要建议就医

输出严格JSON：
```json
{{
  "is_symptom_description": true,
  "symptoms": ["头痛", "发烧", "流鼻涕"],
  "possible_drugs": ["对乙酰氨基酚片", "布洛芬缓释胶囊", "复方氨酚烷胺片"],
  "possible_categories": ["感冒药", "退烧药"],
  "severity": "轻度",
  "needs_doctor": false,
  "possible_condition": "感冒"
}}
```

【重要】
- 只推荐非处方药（OTC），不推荐处方药
- 如果症状严重（高烧不退、剧烈疼痛、呼吸困难等），needs_doctor=true
- 只输出JSON，不要其他文字"""

    result = call_llm(prompt, system="你是药学分析助手，只输出JSON", temperature=0.2, max_tokens=500)

    # 解析JSON
    parsed = None
    m = re.search(r'```json\s*\n?(.*?)```', result, re.DOTALL)
    if m:
        try: parsed = json.loads(m.group(1).strip())
        except: pass
    if not parsed:
        try:
            s, e = result.find('{'), result.rfind('}')
            if s != -1: parsed = json.loads(result[s:e+1])
        except: pass

    if not parsed:
        return {"is_symptom_description": False, "possible_drugs": [], "possible_categories": [],
                "symptoms": [], "needs_doctor": True, "possible_condition": "无法判断"}

    return parsed


# ===== 症状→药品RAG查询 =====
def search_drugs_in_rag(drug_names, categories, top_k=3):
    """
    用提取出的药品名/类别在RAG知识库中检索药品详细信息
    每个药品名分别检索，合并结果
    """
    all_queries = drug_names + [f"{cat}有哪些" for cat in categories]
    all_queries = list(set(all_queries))  # 去重

    all_results = []
    seen_sources = set()

    for query in all_queries:
        results = retrieve(query, top_k=top_k)
        for r in results:
            if r["similarity"] > 0.55 and r["source"] not in seen_sources:
                all_results.append(r)
                seen_sources.add(r["source"])

    return all_results[:8]  # 最多8条


# ===== 完整症状处理链路 =====
def handle_symptom_query(user_input):
    """
    症状描述处理完整链路：
    1. 判断是否是症状描述
    2. LLM提取可能药品
    3. RAG查药品详情
    4. LLM结合症状+药品知识生成建议
    5. 安全提示（不能诊断+建议就医）
    """
    print(f"\n{'='*60}")
    print(f"症状描述处理")
    print(f"用户输入: {user_input}")
    print(f"{'='*60}")

    # 1. 症状分析+药品提取
    print(f"\n步骤1: 症状分析+药品提取...", flush=True)
    t0 = time.time()
    analysis = extract_drugs_from_symptoms(user_input)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.1f}s")
    print(f"  是否症状描述: {analysis.get('is_symptom_description', False)}")
    print(f"  症状: {analysis.get('symptoms', [])}")
    print(f"  可能药品: {analysis.get('possible_drugs', [])}")
    print(f"  药品类别: {analysis.get('possible_categories', [])}")
    print(f"  可能疾病: {analysis.get('possible_condition', '')}")
    print(f"  严重程度: {analysis.get('severity', '')}")
    print(f"  建议就医: {analysis.get('needs_doctor', False)}")

    if not analysis.get("is_symptom_description"):
        print(f"\n  → 不是症状描述，走正常RAG流程")
        return None, analysis

    # 2. RAG查药品详情
    print(f"\n步骤2: RAG查药品详情...", flush=True)
    drug_results = search_drugs_in_rag(
        analysis.get("possible_drugs", []),
        analysis.get("possible_categories", [])
    )
    t2 = time.time()
    print(f"  耗时: {t2-t1:.1f}s")
    print(f"  检索到 {len(drug_results)} 条药品知识")

    if drug_results:
        for i, r in enumerate(drug_results[:3]):
            print(f"    [{i+1}] sim={r['similarity']:.4f} {r['source'][:30]}")
    else:
        print(f"  ⚠️ 知识库未检索到相关药品，将仅基于LLM知识回答")

    # 3. 生成建议（症状+药品知识+安全提示）
    print(f"\n步骤3: 生成用药建议...", flush=True)
    knowledge_text = "\n\n".join(
        f"【药品{i+1}】{r['text'][:200]}" for i, r in enumerate(drug_results)
    ) if drug_results else "（知识库未检索到相关药品信息）"

    symptoms = "、".join(analysis.get("symptoms", []))
    condition = analysis.get("possible_condition", "")
    needs_doctor = analysis.get("needs_doctor", False)

    gen_prompt = f"""你是专业药学咨询助手。用户描述了症状，请结合检索到的药品知识给出建议。

【用户症状描述】
{user_input}

【症状分析】
- 症状：{symptoms}
- 可能情况：{condition}
- 严重程度：{analysis.get('severity', '')}

【知识库检索到的药品信息】
{knowledge_text}

【回答要求】
1. 先简要分析症状（不是诊断，是"可能是XX")
2. 根据知识库的药品信息，推荐适合的非处方药
3. 给出药品的用法用量（基于知识库内容）
4. 注意事项和禁忌
5. 安全提示（非常重要）：
   - 明确说明"我不是医生，以上不是诊断"
   - 如果症状持续或加重，建议及时就医
   - 处方药需凭处方购买
6. 回答风格自然，不要套固定模板"""

    answer = call_llm(gen_prompt,
        system="你是专业药学咨询助手。基于症状描述和药品知识库给出用药建议。不能做诊断，只能给参考建议。",
        temperature=0.4, max_tokens=800)
    t3 = time.time()
    print(f"  耗时: {t3-t2:.1f}s")

    print(f"\n{'─'*60}")
    print(f"【用药建议】")
    print(f"{'─'*60}")
    print(answer)

    return answer, analysis


def demo():
    print("=" * 60)
    print("  症状描述处理演示")
    print("  解决：用户描述病情而非直接问药品时的瘫痪问题")
    print("=" * 60)

    # 测试场景
    tests = [
        # 1. 典型症状描述
        "我头痛发烧流鼻涕三天了，吃什么药好？",
        # 2. 儿童症状
        "我儿子8岁发烧38.5度咳嗽，能吃什么药？",
        # 3. 慢性病+症状
        "我有高血压，最近关节疼，能吃什么止痛药？",
        # 4. 正常药品问题（不应走症状流程）
        "依托度酸片怎么服用？",
        # 5. 严重症状（应建议就医）
        "突然剧烈胸痛呼吸困难出汗，怎么办？",
    ]

    for i, test in enumerate(tests):
        print(f"\n\n{'#'*60}")
        print(f"# 测试{i+1}: {test[:40]}")
        print(f"{'#'*60}")
        try:
            answer, analysis = handle_symptom_query(test)
            if answer is None:
                print(f"\n  → 走正常RAG流程（非症状描述）")
        except Exception as e:
            print(f"\n  错误: {e}")
            # 限流时等一下
            if "429" in str(e):
                print(f"  API限流，等待60秒...")
                time.sleep(60)

    # 汇总
    print(f"\n{'='*60}")
    print(f"症状描述处理策略汇总")
    print(f"{'='*60}")
    print(f"""
【解决的问题】
用户描述病情（"我头痛发烧"）而非直接问药品（"布洛芬怎么吃"）时，
RAG知识库只有药品知识没有症状诊断知识，直接检索会瘫痪。

【解决方案：症状→药品桥梁】
1. LLM分析症状 → 提取可能的非处方药品名
   "头痛发烧流鼻涕" → ["对乙酰氨基酚","布洛芬","感冒药"]
2. 用药品名查RAG → 获取准确的药品用法用量
   检索到：对乙酰氨基酚用法、布洛芬用法...
3. LLM结合症状+药品知识 → 生成用药建议
   "根据您的症状可能是感冒，可选对乙酰氨基酚退烧..."
4. 安全提示 → 不做诊断+建议就医

【关键认知】
- LLM做"症状→药品"的桥梁（LLM自带医学知识）
- RAG做"药品→用法"的精确查询（知识库给准确事实）
- 组合使用：LLM的广度知识 + RAG的精确知识
- 安全底线：AI不做诊断，只给参考建议，严重症状建议就医

【为什么不直接建症状知识库】
1. 症状→药品映射太复杂（同一症状可能是多种疾病）
2. 诊断是医生的事，AI不能做（合规红线）
3. 用LLM桥梁更灵活，覆盖所有症状组合
4. RAG专注药品知识（它擅长的），症状分析交给LLM（它擅长的）
""")


if __name__ == "__main__":
    demo()
