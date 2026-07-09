"""
19_symptom_full_flow.py
症状描述完整端到端流程：
脱敏 → 分诊 → 症状提取 → RAG检索 → 微调生成 → 校验 → 安全层

整合所有模块，演示用户描述症状时的完整处理链路
"""
import os, sys, json, time, re, numpy as np, torch, torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, BASE_DIR, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===== 加载所有模块 =====
from importlib import import_module

# 导入安全层
import importlib.util
def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

safety = load_mod("18_safety_layer")
triage_mod = load_mod("17_triage")

# ===== 模型缓存 =====
_embed_tok = None
_embed_mdl = None
_vectors = None
_chunks = None


def _load_embed():
    global _embed_tok, _embed_mdl, _vectors, _chunks
    if _vectors is not None:
        return
    from transformers import AutoTokenizer, AutoModel
    print("加载embedding模型...", flush=True)
    _embed_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    _embed_mdl = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
    _embed_mdl.eval()
    _vectors = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    _chunks = json.load(open(os.path.join(DATA_DIR, "chunks.json"), encoding="utf-8"))
    print(f"向量库: {len(_chunks)}块", flush=True)


def embed(text):
    _load_embed()
    inputs = _embed_tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = _embed_mdl(**inputs)
    am = inputs["attention_mask"]
    te = out.last_hidden_state
    ime = am.unsqueeze(-1).expand(te.size()).float()
    emb = torch.sum(te * ime, 1) / torch.clamp(ime.sum(1), min=1e-9)
    return F.normalize(emb, p=2, dim=1)[0].numpy()


def retrieve(question, top_k=TOP_K):
    _load_embed()
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


# ===== 脱敏 =====
DESENSITIZE_RULES = [
    (re.compile(r'1[3-9]\d{9}'), '[手机号]'),
    (re.compile(r'\d{17}[\dXx]'), '[身份证]'),
    (re.compile(r'[\w.-]+@[\w.-]+\.\w+'), '[邮箱]'),
    (re.compile(r'\d{16,19}'), '[卡号]'),
    (re.compile(r'[\u4e00-\u9fa5]{2,}(?:省|市|区|县|路|街|号|楼|室|小区|栋|单元|村|镇)[\d\u4e00-\u9fa5]*'), '[地址]'),
]

def desensitize(text):
    masked = text
    for pattern, replacement in DESENSITIZE_RULES:
        masked = pattern.sub(replacement, masked)
    return masked


# ===== 症状提取 =====
def extract_drugs_from_symptoms(user_input):
    prompt = f"""你是药学咨询助手。用户描述了症状，请分析并提取可能的药品信息。

【用户描述】
{user_input}

输出严格JSON：
```json
{{
  "is_symptom_description": true,
  "symptoms": ["头痛", "发烧"],
  "possible_drugs": ["对乙酰氨基酚片", "布洛芬缓释胶囊"],
  "possible_categories": ["感冒药", "退烧药"],
  "severity": "轻度",
  "needs_doctor": false,
  "possible_condition": "感冒"
}}
```
只推荐非处方药(OTC)。如果症状严重，needs_doctor=true。只输出JSON。"""

    result = call_llm(prompt, system="你是药学分析助手，只输出JSON", temperature=0.2, max_tokens=500)
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


def search_drugs_in_rag(drug_names, categories, top_k=3):
    all_queries = drug_names + [f"{cat}有哪些" for cat in categories]
    all_queries = list(set(all_queries))
    all_results = []
    seen_sources = set()
    for query in all_queries:
        results = retrieve(query, top_k=top_k)
        for r in results:
            if r["similarity"] > 0.55 and r["source"] not in seen_sources:
                all_results.append(r)
                seen_sources.add(r["source"])
    return all_results[:8]


# ===== 完整流程 =====
def handle_symptom(user_input):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"用户输入: {user_input}")
    print(f"{'='*60}")

    # 第1层：脱敏
    print(f"\n[第1层] 脱敏过滤...", flush=True)
    masked_input = desensitize(user_input)
    if masked_input != user_input:
        print(f"  原始: {user_input}")
        print(f"  脱敏: {masked_input}")
    else:
        print(f"  无需脱敏")
    current_input = masked_input

    # 第2层：分诊
    print(f"\n[第2层] 分诊评估...", flush=True)
    t0 = time.time()
    triage_result = triage_mod.triage(current_input)
    t1 = time.time()
    level = triage_result.get("level", "未知")
    print(f"  分诊级别: {level} (耗时{t1-t0:.1f}s)")

    # 急症/重症：直接走安全层，不走后续
    if level in ("急症", "重症") or triage_result.get("action") in ("immediate_medical", "see_doctor_soon"):
        print(f"\n  → {level}拦截，不走药品推荐流程")
        safe = safety.apply_safety(current_input, "", [], triage_result)
        print(f"\n{'─'*60}")
        print(f"【最终回复】")
        print(f"{'─'*60}")
        print(safe)
        print(f"\n总耗时: {time.time()-t_start:.1f}s")
        return safe

    # 第3层：症状→药品提取
    print(f"\n[第3层] 症状分析+药品提取...", flush=True)
    t2 = time.time()
    analysis = extract_drugs_from_symptoms(current_input)
    t3 = time.time()
    is_symptom = analysis.get("is_symptom_description", False)
    print(f"  是否症状描述: {is_symptom} (耗时{t3-t2:.1f}s)")
    print(f"  症状: {analysis.get('symptoms', [])}")
    print(f"  可能药品: {analysis.get('possible_drugs', [])}")
    print(f"  可能疾病: {analysis.get('possible_condition', '')}")

    if not is_symptom:
        print(f"\n  → 非症状描述，走正常药品RAG流程")
        retrieved = retrieve(current_input, top_k=TOP_K)
    else:
        # 第4层：RAG查药品详情
        print(f"\n[第4层] RAG查药品详情...", flush=True)
        t4 = time.time()
        retrieved = search_drugs_in_rag(
            analysis.get("possible_drugs", []),
            analysis.get("possible_categories", [])
        )
        t5 = time.time()
        print(f"  检索到 {len(retrieved)} 条 (耗时{t5-t4:.1f}s)")
        for i, r in enumerate(retrieved[:3]):
            print(f"    [{i+1}] sim={r['similarity']:.4f} {r['source'][:30]}")

    # 第5层：超纲检测
    print(f"\n[第5层] 超纲检测...", flush=True)
    max_sim = max((r["similarity"] for r in retrieved), default=0)
    scope_status, _ = safety.detect_out_of_scope(retrieved)
    print(f"  最高相似度: {max_sim:.4f}")
    print(f"  超纲状态: {scope_status}")

    if scope_status == "out_of_scope":
        print(f"  → 超纲拦截，不生成答案")
        safe = safety.apply_safety(current_input, "", retrieved, triage_result)
        print(f"\n{'─'*60}")
        print(f"【最终回复】")
        print(f"{'─'*60}")
        print(safe)
        print(f"\n总耗时: {time.time()-t_start:.1f}s")
        return safe

    # 第6层：LLM生成答案
    print(f"\n[第6层] LLM生成用药建议...", flush=True)
    knowledge_text = "\n\n".join(
        f"【药品{i+1}】{r['text'][:200]}" for i, r in enumerate(retrieved)
    )
    symptoms = "、".join(analysis.get("symptoms", []))
    condition = analysis.get("possible_condition", "")

    gen_prompt = f"""你是专业药学咨询助手。用户描述了症状，请结合检索到的药品知识给出建议。

【用户症状】
{current_input}

【症状分析】
- 症状：{symptoms}
- 可能情况：{condition}

【知识库药品信息】
{knowledge_text}

【回答要求】
1. 简要分析症状（不是诊断，是"可能是XX"）
2. 推荐适合的非处方药+用法用量
3. 注意事项和禁忌
4. 安全提示：明确"我不是医生，以上不是诊断"
5. 回答自然，不要套固定模板"""

    t6 = time.time()
    raw_answer = call_llm(gen_prompt,
        system="你是专业药学咨询助手。基于症状和药品知识库给建议。不做诊断。",
        temperature=0.4, max_tokens=800)
    t7 = time.time()
    print(f"  生成完成 (耗时{t7-t6:.1f}s)")
    print(f"  原始答案: {raw_answer[:100]}...")

    # 第7层：安全层处理
    print(f"\n[第7层] 安全层处理...", flush=True)
    safe_answer = safety.apply_safety(current_input, raw_answer, retrieved, triage_result)
    print(f"  安全处理完成")

    # 最终输出
    print(f"\n{'─'*60}")
    print(f"【最终回复】")
    print(f"{'─'*60}")
    print(safe_answer)

    total = time.time() - t_start
    print(f"\n总耗时: {total:.1f}s")
    return safe_answer


def main():
    print("=" * 60)
    print("  症状描述完整端到端流程演示")
    print("  脱敏→分诊→症状提取→RAG检索→生成→安全层")
    print("=" * 60)

    tests = [
        # 1. 轻症+含隐私
        "我头痛发烧流鼻涕三天了，手机13812345678，吃什么药好？",
        # 2. 急症（应被拦截）
        "突然剧烈胸痛呼吸困难出汗",
        # 3. 正常药品查询（非症状）
        "依托度酸片怎么服用？",
    ]

    for i, test in enumerate(tests):
        print(f"\n\n{'#'*60}")
        print(f"# 测试{i+1}")
        print(f"{'#'*60}")
        try:
            handle_symptom(test)
        except Exception as e:
            print(f"错误: {e}")
            if "429" in str(e):
                print("API限流，等待60秒...")
                time.sleep(60)
                handle_symptom(test)
        if i < len(tests) - 1:
            time.sleep(5)

    # 汇总
    print(f"\n{'='*60}")
    print(f"完整流程总结")
    print(f"{'='*60}")
    print(f"""
【用户描述症状时的完整处理流程】

第1层 脱敏过滤
  手机号/身份证/地址 → 占位符
  病情信息（症状/病史）保留

第2层 分诊评估
  规则引擎（毫秒级）→ 红线症状拦截
  LLM分诊（秒级）→ 4级评估
  ├─ 急症 → 拦截，直接返回"立即就医"
  ├─ 重症 → 拦截，返回"尽快就医，不推药"
  ├─ 中症 → 继续，推药+建议就医
  └─ 轻症 → 继续，推药+观察

第3层 症状→药品提取
  LLM分析症状 → 提取可能OTC药品名
  "头痛发烧" → ["对乙酰氨基酚","布洛芬"]

第4层 RAG检索
  用药品名查知识库 → 获取准确用法用量

第5层 超纲检测
  相似度<0.60 → 拦截编造，返回"知识库没有"
  0.60-0.65 → 前置提醒"信息不完整"

第6层 LLM生成
  结合症状+药品知识 → 生成用药建议

第7层 安全层
  强免责声明（每条必带）
  引导就医（根据分诊级别）

【3种拦截点】
  分诊拦截：急症/重症不推药
  超纲拦截：知识库没有不编造
  安全兜底：每条回答必带免责
""")


if __name__ == "__main__":
    main()
