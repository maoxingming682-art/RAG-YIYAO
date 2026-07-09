"""
15_query_rewrite.py
问题改写 + 对话历史消解
解决用户提问模糊/口语化/指代不清的问题

策略：
1. 对话历史消解：用上下文补全指代（"它"→具体药名）
2. 问题改写：模糊问题→清晰问题（"头疼"→"头疼吃什么药"）
3. 多查询扩展：一个问题生成多个变体分别检索
"""
import os, sys, json, time, numpy as np, torch, torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===== 模型和向量库缓存 =====
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
    """向量检索"""
    _load()
    qv = embed(question)
    norms = _vectors / (np.linalg.norm(_vectors, axis=1, keepdims=True) + 1e-10)
    sims = norms @ qv
    idx = np.argsort(sims)[::-1][:top_k]
    return [{"text": _chunks[i]["text"], "source": _chunks[i]["source"],
             "similarity": float(sims[i])} for i in idx]


def call_llm(prompt, system="你是专业药学咨询助手", temperature=0.3, max_tokens=300):
    """调外部LLM"""
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60)
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


# ===== 策略1：对话历史消解 =====
def resolve_context(question, history):
    """
    用对话历史补全模糊问题中的指代
    history: [{"question": "布洛芬怎么吃", "answer": "..."}, ...]
    """
    if not history:
        return question, False

    # 检测是否包含指代词
    indicators = ["它", "它呢", "那个", "这个", "上面的", "刚才", "呢", "也是吗", "呢？"]
    has_reference = any(ind in question for ind in indicators) or len(question) < 10

    if not has_reference:
        return question, False

    # 构建上下文
    context = "\n".join([f"用户第{i+1}轮问了: {h['question'][:50]}" for i, h in enumerate(history[-3:])])

    prompt = f"""你是问题理解助手。用户当前问题可能包含指代词或省略，请结合对话历史补全为完整问题。

【对话历史】
{context}

【当前问题】
{question}

【任务】
如果当前问题包含"它""那个""呢"等指代词或省略，结合历史补全为完整的药品问题。
如果当前问题本身完整清晰，直接返回原问题。
只输出补全后的问题，不要其他文字。"""

    resolved = call_llm(prompt, system="你是问题理解助手，只输出改写后的问题", temperature=0.1, max_tokens=100)
    return resolved.strip(), resolved.strip() != question


# ===== 策略2：问题改写 =====
def rewrite_query(question):
    """
    模糊问题→清晰问题
    "头疼" → "头疼吃什么药？头疼的用药建议"
    "拉肚子" → "腹泻吃什么药？腹泻的用药治疗"
    """
    # 检测是否过于模糊（短且无明确意图）
    is_vague = len(question) < 12 and not any(kw in question for kw in
        ["怎么", "什么", "哪些", "用法", "用量", "不良反应", "禁忌", "能吃", "能用", "副作用"])

    if not is_vague:
        return question, False

    prompt = f"""你是药学咨询问题的优化助手。用户的问题比较模糊，请改写为更清晰、更适合检索药品知识库的问题。

【原始问题】
{question}

【改写规则】
1. 补充用药意图（"吃什么药""怎么治疗""用药建议"）
2. 把口语转为医学术语（"拉肚子"→"腹泻"，"头疼"→"头痛"）
3. 保留用户原始关键词
4. 只输出改写后的问题，不要其他文字

【示例】
"头疼" → "头痛吃什么药？头痛的用药建议和注意事项"
"拉肚子" → "腹泻吃什么药？腹泻的用药治疗"
"发烧" → "发热吃什么退烧药？发热的用药建议"

请改写："""

    rewritten = call_llm(prompt, system="你是问题优化助手，只输出改写后的问题", temperature=0.2, max_tokens=100)
    return rewritten.strip(), rewritten.strip() != question


# ===== 策略3：多查询扩展 =====
def multi_query_expand(question, n=3):
    """
    一个问题生成n个变体，分别检索后合并
    "感冒了吃什么药" → ["感冒用药推荐", "感冒症状药物治疗", "风寒风热感冒用药"]
    """
    prompt = f"""你是药学检索助手。请把以下问题改写成{n}个不同角度的变体，用于从药品知识库中检索更全面的结果。

【原始问题】
{question}

【要求】
1. 每个变体从不同角度表达相同意图
2. 包含口语化和专业化两种表达
3. 每行一个变体，不要编号不要其他文字

【示例】
问题"感冒了吃什么药"的变体：
感冒用药推荐
感冒症状的药物治疗
风寒风热感冒吃什么药

请生成{question}的{n}个变体："""

    result = call_llm(prompt, system="你是检索变体生成助手，每行输出一个变体", temperature=0.4, max_tokens=150)
    variants = [v.strip() for v in result.split("\n") if v.strip()][:n]
    if not variants:
        variants = [question]
    return variants


def retrieve_multi(questions, top_k=TOP_K):
    """多查询检索，合并去重取top_k"""
    all_results = []
    seen_sources = set()

    for q in questions:
        results = retrieve(q, top_k=top_k)
        for r in results:
            if r["source"] not in seen_sources:
                all_results.append(r)
                seen_sources.add(r["source"])

    # 按相似度排序取top_k
    all_results.sort(key=lambda x: -x["similarity"])
    return all_results[:top_k]


# ===== 完整链路 =====
def smart_retrieve(question, history=None):
    """
    智能检索：对话消解 → 问题改写 → 多查询扩展 → 检索
    返回: (检索结果, 改写信息)
    """
    print(f"\n{'─'*60}")
    print(f"原始问题: {question}")
    print(f"{'─'*60}")

    current_q = question
    steps = []

    # 1. 对话历史消解
    if history:
        resolved, changed = resolve_context(current_q, history)
        if changed:
            print(f"  步骤1 对话消解: {current_q} → {resolved}")
            steps.append(f"对话消解: {resolved}")
            current_q = resolved
        else:
            print(f"  步骤1 对话消解: 无需消解")

    # 2. 问题改写
    rewritten, changed = rewrite_query(current_q)
    if changed:
        print(f"  步骤2 问题改写: {current_q} → {rewritten}")
        steps.append(f"问题改写: {rewritten}")
        current_q = rewritten
    else:
        print(f"  步骤2 问题改写: 无需改写")

    # 3. 多查询扩展
    variants = multi_query_expand(current_q, n=3)
    print(f"  步骤3 多查询扩展: {variants}")

    # 4. 检索（用多个变体）
    if len(variants) > 1:
        results = retrieve_multi(variants, top_k=TOP_K)
        print(f"  步骤4 多查询检索: 合并{len(variants)}个变体 → {len(results)}条结果")
    else:
        results = retrieve(current_q, top_k=TOP_K)
        print(f"  步骤4 单查询检索: {len(results)}条结果")

    # 显示结果
    max_sim = results[0]["similarity"] if results else 0
    print(f"\n  top1相似度: {max_sim:.4f}")
    print(f"  top1来源: {results[0]['source'][:40] if results else '无'}")
    if max_sim < 0.6:
        print(f"  ⚠️ 相似度<0.6，判定为超纲问题")

    return results, {"original": question, "final": current_q, "steps": steps, "variants": variants}


def demo():
    print("=" * 60)
    print("  问题模糊处理策略演示")
    print("=" * 60)

    # 测试场景
    tests = [
        # 1. 太短模糊
        {"q": "头疼", "history": None, "label": "太短模糊"},
        # 2. 口语化
        {"q": "拉肚子吃什么药", "history": None, "label": "口语化"},
        # 3. 指代不清（有对话历史）
        {"q": "它呢", "history": [{"question": "布洛芬缓释胶囊怎么吃", "answer": "..."}], "label": "指代不清"},
        # 4. 正常问题（不应改写）
        {"q": "依托度酸片怎么服用？用法用量是什么？", "history": None, "label": "正常问题"},
        # 5. 多意图
        {"q": "感冒了吃什么药", "history": None, "label": "多查询扩展"},
    ]

    for test in tests:
        print(f"\n{'='*60}")
        print(f"场景: {test['label']}")
        print(f"{'='*60}")
        results, info = smart_retrieve(test["q"], test["history"])
        print(f"\n  改写步骤: {info['steps'] if info['steps'] else '无（原问题已清晰）'}")
        print(f"  变体: {info['variants']}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"策略汇总")
    print(f"{'='*60}")
    print(f"""
【3层模糊问题处理策略】

1. 对话历史消解
   - 检测指代词（它/那个/呢）
   - 结合历史补全为完整问题
   - "它呢" → "布洛芬缓释胶囊的不良反应呢"

2. 问题改写
   - 检测模糊问题（短+无明确意图）
   - 补充用药意图+口语转术语
   - "头疼" → "头痛吃什么药？头痛的用药建议和注意事项"

3. 多查询扩展
   - 一个问题生成3个变体
   - 分别检索后合并去重
   - "感冒了吃什么药" → 3个角度检索更全面

【触发条件】
- 对话消解：有历史+含指代词 → 触发
- 问题改写：问题<12字+无明确意图 → 触发
- 多查询扩展：所有问题都做（提升召回率）
- 正常问题：跳过消解和改写，直接多查询检索
""")


if __name__ == "__main__":
    demo()
