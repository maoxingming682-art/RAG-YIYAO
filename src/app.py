"""
app.py - 药业RAG demo Web服务
前端聊天界面 + 后端API（整合所有模块）
"""
import os, sys, json, time, re, threading, traceback, csv, io, uuid, subprocess
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from config import BASE_DIR, DATA_DIR, TOP_K

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ===== 全局模型缓存（懒加载）=====
_models_loaded = False
_embed_tok = None
_embed_mdl = None
_vectors = None
_chunks = None
_drug_aliases = None

# 本地LLM模型缓存
_llm_tok = None
_llm_mdl = None
_llm_loaded = False

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_LORA_PATH = os.path.join(BASE_DIR, "lora_output")


def load_llm_model():
    """加载本地1.5B微调模型到GPU"""
    global _llm_tok, _llm_mdl, _llm_loaded
    if _llm_loaded:
        return
    print("加载本地1.5B微调模型到GPU...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    _llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    if _llm_tok.pad_token is None:
        _llm_tok.pad_token = _llm_tok.eos_token
    _llm_mdl = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    _llm_mdl = PeftModel.from_pretrained(_llm_mdl, LLM_LORA_PATH)
    _llm_mdl.eval()
    _llm_loaded = True
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"1.5B模型加载完成, GPU显存: {gpu_mem:.1f}GB", flush=True)


def local_generate(question, knowledge_text, max_new_tokens=400):
    """用本地1.5B微调模型生成答案 - 直接复述检索内容"""
    import torch
    load_llm_model()
    # 简单直接的prompt：把检索内容放前面，让模型复述+整理
    prompt = f"""根据以下药品知识回答问题，直接引用知识中的内容，不要编造。

{knowledge_text}

问题：{question}
回答："""
    messages = [
        {"role": "system", "content": "你是药学咨询助手，根据提供的药品知识回答问题，直接引用内容不要编造。"},
        {"role": "user", "content": prompt},
    ]
    text = _llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _llm_tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = _llm_mdl.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.1, do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=_llm_tok.pad_token_id,
        )
    reply = _llm_tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()

def load_embed_models():
    global _models_loaded, _embed_tok, _embed_mdl, _vectors, _chunks
    if _models_loaded:
        return
    print("加载embedding模型和向量库...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    _embed_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    _embed_mdl = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
    _embed_mdl.eval()
    _vectors = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    with open(os.path.join(DATA_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    _models_loaded = True
    print(f"加载完成: {len(_chunks)}块向量库", flush=True)


def embed_query(text):
    import torch, torch.nn.functional as F, numpy as np
    load_embed_models()
    inputs = _embed_tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = _embed_mdl(**inputs)
    am = inputs["attention_mask"]
    te = out.last_hidden_state
    ime = am.unsqueeze(-1).expand(te.size()).float()
    emb = torch.sum(te * ime, 1) / torch.clamp(ime.sum(1), min=1e-9)
    return F.normalize(emb, p=2, dim=1)[0].numpy()


def _extract_query_keywords(question):
    """从问题中提取药品名关键词（去掉问句词，剩下中文片段）
    用于关键词召回，解决"阿莫西林胶囊"vs"阿莫西林口腔崩解片"剂型不匹配导致向量检索漏召的问题。
    返回: [关键词列表]，每个2-6字
    """
    remove_words = ['怎么','如何','什么','有什么','是','的','呢','吗','能','可以','用','服用',
                    '吃','副作用','不良反应','禁忌','用法','用量','剂量','作用','效果',
                    '饭前','饭后','几次','一天','多少','儿童','小孩','孕妇','能吃吗','可以用吗',
                    '有啥','有什么','告诉','请问','帮','查','查询','相关','信息','介绍',
                    '胶囊','片','注射液','注射','分散片','崩解片','缓释','口服','冲剂','颗粒','丸','膏','栓','滴','喷雾','贴','散','糖浆','合剂','溶液','搽剂','霜','凝胶','含片','喷剂','栓剂','气雾剂','贴膏','软膏','乳膏','搽','洗剂','散剂','酊','栓','贴','膜','海绵','凝胶贴','贴剂','膜剂','海绵剂','冲洗剂','灌肠剂','吸入剂','喷鼻剂','滴眼剂','滴耳剂','滴鼻剂','眼膏','耳膏','鼻膏','直肠','阴道','尿道','肛','阴','口含','咀嚼','含服','漱口','冲洗','灌肠','吸入','喷鼻','滴眼','滴耳','滴鼻','眼','耳','鼻','直肠用','阴道用','外用','内服','内用','局部']
    q = question
    for w in remove_words:
        q = q.replace(w, '')
    # 去标点和单字
    chunks_text = re.findall(r'[\u4e00-\u9fa5]{2,8}', q)
    if not chunks_text:
        return []
    keywords = set()
    for c in chunks_text:
        if len(c) >= 2:
            keywords.add(c)
        # 拆2-3字子串，解决"阿莫西林胶囊"匹配"阿莫西林"
        for i in range(len(c) - 1):
            sub2 = c[i:i+2]
            if len(sub2) >= 2:
                keywords.add(sub2)
        for i in range(len(c) - 2):
            sub3 = c[i:i+3]
            if len(sub3) >= 3:
                keywords.add(sub3)
    # 过滤掉太短/太泛的（2字以下不要，但2字药名如"布洛"可能误删，保留2字）
    keywords = {kw for kw in keywords if len(kw) >= 2}
    for alias_target in _expand_aliases(question):
        keywords.add(alias_target)
        short = alias_target.replace("胶囊", "").replace("片", "").replace("颗粒", "")
        if len(short) >= 3:
            keywords.add(short)
    return list(keywords)


def _exact_query_terms(question):
    """High-precision phrases used to prefer exact KB updates over broad keyword hits."""
    terms = []
    cleaned = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", question)
    intent_words = [
        "怎么吃", "怎么服用", "如何服用", "有什么", "有没有", "怎么办",
        "用法用量", "用法", "用量", "剂量", "副作用", "不良反应",
        "禁忌", "作用", "功效", "多少钱", "价格", "吗", "呢", "的",
    ]
    for segment in cleaned.split():
        if len(segment) >= 4:
            terms.append(segment)
        reduced = segment
        for word in intent_words:
            reduced = reduced.replace(word, "")
        if len(reduced) >= 3:
            terms.append(reduced)
    for alias_target in _expand_aliases(question):
        if alias_target not in terms:
            terms.append(alias_target)

    seen = set()
    unique_terms = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms[:8]


def _term_variants(term):
    if not term:
        return []
    variants = [term]
    form_words = [
        "缓释胶囊", "分散片", "崩解片", "咀嚼片", "肠溶片", "控释片",
        "胶囊", "颗粒", "口服液", "注射液", "糖浆", "乳膏", "软膏",
        "凝胶", "喷雾剂", "气雾剂", "滴眼液", "滴耳液", "片", "丸", "散",
    ]
    short = term
    for word in form_words:
        short = short.replace(word, "")
    if len(short) >= 2 and short != term:
        variants.append(short)
    for alias_target in _expand_aliases(term):
        variants.append(alias_target)
        target_short = alias_target
        for word in form_words:
            target_short = target_short.replace(word, "")
        if len(target_short) >= 2 and target_short != alias_target:
            variants.append(target_short)
    seen = []
    for v in variants:
        v = v.strip()
        if len(v) >= 2 and v not in seen:
            seen.append(v)
    return seen


def retrieve(question, top_k=TOP_K):
    """混合检索：向量语义检索 + 关键词召回加分。
    解决纯向量检索在"剂型不匹配但药名相同"时漏召的问题（如阿莫西林胶囊 vs 阿莫西林口腔崩解片）。
    """
    import numpy as np
    load_embed_models()
    qv = embed_query(question)
    norms = _vectors / (np.linalg.norm(_vectors, axis=1, keepdims=True) + 1e-10)
    sims = norms @ qv

    # 1. 向量候选池：取 top 30（比 top_k 大，给关键词召回留空间）
    candidate_pool = 30
    vec_idx = np.argsort(sims)[::-1][:candidate_pool]
    candidate_set = set(int(i) for i in vec_idx)

    # 2. 关键词召回：提取药品名关键词，在知识库中匹配，命中的补入候选池
    keywords = _extract_query_keywords(question)
    exact_terms = _exact_query_terms(question)
    if keywords or exact_terms:
        kw_matches = []
        for i, chunk in enumerate(_chunks):
            text = chunk.get("text", "")
            source = chunk.get("source", "")
            combined = text + " " + source
            hits = sum(1 for kw in keywords if kw in combined)
            exact_hits = sum(1 for term in exact_terms if term in combined)
            source_exact_hits = sum(1 for term in exact_terms if term in source)
            if hits or exact_hits or source_exact_hits:
                kw_matches.append((source_exact_hits, exact_hits, hits, float(sims[i]), i))
        # Prefer exact source/title matches first. This keeps reviewed KB updates from
        # being crowded out by many broad hits for common drug names.
        kw_matches.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        new_kw = [i for *_scores, i in kw_matches if i not in candidate_set][:80]
        candidate_set.update(new_kw)

    # 3. 融合排序：向量相似度 + 关键词加分
    scored = []
    for i in candidate_set:
        vec_sim = float(sims[i])
        # 关键词加分：命中的关键词数 × 0.08，封顶0.20
        if keywords:
            text = _chunks[i].get("text", "")
            source = _chunks[i].get("source", "")
            combined = text + " " + source
            hits = sum(1 for kw in keywords if kw in combined)
            boost = min(hits * 0.08, 0.20)
        else:
            boost = 0.0
        if exact_terms:
            text = _chunks[i].get("text", "")
            source = _chunks[i].get("source", "")
            combined = text + " " + source
            exact_hits = sum(1 for term in exact_terms if term in combined)
            source_exact_hits = sum(1 for term in exact_terms if term in source)
            boost += min(exact_hits * 0.20, 0.55)
            boost += min(source_exact_hits * 0.18, 0.35)
        final_sim = vec_sim + boost
        scored.append((final_sim, i))
    scored.sort(key=lambda x: -x[0])

    # 4. 取 top_k，返回时 similarity 用融合分数
    return [{"text": _chunks[i]["text"], "source": _chunks[i]["source"],
             "similarity": final_sim} for final_sim, i in scored[:top_k]]


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
    count = 0
    for pattern, replacement in DESENSITIZE_RULES:
        new = pattern.sub(replacement, masked)
        if new != masked:
            count += 1
        masked = new
    return masked, count


# ===== 分诊（规则引擎）=====
EMERGENCY_KEYWORDS = [
    "胸痛", "胸闷", "呼吸困难", "气喘", "窒息",
    "意识模糊", "昏迷", "晕厥", "抽搐", "癫痫",
    "大量出血", "吐血", "便血", "咯血",
    "剧烈头痛", "爆炸性头痛", "雷击样头痛",
    "一侧肢体无力", "口角歪斜", "说话不清",
    "过敏休克", "喉头水肿", "面部肿胀",
    "高烧不退", "超高热", "40度", "41度",
    "药物中毒", "服药过量", "误服",
    "自杀", "自残", "心脏骤停",
]

SEVERE_KEYWORDS = [
    "持续高烧", "39度", "39.5", "反复发烧",
    "剧烈疼痛", "难以忍受", "止痛药无效",
    "持续呕吐", "脱水", "无法进食",
    "黄疸", "眼白发黄", "血尿", "尿血",
    "孕期出血", "见红", "胎动减少",
    "儿童精神萎靡", "拒食", "前囟凹陷",
    "症状加重", "持续数日不缓解", "一周未好转",
    "体重骤降", "消瘦", "夜间痛醒", "盗汗",
]


def rule_based_triage(text):
    for kw in EMERGENCY_KEYWORDS:
        if kw in text:
            return {"level": "急症", "action": "immediate_medical",
                    "recommend_drugs": False, "matched": kw}
    for kw in SEVERE_KEYWORDS:
        if kw in text:
            return {"level": "重症", "action": "see_doctor_soon",
                    "recommend_drugs": False, "matched": kw}
    return None


def llm_triage(text):
    from llm_pool import chat
    prompt = f"""你是医学分诊助手。评估用户症状严重程度。

用户描述：{text}

分诊标准：
1. 急症：危及生命→立即就医，不推药
2. 重症：需医生诊断→尽快就医，不推药
3. 中症：可先用OTC药+建议就医
4. 轻症：用OTC药+观察

输出JSON：
```json
{{"level":"轻症","action":"drugs_only","severity_score":3,
"symptoms":["鼻塞"],"possible_condition":"感冒",
"recommend_drugs":true,"needs_doctor":false,
"advice":"可用感冒药缓解","red_flags":["症状加重就医"]}}
```
只输出JSON。"""
    try:
        result = chat([{"role":"user","content":prompt}], temperature=0.15, max_tokens=500)
        m = re.search(r'```json\s*\n?(.*?)```', result, re.DOTALL)
        if m:
            return json.loads(m.group(1).strip())
        s, e = result.find('{'), result.rfind('}')
        if s != -1:
            return json.loads(result[s:e+1])
    except: pass
    return {"level": "中症", "recommend_drugs": False, "needs_doctor": True,
            "advice": "建议咨询医生"}


# ===== 禁用词检测 =====
FORBIDDEN_WORDS = [
    "根治","包治","百分百","100%有效","完全治愈","永不复发",
    "药到病除","祖传秘方","宫廷秘方","神奇疗效",
    "最好","最佳","第一","顶级","唯一","绝对",
    "无副作用","无毒副作用","没有任何副作用","安全无毒",
    "无效退款","假一赔十","保证治好",
    "确诊","诊断为","你得了","你患了","你的病是",
]

def check_forbidden(text):
    found = [w for w in FORBIDDEN_WORDS if w in text]
    return found

def auto_fix_answer(answer):
    fixed = answer
    fixed = fixed.replace("你得了","您可能是")
    fixed = fixed.replace("你患了","您可能是")
    fixed = fixed.replace("确诊为","可能是")
    fixed = fixed.replace("诊断为","可能是")
    for w in FORBIDDEN_WORDS:
        fixed = fixed.replace(w, "***")
    if not any(kw in fixed for kw in ["咨询","医生","药师","医嘱","处方"]):
        fixed += "\n\n具体用药请咨询药师或医生。"
    return fixed


def strip_opening_chitchat(answer, has_history=False):
    """多轮追问时去掉重复问候和自我介绍，让回答像连续对话。"""
    if not has_history or not answer:
        return answer

    text = str(answer).lstrip()
    preserved_prefix = ""
    prefix_match = re.match(r"^(我先提醒一下：.*?回答。\s*\n\n)(.*)$", text, flags=re.S)
    if prefix_match:
        preserved_prefix = prefix_match.group(1)
        text = prefix_match.group(2).lstrip()

    patterns = [
        r"^(您好|你好)[，,。！!\s]*",
        r"^我是[^。！？\n]{0,30}(药学|医药|药业)[^。！？\n]{0,16}助手[。！？!，,\s]*",
        r"^作为[^。！？\n]{0,30}(药学|医药|药业)[^。！？\n]{0,16}助手[，,。！？!\s]*",
        r"^这里是[^。！？\n]{0,30}(药学|医药|药业)[^。！？\n]{0,16}助手[，,。！？!\s]*",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_text = re.sub(pattern, "", text, count=1)
            if new_text != text:
                text = new_text.lstrip()
                changed = True

    return preserved_prefix + text


def normalize_user_question(text):
    """轻量纠错和口语归一化，避免错别字直接污染改写/检索。"""
    normalized = str(text or "").strip()
    changes = []
    for wrong, right in TYPO_CORRECTIONS.items():
        if wrong in normalized:
            normalized = normalized.replace(wrong, right)
            changes.append(f"{wrong}->{right}")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.replace("有没有", "有没有")
    return normalized, changes


def _symptom_topics(text):
    topics = []
    raw = str(text or "")
    for markers, mapped in SYMPTOM_TOPIC_PATTERNS:
        if any(marker in raw for marker in markers):
            for topic in mapped:
                if topic not in topics:
                    topics.append(topic)
    return topics


def _extract_mentioned_entities(text, max_terms=8):
    """从历史回答中抽取被列举的药品/眼药水，用于“这些/它们”类追问。"""
    raw = str(text or "")
    entities = []

    for alias, generic in load_drug_aliases().items():
        if alias in raw and alias not in entities:
            entities.append(alias)
        if generic in raw and generic not in entities:
            entities.append(generic)

    explicit_terms = ["人工泪液", "角膜营养液"]
    for term in explicit_terms:
        if term in raw and term not in entities:
            entities.append(term)

    form_pattern = "|".join(sorted((re.escape(w) for w in DRUG_FORM_WORDS), key=len, reverse=True))
    for match in re.finditer(rf"[\u4e00-\u9fa5A-Za-z0-9]{{2,18}}(?:{form_pattern})", raw):
        term = match.group(0)
        term = re.sub(r"^(如|例如|包括|使用|点用|可用|适当|具体|以下|几种)", "", term).strip()
        if not term or term in GENERIC_FOCUS_WORDS:
            continue
        if any(bad in term for bad in ["知识库", "温馨提示", "说明书", "医生", "药师"]):
            continue
        if term not in entities:
            entities.append(term)
        if len(entities) >= max_terms:
            break

    return entities[:max_terms]


# ===== 免责声明 =====
def get_disclaimer(level="drug_info"):
    templates = {
        "drug_info": "\n\n---\n【温馨提示】\n• 以上信息仅供参考，不能替代医生诊断\n• 具体用药请咨询执业药师或医生\n• 处方药需凭医生处方购买",
        "symptom": "\n\n---\n【温馨提示】\n• 我不是医生，以上建议不是诊断\n• 如症状持续或加重请及时就医\n• 不要仅凭以上建议自行用药",
        "see_doctor": "\n\n---\n【温馨提示】\n• 您的症状建议由医生面诊评估\n• 请前往最近医院就诊，不要自行用药\n• 如症状加重请及时就医",
        "out_of_scope": "\n\n---\n【温馨提示】\n• 我的药品知识库有限，无法回答所有问题\n• 请以药品说明书或医生建议为准",
    }
    return templates.get(level, templates["drug_info"])


# ===== 审计日志 =====
import hashlib
AUDIT_LOG = os.path.join(BASE_DIR, "logs", "audit_log.jsonl")
AUDIT_STATE = os.path.join(BASE_DIR, "logs", "rag_log_state.json")
KB_UPDATES_LOG = os.path.join(DATA_DIR, "kb_updates.jsonl")
REBUILD_STATUS = os.path.join(BASE_DIR, "logs", "rebuild_status.json")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

_admin_state_lock = threading.Lock()
_kb_updates_lock = threading.Lock()
_rebuild_lock = threading.Lock()
_rebuild_thread = None


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_text(value, limit=None):
    text = "" if value is None else str(value)
    return text[:limit] if limit else text


def _read_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _admin_allowed():
    if not ADMIN_TOKEN:
        return True
    token = (
        request.headers.get("X-Admin-Token", "")
        or request.args.get("token", "")
        or request.cookies.get("admin_token", "")
    )
    return token == ADMIN_TOKEN


def _admin_guard():
    if _admin_allowed():
        return None
    return jsonify({
        "success": False,
        "error": "admin token required",
        "token_required": True,
    }), 401


def audit_log(question, answer, level, issues, user_id="web", retrieved=None, steps=None):
    os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
    timestamp = _now()
    raw_id = f"{timestamp}|{user_id}|{question}|{time.time_ns()}|{uuid.uuid4().hex[:8]}"
    answer_text = _safe_text(answer, 4000)
    entry = {
        "id": "log_" + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16],
        "timestamp": timestamp,
        "user_hash": hashlib.md5(user_id.encode()).hexdigest()[:8],
        "question": question[:200],
        "answer": answer_text,
        "answer_preview": answer_text.replace("\n", " ")[:300],
        "triage_level": level,
        "issues": issues,
    }
    if retrieved is not None:
        entry["retrieved"] = retrieved
    if steps is not None:
        entry["steps"] = steps
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


ADMIN_STATUS_LABELS = {
    "answered": "已回答",
    "pending_kb": "待补知识库",
    "invalid": "无效输入",
    "clarify": "需要澄清",
    "review": "需复核",
    "handled": "已处理",
    "ignored": "已忽略",
}
VALID_ADMIN_STATUSES = set(ADMIN_STATUS_LABELS)


def _load_audit_state():
    data = _read_json_file(AUDIT_STATE, {"items": {}})
    if not isinstance(data, dict):
        data = {"items": {}}
    if "items" not in data or not isinstance(data["items"], dict):
        data["items"] = {}
    return data


def _save_audit_state(state):
    _write_json_file(AUDIT_STATE, state)


def _audit_entry_id(entry, line_no):
    if entry.get("id"):
        return str(entry["id"])
    raw = "|".join([
        str(line_no),
        str(entry.get("timestamp", "")),
        str(entry.get("user_hash", "")),
        str(entry.get("question", "")),
    ])
    return "log_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _derive_audit_status(entry):
    issues = [str(item) for item in (entry.get("issues") or [])]
    level = str(entry.get("triage_level", ""))
    if any("超纲" in item or "知识库未收录" in item or "out_of_scope" in item.lower() for item in issues):
        return "pending_kb"
    if any("无效输入" in item for item in issues):
        return "invalid"
    if level == "澄清":
        return "clarify"
    if any("违规" in item or "禁用" in item for item in issues):
        return "review"
    return "answered"


def _load_audit_entries():
    if not os.path.exists(AUDIT_LOG):
        return []
    entries = []
    with open(AUDIT_LOG, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            entry = dict(entry)
            entry["id"] = _audit_entry_id(entry, line_no)
            entry["line_no"] = line_no
            entries.append(entry)
    return entries


def _hydrate_audit_entry(entry, state):
    item_state = state.get("items", {}).get(entry["id"], {})
    derived_status = _derive_audit_status(entry)
    status = item_state.get("status") or entry.get("status") or derived_status
    if status not in VALID_ADMIN_STATUSES:
        status = derived_status
    answer = _safe_text(entry.get("answer") or "")
    hydrated = {
        "id": entry["id"],
        "timestamp": entry.get("timestamp", ""),
        "user_hash": entry.get("user_hash", ""),
        "question": entry.get("question", ""),
        "answer": answer,
        "answer_preview": entry.get("answer_preview") or answer.replace("\n", " ")[:300],
        "triage_level": entry.get("triage_level", ""),
        "issues": entry.get("issues") or [],
        "retrieved": entry.get("retrieved") or [],
        "steps": entry.get("steps") or [],
        "line_no": entry.get("line_no"),
        "status": status,
        "status_label": ADMIN_STATUS_LABELS.get(status, status),
        "derived_status": derived_status,
        "handled": bool(item_state.get("handled")) or status in ("handled", "ignored"),
        "note": item_state.get("note", ""),
        "updated_at": item_state.get("updated_at", ""),
    }
    return hydrated


def _query_audit_logs(status=None, keyword=None, limit=200):
    state = _load_audit_state()
    all_logs = [_hydrate_audit_entry(entry, state) for entry in _load_audit_entries()]
    counts = {"all": len(all_logs)}
    for log in all_logs:
        counts[log["status"]] = counts.get(log["status"], 0) + 1

    keyword = (keyword or "").strip().lower()
    filtered = []
    for log in reversed(all_logs):
        if status and status != "all" and log["status"] != status:
            continue
        if keyword:
            haystack = "\n".join([
                log.get("question", ""),
                log.get("answer", ""),
                log.get("note", ""),
                " ".join(str(x) for x in log.get("issues", [])),
            ]).lower()
            if keyword not in haystack:
                continue
        filtered.append(log)
        if len(filtered) >= limit:
            break
    return filtered, counts


def _get_audit_log(log_id):
    state = _load_audit_state()
    for entry in _load_audit_entries():
        if entry.get("id") == log_id:
            return _hydrate_audit_entry(entry, state)
    return None


def _load_kb_updates():
    if not os.path.exists(KB_UPDATES_LOG):
        return []
    items = []
    with open(KB_UPDATES_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
    return items


def _save_kb_updates(items):
    os.makedirs(os.path.dirname(KB_UPDATES_LOG), exist_ok=True)
    tmp_path = f"{KB_UPDATES_LOG}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_path, KB_UPDATES_LOG)


def _update_audit_state(log_id, status=None, note=None, handled=None):
    with _admin_state_lock:
        state = _load_audit_state()
        current = state["items"].get(log_id, {})
        if status:
            current["status"] = status
        if note is not None:
            current["note"] = _safe_text(note, 1000)
        if handled is not None:
            current["handled"] = bool(handled)
        current["updated_at"] = _now()
        state["items"][log_id] = current
        _save_audit_state(state)
        return current


def generate_kb_draft(log_item, chat_fn=None):
    """Generate a human-review draft for a possible KB update."""
    if not log_item:
        return {"can_draft": False, "reason": "日志不存在"}

    question = _safe_text(log_item.get("question"), 300).strip()
    if not question:
        return {"can_draft": False, "reason": "问题为空，无法生成草稿"}

    retrieved_text = "\n".join(
        f"- {item.get('source', '')} 相似度:{item.get('similarity', '')}"
        for item in (log_item.get("retrieved") or [])[:5]
    ) or "无"
    issues_text = "、".join(str(x) for x in (log_item.get("issues") or [])) or "无"
    answer_preview = _safe_text(log_item.get("answer") or log_item.get("answer_preview"), 900).strip() or "无"
    steps_text = "\n".join(str(x) for x in (log_item.get("steps") or [])[:8]) or "无"

    prompt = f"""你是药学知识库编辑助手。你的任务是根据测试日志生成一版“待人工审核”的知识库草稿，不是直接给用户的最终医疗建议。

【测试问题】
{question}

【原回答或回答摘要】
{answer_preview}

【问题标记】
{issues_text}

【检索命中】
{retrieved_text}

【处理步骤】
{steps_text}

请只输出 JSON，不要输出解释、Markdown 或代码块。JSON 格式：
{{
  "can_draft": true,
  "title": "标准问题，尽量短，适合作为知识库标题",
  "answer": "待审核标准答案，120-500字。必须谨慎、边界清楚，不要诊断，不要编造具体价格。涉及处方药或抗生素时要强调遵医嘱。",
  "source": "AI草稿，需人工核验；建议核验来源：...",
  "review_notes": "给审核人的一句话提醒"
}}

如果问题不适合作为知识库条目，例如寒暄、系统能力、纯价格、缺少具体药品或症状主体、明显虚构药品，请输出：
{{
  "can_draft": false,
  "reason": "不建议生成草稿的原因",
  "review_notes": "建议后台如何处理"
}}

要求：
- 不要使用星号、粗体、标题符号等 Markdown。
- 草稿必须写明需要人工核验来源，不要假装已经核验。
- 药品用法用量、禁忌、不良反应等必须提示以说明书、医生或药师审核为准。
- 若可以生成草稿，source 必须以“AI草稿，需人工核验；建议核验来源：”开头。"""

    try:
        if chat_fn is None:
            from llm_pool import chat
            chat_fn = chat
        raw = chat_fn([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=800)
        parsed = _json_from_llm(raw)
    except Exception as exc:
        return {
            "can_draft": False,
            "reason": f"AI草稿生成失败: {str(exc)[:160]}",
            "review_notes": "请稍后重试，或手动填写待审核知识条目。",
        }

    can_draft = bool(parsed.get("can_draft"))
    if not can_draft:
        return {
            "can_draft": False,
            "reason": _safe_text(parsed.get("reason") or "模型判断不适合生成知识条目", 500),
            "review_notes": _safe_text(parsed.get("review_notes") or "", 500),
        }

    title = _safe_text(parsed.get("title") or question, 300).strip()
    answer = _safe_text(parsed.get("answer"), 6000).strip()
    source = _safe_text(parsed.get("source"), 1000).strip()
    review_notes = _safe_text(parsed.get("review_notes"), 1000).strip()

    if not answer or len(answer) < 30:
        return {
            "can_draft": False,
            "reason": "AI草稿内容过短，未达到入库审核要求",
            "review_notes": review_notes or "建议手动填写，或补充更明确的问题后再生成。",
        }
    if not source.startswith("AI草稿，需人工核验"):
        source = "AI草稿，需人工核验；建议核验来源：" + (source or "药品说明书、国家药监局、医院/卫健委科普或专业指南")

    return {
        "can_draft": True,
        "title": title,
        "answer": answer,
        "source": source,
        "review_notes": review_notes,
    }


def _append_to_drug_knowledge(update_item):
    knowledge_path = os.path.join(DATA_DIR, "drug_knowledge.json")
    if not os.path.exists(knowledge_path):
        raise FileNotFoundError("data/drug_knowledge.json not found")
    with open(knowledge_path, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
    if not isinstance(knowledge, list):
        raise ValueError("drug_knowledge.json format must be a list")

    question = (update_item.get("title") or update_item.get("question") or "").strip()
    answer = (update_item.get("answer") or "").strip()
    if not question or not answer:
        raise ValueError("question/title and answer are required")

    if update_item.get("knowledge_id"):
        return update_item["knowledge_id"], False

    raw_id = f"{question}|{answer}|{time.time_ns()}"
    knowledge_id = "kb_" + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
    knowledge.append({
        "id": knowledge_id,
        "question": question,
        "answer": answer,
        "text": f"问题：{question}\n答案：{answer}",
        "source": update_item.get("source", ""),
        "created_from_log": update_item.get("log_id", ""),
    })
    tmp_path = f"{knowledge_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, knowledge_path)
    return knowledge_id, True


def _write_rebuild_status(data):
    data["updated_at"] = _now()
    _write_json_file(REBUILD_STATUS, data)


def _read_rebuild_status():
    return _read_json_file(REBUILD_STATUS, {
        "running": False,
        "status": "idle",
        "message": "未开始",
        "updated_at": "",
    })


def _reload_vector_files_if_ready():
    global _vectors, _chunks, _models_loaded
    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    vectors_path = os.path.join(DATA_DIR, "vectors.npy")
    if not (os.path.exists(chunks_path) and os.path.exists(vectors_path)):
        return
    if _embed_tok is None or _embed_mdl is None:
        _models_loaded = False
        return
    _vectors = np.load(vectors_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    _models_loaded = True


def _run_rebuild_job():
    global _rebuild_thread
    log_path = os.path.join(BASE_DIR, "logs", "embed_rebuild_admin.log")
    python_exe = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    cmd = [python_exe, "-u", os.path.join(BASE_DIR, "src", "02_chunk_embed.py")]
    _write_rebuild_status({
        "running": True,
        "status": "running",
        "message": "正在重建向量库",
        "log_path": log_path,
        "started_at": _now(),
    })
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n\n===== rebuild started at {_now()} =====\n")
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            code = proc.wait()
        if code == 0:
            _reload_vector_files_if_ready()
            _write_rebuild_status({
                "running": False,
                "status": "done",
                "message": "向量库重建完成；如服务未预加载模型，请重启服务后再测试新知识",
                "log_path": log_path,
                "finished_at": _now(),
            })
        else:
            _write_rebuild_status({
                "running": False,
                "status": "failed",
                "message": f"向量库重建失败，退出码 {code}",
                "log_path": log_path,
                "finished_at": _now(),
            })
    except Exception as exc:
        _write_rebuild_status({
            "running": False,
            "status": "failed",
            "message": str(exc),
            "log_path": log_path,
            "finished_at": _now(),
        })
    finally:
        with _rebuild_lock:
            _rebuild_thread = None


# ===== 问题类型识别 → 对应微调训练数据的4种回答风格 =====
def classify_drug_question(question):
    """
    判断药品问题类型，对应微调训练数据的4种风格：
    - dosage  (用法用量) → 简洁直接给剂量+简短提醒
    - adverse (不良反应) → 列举反应+就医提醒
    - contraindication (禁忌/孕妇/儿童) → 重点警告+遵医嘱
    - general (一般咨询) → 自然对话
    返回: (type_key, style_prompt)
    """
    q = question
    if any(k in q for k in ["怎么服用","怎么吃","如何服用","用法用量","用法","用量",
                            "吃多少","服用","剂量","服","口服","饭前","饭后","几次","一天"]):
        return "dosage", "【回答风格：用法用量】简洁直接给剂量，附简短提醒。格式：先给用法用量数值，末尾一句\"具体用药请遵医嘱\"。"
    if any(k in q for k in ["不良反应","副作用","反应","不适","有啥反应","有反应","危害","毒性"]):
        return "adverse", "【回答风格：不良反应】先列举可能的不良反应，末尾加\"如出现严重不良反应请立即停药并就医\"。"
    if any(k in q for k in ["禁忌","孕妇","儿童","小孩","幼儿","哺乳","怀孕","经期",
                            "禁用","慎用","不能用","能吃吗","可以用吗","过敏"]):
        return "contraindication", "【回答风格：禁忌人群】开头用⚠️提示重点警告，强调\"请务必在医生指导下使用，切勿自行用药\"。"
    return "general", "【回答风格：一般咨询】自然对话，基于知识库回答，末尾简短提醒咨询药师或医生。"


# ===== 会话理解：提取当前主题、识别追问、构建多查询 =====
FOCUS_STOP_PHRASES = [
    "请问", "帮我", "帮忙", "想问一下", "我想问", "麻烦", "一下",
    "怎么吃", "怎么服用", "如何服用", "用法用量", "用法", "用量",
    "有什么", "有啥", "哪些", "什么", "是不是", "可不可以",
    "能不能", "可以吗", "能吃吗", "能用吗", "吗", "呢", "呀", "啊",
    "小孩", "儿童", "孕妇", "老人", "哺乳", "过敏",
    "作用", "功效", "不良反应", "副作用", "禁忌", "注意事项",
    "饭前", "饭后", "空腹", "一天几次", "吃多少", "服多少", "吃还是",
    "会不会", "相似", "一样", "区别", "对比", "相比", "和", "跟", "与",
    "这个", "那个", "这种", "该药", "它", "上面", "刚才", "刚刚", "的",
    "最近", "因为", "是否", "是否因为", "长时间", "电脑看久", "看电脑",
    "这些", "那些", "这几个", "它们", "多少钱", "价格", "费用",
]

GENERIC_FOCUS_WORDS = {
    "药品", "药物", "医生", "药师", "症状", "建议", "信息", "知识库", "回答",
    "温馨提示", "免责声明", "用药问题", "可以帮您", "具体用药", "咨询",
    "小孩", "儿童", "孕妇", "老人", "吃还是", "饭前饭后",
    "电脑看久", "看电脑", "长时间", "最近", "这些", "那些", "这几个",
    "多少钱", "价格", "费用", "最推荐",
}

FOLLOWUP_MARKERS = [
    "那", "那么", "这个", "那个", "这种", "该药", "它", "上面", "刚才", "刚刚",
    "还有", "还会", "还要", "呢", "禁忌", "副作用", "不良反应", "饭前", "饭后",
    "空腹", "小孩", "儿童", "孕妇", "老人", "哺乳", "过敏", "能吃", "能用",
    "可以吃", "可以用", "会不会", "相似", "一样", "区别", "对比", "相比",
    "哪些", "这些", "那些", "这几个", "那几个", "它们", "价格", "多少钱",
    "用量",
]

COMPARISON_MARKERS = ["相似", "一样", "区别", "对比", "相比", "和", "跟", "与"]

TYPO_CORRECTIONS = {
    "这写": "这些",
    "那写": "那些",
    "有写": "有些",
    "眼药谁": "眼药水",
    "眼要水": "眼药水",
    "眼液水": "眼药水",
    "副做用": "副作用",
    "不量反应": "不良反应",
    "不梁反应": "不良反应",
    "用凉": "用量",
    "剂凉": "剂量",
    "多钱": "多少钱",
    "有没用量": "有没有用量",
}

REFERENCE_MARKERS = [
    "这些", "那些", "这几个", "那几个", "上述", "上面", "前面",
    "它们", "这几种", "这类", "这些药", "这些眼药水",
]

PRICE_INTENT_WORDS = ["多少钱", "价格", "费用", "售价", "贵不贵"]
DRUG_FORM_WORDS = [
    "缓释胶囊", "分散片", "崩解片", "咀嚼片", "肠溶片", "控释片",
    "胶囊", "颗粒", "口服液", "注射液", "糖浆", "乳膏", "软膏",
    "凝胶", "喷雾剂", "气雾剂", "滴眼液", "滴耳液", "滴鼻液",
    "眼液", "眼药水", "片", "丸", "散",
]

SYMPTOM_TOPIC_PATTERNS = [
    (["眼睛干涩", "眼干涩", "眼睛很涩", "眼睛涩", "眼干", "干眼", "干眼症"], ["眼睛干涩", "干眼症"]),
    (["看电脑", "电脑看久", "长时间盯着电脑"], ["眼睛干涩", "干眼症"]),
    (["喉咙痛", "嗓子痛", "咽喉痛"], ["喉咙痛"]),
    (["鼻塞", "流鼻涕", "感冒"], ["感冒"]),
]

DRUG_ALIAS_PATH = os.path.join(DATA_DIR, "drug_aliases.csv")

# 内置兜底。正式维护时优先改 data/drug_aliases.csv。
DEFAULT_DRUG_ALIASES = {
    "感康": "复方氨酚烷胺胶囊",
    "快克": "复方氨酚烷胺胶囊",
    "白加黑": "氨酚伪麻美芬片",
    "芬必得": "布洛芬缓释胶囊",
    "护彤": "小儿氨酚黄那敏颗粒",
    "达喜": "铝碳酸镁咀嚼片",
    "吗丁啉": "多潘立酮片",
    "开瑞坦": "氯雷他定片",
    "息斯敏": "氯雷他定片",
    "泰诺": "酚麻美敏片",
}

TONE_PROMPT = """【沟通语气】
1. 像药店执业药师一样沟通：先接住用户关心点，再给结论，不要只堆条款。
2. 语气温和、明确、有边界。可以说"我理解你主要担心..."，但不要过度寒暄。
3. 如果知识库证据不足，要坦诚说明"我不能硬编"，并告诉用户下一步可以补充什么信息。
4. 多轮追问时要自然衔接上一轮，直接回答当前追问，不要说"您好"、"我是药学咨询助手"、"作为药学助手"等开场白。
5. 回答控制在用户能读完的长度，优先给可执行提醒。"""


def _clean_history(history, limit=5):
    """前端传来的历史可能很长，这里只保留最近几轮的关键文本。"""
    if not isinstance(history, list):
        return []
    cleaned = []
    for h in history[-limit:]:
        if not isinstance(h, dict):
            continue
        q = str(h.get("question", "")).strip()
        a = str(h.get("answer", "")).strip()
        if q:
            cleaned.append({"question": q[:120], "answer": a[:220]})
    return cleaned


_CHOICE_NUMERALS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _choice_number(question):
    q = str(question or "").strip()
    q = q.replace("．", ".").replace("。", ".")
    digit_match = re.fullmatch(
        r"(?:选|选择|第)?\s*([1-9])\s*(?:个|项|条|号|种|\.|、|）|\))?\s*(?:吧|呢)?",
        q,
    )
    if digit_match:
        return int(digit_match.group(1))

    cn_match = re.fullmatch(
        r"(?:选|选择|第)?\s*([一二两三四五六七八九])\s*(?:个|项|条|号|种)?\s*(?:吧|呢)?",
        q,
    )
    if cn_match:
        return _CHOICE_NUMERALS.get(cn_match.group(1))
    return None


def _last_history_answer(history):
    if not isinstance(history, list):
        return ""
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        answer = str(item.get("answer") or "").strip()
        if answer:
            return answer
    return ""


def _extract_numbered_options(text, max_options=9):
    raw = str(text or "")
    if not raw.strip():
        return {}

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I)
    # Handle compact text like "1. A 2. B 3. C" by putting choices on separate lines.
    raw = re.sub(r"(?<![\d.])\s+(?=[1-9][\.\)）、．]\s+)", "\n", raw)

    pattern = re.compile(
        r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?([1-9])[\.\)）、．]\s*(.+?)(?=(?:\n\s*(?:[-*]\s*)?(?:\*\*)?[1-9][\.\)）、．]\s+)|\Z)",
        re.S,
    )
    options = {}
    for num, option in pattern.findall(raw):
        option = re.split(r"\n\s*\n|---|【温馨提示】|温馨提示", option, maxsplit=1)[0]
        option = re.sub(r"\*\*", "", option)
        option = re.sub(r"\s+", " ", option).strip()
        option = option.strip(" \t-—:：;；，,。")
        if not option:
            continue
        options.setdefault(int(num), option[:160])
        if len(options) >= max_options:
            break
    return options


def _detect_numeric_choice(question, history):
    """Detect replies like "2" and capture the selected previous option."""
    number = _choice_number(question)
    if not number:
        return None

    options = _extract_numbered_options(_last_history_answer(history))
    selected = options.get(number)
    if not selected:
        return None

    return {
        "number": number,
        "option": selected,
        "options": options,
    }


def _json_from_llm(text):
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _fallback_numeric_choice_completion(choice_info, context):
    selected = choice_info.get("option", "")
    subject = context.get("last_subject", "")
    needs_more_info_markers = [
        "药品完整名称", "规格", "厂家", "补充", "年龄", "持续时间",
        "想问的是", "用法用量", "不良反应", "禁忌", "能不能",
    ]
    if any(marker in selected for marker in needs_more_info_markers):
        return {
            "action": "clarify",
            "reply": f"可以。您选的是第{choice_info['number']}项：{selected}。请再补充具体药品名或您最想确认的一点，我再帮您查。",
        }

    rewritten = selected
    if subject and not _contains_subject(selected, subject):
        rewritten = f"关于{subject}，{selected}"
    return {"action": "rewrite", "question": rewritten}


def complete_numeric_choice(question, history, context, choice_info):
    """Let the rewrite model decide what a numeric option reply means."""
    options_text = "\n".join(
        f"{num}. {text}" for num, text in sorted(choice_info.get("options", {}).items())
    )
    ctx = _build_history_context_text(context)
    prompt = f"""你是多轮问句补全助手。用户这轮只回复了一个选项编号，请结合上一轮上下文和选项内容判断它的真实含义。

【对话上下文】
{ctx}

【上一轮编号选项】
{options_text}

【用户本轮输入】
{question}

【用户选择】
第{choice_info['number']}项：{choice_info['option']}

请只输出 JSON，不要输出解释。JSON 格式二选一：
1. 如果已经能补全成一个可检索、可回答的完整问题：
{{"action":"rewrite","question":"补全后的完整问题"}}
2. 如果用户只是选择了一个澄清方向，但仍缺少关键对象（例如具体药品名、症状细节、年龄、持续时间等）：
{{"action":"clarify","reply":"一句自然的追问，说明还需要用户补什么"}}

要求：
- 不要机械拼接选项文字，要理解用户真正想继续哪条路。
- 医药问题里，如果缺少具体药品名或症状细节，优先 action=clarify。
- 不要编造知识库没有的信息。"""

    try:
        from llm_pool import chat
        raw = chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=180)
        parsed = _json_from_llm(raw)
        action = parsed.get("action")
        if action == "rewrite" and parsed.get("question"):
            return {"action": "rewrite", "question": str(parsed["question"]).strip()}
        if action == "clarify" and parsed.get("reply"):
            return {"action": "clarify", "reply": str(parsed["reply"]).strip()}
    except Exception as e:
        print(f"[choice] 模型补全失败，使用兜底: {e}", flush=True)

    return _fallback_numeric_choice_completion(choice_info, context)


def load_drug_aliases():
    """加载商品名/俗称 -> 通用名映射。CSV优先，内置表兜底。"""
    global _drug_aliases
    if _drug_aliases is not None:
        return _drug_aliases

    aliases = dict(DEFAULT_DRUG_ALIASES)
    if os.path.exists(DRUG_ALIAS_PATH):
        try:
            with open(DRUG_ALIAS_PATH, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alias = (row.get("alias") or "").strip()
                    canonical = (row.get("canonical") or "").strip()
                    if alias and canonical:
                        aliases[alias] = canonical
        except Exception as e:
            print(f"[alias] 加载别名表失败，使用内置别名: {e}", flush=True)

    _drug_aliases = aliases
    return aliases


def _expand_aliases(text):
    if not text:
        return []
    found = []
    for alias, generic in load_drug_aliases().items():
        if alias in text and generic not in found:
            found.append(generic)
    return found


def _extract_focus_terms(text, max_terms=4):
    """从用户话语中抽取可能的药品名/症状主题，用于多轮追问补主语。"""
    if not text:
        return []
    text, _ = normalize_user_question(text)
    terms = []
    for topic in _symptom_topics(text):
        if topic not in terms:
            terms.append(topic)
        if len(terms) >= max_terms:
            return terms
    for entity in _extract_mentioned_entities(text, max_terms=max_terms):
        if entity not in terms:
            terms.append(entity)
        if len(terms) >= max_terms:
            return terms
    cleaned = re.sub(r"【.*?】|---|⚠️|[0-9]+[\.、]", " ", str(text))
    for phrase in FOCUS_STOP_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"[，。！？；：、,.!?;:\n\r\t（）()《》<>“”\"'·\-_/]+", " ", cleaned)
    chunks = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}", cleaned)
    for chunk in chunks:
        chunk = re.sub(r"(的|了|吧)$", "", chunk.strip())
        if not chunk or chunk in GENERIC_FOCUS_WORDS:
            continue
        if any(g in chunk for g in GENERIC_FOCUS_WORDS):
            continue
        # 过长句子通常不是药名，截取会破坏语义，直接跳过。
        if len(chunk) > 16 and not any(form in chunk for form in ["片", "胶囊", "颗粒", "糖浆", "口服液"]):
            continue
        if chunk not in terms:
            terms.append(chunk)
        if len(terms) >= max_terms:
            break
    return terms


def _dialog_context(history):
    history = _clean_history(history)
    subjects = []
    mentioned_entities = []
    for h in reversed(history):
        for entity in _extract_mentioned_entities(h.get("answer", ""), max_terms=6):
            if entity not in mentioned_entities:
                mentioned_entities.append(entity)
        combined_text = f"{h.get('question', '')}\n{h.get('answer', '')}"
        for term in _extract_focus_terms(combined_text, max_terms=5):
            if term not in subjects:
                subjects.append(term)
    return {
        "history": history,
        "last_subject": subjects[0] if subjects else "",
        "subjects": subjects[:5],
        "mentioned_entities": mentioned_entities[:8],
    }


def _contains_subject(question, subject):
    if not subject:
        return False
    if subject in question:
        return True
    subject_keys = _extract_query_keywords(subject)
    return any(k in question for k in subject_keys if len(k) >= 3)


def _looks_like_followup(question):
    q = question.strip()
    if len(q) <= 18:
        return True
    return any(marker in q for marker in FOLLOWUP_MARKERS)


def _refers_to_mentioned_entities(question):
    return any(marker in question for marker in REFERENCE_MARKERS)


def _context_entity_phrase(context, limit=4):
    entities = []
    for entity in context.get("mentioned_entities", []):
        if entity not in entities and entity not in GENERIC_FOCUS_WORDS:
            entities.append(entity)
        if len(entities) >= limit:
            break
    return "、".join(entities)


def _rule_rewrite_followup(question, context):
    """先用可解释规则修常见追问，减少LLM改写漂移。"""
    subject = context.get("last_subject", "")
    q = question.strip()
    if not q:
        return question, False

    entity_phrase = _context_entity_phrase(context)
    if entity_phrase and _refers_to_mentioned_entities(q):
        ref_pattern = r"(这些眼药水|这些药|这几种|这几个|那几个|这些|那些|它们|上述|上面|前面)"
        rewritten = re.sub(ref_pattern, entity_phrase, q, count=1)
        if rewritten == q:
            rewritten = f"{entity_phrase}{q}"
        return rewritten, rewritten != q

    if not subject:
        return question, False
    if _contains_subject(q, subject):
        return question, False

    pronoun_pattern = r"(这个药|这种药|该药|这个|那个|这种|它)"
    if re.search(pronoun_pattern, q):
        rewritten = re.sub(pronoun_pattern, subject, q, count=1)
        return rewritten, rewritten != q

    is_compare = any(marker in q for marker in COMPARISON_MARKERS)
    if is_compare:
        # "和感康相似吗" -> "阿莫西林胶囊和感康相似吗"
        return f"{subject}{q}", True

    if _looks_like_followup(q):
        q2 = re.sub(r"^(那|那么|还有|还|再问一下|另外)", "", q).strip()
        return f"{subject}{q2}", True

    return question, False


def _build_history_context_text(context):
    history = context.get("history", [])
    lines = []
    if context.get("last_subject"):
        lines.append(f"当前主要对象/主题：{context['last_subject']}")
    for i, h in enumerate(history[-4:], 1):
        answer = h.get("answer", "").replace("\n", " ")
        lines.append(f"第{i}轮 用户：{h.get('question','')[:80]}；回答要点：{answer[:120]}")
    return "\n".join(lines)


def build_knowledge_gap_reply(question, max_sim=0.0, context=None):
    """更有温度的超纲/知识不足回复：不硬编，但给下一步。"""
    context = context or {}
    latin_tokens = [
        t for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", question)
        if t.upper() not in {"AI", "API", "RAG", "OTC", "GLM", "HTTP"}
    ]
    terms = _extract_focus_terms(question, max_terms=2)
    subject = latin_tokens[0] if latin_tokens else ""
    if not subject:
        subject = context.get("last_subject", "") if _looks_like_followup(question) else ""
    if not subject:
        subject = terms[0] if terms else context.get("last_subject", "")
    if subject:
        intro = f"这个问题我现在没有足够可靠的知识库依据，不能硬编「{subject}」的答案。"
    else:
        intro = "这个问题我还没定位到足够可靠的药品知识，不能硬编答案。"

    return (
        f"{intro}\n\n"
        "您可以继续补充其中一种信息，我会再帮您查：\n"
        "1. 药品完整名称、规格或厂家\n"
        "2. 想问的是用法用量、不良反应、禁忌，还是能不能和其他药一起用\n"
        "3. 如果是在描述症状，请补充年龄、持续时间、是否发热/过敏/正在用药\n\n"
        "在信息不充分时，更稳妥的做法是查看说明书或咨询药师。"
        + get_disclaimer("out_of_scope")
    )


def build_price_gap_prefix(question):
    if any(word in question for word in PRICE_INTENT_WORDS):
        return "关于价格，知识库暂无价格信息；实际价格会因规格、地区和购买渠道变化，我不能编价格。\n\n"
    return ""


def strip_duplicate_price_notice(text):
    if not text:
        return text
    return re.sub(
        r"^\s*(?:关于(?:您)?(?:询问的)?)?\*?\*?价格\*?\*?[，,:：\s]*"
        r"\*?\*?知识库暂无价格信息\*?\*?[。；;]?\s*",
        "",
        str(text),
        count=1,
    ).lstrip()


def build_retrieval_queries(question, context=None):
    """为多意图/比较型追问生成少量检索变体，再合并结果。"""
    context = context or {}
    question, _ = normalize_user_question(question)
    queries = [question]
    subject = context.get("last_subject", "")
    terms = _extract_focus_terms(question, max_terms=3)
    mentioned_entities = context.get("mentioned_entities", [])

    if subject and not _contains_subject(question, subject) and _looks_like_followup(question):
        queries.append(f"{subject} {question}")
    if mentioned_entities and _refers_to_mentioned_entities(question):
        queries.append(f"{'、'.join(mentioned_entities[:4])} {question}")

    focus_terms = []
    if subject:
        focus_terms.append(subject)
    if _refers_to_mentioned_entities(question):
        for entity in mentioned_entities[:4]:
            if entity not in focus_terms:
                focus_terms.append(entity)
    for term in terms:
        if term not in focus_terms:
            focus_terms.append(term)
    for alias_target in _expand_aliases(question):
        if alias_target not in focus_terms:
            focus_terms.append(alias_target)

    intent_words = []
    if any(k in question for k in ["禁忌", "不能", "慎用", "孕妇", "儿童", "小孩", "过敏"]):
        intent_words.append("禁忌 注意事项")
    if any(k in question for k in ["不良反应", "副作用", "不舒服", "胃不舒服"]):
        intent_words.append("不良反应 副作用")
    if any(k in question for k in ["怎么吃", "怎么服用", "饭前", "饭后", "空腹", "用法", "用量"]):
        intent_words.append("用法用量 服用")
    if any(k in question for k in ["作用", "功效", "相似", "一样", "区别", "对比"]):
        intent_words.append("作用 药理")
    if any(k in question for k in ["眼药水", "滴眼液", "眼液", "人工泪液"]):
        intent_words.append("眼药水 滴眼液 人工泪液")
    if any(k in question for k in PRICE_INTENT_WORDS):
        intent_words.append("价格 多少钱")

    for term in focus_terms[:3]:
        for intent in intent_words[:3]:
            queries.append(f"{term} {intent}")
    for alias_target in _expand_aliases(question):
        queries.append(f"{alias_target} {question}")

    unique = []
    for q in queries:
        q = q.strip()
        if q and q not in unique:
            unique.append(q)
    return unique[:6]


def _retrieval_focus_terms(question, context=None):
    context = context or {}
    question, _ = normalize_user_question(question)
    terms = []
    subject = context.get("last_subject", "")
    if subject:
        terms.append(subject)
    if _refers_to_mentioned_entities(question):
        for entity in context.get("mentioned_entities", [])[:4]:
            if entity not in terms:
                terms.append(entity)
    for term in _extract_focus_terms(question, max_terms=4):
        if term not in terms:
            terms.append(term)
    for alias_target in _expand_aliases(question):
        if alias_target not in terms:
            terms.append(alias_target)
    return terms[:6]


def _combined_retrieved_text(retrieved):
    return " ".join(
        f"{r.get('source','')} {r.get('text','')}"
        for r in (retrieved or [])
    )


def _item_contains_term(item, term):
    combined = f"{item.get('source','')} {item.get('text','')}"
    return any(v in combined for v in _term_variants(term))


def _retrieval_intent_terms(question):
    terms = []
    intent_groups = [
        (["禁忌", "不能", "慎用", "孕妇", "儿童", "小孩", "过敏"], ["禁忌", "注意事项", "慎用"]),
        (["不良反应", "副作用", "不舒服", "胃不舒服"], ["不良反应", "副作用"]),
        (["怎么吃", "怎么服用", "饭前", "饭后", "空腹", "用法", "用量"], ["用法用量", "服用", "一次", "一日"]),
        (["作用", "功效", "相似", "一样", "区别", "对比"], ["作用", "药理", "功能主治"]),
        (["眼药水", "滴眼液", "眼液", "人工泪液"], ["眼药水", "滴眼液", "眼液", "人工泪液"]),
        (PRICE_INTENT_WORDS, ["价格", "多少钱", "费用"]),
    ]
    for markers, kws in intent_groups:
        if any(marker in question for marker in markers):
            for kw in kws:
                if kw not in terms:
                    terms.append(kw)
    return terms


def _rerank_retrieval_results(results, question, context=None):
    context = context or {}
    focus_terms = _retrieval_focus_terms(question, context)
    intent_terms = _retrieval_intent_terms(question)
    reranked = []
    for idx, item in enumerate(results):
        combined = f"{item.get('source', '')} {item.get('text', '')}"
        score = float(item.get("similarity", 0))
        focus_hits = sum(1 for term in focus_terms if _item_contains_term(item, term))
        intent_hits = sum(1 for kw in intent_terms if kw in combined)
        score += min(focus_hits * 0.16, 0.42)
        score += min(intent_hits * 0.06, 0.24)
        if context.get("last_subject") and _item_contains_term(item, context["last_subject"]):
            score += 0.08
        item = dict(item)
        item["similarity"] = score
        item["rerank_hits"] = {"focus": focus_hits, "intent": intent_hits}
        reranked.append((score, -idx, item))
    reranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in reranked]


def retrieve_with_context(question, context=None, top_k=TOP_K):
    """多查询检索合并，保留每条知识的最高得分。"""
    context = context or {}
    merged = {}
    for q in build_retrieval_queries(question, context):
        for r in retrieve(q, top_k=top_k):
            key = (r.get("source", ""), r.get("text", "")[:120])
            old = merged.get(key)
            if old is None or r.get("similarity", 0) > old.get("similarity", 0):
                item = dict(r)
                item["query"] = q
                merged[key] = item
    results = sorted(merged.values(), key=lambda x: x.get("similarity", 0), reverse=True)
    results = _rerank_retrieval_results(results, question, context)

    selected = []
    selected_keys = set()
    for term in _retrieval_focus_terms(question, context):
        match = next((r for r in results if _item_contains_term(r, term)), None)
        if match:
            key = (match.get("source", ""), match.get("text", "")[:120])
            if key not in selected_keys:
                selected.append(match)
                selected_keys.add(key)
        if len(selected) >= top_k:
            break

    for r in results:
        key = (r.get("source", ""), r.get("text", "")[:120])
        if key not in selected_keys:
            selected.append(r)
            selected_keys.add(key)
        if len(selected) >= top_k:
            break
    selected.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return selected[:top_k]


def evidence_supports_question(question, retrieved, context=None):
    """相似度之外的硬校验：检索证据需要覆盖用户提到的明确药品/别名。"""
    if not retrieved:
        return False, "没有检索结果"

    combined = _combined_retrieved_text(retrieved).lower()

    latin_tokens = [
        t for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", question)
        if t.upper() not in {"AI", "API", "RAG", "OTC", "GLM", "HTTP"}
    ]
    for token in latin_tokens:
        if token.lower() not in combined:
            return False, f"检索证据未覆盖英文/编码药名: {token}"

    drug_intent = any(k in question for k in [
        "怎么吃", "怎么服用", "用法", "用量", "副作用", "不良反应",
        "禁忌", "作用", "功效", "能吃", "能用", "相似", "区别", "一起吃",
    ])
    if not drug_intent:
        return True, ""

    terms = _retrieval_focus_terms(question, context)
    if not terms:
        return True, ""

    supported = []
    for term in terms:
        if any(v.lower() in combined for v in _term_variants(term)):
            supported.append(term)

    if not supported:
        return False, f"检索证据未覆盖问题主体: {', '.join(terms[:3])}"
    return True, ""


# ===== Query Rewrite：把追问改写成完整问题，解决多轮指代 =====
def rewrite_query(question, history):
    """
    多轮对话中，用户常省略主语（如"那可以用什么药"→指代牙龈疼的药）。
    有对话历史时一律交给大模型判断：若问题本身完整则原样输出，否则补全省略的主语。
    history: [{"question":..,"answer":..}, ...] 最近几轮
    返回: (rewritten_question, is_rewritten)
    """
    question, _ = normalize_user_question(question)
    context = _dialog_context(history)
    rule_rewritten, did_rule = _rule_rewrite_followup(question, context)
    if did_rule:
        return rule_rewritten, True

    history = context.get("history", [])
    if not history:
        return question, False

    # 构建上下文（最近3轮）
    ctx = _build_history_context_text(context)

    prompt = f"""你是问句改写助手。当前问题是多轮对话中的追问，可能省略了主语/指代词。
请结合上下文判断：如果当前问题本身已完整独立（不依赖上文也能理解），原样输出；如果省略了主语/宾语，补全后输出。

【对话上下文】
{ctx}

【当前问题】
{question}

【输出要求】
1. 只输出改写后的问题，不要解释，不要引号
2. 完整问题原样输出，不要画蛇添足
3. 保留用户口语风格，只补全省略的主语/宾语
4. 示例：
   - 上文问"依托度酸片怎么服用"，当前问"有什么禁忌吗"→输出"依托度酸片有什么禁忌吗"
   - 上文问"便秘怎么办"，当前问"吃香蕉有用吗"→输出"便秘吃香蕉有用吗"
   - 上文问"布洛芬不良反应"，当前问"依托度酸片呢"→输出"依托度酸片有什么不良反应"
   - 上文问"阿莫西林胶囊怎么吃"，当前问"和感康的作用相似吗"→输出"阿莫西林胶囊和感康的作用相似吗"
   - 当前问题本身完整如"阿莫西林怎么吃"→原样输出"阿莫西林怎么吃"
"""

    try:
        from llm_pool import chat
        rewritten = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=120,
        )
        rewritten = rewritten.strip().strip('"').strip("「」").strip("【】")
        if rewritten and rewritten != question:
            return rewritten, True
    except Exception as e:
        print(f"[rewrite] 失败,用原问题: {e}", flush=True)
    return question, False


# ===== 核心问答链路 =====
def process_question(question, history=None):
    """完整处理链路：脱敏→分诊→检索→生成→审核→安全层
    history: [{"question":..,"answer":..}, ...] 多轮上下文
    """
    question, normalize_notes = normalize_user_question(question)
    choice_info = _detect_numeric_choice(question, history)
    context = _dialog_context(history)
    result = {
        "question": question,
        "answer": "",
        "retrieved": [],
        "triage": None,
        "steps": [],
        "issues": [],
    }

    if choice_info:
        result["steps"].append(f"选项选择：{choice_info['number']} -> {choice_info['option'][:30]}")
        choice_result = complete_numeric_choice(question, history, context, choice_info)
        if choice_result.get("action") == "clarify":
            answer = choice_result.get("reply") or "可以，请再补充一点具体信息，我再帮您查。"
            result["answer"] = answer
            result["triage"] = {"level": "澄清", "recommend_drugs": False, "needs_doctor": False}
            result["steps"].append("选项补全：需要继续澄清")
            result["question"] = f"选择第{choice_info['number']}项：{choice_info['option']}"
            audit_log(result["question"], answer, "澄清", [], retrieved=result["retrieved"], steps=result["steps"])
            return result
        rewritten = choice_result.get("question")
        if rewritten and rewritten != question:
            question = rewritten
            result["question"] = rewritten
            result["steps"].append(f"选项补全: {rewritten[:30]}")

    # 1. 脱敏
    masked, mask_count = desensitize(question)
    if mask_count > 0:
        result["steps"].append(f"已脱敏{mask_count}处隐私信息")
    if normalize_notes:
        result["steps"].append(f"输入纠错：{'；'.join(normalize_notes)}")

    # 2. 分诊——规则引擎优先，简单问题不调LLM
    triage = rule_based_triage(masked)
    if triage is None:
        # 判断是否是简单药品查询（不含症状描述词）
        symptom_words = ["头痛","发烧","咳嗽","感冒","疼痛","拉肚子","腹泻","失眠",
                         "恶心","呕吐","过敏","皮疹","痒","头晕","胸闷","心慌",
                         "鼻塞","流鼻"," sore","痛","烧","咳","吐","麻","晕"]
        is_symptom = any(kw in masked for kw in symptom_words)

        if is_symptom or len(masked) > 30:
            # 有症状描述才调LLM分诊
            try:
                triage = llm_triage(masked)
            except Exception as e:
                triage = {"level": "轻症", "recommend_drugs": True, "needs_doctor": False}
        else:
            # 简单药品查询直接轻症，跳过LLM分诊（省5-10秒）
            triage = {"level": "轻症", "action": "drugs_only",
                      "recommend_drugs": True, "needs_doctor": False}

    result["triage"] = triage
    level = triage.get("level", "轻症")
    result["steps"].append(f"分诊：{level}")

    # 急症/重症：直接引导就医
    if level in ("急症", "重症") or triage.get("action") in ("immediate_medical", "see_doctor_soon"):
        if level == "急症":
            answer = f"您好，根据您描述的情况，建议您尽快前往医院就诊，由专业医生为您评估。\n\n请不要自行用药，前往最近医院就诊即可。"
            answer += get_disclaimer("see_doctor")
        else:
            advice = triage.get("advice", "您的症状建议尽快就医，不建议自行用药。")
            answer = f"您好，根据您描述的情况，建议您尽快前往医院就诊。\n\n{advice}\n\n【为什么不建议自行用药】\n• 需要专业医生面诊\n• 自行用药可能掩盖病情"
            red_flags = triage.get("red_flags", [])
            if red_flags:
                answer += "\n\n【需要留意的情况】\n"
                for rf in red_flags:
                    answer += f"• {rf}\n"
            answer += get_disclaimer("see_doctor")
        result["answer"] = answer
        result["steps"].append("已拦截：引导就医")
        audit_log(masked, answer, level, [], retrieved=result["retrieved"], steps=result["steps"])
        return result

    # 2.4 无效输入拦截：单字符/纯数字/无意义输入
    cleaned = re.sub(r'[\s\W_]+', '', masked)
    if not choice_info and (len(cleaned) < 2 or cleaned.isdigit() or len(masked.strip()) < 2):
        answer = "您的问题似乎不太完整，能再详细描述一下吗？比如您的症状或想了解的药品名称。"
        result["answer"] = answer
        result["steps"].append("无效输入：过短/纯数字")
        result["triage"] = {"level": "闲聊", "recommend_drugs": False, "needs_doctor": False}
        audit_log(masked, answer, "闲聊", ["无效输入"], retrieved=result["retrieved"], steps=result["steps"])
        return result

    # 2.5 闲聊/非用药问题检测（在检索前拦截）
    #   注意：有多轮历史时跳过闲聊兜底——短问题很可能是追问而非闲聊
    chitchat_keywords = [
        "你好", "您好", "hi", "hello", "谢谢", "感谢", "再见", "拜拜",
        "你是谁", "你叫什么", "你能做什么", "帮助",
        "在吗", "在不在", "有人吗",
        "对话几轮", "能对话几轮", "记得上文吗", "能记住吗", "能追问吗",
    ]
    # 仅当精确命中闲聊关键词（且无多轮上下文）才判为闲聊
    lower_q = masked.lower().strip()
    is_chitchat = (
        lower_q in chitchat_keywords
        or (not history and len(masked) < 20 and any(kw in lower_q for kw in chitchat_keywords))
    )

    if is_chitchat:
        # 闲聊直接本地回复，不调LLM（省5-10秒）
        chitchat_replies = {
            "你好": "您好！我是药业智能咨询助手，有什么用药问题可以帮您吗？",
            "您好": "您好！我是药业智能咨询助手，有什么用药问题可以帮您吗？",
            "hi": "您好！有什么用药问题可以帮您吗？",
            "hello": "您好！有什么用药问题可以帮您吗？",
            "谢谢": "不客气！如果还有其他用药问题随时问我。祝您健康！",
            "感谢": "不客气！如果还有其他用药问题随时问我。祝您健康！",
            "再见": "再见！祝您健康，有用药问题随时来找我。",
            "拜拜": "再见！祝您健康。",
            "你是谁": "我是药业智能咨询助手，可以为您解答药品用法用量、不良反应、禁忌等问题，也可以描述症状我会给建议。",
            "你叫什么": "我是药业智能咨询助手，可以为您解答用药问题。",
            "你能做什么": "我可以为您解答药品用法用量、不良反应、禁忌等问题，也可以描述症状我会给建议。请问有什么可以帮您的？",
            "在吗": "在的！有什么用药问题可以帮您吗？",
            "在不在": "在的！有什么用药问题可以帮您吗？",
            "有人吗": "在的！我是药业智能咨询助手，有什么可以帮您的？",
            "对话几轮": "可以连续追问。我会参考最近5轮对话，尽量理解“它呢”“有什么禁忌”“能和这个一起吃吗”这类上下文问题。",
            "能对话几轮": "可以连续追问。我会参考最近5轮对话，尽量理解“它呢”“有什么禁忌”“能和这个一起吃吗”这类上下文问题。",
            "记得上文吗": "我会参考最近5轮对话来理解追问，但重要药名、年龄、正在服用的药，建议您在关键问题里再说一遍，会更稳妥。",
            "能记住吗": "我会参考最近5轮对话来理解追问，但重要药名、年龄、正在服用的药，建议您在关键问题里再说一遍，会更稳妥。",
            "能追问吗": "可以追问。比如先问“布洛芬有什么不良反应”，再问“饭后吃会不会好一点”，我会尽量带着上文理解。",
        }
        reply = chitchat_replies.get(lower_q)
        if not reply:
            # 不在预设里的闲聊，用简单模板
            reply = "您好！我是药业智能咨询助手，可以为您解答药品用法用量、不良反应等问题。请问有什么可以帮您的？"
        result["answer"] = reply
        result["steps"].append("闲聊处理：本地秒回")
        result["triage"] = {"level": "闲聊", "recommend_drugs": False, "needs_doctor": False}
        audit_log(masked, reply, "闲聊", [], retrieved=result["retrieved"], steps=result["steps"])
        return result

    # 3. Query Rewrite：多轮追问改写成完整问题（解决"那用什么药"指代不明）
    retrieve_question = masked
    if history and not choice_info:
        rewritten, is_rw = rewrite_query(masked, history)
        if is_rw:
            result["steps"].append(f"追问改写: {rewritten[:30]}")
            retrieve_question = rewritten
            masked = rewritten  # 后续生成也用改写后的完整问题
            result["question"] = rewritten

    # 4. 检索
    try:
        retrieved = retrieve_with_context(retrieve_question, context=context, top_k=TOP_K)
    except Exception as e:
        retrieved = []
    result["retrieved"] = [{"source": r["source"], "similarity": round(r["similarity"], 4)} for r in retrieved[:3]]
    max_sim = max((r["similarity"] for r in retrieved), default=0)
    result["steps"].append(f"检索到{len(retrieved)}条相关知识（最高相似度{max_sim:.2f}）")
    evidence_ok, evidence_reason = evidence_supports_question(masked, retrieved, context)
    if not evidence_ok:
        result["steps"].append(f"证据校验未通过：{evidence_reason}")

    # 5. 超纲检测（阈值从0.60提到0.65，减少检索到不相关内容导致跑偏）
    if max_sim < 0.65 or not evidence_ok:
        answer = build_knowledge_gap_reply(masked, max_sim, context)
        result["answer"] = answer
        result["steps"].append("超纲兜底：知识库未收录")
        audit_log(masked, answer, level, ["超纲"], retrieved=result["retrieved"], steps=result["steps"])
        return result

    # 6. 生成答案：默认用大模型(llm_pool)，本地1.5B仅作离线兜底
    #    大模型理解力远超1.5B，能避免"牙龈疼给坐骨神经痛方案"这种跑偏
    knowledge_text = "\n\n".join(
        f"【知识{i+1}】{r['text'][:250]}" for i, r in enumerate(retrieved[:5])
    )
    partial_prefix = ""
    if max_sim < 0.72:
        partial_prefix = "我先提醒一下：检索到的信息可能不完整，下面只能基于现有知识库内容回答。\n\n"
    price_prefix = build_price_gap_prefix(masked)
    partial_prefix += price_prefix

    # 多轮上下文提示
    history_hint = ""
    if history:
        history_hint = f"\n\n【多轮对话上下文】\n{_build_history_context_text(context)}\n当前问题可能是追问，请自然衔接，不要重复已说过的内容。不要重新问候或自我介绍，不要说“您好，我是药学咨询助手”。"

    q_type, style_prompt = classify_drug_question(masked)
    price_generation_hint = ""
    if any(word in masked for word in PRICE_INTENT_WORDS):
        price_generation_hint = "\n【价格处理】回答开头已由系统说明知识库暂无价格信息，后文不要重复这句话，不要编价格，只补充知识库支持的用法/注意事项。"

    gen_prompt = f"""你是专业药学咨询助手。请基于知识库内容回答用户问题，严格遵循以下要求：

【用户问题】
{masked}

【知识库内容】（只能基于这些回答，不要编造知识库里没有的药品/疗法）
{knowledge_text}
{history_hint}

【回答要求】
1. 只推荐知识库中提到的药品，知识库没有的不要编造
2. 用法用量/不良反应/禁忌必须来自知识库，不要凭空生成
3. 如果知识库内容与问题不匹配（如问牙龈疼但知识库是坐骨神经痛），明确说明"知识库暂无相关药品信息"，不要强行套用
4. 回答自然、简洁，紧扣用户问题，不要套用与问题无关的治疗方案
5. 不要做诊断，用"可能是""建议咨询医生"等措辞
6. 如果用户问价格/多少钱，而知识库没有价格信息，要明确说"知识库暂无价格信息"，不要编造价格
7. 用户问题如有错别字，应按纠正后的语义回答，不要纠结错字本身

{price_generation_hint}
{style_prompt}

{TONE_PROMPT}"""

    try:
        from llm_pool import chat
        raw_answer = chat(
            [
                {"role": "system", "content": "你是专业、温和、边界清晰的药学咨询助手。严格基于知识库回答，不编造，不套用无关治疗方案。"},
                {"role": "user", "content": gen_prompt},
            ],
            temperature=0.3, max_tokens=700,
        )
        if not raw_answer:
            raw_answer = "抱歉，未能生成回答，请重新提问。"
        result["steps"].append("大模型生成完成")
    except Exception as e:
        # 大模型全部失败时，降级到本地1.5B
        result["steps"].append(f"大模型异常，降级本地1.5B: {str(e)[:40]}")
        try:
            raw_answer = local_generate(masked, knowledge_text, max_new_tokens=400)
            if not raw_answer:
                raw_answer = "抱歉，服务暂时不可用，请稍后重试。"
        except Exception as e2:
            raw_answer = "抱歉，服务暂时不可用，请稍后重试。"

    # 6. 禁用词检测+自动修复
    raw_answer = strip_opening_chitchat(raw_answer, has_history=bool(history))
    if price_prefix:
        raw_answer = strip_duplicate_price_notice(raw_answer)
    forbidden = check_forbidden(raw_answer)
    if forbidden:
        result["issues"].append(f"发现违规词: {forbidden}")
        raw_answer = auto_fix_answer(raw_answer)
        result["steps"].append("已自动修复违规内容")

    # 7. 免责声明
    if triage.get("needs_doctor") and triage.get("recommend_drugs"):
        disclaimer = get_disclaimer("symptom")
    else:
        disclaimer = get_disclaimer("drug_info")

    result["answer"] = partial_prefix + raw_answer + disclaimer
    audit_log(masked, result["answer"], level, result["issues"], retrieved=result["retrieved"], steps=result["steps"])
    return result


# ===== API路由 =====
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/admin/rag-logs")
def admin_rag_logs_page():
    return send_from_directory(BASE_DIR, "admin_rag_logs.html")

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    history = data.get("history", []) or []  # 多轮对话历史
    if not question:
        return jsonify({"success": False, "error": "问题不能为空"}), 400

    try:
        result = process_question(question, history=history)
        # 打印问题和回答到日志
        print(f"\n{'='*50}", flush=True)
        print(f"[提问] {question}", flush=True)
        if history:
            print(f"[多轮] 已带入{len(history)}轮上下文", flush=True)
        print(f"[分诊] {result.get('triage',{}).get('level','未知')}", flush=True)
        print(f"[步骤] {' → '.join(result.get('steps',[]))}", flush=True)
        answer_preview = result.get('answer','')[:200].replace('\n',' ')
        print(f"[回答] {answer_preview}", flush=True)
        print(f"{'='*50}", flush=True)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    from llm_pool import get_status
    return jsonify({
        "vector_db": _models_loaded,
        "chunks": len(_chunks) if _models_loaded else 0,
        "apis": get_status(),
    })


@app.route("/api/admin/status", methods=["GET"])
def api_admin_status():
    denied = _admin_guard()
    if denied:
        return denied
    logs, counts = _query_audit_logs(limit=1)
    return jsonify({
        "success": True,
        "protected": bool(ADMIN_TOKEN),
        "token_required": bool(ADMIN_TOKEN),
        "counts": counts,
        "latest": logs[0] if logs else None,
        "rebuild": _read_rebuild_status(),
    })


@app.route("/api/admin/rag-logs", methods=["GET"])
def api_admin_rag_logs():
    denied = _admin_guard()
    if denied:
        return denied
    status = request.args.get("status", "all")
    keyword = request.args.get("q", "")
    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, 2000))
    logs, counts = _query_audit_logs(status=status, keyword=keyword, limit=limit)
    return jsonify({
        "success": True,
        "logs": logs,
        "counts": counts,
        "status_labels": ADMIN_STATUS_LABELS,
    })


@app.route("/api/admin/rag-logs/<log_id>/mark", methods=["POST"])
def api_admin_mark_log(log_id):
    denied = _admin_guard()
    if denied:
        return denied
    data = request.get_json(force=True) or {}
    status = (data.get("status") or "").strip()
    if status not in VALID_ADMIN_STATUSES:
        return jsonify({"success": False, "error": "invalid status"}), 400
    note = data.get("note")
    handled = data.get("handled")
    if handled is None:
        handled = status in ("handled", "ignored")
    state = _update_audit_state(log_id, status=status, note=note, handled=handled)
    return jsonify({"success": True, "state": state})


@app.route("/api/admin/rag-logs/<log_id>/draft-kb", methods=["POST"])
def api_admin_draft_kb_from_log(log_id):
    denied = _admin_guard()
    if denied:
        return denied
    log_item = _get_audit_log(log_id)
    if not log_item:
        return jsonify({"success": False, "error": "log not found"}), 404
    draft = generate_kb_draft(log_item)
    return jsonify({"success": True, "draft": draft})


@app.route("/api/admin/rag-logs/export.csv", methods=["GET"])
def api_admin_export_logs():
    denied = _admin_guard()
    if denied:
        return denied
    status = request.args.get("status", "all")
    keyword = request.args.get("q", "")
    logs, _counts = _query_audit_logs(status=status, keyword=keyword, limit=10000)
    output = io.StringIO()
    fields = ["timestamp", "status_label", "triage_level", "question", "answer_preview", "issues", "note"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for log in logs:
        writer.writerow({
            "timestamp": log.get("timestamp", ""),
            "status_label": log.get("status_label", ""),
            "triage_level": log.get("triage_level", ""),
            "question": log.get("question", ""),
            "answer_preview": log.get("answer_preview", ""),
            "issues": " | ".join(str(x) for x in log.get("issues", [])),
            "note": log.get("note", ""),
        })
    csv_text = "\ufeff" + output.getvalue()
    return Response(
        csv_text,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=rag_logs.csv"},
    )


@app.route("/api/admin/kb-updates", methods=["GET"])
def api_admin_kb_updates():
    denied = _admin_guard()
    if denied:
        return denied
    status = request.args.get("status", "all")
    items = list(reversed(_load_kb_updates()))
    counts = {"all": len(items)}
    for item in items:
        counts[item.get("status", "draft")] = counts.get(item.get("status", "draft"), 0) + 1
    if status and status != "all":
        items = [item for item in items if item.get("status", "draft") == status]
    return jsonify({"success": True, "items": items, "counts": counts})


@app.route("/api/admin/kb-updates", methods=["POST"])
def api_admin_create_kb_update():
    denied = _admin_guard()
    if denied:
        return denied
    data = request.get_json(force=True) or {}
    question = _safe_text(data.get("question"), 300).strip()
    title = _safe_text(data.get("title") or question, 300).strip()
    answer = _safe_text(data.get("answer"), 6000).strip()
    source = _safe_text(data.get("source"), 1000).strip()
    log_id = _safe_text(data.get("log_id"), 80).strip()
    if not title or not answer:
        return jsonify({"success": False, "error": "title and answer are required"}), 400
    item = {
        "id": "kbupd_" + uuid.uuid4().hex[:16],
        "created_at": _now(),
        "updated_at": _now(),
        "log_id": log_id,
        "question": question,
        "title": title,
        "answer": answer,
        "source": source,
        "note": _safe_text(data.get("note"), 1000).strip(),
        "status": "draft",
    }
    with _kb_updates_lock:
        items = _load_kb_updates()
        items.append(item)
        _save_kb_updates(items)
    if log_id:
        _update_audit_state(log_id, status="pending_kb", note="已创建待审核知识条目", handled=False)
    return jsonify({"success": True, "item": item})


@app.route("/api/admin/kb-updates/<update_id>/mark", methods=["POST"])
def api_admin_mark_kb_update(update_id):
    denied = _admin_guard()
    if denied:
        return denied
    data = request.get_json(force=True) or {}
    status = (data.get("status") or "").strip()
    if status not in {"draft", "approved", "applied", "ignored"}:
        return jsonify({"success": False, "error": "invalid status"}), 400
    with _kb_updates_lock:
        items = _load_kb_updates()
        target = None
        for item in items:
            if item.get("id") == update_id:
                target = item
                break
        if not target:
            return jsonify({"success": False, "error": "not found"}), 404
        target["status"] = status
        if data.get("note") is not None:
            target["note"] = _safe_text(data.get("note"), 1000)
        target["updated_at"] = _now()
        _save_kb_updates(items)
    return jsonify({"success": True, "item": target})


@app.route("/api/admin/kb-updates/<update_id>/apply", methods=["POST"])
def api_admin_apply_kb_update(update_id):
    denied = _admin_guard()
    if denied:
        return denied
    with _kb_updates_lock:
        items = _load_kb_updates()
        target = None
        for item in items:
            if item.get("id") == update_id:
                target = item
                break
        if not target:
            return jsonify({"success": False, "error": "not found"}), 404
        try:
            knowledge_id, appended = _append_to_drug_knowledge(target)
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500
        target["status"] = "applied"
        target["knowledge_id"] = knowledge_id
        target["applied_at"] = _now()
        target["updated_at"] = _now()
        _save_kb_updates(items)
    if target.get("log_id"):
        _update_audit_state(target["log_id"], status="handled", note=f"已入库: {knowledge_id}", handled=True)
    return jsonify({"success": True, "item": target, "appended": appended})


@app.route("/api/admin/kb-updates/export.jsonl", methods=["GET"])
def api_admin_export_kb_updates():
    denied = _admin_guard()
    if denied:
        return denied
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in _load_kb_updates())
    return Response(
        text,
        mimetype="application/x-jsonlines; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=kb_updates.jsonl"},
    )


@app.route("/api/admin/rebuild-index", methods=["POST"])
def api_admin_rebuild_index():
    denied = _admin_guard()
    if denied:
        return denied
    global _rebuild_thread
    with _rebuild_lock:
        if _rebuild_thread and _rebuild_thread.is_alive():
            return jsonify({"success": True, "rebuild": _read_rebuild_status(), "already_running": True})
        _rebuild_thread = threading.Thread(target=_run_rebuild_job, daemon=True)
        _rebuild_thread.start()
    return jsonify({"success": True, "rebuild": _read_rebuild_status(), "started": True})


@app.route("/api/admin/rebuild-status", methods=["GET"])
def api_admin_rebuild_status():
    denied = _admin_guard()
    if denied:
        return denied
    return jsonify({"success": True, "rebuild": _read_rebuild_status()})


# ===== 流式问答接口（SSE）=====
@app.route("/api/ask_stream", methods=["POST"])
def api_ask_stream():
    """流式问答：先推送元信息(分诊/检索), 再逐token推送生成内容。
    前端用 fetch + ReadableStream 接收。SSE 格式: data: <json>\n\n
    """
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    history = data.get("history", []) or []
    question, normalize_notes = normalize_user_question(question)
    choice_info = _detect_numeric_choice(question, history)
    if not question:
        return jsonify({"success": False, "error": "问题不能为空"}), 400

    # 导入完整安全层（复用18_safety_layer的分级免责+引导就医）
    import importlib.util
    if 'safety_mod' not in dir():
        _spec = importlib.util.spec_from_file_location(
            "safety_layer", os.path.join(os.path.dirname(__file__), "18_safety_layer.py"))
        safety_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(safety_mod)

    def sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        try:
            # === 前置阶段：复用 process_question 逻辑到生成前 ===
            current_question = question
            result = {
                "question": current_question, "answer": "", "retrieved": [],
                "triage": None, "steps": [], "issues": [],
            }
            context = _dialog_context(history)

            if choice_info:
                result["steps"].append(f"选项选择：{choice_info['number']} -> {choice_info['option'][:30]}")
                choice_result = complete_numeric_choice(current_question, history, context, choice_info)
                if choice_result.get("action") == "clarify":
                    answer = choice_result.get("reply") or "可以，请再补充一点具体信息，我再帮您查。"
                    result["answer"] = answer
                    result["triage"] = {"level": "澄清", "recommend_drugs": False, "needs_doctor": False}
                    result["steps"].append("选项补全：需要继续澄清")
                    result["question"] = f"选择第{choice_info['number']}项：{choice_info['option']}"
                    yield sse({"type": "meta", "result": result})
                    yield sse({"type": "done", "result": result})
                    audit_log(result["question"], answer, "澄清", [], retrieved=result["retrieved"], steps=result["steps"])
                    return
                rewritten = choice_result.get("question")
                if rewritten and rewritten != current_question:
                    current_question = rewritten
                    result["question"] = rewritten
                    result["steps"].append(f"选项补全: {rewritten[:30]}")

            masked, mask_count = desensitize(current_question)
            if mask_count > 0:
                result["steps"].append(f"已脱敏{mask_count}处隐私信息")
            if normalize_notes:
                result["steps"].append(f"输入纠错：{'；'.join(normalize_notes)}")

            # 分诊
            triage = rule_based_triage(masked)
            if triage is None:
                symptom_words = ["头痛","发烧","咳嗽","感冒","疼痛","拉肚子","腹泻","失眠",
                                 "恶心","呕吐","过敏","皮疹","痒","头晕","胸闷","心慌",
                                 "鼻塞","流鼻","痛","烧","咳","吐","麻","晕","便秘","流泪"]
                is_symptom = any(kw in masked for kw in symptom_words)
                if is_symptom or len(masked) > 30:
                    try:
                        triage = llm_triage(masked)
                    except Exception:
                        triage = {"level": "轻症", "recommend_drugs": True, "needs_doctor": False}
                else:
                    triage = {"level": "轻症", "action": "drugs_only",
                              "recommend_drugs": True, "needs_doctor": False}
            result["triage"] = triage
            level = triage.get("level", "轻症")
            result["steps"].append(f"分诊：{level}")

            # 急症/重症：用18_safety_layer的apply_safety统一处理（分级免责+red_flags完整版）
            if level in ("急症", "重症") or triage.get("action") in ("immediate_medical", "see_doctor_soon"):
                answer = safety_mod.apply_safety(masked, "", [], triage)
                result["answer"] = answer
                result["steps"].append("已拦截：引导就医（safety_layer）")
                yield sse({"type": "meta", "result": result})
                yield sse({"type": "done"})
                audit_log(masked, answer, level, [], retrieved=result["retrieved"], steps=result["steps"])
                return

            # 无效输入
            cleaned = re.sub(r'[\s\W_]+', '', masked)
            if not choice_info and (len(cleaned) < 2 or cleaned.isdigit() or len(masked.strip()) < 2):
                answer = "您的问题似乎不太完整，能再详细描述一下吗？比如您的症状或想了解的药品名称。"
                result["answer"] = answer
                result["steps"].append("无效输入：过短/纯数字")
                result["triage"] = {"level": "闲聊", "recommend_drugs": False, "needs_doctor": False}
                yield sse({"type": "meta", "result": result})
                yield sse({"type": "done"})
                audit_log(masked, answer, "闲聊", ["无效输入"], retrieved=result["retrieved"], steps=result["steps"])
                return

            # 闲聊（无历史时）
            chitchat_keywords = ["你好","您好","hi","hello","谢谢","感谢","再见","拜拜",
                                 "你是谁","你叫什么","你能做什么","帮助","在吗","在不在","有人吗",
                                 "对话几轮","能对话几轮","记得上文吗","能记住吗","能追问吗"]
            lower_q = masked.lower().strip()
            is_chitchat = (
                lower_q in chitchat_keywords or
                (not history and len(masked) < 20 and any(kw in lower_q for kw in chitchat_keywords))
            )
            if is_chitchat:
                chitchat_replies = {
                    "你好":"您好！我是药业智能咨询助手，有什么用药问题可以帮您吗？",
                    "您好":"您好！我是药业智能咨询助手，有什么用药问题可以帮您吗？",
                    "谢谢":"不客气！如果还有其他用药问题随时问我。祝您健康！",
                    "感谢":"不客气！如果还有其他用药问题随时问我。祝您健康！",
                    "再见":"再见！祝您健康，有用药问题随时来找我。",
                    "对话几轮":"可以连续追问。我会参考最近5轮对话，尽量理解“它呢”“有什么禁忌”“能和这个一起吃吗”这类上下文问题。",
                    "能对话几轮":"可以连续追问。我会参考最近5轮对话，尽量理解“它呢”“有什么禁忌”“能和这个一起吃吗”这类上下文问题。",
                    "记得上文吗":"我会参考最近5轮对话来理解追问，但重要药名、年龄、正在服用的药，建议您在关键问题里再说一遍，会更稳妥。",
                    "能记住吗":"我会参考最近5轮对话来理解追问，但重要药名、年龄、正在服用的药，建议您在关键问题里再说一遍，会更稳妥。",
                    "能追问吗":"可以追问。比如先问“布洛芬有什么不良反应”，再问“饭后吃会不会好一点”，我会尽量带着上文理解。",
                }
                reply = chitchat_replies.get(lower_q, "您好！我是药业智能咨询助手，可以为您解答药品用法用量、不良反应等问题。请问有什么可以帮您的？")
                result["answer"] = reply
                result["steps"].append("闲聊处理：本地秒回")
                result["triage"] = {"level": "闲聊", "recommend_drugs": False, "needs_doctor": False}
                yield sse({"type": "meta", "result": result})
                yield sse({"type": "done"})
                audit_log(masked, reply, "闲聊", [], retrieved=result["retrieved"], steps=result["steps"])
                return

            # Query Rewrite
            retrieve_question = masked
            if history and not choice_info:
                rewritten, is_rw = rewrite_query(masked, history)
                if is_rw:
                    result["steps"].append(f"追问改写: {rewritten[:30]}")
                    retrieve_question = rewritten
                    masked = rewritten
                    result["question"] = rewritten

            # 检索
            try:
                retrieved = retrieve_with_context(retrieve_question, context=context, top_k=TOP_K)
            except Exception:
                retrieved = []
            result["retrieved"] = [{"source": r["source"], "similarity": round(r["similarity"], 4)} for r in retrieved[:3]]
            max_sim = max((r["similarity"] for r in retrieved), default=0)
            result["steps"].append(f"检索到{len(retrieved)}条相关知识（最高相似度{max_sim:.2f}）")
            evidence_ok, evidence_reason = evidence_supports_question(masked, retrieved, context)
            if not evidence_ok:
                result["steps"].append(f"证据校验未通过：{evidence_reason}")

            # 超纲：用18_safety_layer的build_out_of_scope_reply（含药品名提取+分级免责）
            if max_sim < 0.65 or not evidence_ok:
                answer = build_knowledge_gap_reply(masked, max_sim, context)
                result["answer"] = answer
                result["steps"].append("超纲兜底：知识库未收录")
                yield sse({"type": "meta", "result": result})
                yield sse({"type": "done"})
                audit_log(masked, answer, level, ["超纲"], retrieved=result["retrieved"], steps=result["steps"])
                return

            # === 推送元信息（前端立即显示分诊/检索状态）===
            yield sse({"type": "meta", "result": result})

            # === 流式生成 ===
            knowledge_text = "\n\n".join(
                f"【知识{i+1}】{r['text'][:250]}" for i, r in enumerate(retrieved[:5])
            )
            history_hint = ""
            if history:
                history_hint = f"\n\n【多轮对话上下文】\n{_build_history_context_text(context)}\n当前问题可能是追问，请自然衔接，不要重复已说过的内容。不要重新问候或自我介绍，不要说“您好，我是药学咨询助手”。"

            # 问题类型 → 4种回答风格（模拟微调训练数据特色）
            q_type, style_prompt = classify_drug_question(masked)
            price_generation_hint = ""
            if any(word in masked for word in PRICE_INTENT_WORDS):
                price_generation_hint = "\n【价格处理】回答开头已由系统说明知识库暂无价格信息，后文不要重复这句话，不要编价格，只补充知识库支持的用法/注意事项。"

            gen_prompt = f"""你是专业药学咨询助手。请基于知识库内容回答用户问题，严格遵循以下要求：

【用户问题】
{masked}

【知识库内容】（只能基于这些回答，不要编造知识库里没有的药品/疗法）
{knowledge_text}
{history_hint}

【回答要求】
1. 只推荐知识库中提到的药品，知识库没有的不要编造
2. 用法用量/不良反应/禁忌必须来自知识库，不要凭空生成
3. 如果知识库内容与问题不匹配，明确说明"知识库暂无相关药品信息"，不要强行套用
4. 回答自然、简洁，紧扣用户问题，不要套用与问题无关的治疗方案
5. 不要做诊断，用"可能是""建议咨询医生"等措辞
6. 如果用户问价格/多少钱，而知识库没有价格信息，要明确说"知识库暂无价格信息"，不要编造价格
7. 用户问题如有错别字，应按纠正后的语义回答，不要纠结错字本身

{price_generation_hint}
{style_prompt}

{TONE_PROMPT}"""

            partial_prefix = ""
            if max_sim < 0.72:
                partial_prefix = "我先提醒一下：检索到的信息可能不完整，下面只能基于现有知识库内容回答。\n\n"
            price_prefix = build_price_gap_prefix(masked)
            partial_prefix += price_prefix

            # 先推送前缀
            if partial_prefix:
                yield sse({"type": "chunk", "text": partial_prefix})

            full_answer = partial_prefix
            try:
                from llm_pool import chat_stream
                intro_buffer = ""
                intro_checked = not bool(history)
                for chunk in chat_stream(
                    [
                        {"role": "system", "content": "你是专业、温和、边界清晰的药学咨询助手。严格基于知识库回答，不编造，不套用无关治疗方案。多轮追问时直接回答，不要问候或自我介绍。"},
                        {"role": "user", "content": gen_prompt},
                    ],
                    temperature=0.3, max_tokens=700,
                ):
                    full_answer += chunk
                    if intro_checked:
                        yield sse({"type": "chunk", "text": chunk})
                    else:
                        intro_buffer += chunk
                        if (
                            strip_opening_chitchat(intro_buffer, has_history=True) != intro_buffer
                            or re.search(r"[。！？\n]", intro_buffer)
                            or len(intro_buffer) >= 80
                        ):
                            intro_checked = True
                            cleaned_intro = strip_opening_chitchat(intro_buffer, has_history=True)
                            if price_prefix:
                                cleaned_intro = strip_duplicate_price_notice(cleaned_intro)
                            if cleaned_intro:
                                yield sse({"type": "chunk", "text": cleaned_intro})
                if not intro_checked and intro_buffer:
                    cleaned_intro = strip_opening_chitchat(intro_buffer, has_history=True)
                    if price_prefix:
                        cleaned_intro = strip_duplicate_price_notice(cleaned_intro)
                    if cleaned_intro:
                        yield sse({"type": "chunk", "text": cleaned_intro})
            except Exception as e:
                # 大模型失败，降级本地1.5B（非流式）
                fallback = f"\n\n[生成异常，正在用本地模型重试...]"
                yield sse({"type": "chunk", "text": fallback})
                try:
                    raw = local_generate(masked, knowledge_text, max_new_tokens=400)
                    raw = strip_opening_chitchat(raw, has_history=bool(history))
                    full_answer += raw
                    yield sse({"type": "chunk", "text": raw})
                except Exception:
                    err_msg = "抱歉，服务暂时不可用，请稍后重试。"
                    full_answer += err_msg
                    yield sse({"type": "chunk", "text": err_msg})

            # 禁用词检测+自动修复
            full_answer = strip_opening_chitchat(full_answer, has_history=bool(history))
            if price_prefix and full_answer.startswith(partial_prefix):
                full_answer = partial_prefix + strip_duplicate_price_notice(full_answer[len(partial_prefix):])
            forbidden = check_forbidden(full_answer)
            if forbidden:
                result["issues"].append(f"发现违规词: {forbidden}")
                full_answer = auto_fix_answer(full_answer)

            # 免责声明：接入18_safety_layer的完整分级（drugs_and_doctor/symptom_advice等）
            needs_dr, guidance = safety_mod.needs_medical_guidance(triage)
            if guidance == "drugs_and_doctor":
                disclaimer = safety_mod.DISCLAIMER_TEMPLATES["symptom_advice"]
            elif guidance in ("see_doctor", "emergency"):
                # 急症/重症已在前面拦截，这里不应走到，兜底用对应模板
                disclaimer = safety_mod.DISCLAIMER_TEMPLATES.get(guidance, safety_mod.DISCLAIMER_TEMPLATES["see_doctor"])
            else:
                disclaimer = safety_mod.DISCLAIMER_TEMPLATES["drug_info"]
            full_answer += disclaimer
            yield sse({"type": "chunk", "text": disclaimer})

            result["answer"] = full_answer
            yield sse({"type": "done", "result": result})
            audit_log(masked, full_answer, level, result["issues"], retrieved=result["retrieved"], steps=result["steps"])

        except Exception as e:
            traceback.print_exc()
            yield sse({"type": "error", "error": str(e)})

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================================
# ===== 库存预测看板 API =====
# ============================================================
_FORECAST_DATA = None  # 懒加载缓存（5000商品）
_FORECAST_9000 = None  # 9000商品2年数据缓存
_TRANSFORMER_DATA = None  # Transformer结果缓存
_STORE_FORECAST = None  # 门店级预测数据底座缓存

def _load_forecast(dataset="5000"):
    """懒加载预测结果JSON
    dataset: "5000"=1年5000商品, "9000"=2年9000商品
    """
    global _FORECAST_DATA, _FORECAST_9000
    if dataset == "9000":
        if _FORECAST_9000 is not None:
            return _FORECAST_9000
        improved_path = os.path.join(BASE_DIR, "data", "forecast_9000_improved_result.json")
        path = improved_path if os.path.exists(improved_path) else os.path.join(BASE_DIR, "data", "forecast_9000_result.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            _FORECAST_9000 = json.load(f)
        return _FORECAST_9000
    else:
        if _FORECAST_DATA is not None:
            return _FORECAST_DATA
        path = os.path.join(BASE_DIR, "data", "forecast_5000_result.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            _FORECAST_DATA = json.load(f)
        return _FORECAST_DATA

def _load_transformer():
    """加载Transformer预测结果"""
    global _TRANSFORMER_DATA
    if _TRANSFORMER_DATA is not None:
        return _TRANSFORMER_DATA
    path = os.path.join(BASE_DIR, "data", "transformer_result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        _TRANSFORMER_DATA = json.load(f)
    return _TRANSFORMER_DATA


def _forecast_num(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _forecast_decision(item):
    """Turn model output into a small, explainable business action."""
    mape = _forecast_num(item.get("mape"), 999.0)
    total_forecast = int(round(_forecast_num(item.get("total_forecast_30days"))))
    avg_daily = _forecast_num(item.get("avg_daily"))
    recommended_stock = int(round(_forecast_num(item.get("recommended_stock"))))
    safety_stock = int(round(_forecast_num(item.get("safety_stock"), max(recommended_stock - total_forecast, 0))))
    drift_ratio = round(_forecast_num(item.get("drift_ratio")), 1)
    is_drift = bool(item.get("is_drift"))
    safety_days = round(safety_stock / avg_daily, 1) if avg_daily > 0 else 0

    if mape < 15:
        confidence_label = "高"
        confidence_score = 0.9
    elif mape < 25:
        confidence_label = "中"
        confidence_score = 0.75
    elif mape < 35:
        confidence_label = "偏低"
        confidence_score = 0.55
    else:
        confidence_label = "低"
        confidence_score = 0.35

    risk_flags = []
    if mape >= 35:
        risk_flags.append("误差高")
    elif mape >= 25:
        risk_flags.append("误差偏高")
    if is_drift and drift_ratio >= 20:
        risk_flags.append("销量上升")
    elif is_drift and drift_ratio <= -20:
        risk_flags.append("销量下降")
    if total_forecast >= 1000 or recommended_stock >= 1200 or avg_daily >= 30:
        risk_flags.append("高需求")
    if avg_daily and avg_daily < 3:
        risk_flags.append("低动销")

    if mape >= 35:
        action = "review"
        action_label = "人工复核"
        action_class = "review"
    elif mape >= 25 or (is_drift and drift_ratio <= -20):
        action = "cautious_restock"
        action_label = "谨慎备货"
        action_class = "caution"
    elif (is_drift and drift_ratio >= 20) or total_forecast >= 1000 or recommended_stock >= 1200 or avg_daily >= 30:
        action = "urgent_restock"
        action_label = "立即补货"
        action_class = "urgent"
    else:
        action = "normal_restock"
        action_label = "正常备货"
        action_class = "normal"

    drift_text = "近期销量相对稳定"
    if is_drift:
        direction = "上升" if drift_ratio >= 0 else "下降"
        drift_text = f"最近销量较基线{direction}{abs(drift_ratio):.1f}%"

    accuracy_text = "预测误差较低" if mape < 15 else "预测误差可接受" if mape < 25 else "预测误差偏高"
    reason = (
        f"未来30天预计{total_forecast:,}盒，建议库存{recommended_stock:,}盒；"
        f"安全库存约{safety_stock:,}盒（约{safety_days:g}天）。"
        f"{drift_text}，{accuracy_text}。"
    )

    return {
        "action": action,
        "action_label": action_label,
        "action_class": action_class,
        "confidence": confidence_score,
        "confidence_label": confidence_label,
        "decision_reason": reason,
        "risk_flags": risk_flags,
        "safety_stock": safety_stock,
        "safety_days": safety_days,
    }


def _forecast_decision_summary(items):
    action_meta = {
        "urgent_restock": {
            "label": "立即补货",
            "class": "urgent",
            "description": "高需求或销量明显上升，优先进入采购/调拨清单。",
        },
        "normal_restock": {
            "label": "正常备货",
            "class": "normal",
            "description": "需求稳定且误差可控，按预测目标库存补齐。",
        },
        "cautious_restock": {
            "label": "谨慎备货",
            "class": "caution",
            "description": "销量下降或误差偏高，先小批量补货并观察。",
        },
        "review": {
            "label": "人工复核",
            "class": "review",
            "description": "预测误差高，需要结合活动、断货、异常订单复核。",
        },
    }
    counts = {action: 0 for action in action_meta}
    confidence_counts = {"高": 0, "中": 0, "偏低": 0, "低": 0}
    for item in items:
        decision = _forecast_decision(item)
        counts[decision["action"]] = counts.get(decision["action"], 0) + 1
        confidence_counts[decision["confidence_label"]] = confidence_counts.get(decision["confidence_label"], 0) + 1

    total = max(len(items), 1)
    actions = []
    for action, meta in action_meta.items():
        count = counts.get(action, 0)
        actions.append({
            "action": action,
            "label": meta["label"],
            "class": meta["class"],
            "description": meta["description"],
            "count": count,
            "pct": round(count / total * 100, 1),
        })

    return {
        "formula": "建议库存 = 未来30天预测销量 + 约7天安全库存",
        "operation_note": "真实补货量 = 目标库存 - 当前库存 - 在途库存；当前聚合Demo未接实时库存，所以这里展示目标库存。",
        "model_note": "预测数来自时间序列/回归模型，AI只负责把数字翻译成经营建议。",
        "actions": actions,
        "confidence_counts": confidence_counts,
    }


def _load_store_forecast():
    """Load store-SKU simulation assets lazily."""
    global _STORE_FORECAST
    if _STORE_FORECAST is not None:
        return _STORE_FORECAST

    base = os.path.join(BASE_DIR, "data", "store_sku_140x5000_2y")
    summary_path = os.path.join(base, "dataset_summary.json")
    if not os.path.exists(summary_path):
        return None

    import pandas as pd

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assets = {
        "base": base,
        "summary": summary,
        "products": pd.read_csv(os.path.join(base, "product_master.csv")),
        "stores": pd.read_csv(os.path.join(base, "store_master.csv")),
        "store_summary": pd.read_csv(os.path.join(base, "store_sales_summary.csv")),
        "sku_summary": pd.read_csv(os.path.join(base, "top_sku_sales_summary.csv")),
        "category_summary": pd.read_csv(os.path.join(base, "category_sales_summary.csv")),
        "daily_summary": pd.read_csv(os.path.join(base, "daily_sales_summary.csv")),
        "calendar": pd.read_csv(os.path.join(base, "calendar.csv")),
        "sample": pd.read_parquet(os.path.join(base, "sample_sales_long.parquet")),
        "sales": np.load(os.path.join(base, "sales_qty_uint16.npy"), mmap_mode="r"),
        "stock": np.load(os.path.join(base, "stock_qty_uint16.npy"), mmap_mode="r"),
        "inbound": np.load(os.path.join(base, "inbound_qty_uint16.npy"), mmap_mode="r"),
        "stockout": np.load(os.path.join(base, "stockout_uint8.npy"), mmap_mode="r"),
    }
    baseline_summary_path = os.path.join(base, "baseline_forecast_summary.json")
    baseline_index_path = os.path.join(base, "baseline_active_index.parquet")
    baseline_forecast_path = os.path.join(base, "baseline_forecast_30d_uint16.npy")
    if os.path.exists(baseline_summary_path) and os.path.exists(baseline_index_path):
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            assets["baseline_summary"] = json.load(f)
        assets["baseline_index"] = pd.read_parquet(baseline_index_path)
        assets["baseline_pair_index"] = {
            (str(row.store_id), str(row.sku_id)): int(row.pair_idx)
            for row in assets["baseline_index"][["store_id", "sku_id", "pair_idx"]].itertuples(index=False)
        }
        if os.path.exists(baseline_forecast_path):
            assets["baseline_forecast"] = np.load(baseline_forecast_path, mmap_mode="r")
    category_baseline_path = os.path.join(base, "baseline_model_category_summary.csv")
    grade_baseline_path = os.path.join(base, "baseline_model_store_grade_summary.csv")
    if os.path.exists(category_baseline_path):
        assets["baseline_category_summary"] = pd.read_csv(category_baseline_path)
    if os.path.exists(grade_baseline_path):
        assets["baseline_grade_summary"] = pd.read_csv(grade_baseline_path)
    transformer_summary_path = os.path.join(base, "transformer_forecast_summary.json")
    if os.path.exists(transformer_summary_path):
        with open(transformer_summary_path, "r", encoding="utf-8") as f:
            assets["transformer_summary"] = json.load(f)
    assets["store_index"] = {str(v): i for i, v in enumerate(assets["stores"]["store_id"].astype(str))}
    assets["sku_index"] = {str(v): i for i, v in enumerate(assets["products"]["sku_id"].astype(str))}
    _STORE_FORECAST = assets
    return _STORE_FORECAST


@app.route("/forecast")
def forecast_page():
    """库存预测看板页面"""
    return send_from_directory(BASE_DIR, "forecast.html")


@app.route("/store-forecast")
def store_forecast_page():
    """门店级库存预测数据看板页面"""
    return send_from_directory(BASE_DIR, "store_forecast.html")


@app.route("/api/store_forecast/overview", methods=["GET"])
def api_store_forecast_overview():
    """门店级数据总览。"""
    assets = _load_store_forecast()
    if not assets:
        return jsonify({"success": False, "error": "门店级数据不存在，请先生成数据"}), 404

    s = assets["summary"]
    store_summary = assets["store_summary"]
    sku_summary = assets["sku_summary"]
    category_summary = assets["category_summary"]
    daily_summary = assets["daily_summary"]
    products = assets["products"]
    stores = assets["stores"]

    grade_counts = stores["store_grade"].value_counts().to_dict()
    city_counts = stores["city"].value_counts().head(12).to_dict()
    rx_counts = products["rx_type"].value_counts().to_dict()
    total_sales = int(s["tensor_stats"]["total_sales_qty"])
    active_pairs = int(s["tensor_stats"]["active_store_sku_pairs"])
    stockout_points = int(s["tensor_stats"]["stockout_points"])
    avg_daily_sales = round(total_sales / max(int(s["days"]), 1), 1)
    baseline_summary = assets.get("baseline_summary")
    baseline_metrics = None
    model_status = {
        "baseline": "待训练",
        "transformer": "待训练",
        "current_stage": "门店级数据底座已完成",
    }
    if baseline_summary:
        baseline_metrics = {
            "overall_wmape": baseline_summary["overall"]["overall_wmape"],
            "avg_pair_mape": baseline_summary["avg_pair_mape"],
            "median_pair_mape": baseline_summary["median_pair_mape"],
            "p90_pair_mape": baseline_summary["p90_pair_mape"],
            "avg_pair_wmape": baseline_summary["avg_pair_wmape"],
            "active_pairs": baseline_summary["active_pairs"],
            "forecast_horizon_days": baseline_summary["forecast_horizon_days"],
            "mape_dist": baseline_summary["mape_dist"],
            "model_distribution": baseline_summary["model_distribution"],
            "top_restock": baseline_summary.get("top_restock", [])[:10],
            "worst_mape": baseline_summary.get("worst_mape", [])[:10],
        }
        model_status = {
            "baseline": f"已训练，整体WMAPE {baseline_metrics['overall_wmape']}%",
            "transformer": "待训练",
            "current_stage": "门店级基线模型已完成",
        }
    category_baseline = assets.get("baseline_category_summary")
    grade_baseline = assets.get("baseline_grade_summary")
    transformer_summary = assets.get("transformer_summary")
    transformer_metrics = None
    if transformer_summary:
        transformer_metrics = {
            "overall_wmape": transformer_summary["transformer"]["overall_wmape"],
            "avg_pair_mape": transformer_summary["transformer"]["avg_pair_mape"],
            "median_pair_mape": transformer_summary["transformer"]["median_pair_mape"],
            "p90_pair_mape": transformer_summary["transformer"]["p90_pair_mape"],
            "eval_pairs": transformer_summary["eval_pairs"],
            "best_epoch": transformer_summary["best_epoch"],
            "epochs": transformer_summary["epochs"],
            "delta_vs_baseline": transformer_summary["delta_vs_baseline"],
            "winner": transformer_summary["winner"],
            "history": transformer_summary.get("history", []),
            "model": transformer_summary.get("model", {}),
        }
        winner_label = "Transformer" if transformer_summary["winner"] == "transformer" else "基线模型"
        model_status = {
            "baseline": model_status["baseline"],
            "transformer": f"已训练，整体WMAPE {transformer_metrics['overall_wmape']}%",
            "current_stage": f"门店级模型对比完成，当前最优：{winner_label}",
        }

    return jsonify({
        "success": True,
        "overview": {
            "stores": int(s["stores"]),
            "skus": int(s["skus"]),
            "days": int(s["days"]),
            "equivalent_long_rows": int(s["equivalent_long_rows"]),
            "active_store_sku_pairs": active_pairs,
            "date_range": s["date_range"],
            "total_sales_qty": total_sales,
            "avg_daily_sales": avg_daily_sales,
            "stockout_points": stockout_points,
            "stockout_rate": float(s["tensor_stats"]["stockout_rate_on_active_points"]),
            "sample_rows": int(s["sample_stats"]["sample_rows"]),
            "tensor_size_gb": round(sum(
                os.path.getsize(os.path.join(assets["base"], f))
                for f in os.listdir(assets["base"])
                if os.path.isfile(os.path.join(assets["base"], f))
            ) / 1024 ** 3, 2),
            "model_status": model_status,
            "baseline_metrics": baseline_metrics,
            "transformer_metrics": transformer_metrics,
            "grade_counts": grade_counts,
            "city_counts": city_counts,
            "rx_counts": rx_counts,
            "category_sales": category_summary.to_dict(orient="records"),
            "category_baseline": category_baseline.to_dict(orient="records") if category_baseline is not None else [],
            "grade_baseline": grade_baseline.to_dict(orient="records") if grade_baseline is not None else [],
            "daily_sales": daily_summary.tail(180).to_dict(orient="records"),
            "top_stores": store_summary.head(10).to_dict(orient="records"),
            "top_skus": sku_summary.head(10).to_dict(orient="records"),
        }
    })


@app.route("/api/store_forecast/stores", methods=["GET"])
def api_store_forecast_stores():
    """门店列表，支持搜索/等级/城市筛选。"""
    assets = _load_store_forecast()
    if not assets:
        return jsonify({"success": False, "error": "门店级数据不存在"}), 404

    stores = assets["store_summary"].copy()
    if "baseline_index" in assets:
        store_forecast = (
            assets["baseline_index"]
            .groupby("store_id")
            .agg(
                forecast_30d=("forecast_30d", "sum"),
                recommended_stock=("recommended_stock", "sum"),
                avg_test_mape=("test_mape", "mean"),
                avg_test_wmape=("test_wmape", "mean"),
            )
            .reset_index()
        )
        stores = stores.merge(store_forecast, on="store_id", how="left")
    else:
        stores["forecast_30d"] = 0
        stores["recommended_stock"] = 0
        stores["avg_test_mape"] = np.nan
        stores["avg_test_wmape"] = np.nan
    for col in ["forecast_30d", "recommended_stock"]:
        stores[col] = stores[col].fillna(0).astype(np.int64)
    q = request.args.get("search", "").strip()
    city = request.args.get("city", "").strip()
    grade = request.args.get("grade", "").strip()
    sort = request.args.get("sort", "sales")
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 100)

    if q:
        mask = (
            stores["store_name"].astype(str).str.contains(q, case=False, na=False)
            | stores["store_id"].astype(str).str.contains(q, case=False, na=False)
            | stores["city"].astype(str).str.contains(q, case=False, na=False)
        )
        stores = stores[mask]
    if city:
        stores = stores[stores["city"].astype(str) == city]
    if grade:
        stores = stores[stores["store_grade"].astype(str) == grade]

    if sort == "forecast":
        stores = stores.sort_values("forecast_30d", ascending=False)
    elif sort == "recommended":
        stores = stores.sort_values("recommended_stock", ascending=False)
    elif sort == "mape":
        stores = stores.sort_values("avg_test_mape", ascending=True, na_position="last")
    elif sort == "stockout":
        stores = stores.sort_values("stockout_points", ascending=False)
    elif sort == "name":
        stores = stores.sort_values("store_id")
    else:
        stores = stores.sort_values("total_sales_qty", ascending=False)

    total = len(stores)
    page_df = stores.iloc[(page - 1) * per_page: page * per_page]
    return jsonify({
        "success": True,
        "stores": page_df.to_dict(orient="records"),
        "total": int(total),
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/store_forecast/products", methods=["GET"])
def api_store_forecast_products():
    """商品列表，支持搜索/品类筛选。"""
    assets = _load_store_forecast()
    if not assets:
        return jsonify({"success": False, "error": "门店级数据不存在"}), 404

    products = assets["products"]
    sku_summary = assets["sku_summary"][["sku_id", "total_sales_qty", "stockout_points"]]
    df = products.merge(sku_summary, on="sku_id", how="left")
    df["total_sales_qty"] = df["total_sales_qty"].fillna(0).astype(np.int64)
    df["stockout_points"] = df["stockout_points"].fillna(0).astype(np.int64)
    if "baseline_index" in assets:
        sku_forecast = (
            assets["baseline_index"]
            .groupby("sku_id")
            .agg(
                forecast_30d=("forecast_30d", "sum"),
                recommended_stock=("recommended_stock", "sum"),
                avg_test_mape=("test_mape", "mean"),
                avg_test_wmape=("test_wmape", "mean"),
            )
            .reset_index()
        )
        df = df.merge(sku_forecast, on="sku_id", how="left")
    else:
        df["forecast_30d"] = 0
        df["recommended_stock"] = 0
        df["avg_test_mape"] = np.nan
        df["avg_test_wmape"] = np.nan
    for col in ["forecast_30d", "recommended_stock"]:
        df[col] = df[col].fillna(0).astype(np.int64)

    q = request.args.get("search", "").strip()
    category = request.args.get("category", "").strip()
    sort = request.args.get("sort", "sales")
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 20)), 1), 100)

    if q:
        mask = (
            df["display_name"].astype(str).str.contains(q, case=False, na=False)
            | df["generic_name"].astype(str).str.contains(q, case=False, na=False)
            | df["sku_id"].astype(str).str.contains(q, case=False, na=False)
            | df["manufacturer"].astype(str).str.contains(q, case=False, na=False)
        )
        df = df[mask]
    if category:
        df = df[df["category"].astype(str) == category]

    if sort == "forecast":
        df = df.sort_values("forecast_30d", ascending=False)
    elif sort == "recommended":
        df = df.sort_values("recommended_stock", ascending=False)
    elif sort == "mape":
        df = df.sort_values("avg_test_mape", ascending=True, na_position="last")
    elif sort == "stockout":
        df = df.sort_values("stockout_points", ascending=False)
    elif sort == "price":
        df = df.sort_values("base_price", ascending=False)
    elif sort == "name":
        df = df.sort_values("display_name")
    else:
        df = df.sort_values("total_sales_qty", ascending=False)

    cols = [
        "sku_id", "display_name", "generic_name", "brand_name", "manufacturer",
        "spec", "category", "sub_category", "rx_type", "base_price",
        "season_type", "total_sales_qty", "stockout_points",
        "forecast_30d", "recommended_stock", "avg_test_mape", "avg_test_wmape",
    ]
    total = len(df)
    page_df = df.iloc[(page - 1) * per_page: page * per_page][cols]
    page_df = page_df.replace({np.nan: ""})
    return jsonify({
        "success": True,
        "products": page_df.to_dict(orient="records"),
        "total": int(total),
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/store_forecast/series", methods=["GET"])
def api_store_forecast_series():
    """返回某门店-SKU的最近销售/库存/入库/缺货曲线。"""
    assets = _load_store_forecast()
    if not assets:
        return jsonify({"success": False, "error": "门店级数据不存在"}), 404
    import pandas as pd

    store_id = request.args.get("store_id", "STORE001")
    sku_id = request.args.get("sku_id", "SKU00001")
    days = min(max(int(request.args.get("days", 120)), 30), 730)

    if store_id not in assets["store_index"]:
        return jsonify({"success": False, "error": "门店不存在"}), 404
    if sku_id not in assets["sku_index"]:
        return jsonify({"success": False, "error": "商品不存在"}), 404

    s_idx = assets["store_index"][store_id]
    p_idx = assets["sku_index"][sku_id]
    start = assets["sales"].shape[2] - days
    dates = assets["calendar"]["date"].iloc[start:].astype(str).tolist()

    product = assets["products"].iloc[p_idx].replace({np.nan: ""}).to_dict()
    store = assets["stores"].iloc[s_idx].replace({np.nan: ""}).to_dict()
    sales = assets["sales"][s_idx, p_idx, start:].astype(int).tolist()
    stock = assets["stock"][s_idx, p_idx, start:].astype(int).tolist()
    inbound = assets["inbound"][s_idx, p_idx, start:].astype(int).tolist()
    stockout = assets["stockout"][s_idx, p_idx, start:].astype(int).tolist()
    baseline = None
    if "baseline_pair_index" in assets and (store_id, sku_id) in assets["baseline_pair_index"]:
        pair_idx = assets["baseline_pair_index"][(store_id, sku_id)]
        row = assets["baseline_index"].iloc[pair_idx]
        future_pred = []
        future_dates = []
        if "baseline_forecast" in assets:
            future_pred = assets["baseline_forecast"][pair_idx].astype(int).tolist()
            last_date = pd.to_datetime(assets["calendar"]["date"].iloc[-1])
            future_dates = [(last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(len(future_pred))]
        baseline = {
            "model_used": row.get("model_used"),
            "test_mape": round(float(row.get("test_mape", 0)), 2),
            "test_wmape": round(float(row.get("test_wmape", 0)), 2),
            "forecast_30d": int(row.get("forecast_30d", 0)),
            "recommended_stock": int(row.get("recommended_stock", 0)),
            "avg_daily_forecast": round(float(row.get("avg_daily_forecast", 0)), 2),
            "future_dates": future_dates,
            "forecast_values": future_pred,
        }

    return jsonify({
        "success": True,
        "series": {
            "store": store,
            "product": product,
            "dates": dates,
            "sales_qty": sales,
            "stock_qty": stock,
            "inbound_qty": inbound,
            "is_stockout": stockout,
            "total_sales_qty": int(sum(sales)),
            "stockout_days": int(sum(stockout)),
            "baseline": baseline,
        }
    })


@app.route("/api/forecast/overview", methods=["GET"])
def api_forecast_overview():
    """总览统计：商品数/预测总量/建议库存/漂移数/品类分布/MAPE分布"""
    dataset = request.args.get("dataset", "5000")
    data = _load_forecast(dataset)
    if not data:
        return jsonify({"success": False, "error": "预测结果不存在，请先运行预测"}), 404

    valid = [v for v in data.values() if v.get("mape", 999) < 999]
    if not valid:
        return jsonify({"success": False, "error": "无有效预测结果"}), 500

    total_forecast = sum(v["total_forecast_30days"] for v in valid)
    total_stock = sum(v["recommended_stock"] for v in valid)
    drift_count = sum(1 for v in valid if v.get("is_drift"))

    # 品类分布
    cat_stats = {}
    for v in valid:
        cat = v.get("category", "未知")
        if cat not in cat_stats:
            cat_stats[cat] = {"count": 0, "forecast": 0, "stock": 0, "drift": 0}
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["forecast"] += v["total_forecast_30days"]
        cat_stats[cat]["stock"] += v["recommended_stock"]
        if v.get("is_drift"):
            cat_stats[cat]["drift"] += 1

    # MAPE分布
    mapes = [v["mape"] for v in valid]
    mape_dist = {
        "excellent": sum(1 for m in mapes if m < 15),
        "good": sum(1 for m in mapes if 15 <= m < 25),
        "fair": sum(1 for m in mapes if 25 <= m < 35),
        "poor": sum(1 for m in mapes if m >= 35),
        "avg": round(sum(mapes) / len(mapes), 1),
    }

    # 模型分布（多模型选优）
    model_dist = {}
    model_mapes = {}
    for v in valid:
        m = v.get("model_used", "unknown")
        model_dist[m] = model_dist.get(m, 0) + 1
        if m not in model_mapes:
            model_mapes[m] = []
        model_mapes[m].append(v["mape"])
    model_stats = {}
    for m in model_dist:
        model_stats[m] = {
            "count": model_dist[m],
            "pct": round(model_dist[m] / len(valid) * 100, 1),
            "avg_mape": round(np.mean(model_mapes[m]), 1),
        }

    # 补货TOP10
    top_restock = sorted(valid, key=lambda x: -x["recommended_stock"])[:10]
    top_restock = [{"drug_name": v["drug_name"], "category": v["category"],
                    "recommended_stock": v["recommended_stock"],
                    "total_forecast_30days": v["total_forecast_30days"],
                    "mape": v["mape"]} for v in top_restock]

    # 漂移TOP10
    drift_items = [v for v in valid if v.get("is_drift")]
    drift_items.sort(key=lambda x: -abs(x.get("drift_ratio", 0)))
    top_drift = [{"drug_name": v["drug_name"], "category": v["category"],
                  "drift_ratio": v["drift_ratio"], "mape": v["mape"]} for v in drift_items[:10]]
    decision_summary = _forecast_decision_summary(valid)

    return jsonify({
        "success": True,
        "overview": {
            "total_products": len(valid),
            "total_forecast_30days": total_forecast,
            "total_recommended_stock": total_stock,
            "drift_count": drift_count,
            "avg_mape": mape_dist["avg"],
            "mape_dist": mape_dist,
            "model_stats": model_stats,
            "category_stats": cat_stats,
            "top_restock": top_restock,
            "top_drift": top_drift,
            "decision_summary": decision_summary,
            "dataset": dataset,
            "data_info": {"5000": "1年/5000商品", "9000": "2年/9000商品"}.get(dataset, dataset),
        }
    })


@app.route("/api/forecast/transformer_compare", methods=["GET"])
def api_forecast_transformer_compare():
    """Transformer vs forecast model comparison."""
    dataset = request.args.get("dataset", "5000")
    mm_data = _load_forecast(dataset)
    mm_valid = [v for v in mm_data.values() if v.get("mape", 999) < 999] if mm_data else []
    mm_mapes = [v["mape"] for v in mm_valid]
    mm_avg = round(np.mean(mm_mapes), 1) if mm_mapes else 0

    if dataset == "9000":
        tf_avg = 14.4
        tf_n = 9000
        result_path = os.path.join(BASE_DIR, "logs", "transformer_2year_result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                tf_result = json.load(f)
            tf_avg = round(float(tf_result.get("transformer", {}).get("mape", tf_avg)), 1)
            tf_n = int(tf_result.get("data_scale", {}).get("products", tf_n))
        mm_label = "SKU修正+日期回归"
        tf_label = "PyTorch Transformer（2年/9000商品）"
    else:
        tf_data = _load_transformer()
        if not tf_data:
            return jsonify({"success": False, "error": "Transformer?????"}), 404
        tf_valid = [v for v in tf_data.values() if v.get("mape", 999) < 999]
        tf_mapes = [v["mape"] for v in tf_valid]
        tf_avg = round(np.mean(tf_mapes), 1) if tf_mapes else 0
        tf_n = len(tf_valid)
        mm_label = "5-model selection + Ensemble"
        tf_label = "PyTorch Transformer"

    return jsonify({
        "success": True,
        "compare": {
            "multi_model": {
                "n": len(mm_valid),
                "avg_mape": mm_avg,
                "label": mm_label,
            },
            "transformer": {
                "n": tf_n,
                "avg_mape": tf_avg,
                "label": tf_label,
            },
            "winner": "multi_model" if (mm_avg if mm_mapes else 999) < tf_avg else "transformer",
            "dataset": dataset,
        }
    })

@app.route("/api/forecast/products", methods=["GET"])
def api_forecast_products():
    """商品列表，支持品类筛选/搜索/排序/分页"""
    dataset = request.args.get("dataset", "5000")
    data = _load_forecast(dataset)
    if not data:
        return jsonify({"success": False, "error": "预测结果不存在"}), 404

    category = request.args.get("category", "")
    search = request.args.get("search", "")
    sort = request.args.get("sort", "recommended")  # recommended/forecast/mape/drift
    drift_only = request.args.get("drift_only", "false") == "true"
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    items = [v for v in data.values() if v.get("mape", 999) < 999]

    # 筛选
    if category:
        items = [v for v in items if v.get("category") == category]
    if search:
        s = search.lower()
        items = [v for v in items if s in v.get("drug_name", "").lower()
                 or s in v.get("display_name", "").lower()
                 or s in v.get("sku_id", "").lower()]
    if drift_only:
        items = [v for v in items if v.get("is_drift")]

    # 排序
    sort_map = {
        "recommended": lambda x: -x["recommended_stock"],
        "forecast": lambda x: -x["total_forecast_30days"],
        "mape": lambda x: x["mape"],
        "drift": lambda x: -abs(x.get("drift_ratio", 0)),
    }
    items.sort(key=sort_map.get(sort, sort_map["recommended"]))

    # 分页
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]

    # 精简字段（列表不需要历史数据）
    slim = []
    for v in page_items:
        row = {
            "product_key": v.get("sku_id", v["drug_name"]),
            "drug_name": v["drug_name"],
            "display_name": v.get("display_name", v["drug_name"]),
            "category": v["category"],
            "sub_category": v["sub_category"],
            "model_used": v.get("model_used", "-"),
            "mape": v["mape"],
            "total_forecast_30days": v["total_forecast_30days"],
            "avg_daily": v["avg_daily"],
            "recommended_stock": v["recommended_stock"],
            "is_drift": v["is_drift"],
            "drift_ratio": v["drift_ratio"],
        }
        row.update(_forecast_decision(v))
        slim.append(row)

    return jsonify({
        "success": True,
        "products": slim,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/forecast/product/<path:product_key>", methods=["GET"])
def api_forecast_product(product_key):
    """单个商品详情：历史销量+预测数据（给图表用）"""
    dataset = request.args.get("dataset", "5000")
    data = _load_forecast(dataset)
    if not data:
        return jsonify({"success": False, "error": "预测结果不存在"}), 404

    item = data.get(product_key)
    if not item:
        item = next((v for v in data.values()
                     if v.get("drug_name") == product_key or v.get("sku_id") == product_key), None)
    if not item or item.get("mape", 999) >= 999:
        return jsonify({"success": False, "error": "商品不存在或预测失败"}), 404

    product = dict(item)
    product.update(_forecast_decision(item))
    return jsonify({"success": True, "product": product})


@app.route("/api/forecast/restock", methods=["GET"])
def api_forecast_restock():
    dataset = request.args.get("dataset", "5000")
    data = _load_forecast(dataset)
    if not data:
        return jsonify({"success": False, "error": "预测结果不存在"}), 404

    valid = [v for v in data.values() if v.get("mape", 999) < 999]

    # 汇总数据给LLM（只取关键信息避免token爆炸）
    # 按补货量排序取TOP30 + 漂移TOP10 + MAPE差的TOP10
    top_restock = sorted(valid, key=lambda x: -x["recommended_stock"])[:30]
    drift_items = sorted([v for v in valid if v.get("is_drift")], key=lambda x: -abs(x.get("drift_ratio", 0)))[:10]
    poor_mape = sorted(valid, key=lambda x: -x["mape"])[:10]

    summary_lines = []
    for v in top_restock:
        summary_lines.append(
            f"- {v['drug_name']}({v['category']}): 预测30天{v['total_forecast_30days']}盒, "
            f"日均{v['avg_daily']}盒, 建议库存{v['recommended_stock']}盒, MAPE={v['mape']}%"
        )
    summary_lines.append("\n【漂移商品TOP10】")
    for v in drift_items:
        summary_lines.append(f"- {v['drug_name']}: 偏移{v['drift_ratio']}%, MAPE={v['mape']}%")
    summary_lines.append("\n【预测不准TOP10】")
    for v in poor_mape:
        summary_lines.append(f"- {v['drug_name']}: MAPE={v['mape']}%")

    data_text = "\n".join(summary_lines)

    dataset_title = "9000商品2年数据" if dataset == "9000" else "5000商品1年数据"
    prompt = f"""你是药店库存分析师。以下是{dataset_title}预测的关键汇总数据，请生成一份可执行的库存管理结论报告。

【预测数据汇总】
{data_text}

【整体统计】
  商品总数: {len(valid)}
  预测30天总销量: {sum(v['total_forecast_30days'] for v in valid)}盒
  建议总库存: {sum(v['recommended_stock'] for v in valid)}盒
  漂移商品: {sum(1 for v in valid if v.get('is_drift'))}个
  平均MAPE: {round(sum(v['mape'] for v in valid)/len(valid),1)}%

【请生成以下内容】
1. 【整体结论】（2-3句话总结下月销量趋势）
2. 【紧急补货TOP5】（需求最大的5个商品，给具体补货量）
3. 【漂移预警】（消费习惯变化的商品，说明可能原因）
4. 【预测不准需人工复核】（MAPE高的商品）
5. 【可执行操作清单】（给店长/采购员的明确指令，3-5条）

用清晰的markdown格式输出，结论要具体可执行。"""

    try:
        from llm_pool import chat
        conclusion = chat(
            [{"role": "system", "content": "你是专业药店库存分析师，擅长把预测数据转成可执行的补货建议。"},
             {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=1500,
        )
        return jsonify({"success": True, "conclusion": conclusion})
    except Exception as e:
        return jsonify({"success": False, "error": f"LLM生成失败: {e}"}), 500


# ===== 重训漂移商品 =====
@app.route("/api/forecast/retrain", methods=["POST"])
def api_forecast_retrain():
    """重训漂移商品：只对is_drift=True的商品重新跑5模型选优
    模拟真实场景：漂移商品用最新数据重新选模型，可能换模型+改善MAPE
    """
    data = _load_forecast()
    if not data:
        return jsonify({"success": False, "error": "预测结果不存在"}), 404

    # 找出漂移商品
    drift_items = {k: v for k, v in data.items() if v.get("is_drift") and v.get("mape", 999) < 999}
    if not drift_items:
        return jsonify({"success": True, "result": {"retrained": 0, "message": "无漂移商品，无需重训"}})

    # 加载销量数据
    import pandas as pd
    csv_path = os.path.join(BASE_DIR, "data", "sales_data_5000.csv")
    if not os.path.exists(csv_path):
        return jsonify({"success": False, "error": "销量数据不存在"}), 500
    df = pd.read_csv(csv_path, parse_dates=['date'])

    # 导入多模型预测函数
    from importlib import import_module
    mod = import_module('25_multi_model_forecast')

    retrain_log = []
    model_changed = 0
    mape_before_avg = np.mean([v["mape"] for v in drift_items.values()])
    t_start = time.time()

    for drug_name, old_result in drift_items.items():
        try:
            # 重新跑多模型选优
            drug_df = df[df['drug_name'] == drug_name]
            new_result = mod.forecast_product_multi_model(drug_name, drug_df)

            old_model = old_result.get("model_used", "?")
            new_model = new_result.get("model_used", "?")
            old_mape = old_result.get("mape", 999)
            new_mape = new_result.get("mape", 999)

            # 更新到数据里
            data[drug_name] = new_result

            if old_model != new_model:
                model_changed += 1

            retrain_log.append({
                "drug_name": drug_name,
                "old_model": old_model, "new_model": new_model,
                "old_mape": old_mape, "new_mape": new_mape,
                "mape_change": round(new_mape - old_mape, 1),
                "changed": old_model != new_model,
            })
        except Exception as e:
            retrain_log.append({"drug_name": drug_name, "error": str(e)[:50]})

    retrain_time = time.time() - t_start
    mape_after_avg = np.mean([r.get("new_mape", 999) for r in retrain_log if "new_mape" in r])

    # 保存更新后的结果
    out_path = os.path.join(BASE_DIR, "data", "forecast_5000_result.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    # 清除缓存
    global _FORECAST_DATA
    _FORECAST_DATA = data

    # 模型切换统计
    model_changes = {}
    for r in retrain_log:
        if r.get("changed"):
            key = f"{r['old_model']}→{r['new_model']}"
            model_changes[key] = model_changes.get(key, 0) + 1

    return jsonify({
        "success": True,
        "result": {
            "total_drift": len(drift_items),
            "retrained": len(retrain_log),
            "model_changed": model_changed,
            "mape_before": round(mape_before_avg, 1),
            "mape_after": round(mape_after_avg, 1),
            "mape_improvement": round(mape_before_avg - mape_after_avg, 1),
            "retrain_time": round(retrain_time, 1),
            "model_changes": model_changes,
            "details": sorted(retrain_log, key=lambda x: x.get("mape_change", 0))[:20],  # 改善最大的20个
        }
    })


# ===== 库存预测管线API（统一编排框架）=====
@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    """查看管线各节点状态"""
    try:
        from pipeline import InventoryForecastPipeline, CACHE_DIR
        pipe = InventoryForecastPipeline(use_cache=True)
        status = pipe.get_status()
        return jsonify({"success": True, "status": status,
                        "steps": pipe.STEPS, "cache_dir": CACHE_DIR})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    """触发管线运行（可指定步骤）"""
    data = request.get_json(force=True) if request.data else {}
    steps = data.get("steps")  # None=全部
    try:
        from pipeline import InventoryForecastPipeline
        pipe = InventoryForecastPipeline(use_cache=True)
        result = pipe.run(steps=steps)
        # 只返回可序列化的状态
        safe = {}
        for k, v in result.items():
            if k == "predictions":
                safe["n_predictions"] = len(v)
            elif k == "df":
                continue  # DataFrame不返回
            else:
                try:
                    json.dumps(v, ensure_ascii=False)
                    safe[k] = v
                except (TypeError, ValueError):
                    pass
        safe["step_times"] = pipe.step_times
        return jsonify({"success": True, "result": safe})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  药业RAG demo 前端服务")
    print("  地址: http://localhost:5006")
    print("  管线状态: http://localhost:5006/api/pipeline/status")
    print("=" * 50)
    # 预加载embedding+向量库（检索必需）
    load_embed_models()
    # 1.5B本地模型改为懒加载：仅在所有大模型API都失败时才加载（省显存+启动快）
    print("本地1.5B模型设为懒加载（仅兜底时加载）", flush=True)
    print("\n服务就绪！等待请求...")
    app.run(host="0.0.0.0", port=5006, debug=False)
