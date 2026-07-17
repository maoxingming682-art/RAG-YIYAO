"""
知识路由层。

职责：判断用户问题应该优先走问诊、通用科普、药品知识库，还是混合处理。
这里不生成答案，只返回路由决策，便于在 app.py 里记录步骤和审计。
"""
from medical_science import detect_science_topic, is_science_education_question


PERSONAL_OR_ACUTE_MARKERS = [
    "我", "孩子", "小孩", "老人", "孕妇", "今天", "现在", "最近", "一直", "刚",
    "一天", "半天", "几天", "吃了", "吃过", "喝了", "用了", "戴了",
    "怎么办", "怎么处理", "怎么缓解", "用什么", "吃什么", "能不能", "可以用", "可以吃",
    "感觉", "不舒服", "疼", "痛", "胀", "干", "涩", "痒", "拉", "烧", "咳", "吐",
]

DRUG_INTENT_MARKERS = [
    "怎么吃", "怎么用", "用法", "用量", "剂量", "不良反应", "副作用",
    "禁忌", "慎用", "能不能一起", "同服", "饭前", "饭后", "说明书",
]


def looks_personal_or_acute(question):
    q = str(question or "")
    return any(marker in q for marker in PERSONAL_OR_ACUTE_MARKERS)


def looks_drug_fact_question(question):
    q = str(question or "")
    return any(marker in q for marker in DRUG_INTENT_MARKERS)


def looks_generic_education(question):
    q = str(question or "")
    education_markers = ["是什么", "为什么", "原因", "科普", "讲讲", "解释", "怎么回事", "常识"]
    strong_personal_markers = [
        "我", "孩子", "小孩", "老人", "孕妇", "今天", "现在", "最近", "一直", "刚",
        "一天", "半天", "几天", "吃了", "吃过", "喝了", "用了", "戴了",
        "怎么办", "用什么", "吃什么", "能不能", "可以用", "可以吃", "不舒服",
    ]
    return any(marker in q for marker in education_markers) and not any(marker in q for marker in strong_personal_markers)


def route_question(question, has_named_drug=False, has_history=False, max_sim=None, evidence_ok=True):
    """返回路由结果。

    route:
    - intake: 症状问诊优先
    - science: 通用科普优先
    - mixed: 药品名 + 科普/症状边界，后续可结合药品库
    - drug: 药品知识库优先
    - drug_or_gap: 未明确，按原 RAG 检索或兜底
    """
    q = str(question or "").strip()
    topic = detect_science_topic(q)
    science_like = is_science_education_question(q)

    if not q:
        return {"route": "drug_or_gap", "topic": "", "reason": "empty"}

    if has_named_drug:
        if topic and science_like and not looks_drug_fact_question(q):
            return {"route": "mixed", "topic": topic, "reason": "drug_name_with_science_boundary"}
        return {"route": "drug", "topic": topic, "reason": "named_drug"}

    if topic and looks_generic_education(q):
        return {"route": "science", "topic": topic, "reason": "generic_science"}

    if topic and (looks_personal_or_acute(q) or has_history):
        return {"route": "intake", "topic": topic, "reason": "personal_symptom"}

    if science_like or topic:
        return {"route": "science", "topic": topic, "reason": "science_or_symptom_education"}

    if max_sim is not None and (max_sim < 0.65 or not evidence_ok):
        return {"route": "science" if topic else "drug_or_gap", "topic": topic, "reason": "weak_drug_evidence"}

    return {"route": "drug_or_gap", "topic": topic, "reason": "fallback"}
