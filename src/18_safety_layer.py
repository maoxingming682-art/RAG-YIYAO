"""
18_safety_layer.py
核心安全机制：超纲兜底 + 强免责 + 引导就医
作为所有回答的统一安全中间件

三个机制：
1. 超纲兜底：知识库没有的→诚实说没有，不编造
2. 强免责：每次回答都带"我不是医生，以上不是诊断"
3. 引导就医：根据分诊级别决定推药还是转就医

使用方式：
  from safety_layer import apply_safety
  safe_answer = apply_safety(question, answer, retrieved, triage_result)
"""
import os, sys, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 免责声明模板（按场景分级）=====
DISCLAIMER_TEMPLATES = {
    "drug_info": """
---
⚠️ 【免责声明】
- 以上信息仅供参考，不能替代医生诊断和处方
- 具体用药请咨询执业药师或医生
- 处方药需凭医生处方购买
- 如出现不良反应请立即停药并就医""",

    "symptom_advice": """
---
⚠️ 【免责声明】
- 我不是医生，以上建议不是诊断
- 症状可能由多种原因引起，需医生面诊确诊
- 如症状持续或加重请及时就医
- 不要仅凭以上建议自行用药""",

    "see_doctor": """
---
⚠️ 【免责声明】
- 您的症状需要专业医生面诊
- AI不能替代医生诊断
- 请尽快前往医院就诊，不要自行用药
- 延误就医可能导致病情恶化""",

    "emergency": """
---
【温馨提示】
- 您的症状建议尽快由医生面诊评估
- 请前往最近医院就诊，不要自行用药
- 如症状加重请及时就医""",

    "out_of_scope": """
---
⚠️ 【免责声明】
- 我的药品知识库有限，无法回答所有问题
- 以上信息可能不完整或不准确
- 请以药品说明书或医生建议为准""",
}

# ===== 超纲检测阈值 =====
OUT_OF_SCOPE_THRESHOLD = 0.60  # 相似度低于此值判定超纲
PARTIAL_MATCH_THRESHOLD = 0.65  # 0.60-0.65之间为"信息不完整"

# ===== 超纲兜底回复 =====
OUT_OF_SCOPE_REPLIES = [
    "抱歉，我的药品知识库中暂未收录「{drug}」的详细信息。",
    "我的知识库中暂时没有「{drug}」的相关记录。",
    "关于「{drug}」，我的药品知识库暂未收录该药品信息。",
]

PARTIAL_MATCH_REPLIES = [
    "我的知识库中有一些相关信息，但可能不完整。以下是基于现有知识的回答：",
    "检索到了部分相关内容，但信息可能不够全面，请结合药品说明书参考：",
]


def detect_out_of_scope(retrieved_chunks):
    """
    超纲检测：基于检索相似度判断
    返回: ("ok"|"partial"|"out_of_scope", max_similarity)
    """
    if not retrieved_chunks:
        return "out_of_scope", 0.0

    max_sim = max(c.get("similarity", 0) for c in retrieved_chunks)

    if max_sim < OUT_OF_SCOPE_THRESHOLD:
        return "out_of_scope", max_sim
    elif max_sim < PARTIAL_MATCH_THRESHOLD:
        return "partial", max_sim
    else:
        return "ok", max_sim


def extract_drug_name(question):
    """从问题中提取药品名关键词"""
    # 去掉常见问题词
    remove_words = ["怎么", "什么", "如何", "用法", "用量", "服用", "吃",
                    "不良反应", "副作用", "禁忌", "注意事项", "孕妇", "儿童",
                    "能吃吗", "能用吗", "？", "?", "呢", "吗", "的", "是"]
    text = question
    for w in remove_words:
        text = text.replace(w, "")
    # 提取中文片段
    chunks = re.findall(r'[\u4e00-\u9fa5]{2,15}', text)
    return chunks[0] if chunks else question[:10]


def build_out_of_scope_reply(question, max_sim):
    """构建超纲兜底回复"""
    drug = extract_drug_name(question)
    import random
    reply = random.choice(OUT_OF_SCOPE_REPLIES).format(drug=drug)
    reply += f"\n\n建议您：\n1. 查看药品随附的说明书\n2. 咨询执业药师\n3. 就诊时向医生咨询"
    reply += DISCLAIMER_TEMPLATES["out_of_scope"]
    return reply


def build_partial_match_prefix(max_sim):
    """构建信息不完整的前置提醒"""
    import random
    return random.choice(PARTIAL_MATCH_REPLIES)


def needs_medical_guidance(triage_result):
    """判断是否需要引导就医"""
    if not triage_result:
        return False, "unknown"

    level = triage_result.get("level", "").lower()
    needs_doctor = triage_result.get("needs_doctor", False)
    recommend_drugs = triage_result.get("recommend_drugs", True)

    if level in ("急症", "emergency") or triage_result.get("action") == "immediate_medical":
        return True, "emergency"
    if level in ("重症", "severe") or triage_result.get("action") == "see_doctor_soon":
        return True, "see_doctor"
    if needs_doctor and not recommend_drugs:
        return True, "see_doctor"
    if needs_doctor:
        return True, "drugs_and_doctor"
    return False, "drugs_only"


def apply_safety(question, answer, retrieved_chunks=None, triage_result=None):
    """
    核心安全机制：超纲兜底 + 强免责 + 引导就医
    所有回答都必须经过这个函数处理

    参数：
    - question: 用户问题
    - answer: LLM生成的原始答案
    - retrieved_chunks: 检索结果
    - triage_result: 分诊结果

    返回：处理后的安全回答
    """
    # ===== 机制1：引导就医（最高优先级）=====
    needs_doctor, guidance_type = needs_medical_guidance(triage_result)

    if guidance_type == "emergency":
        # 急症：直接返回紧急提醒，不展示原始答案
        emergency_msg = triage_result.get("message", "您描述的症状建议尽快由医生面诊评估。")
        return f"您好，根据您描述的情况，建议您尽快前往医院就诊，由专业医生为您评估。\n\n请不要自行用药，前往最近医院就诊即可。\n{DISCLAIMER_TEMPLATES['emergency']}"

    if guidance_type == "see_doctor":
        # 重症：不推药，引导就医
        advice = triage_result.get("advice", triage_result.get("message", "您的症状需要医生面诊，建议尽快就医。"))
        reason = triage_result.get("urgency_reason", "")
        red_flags = triage_result.get("red_flags", [])
        red_flags_text = "".join([f"  ⚠️ {rf}\n" for rf in red_flags])
        return f"⚠️ 建议尽快就医\n\n{advice}\n\n{reason}\n\n【为什么不建议自行用药】\n- 需要专业医生面诊\n- 自行用药可能掩盖病情\n- 请前往医院就诊\n\n{red_flags_text}{DISCLAIMER_TEMPLATES['see_doctor']}"

    # ===== 机制2：超纲兜底 =====
    if retrieved_chunks:
        scope_status, max_sim = detect_out_of_scope(retrieved_chunks)

        if scope_status == "out_of_scope":
            # 超纲：不展示原始答案，返回兜底回复
            return build_out_of_scope_reply(question, max_sim)

        elif scope_status == "partial":
            # 信息不完整：前置提醒 + 原始答案 + 免责
            prefix = build_partial_match_prefix(max_sim)
            return f"{prefix}\n\n{answer}{DISCLAIMER_TEMPLATES['drug_info']}"

    # ===== 机制3：强免责（所有回答都必须带）=====
    # 判断用哪种免责模板
    if guidance_type == "drugs_and_doctor":
        disclaimer = DISCLAIMER_TEMPLATES["symptom_advice"]
    else:
        disclaimer = DISCLAIMER_TEMPLATES["drug_info"]

    safe_answer = f"{answer}{disclaimer}"
    return safe_answer


# ===== 集成测试 =====
def demo():
    print("=" * 60)
    print("  核心安全机制演示")
    print("  超纲兜底 + 强免责 + 引导就医")
    print("=" * 60)

    # 模拟场景
    scenarios = [
        {
            "label": "场景1：正常药品查询",
            "question": "依托度酸片怎么服用？",
            "answer": "依托度酸片用法：口服，一次200-400mg，每8小时一次，每日最大不超过1.2g。",
            "retrieved": [{"similarity": 0.85, "source": "依托度酸片用法用量"}],
            "triage": None,
        },
        {
            "label": "场景2：超纲药品（知识库没有）",
            "question": "阿莫西林胶囊怎么吃？",
            "answer": "阿莫西林用法：成人每次0.5g，每日3次...",  # LLM编造的答案
            "retrieved": [{"similarity": 0.55, "source": "阿昔莫司胶囊禁忌"}],
            "triage": None,
        },
        {
            "label": "场景3：信息不完整（相似度中等）",
            "question": "对乙酰氨基酚片儿童能用吗",
            "answer": "对乙酰氨基酚可用于儿童退烧，但需根据体重计算剂量。",
            "retrieved": [{"similarity": 0.63, "source": "对乙酰氨基酚缓释片用法"}],
            "triage": None,
        },
        {
            "label": "场景4：急症（胸痛→建议就医）",
            "question": "突然胸痛呼吸困难",
            "answer": "可能是心脏问题，建议...",  # 不应展示
            "retrieved": [{"similarity": 0.3}],
            "triage": {"level": "急症", "action": "immediate_medical",
                       "message": "胸痛症状建议尽快由医生面诊评估，请不要自行用药。"},
        },
        {
            "label": "场景5：重症（高烧不退→就医不推药）",
            "question": "发烧39.5度三天不退",
            "answer": "可以吃布洛芬退烧...",  # 不应展示
            "retrieved": [{"similarity": 0.7}],
            "triage": {"level": "重症", "action": "see_doctor_soon",
                       "advice": "持续高烧39.5度需要就医",
                       "urgency_reason": "可能需要抗感染治疗",
                       "red_flags": ["体温超过40度立即就医", "出现呼吸困难请及时就医"]},
        },
        {
            "label": "场景6：中症（推药+建议就医）",
            "question": "发烧38度有点咳嗽",
            "answer": "可以服用对乙酰氨基酚退烧，多喝水多休息。",
            "retrieved": [{"similarity": 0.75}],
            "triage": {"level": "中症", "needs_doctor": True, "recommend_drugs": True,
                       "advice": "可先用药缓解，建议就医确诊"},
        },
    ]

    for s in scenarios:
        print(f"\n{'='*60}")
        print(f"{s['label']}")
        print(f"{'='*60}")
        print(f"问题: {s['question']}")
        print(f"原始答案: {s['answer'][:60]}...")
        print(f"检索相似度: {s['retrieved'][0]['similarity'] if s['retrieved'] else '无'}")
        print(f"分诊: {s['triage']['level'] if s['triage'] else '无'}")
        print(f"\n--- 安全处理后 ---")

        safe = apply_safety(s["question"], s["answer"], s["retrieved"], s["triage"])
        print(safe)

    # 汇总
    print(f"\n{'='*60}")
    print(f"核心安全机制总结")
    print(f"{'='*60}")
    print(f"""
【三个机制，优先级从高到低】

机制1：引导就医（最高优先级）
  ├─ 急症 → 拦截原始答案，返回"立即就医"
  ├─ 重症 → 拦截原始答案，返回"尽快就医，不推药"
  └─ 中症 → 允许推药 + 附带"建议就医"

机制2：超纲兜底
  ├─ 相似度<0.60 → 拦截LLM编造的答案，返回"知识库没有+建议咨询"
  ├─ 0.60-0.65   → 前置提醒"信息不完整" + 展示答案
  └─ >0.65       → 正常展示答案

机制3：强免责（所有回答必带）
  ├─ 药品查询 → "不能替代医生诊断+咨询药师"
  ├─ 症状建议 → "我不是医生+以上不是诊断"
  ├─ 就医引导 → "需医生面诊+不要自行用药"
  └─ 超纲兜底 → "知识库有限+以说明书为准"

【拦截逻辑】
  急症 → 拦截LLM答案，不展示，只给就医提醒
  重症 → 拦截LLM答案，不展示，只给就医建议
  超纲 → 拦截LLM编造的答案，不展示，只给兜底回复
  其余 → 展示LLM答案 + 追加强免责声明

【设计原则】
  - 宁可拦截也不让危险答案出去
  - 宁可保守也不让编造信息出去
  - 每条回答都必须带免责声明
  - 急症/重症不浪费一秒在LLM答案上
""")


if __name__ == "__main__":
    demo()
