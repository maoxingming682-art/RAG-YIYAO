"""
21_compliance_framework.py
医疗合规风控完整框架

4层防护：
  第1层 输入侧：禁用问题过滤 + 脱敏
  第2层 检索侧：超纲检测 + 处方药拦截
  第3层 生成侧：禁用词检测 + 处方药推荐拦截
  第4层 输出侧：强免责 + 审计日志 + 人工兜底

整合已有的：脱敏/分诊/超纲兜底/强免责/复杂症状
新增的：禁用词/处方药拦截/审计日志/人工兜底
"""
import os, sys, json, re, time, hashlib

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

import importlib.util
def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

safety = load_mod("18_safety_layer")

# ===== 审计日志存储 =====
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "logs", "audit_log.jsonl")


# ============================================================
# 第1层：输入侧防护
# ============================================================

# 禁止回答的问题类型（输入侧拦截）
FORBIDDEN_QUESTION_PATTERNS = [
    # 诱导诊断
    (re.compile(r'(确诊|诊断|我得了什么病|是什么病|严不严重)'), "诊断类问题",
     "AI不能进行诊断，建议就医由医生评估。"),
    # 处方药购买
    (re.compile(r'(哪里能买|怎么买|多少钱|价格|代购|哪里买到处方药)'), "购药引导问题",
     "请前往正规医疗机构或药店购买药品，处方药需凭医生处方购买。"),
    # 药物滥用
    (re.compile(r'(过量服用|多吃几片|加大剂量|怎么吃能死|安眠药.*致死|药物.*自杀)'), "药物滥用问题",
     "请勿自行调整用药剂量。如有心理困扰请拨打心理援助热线或寻求专业帮助。"),
    # 儿童剂量精确要求（需医生）
    (re.compile(r'(婴儿|新生儿|几个月.*宝宝.*用药|未满.*岁.*用药)'), "婴幼儿用药问题",
     "婴幼儿用药需严格遵医嘱，请咨询儿科医生。"),
    # 替代医生
    (re.compile(r'(不用看医生|不想去医院|不去医院|代替医生|自己治)'), "拒绝就医",
     "您的健康很重要，建议及时就医由专业医生评估。"),
]

# 脱敏规则
DESENSITIZE_RULES = [
    (re.compile(r'1[3-9]\d{9}'), '[手机号]'),
    (re.compile(r'\d{17}[\dXx]'), '[身份证]'),
    (re.compile(r'[\w.-]+@[\w.-]+\.\w+'), '[邮箱]'),
    (re.compile(r'\d{16,19}'), '[卡号]'),
    (re.compile(r'[\u4e00-\u9fa5]{2,}(?:省|市|区|县|路|街|号|楼|室|小区|栋|单元|村|镇)[\d\u4e00-\u9fa5]*'), '[地址]'),
]


def input_filter(user_input):
    """
    第1层：输入侧防护
    返回: (filtered_text, blocked, reason)
    """
    # 1. 脱敏
    masked = user_input
    masked_count = 0
    for pattern, replacement in DESENSITIZE_RULES:
        new = pattern.sub(replacement, masked)
        if new != masked:
            masked_count += 1
        masked = new

    # 2. 禁止问题检测
    for pattern, category, response in FORBIDDEN_QUESTION_PATTERNS:
        if pattern.search(masked):
            return masked, True, {"category": category, "response": response, "matched": category}

    return masked, False, {"masked_count": masked_count}


# ============================================================
# 第2层：检索侧防护
# ============================================================

# 处方药关键词（知识库里如果有这些，标注为处方药）
PRESCRIPTION_DRUG_KEYWORDS = [
    "注射液", "注射剂", "输液", "针剂",
    "控释片", "缓释片",  # 部分是处方药
    "精神类", "麻醉类", "毒性药品",
    "抗肿瘤", "化疗", "靶向药",
]

# OTC药品标识关键词
OTC_KEYWORDS = ["非处方", "OTC", "甲类OTC", "乙类OTC"]


def check_prescription_drug(retrieved_chunks):
    """
    第2层：检测检索到的药品是否是处方药
    返回: (is_prescription, drug_names)
    """
    prescription_drugs = []
    for chunk in retrieved_chunks:
        source = chunk.get("source", "")
        text = chunk.get("text", "")
        # 检查是否是处方药
        for kw in PRESCRIPTION_DRUG_KEYWORDS:
            if kw in source or kw in text[:100]:
                prescription_drugs.append(source[:20])
                break

    return len(prescription_drugs) > 0, prescription_drugs


# ============================================================
# 第3层：生成侧防护（输出审核）
# ============================================================

# 违规宣传禁用词
FORBIDDEN_WORDS = [
    # 疗效夸大
    "根治", "包治", "百分百", "100%有效", "完全治愈", "永不复发",
    "药到病除", "祖传秘方", "宫廷秘方", "神奇疗效",
    # 绝对化用语
    "最好", "最佳", "第一", "顶级", "唯一", "绝对",
    "无副作用", "无毒副作用", "没有任何副作用", "安全无毒",
    # 违规承诺
    "无效退款", "假一赔十", "签合同治疗", "保证治好",
    # 诱导用语
    "限时优惠", "马上抢购", "仅剩", "赶紧买", "立即购买",
    # 非法行医
    "确诊", "诊断为", "你得了", "你患了", "你的病是",
]

# 必须出现的安全提示关键词（至少一个）
REQUIRED_SAFETY_WORDS = ["咨询", "医生", "药师", "医嘱", "处方"]


def output_review(answer, question_type="drug_info"):
    """
    第3层：输出审核
    检测答案中的违规词 + 必须包含的安全提示
    返回: (passed, issues, fixed_answer)
    """
    issues = []

    # 1. 禁用词检测
    found_forbidden = []
    for word in FORBIDDEN_WORDS:
        if word in answer:
            found_forbidden.append(word)

    if found_forbidden:
        issues.append(f"发现违规词: {found_forbidden}")

    # 2. 安全提示检测
    has_safety = any(kw in answer for kw in REQUIRED_SAFETY_WORDS)
    if not has_safety:
        issues.append("缺少安全提示（咨询/医生/药师）")

    # 3. 诊断性语言检测
    diagnostic_phrases = ["你得了", "你患了", "确诊为", "诊断为", "你的病是"]
    found_diagnostic = [p for p in diagnostic_phrases if p in answer]
    if found_diagnostic:
        issues.append(f"发现诊断性语言: {found_diagnostic}")

    # 4. 自动修复
    fixed = answer
    # 替换诊断性语言
    fixed = fixed.replace("你得了", "您可能是")
    fixed = fixed.replace("你患了", "您可能是")
    fixed = fixed.replace("确诊为", "可能是")
    fixed = fixed.replace("诊断为", "可能是")
    fixed = fixed.replace("你的病是", "您的情况可能是")

    # 删除违规宣传词
    for word in found_forbidden:
        fixed = fixed.replace(word, "***")

    # 补安全提示
    if not has_safety:
        fixed += "\n\n具体用药请咨询药师或医生。"

    passed = len(issues) == 0
    return passed, issues, fixed


# ============================================================
# 第4层：审计日志
# ============================================================

def audit_log(question, answer, triage_result, review_result, user_id="anonymous"):
    """
    第4层：审计日志
    每个问答留痕，出问题可追溯
    """
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_hash": hashlib.md5(user_id.encode()).hexdigest()[:8],  # 用户ID脱敏
        "question": question[:200],
        "answer_preview": answer[:200] if answer else "(被拦截)",
        "triage_level": triage_result.get("level", "未分诊") if triage_result else "未分诊",
        "review_passed": review_result.get("passed", True),
        "review_issues": review_result.get("issues", []),
        "session_id": hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()[:8],
    }

    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return log_entry


# ============================================================
# 人工兜底
# ============================================================

NEEDS_HUMAN_REVIEW = [
    # 校验不通过
    "review_failed",
    # 处方药涉及
    "prescription_drug",
    # 用户主动要求人工
    "user_request_human",
    # 分诊为中症（建议人工确认）
    "moderate_symptom",
]


def should_escalate_to_human(triage_result, review_result, is_prescription):
    """判断是否需要转人工"""
    reasons = []

    if review_result and not review_result.get("passed", True):
        reasons.append("答案审核未通过")

    if is_prescription:
        reasons.append("涉及处方药")

    if triage_result:
        level = triage_result.get("level", "")
        if level in ("急症", "重症"):
            reasons.append(f"分诊为{level}")

    return len(reasons) > 0, reasons


def build_human_escalation(reasons, original_answer):
    """构建人工兜底回复"""
    reply = "您好，根据您的问题，建议您咨询我们的执业药师或前往门店由药师为您服务。\n\n"
    reply += "【转人工原因】\n"
    for r in reasons:
        reply += f"• {r}\n"
    reply += f"\n\n【参考信息】\n{original_answer[:200]}\n"
    reply += "\n---\n【温馨提示】以上信息仅供参考，执业药师将为您提供更专业的用药指导。"
    return reply


# ============================================================
# 完整合规风控框架
# ============================================================

def compliance_pipeline(question, answer, retrieved_chunks=None, triage_result=None, user_id="anonymous"):
    """
    合规风控完整流水线
    在LLM生成答案后、返回用户前调用

    参数：
    - question: 用户问题（已脱敏后的）
    - answer: LLM生成的原始答案
    - retrieved_chunks: 检索结果
    - triage_result: 分诊结果
    - user_id: 用户标识

    返回: {
        "final_answer": 最终安全回答,
        "blocked": 是否被拦截,
        "block_reason": 拦截原因,
        "review_issues": 审核问题,
        "needs_human": 是否需要人工,
        "audit": 审计日志,
    }
    """
    result = {
        "final_answer": answer,
        "blocked": False,
        "block_reason": None,
        "review_issues": [],
        "needs_human": False,
        "human_reasons": [],
        "audit": None,
    }

    # 第1层：输入侧（在问题阶段已做，这里跳过）

    # 第2层：检索侧 - 处方药检测
    is_prescription = False
    prescription_drugs = []
    if retrieved_chunks:
        is_prescription, prescription_drugs = check_prescription_drug(retrieved_chunks)
        if is_prescription:
            result["review_issues"].append(f"涉及处方药: {prescription_drugs}")

    # 第3层：输出审核 - 禁用词+安全提示+诊断语言
    question_type = "symptom" if triage_result and triage_result.get("level") not in ("轻症", None) else "drug_info"
    review_passed, review_issues, fixed_answer = output_review(answer, question_type)
    result["review_issues"].extend(review_issues)
    result["final_answer"] = fixed_answer  # 用修复后的答案

    # 第4层：安全层（超纲兜底+强免责+引导就医）
    result["final_answer"] = safety.apply_safety(
        question, result["final_answer"], retrieved_chunks, triage_result
    )

    # 人工兜底判断
    needs_human, human_reasons = should_escalate_to_human(triage_result, {"passed": review_passed, "issues": review_issues}, is_prescription)
    result["needs_human"] = needs_human
    result["human_reasons"] = human_reasons

    if needs_human and review_issues:
        # 有审核问题且需要人工 → 用人工兜底回复
        result["final_answer"] = build_human_escalation(human_reasons, fixed_answer)

    # 审计日志
    result["audit"] = audit_log(
        question, result["final_answer"], triage_result,
        {"passed": review_passed, "issues": review_issues}, user_id
    )

    return result


def input_compliance_check(user_input, user_id="anonymous"):
    """
    输入侧合规检查（在检索前调用）
    返回: (processed_input, blocked, block_info)
    """
    filtered, blocked, info = input_filter(user_input)

    if blocked:
        # 记录审计日志
        audit_log(user_input, f"(被拦截:{info['category']})", None,
                  {"passed": False, "issues": [info["category"]]}, user_id)

    return filtered, blocked, info


# ============================================================
# 演示
# ============================================================

def demo():
    print("=" * 60)
    print("  医疗合规风控完整框架演示")
    print("  4层防护 + 审计日志 + 人工兜底")
    print("=" * 60)

    # ===== 第1层演示：输入侧拦截 =====
    print(f"\n{'='*60}")
    print(f"第1层：输入侧防护")
    print(f"{'='*60}")

    input_tests = [
        "帮我诊断一下我得了什么病",
        "安眠药吃多少能致死",
        "哪里能买到便宜的处方药",
        "依托度酸片怎么服用？手机13812345678",
    ]

    for test in input_tests:
        filtered, blocked, info = input_compliance_check(test)
        status = "🚫拦截" if blocked else "✅放行"
        print(f"\n  输入: {test[:40]}")
        print(f"  状态: {status}")
        if blocked:
            print(f"  拦截原因: {info['category']}")
            print(f"  回复: {info['response']}")
        else:
            print(f"  脱敏后: {filtered[:40]}")

    # ===== 第2层演示：处方药检测 =====
    print(f"\n{'='*60}")
    print(f"第2层：处方药检测")
    print(f"{'='*60}")

    fake_retrieved_otc = [{"source": "对乙酰氨基酚片用法用量", "text": "口服，一次1片"}]
    fake_retrieved_rx = [{"source": "地西泮注射液用法用量", "text": "注射液，肌内注射"}]

    for label, chunks in [("OTC药品", fake_retrieved_otc), ("处方药(注射液)", fake_retrieved_rx)]:
        is_rx, drugs = check_prescription_drug(chunks)
        print(f"  {label}: 处方药={is_rx}, 药品={drugs}")

    # ===== 第3层演示：输出审核 =====
    print(f"\n{'='*60}")
    print(f"第3层：输出审核（禁用词+安全提示+诊断语言）")
    print(f"{'='*60}")

    output_tests = [
        ("正常回答", "依托度酸片口服一次200mg，每日3次。具体用药请咨询医生。"),
        ("违规宣传词", "这个药能根治你的头痛，百分百有效，无副作用！"),
        ("诊断性语言", "你得了感冒，需要吃阿莫西林。"),
        ("缺安全提示", "依托度酸片口服一次200mg，每日3次。"),
    ]

    for label, answer in output_tests:
        passed, issues, fixed = output_review(answer)
        print(f"\n  [{label}]")
        print(f"  原始: {answer[:50]}")
        print(f"  审核: {'✅通过' if passed else '❌不通过'}")
        if issues:
            print(f"  问题: {issues}")
            print(f"  修复: {fixed[:60]}")

    # ===== 第4层演示：审计日志 =====
    print(f"\n{'='*60}")
    print(f"第4层：审计日志")
    print(f"{'='*60}")

    # 模拟几条问答
    audit_tests = [
        {"q": "依托度酸片怎么吃", "a": "口服一次200mg...咨询医生", "triage": {"level": "轻症"}, "rx": False},
        {"q": "胸痛怎么办", "a": "(被拦截:建议就医)", "triage": {"level": "急症"}, "rx": False},
        {"q": "地西泮注射液用法", "a": "肌内注射...", "triage": {"level": "轻症"}, "rx": True},
    ]

    for t in audit_tests:
        review = {"passed": not t["rx"], "issues": ["处方药"] if t["rx"] else []}
        log = audit_log(t["q"], t["a"], t["triage"], review, "user_001")
        print(f"  [{log['timestamp']}] {log['triage_level']} | {log['question'][:20]} | 审核:{log['review_passed']}")

    print(f"\n  审计日志文件: {AUDIT_LOG_PATH}")

    # ===== 人工兜底演示 =====
    print(f"\n{'='*60}")
    print(f"人工兜底")
    print(f"{'='*60}")

    human_tests = [
        ("审核不通过+处方药", {"passed": False, "issues": ["违规词:根治"]}, True, {"level": "轻症"}),
        ("正常OTC", {"passed": True, "issues": []}, False, {"level": "轻症"}),
        ("急症", {"passed": True, "issues": []}, False, {"level": "急症"}),
    ]

    for label, review, is_rx, triage in human_tests:
        needs, reasons = should_escalate_to_human(triage, review, is_rx)
        print(f"  [{label}] 转人工: {needs}, 原因: {reasons}")

    # ===== 完整流水线演示 =====
    print(f"\n{'='*60}")
    print(f"完整合规流水线演示")
    print(f"{'='*60}")

    pipeline_tests = [
        {
            "label": "正常OTC药品查询",
            "question": "依托度酸片怎么服用？",
            "answer": "依托度酸片口服，一次200-400mg，每8小时一次。具体用药请咨询医生。",
            "retrieved": [{"similarity": 0.85, "source": "依托度酸片用法用量", "text": "口服一次200mg"}],
            "triage": {"level": "轻症", "action": "drugs_only", "needs_doctor": False, "recommend_drugs": True},
        },
        {
            "label": "违规宣传词+缺安全提示",
            "question": "这个药能治好吗",
            "answer": "这个药能根治你的病，百分百有效，无副作用！",
            "retrieved": [{"similarity": 0.7, "source": "某药品", "text": "..."}],
            "triage": None,
        },
        {
            "label": "处方药+诊断语言",
            "question": "地西泮注射液怎么用？我确诊了焦虑症",
            "answer": "你得了焦虑症，地西泮注射液肌内注射，一次10mg。",
            "retrieved": [{"similarity": 0.8, "source": "地西泮注射液用法", "text": "注射液肌内注射"}],
            "triage": {"level": "中症", "action": "drugs_and_doctor", "needs_doctor": True},
        },
    ]

    for test in pipeline_tests:
        print(f"\n{'─'*60}")
        print(f"场景: {test['label']}")
        print(f"{'─'*60}")
        print(f"问题: {test['question']}")
        print(f"原始答案: {test['answer'][:60]}")

        result = compliance_pipeline(
            question=test["question"],
            answer=test["answer"],
            retrieved_chunks=test["retrieved"],
            triage_result=test["triage"],
            user_id="demo_user"
        )

        print(f"\n审核问题: {result['review_issues']}")
        print(f"转人工: {result['needs_human']} {result['human_reasons']}")
        print(f"\n【最终回复】")
        print(result["final_answer"][:300])

    # 汇总
    print(f"\n{'='*60}")
    print(f"合规风控框架总结")
    print(f"{'='*60}")
    print(f"""
【4层防护】

第1层 输入侧
  ├─ 脱敏：手机号/身份证/地址 → 占位符
  ├─ 禁止问题：诊断类/购药引导/药物滥用/婴幼儿用药/拒绝就医
  └─ 命中即拦截，返回安全提示

第2层 检索侧
  ├─ 超纲检测：相似度<0.60 → 不展示编造答案
  └─ 处方药检测：注射液/精神类/抗肿瘤 → 标注+转人工

第3层 生成侧（输出审核）
  ├─ 禁用词检测：根治/包治/百分百/无副作用/确诊为...
  ├─ 安全提示检测：必须含"咨询/医生/药师"
  ├─ 诊断语言检测：你得了/你患了/确诊为 → 自动替换
  └─ 自动修复：替换违规词+补安全提示

第4层 输出侧
  ├─ 强免责声明（每条必带）
  ├─ 引导就医（分诊级别决定）
  ├─ 审计日志（每个问答留痕）
  └─ 人工兜底（审核不过/处方药/急症 → 转人工）

【禁用词清单】
  疗效夸大：根治/包治/百分百/药到病除/祖传秘方
  绝对化：最好/最佳/第一/唯一/绝对/无副作用
  违规承诺：无效退款/保证治好
  诱导消费：限时优惠/马上抢购
  非法行医：确诊/诊断为/你得了/你患了

【人工兜底触发条件】
  • 答案审核不通过（有违规词/缺安全提示）
  • 涉及处方药
  • 分诊为急症/重症
  • 用户主动要求人工

【审计日志】
  • 每个问答留痕（时间/用户哈希/问题/答案/分诊/审核）
  • 存储在 logs/audit_log.jsonl
  • 出问题可追溯

【为什么不能只靠prompt】
  1. prompt会被绕过（用户巧妙提问可诱导违规输出）
  2. prompt是软约束（LLM可能不遵守）
  3. 代码层是硬约束（规则引擎100%拦截）
  4. 审计日志是prompt做不到的
  5. 人工兜底是prompt做不到的
""")


if __name__ == "__main__":
    demo()
