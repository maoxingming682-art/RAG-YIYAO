"""
17_triage.py
症状分诊系统：判断轻重缓急 → 决定推药还是转就医

4级分诊：
  轻症 → 推OTC药 + 观察建议
  中症 → 推OTC药 + 建议就医
  重症 → 不推药 + 尽快就医
  急症 → 不推药 + 立即就医
"""
import os, sys, json, time, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# ===== 红线症状清单（急症，绝对不推药）=====
EMERGENCY_KEYWORDS = [
    "胸痛", "胸闷", "呼吸困难", "气喘", "窒息",
    "意识模糊", "昏迷", "晕厥", "抽搐", "癫痫",
    "大量出血", "吐血", "便血", "咯血",
    "剧烈头痛", "爆炸性头痛", "雷击样头痛",
    "一侧肢体无力", "口角歪斜", "说话不清",  # 中风
    "过敏休克", "喉头水肿", "面部肿胀",
    "高烧不退", "超高热", "40度", "41度",
    "药物中毒", "服药过量", "误服",
    "自杀", "自残",
    "心脏骤停", "没有脉搏", "没有呼吸",
]

# 重症关键词（建议尽快就医，不推药）
SEVERE_KEYWORDS = [
    "持续高烧", "39度", "39.5", "反复发烧",
    "剧烈疼痛", "难以忍受", "止痛药无效",
    "持续呕吐", "脱水", "无法进食",
    "黄疸", "眼白发黄",
    "血尿", "尿血",
    "孕期出血", "见红", "胎动减少",
    "儿童精神萎靡", "拒食", "前囟凹陷",
    "症状加重", "持续数日不缓解", "一周未好转",
    "体重骤降", "消瘦",
    "夜间痛醒", "盗汗",
]


def call_llm(prompt, system="你是专业分诊助手", temperature=0.2, max_tokens=600):
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


def rule_based_triage(user_input):
    """
    规则引擎分诊（第一道防线，毫秒级）
    基于关键词匹配，不依赖LLM
    """
    text = user_input.lower()

    # 检查急症关键词
    for kw in EMERGENCY_KEYWORDS:
        if kw in text:
            return {
                "level": "急症",
                "action": "immediate_medical",
                "matched_keyword": kw,
                "recommend_drugs": False,
                "message": f"您描述的症状（{kw}）建议尽快由医生面诊评估，请不要自行用药，前往最近医院就诊。",
            }

    # 检查重症关键词
    for kw in SEVERE_KEYWORDS:
        if kw in text:
            return {
                "level": "重症",
                "action": "see_doctor_soon",
                "matched_keyword": kw,
                "recommend_drugs": False,
                "message": f"您描述的症状（{kw}）需要尽快就医，不建议自行用药。请前往医院就诊。",
            }

    return None  # 规则未命中，交给LLM分诊


def llm_triage(user_input):
    """
    LLM分诊（第二道防线，更精准）
    规则未命中时用LLM判断
    """
    prompt = f"""你是专业医学分诊助手。请评估用户症状的严重程度并决定处理方式。

【用户描述】
{user_input}

【分诊标准】
1. 急症：危及生命的症状（胸痛/呼吸困难/昏迷/大出血/中风等）→ 立即就医，不推药
2. 重症：需要医生诊断但非立即危及生命（持续高烧/剧烈疼痛/脱水等）→ 尽快就医，不推药
3. 中症：可以先用OTC药缓解，但建议就医确诊（发烧38.5以下/中度疼痛/疑似感染等）→ 推药+建议就医
4. 轻症：常见轻微症状（轻微感冒/轻度头痛/小擦伤等）→ 推OTC药+观察

输出严格JSON：
```json
{{
  "level": "急症/重症/中症/轻症",
  "action": "immediate_medical/see_doctor_soon/drugs_and_doctor/drugs_only",
  "severity_score": 8,
  "symptoms": ["头痛","发烧"],
  "possible_condition": "感冒",
  "recommend_drugs": false,
  "drug_categories": ["退烧药"],
  "needs_doctor": true,
  "urgency_reason": "持续高烧39度可能需要抗感染治疗",
  "advice": "建议先服用对乙酰氨基酚退烧，同时尽快就医检查感染原因",
  "red_flags": ["如果体温超过39.5请立即就医","如果出现呼吸困难请及时就医"]
}}
```
只输出JSON。"""

    result = call_llm(prompt, system="你是专业医学分诊助手，只输出JSON", temperature=0.15, max_tokens=600)

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
        # 解析失败，保守处理
        return {
            "level": "中症",
            "action": "drugs_and_doctor",
            "recommend_drugs": False,
            "needs_doctor": True,
            "advice": "无法准确判断严重程度，建议咨询医生",
        }

    return parsed


def triage(user_input):
    """
    完整分诊：规则引擎 → LLM → 决策
    """
    print(f"\n{'─'*60}")
    print(f"分诊评估: {user_input[:50]}")
    print(f"{'─'*60}")

    # 第1道：规则引擎（毫秒级）
    rule_result = rule_based_triage(user_input)
    if rule_result:
        print(f"  规则引擎: 命中[{rule_result['matched_keyword']}]")
        print(f"  分诊级别: {rule_result['level']}")
        print(f"  处理方式: {rule_result['action']}")
        print(f"  推药: {'否' if not rule_result['recommend_drugs'] else '是'}")
        print(f"  消息: {rule_result['message']}")
        return rule_result

    # 第2道：LLM分诊
    print(f"  规则引擎: 未命中，交给LLM分诊...")
    t0 = time.time()
    result = llm_triage(user_input)
    t1 = time.time()
    print(f"  LLM分诊耗时: {t1-t0:.1f}s")
    print(f"  分诊级别: {result.get('level', '未知')}")
    print(f"  严重程度评分: {result.get('severity_score', 'N/A')}/10")
    print(f"  症状: {result.get('symptoms', [])}")
    print(f"  可能疾病: {result.get('possible_condition', '')}")
    print(f"  处理方式: {result.get('action', '')}")
    print(f"  推药: {'是' if result.get('recommend_drugs') else '否'}")
    print(f"  建议就医: {'是' if result.get('needs_doctor') else '否'}")
    if result.get('urgency_reason'):
        print(f"  紧急原因: {result['urgency_reason']}")
    if result.get('advice'):
        print(f"  建议: {result['advice'][:100]}")
    if result.get('red_flags'):
        print(f"  警示信号:")
        for rf in result['red_flags']:
            print(f"    ⚠️ {rf}")
    if result.get('drug_categories'):
        print(f"  可推药品类别: {result['drug_categories']}")

    return result


def format_response(triage_result, user_input):
    """根据分诊结果格式化最终回复"""
    level = triage_result.get("level", "中症")
    action = triage_result.get("action", "drugs_and_doctor")

    print(f"\n{'='*60}")
    print(f"最终回复")
    print(f"{'='*60}")

    red_flags = triage_result.get('red_flags', [])
    red_flags_text = "".join([f"  ⚠️ {rf}\n" for rf in red_flags])

    if level == "急症":
        msg = triage_result.get('message', '您描述的症状建议尽快由医生面诊评估。')
        print(f"""
您好，根据您描述的情况，建议您尽快前往医院就诊，由专业医生为您评估。

{msg}

请不要自行用药，前往最近医院就诊即可。

【温馨提示】
- 您的症状建议尽快由医生面诊评估
- 请前往最近医院就诊，不要自行用药
- 如症状加重请及时就医""")

    elif level == "重症":
        msg = triage_result.get('message', triage_result.get('advice', '您描述的症状需要医生诊断，不建议自行用药。'))
        reason = triage_result.get('urgency_reason', '')
        print(f"""
⚠️ 建议尽快就医

{msg}

{reason}

【为什么不建议自行用药】
- 您的症状需要专业医生面诊
- 自行用药可能掩盖病情，延误治疗
- 请前往医院相关科室就诊

{red_flags_text}""")

    elif level == "中症":
        advice = triage_result.get('advice', '')
        cats = ', '.join(triage_result.get('drug_categories', ['请咨询药师']))
        print(f"""
📋 用药建议 + 就医建议

{advice}

【可参考的非处方药类别】
{cats}

【重要提醒】
- 以上建议仅供参考，不能替代医生诊断
- 建议在用药同时尽快就医，明确病因
{red_flags_text}- 处方药需凭处方购买""")

    else:  # 轻症
        advice = triage_result.get('advice', '根据您的症状，可以参考以下用药建议。')
        cats = ', '.join(triage_result.get('drug_categories', ['请咨询药师']))
        print(f"""
💊 用药建议

{advice}

【可参考的非处方药类别】
{cats}

【观察建议】
- 用药后观察症状变化
- 如症状持续3天未缓解或加重，请就医
{red_flags_text}- 具体用药请咨询药师或医生""")


def demo():
    print("=" * 60)
    print("  症状分诊系统演示")
    print("  判断轻重缓急 → 决定推药还是转就医")
    print("=" * 60)

    # 测试5个不同严重程度的场景
    tests = [
        # 1. 急症（规则引擎应该直接拦截）
        {"input": "突然剧烈胸痛呼吸困难出汗", "expect": "急症→建议就医"},
        # 2. 急症（脑卒中症状）
        {"input": "我妈突然口角歪斜说话不清一侧手脚无力", "expect": "急症→立即就医"},
        # 3. 重症（持续高烧）
        {"input": "发烧39.5度三天不退吃了退烧药也没用", "expect": "重症→就医不推药"},
        # 4. 中症（儿童发烧）
        {"input": "我儿子8岁发烧38.5度咳嗽两天了", "expect": "中症→推药+就医"},
        # 5. 轻症（普通感冒）
        {"input": "有点鼻塞流清鼻涕轻微头痛", "expect": "轻症→推OTC药"},
    ]

    for i, test in enumerate(tests):
        print(f"\n\n{'#'*60}")
        print(f"# 测试{i+1}: {test['input'][:40]}")
        print(f"# 期望: {test['expect']}")
        print(f"{'#'*60}")

        result = triage(test["input"])
        format_response(result, test["input"])

        # 防止API限流
        if i < len(tests) - 1:
            time.sleep(3)

    # 汇总
    print(f"\n{'='*60}")
    print(f"分诊系统总结")
    print(f"{'='*60}")
    print(f"""
【两道防线】

第1道：规则引擎（毫秒级，不依赖LLM）
  - 红线症状清单（急症关键词）：胸痛/呼吸困难/昏迷/大出血/中风...
  - 重症关键词：持续高烧/剧烈疼痛/脱水/孕期出血...
  - 命中即拦截，不等LLM，直接给安全建议
  - 优势：零延迟、零成本、100%可靠

第2道：LLM分诊（秒级，更精准）
  - 规则未命中时用LLM判断
  - 评估4个维度：严重程度/症状/可能疾病/是否推药
  - 输出结构化分诊结果

【4级处理】

  急症 → 🚨 不推药 + 建议尽快就医
    规则引擎拦截：胸痛/呼吸困难/昏迷/中风/大出血
    
  重症 → ⚠️ 不推药 + 尽快就医
    规则引擎或LLM判断：持续高烧/剧烈疼痛/脱水
    
  中症 → 📋 推OTC药 + 建议就医
    LLM判断：38.5以下发烧/中度疼痛/疑似感染
    
  轻症 → 💊 推OTC药 + 观察
    LLM判断：轻微感冒/轻度头痛/小擦伤

【关键原则】
  - 宁可保守：分不清轻重时按重的处理
  - 不做诊断：只判断"该不该推药"，不诊断"什么病"
  - 红线优先：急症症状不管其他信息，先拦住
  - 推药有条件：只推OTC非处方药，处方药不推
""")


if __name__ == "__main__":
    demo()
