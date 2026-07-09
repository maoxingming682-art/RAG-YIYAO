"""
20_chronic_symptom.py
慢性/复杂症状处理：失眠、手脚发麻等不适合直接推OTC药的症状

策略：
1. 识别"复杂症状"（多个系统症状/慢性症状/病因不明的症状）
2. 不强行推OTC药
3. 给出可能方向（不是诊断）
4. 重点引导就医 + 日常建议
"""
import os, sys, json, re, time

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

import importlib.util
def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

safety = load_mod("18_safety_layer")


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


def analyze_complex_symptom(user_input):
    """
    分析复杂/慢性症状
    区别于简单症状（感冒→退烧药），复杂症状需要：
    - 不强行推OTC药
    - 给出可能方向（不是诊断）
    - 引导就医
    - 给日常建议
    """
    prompt = f"""你是专业药学咨询助手。用户描述了一些症状，请分析并给出建议。

【用户描述】
{user_input}

【分析任务】
1. 判断症状类型：
   - simple_otc：简单症状，有明确OTC药对应（如感冒→退烧药，头痛→止痛药）
   - complex_need_doctor：复杂/慢性/多系统症状，不建议自行用药，需医生诊断
   - lifestyle：生活方式相关，可给日常建议+可选OTC辅助

2. 如果是complex_need_doctor：
   - 不要推荐具体药品
   - 给出可能涉及的方向（不是诊断，是"可能跟XX有关，建议看XX科"）
   - 建议就诊科室
   - 给日常护理建议

3. 如果是simple_otc：
   - 推荐OTC药品类别
   - 但仍建议就医确诊

输出严格JSON：
```json
{{
  "symptom_type": "complex_need_doctor",
  "symptoms": ["失眠", "手脚发麻"],
  "possible_directions": [
    {{"direction": "可能与神经系统有关", "department": "神经内科"}},
    {{"direction": "可能与颈椎有关", "department": "骨科"}},
    {{"direction": "可能与焦虑/压力有关", "department": "心理科或精神科"}}
  ],
  "recommend_otc": false,
  "otc_categories": [],
  "lifestyle_advice": ["保持规律作息", "睡前避免手机", "适当运动", "注意颈椎姿势"],
  "needs_doctor": true,
  "urgency": "非紧急，建议近期就医",
  "advice": "您的症状涉及多个系统，不建议自行用药。建议先调整作息观察1-2周，如无改善请就医。",
  "red_flags": ["如手脚发麻加重或扩散", "如出现言语不清或肢体无力", "如失眠严重影响生活"]
}}
```
只输出JSON。"""

    result = call_llm(prompt, system="你是专业药学分析助手，只输出JSON", temperature=0.2, max_tokens=700)

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
        return {
            "symptom_type": "complex_need_doctor",
            "recommend_otc": False,
            "needs_doctor": True,
            "advice": "您的症状建议由医生评估，不建议自行用药。",
            "possible_directions": [],
            "lifestyle_advice": [],
            "red_flags": [],
        }

    return parsed


def build_complex_symptom_reply(analysis, user_input):
    """构建复杂症状的回复（不推药，引导就医+日常建议）"""
    symptoms = "、".join(analysis.get("symptoms", []))
    directions = analysis.get("possible_directions", [])
    lifestyle = analysis.get("lifestyle_advice", [])
    red_flags = analysis.get("red_flags", [])
    advice = analysis.get("advice", "")
    urgency = analysis.get("urgency", "建议近期就医")

    # 症状分析
    reply = f"您好，根据您描述的情况（{symptoms}），以下是一些分析和建议：\n\n"

    # 可能方向（不是诊断）
    if directions:
        reply += "【可能涉及的方向】\n"
        reply += "（以下仅为参考，不是诊断，具体需要医生评估）\n"
        for d in directions:
            direction = d.get("direction", "")
            dept = d.get("department", "")
            reply += f"• {direction}，建议就诊科室：{dept}\n"
        reply += "\n"

    # 日常建议
    if lifestyle:
        reply += "【日常建议】\n"
        for ls in lifestyle:
            reply += f"• {ls}\n"
        reply += "\n"

    # 建议
    if advice:
        reply += f"【建议】\n{advice}\n\n"

    # 警示信号
    if red_flags:
        reply += "【需要留意的情况】\n"
        for rf in red_flags:
            reply += f"• {rf}\n"
        reply += "\n"

    # 不推药说明
    if not analysis.get("recommend_otc", False):
        reply += "【关于用药】\n"
        reply += "您的症状不建议自行服用药物。原因：\n"
        reply += "• 症状可能涉及多种原因，需医生诊断后对症治疗\n"
        reply += "• 自行用药可能掩盖真实病情\n"
        reply += "• 部分相关药品属于处方药，需凭处方购买\n\n"

    # 免责
    reply += "---\n"
    reply += "【温馨提示】\n"
    reply += "• 以上分析仅供参考，不能替代医生诊断\n"
    reply += f"• {urgency}\n"
    reply += "• 建议前往医院相关科室就诊\n"

    return reply


def handle_symptom_v2(user_input):
    """
    升级版症状处理：
    先判断简单还是复杂
    - 简单 → 走原来的症状→药品桥梁
    - 复杂 → 不推药，给方向+日常建议+引导就医
    """
    print(f"\n{'='*60}")
    print(f"用户输入: {user_input}")
    print(f"{'='*60}")

    # 分析症状类型
    print(f"\n[分析症状类型]...", flush=True)
    t0 = time.time()
    analysis = analyze_complex_symptom(user_input)
    t1 = time.time()

    symptom_type = analysis.get("symptom_type", "complex_need_doctor")
    print(f"  类型: {symptom_type} (耗时{t1-t0:.1f}s)")
    print(f"  症状: {analysis.get('symptoms', [])}")
    print(f"  推荐OTC: {analysis.get('recommend_otc', False)}")
    print(f"  需就医: {analysis.get('needs_doctor', True)}")

    if symptom_type == "complex_need_doctor":
        # 复杂症状：不推药，给方向+建议
        print(f"  → 复杂症状，不推药，引导就医")
        reply = build_complex_symptom_reply(analysis, user_input)
        print(f"\n{'─'*60}")
        print(f"【回复】")
        print(f"{'─'*60}")
        print(reply)
        return reply
    else:
        # 简单症状：走原来的流程（症状→药品桥梁）
        print(f"  → 简单症状，可推OTC药")
        # 这里接原来的16_symptom_handler流程
        return "（走症状→药品桥梁流程，见16_symptom_handler.py）"


def demo():
    print("=" * 60)
    print("  慢性/复杂症状处理演示")
    print("=" * 60)

    tests = [
        # 1. 复杂症状（多系统，不该推药）
        "我最近睡觉总是失眠，手脚发麻，这是什么情况？",
        # 2. 慢性症状
        "最近三个月总是胃不舒服，吃完饭就胀，有时候还反酸",
        # 3. 简单症状（可以推OTC）
        "我有点鼻塞流鼻涕，吃什么感冒药好？",
        # 4. 复杂症状（可能严重）
        "最近经常头晕，有时候眼前发黑，还耳鸣",
    ]

    for i, test in enumerate(tests):
        print(f"\n\n{'#'*60}")
        print(f"# 测试{i+1}: {test[:40]}")
        print(f"{'#'*60}")
        try:
            handle_symptom_v2(test)
        except Exception as e:
            print(f"错误: {e}")
            if "429" in str(e):
                print("API限流，等待60秒...")
                time.sleep(60)
                handle_symptom_v2(test)
        if i < len(tests) - 1:
            time.sleep(5)

    print(f"\n{'='*60}")
    print(f"复杂症状处理策略总结")
    print(f"{'='*60}")
    print(f"""
【两类症状，两种处理】

  简单症状（simple_otc）
    "头痛发烧流鼻涕" → 有明确OTC药对应
    → 走症状→药品桥梁：提取药品→RAG查用法→推荐OTC+建议就医

  复杂症状（complex_need_doctor）
    "失眠+手脚发麻" → 病因不明，多系统，不适合自行用药
    → 不推药：给可能方向（不是诊断）+ 建议就诊科室 + 日常建议 + 引导就医

【复杂症状的回复结构】
  1. 症状分析（"可能涉及的方向"，不是"你得了什么病"）
  2. 建议就诊科室（神经内科/骨科/消化科等）
  3. 日常建议（作息/饮食/运动等）
  4. 不推药说明（为什么不建议自行用药）
  5. 警示信号（什么情况要紧急就医）
  6. 免责声明

【关键区别】
  简单症状：推OTC药 + 建议就医
  复杂症状：不推药 + 给方向 + 引导就医 + 日常建议

【为什么不推药】
  • 失眠+手脚发麻可能是：颈椎病/神经病变/焦虑/糖尿病/多种原因
  • 没有OTC药能"对症"（安眠药是处方药，手脚发麻要查病因）
  • 自行用药会掩盖真实病情
""")


if __name__ == "__main__":
    demo()
