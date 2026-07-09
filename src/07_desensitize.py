"""
07_desensitize.py
信息脱敏 demo：演示如何在外部API调用前保护用户隐私

脱敏策略（医药咨询场景）：
- 手机号、身份证、地址 → 替换占位符（与用药无关，必须脱敏）
- 姓名 → 替换占位符（不影响用药建议）
- 年龄、性别、病史、用药情况 → 保留（这些是用药咨询的必要信息）

对比演示：
  1. 原始问题（含隐私）→ 直接传API（危险）
  2. 脱敏后问题（去隐私）→ 传API（安全）
  3. 对比答案质量是否受影响
"""
import os
import sys
import re
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K


# ============================================================
# 第一部分：脱敏规则引擎
# ============================================================

# 脱敏规则：正则匹配 → 替换占位符
DESENSITIZE_RULES = [
    # 手机号：11位数字，1开头
    (re.compile(r'1[3-9]\d{9}'), '[手机号]'),
    # 身份证：18位，最后一位可能是X
    (re.compile(r'\d{17}[\dXx]'), '[身份证]'),
    # 邮箱
    (re.compile(r'[\w.-]+@[\w.-]+\.\w+'), '[邮箱]'),
    # 银行卡：16-19位连续数字
    (re.compile(r'\d{16,19}'), '[卡号]'),
    # 地址：包含"路""街""号""楼""室""小区""栋""单元"的关键词片段
    (re.compile(r'[\u4e00-\u9fa5]{2,}(?:省|市|区|县|路|街|号|楼|室|小区|栋|单元|村|镇)[\d\u4e00-\u9fa5]*'), '[地址]'),
    # QQ号
    (re.compile(r'QQ[：:]?\s*\d{5,12}', re.IGNORECASE), 'QQ[账号]'),
    # 微信号
    (re.compile(r'微信[：:]?\s*[\w-]{6,20}'), '微信[账号]'),
]

# 姓名脱敏：常见称呼 + 姓名模式
NAME_PATTERNS = [
    (re.compile(r'(?:我叫|我是|本人|患者[：:]|姓名[：:])([\u4e00-\u9fa5]{2,4})'), '我叫[姓名]'),
    (re.compile(r'(?:先生|女士|师傅|老板)([\u4e00-\u9fa5]{2,4})'), '[姓名]'),
]


def desensitize(text):
    """
    对文本进行脱敏处理
    返回: (脱敏后文本, 脱敏记录列表)
    """
    original = text
    masked = text
    records = []

    # 应用通用规则
    for pattern, replacement in DESENSITIZE_RULES:
        matches = pattern.findall(masked)
        if matches:
            for m in matches:
                if isinstance(m, str) and len(m) > 2:
                    records.append({
                        "type": replacement,
                        "original": m[:3] + "***",
                        "action": "已替换为" + replacement,
                    })
            masked = pattern.sub(replacement, masked)

    # 姓名脱敏（保守，只匹配明确自称模式）
    for pattern, replacement in NAME_PATTERNS:
        masked = pattern.sub(replacement, masked)

    # 记录变更
    if masked != original:
        masked_count = len(records)
    else:
        masked_count = 0

    return masked, records, masked_count


# ============================================================
# 第二部分：调用外部API（模拟第三方看到的内容）
# ============================================================

def call_external_api(question_with_context):
    """调用外部 LLM API（模拟第三方服务器收到的内容）"""
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60)

    prompt = f"""你是专业药学咨询助手。请根据用户的描述给出用药建议。
回答要包含：1.可能的用药建议 2.注意事项 3.必须咨询医生的提示

用户描述：{question_with_context}

请回答："""

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是专业药学咨询助手，基于用户描述给出用药建议。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    chunks.append(delta.content)
        return "".join(chunks).strip()
    finally:
        client.close()


# ============================================================
# 第三部分：对比演示
# ============================================================

def demo_desensitize():
    print("=" * 70)
    print("  信息脱敏 Demo：外部API调用如何保护用户隐私")
    print("=" * 70)

    # 模拟真实用户问题（含各种隐私）
    test_cases = [
        {
            "label": "场景1：手机号+病史咨询",
            "raw": "我叫张明华，手机13812345678，今年45岁，男性，有糖尿病史5年，最近感冒了能吃什么感冒药？",
        },
        {
            "label": "场景2：身份证+地址+用药咨询",
            "raw": "我的身份证号是522121199003151234，住在贵阳市南明区花果园大街1号，我怀孕3个月了，请问布洛芬缓释胶囊孕妇能吃吗？",
        },
        {
            "label": "场景3：微信号+儿童用药",
            "raw": "微信zhangsan2024，我儿子8岁，体重30公斤，发烧38.5度，对乙酰氨基酚片怎么吃？",
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'█'*70}")
        print(f"█  {case['label']}")
        print(f"{'█'*70}")

        # 1. 原始问题
        print(f"\n【1. 用户原始输入】（含隐私信息）")
        print(f"  {case['raw']}")

        # 2. 脱敏处理
        masked, records, count = desensitize(case['raw'])
        print(f"\n【2. 脱敏处理后】（发送给外部API的内容）")
        print(f"  {masked}")
        print(f"  脱敏项: {count} 处")
        for r in records:
            print(f"    - {r['original']} → {r['action']}")

        # 3. 危险对比：不脱敏直接传
        print(f"\n【3. ⚠️ 如果不脱敏，外部API会看到】")
        print(f"  {case['raw']}")
        print(f"  泄露: 姓名/手机号/身份证/地址等身份识别信息")

        # 4. 调用API（用脱敏后的内容）
        print(f"\n【4. 调用外部API（传入脱敏后内容）】")
        print(f"  发送: {masked[:80]}...")
        t0 = time.time()
        answer = call_external_api(masked)
        t1 = time.time()
        print(f"  耗时: {t1-t0:.1f}s")
        print(f"\n【5. API返回的答案】")
        print(f"  {answer}")

        # 5. 关键分析
        print(f"\n【6. 关键分析】")
        print(f"  ✓ 外部API只看到脱敏后的内容，不知道用户是谁")
        print(f"  ✓ 但年龄/性别/病史/用药情况都保留了，答案质量不受影响")
        print(f"  ✓ 即使API提供商记录了对话，也无法追溯到具体用户")
        print()

    # 汇总
    print(f"\n{'='*70}")
    print(f"  脱敏策略总结")
    print(f"{'='*70}")
    print(f"""
【必须脱敏】（身份识别信息，与用药无关）
  - 手机号    → [手机号]
  - 身份证    → [身份证]
  - 地址      → [地址]
  - 姓名      → [姓名]
  - 邮箱/QQ/微信 → [账号]
  - 银行卡    → [卡号]

【必须保留】（用药咨询的必要信息）
  - 年龄      → 影响剂量（儿童/成人/老人用药不同）
  - 性别      → 影响用药（孕妇/哺乳期禁用某些药）
  - 体重      → 影响剂量（mg/kg计算）
  - 病史      → 影响用药禁忌（糖尿病/高血压禁用某些药）
  - 当前用药  → 影响药物相互作用判断
  - 症状      → 影响用药建议

【核心原则】
  外部API需要知道"什么情况、用什么药"，不需要知道"你是谁"。
  脱敏去掉"你是谁"，保留"什么情况"，答案质量不受影响。
""")


if __name__ == "__main__":
    demo_desensitize()
