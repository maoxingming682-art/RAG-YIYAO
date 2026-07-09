"""
08_prepare_finetune_data.py  (升级版)
改造训练数据：4种回答风格 + 超纲兜底 + 错字纠错 + 闲聊
解决微调后回答机械化问题
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


def style_dosage(question, raw_answer):
    """风格1：用法用量 - 简洁直接"""
    return f"""{raw_answer}

温馨提示：具体用药请遵医嘱，如不确定请查看药品说明书或咨询药师。"""


def style_adverse(question, raw_answer):
    """风格2：不良反应 - 列举+提醒"""
    return f"""该药品的不良反应如下：
{raw_answer}

如出现严重不良反应请立即停药并就医。用药前请咨询医生或药师。"""


def style_contraindication(question, raw_answer):
    """风格3：禁忌/孕妇/儿童 - 重点警告"""
    return f"""⚠️ 用药安全提醒：
{raw_answer}

请务必在医生指导下使用，切勿自行用药。"""


def style_general(question, raw_answer):
    """风格4：一般咨询 - 自然对话"""
    return f"""{raw_answer}

希望对您有帮助。如有其他用药问题随时问我，也建议咨询您的药师或医生。"""


# 超纲兜底样例
OUT_OF_SCOPE_SAMPLES = [
    {"instruction": "阿莫西林胶囊怎么吃？", "input": "",
     "output": "抱歉，我目前的知识库中暂未收录阿莫西林胶囊的详细信息。建议您：\n1. 查看药品随附的说明书\n2. 咨询执业药师或医生\n\n我会持续更新药品知识，为您提供更好的服务。"},
    {"instruction": "头孢克肟片的用法用量", "input": "",
     "output": "我的药品知识库中暂未收录头孢克肟片的相关信息。建议您查看药品说明书或咨询医生获取准确用药指导。"},
    {"instruction": "伟哥怎么买", "input": "",
     "output": "这个问题涉及处方药购买，我无法提供购买建议。请前往正规医疗机构就诊，凭医生处方在正规药店购买。"},
    {"instruction": "安眠药吃多少能睡着", "input": "",
     "output": "安眠药属于处方药，必须严格遵医嘱使用，切勿自行调整剂量。过量服用可能造成严重健康风险。请立即咨询医生获取专业指导。"},
    {"instruction": "那个药怎么吃", "input": "",
     "output": "请问您指的是哪个药品呢？请告诉我具体的药品名称，我可以为您查询相关用药信息。"},
    {"instruction": "感冒了吃什么药好", "input": "",
     "output": "感冒用药需要根据具体症状来选择。请问您目前有哪些症状（如发烧、咳嗽、鼻塞等）？是否有其他慢性疾病或过敏史？建议咨询医生或药师获取适合您的用药方案。"},
    {"instruction": "降压药能和感冒药一起吃吗", "input": "",
     "output": "降压药与感冒药的相互作用需要具体看是哪种降压药和哪种感冒药。部分感冒药中的伪麻黄碱可能影响血压控制。建议您将正在服用的降压药名称告诉我，或直接咨询您的医生。"},
    {"instruction": "这个药过期了还能吃吗", "input": "",
     "output": "药品过期后不建议服用。过期药品的有效成分可能降解，疗效降低，还可能产生有害物质。请妥善处理过期药品，如需继续用药请购买新批次。"},
]

# 闲聊/非用药问题样例
CHITCHAT_SAMPLES = [
    {"instruction": "你好", "input": "",
     "output": "您好！我是药学咨询助手，有什么用药问题可以帮您吗？"},
    {"instruction": "谢谢", "input": "",
     "output": "不客气！如果还有其他用药问题随时问我。祝您健康！"},
    {"instruction": "你是谁", "input": "",
     "output": "我是专业药学咨询助手，可以为您解答药品用法用量、不良反应、用药禁忌等问题。请问有什么可以帮您的？"},
    {"instruction": "今天天气怎么样", "input": "",
     "output": "我是药学咨询助手，只能回答用药相关的问题哦。请问有什么药品或用药方面的问题需要咨询？"},
    {"instruction": "帮我推荐个感冒药", "input": "",
     "output": "我可以为您提供药品的用法用量等信息，但具体的药品推荐需要根据您的症状和身体状况来判断。建议您咨询医生或药师获取个性化的用药建议。"},
]

# 错字纠错样例
TYPO_CORRECTION_SAMPLES = [
    {"instruction": "阿斯匹林肠溶胶囊的用法用量", "input": "",
     "output": "您提到的\"阿斯匹林\"可能是\"阿司匹林\"（Aspirin）。以下是阿司匹林肠溶胶囊的用法用量：\n用于抗血栓形成应用小剂量，每日75～300mg（1～4粒），每日1次。\n\n温馨提示：具体用药请遵医嘱。"},
    {"instruction": "布洛分缓释胶囊怎么吃", "input": "",
     "output": "您提到的\"布洛分\"可能是\"布洛芬\"（Ibuprofen）。布洛芬缓释胶囊通常饭后服用，具体剂量请遵照药品说明书或医嘱。\n\n如不确定请咨询药师或医生。"},
    {"instruction": "阿莫西啉胶囊用法用量", "input": "",
     "output": "您提到的\"阿莫西啉\"可能是\"阿莫西林\"（Amoxicillin）。我的知识库中暂未收录该药品的详细信息，建议您查看药品说明书或咨询药师获取准确用法。"},
]


def classify_question_style(question):
    """根据问题类型选择回答风格"""
    q = question
    if any(kw in q for kw in ["禁忌", "孕妇", "哺乳", "儿童用药", "老年用药", "不能用", "不能用"]):
        return "contraindication"
    if any(kw in q for kw in ["不良反应", "副作用", "有什么反应", "不适"]):
        return "adverse"
    if any(kw in q for kw in ["用法用量", "用法", "用量", "怎么服用", "怎么吃", "怎么用", "服用"]):
        return "dosage"
    return "general"


def format_answer(question, raw_answer):
    """根据问题类型用不同风格格式化答案"""
    style = classify_question_style(question)
    if style == "dosage":
        return style_dosage(question, raw_answer)
    elif style == "adverse":
        return style_adverse(question, raw_answer)
    elif style == "contraindication":
        return style_contraindication(question, raw_answer)
    else:
        return style_general(question, raw_answer)


def prepare_finetune_data():
    print("=" * 60)
    print("第2步：准备多样化微调训练数据")
    print("=" * 60)

    knowledge_path = os.path.join(DATA_DIR, "drug_knowledge.json")
    with open(knowledge_path, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
    print(f"加载 {len(knowledge)} 条药品知识")

    # 筛选适合微调的条目
    style_keywords = ["用法用量", "用法", "用量", "怎么服用", "怎么吃",
                      "不良反应", "副作用", "注意事项", "禁忌",
                      "孕妇", "儿童", "老人"]
    candidates = [k for k in knowledge
                  if any(kw in k["question"] for kw in style_keywords)
                  and 30 < len(k["answer"]) < 500]

    # 按类型分组确保多样性
    dosage_items = [k for k in candidates if classify_question_style(k["question"]) == "dosage"]
    adverse_items = [k for k in candidates if classify_question_style(k["question"]) == "adverse"]
    contra_items = [k for k in candidates if classify_question_style(k["question"]) == "contraindication"]
    general_items = [k for k in candidates if classify_question_style(k["question"]) == "general"]

    print(f"  用法用量类: {len(dosage_items)} 条")
    print(f"  不良反应类: {len(adverse_items)} 条")
    print(f"  禁忌人群类: {len(contra_items)} 条")
    print(f"  一般咨询类: {len(general_items)} 条")

    # 各类型取样，保证均衡
    random.seed(42)
    sampled = []
    for items, n in [(dosage_items, 150), (adverse_items, 50),
                     (contra_items, 50), (general_items, 50)]:
        random.shuffle(items)
        sampled.extend(items[:n])

    print(f"  采样总计: {len(sampled)} 条")

    # 生成训练数据
    train_data = []
    for item in sampled:
        train_data.append({
            "instruction": item["question"],
            "input": "",
            "output": format_answer(item["question"], item["answer"]),
            "style": classify_question_style(item["question"]),
        })

    # 加入超纲兜底样例
    for s in OUT_OF_SCOPE_SAMPLES:
        s["style"] = "out_of_scope"
        train_data.append(s)

    # 加入闲聊样例
    for s in CHITCHAT_SAMPLES:
        s["style"] = "chitchat"
        train_data.append(s)

    # 加入错字纠错样例
    for s in TYPO_CORRECTION_SAMPLES:
        s["style"] = "typo_correction"
        train_data.append(s)

    # 打乱顺序
    random.shuffle(train_data)

    # 划分训练集和测试集
    split = int(len(train_data) * 0.9)
    train_set = train_data[:split]
    test_set = train_data[split:]

    # 保存
    train_path = os.path.join(DATA_DIR, "finetune_train.json")
    test_path = os.path.join(DATA_DIR, "finetune_test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    # 统计各风格分布
    style_count = {}
    for d in train_set:
        s = d.get("style", "unknown")
        style_count[s] = style_count.get(s, 0) + 1

    print(f"\n微调数据生成完成:")
    print(f"  训练集: {len(train_set)} 条")
    print(f"  测试集: {len(test_set)} 条")
    print(f"  风格分布: {style_count}")

    print(f"\n各风格样例:")
    shown = set()
    for d in train_set:
        s = d.get("style", "unknown")
        if s not in shown:
            shown.add(s)
            print(f"\n  [{s}] Q: {d['instruction']}")
            print(f"       A: {d['output'][:120]}...")

    print(f"\n【关键改进】")
    print(f"  1. 4种回答风格：用法用量简洁/不良反应列举/禁忌警告/一般自然")
    print(f"  2. 超纲兜底：{len(OUT_OF_SCOPE_SAMPLES)}条（库没有→建议咨询）")
    print(f"  3. 闲聊变通：{len(CHITCHAT_SAMPLES)}条（非用药→正常对话）")
    print(f"  4. 错字纠错：{len(TYPO_CORRECTION_SAMPLES)}条（提示正确药名）")

    return train_set, test_set


if __name__ == "__main__":
    train, test = prepare_finetune_data()
    print(f"\n第2步完成")
