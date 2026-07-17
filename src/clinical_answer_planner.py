"""
结构化症状回答规划器。

它把 data/science_kb 里的结构化科普卡组织成“鉴别分析式回答”：
- 先说明边界；
- 再列可能方向和更像/不像的线索；
- 给自查问题、检查建议、居家处理和用药边界；
- 保留来源编号，后续可升级成真正的科普库检索。
"""
import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCIENCE_KB_DIR = os.path.join(BASE_DIR, "data", "science_kb")

FOOT_ITCH_MARKERS = ["脚痒", "足痒", "脚趾缝痒", "脚底痒", "脚气", "足癣"]
DIARRHEA_MARKERS = ["腹泻", "拉肚子", "肚子拉", "稀便", "水样便", "闹肚子", "急性胃肠炎", "肠胃炎", "胃肠炎"]
ANALYSIS_MARKERS = ["分析", "状况", "情况", "原因", "没分析", "不分析", "是不是", "会不会", "为什么"]


def _load_cards():
    cards = []
    if not os.path.isdir(SCIENCE_KB_DIR):
        return cards
    for name in os.listdir(SCIENCE_KB_DIR):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(SCIENCE_KB_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cards.append(json.loads(line))
    return cards


_CARDS = None


def get_cards():
    global _CARDS
    if _CARDS is None:
        _CARDS = _load_cards()
    return _CARDS


def _history_questions(history):
    if not isinstance(history, list):
        return []
    questions = []
    for item in history[-5:]:
        if isinstance(item, dict):
            q = str(item.get("question") or "").strip()
            if q:
                questions.append(q)
    return questions


def _combined_text(question, history=None):
    parts = [str(question or "")]
    parts.extend(_history_questions(history))
    return " ".join(part for part in parts if part).strip()


def _has_any(text, markers):
    return any(marker in text for marker in markers)


def is_foot_itch_context(question, history=None):
    text = _combined_text(question, history)
    return _has_any(text, FOOT_ITCH_MARKERS) or ("脚" in text and "痒" in text)


def is_diarrhea_context(question, history=None):
    text = _combined_text(question, history)
    return _has_any(text, DIARRHEA_MARKERS) or ("拉" in text and "肚子" in text)


def is_analysis_request(question):
    q = str(question or "")
    return _has_any(q, ANALYSIS_MARKERS)


def get_foot_itch_card():
    for card in get_cards():
        if card.get("topic") == "脚痒":
            return card
    return None


def get_diarrhea_card():
    for card in get_cards():
        if card.get("topic") == "腹泻":
            return card
    return None


def _source_ref(index):
    return f"[{index}]"


def _source_refs(start, count):
    return "".join(_source_ref(i) for i in range(start, start + count))


def _missing_questions(card, text, limit=4):
    missing = []
    for slot in card.get("ask_slots", []):
        markers = slot.get("markers") or []
        if not _has_any(text, markers):
            missing.append(slot.get("question", ""))
    return [item for item in missing if item][:limit]


def _source_lines(card):
    lines = []
    for idx, source in enumerate(card.get("sources", []), 1):
        title = source.get("title", "本地审核资料")
        source_type = source.get("type", "本地资料")
        lines.append(f"{idx}. {title}（{source_type}）")
    return lines


def _format_list(items, start=1):
    return [f"{idx}. {item}" for idx, item in enumerate(items, start)]


def build_foot_itch_professional_reply(question, history=None, mode="intake", disclaimer=""):
    card = get_foot_itch_card()
    if not card:
        return None

    text = _combined_text(question, history)
    more_like = card.get("more_like", {})
    tinea = more_like.get("足癣/脚气", [])
    eczema = more_like.get("湿疹/汗疱疹/接触性皮炎", [])
    missing = _missing_questions(card, text, limit=5)
    analysis_mode = mode == "analysis" or is_analysis_request(question)

    lines = []
    if analysis_mode:
        if "是不是" in str(question or "") or "会不会" in str(question or ""):
            lines.append("有这个可能，但仅凭“痒”还不能下结论。")
        else:
            lines.append("你提醒得对，这类问题要先做状况分析，不能只给一句处理建议。")
    else:
        lines.append("脚痒确实常见于脚气，也就是足癣，但仅凭“痒”还不能下结论。")
    causes = "、".join(card.get("possible_causes", [])[:6])
    lines.append(f"常见方向包括：{causes}。需要结合部位、皮损形态、诱因、是否反复和危险信号判断。{_source_ref(1)}")

    lines.extend([
        "",
        "先按线索分几类看：",
        "",
        "更像足癣/脚气的线索：",
    ])
    lines.extend(_format_list(tinea[:5]))

    lines.extend([
        "",
        "更像湿疹、汗疱疹或接触性皮炎的线索：",
    ])
    lines.extend(_format_list(eczema[:5]))

    lines.extend([
        "",
        f"如果两类表现重叠，或者准备用激素类药膏前，建议优先做真菌镜检来明确方向{_source_ref(2)}。",
        "检查思路：",
    ])
    lines.extend(_format_list(card.get("tests", [])[:3]))

    if missing:
        lines.extend(["", "你可以先自查并补充这几项："])
        lines.extend(_format_list(missing))

    lines.extend([
        "",
        "现在可以先做的事：",
    ])
    lines.extend(_format_list(card.get("home_care", [])[:4]))

    lines.extend([
        "",
        f"用药边界：{card.get('drug_boundary', '')}{_source_ref(3)}",
        "",
        "需要尽快就医的情况：",
    ])
    lines.extend(_format_list(card.get("red_flags", [])[:6]))

    source_lines = _source_lines(card)
    if source_lines:
        lines.extend(["", "本地审核资料："])
        lines.extend(source_lines)

    if disclaimer:
        lines.append(disclaimer)
    return "\n".join(line for line in lines if line is not None)


def build_diarrhea_professional_reply(question, history=None, mode="intake", level="", observed_summary="", disclaimer=""):
    card = get_diarrhea_card()
    if not card:
        return None

    text = _combined_text(question, history)
    more_like = card.get("more_like", {})
    mild = more_like.get("饮食刺激/轻症胃肠不适", [])
    infection = more_like.get("急性胃肠炎/感染性腹泻", [])
    dehydration = more_like.get("脱水或中高风险", [])
    missing = _missing_questions(card, text, limit=5)
    analysis_mode = mode == "analysis" or is_analysis_request(question)

    lines = []
    if analysis_mode:
        if "是不是" in str(question or "") or "会不会" in str(question or ""):
            lines.append("有这个可能，但不能只凭“拉肚子”判断就是急性胃肠炎。")
        else:
            lines.append("腹泻要先做状况分析，不能直接跳到“吃什么药”。")
    else:
        lines.append("腹泻很常见，但不能只凭“拉肚子”判断原因。")

    if level:
        lines.append(f"当前更偏向：{level}。")
    if observed_summary:
        lines.append(observed_summary)

    if _has_any(text, ["血便", "便血", "黑便", "高热", "剧烈腹痛", "持续加重腹痛", "反复呕吐", "尿少", "头晕", "喝不进水"]):
        lines.append("你描述里有需要提高警惕的信号，重点先看脱水、感染和是否需要线下检查。")

    causes = "、".join(card.get("possible_causes", [])[:7])
    lines.append(f"常见方向包括：{causes}。要结合次数、持续时间、大便形态、发热腹痛、呕吐、脱水表现和特殊人群风险判断。{_source_ref(1)}")

    lines.extend([
        "",
        "先按线索分几类看：",
        "",
        "更像饮食刺激或轻症胃肠不适的线索：",
    ])
    lines.extend(_format_list(mild[:5]))

    lines.extend([
        "",
        "更像急性胃肠炎或感染性腹泻的线索：",
    ])
    lines.extend(_format_list(infection[:5]))

    lines.extend([
        "",
        "提示脱水或中高风险的线索：",
    ])
    lines.extend(_format_list(dehydration[:5]))

    lines.extend([
        "",
        f"检查思路：轻症、短时间、没有危险信号时通常先补液观察；如果有血便或黑便、高热、明显腹痛、反复呕吐或脱水表现，建议就医评估，必要时做大便常规/培养、电解质等检查{_source_ref(2)}。",
    ])
    lines.extend(_format_list(card.get("tests", [])[:4]))

    if missing:
        lines.extend(["", "你可以先自查并补充这几项："])
        lines.extend(_format_list(missing))

    lines.extend([
        "",
        "现在可以先做的事：",
    ])
    lines.extend(_format_list(card.get("home_care", [])[:5]))

    lines.extend([
        "",
        f"用药边界：{card.get('drug_boundary', '')}{_source_ref(3)}",
        "",
        "需要尽快就医的情况：",
    ])
    lines.extend(_format_list(card.get("red_flags", [])[:7]))

    source_lines = _source_lines(card)
    if source_lines:
        lines.extend(["", "本地审核资料："])
        lines.extend(source_lines)

    if disclaimer:
        lines.append(disclaimer)
    return "\n".join(line for line in lines if line is not None)
