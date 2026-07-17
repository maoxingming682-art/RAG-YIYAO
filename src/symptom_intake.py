"""
问诊式症状入口。

这个模块只处理“用户在描述症状/不适”的第一层分流：
- 首轮信息不足时，追问关键槽位；
- 多轮信息已有时，给阶段性风险判断；
- 不直接给处方药或剂量建议。
"""
import re

try:
    from clinical_answer_planner import (
        build_diarrhea_professional_reply,
        build_foot_itch_professional_reply,
        is_diarrhea_context,
        is_foot_itch_context,
    )
except Exception:
    build_diarrhea_professional_reply = None
    build_foot_itch_professional_reply = None

    def is_diarrhea_context(question, history=None):
        return False

    def is_foot_itch_context(question, history=None):
        return False


NEGATION_WORDS = ["没有", "没", "无", "未", "不", "否认"]
UNKNOWN_WORDS = ["不知道", "不清楚", "没测", "没量", "未知", "说不准"]

DRUG_HINTS = [
    "左氧氟沙星", "氧氟沙星", "阿莫西林", "头孢", "罗红霉素", "阿奇霉素",
    "布洛芬", "对乙酰氨基酚", "蒙脱石散", "诺氟沙星", "黄连素", "人工泪液",
]
DRUG_FORM_WORDS = [
    "片", "胶囊", "颗粒", "口服液", "注射液", "滴眼液", "眼液", "眼药水",
    "乳膏", "软膏", "喷雾", "糖浆", "散", "丸", "栓", "凝胶",
]


def default_disclaimer(kind="symptom"):
    if kind == "symptom":
        return "\n\n提示：以上只能作为健康咨询和风险分层参考，不能替代医生面诊；症状加重或拿不准时，优先就医或咨询药师。"
    return "\n\n提示：具体用药请以说明书、医生或药师建议为准。"


TOPICS = {
    "腹泻": {
        "priority": 10,
        "markers": ["拉肚子", "腹泻", "肚子拉", "水样便", "稀便", "闹肚子"],
        "stage_groups": [
            ["腹泻", "拉肚子", "肚子拉", "水样便", "稀便", "闹肚子"],
            ["发热", "发烧", "高热", "低热", "体温"],
            ["腹痛", "肚子痛", "肚子疼", "胃痛", "胃疼"],
            ["呕吐", "恶心"],
            ["口干", "尿少", "头晕", "乏力", "脱水"],
            ["血便", "黑便", "黏液", "脓血"],
            ["冰", "西瓜", "生冷", "外卖", "海鲜", "隔夜"],
            ["孕妇", "儿童", "小孩", "老人", "慢性病", "基础病"],
        ],
        "intro": "腹泻先别急着直接定原因，要先看次数、持续时间和有没有脱水或感染风险。",
        "questions": [
            ("frequency", ["几次", "多少次", "次", "趟", "水样", "稀便", "成形"], "今天大概拉了几次？大便是水样、稀便，还是有黏液、血丝或黑便？"),
            ("duration", ["今天", "一天", "半天", "小时", "昨天", "持续", "开始"], "从什么时候开始的？有没有吃生冷、隔夜、外卖或海鲜后出现？"),
            ("symptoms", ["发烧", "发热", "腹痛", "肚子痛", "肚子疼", "呕吐", "恶心"], "有没有发热、明显腹痛、恶心呕吐？腹痛是阵发性还是持续加重？"),
            ("dehydration", ["口干", "尿少", "头晕", "乏力", "脱水", "喝不下"], "有没有口干、尿少、头晕、明显乏力这些脱水表现？"),
            ("risk", ["成人", "儿童", "小孩", "老人", "孕妇", "慢性病", "基础病"], "你是成人吗？有没有孕期、老人儿童、慢性病或正在用药的情况？"),
        ],
        "interim": "先少量多次补水；如果家里有口服补液盐，可以按说明书冲服。饮食清淡，暂时避开冰冷、油腻、辛辣、酒精和奶制品。",
        "red_flags": "血便或黑便、高热、剧烈或持续加重腹痛、反复呕吐喝不进水、明显尿少头晕，都建议尽快就医。",
    },
    "腹痛": {
        "priority": 9,
        "markers": ["肚子痛", "腹痛", "胃痛", "肚子疼", "胃疼"],
        "stage_groups": [
            ["腹痛", "肚子痛", "肚子疼", "胃痛", "胃疼"],
            ["发热", "发烧", "体温"],
            ["呕吐", "恶心"],
            ["腹泻", "拉肚子", "便血", "黑便", "尿痛"],
            ["右下腹", "上腹", "下腹", "肚脐"],
            ["孕妇", "儿童", "小孩", "老人", "慢性病", "基础病"],
        ],
        "intro": "腹痛需要先确认部位、疼痛程度和伴随表现，不能只按一个症状下结论。",
        "questions": [
            ("location", ["上腹", "下腹", "左", "右", "肚脐", "胃"], "疼痛主要在哪个位置？上腹、下腹、右下腹，还是肚脐周围？"),
            ("severity", ["轻微", "剧烈", "阵发", "持续", "越来越痛"], "疼痛程度如何？是阵发性，还是持续加重？"),
            ("symptoms", ["发烧", "发热", "呕吐", "腹泻", "便血", "黑便", "尿痛"], "有没有发热、呕吐、腹泻、便血黑便或尿痛？"),
            ("risk", ["儿童", "小孩", "老人", "孕妇", "慢性病", "基础病"], "你是成人吗？有没有孕期、老人儿童、慢性病或正在用药？"),
        ],
        "interim": "先暂时避免油腻辛辣和饮酒，不要盲目吃止痛药掩盖病情。",
        "red_flags": "腹痛剧烈或持续加重、右下腹明显疼痛、伴高热/呕吐/便血黑便，建议尽快就医。",
    },
    "消化不适": {
        "priority": 8,
        "markers": ["胃口不好", "食欲不振", "腹胀", "肚子胀", "胃胀", "反酸", "嗳气"],
        "stage_groups": [
            ["胃口不好", "食欲不振", "腹胀", "肚子胀", "胃胀", "反酸", "嗳气"],
            ["腹痛", "肚子痛", "肚子疼", "胃痛", "胃疼"],
            ["恶心", "呕吐", "腹泻", "发热", "发烧"],
            ["黑便", "血便", "呕血", "体重下降", "消瘦", "吞咽困难"],
            ["孕妇", "儿童", "小孩", "老人", "慢性病", "基础病"],
        ],
        "intro": "胃口不好、腹胀这类消化不适，要先看持续时间、伴随症状和有没有危险信号。",
        "questions": [
            ("duration", ["今天", "几天", "一周", "持续", "开始"], "这种情况持续多久了？是这两天刚出现，还是已经一周以上？"),
            ("location", ["上腹", "胃", "肚脐", "下腹", "右下腹"], "不舒服主要在胃部上腹，还是肚脐周围、下腹或右下腹？"),
            ("symptoms", ["腹痛", "胃痛", "反酸", "嗳气", "恶心", "呕吐", "腹泻", "发热"], "有没有腹痛、反酸嗳气、恶心呕吐、腹泻或发热？"),
            ("red_flag", ["体重", "消瘦", "黑便", "血便", "呕血", "吞咽困难"], "有没有体重下降、黑便血便、呕血或吞咽困难？"),
            ("risk", ["成人", "儿童", "小孩", "老人", "孕妇", "慢性病", "基础病", "用药"], "你是成人吗？有没有孕期、老人儿童、慢性病或正在用药的情况？"),
        ],
        "interim": "先少量多餐、饮食清淡，暂时少吃油腻、辛辣、酒精和过甜食物；不要自行叠加止痛药、抗生素或多种胃药。",
        "red_flags": "持续或加重腹痛、反复呕吐、发热、黑便血便、呕血、体重明显下降或吞咽困难，建议及时就医。",
    },
    "发热": {
        "priority": 7,
        "markers": ["发烧", "发热", "低烧", "高烧", "体温"],
        "stage_groups": [
            ["发热", "发烧", "低烧", "高烧", "体温"],
            ["咳嗽", "咳痰", "流鼻涕", "鼻塞", "喉咙痛", "咽痛"],
            ["腹泻", "拉肚子", "腹痛", "肚子痛", "肚子疼"],
            ["胸闷", "呼吸困难", "气促"],
            ["皮疹"],
            ["儿童", "小孩", "老人", "孕妇", "慢性病", "基础病"],
        ],
        "intro": "发热要先看体温范围、持续时间和伴随症状，再判断居家观察还是就医。",
        "questions": [
            ("temperature", ["度", "℃", "体温", "38", "39", "40"], "现在最高体温多少？是腋温、耳温还是额温？"),
            ("duration", ["今天", "昨天", "小时", "天", "持续", "反复"], "发热持续多久了？是一直烧，还是退了又升？"),
            ("symptoms", ["咳嗽", "喉咙痛", "咽痛", "流鼻涕", "腹泻", "头痛", "皮疹", "胸闷", "呼吸困难"], "有没有咳嗽、咽痛、流鼻涕、腹泻、皮疹、胸闷或呼吸困难？"),
            ("risk", ["儿童", "小孩", "老人", "孕妇", "慢性病", "基础病"], "是成人还是儿童/老人/孕妇？有没有慢性病或正在用药？"),
        ],
        "interim": "休息、补水、观察体温变化，避免自行叠加多种退烧药。",
        "red_flags": "体温接近或超过39℃、精神很差、呼吸困难、胸痛、意识异常、持续不退或反复加重，建议及时就医。",
    },
    "感冒": {
        "priority": 6,
        "markers": ["感冒", "鼻塞", "流鼻涕", "流涕", "咳嗽", "喉咙痛", "咽痛"],
        "stage_groups": [
            ["感冒", "鼻塞", "流鼻涕", "流涕", "咳嗽", "喉咙痛", "咽痛"],
            ["发热", "发烧", "体温"],
            ["黄痰", "胸闷", "气促", "呼吸困难"],
            ["儿童", "小孩", "老人", "孕妇", "慢性病", "基础病", "过敏"],
        ],
        "intro": "感冒样症状也要先分清主要表现和严重程度，尤其要排除高热、气促、基础病等风险。",
        "questions": [
            ("duration", ["今天", "昨天", "天", "小时", "持续"], "症状持续多久了？是刚开始，还是已经几天没有好转？"),
            ("temperature", ["发烧", "发热", "体温", "度", "℃"], "有没有发烧？最高体温多少？"),
            ("symptoms", ["咳嗽", "咳痰", "黄痰", "鼻塞", "流鼻涕", "流涕", "喉咙痛", "咽痛", "胸闷", "气促"], "主要是鼻塞流涕、喉咙痛、咳嗽，还是有黄痰、胸闷气促？"),
            ("risk", ["儿童", "小孩", "老人", "孕妇", "慢性病", "基础病", "过敏"], "你是成人吗？有没有孕期、老人儿童、慢性病、药物过敏或正在用药？"),
        ],
        "interim": "先休息、补水，观察体温和呼吸情况。不要自行使用抗生素。",
        "red_flags": "持续高热、呼吸困难、胸痛、明显乏力加重，或儿童/孕妇/老人/有慢性病人群症状明显，建议及时就医。",
    },
    "眼干": {
        "priority": 5,
        "markers": ["眼睛干", "眼干", "干眼", "干涩", "眼涩", "眼疲劳", "视疲劳"],
        "stage_groups": [
            ["眼睛干", "眼干", "干眼", "干涩", "眼涩", "眼疲劳", "视疲劳"],
            ["电脑", "手机", "屏幕", "熬夜", "空调", "隐形眼镜", "美瞳"],
            ["疼", "痛", "红", "畏光", "分泌物", "视力下降", "异物感"],
            ["儿童", "孕妇", "老人", "青光眼", "眼病", "手术"],
        ],
        "intro": "眼睛干涩常见，但要先区分普通视疲劳/干眼，还是有感染、角膜损伤等风险。",
        "questions": [
            ("duration", ["今天", "最近", "几天", "一周", "持续", "多久"], "干涩持续多久了？是看屏幕后加重，还是一天都明显？"),
            ("trigger", ["电脑", "手机", "屏幕", "熬夜", "空调", "隐形眼镜", "美瞳"], "最近是否长时间看电脑手机、熬夜、吹空调，或戴隐形眼镜/美瞳？"),
            ("red_flag", ["疼", "痛", "红", "畏光", "分泌物", "视力下降", "异物感"], "有没有眼痛、明显发红、畏光、分泌物增多、异物感或视力下降？"),
            ("risk", ["儿童", "孕妇", "老人", "青光眼", "眼病", "手术", "用药"], "有没有青光眼、眼部手术史、眼病，或正在使用其他眼药水？"),
        ],
        "interim": "先减少连续盯屏，主动眨眼，每20分钟看远处休息一下；避免揉眼，暂时少戴隐形眼镜。需要用眼药水时，优先核对说明书或问药师。",
        "red_flags": "如果有眼痛、明显红肿、畏光、分泌物多、视力下降，或戴隐形眼镜后明显不适，建议尽快看眼科。",
    },
    "皮肤过敏": {
        "priority": 5,
        "markers": ["过敏", "皮疹", "荨麻疹", "红疹", "瘙痒", "发痒", "起疹", "脚痒", "足痒", "痒"],
        "stage_groups": [
            ["过敏", "皮疹", "荨麻疹", "红疹", "瘙痒", "发痒", "起疹", "脚痒", "足痒", "痒"],
            ["脚趾缝", "脚底", "脱皮", "水疱", "水泡", "渗液", "脚气", "潮湿", "闷热"],
            ["吃了", "用药", "药", "海鲜", "接触", "化妆品", "新鞋", "袜子", "洗衣液"],
            ["呼吸困难", "胸闷", "喉咙紧", "嘴唇肿", "眼睑肿", "全身", "红肿热痛", "流脓", "发热"],
            ["儿童", "孕妇", "老人", "哮喘", "糖尿病", "慢性病"],
        ],
        "intro": "皮肤瘙痒要先看部位、皮疹形态、诱因和有没有感染或严重过敏信号，不能只按“过敏”处理。",
        "questions": [
            ("location", ["脚趾缝", "脚底", "脚背", "脚踝", "手", "腿", "胳膊", "脸", "全身", "局部"], "痒主要在哪个部位？是脚趾缝、脚底，还是脚背/脚踝？"),
            ("rash", ["脱皮", "水疱", "水泡", "红疹", "风团", "渗液", "破溃", "红肿"], "皮肤表面有没有脱皮、水疱/水泡、红疹、风团、渗液、破溃或红肿？"),
            ("trigger", ["吃了", "用药", "药", "海鲜", "接触", "化妆品", "新鞋", "袜子", "洗衣液"], "最近有没有新鞋袜、洗衣液/沐浴露、外用药膏，或吃过可疑食物/新用药？"),
            ("severity", ["呼吸困难", "胸闷", "喉咙紧", "头晕", "发热", "疼痛", "流脓"], "有没有明显疼痛、红肿热痛、流脓、发热，或胸闷/呼吸困难/喉咙发紧？"),
            ("risk", ["儿童", "孕妇", "老人", "哮喘", "糖尿病", "慢性病", "用药"], "是否儿童、孕妇、老人，或有糖尿病、哮喘、慢性病/正在用药？"),
        ],
        "interim": "先避免抓挠和热水烫洗，保持局部清洁干燥；如果是脚部瘙痒，先勤换袜、保持鞋内干爽，不要自行混用多种药膏。",
        "red_flags": "出现红肿热痛、流脓、发热、迅速扩散，或胸闷/呼吸困难/喉咙发紧、嘴唇眼睑肿，建议及时就医。",
    },
    "便秘": {
        "priority": 4,
        "markers": ["便秘", "大便干", "排便困难", "拉不出来", "几天没大便"],
        "stage_groups": [
            ["便秘", "大便干", "排便困难", "拉不出来", "几天没大便"],
            ["腹痛", "腹胀", "呕吐", "便血", "黑便"],
            ["老人", "儿童", "孕妇", "慢性病", "用药"],
        ],
        "intro": "便秘要先看持续时间、排便困难程度和有没有腹痛便血等危险信号。",
        "questions": [
            ("duration", ["几天", "多久", "今天", "一周", "持续"], "已经多久没有正常排便了？平时排便规律怎样？"),
            ("stool", ["干", "硬", "费力", "出血", "黑便"], "大便是否很干硬、排便很费力？有没有出血或黑便？"),
            ("symptoms", ["腹痛", "腹胀", "呕吐", "发热"], "有没有明显腹痛、腹胀、呕吐或发热？"),
            ("risk", ["老人", "儿童", "孕妇", "慢性病", "用药"], "是否老人、儿童、孕妇，或最近用了止痛药、补铁、钙片等药物？"),
        ],
        "interim": "先增加饮水和膳食纤维，能活动就适当活动，不要长期依赖刺激性泻药。",
        "red_flags": "严重腹痛腹胀、呕吐、便血黑便、突然排便习惯明显改变，或老人儿童孕妇症状明显，建议就医。",
    },
    "头痛": {
        "priority": 4,
        "markers": ["头痛", "头疼", "偏头痛", "头胀"],
        "stage_groups": [
            ["头痛", "头疼", "偏头痛", "头胀"],
            ["发热", "发烧", "颈部僵硬", "呕吐", "视物模糊"],
            ["突然", "剧烈", "越来越痛", "外伤", "麻木", "说话不清"],
            ["高血压", "孕妇", "老人", "儿童", "慢性病"],
        ],
        "intro": "头痛要先排除危险信号，再看是否像疲劳、感冒、偏头痛或血压相关。",
        "questions": [
            ("onset", ["今天", "突然", "多久", "持续", "反复"], "头痛从什么时候开始？是突然剧烈，还是慢慢出现、反复发作？"),
            ("severity", ["轻微", "剧烈", "越来越痛", "影响睡觉"], "疼痛程度如何？有没有越来越重或影响睡眠？"),
            ("symptoms", ["发热", "呕吐", "视物模糊", "麻木", "说话不清", "颈部僵硬"], "有没有发热、呕吐、视物模糊、肢体麻木、说话不清或颈部僵硬？"),
            ("risk", ["高血压", "孕妇", "老人", "儿童", "外伤", "慢性病"], "有没有高血压、外伤、孕期、老人儿童或其他慢性病情况？"),
        ],
        "interim": "先休息、补水，避免饮酒和熬夜；如果怀疑血压问题，可以先测血压。",
        "red_flags": "突然爆发样剧烈头痛、伴发热颈僵/呕吐、肢体麻木无力、说话不清、外伤后头痛或血压很高，应尽快就医。",
    },
}


def _clean_history(history):
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-5:]:
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            if question:
                cleaned.append({"question": question})
    return cleaned


def _history_question_text(history):
    return " ".join(item["question"] for item in _clean_history(history)).strip()


def _history_has_clinical_topic(history):
    for item in _clean_history(history):
        question = item.get("question", "")
        if any(has_positive_marker(question, spec["markers"]) for spec in TOPICS.values()):
            return True
    return False


def clinical_user_context(question, history=None):
    parts = [str(question or "").strip()]
    history_text = _history_question_text(history)
    if history_text:
        parts.append(history_text)
    return " ".join(part for part in parts if part).strip()


def has_named_drug(question):
    q = str(question or "")
    if any(drug in q for drug in DRUG_HINTS):
        return True
    form_pattern = "|".join(sorted((re.escape(w) for w in DRUG_FORM_WORDS), key=len, reverse=True))
    return bool(re.search(rf"[\u4e00-\u9fa5A-Za-z0-9]{{2,18}}(?:{form_pattern})", q))


def _has_unknown(text, marker):
    around = str(text or "")
    idx = around.find(marker)
    if idx < 0:
        return False
    window = around[max(0, idx - 6):idx + len(marker) + 8]
    return any(word in window for word in UNKNOWN_WORDS)


def has_positive_marker(text, markers):
    raw = str(text or "")
    for marker in markers:
        start = 0
        while True:
            idx = raw.find(marker, start)
            if idx < 0:
                break
            before = raw[max(0, idx - 4):idx]
            around = raw[max(0, idx - 6):idx + len(marker) + 8]
            if not any(word in before for word in NEGATION_WORDS) and not any(word in around for word in UNKNOWN_WORDS):
                return True
            start = idx + len(marker)
    return False


def _slot_present(user_text, markers):
    raw = str(user_text or "")
    return any(marker in raw for marker in markers)


def _clinical_number(value):
    if not value:
        return None
    if value.isdigit():
        return int(value)
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if value == "十":
        return 10
    if value.startswith("十") and len(value) == 2:
        return 10 + mapping.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return mapping.get(value[0], 0) * 10
    if "十" in value and len(value) == 3:
        return mapping.get(value[0], 0) * 10 + mapping.get(value[2], 0)
    return mapping.get(value)


def diarrhea_frequency(text):
    raw = str(text or "")
    match = re.search(r"(?:拉|腹泻|大便|排便).{0,4}?([0-9一二两三四五六七八九十]{1,3})\s*(?:次|趟)", raw)
    return _clinical_number(match.group(1)) if match else None


def fever_temperature(text):
    raw = str(text or "")
    match = re.search(r"([3-4][0-9](?:\.[0-9])?)\s*(?:度|℃)?", raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def topic_stage_score(topic, text):
    spec = TOPICS.get(topic) or {}
    score = 0
    for group in spec.get("stage_groups", []):
        if has_positive_marker(text, group):
            score += 1
    if topic == "腹泻" and (diarrhea_frequency(text) or re.search(r"拉.{0,8}肚子|肚子.{0,8}拉", str(text or ""))):
        score += 1
    if topic == "发热" and fever_temperature(text):
        score += 1
    return score


def detect_clinical_intake_topic(question, history=None):
    q = clinical_user_context(question, history)
    if not q:
        return ""

    scored = []
    for topic, spec in TOPICS.items():
        score = 0
        if has_positive_marker(q, spec["markers"]):
            score += 2
        score += topic_stage_score(topic, q)
        if topic == "腹泻" and re.search(r"拉.{0,8}肚子|肚子.{0,8}拉", q):
            score += 2
        if score > 0:
            scored.append((score, spec["priority"], topic))

    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _looks_personal_or_acute(question, history=None):
    q = str(question or "")
    if history:
        return True
    markers = [
        "我", "孩子", "小孩", "老人", "孕妇", "今天", "现在", "最近", "一直", "刚",
        "一天", "半天", "几天", "吃了", "吃过", "喝了", "戴了", "用了",
        "怎么回事", "怎么办", "用什么", "吃什么", "能不能", "可以用", "可以吃",
        "原因", "是不是", "会是", "会不会", "为什么", "感觉", "不舒服", "不好",
        "胀", "疼", "痛", "干", "涩", "痒", "拉", "烧", "咳", "吐", "便秘",
    ]
    return any(marker in q for marker in markers)


def _is_generic_education_request(question, history=None):
    if history:
        return False
    q = str(question or "")
    education_markers = ["是什么", "为什么", "原因", "科普", "讲讲", "解释", "怎么回事", "常识"]
    strong_personal_markers = [
        "我", "孩子", "小孩", "老人", "孕妇", "今天", "现在", "最近", "一直", "刚",
        "一天", "半天", "几天", "吃了", "吃过", "喝了", "用了", "戴了",
        "怎么办", "用什么", "吃什么", "能不能", "可以用", "可以吃", "不舒服",
    ]
    return any(marker in q for marker in education_markers) and not any(marker in q for marker in strong_personal_markers)


def _is_analysis_request(question):
    q = str(question or "")
    markers = ["分析", "状况", "情况", "原因", "没分析", "不分析", "就这些", "为什么", "是不是", "会不会"]
    return any(marker in q for marker in markers)


def _planned_foot_itch_reply(question, history=None, mode="intake"):
    if not build_foot_itch_professional_reply:
        return None
    if not is_foot_itch_context(question, history):
        return None
    return build_foot_itch_professional_reply(
        question,
        history=history,
        mode=mode,
        disclaimer=default_disclaimer("symptom"),
    )


def _is_planned_symptom_context(question, history=None):
    return is_foot_itch_context(question, history) or is_diarrhea_context(question, history)


def _planned_diarrhea_reply(question, history=None, mode="intake", level=""):
    if not build_diarrhea_professional_reply:
        return None
    if not is_diarrhea_context(question, history):
        return None
    user_context = clinical_user_context(question, history)
    freq = diarrhea_frequency(user_context)
    observed_summary = ""
    if freq:
        observed_summary = f"已提到今天腹泻约{freq}次，次数越多越要重点防脱水。"
    return build_diarrhea_professional_reply(
        question,
        history=history,
        mode=mode,
        level=level,
        observed_summary=observed_summary,
        disclaimer=default_disclaimer("symptom"),
    )


def _topic_stage_level(topic, score, has_history, text):
    temp = fever_temperature(text)
    if topic == "腹泻":
        freq = diarrhea_frequency(text)
        if freq and freq >= 6:
            return "中症"
        if has_positive_marker(text, ["血便", "黑便", "明显尿少", "反复呕吐", "高热"]):
            return "中症"
        if has_positive_marker(text, ["发热", "发烧", "腹痛", "肚子痛", "肚子疼", "呕吐", "尿少", "头晕", "口干", "乏力"]):
            return "中症"
    elif topic == "发热":
        if (temp and temp >= 39) or has_positive_marker(text, ["呼吸困难", "胸痛", "意识异常", "高热", "精神差"]):
            return "中症"
    elif topic == "感冒":
        if (temp and temp >= 39) or has_positive_marker(text, ["高热", "呼吸困难", "胸痛", "黄痰", "气促"]):
            return "中症"
    elif topic == "腹痛":
        if has_positive_marker(text, ["右下腹", "血便", "黑便", "持续加重", "高热", "呕吐", "剧烈"]):
            return "中症"
    elif topic == "消化不适":
        if has_positive_marker(text, ["黑便", "血便", "呕血", "体重下降", "消瘦", "吞咽困难", "持续加重", "高热", "反复呕吐"]):
            return "中症"
    elif topic == "眼干":
        if has_positive_marker(text, ["眼痛", "明显红", "畏光", "分泌物", "视力下降", "异物感"]):
            return "中症"
    elif topic == "皮肤过敏":
        if has_positive_marker(text, ["呼吸困难", "胸闷", "喉咙紧", "嘴唇肿", "眼睑肿", "头晕", "全身迅速", "红肿热痛", "流脓", "发热"]):
            return "中症"
    elif topic == "便秘":
        if has_positive_marker(text, ["严重腹痛", "腹胀", "呕吐", "便血", "黑便"]):
            return "中症"
    elif topic == "头痛":
        if has_positive_marker(text, ["突然", "剧烈", "越来越痛", "外伤", "麻木", "说话不清", "颈部僵硬", "视物模糊"]):
            return "中症"
    return "轻症"


def _clinical_question_missing(spec, user_text):
    missing = []
    for _key, markers, prompt in spec["questions"]:
        if not _slot_present(user_text, markers):
            missing.append(prompt)
    return missing


def _stage_lines(topic, level, user_context, has_history):
    lines = ["我先按你已经补充的信息做阶段性判断。"]

    if topic == "腹泻":
        freq = diarrhea_frequency(user_context)
        if freq and freq >= 6:
            lines.append(f"今天已经腹泻约{freq}次，次数偏多，重点是防脱水，并留意发热、血便或腹痛加重。")
        if has_positive_marker(user_context, ["发热", "发烧", "腹痛", "肚子痛", "肚子疼"]):
            lines.append("合并发热或腹痛时，比单纯腹泻更需要重视，常见要考虑急性胃肠炎或感染性腹泻。")
        elif has_positive_marker(user_context, ["口干", "尿少", "头晕", "乏力"]):
            lines.append("你提到有乏力、口干、尿少或头晕这类表现，要重点观察脱水风险。")
        lines.append("现在先少量多次补液，饮食清淡，暂时避开冰冷、油腻、辛辣和酒精。")
    elif topic == "发热":
        temp = fever_temperature(user_context)
        if temp:
            lines.append(f"目前提到的体温约{temp:g}℃，后续要看是否持续升高、是否退了又升。")
        if has_positive_marker(user_context, ["咳嗽", "咽痛", "流鼻涕", "鼻塞"]):
            lines.append("伴随咳嗽、咽痛、流涕时，更像上呼吸道感染方向，但仍要看持续时间和呼吸情况。")
        if has_positive_marker(user_context, ["腹痛", "腹泻"]):
            lines.append("如果发热还合并腹痛或腹泻，也要考虑胃肠道感染的可能。")
        lines.append("先休息、补水，避免自行叠加多种退烧药。")
    elif topic == "感冒":
        if has_positive_marker(user_context, ["发热", "发烧"]):
            lines.append("感冒样症状合并发热时，要看最高体温、持续几天和精神状态。")
        if has_positive_marker(user_context, ["黄痰", "胸闷", "气促", "呼吸困难"]):
            lines.append("如果有黄痰、胸闷、气促或呼吸困难，就不能只按普通感冒看。")
        lines.append("先休息、补水，不要自行使用抗生素。")
    elif topic == "腹痛":
        if has_positive_marker(user_context, ["发热", "腹泻", "呕吐", "恶心"]):
            lines.append("腹痛合并发热、腹泻或呕吐时，要优先考虑急性胃肠炎、肠道感染等方向。")
        if has_positive_marker(user_context, ["右下腹", "下腹"]):
            lines.append("如果疼痛主要在右下腹或持续加重，更需要尽快就医排查。")
        lines.append("先避免油腻辛辣和酒精，不要盲目吃止痛药掩盖病情。")
    elif topic == "消化不适":
        if has_positive_marker(user_context, ["反酸", "嗳气", "胃胀", "腹胀", "肚子胀"]):
            lines.append("胃口差、腹胀、反酸嗳气常见和饮食刺激、胃肠动力或消化道炎症有关，但不能直接下诊断。")
        lines.append("现在先少量多餐、清淡饮食，暂时避开油腻辛辣、酒精、浓茶咖啡和过甜食物。")
    elif topic == "眼干":
        if has_positive_marker(user_context, ["电脑", "手机", "屏幕", "熬夜", "空调"]):
            lines.append("长时间看屏幕、熬夜或空调环境会减少眨眼和泪膜稳定性，容易加重干涩。")
        if has_positive_marker(user_context, ["隐形眼镜", "美瞳"]):
            lines.append("戴隐形眼镜或美瞳后干涩明显时，要先减少佩戴，避免角膜刺激。")
        lines.append("先减少连续盯屏、主动眨眼、规律休息，避免揉眼。需要眼药水时，要核对说明书或问药师。")
    elif topic == "皮肤过敏":
        if has_positive_marker(user_context, ["脚", "脚趾缝", "脚底", "脚痒", "足痒"]):
            lines.append("脚部瘙痒越来越重，常见方向包括足癣/脚气、湿疹或接触性皮炎，也可能和鞋袜闷热、潮湿摩擦有关。")
            lines.append("如果有脚趾缝脱皮、水疱、渗液、糜烂或反复发作，会更偏向真菌感染或湿疹方向，需要进一步确认。")
        else:
            lines.append("皮肤瘙痒可能和过敏、湿疹/接触性皮炎、虫咬、感染或皮肤干燥有关，要看皮疹形态和诱因。")
        if has_positive_marker(user_context, ["吃了", "用药", "药", "海鲜", "化妆品", "新鞋", "袜子", "洗衣液"]):
            lines.append("如果和新食物、新药、鞋袜、清洁用品或化妆品时间上相关，要把这个线索记下来。")
        lines.append("先避免抓挠和热水烫洗，保持局部清洁干燥，不要自行混用多种药膏。")
    elif topic == "便秘":
        lines.append("短期便秘先从饮水、膳食纤维、活动和排便习惯调整入手，不建议长期依赖刺激性泻药。")
    elif topic == "头痛":
        lines.append("头痛先排除突然剧烈、神经系统异常、发热颈僵、外伤和血压明显异常这些危险线索。")
        lines.append("如果没有危险信号，可以先休息、补水，必要时测血压，观察是否和熬夜、紧张、感冒有关。")

    if level == "中症":
        lines.append("目前信息里有需要提高警惕的点，建议尽快线下就医或至少咨询医生/药师进一步评估。")
    else:
        lines.append("目前更像可以先观察和对症处理的轻症方向，但如果症状加重或出现危险信号，要及时就医。")
    return lines


def build_clinical_stage_reply(question, history=None):
    q = str(question or "").strip()
    topic = detect_clinical_intake_topic(q, history)
    if not topic:
        return None
    if has_named_drug(q) and not _is_planned_symptom_context(q, history):
        return None
    if _is_generic_education_request(q, history):
        return None

    has_history = _history_has_clinical_topic(history)
    user_context = clinical_user_context(q, history)
    score = topic_stage_score(topic, user_context)

    if not has_history and score < 2:
        return None
    if has_history and score < 1:
        return None

    level = _topic_stage_level(topic, score, has_history, user_context)
    if topic == "腹泻":
        planned_reply = _planned_diarrhea_reply(q, history, mode="intake", level=level)
        if planned_reply:
            return {
                "reply": planned_reply,
                "level": level,
                "needs_doctor": level == "中症",
                "score": score,
                "topic": "腹泻",
            }

    if topic == "皮肤过敏":
        planned_reply = _planned_foot_itch_reply(q, history, mode="intake")
        if planned_reply:
            return {
                "reply": planned_reply,
                "level": level,
                "needs_doctor": level == "中症",
                "score": score,
                "topic": "脚痒",
            }

    spec = TOPICS[topic]
    lines = _stage_lines(topic, level, user_context, has_history)
    missing = _clinical_question_missing(spec, user_context)
    if missing:
        lines.append(f"还差一个关键点：{missing[0]}")
    lines.extend([
        "",
        f"当前更偏向：{level}。",
        "",
        f"需要尽快就医的情况：{spec['red_flags']}",
        default_disclaimer("symptom"),
    ])
    return {
        "reply": "\n".join(line for line in lines if line is not None),
        "level": level,
        "needs_doctor": level == "中症",
        "score": score,
        "topic": topic,
    }


def build_clinical_analysis_reply(question, history=None):
    q = str(question or "").strip()
    if not history or not _is_analysis_request(q):
        return None
    topic = detect_clinical_intake_topic(q, history)
    if not topic:
        return None
    if has_named_drug(q) and not _is_planned_symptom_context(q, history):
        return None

    user_context = clinical_user_context(q, history)
    score = topic_stage_score(topic, user_context)
    level = _topic_stage_level(topic, score, True, user_context)
    spec = TOPICS[topic]
    planned_reply = _planned_diarrhea_reply(q, history, mode="analysis", level=level)
    if planned_reply:
        return {
            "reply": planned_reply,
            "level": level,
            "needs_doctor": level == "中症",
            "score": score,
            "topic": "腹泻",
        }

    planned_reply = _planned_foot_itch_reply(q, history, mode="analysis")
    if planned_reply:
        return {
            "reply": planned_reply,
            "level": level,
            "needs_doctor": level == "中症",
            "score": score,
            "topic": "脚痒",
        }

    lines = ["你提醒得对，我应该先把状况拆开分析，而不是只给一句处理建议。"]
    if topic == "皮肤过敏":
        lines.extend([
            "",
            "按你前面说的“脚痒越来越严重”，常见要分几类看：",
            "1. 足癣/脚气：常见在脚趾缝或脚底，可能有脱皮、水疱、潮湿后加重、反复发作。",
            "2. 湿疹或接触性皮炎：可能和新鞋袜、洗衣液、闷热摩擦、外用药膏刺激有关，常有红疹、渗出或反复瘙痒。",
            "3. 过敏或虫咬：可能突然起风团、红疙瘩，通常和接触或叮咬时间相关。",
            "4. 感染风险：如果有红肿热痛、流脓、发热或范围迅速扩大，就不能只当普通瘙痒处理。",
        ])
        next_questions = [
            "痒的位置是在脚趾缝、脚底，还是脚背/脚踝？",
            "有没有脱皮、水疱/水泡、渗液、破溃、红肿或明显疼痛？",
            "最近有没有换新鞋袜、洗衣液、外用药膏，或脚部长期潮湿闷热？",
        ]
    else:
        lines.extend([""])
        lines.extend(_stage_lines(topic, level, user_context, True))
        next_questions = _clinical_question_missing(spec, user_context)[:3]

    lines.extend(["", "为了继续判断，我需要你补充："])
    lines.extend([f"{idx}. {item}" for idx, item in enumerate(next_questions, 1)])
    lines.extend([
        "",
        f"当前先按安全原则：{spec['interim']}",
        "",
        f"需要及时就医的情况：{spec['red_flags']}",
        default_disclaimer("symptom"),
    ])
    return {
        "reply": "\n".join(line for line in lines if line is not None),
        "level": level,
        "needs_doctor": level == "中症",
        "score": score,
        "topic": topic,
    }


def build_clinical_intake_reply(question, history=None):
    q = str(question or "").strip()
    topic = detect_clinical_intake_topic(q, history)
    if not topic:
        return None
    if has_named_drug(q) and not _is_planned_symptom_context(q, history):
        return None
    if _is_generic_education_request(q, history):
        return None
    if not _looks_personal_or_acute(q, history):
        return None

    if topic == "腹泻":
        planned_reply = _planned_diarrhea_reply(q, history, mode="intake")
        if planned_reply:
            return planned_reply

    if topic == "皮肤过敏":
        planned_reply = _planned_foot_itch_reply(q, history, mode="intake")
        if planned_reply:
            return planned_reply

    spec = TOPICS[topic]
    user_context = clinical_user_context(q, history)
    missing = _clinical_question_missing(spec, user_context)
    if len(missing) <= 1 and _history_has_clinical_topic(history):
        return None

    questions = missing[:4] or [prompt for _key, _markers, prompt in spec["questions"][:4]]
    lines = [
        spec["intro"],
        "",
        "我先确认几个关键点：",
    ]
    lines.extend([f"{idx}. {item}" for idx, item in enumerate(questions, 1)])
    lines.extend([
        "",
        f"在你补充前，可以先这样处理：{spec['interim']}",
        "",
        f"需要尽快就医的情况：{spec['red_flags']}",
        default_disclaimer("symptom"),
    ])
    return "\n".join(line for line in lines if line is not None)
