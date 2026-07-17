"""
问诊/科普路由回归检查。

只覆盖不需要真实药品检索和外部模型的核心链路：
- 纯科普走科普库；
- 个人症状先问诊；
- 多轮补充能阶段判断；
- 回答不重复自我介绍，不输出 Markdown 加粗星号。
"""
import sys

sys.path.insert(0, ".")

import app  # noqa: E402


CASES = [
    {
        "name": "pure_eye_science",
        "question": "干眼症是什么原因",
        "history": [],
        "must_steps": ["知识路由：科普库"],
        "must_answer": ["眼睛干涩", "我还需要确认"],
    },
    {
        "name": "eye_dry_intake",
        "question": "眼睛干涩可以用什么",
        "history": [],
        "must_steps": ["问诊式澄清"],
        "must_answer": ["先确认几个关键点", "眼痛"],
    },
    {
        "name": "personal_diarrhea_stage",
        "question": "我今天拉肚子是怎么回事",
        "history": [],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["不能只凭", "当前更偏向", "拉了几次", "口服补液盐", "不建议自行使用抗生素"],
    },
    {
        "name": "diarrhea_followup_frequency",
        "question": "拉7次 体温不知道",
        "history": [{"question": "我今天拉肚子", "answer": "请补充次数和体温。"}],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["7次", "中症", "脱水", "血便或黑便"],
    },
    {
        "name": "diarrhea_gastroenteritis_analysis",
        "question": "拉肚子是不是急性肠胃炎",
        "history": [],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["有这个可能", "不能只凭", "急性胃肠炎", "感染性腹泻"],
    },
    {
        "name": "diarrhea_after_ice_watermelon",
        "question": "吃了冰西瓜后拉肚子",
        "history": [],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["饮食刺激", "冰西瓜", "口服补液盐"],
    },
    {
        "name": "diarrhea_watery_fever",
        "question": "水样便还发热",
        "history": [{"question": "我拉肚子", "answer": "请补充次数和体温。"}],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["急性胃肠炎", "感染性腹泻", "脱水", "中症"],
    },
    {
        "name": "diarrhea_bloody_black_stool",
        "question": "拉肚子还有便血黑便",
        "history": [],
        "must_steps": ["已拦截：引导就医"],
        "must_answer": ["需要提高警惕", "血便或黑便", "尽快就医"],
    },
    {
        "name": "antibiotic_cold_science",
        "question": "感冒为什么不建议自己吃抗生素",
        "history": [],
        "must_steps": ["知识路由：科普库"],
        "must_answer": ["普通感冒多数是病毒感染", "不建议自行"],
    },
    {
        "name": "indigestion_intake",
        "question": "最近胃口不好",
        "history": [],
        "must_steps": ["问诊式澄清"],
        "must_answer": ["消化不适", "持续多久"],
    },
    {
        "name": "foot_itch_after_chitchat_intake",
        "question": "最近脚痒越来越严重了怎么办？",
        "history": [{"question": "HI", "answer": "您好，请问有什么可以帮您？"}],
        "must_steps": ["问诊式澄清"],
        "must_answer": ["不能下结论", "更像足癣/脚气", "真菌镜检", "用药边界"],
    },
    {
        "name": "analysis_followup_no_repeat",
        "question": "不分析状况吗",
        "history": [{"question": "最近脚痒越来越严重了怎么办？", "answer": "请补充更多信息。"}],
        "must_steps": ["问诊式状况分析"],
        "must_answer": ["足癣/脚气", "湿疹、汗疱疹或接触性皮炎", "本地审核资料"],
    },
    {
        "name": "is_it_athletes_foot",
        "question": "是不是脚气？",
        "history": [{"question": "最近脚痒越来越严重了怎么办？", "answer": "请补充更多信息。"}],
        "must_steps": ["问诊式状况分析"],
        "must_answer": ["仅凭“痒”还不能下结论", "更像足癣/脚气", "真菌镜检"],
    },
    {
        "name": "toe_web_scaling",
        "question": "脚趾缝脱皮还痒",
        "history": [],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["脚趾缝", "足癣/脚气", "湿疹、汗疱疹或接触性皮炎"],
    },
    {
        "name": "steroid_cream_worse",
        "question": "用了激素药膏更严重",
        "history": [{"question": "最近脚痒越来越严重了怎么办？", "answer": "请补充更多信息。"}],
        "must_steps": ["问诊式阶段判断"],
        "must_answer": ["激素", "真菌感染", "用药边界"],
    },
]


FORBIDDEN_ANSWER = ["您好，我是", "作为药学助手", "**"]


def assert_contains(label, value, needles):
    missing = [needle for needle in needles if needle not in value]
    if missing:
        raise AssertionError(f"{label} missing: {missing}\nvalue={value[:500]}")


def main():
    failures = []
    for case in CASES:
        try:
            result = app.process_question(case["question"], history=case["history"])
            steps = " | ".join(result.get("steps", []))
            answer = result.get("answer", "")
            assert_contains("steps", steps, case["must_steps"])
            assert_contains("answer", answer, case["must_answer"])
            forbidden_hits = [item for item in FORBIDDEN_ANSWER if item in answer]
            if forbidden_hits:
                raise AssertionError(f"forbidden answer fragments: {forbidden_hits}")
            print(f"[PASS] {case['name']} :: {steps}")
        except Exception as exc:
            failures.append((case["name"], str(exc)))
            print(f"[FAIL] {case['name']} :: {exc}")

    if failures:
        print("\nFailed cases:")
        for name, error in failures:
            print(f"- {name}: {error}")
        return 1
    print(f"\nAll {len(CASES)} medical assistant checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
