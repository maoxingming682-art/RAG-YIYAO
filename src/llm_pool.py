"""
llm_pool.py
多API轮询模块：两个API自动切换，一个限流/报错自动切另一个

API1: keungliang（25个模型，主力）
API2: zhenhaoji（glm-4.7-flash，备用）
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _env_api(index, default_name, default_base_url, default_model, default_priority):
    return {
        "name": os.getenv(f"LLM_API_{index}_NAME", default_name),
        "base_url": os.getenv(f"LLM_API_{index}_BASE_URL", default_base_url),
        "api_key": os.getenv(f"LLM_API_{index}_KEY", ""),
        "model": os.getenv(f"LLM_API_{index}_MODEL", default_model),
        "priority": int(os.getenv(f"LLM_API_{index}_PRIORITY", str(default_priority))),
        "last_fail": 0,
        "cooldown": int(os.getenv(f"LLM_API_{index}_COOLDOWN", "15")),
    }


# ===== API配置（从 .env 读取，避免密钥进入仓库）=====
RAW_APIS = [
    _env_api(
        1,
        "agentrs",
        "https://agentrs.jd.com/api/saas/openai-u/v1",
        "GLM-5.1",
        1,
    ),
    _env_api(
        2,
        "keungliang",
        "https://keungliang.dpdns.org/v1",
        "glm-5.2",
        2,
    ),
    _env_api(
        3,
        "zhenhaoji",
        "https://api.zhenhaoji.qzz.io/v1",
        "glm-4.7-flash",
        3,
    ),
    _env_api(
        4,
        "runanytime-glm-5.2",
        "https://runanytime.hxi.me/v1",
        "z-ai/glm-5.2",
        1,
    ),
    _env_api(
        5,
        "runanytime-mimo-v2.5-pro",
        "https://runanytime.hxi.me/v1",
        "xiaomi/mimo-v2.5-pro",
        2,
    ),
]

APIS = [api for api in RAW_APIS if api.get("api_key")]

if os.getenv("LLM_API_KEY"):
    APIS.append(
        {
            "name": os.getenv("LLM_API_NAME", "default"),
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("LLM_API_KEY"),
            "model": os.getenv("LLM_MODEL", "glm-4.7-flash"),
            "priority": int(os.getenv("LLM_API_PRIORITY", "99")),
            "last_fail": 0,
            "cooldown": int(os.getenv("LLM_API_COOLDOWN", "15")),
        }
    )

"""旧版示例结构，真实 key 已改为从 .env 读取：
APIS = [
    {
        "name": "agentrs",
        "base_url": "https://agentrs.jd.com/api/saas/openai-u/v1",
        "api_key": "...",
        "model": "GLM-5.1",
        "priority": 1,  # 最稳定，主力
        "last_fail": 0,
        "cooldown": 15,  # 失败后冷却15秒（不等60秒）
    },
    {
        "name": "keungliang",
        "base_url": "https://keungliang.dpdns.org/v1",
        "api_key": "...",
        "model": "glm-5.2",
        "priority": 2,
        "last_fail": 0,
        "cooldown": 15,
    },
    {
        "name": "zhenhaoji",
        "base_url": "https://api.zhenhaoji.qzz.io/v1",
        "api_key": "...",
        "model": "glm-4.7-flash",
        "priority": 3,  # 备用
        "last_fail": 0,
        "cooldown": 15,
    },
]
"""


def _is_available(api):
    """检查API是否在冷却期外"""
    if api["last_fail"] == 0:
        return True
    return time.time() - api["last_fail"] >= api["cooldown"]


def _mark_fail(api):
    """标记API失败"""
    api["last_fail"] = time.time()


def _call_single(api, messages, temperature, max_tokens):
    """调用单个API"""
    client = OpenAI(base_url=api["base_url"], api_key=api["api_key"], timeout=90)
    try:
        stream = client.chat.completions.create(
            model=api["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    chunks.append(delta.content)
        result = "".join(chunks).strip()
        if not result:
            raise Exception("空回复")
        return result
    finally:
        client.close()


def chat(messages, temperature=0.3, max_tokens=1024, retry=2):
    """
    多API轮询调用
    - 按优先级从高到低依次尝试
    - 某API失败/限流自动切下一个
    - 全部失败则重试一轮
    返回: 模型回复文本
    """
    if not APIS:
        raise RuntimeError("未配置 LLM API Key，请在项目根目录 .env 中设置 LLM_API_1_KEY 或 LLM_API_KEY。")

    last_error = None

    for attempt in range(retry + 1):
        # 按优先级排序，跳过冷却中的
        available = [a for a in sorted(APIS, key=lambda x: x["priority"]) if _is_available(a)]
        if not available:
            # 全在冷却中，等最短的
            min_wait = min(a["cooldown"] - (time.time() - a["last_fail"]) for a in APIS)
            wait = max(5, min_wait)
            print(f"  所有API冷却中，等待{wait:.0f}秒...", flush=True)
            time.sleep(wait)
            continue

        for api in available:
            try:
                result = _call_single(api, messages, temperature, max_tokens)
                api["last_fail"] = 0  # 成功就清除冷却
                return result
            except Exception as e:
                last_error = e
                err = str(e).lower()
                is_rate_limit = "429" in err or "rate" in err or "limit" in err
                is_server = "500" in err or "502" in err or "503" in err or "504" in err or "timeout" in err or "524" in err

                if is_rate_limit or is_server:
                    print(f"  [{api['name']}] 限流/服务异常，切换下一个API...", flush=True)
                    _mark_fail(api)
                    continue
                else:
                    # 其他错误也切换
                    print(f"  [{api['name']}] 错误: {str(e)[:80]}，切换...", flush=True)
                    _mark_fail(api)
                    continue

        if attempt < retry:
            time.sleep(3)

    raise last_error


def chat_stream(messages, temperature=0.3, max_tokens=1024, retry=1):
    """
    流式版多API轮询：逐token yield生成内容。
    用法: for chunk in chat_stream(messages): send(chunk)
    失败自动切下一个API；首个API成功后中途断开则抛错（不再切换，避免输出错乱）。
    """
    if not APIS:
        raise RuntimeError("未配置 LLM API Key，请在项目根目录 .env 中设置 LLM_API_1_KEY 或 LLM_API_KEY。")

    last_error = None
    for attempt in range(retry + 1):
        available = [a for a in sorted(APIS, key=lambda x: x["priority"]) if _is_available(a)]
        if not available:
            min_wait = min(a["cooldown"] - (time.time() - a["last_fail"]) for a in APIS)
            wait = max(3, min_wait)
            print(f"  [stream] 所有API冷却中，等待{wait:.0f}秒...", flush=True)
            time.sleep(wait)
            continue
        for api in available:
            client = OpenAI(base_url=api["base_url"], api_key=api["api_key"], timeout=120)
            got = False
            try:
                stream = client.chat.completions.create(
                    model=api["model"], messages=messages,
                    temperature=temperature, max_tokens=max_tokens, stream=True,
                )
                for chunk in stream:
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            got = True
                            yield delta.content
                api["last_fail"] = 0
                if not got:
                    raise Exception("空回复")
                client.close()
                return
            except Exception as e:
                last_error = e
                try: client.close()
                except: pass
                if not got:
                    # 还没输出过内容，可以安全切换
                    print(f"  [stream][{api['name']}] 切换: {str(e)[:60]}", flush=True)
                    _mark_fail(api)
                    continue
                else:
                    # 已经输出过内容，不能再切换，直接结束
                    print(f"  [stream][{api['name']}] 中途断开: {str(e)[:60]}", flush=True)
                    return
        if attempt < retry:
            time.sleep(2)
    raise last_error


def get_status():
    """获取各API状态"""
    result = []
    configured = {id(api) for api in APIS}
    for api in RAW_APIS:
        if id(api) not in configured and not api.get("api_key"):
            result.append({
                "name": api["name"],
                "model": api["model"],
                "priority": api["priority"],
                "status": "未配置",
            })
            continue
        if _is_available(api):
            status = "可用"
        else:
            remaining = int(api["cooldown"] - (time.time() - api["last_fail"]))
            status = f"冷却中({remaining}s)"
        result.append({"name": api["name"], "model": api["model"], "priority": api["priority"], "status": status})
    for api in APIS:
        if api not in RAW_APIS:
            if _is_available(api):
                status = "可用"
            else:
                remaining = int(api["cooldown"] - (time.time() - api["last_fail"]))
                status = f"冷却中({remaining}s)"
            result.append({"name": api["name"], "model": api["model"], "priority": api["priority"], "status": status})
    return result


if __name__ == "__main__":
    # 测试轮询
    print("=" * 50)
    print("  多API轮询测试")
    print("=" * 50)

    print("\nAPI状态:")
    for s in get_status():
        print(f"  {s['name']} ({s['model']}): {s['status']}")

    # 测试3次调用
    for i in range(3):
        print(f"\n--- 调用{i+1} ---")
        try:
            reply = chat(
                [{"role": "user", "content": f"回复OK，这是第{i+1}次测试"}],
                temperature=0.1, max_tokens=50
            )
            print(f"  回复: {reply[:60]}")
        except Exception as e:
            print(f"  失败: {e}")

    print("\n最终API状态:")
    for s in get_status():
        print(f"  {s['name']} ({s['model']}): {s['status']}")
