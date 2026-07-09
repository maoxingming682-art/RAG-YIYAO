"""
11_serve_finetuned.py
部署微调后模型为 OpenAI 兼容 API 服务
启动后你的 RAG demo 只需把 LLM_BASE_URL 改成 http://localhost:8001/v1

启动：python 11_serve_finetuned.py
切换：把 config.py 的 LLM_BASE_URL 改成 http://localhost:8001/v1
"""
import os
import sys
import json
import time
import torch
from flask import Flask, request, jsonify

# 安全地重配置编码（后台启动时stdout可能不支持reconfigure）
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = os.path.join(BASE_DIR, "lora_output")
PORT = 8001
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = Flask(__name__)

# 全局模型和tokenizer（启动时加载一次）
_model = None
_tokenizer = None


def load_model():
    """加载微调后模型"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"加载基础模型: {MODEL_NAME}...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, trust_remote_code=True
    ).to("cuda")

    # 加载LoRA适配器
    print(f"加载LoRA适配器: {LORA_PATH}...", flush=True)
    from peft import PeftModel
    _model = PeftModel.from_pretrained(_model, LORA_PATH)
    _model.eval()

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"模型加载完成! GPU显存: {gpu_mem:.1f}GB", flush=True)
    print(f"API服务启动在 http://localhost:{PORT}", flush=True)
    print(f"OpenAI兼容端点: http://localhost:{PORT}/v1/chat/completions", flush=True)
    return _model, _tokenizer


def generate(messages, max_tokens=512, temperature=0.3):
    """生成回答"""
    model, tokenizer = load_model()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI 兼容的 chat completions 接口"""
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.3)
    stream = data.get("stream", False)

    t0 = time.time()
    try:
        answer = generate(messages, max_tokens, temperature)
        t1 = time.time()

        if stream:
            # 简单流式：把答案分成小块返回
            def generate_stream():
                chunk_size = 5
                for i in range(0, len(answer), chunk_size):
                    chunk = {
                        "choices": [{"delta": {"content": answer[i:i+chunk_size]}, "index": 0}]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return app.response_class(generate_stream(), mimetype="text/event-stream")
        else:
            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen2.5-1.5b-pharmacist-finetuned",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "processing_time": f"{t1-t0:.1f}s",
            })
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "internal_error"}}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """模型列表"""
    return jsonify({
        "data": [{"id": "qwen2.5-1.5b-pharmacist-finetuned", "object": "model"}]
    })


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "model": "qwen2.5-1.5b-pharmacist-finetuned"})


if __name__ == "__main__":
    print("=" * 60)
    print("  微调模型 API 服务")
    print("  模型: Qwen2.5-1.5B + LoRA(药学咨询微调)")
    print(f"  端口: {PORT}")
    print(f"  端点: http://localhost:{PORT}/v1/chat/completions")
    print("=" * 60)

    # 启动时预加载模型
    load_model()

    print(f"\n服务就绪! 等待请求...")
    print(f"测试: curl http://localhost:{PORT}/health")
    print(f"切换: 把 config.py 的 LLM_BASE_URL 改成 http://localhost:{PORT}/v1")
    print(f"按 Ctrl+C 停止服务\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
