"""
09_qlora_finetune.py
QLoRA/LoRA 微调训练 - 纯PyTorch循环版（不用Trainer，Windows兼容）
在 4070Ti 上微调 Qwen2.5-1.5B-Instruct
"""
import os
import sys
import json
import torch
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, BASE_DIR

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4  # 等效batch_size=8
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
OUTPUT_DIR = os.path.join(BASE_DIR, "lora_output")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def finetune():
    print("=" * 60)
    print("第2-4步：LoRA 微调训练（纯PyTorch版）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"训练: epochs={EPOCHS}, batch={BATCH_SIZE}, accum={GRAD_ACCUM}")
    print()

    # 1. 加载tokenizer
    print("加载 tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型（直接to cuda，不用device_map）
    print("加载模型（fp16）...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to("cuda")
    model.config.use_cache = False
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"模型加载完成, GPU显存: {gpu_mem:.1f} GB", flush=True)

    # 3. 配置LoRA
    print("配置 LoRA...", flush=True)
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    gpu_mem2 = torch.cuda.memory_allocated() / 1024**3
    print(f"LoRA配置后, GPU显存: {gpu_mem2:.1f} GB", flush=True)

    # 4. 加载训练数据
    print("加载训练数据...", flush=True)
    train_path = os.path.join(DATA_DIR, "finetune_train.json")
    with open(train_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"训练数据: {len(raw_data)} 条", flush=True)

    # 5. Tokenize
    print("Tokenizing...", flush=True)
    tokenized_data = []
    for item in raw_data:
        messages = [
            {"role": "system", "content": "你是专业药学咨询助手，回答用药问题。回答必须包含用药建议、注意事项和安全提示。"},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
        tokenized_data.append({
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": enc["input_ids"][0].clone(),
        })
    print(f"Tokenize完成: {len(tokenized_data)} 条", flush=True)

    # 6. 纯PyTorch训练循环
    print("\n开始训练...", flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = (len(tokenized_data) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"总步数: ~{total_steps}", flush=True)

    step = 0
    accum_count = 0
    losses = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_steps = 0

        # 打乱数据
        import random
        random.seed(42 + epoch)
        random.shuffle(tokenized_data)

        for i in range(0, len(tokenized_data), BATCH_SIZE):
            batch = tokenized_data[i:i+BATCH_SIZE]

            input_ids = torch.stack([b["input_ids"] for b in batch]).to("cuda")
            attention_mask = torch.stack([b["attention_mask"] for b in batch]).to("cuda")
            labels = torch.stack([b["labels"] for b in batch]).to("cuda")

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM

            # 反向传播
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM
            accum_count += 1

            # 梯度累积后更新
            if accum_count >= GRAD_ACCUM:
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1
                epoch_steps += 1

                if step % 10 == 0:
                    avg_loss = epoch_loss / (epoch_steps * GRAD_ACCUM)
                    elapsed = time.time() - t0
                    remaining = elapsed / step * (total_steps - step)
                    print(f"  epoch {epoch+1}/{EPOCHS} step {step}/{total_steps} "
                          f"loss={avg_loss:.4f} "
                          f"elapsed={elapsed:.0f}s remaining={remaining:.0f}s "
                          f"GPU={torch.cuda.memory_allocated()/1024**3:.1f}GB",
                          flush=True)

        print(f"Epoch {epoch+1} 完成, 平均loss={epoch_loss/len(tokenized_data):.4f}", flush=True)

    t1 = time.time()
    print(f"\n训练完成! 耗时 {t1-t0:.0f}秒 ({(t1-t0)/60:.1f}分钟)", flush=True)

    # 7. 保存LoRA适配器
    print(f"保存LoRA适配器到 {OUTPUT_DIR}", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 打印结果
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n{'='*60}")
    print(f"微调结果:")
    print(f"  训练数据: {len(raw_data)} 条")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  训练耗时: {(t1-t0)/60:.1f} 分钟")
    print(f"  峰值显存: {peak_mem:.1f} GB / 16 GB")
    print(f"  LoRA适配器: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\n下一步: 用 10_eval_finetune.py 评估微调效果", flush=True)


if __name__ == "__main__":
    finetune()
