"""
02_chunk_embed.py (混合切分版)
3层递进切分：问答对 → 句子边界 → 字数兜底
每个子块带问题前缀，解决问答对断裂问题
"""
import os
import sys
import time
import json
import re
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_tokenizer = None
_model = None

# 句子分隔符（中文标点）
SENTENCE_ENDINGS = re.compile(r'[。！？；\n]')


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    from transformers import AutoTokenizer, AutoModel
    print(f"加载 embedding 模型: {EMBEDDING_MODEL_NAME}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    _model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    _model.eval()
    print("模型加载完成", flush=True)
    return _tokenizer, _model


def encode_texts(texts, batch_size=32):
    """用transformers编码文本为向量（mean pooling + L2归一化）"""
    tokenizer, model = _load_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.extend(embeddings.tolist())
        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(texts):
            print(f"  已向量化 {len(all_embeddings)}/{len(texts)} 块", flush=True)
    return all_embeddings


def split_by_sentences(text, max_length=300):
    """
    按句子边界切分文本，保证不切断句子
    在。！？；\n处切，每个块不超过max_length字
    """
    # 找所有句子结束位置
    sentences = []
    last_end = 0
    for m in SENTENCE_ENDINGS.finditer(text):
        end = m.end()
        sentences.append(text[last_end:end])
        last_end = end
    if last_end < len(text):
        sentences.append(text[last_end:])  # 最后一段

    # 把句子拼成块，每块不超过max_length
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_length:
            current += sent
        else:
            if current:
                chunks.append(current)
            # 单个句子超过max_length，硬切兜底
            if len(sent) > max_length:
                for i in range(0, len(sent), max_length):
                    chunks.append(sent[i:i+max_length])
            else:
                current = sent
                continue
            current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_qa_pair(item, max_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    混合切分：按问答对 → 句子边界 → 字数兜底
    每个子块带问题前缀，解决问答断裂问题

    改进点：
    1. 先分离"问题"和"答案"
    2. 答案按句子切分（不断句号）
    3. 每个子块前缀加上"[药品名] " 让检索知道在回答什么药
    """
    text = item["text"]
    question = item["question"]

    # 如果整条不超过max_size，整条保留
    if len(text) <= max_size:
        return [{"id": item["id"], "text": text, "source": question}]

    # 分离问题和答案
    # 格式："问题：xxx\n答案：xxx"
    parts = text.split("\n答案：", 1)
    question_part = parts[0]  # "问题：xxx"
    answer_part = parts[1] if len(parts) > 1 else text

    # 提取药品名/关键词作为前缀（问题去掉"问题："前缀）
    drug_prefix = question_part.replace("问题：", "").strip()[:30]

    # 按句子切分答案
    answer_chunks = split_by_sentences(answer_part, max_size - len(drug_prefix) - 10)

    # 每个子块加上药品名前缀
    all_chunks = []
    for j, ans_chunk in enumerate(answer_chunks):
        # 给每个子块加前缀，让检索知道这是哪个药的答案
        prefixed_text = f"[{drug_prefix}] {ans_chunk}"
        all_chunks.append({
            "id": f"{item['id']}_c{j}",
            "text": prefixed_text,
            "source": question,
        })

    return all_chunks


def build_vector_db():
    print("=" * 60)
    print("第2步：混合切分 + 向量化 + 入库")
    print("=" * 60)

    data_path = os.path.join(DATA_DIR, "drug_knowledge.json")
    if not os.path.exists(data_path):
        print("错误：请先运行 01_prepare_data.py")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
    print(f"加载 {len(knowledge)} 条药品知识", flush=True)

    # 混合切分
    print(f"切分方式：问答对 → 句子边界 → 字数兜底（{CHUNK_SIZE}字）", flush=True)
    print(f"每个子块带药品名前缀，解决问答断裂问题", flush=True)

    all_chunks = []
    for item in knowledge:
        chunks = chunk_qa_pair(item, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    # 统计切分情况
    lens = [len(c["text"]) for c in all_chunks]
    uncut = sum(1 for item in knowledge if len(item["text"]) <= CHUNK_SIZE)
    cut_items = len(knowledge) - uncut

    print(f"\n切分完成：{len(knowledge)} 条 → {len(all_chunks)} 块", flush=True)
    print(f"  未切分（整条保留）: {uncut} 条", flush=True)
    print(f"  已切分（按句子）: {cut_items} 条 → {len(all_chunks)-uncut} 块", flush=True)
    print(f"  块长度: 最短{min(lens)}字, 最长{max(lens)}字, 平均{sum(lens)//len(lens)}字", flush=True)

    # 显示切分样例
    print(f"\n切分样例（前5块）:", flush=True)
    for i, c in enumerate(all_chunks[:5]):
        print(f"  [{i}] id={c['id']} len={len(c['text'])}字 source={c['source'][:25]}", flush=True)
        print(f"      内容: {c['text'][:100]}...", flush=True)

    # 向量化
    print(f"\n开始向量化（模型: {EMBEDDING_MODEL_NAME}, 维度: {EMBEDDING_DIM}）...", flush=True)
    texts = [c["text"] for c in all_chunks]
    t0 = time.time()
    embeddings = encode_texts(texts)
    t1 = time.time()
    print(f"向量化完成: {len(embeddings)} 个向量, 耗时 {t1-t0:.1f}s", flush=True)

    # 保存
    import numpy as np
    vectors = np.array(embeddings, dtype=np.float32)
    vec_path = os.path.join(DATA_DIR, "vectors.npy")
    np.save(vec_path, vectors)
    meta_path = os.path.join(DATA_DIR, "chunks.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(c["text"]) for c in all_chunks)
    est_tokens = int(total_chars * 1.5)
    print(f"\n向量库构建完成:", flush=True)
    print(f"  总块数: {len(all_chunks)}")
    print(f"  向量维度: {len(embeddings[0])}")
    print(f"  向量文件: {vec_path} ({vectors.nbytes/1024/1024:.1f} MB)")
    print(f"  全量注入估算 token: ~{est_tokens:,}")
    print(f"  向量检索 top-5 估算 token: ~{int(5 * CHUNK_SIZE * 1.5):,}")
    print(f"\n【改进说明】", flush=True)
    print(f"  1. 按句子切分：不再切断句子（在。！？；处切）", flush=True)
    print(f"  2. 问题前缀：每个子块带[药品名]前缀，检索知道回答什么药", flush=True)
    print(f"  3. 3层递进：问答对→句子→字数兜底", flush=True)
    return all_chunks


if __name__ == "__main__":
    build_vector_db()
    print("\n第2步完成")
