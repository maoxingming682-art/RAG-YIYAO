"""
03_retrieve.py
向量检索：用户问题 → top-K 相关块（numpy 暴力搜索）
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, EMBEDDING_MODEL_NAME, TOP_K

VECTORS_PATH = os.path.join(BASE_DIR, "data", "vectors.npy")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.json")

_vectors_cache = None
_chunks_cache = None
_embedder_cache = None


def _load_vector_db():
    global _vectors_cache, _chunks_cache
    if _vectors_cache is not None:
        return _vectors_cache, _chunks_cache
    _vectors_cache = np.load(VECTORS_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        _chunks_cache = json.load(f)
    return _vectors_cache, _chunks_cache


def get_embedding(text):
    """用transformers直接加载bge获取向量"""
    global _embedder_cache
    if _embedder_cache is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer, AutoModel
        _embedder_cache = (
            AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
            AutoModel.from_pretrained(EMBEDDING_MODEL_NAME),
        )
        _embedder_cache[1].eval()
    tokenizer, model = _embedder_cache
    import torch.nn.functional as F
    inputs = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()


def cosine_similarity(query_vec, all_vectors):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    all_norms = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-10)
    return all_norms @ query_norm


def retrieve(question, top_k=TOP_K):
    print("=" * 60)
    print("第3步：向量检索")
    print("=" * 60)
    print(f"问题: {question}")
    print(f"top-K: {top_k}")

    t0 = time.time()
    vectors, chunks = _load_vector_db()
    t1 = time.time()
    print(f"加载向量库: {t1-t0:.3f}s ({len(vectors)} 块)")

    t2 = time.time()
    query_embedding = get_embedding(question)
    query_vec = np.array(query_embedding, dtype=np.float32)
    t3 = time.time()
    print(f"问题向量化: {t3-t2:.3f}s")

    t4 = time.time()
    sims = cosine_similarity(query_vec, vectors)
    top_indices = np.argsort(sims)[::-1][:top_k]
    t5 = time.time()
    print(f"相似度搜索: {t5-t4:.4f}s")

    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "id": chunks[idx]["id"],
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "similarity": float(sims[idx]),
        })

    print(f"\n检索结果 ({len(retrieved)} 条):")
    for i, r in enumerate(retrieved):
        print(f"\n  [{i+1}] 相似度: {r['similarity']:.4f} | 来源: {r['source'][:30]}")
        print(f"      内容: {r['text'][:100]}...")
    return retrieved


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "依托度酸片怎么服用？"
    results = retrieve(q)
    print(f"\n第3步完成，检索到 {len(results)} 条")
