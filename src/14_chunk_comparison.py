"""
14_chunk_comparison.py
对比3种切分方式：固定长度 vs 句子边界 vs 语义切分
取1000条数据做小规模对比，看检索准确性差异
"""
import os, sys, json, re, time, numpy as np, torch, torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 加载模型
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
mdl = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
mdl.eval()

SENTENCE_ENDINGS = re.compile(r'[。！？；\n]')

def embed_batch(texts):
    inputs = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**inputs)
    am = inputs["attention_mask"]
    te = out.last_hidden_state
    ime = am.unsqueeze(-1).expand(te.size()).float()
    emb = torch.sum(te * ime, 1) / torch.clamp(ime.sum(1), min=1e-9)
    return F.normalize(emb, p=2, dim=1).numpy()

def embed_one(text):
    return embed_batch([text])[0]


# ============ 切分方式1：固定长度滑动窗口 ============
def chunk_fixed(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text): break
        start = end - overlap
    return chunks


# ============ 切分方式2：句子边界 + 药品名前缀 ============
def split_by_sentences(text, max_length=300):
    sentences = []
    last = 0
    for m in SENTENCE_ENDINGS.finditer(text):
        sentences.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        sentences.append(text[last:])
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_length:
            current += s
        else:
            if current: chunks.append(current)
            if len(s) > max_length:
                for i in range(0, len(s), max_length):
                    chunks.append(s[i:i+max_length])
            else:
                current = s
                continue
            current = ""
    if current: chunks.append(current)
    return chunks

def chunk_sentence(text, question, max_size=300):
    if len(text) <= max_size:
        return [text]
    parts = text.split("\n答案：", 1)
    answer = parts[1] if len(parts) > 1 else text
    prefix = question[:30]
    sub = split_by_sentences(answer, max_size - len(prefix) - 10)
    return [f"[{prefix}] {s}" for s in sub]


# ============ 切分方式3：语义切分 ============
def chunk_semantic(text, question, max_size=300, threshold=0.5):
    """
    语义切分：把答案按句子拆开，算相邻句子的embedding相似度，
    相似度低于threshold处切分，保证每块语义聚焦
    """
    if len(text) <= max_size:
        return [text]
    parts = text.split("\n答案：", 1)
    answer = parts[1] if len(parts) > 1 else text
    prefix = question[:30]

    # 先按句子拆
    sentences = []
    last = 0
    for m in SENTENCE_ENDINGS.finditer(answer):
        s = answer[last:m.end()].strip()
        if s: sentences.append(s)
        last = m.end()
    if last < len(answer):
        s = answer[last:].strip()
        if s: sentences.append(s)

    if len(sentences) <= 1:
        return [f"[{prefix}] {answer}"]

    # 算相邻句子的相似度
    sent_vecs = embed_batch(sentences)
    sims = [float(F.cosine_similarity(
        torch.tensor(sent_vecs[i]), torch.tensor(sent_vecs[i+1]), dim=0
    )) for i in range(len(sentences)-1)]

    # 在语义转折点（相似度<threshold）切分
    chunks = []
    current_sents = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(len(sims)):
        sim = sims[i]
        next_sent = sentences[i+1]

        # 切分条件：语义转折 OR 超长
        if sim < threshold or current_len + len(next_sent) > max_size:
            chunks.append(f"[{prefix}] " + "".join(current_sents))
            current_sents = [next_sent]
            current_len = len(next_sent)
        else:
            current_sents.append(next_sent)
            current_len += len(next_sent)

    if current_sents:
        chunks.append(f"[{prefix}] " + "".join(current_sents))

    return chunks


def compare():
    print("=" * 70)
    print("  3种切分方式对比测试")
    print("=" * 70)

    # 加载500条数据（小规模快速对比）
    knowledge = json.load(open(os.path.join(DATA_DIR, "drug_knowledge.json"), encoding="utf-8"))
    test_data = knowledge[:500]
    print(f"测试数据: {len(test_data)} 条\n")

    # 3种切分
    results = {}
    for method_name, chunk_fn in [
        ("固定长度", lambda item: chunk_fixed(item["text"], 300, 50)),
        ("句子边界", lambda item: chunk_sentence(item["text"], item["question"], 300)),
        ("语义切分", lambda item: chunk_semantic(item["text"], item["question"], 300, 0.5)),
    ]:
        print(f"\n{'='*50}")
        print(f"切分方式: {method_name}")
        print(f"{'='*50}")

        all_chunks = []
        for item in test_data:
            chunks = chunk_fn(item)
            for j, c in enumerate(chunks):
                all_chunks.append({"text": c, "source": item["question"], "id": f"{item['id']}_{j}"})

        lens = [len(c["text"]) for c in all_chunks]
        print(f"  总块数: {len(all_chunks)}")
        print(f"  平均长度: {sum(lens)//len(lens)}字, 最短{min(lens)}, 最长{max(lens)}")

        # 向量化
        t0 = time.time()
        texts = [c["text"] for c in all_chunks]
        vecs = embed_batch(texts)
        vecs = np.array(vecs, dtype=np.float32)
        t1 = time.time()
        print(f"  向量化: {t1-t0:.1f}s")

        # 检索测试：用每条原始问题检索，看top1是不是对应自己的块
        correct = 0
        total = 0
        sim_scores = []

        for item in test_data[:100]:  # 测前100条
            q_vec = embed_one(item["question"])
            norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
            sims = norms @ q_vec
            top_idx = np.argsort(sims)[::-1][0]
            top_chunk = all_chunks[top_idx]
            top_sim = float(sims[top_idx])

            # 判断top1是否来自同一个药品（source匹配）
            is_correct = item["question"] == top_chunk["source"]
            if is_correct:
                correct += 1
            sim_scores.append(top_sim)
            total += 1

        accuracy = correct / total * 100
        avg_sim = sum(sim_scores) / len(sim_scores)
        print(f"  检索准确率: {accuracy:.1f}% ({correct}/{total} top1命中正确药品)")
        print(f"  平均top1相似度: {avg_sim:.4f}")

        results[method_name] = {
            "chunks": len(all_chunks),
            "avg_len": sum(lens)//len(lens),
            "accuracy": accuracy,
            "avg_sim": avg_sim,
            "embed_time": t1-t0,
        }

    # 汇总对比
    print(f"\n{'='*70}")
    print(f"  汇总对比")
    print(f"{'='*70}")
    print(f"{'方式':<12} {'块数':<8} {'平均长度':<10} {'准确率':<10} {'平均相似度':<12} {'向量化耗时':<10}")
    print(f"{'-'*70}")
    for name, r in results.items():
        print(f"{name:<12} {r['chunks']:<8} {r['avg_len']:<10} {r['accuracy']:.1f}%{'':<4} {r['avg_sim']:.4f}{'':<5} {r['embed_time']:.1f}s")
    print(f"{'='*70}")

    # 结论
    best = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n【结论】")
    print(f"  准确率最高: {best[0]} ({best[1]['accuracy']:.1f}%)")
    print(f"  语义切分 vs 句子边界: 准确率差 {results['语义切分']['accuracy'] - results['句子边界']['accuracy']:+.1f}%")
    print(f"  语义切分 vs 固定长度: 准确率差 {results['语义切分']['accuracy'] - results['固定长度']['accuracy']:+.1f}%")
    print(f"\n  语义切分向量化耗时: {results['语义切分']['embed_time']:.1f}s (需额外算句子间相似度)")
    print(f"  句子边界向量化耗时: {results['句子边界']['embed_time']:.1f}s")


if __name__ == "__main__":
    compare()
