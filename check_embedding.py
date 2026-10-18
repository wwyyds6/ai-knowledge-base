import time

import numpy as np
from sentence_transformers import SentenceTransformer

print("正在加载模型（首次会从 hf-mirror 下载，约 100MB）...")
t0 = time.time()
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"模型加载耗时：{time.time() - t0:.1f} 秒")

texts = [
    "检索增强生成需要先把文档分块",
    "RAG 的核心是向量检索",
    "今天天气不错",
]

t1 = time.time()
vectors = model.encode(texts, normalize_embeddings=True)
print(f"编码耗时：{time.time() - t1:.3f} 秒")
print(f"向量维度：{vectors.shape}")

sim_01 = float(np.dot(vectors[0], vectors[1]))
sim_02 = float(np.dot(vectors[0], vectors[2]))
print(f"句1 vs 句2（语义相关）：{sim_01:.4f}")
print(f"句1 vs 句3（语义无关）：{sim_02:.4f}")