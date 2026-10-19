"""向量存储与语义检索（当前为内存实现，后续可换 pgvector）"""
import numpy as np
from fastapi import HTTPException

from app.config import QUERY_INSTRUCTION
from app.embedding import get_model

# ==================== 内存存储 ====================
_documents: dict[str, dict] = {}   # doc_id -> 文档元信息
_chunk_store: list[dict] = []      # 每条分块 + 它的向量


def add_document(doc_id: str, filename: str, meta: dict, chunks: list[dict], vectors) -> None:
    """把分块与对应向量写入存储"""
    for c, vec in zip(chunks, vectors):
        _chunk_store.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": c["index"],
            "text": c["text"],
            "vector": vec,
        })
    _documents[doc_id] = meta


def list_documents() -> dict:
    """返回已索引文档概要"""
    return {
        "count": len(_documents),
        "documents": _documents,
        "total_chunks": len(_chunk_store),
    }


def retrieve(query: str, top_k: int) -> list[dict]:
    """query 向量化 → 与所有分块算余弦相似度 → 返回 Top-K"""
    if not _chunk_store:
        raise HTTPException(status_code=400, detail="向量库为空，请先调用 /documents/index 索引文档")

    model = get_model()
    q_vec = model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)

    # 所有分块向量堆成矩阵，一次矩阵乘法算完所有相似度
    matrix = np.vstack([c["vector"] for c in _chunk_store])
    scores = matrix @ q_vec          # 已归一化 → 点积即余弦相似度

    top_idx = np.argsort(-scores)[:top_k]

    return [
        {
            "score": round(float(scores[i]), 4),
            "doc_id": _chunk_store[i]["doc_id"],
            "filename": _chunk_store[i]["filename"],
            "chunk_index": _chunk_store[i]["chunk_index"],
            "text": _chunk_store[i]["text"],
        }
        for i in top_idx
    ]