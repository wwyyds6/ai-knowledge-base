"""向量化：模型懒加载单例"""
import time

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """懒加载：第一次调用时加载模型，之后复用（首次加载约 20 秒）"""
    global _model
    if _model is None:
        print("[embedding] 正在加载模型...")
        t0 = time.time()
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[embedding] 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return _model