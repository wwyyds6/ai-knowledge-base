import os
import time
import uuid

import httpx
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AI Knowledge Base", version="0.1.0")

# ==================== 配置 ====================
ALLOWED_EXT = (".txt", ".md")
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# bge 系列在检索场景下，query 需要加中文指令前缀；文档片段不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# 检索分数阈值：Top-1 相似度低于此值时，认为库里没有相关内容，不调用模型
# 经验值，随语料规模和 embedding 模型调整
MIN_SCORE = 0.45

SYSTEM_PROMPT = (
    "你是一个严谨的技术知识库助手。\n"
    "请只依据【参考资料】回答问题，不要编造资料之外的内容。\n"
    "如果参考资料不足以回答，请明确回答「根据现有资料无法回答」。\n"
    "回答时用 [1][2] 的形式标注引用了哪几条资料。"
)


# ==================== 模型：全局单例，只加载一次 ====================
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


# ==================== 内存向量库（第 1 版；后续换 pgvector） ====================
_documents: dict[str, dict] = {}   # doc_id -> 文档元信息
_chunk_store: list[dict] = []      # 每个元素是一条分块 + 它的向量


# ==================== 工具函数 ====================
def decode_text(raw: bytes) -> str:
    """统一文本解码：去 BOM + 归一化换行"""
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件不是 UTF-8 编码，请另存为 UTF-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 60) -> list[dict]:
    """按固定长度切分，相邻块保留 overlap 个字符的重叠"""
    if chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="overlap 必须在 [0, chunk_size) 区间内")

    text = text.strip()
    if not text:
        return []

    chunks: list[dict] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append({"index": idx, "text": piece, "start": start, "end": end})
            idx += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def retrieve(query: str, top_k: int) -> list[dict]:
    """检索：把 query 向量化，与所有分块算余弦相似度，返回 Top-K"""
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


def build_prompt(query: str, hits: list[dict]) -> str:
    """把检索到的资料拼进 Prompt，并编号供引用"""
    lines = ["【参考资料】"]
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] {h['text']}")
    lines.append("")
    lines.append(f"【问题】{query}")
    lines.append("")
    lines.append("请依据上面的资料回答，并用 [编号] 标注引用。")
    return "\n".join(lines)


def call_llm(prompt: str) -> dict:
    """调用 DeepSeek（OpenAI 兼容协议）"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="未配置 DEEPSEEK_API_KEY，请在启动服务前设置环境变量",
        )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,          # 事实类任务用低温，减少随机编造
        "stream": False,
    }

    t0 = time.time()
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    cost_ms = round((time.time() - t0) * 1000)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"模型调用失败：{resp.status_code} {resp.text[:200]}",
        )

    data = resp.json()
    return {
        "answer": data["choices"][0]["message"]["content"],
        "cost_ms": cost_ms,
        "usage": data.get("usage", {}),
    }


# ==================== 请求模型（Pydantic 校验） ====================
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="检索问题")
    top_k: int = Field(3, ge=1, le=10, description="返回前 K 条")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="问题")
    top_k: int = Field(3, ge=1, le=10, description="检索几条资料")


# ==================== 接口 ====================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并提取纯文本（不索引）"""
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"只支持 {ALLOWED_EXT} 格式")

    raw = await file.read()
    text = decode_text(raw)

    return {
        "filename": filename,
        "size_bytes": len(raw),
        "char_count": len(text),
        "text": text,
    }


@app.post("/documents/chunks")
async def preview_chunks(
    file: UploadFile = File(...),
    chunk_size: int = 300,
    overlap: int = 60,
):
    """只做分块预览，不向量化"""
    raw = await file.read()
    text = decode_text(raw)
    chunks = chunk_text(text, chunk_size, overlap)
    return {
        "filename": file.filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.post("/documents/index")
async def index_document(
    file: UploadFile = File(...),
    chunk_size: int = 300,
    overlap: int = 60,
):
    """上传 → 分块 → 向量化 → 存入内存向量库"""
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"只支持 {ALLOWED_EXT} 格式")

    raw = await file.read()
    text = decode_text(raw)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档没有可用内容")

    model = get_model()
    texts = [c["text"] for c in chunks]

    t0 = time.time()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    encode_ms = round((time.time() - t0) * 1000)

    doc_id = uuid.uuid4().hex[:8]
    for c, vec in zip(chunks, vectors):
        _chunk_store.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": c["index"],
            "text": c["text"],
            "vector": vec,
        })

    _documents[doc_id] = {
        "filename": filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "encode_ms": encode_ms,
    }

    return {
        "doc_id": doc_id,
        "filename": filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "encode_ms": encode_ms,
    }


@app.get("/documents")
def list_documents():
    """列出已索引的文档"""
    return {
        "count": len(_documents),
        "documents": _documents,
        "total_chunks": len(_chunk_store),
    }


@app.post("/search")
def search(req: SearchRequest):
    """语义检索：只返回相关分块，不生成回答"""
    t0 = time.time()
    results = retrieve(req.query, req.top_k)
    return {
        "query": req.query,
        "search_ms": round((time.time() - t0) * 1000),
        "results": results,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG 问答：检索 → (阈值兜底) → 拼 Prompt → 生成 → 返回答案 + 引用来源"""
    # 1. 检索
    hits = retrieve(req.query, req.top_k)

    # 2. ⭐ 分数阈值兜底：Top-1 太差 → 说明库里没有相关内容，不浪费模型调用
    top_score = hits[0]["score"] if hits else 0.0
    if top_score < MIN_SCORE:
        return {
            "query": req.query,
            "answer": "未找到相关内容，当前知识库中没有与你的问题相关的资料。",
            "cost_ms": 0,
            "usage": {},
            "sources": [],
            "note": f"Top-1 相似度 {top_score} 低于阈值 {MIN_SCORE}，已跳过模型调用",
        }

    # 3. 拼 Prompt
    prompt = build_prompt(req.query, hits)

    # 4. 调模型生成
    result = call_llm(prompt)

    # 5. 返回答案 + 引用来源（前端可点开核对原文）
    return {
        "query": req.query,
        "answer": result["answer"],
        "cost_ms": result["cost_ms"],
        "usage": result["usage"],
        "sources": [
            {
                "ref": i,
                "score": h["score"],
                "filename": h["filename"],
                "chunk_index": h["chunk_index"],
                "text": h["text"],
            }
            for i, h in enumerate(hits, start=1)
        ],
    }