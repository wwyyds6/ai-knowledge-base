"""文档接口：上传 / 分块预览 / 索引 / 列表"""
import time
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.chunking import chunk_text, decode_text
from app.config import ALLOWED_EXT
from app.embedding import get_model
from app.retrieval import add_document, list_documents

router = APIRouter(prefix="/documents", tags=["documents"])


def _check_ext(filename: str) -> None:
    """校验扩展名，不在白名单直接 400"""
    if not filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"只支持 {ALLOWED_EXT} 格式")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并提取纯文本（不索引）"""
    filename = file.filename or ""
    _check_ext(filename)

    raw = await file.read()
    text = decode_text(raw)

    return {
        "filename": filename,
        "size_bytes": len(raw),
        "char_count": len(text),
        "text": text,
    }


@router.post("/chunks")
async def preview_chunks(
    file: UploadFile = File(...),
    chunk_size: int = 300,
    overlap: int = 60,
):
    """只做分块预览，不向量化（用于验证分块策略）"""
    raw = await file.read()
    text = decode_text(raw)
    chunks = chunk_text(text, chunk_size, overlap)
    return {
        "filename": file.filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@router.post("/index")
async def index_document(
    file: UploadFile = File(...),
    chunk_size: int = 300,
    overlap: int = 60,
):
    """上传 → 分块 → 向量化 → 存入向量库"""
    filename = file.filename or ""
    _check_ext(filename)

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
    meta = {
        "filename": filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "encode_ms": encode_ms,
    }
    add_document(doc_id, filename, meta, chunks, vectors)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "encode_ms": encode_ms,
    }


@router.get("")
def get_documents():
    """列出已索引的文档"""
    return list_documents()