"""文本解码与分块"""
from fastapi import HTTPException


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