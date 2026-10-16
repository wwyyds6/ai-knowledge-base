from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="AI Knowledge Base", version="0.1.0")

ALLOWED_EXT = (".txt", ".md")


@app.get("/health")
def health():
    """健康检查：确认服务活着"""
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并提取纯文本
    第 1 版只支持 TXT / Markdown（PDF 后续再补）
    """
    # 1. 校验扩展名
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"只支持 {ALLOWED_EXT} 格式")

    # 2. 读取文件内容
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")

    # 3. 解码为文本（必须是 UTF-8）
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件不是 UTF-8 编码，请另存为 UTF-8")

    # 4. 返回：文件名 + 字符数 + 文本内容
    return {
        "filename": filename,
        "size_bytes": len(raw),
        "char_count": len(text),
        "text": text,
    }