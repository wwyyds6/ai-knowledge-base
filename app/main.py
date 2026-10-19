"""FastAPI 应用入口：只负责组装路由"""
from fastapi import FastAPI

from app.api import chat, documents

app = FastAPI(title="AI Knowledge Base", version="0.1.0")

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}