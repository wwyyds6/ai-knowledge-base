"""检索与问答接口"""
import time

from fastapi import APIRouter

from app.config import MIN_SCORE
from app.llm import build_prompt, call_llm
from app.models import ChatRequest, SearchRequest
from app.retrieval import retrieve

router = APIRouter(tags=["chat"])


@router.post("/search")
def search(req: SearchRequest):
    """语义检索：只返回相关分块，不生成回答"""
    t0 = time.time()
    results = retrieve(req.query, req.top_k)
    return {
        "query": req.query,
        "search_ms": round((time.time() - t0) * 1000),
        "results": results,
    }


@router.post("/chat")
def chat(req: ChatRequest):
    """RAG 问答：检索 →（阈值兜底）→ 拼 Prompt → 生成 → 返回答案 + 引用来源"""
    # 1. 检索
    hits = retrieve(req.query, req.top_k)

    # 2. 分数阈值兜底：Top-1 太差说明库里没有相关内容，不浪费模型调用
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