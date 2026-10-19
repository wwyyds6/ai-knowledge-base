"""大模型调用与 Prompt 构建"""
import time

import httpx
from fastapi import HTTPException

from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)


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
        "temperature": LLM_TEMPERATURE,   # 事实类任务用低温，减少随机编造
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