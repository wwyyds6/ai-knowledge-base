"""接口请求模型（Pydantic 校验）"""
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="检索问题")
    top_k: int = Field(3, ge=1, le=10, description="返回前 K 条")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="问题")
    top_k: int = Field(3, ge=1, le=10, description="检索几条资料")