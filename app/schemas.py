"""Pydantic 请求/响应模型。"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class Source(BaseModel):
    document: str
    chunk_id: int
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
