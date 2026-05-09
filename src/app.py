"""FastAPI HTTP 服务：``POST /chat``（RAG 问答，契约见作业说明）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import rag_answer

_DEFAULT_CHROMA = Path(__file__).resolve().parent.parent / "data" / "chroma_db"


def _chroma_dir() -> Path:
    raw = os.environ.get("CHROMA_PERSIST_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_CHROMA.resolve()


def _collection_name() -> str:
    return os.environ.get("CHROMA_COLLECTION", "pdf_chunks")


def _top_k() -> int:
    try:
        return max(1, int(os.environ.get("RAG_TOP_K", "4")))
    except ValueError:
        return 4


app = FastAPI(
    title="智能文档问答 Agent",
    version="0.1.0",
    description="扫描版 PDF → OCR → 分块 → Chroma + OpenAI RAG；``POST /chat`` 为作业约定接口。",
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户自然语言问题")
    conversation_id: str = Field(..., description="会话 ID（预留多轮；当前实现未使用）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "这个产品的最大并发数是多少？",
                "conversation_id": "user_123",
            }
        }
    }


class SourceItem(BaseModel):
    page: int
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    confidence: Optional[float] = Field(
        None,
        description="检索置信度启发式分数；无检索距离时为 null（见 README 说明）",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "根据产品手册第3页，该产品的最大并发数为 5000 QPS。",
                "sources": [
                    {
                        "page": 3,
                        "content_preview": "性能规格：最大并发数 5000 QPS，响应时间 ≤ 100ms...",
                    }
                ],
                "confidence": 0.89,
            }
        }
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(body: ChatRequest) -> ChatResponse:
    """
    接收问题，执行向量检索与 LLM 生成，返回答案与来源（页码 + 片段预览）。

    ``confidence``：由 Chroma 最优命中距离换算得到 ``1/(1+d)`` 并限制在 ``[0.05,1]``，仅供参考。
    """
    _ = body.conversation_id  # 预留多轮对话 / 会话隔离

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question 不能为空")

    try:
        out = rag_answer(
            q,
            persist_directory=_chroma_dir(),
            collection_name=_collection_name(),
            top_k=_top_k(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"检索或模型调用失败：{e}",
        ) from e

    sources = [
        SourceItem(
            page=int(s.get("page") or 0),
            content_preview=str(s.get("content_preview") or ""),
        )
        for s in out.get("sources") or []
    ]

    return ChatResponse(
        answer=out.get("answer") or "",
        sources=sources,
        confidence=out.get("confidence"),
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
