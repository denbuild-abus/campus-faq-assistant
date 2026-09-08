"""HTTP 接口：健康检查、问答、文档上传。"""
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.chunker import chunk_document
from app.core.loader import load_document
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.storage.index_store import IndexStore

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipe = request.app.state.pipeline
    return HealthResponse(status="ok", indexed_chunks=len(pipe.retriever.chunks))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    pipe = request.app.state.pipeline
    result = pipe.ask(req.query, top_k=settings.top_k)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=pipe.generator.model,
    )


@router.post("/documents")
async def upload_documents(
    files: list[UploadFile] = File(...), request: Request = None
) -> JSONResponse:
    pipe = request.app.state.pipeline
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for f in files:
        raw = await f.read()
        dest = docs_dir / f.filename
        dest.write_bytes(raw)
        try:
            text = load_document(dest)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        chunks = chunk_document(text, settings.chunk_size, settings.chunk_overlap)
        pipe.retriever.add(chunks, [{"document": f.filename} for _ in chunks])
        added += len(chunks)

    IndexStore(settings.index_dir).save(pipe.retriever)
    return JSONResponse(
        {
            "status": "ok",
            "files": len(files),
            "added_chunks": added,
            "total_chunks": len(pipe.retriever.chunks),
        }
    )
