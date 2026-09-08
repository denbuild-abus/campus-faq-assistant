"""FastAPI 入口：装配流水线、挂载路由与前端静态资源。"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.embedder import Embedder
from app.core.generator import Generator
from app.core.pipeline import FAQPipeline
from app.core.retriever import HybridRetriever
from app.routes.chat import router as chat_router
from app.storage.index_store import IndexStore


def build_pipeline() -> FAQPipeline:
    """生产环境装配：加载已有索引（若有），否则空库等待上传。"""
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(embedder)
    store = IndexStore(settings.index_dir)
    if store.exists():
        retriever = store.load(retriever)
    generator = Generator(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    return FAQPipeline(retriever, generator)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_pipeline()
    yield


def create_app(pipeline: FAQPipeline = None) -> FastAPI:
    """工厂函数：测试可传入 fake pipeline，生产环境走 lifespan 装配。"""
    if pipeline is None:
        app = FastAPI(title="校园新生 FAQ 助手", version="1.0.0", lifespan=lifespan)
    else:
        app = FastAPI(title="校园新生 FAQ 助手", version="1.0.0")
        app.state.pipeline = pipeline

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)

    frontend = Path(__file__).resolve().parent.parent / "frontend"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
    return app


app = create_app()
