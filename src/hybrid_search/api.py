import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from psycopg_pool import PoolTimeout

from hybrid_search.domain import SearchRequest
from hybrid_search.models import SPECS, OnnxModels
from hybrid_search.search import SearchResponse, SearchService
from hybrid_search.store import CatalogNotReady, Store


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        if service is not None:
            app.state.service = service
            yield
            return
        models = OnnxModels(threads=int(os.environ.get("INFERENCE_THREADS", "2")))
        models.warmup()
        store = Store(os.environ["DATABASE_URL"])
        app.state.service = SearchService(store, models)
        try:
            yield
        finally:
            store.close()

    app = FastAPI(
        title="Hybrid Product Search",
        version="0.1.0",
        lifespan=lifespan,
        description="Synthetic local demo: PostgreSQL FTS, E5/pgvector, RRF and BGE. "
        "Scores have different scales; BGE logits are not probabilities. "
        "The demo has no authentication and should be bound to loopback.",
    )
    slots = threading.BoundedSemaphore(4)

    assets = Path(__file__).with_name("static")
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def explorer():
        return FileResponse(assets / "index.html")

    @app.exception_handler(CatalogNotReady)
    @app.exception_handler(psycopg.Error)
    @app.exception_handler(PoolTimeout)
    async def unavailable(_request, _error):
        return JSONResponse(
            status_code=503,
            content={"detail": "catalog unavailable; check indexing and database health"},
        )

    @app.get("/health", responses={503: {"description": "Catalog or model signature unavailable"}})
    def health(request: Request):
        current = request.app.state.service
        state = current.store.state(current.models.signature)
        return dict(status="ok", **state)

    @app.get("/models")
    def model_contract():
        return SPECS

    @app.post(
        "/search",
        response_model=SearchResponse,
        responses={503: {"description": "Catalog unavailable or local inference queue full"}},
    )
    def search(body: SearchRequest, request: Request):
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "local inference queue is full", headers={"Retry-After": "1"})
        try:
            return request.app.state.service.search(body)
        finally:
            slots.release()

    return app
