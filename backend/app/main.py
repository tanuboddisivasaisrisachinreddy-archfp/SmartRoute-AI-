"""SmartRoute AI – FastAPI application entry point.

Run locally (from the repository root):
    uvicorn backend.app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.app.api.routes import router
from backend.app.core.logging import setup_logging
from backend.app.core.settings import settings
from backend.app.core.state import state

setup_logging(settings.log_level)
log = logging.getLogger("smartroute")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.engine is None and state.startup_error is None:
        state.load()
    yield
    state.trainer.stop()


app = FastAPI(
    title="SmartRoute AI API",
    description="Reinforcement Learning for Dynamic Route Optimization in Hyderabad. Compares Dijkstra, "
                "tabular Q-Learning and a Deep Q-Network on the OpenStreetMap road graph of Hyderabad.",
    version=settings.version,
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_origin_regex=r"https://.*\.onrender\.com" if settings.environment == "production" else None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api"):
        log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t) * 1000)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc.__class__.__name__}"})


app.include_router(router)

# Single-service deployment (Docker image): when the built frontend is present,
# the API also serves it, so frontend and API share one URL (no CORS needed).
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))


@app.get("/", include_in_schema=False)
def root():
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}


if (FRONTEND_DIST / "index.html").exists():

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        root_dir = FRONTEND_DIST.resolve()
        f = (root_dir / full_path).resolve()
        if f.is_file() and root_dir in f.parents:
            return FileResponse(f)
        return FileResponse(root_dir / "index.html")  # client-side routes (/route, /traffic, ...)
