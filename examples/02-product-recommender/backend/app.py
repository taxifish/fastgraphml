"""REST API of the product recommender.

uvicorn backend.app:app --port 8000          # from examples/02-product-recommender
FGML_MODE=offline uvicorn backend.app:app    # without ArangoDB
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, pipeline
from .store import ArangoStore, MemoryStore

app = FastAPI(title="Product recommender", description=__doc__)
app.add_middleware(  # the Vite dev server runs on another port
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

job = pipeline.Job()
_store: Any = None
_store_error: Optional[str] = None


def get_store() -> Any:
    """Connects on first use, so the app also starts while the database is down."""
    global _store, _store_error
    if _store is None:
        try:
            _store = (
                MemoryStore()
                if config.mode() == "offline"
                else ArangoStore(config.connect())
            )
            _store_error = None
        except Exception as e:
            _store_error = (
                f"Could not connect to ArangoDB ({e}). Check examples/.env, "
                "or start the backend with FGML_MODE=offline."
            )
            raise HTTPException(503, _store_error)
    return _store


class TrainRequest(BaseModel):
    model: str = Field("SAGE", pattern="^(SAGE|GAT)$")
    epochs: int = Field(5, ge=1, le=100)
    embedding_size: int = Field(64, ge=8, le=512)


@app.get("/api/status")
def status() -> Dict[str, Any]:
    try:
        store = get_store()
    except HTTPException as e:
        return {
            "mode": config.mode(),
            "error": e.detail,
            "has_graph": False,
            "has_embeddings": False,
            "counts": {},
            "job": job.as_dict(),
        }
    has_graph = store.has_graph()
    hint = None
    if not has_graph and config.mode() == "arangodb":
        hint = (
            "The product graph is not in the database yet. Load it with: "
            "python -m backend.load_data"
        )
    return {
        "mode": config.mode(),
        "error": hint,
        "has_graph": has_graph,
        "has_embeddings": store.has_embeddings(),
        "counts": store.counts(),
        "job": job.as_dict(),
    }


@app.post("/api/train", status_code=202)
def train(request: TrainRequest) -> Dict[str, Any]:
    store = get_store()
    if config.mode() == "arangodb" and not store.has_graph():
        raise HTTPException(
            409, "Load the product graph first: python -m backend.load_data"
        )
    with job.lock:
        if job.state == "running":
            raise HTTPException(409, "A training is already running.")
        job.state = "running"
    threading.Thread(
        target=pipeline.train,
        args=(job, store, request.model, request.epochs, request.embedding_size),
        daemon=True,
    ).start()
    return job.as_dict()


@app.get("/api/categories")
def categories() -> List[Dict[str, Any]]:
    store = get_store()
    return store.categories() if store.has_graph() else []


@app.get("/api/products")
def products(
    category: Optional[int] = None,
    q: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    store = get_store()
    return store.products(category, q, limit, offset) if store.has_graph() else []


def _existing(key: str) -> Dict[str, Any]:
    found = get_store().get(key)
    if found is None:
        raise HTTPException(404, f"No product with key {key}.")
    return dict(found)


@app.get("/api/products/{key}")
def product(key: str) -> Dict[str, Any]:
    return _existing(key)


@app.get("/api/products/{key}/similar")
def similar(key: str, k: int = Query(12, ge=1, le=100)) -> List[Dict[str, Any]]:
    """Recommendations: the products with the most similar embeddings."""
    _existing(key)
    store = get_store()
    if not store.has_embeddings():
        raise HTTPException(409, "There are no embeddings yet. Train a model first.")
    return list(store.similar(key, k))


@app.get("/api/products/{key}/bought-together")
def bought_together(
    key: str, limit: int = Query(12, ge=1, le=100)
) -> List[Dict[str, Any]]:
    """For comparison: the products directly connected in the graph."""
    _existing(key)
    return list(get_store().bought_together(key, limit))


@app.get("/api/map")
def embedding_map() -> Dict[str, Any]:
    store = get_store()
    if not store.has_embeddings():
        return {"keys": [], "x": [], "y": [], "labels": []}
    return dict(store.map_points())


# the built React app, if there is one (npm run build in frontend/)
_dist = config.EXAMPLE_DIR / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
