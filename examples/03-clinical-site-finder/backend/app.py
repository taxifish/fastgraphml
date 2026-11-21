"""REST API of the clinical site finder.

uvicorn backend.app:app --port 8001          # from examples/03-clinical-site-finder
FGML_MODE=offline uvicorn backend.app:app --port 8001   # without ArangoDB
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, pipeline
from .store import ArangoStore, MemoryStore

app = FastAPI(title="Clinical site finder", description=__doc__)
app.add_middleware(  # the Vite dev server runs on another port
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

job = pipeline.Job()
_store: Any = None


def get_store() -> Any:
    """Connects on first use, so the app also starts while the database is down."""
    global _store
    if _store is None:
        try:
            _store = (
                MemoryStore()
                if config.mode() == "offline"
                else ArangoStore(config.connect())
            )
        except Exception as e:
            raise HTTPException(
                503,
                f"Could not connect to ArangoDB ({e}). Check examples/.env, "
                "or start the backend with FGML_MODE=offline.",
            )
    return _store


def need_embeddings(store: Any) -> None:
    if not store.has_embeddings():
        raise HTTPException(409, "There are no embeddings yet. Train the model first.")


class TrainRequest(BaseModel):
    epochs: int = Field(15, ge=1, le=100)
    embedding_size: int = Field(64, ge=8, le=256)


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
    if not has_graph:
        hint = (
            "The clinical trial graph is not in the database yet. Load it with: "
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
    if not store.has_graph():
        raise HTTPException(409, "Load the graph first: python -m backend.load_data")
    with job.lock:
        if job.state == "running":
            raise HTTPException(409, "A training is already running.")
        job.state = "running"
    threading.Thread(
        target=pipeline.train,
        args=(job, store, request.epochs, request.embedding_size),
        daemon=True,
    ).start()
    return job.as_dict()


@app.get("/api/facets")
def facets() -> Dict[str, Any]:
    """Indications and countries for the query form."""
    store = get_store()
    return (
        dict(store.facets())
        if store.has_graph()
        else {"indications": [], "countries": []}
    )


@app.get("/api/recommend")
def recommend(
    indication: str,
    country: Optional[str] = None,
    k: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    """Sites for a new trial in an indication.

    1. Experienced sites: the ones that already ran trials in the indication
       (a graph traversal).
    2. Their embeddings are averaged, weighted by the number of trials, into one
       vector describing 'a site that runs this kind of trial'.
    3. All sites are ranked by cosine similarity to that vector. The best ranked
       sites without any trial in the indication are the lookalikes: candidates a
       search by past experience can never find.
    """
    store = get_store()
    need_embeddings(store)
    found = store.indication(indication)
    if found is None:
        raise HTTPException(404, f"No indication with key {indication}.")

    experience: Dict[str, int] = store.experience(indication)
    if not experience:
        return {
            "indication": found,
            "experienced": [],
            "lookalikes": [],
            "seed_sites": 0,
        }
    vectors = store.vectors(list(experience))
    centroid = np.zeros(len(next(iter(vectors.values()))), dtype=np.float32)
    for key, vector in vectors.items():
        centroid += experience[key] * vector / max(float(np.linalg.norm(vector)), 1e-12)

    ranked = store.rank(centroid, country, limit=100_000)
    experienced = [
        {**site, "trials": experience[site["key"]]}
        for site in ranked
        if site["key"] in experience
    ]
    experienced.sort(key=lambda s: (-s["trials"], -s["score"]))
    lookalikes = [
        {**site, "trials": 0} for site in ranked if site["key"] not in experience
    ]
    return {
        "indication": found,
        "seed_sites": len(vectors),
        "experienced_total": len(experienced),
        "experienced": experienced[:k],
        "lookalikes": lookalikes[:k],
    }


@app.get("/api/sites")
def sites(
    country: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = Query(30, ge=1, le=200),
) -> List[Dict[str, Any]]:
    store = get_store()
    return list(store.sites(country, q, limit)) if store.has_graph() else []


@app.get("/api/sites/{key}")
def site(key: str) -> Dict[str, Any]:
    found = get_store().site(key)
    if found is None:
        raise HTTPException(404, f"No site with key {key}.")
    return dict(found)


@app.get("/api/sites/{key}/similar")
def similar(key: str, k: int = Query(8, ge=1, le=50)) -> List[Dict[str, Any]]:
    """Sites with the most similar embedding, i.e. a similar trial history."""
    store = get_store()
    need_embeddings(store)
    vectors = store.vectors([key])
    if key not in vectors:
        raise HTTPException(404, f"No embedding for site {key}.")
    ranked = store.rank(vectors[key], None, limit=k + 1)
    return [s for s in ranked if s["key"] != key][:k]


@app.get("/api/map")
def embedding_map() -> List[Dict[str, Any]]:
    store = get_store()
    return list(store.map_points()) if store.has_embeddings() else []


# the built React app, if there is one (npm run build in frontend/)
_dist = config.EXAMPLE_DIR / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
