"""Where the API reads products, co-purchases and embeddings from.

Two implementations with the same methods:

* ArangoStore  - the real thing. Everything is an AQL query against the graph and
                 against the embedding collection written by fastgraphml.
* MemoryStore  - offline mode. The PyG dataset and the embeddings live in numpy
                 arrays, so the app can be tried without a database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import config

Product = Dict[str, Any]


def product(key: str, label: Optional[int]) -> Product:
    """The dataset has no product titles, so the name is category + key."""
    category = (
        config.CATEGORIES.get(label, "Unknown") if label is not None else "Unknown"
    )
    return {
        "key": key,
        "name": f"{category} {key}",
        "category": category,
        "label": label,
    }


class ArangoStore:
    def __init__(self, db: Any):
        self.db = db

    def _aql(self, query: str, **bind_vars: Any) -> List[Any]:
        return list(self.db.aql.execute(query, bind_vars=bind_vars))

    # ---- graph data ------------------------------------------------------- #
    def has_graph(self) -> bool:
        return bool(
            self.db.has_collection(config.PRODUCTS)
            and self.db.has_collection(config.BOUGHT_TOGETHER)
        )

    def counts(self) -> Dict[str, int]:
        if not self.has_graph():
            return {"products": 0, "co_purchases": 0}
        return {
            "products": self.db.collection(config.PRODUCTS).count(),
            "co_purchases": self.db.collection(config.BOUGHT_TOGETHER).count(),
        }

    def categories(self) -> List[Dict[str, Any]]:
        rows = self._aql(
            """
            FOR p IN @@products
                COLLECT label = p.label WITH COUNT INTO n
                SORT label
                RETURN { label: label, count: n }
            """,
            **{"@products": config.PRODUCTS},
        )
        return [
            {
                "label": r["label"],
                "name": config.CATEGORIES.get(r["label"], "Unknown"),
                "count": r["count"],
            }
            for r in rows
        ]

    def products(
        self, label: Optional[int], q: Optional[str], limit: int, offset: int
    ) -> List[Product]:
        rows = self._aql(
            """
            FOR p IN @@products
                FILTER @label == null OR p.label == @label
                FILTER @q == null OR CONTAINS(p._key, @q)
                SORT p._key
                LIMIT @offset, @limit
                RETURN { key: p._key, label: p.label }
            """,
            **{
                "@products": config.PRODUCTS,
                "label": label,
                "q": q or None,
                "limit": limit,
                "offset": offset,
            },
        )
        return [product(r["key"], r["label"]) for r in rows]

    def get(self, key: str) -> Optional[Product]:
        rows = self._aql(
            """
            LET p = DOCUMENT(CONCAT(@products, "/", @key))
            FILTER p != null
            LET degree = LENGTH(FOR v IN 1..1 ANY p @@edges RETURN DISTINCT v._key)
            RETURN { key: p._key, label: p.label, degree: degree }
            """,
            **{
                "products": config.PRODUCTS,
                "@edges": config.BOUGHT_TOGETHER,
                "key": key,
            },
        )
        if not rows:
            return None
        return {
            **product(rows[0]["key"], rows[0]["label"]),
            "degree": rows[0]["degree"],
        }

    def bought_together(self, key: str, limit: int) -> List[Product]:
        """Direct neighbours in the graph, with their embedding similarity (once
        there are embeddings)."""
        start = f"{config.PRODUCTS}/{key}"
        if not self.has_embeddings():
            rows = self._aql(
                """
                FOR v IN 1..1 ANY @start @@edges
                    COLLECT k = v._key, label = v.label
                    SORT k
                    LIMIT @limit
                    RETURN { key: k, label: label, score: null }
                """,
                **{"@edges": config.BOUGHT_TOGETHER, "start": start, "limit": limit},
            )
        else:
            rows = self._aql(
                """
                LET q = DOCUMENT(CONCAT(@emb, "/", @key)).embedding
                FOR v IN 1..1 ANY @start @@edges
                    COLLECT k = v._key, label = v.label
                    LET e = DOCUMENT(CONCAT(@emb, "/", k))
                    LET score = e == null ? null : COSINE_SIMILARITY(e.embedding, q)
                    SORT score DESC
                    LIMIT @limit
                    RETURN { key: k, label: label, score: score }
                """,
                **{
                    "@edges": config.BOUGHT_TOGETHER,
                    "emb": config.EMBEDDINGS,
                    "start": start,
                    "key": key,
                    "limit": limit,
                },
            )
        return [{**product(r["key"], r["label"]), "score": r["score"]} for r in rows]

    # ---- embeddings ------------------------------------------------------- #
    def has_embeddings(self) -> bool:
        return bool(
            self.db.has_collection(config.EMBEDDINGS)
            and self.db.collection(config.EMBEDDINGS).count() > 0
        )

    def similar(self, key: str, k: int) -> List[Product]:
        """Products with the most similar embeddings. Brute force cosine similarity
        in AQL: fine for tens of thousands of documents, use a vector index beyond."""
        rows = self._aql(
            """
            LET q = DOCUMENT(CONCAT(@emb, "/", @key)).embedding
            LET neighbours = (
                FOR v IN 1..1 ANY @start @@edges RETURN DISTINCT v._key
            )
            FOR e IN @@emb
                FILTER e._key != @key
                LET score = COSINE_SIMILARITY(e.embedding, q)
                SORT score DESC
                LIMIT @k
                RETURN { key: e._key, label: e.label, score: score,
                         bought_together: e._key IN neighbours }
            """,
            **{
                "@emb": config.EMBEDDINGS,
                "emb": config.EMBEDDINGS,
                "@edges": config.BOUGHT_TOGETHER,
                "start": f"{config.PRODUCTS}/{key}",
                "key": key,
                "k": k,
            },
        )
        return [
            {
                **product(r["key"], r["label"]),
                "score": r["score"],
                "bought_together": r["bought_together"],
            }
            for r in rows
        ]

    def save_map(self, keys: List[str], xy: np.ndarray) -> None:
        """Adds the 2D map position to the embedding documents."""
        docs = [
            {"_key": key, "xy": [round(float(x), 4), round(float(y), 4)]}
            for key, (x, y) in zip(keys, xy)
        ]
        collection = self.db.collection(config.EMBEDDINGS)
        for start in range(0, len(docs), 1000):
            collection.import_bulk(docs[start : start + 1000], on_duplicate="update")

    def map_points(self) -> Dict[str, Any]:
        rows = self._aql(
            "FOR e IN @@emb FILTER e.xy != null RETURN [e._key, e.xy[0], e.xy[1], e.label]",
            **{"@emb": config.EMBEDDINGS},
        )
        return {
            "keys": [r[0] for r in rows],
            "x": [r[1] for r in rows],
            "y": [r[2] for r in rows],
            "labels": [r[3] for r in rows],
        }


class MemoryStore:
    """Offline mode: the same answers, computed with numpy."""

    def __init__(self) -> None:
        self.keys: List[str] = []
        self.index: Dict[str, int] = {}
        self.labels = np.zeros(0, dtype=np.int64)
        self.neighbours: List[np.ndarray] = []
        self.num_edges = 0
        self.emb: Optional[np.ndarray] = None  # L2 normalised
        self.xy: Optional[np.ndarray] = None
        self.load()

    # ---- persistence, so that a restart does not need a new training ------- #
    @property
    def _file(self) -> Any:
        return config.CACHE_DIR / "products.npz"

    def set_graph(self, labels: np.ndarray, edge_index: np.ndarray) -> None:
        n = len(labels)
        self.keys = [str(i) for i in range(n)]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.labels = labels.astype(np.int64)
        order = np.argsort(edge_index[0], kind="stable")
        src, dst = edge_index[0][order], edge_index[1][order]
        bounds = np.searchsorted(src, np.arange(n + 1))
        self.neighbours = [np.unique(dst[bounds[i] : bounds[i + 1]]) for i in range(n)]
        self.num_edges = int(edge_index.shape[1])
        self._edge_index = edge_index

    def set_embeddings(self, emb: np.ndarray, xy: np.ndarray) -> None:
        emb = emb.astype(np.float32)
        self.emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        self.xy = xy.astype(np.float32)
        config.CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            self._file,
            labels=self.labels,
            edge_index=self._edge_index,
            emb=self.emb,
            xy=self.xy,
        )

    def load(self) -> None:
        if not self._file.is_file():
            return
        with np.load(self._file) as data:
            self.set_graph(data["labels"], data["edge_index"])
            self.emb, self.xy = data["emb"], data["xy"]

    # ---- same methods as ArangoStore ---------------------------------------- #
    def has_graph(self) -> bool:
        return len(self.keys) > 0

    def counts(self) -> Dict[str, int]:
        return {"products": len(self.keys), "co_purchases": self.num_edges // 2}

    def categories(self) -> List[Dict[str, Any]]:
        counts = np.bincount(self.labels, minlength=len(config.CATEGORIES))
        return [
            {"label": label, "name": name, "count": int(counts[label])}
            for label, name in config.CATEGORIES.items()
        ]

    def _product(self, i: int) -> Product:
        return product(self.keys[i], int(self.labels[i]))

    def products(
        self, label: Optional[int], q: Optional[str], limit: int, offset: int
    ) -> List[Product]:
        ids = np.arange(len(self.keys))
        if label is not None:
            ids = ids[self.labels == label]
        if q:
            ids = np.array([i for i in ids if q in self.keys[i]], dtype=np.int64)
        return [self._product(int(i)) for i in ids[offset : offset + limit]]

    def get(self, key: str) -> Optional[Product]:
        if key not in self.index:
            return None
        i = self.index[key]
        return {**self._product(i), "degree": int(len(self.neighbours[i]))}

    def bought_together(self, key: str, limit: int) -> List[Product]:
        i = self.index[key]
        ids = self.neighbours[i]
        if self.emb is not None:
            scores = self.emb[ids] @ self.emb[i]
            order = np.argsort(-scores)[:limit]
            return [
                {**self._product(int(ids[j])), "score": float(scores[j])} for j in order
            ]
        return [{**self._product(int(j)), "score": None} for j in ids[:limit]]

    def has_embeddings(self) -> bool:
        return self.emb is not None

    def similar(self, key: str, k: int) -> List[Product]:
        assert self.emb is not None
        i = self.index[key]
        scores = self.emb @ self.emb[i]
        scores[i] = -np.inf
        neighbours = set(self.neighbours[i].tolist())
        return [
            {
                **self._product(int(j)),
                "score": float(scores[j]),
                "bought_together": int(j) in neighbours,
            }
            for j in np.argsort(-scores)[:k]
        ]

    def map_points(self) -> Dict[str, Any]:
        if self.xy is None:
            return {"keys": [], "x": [], "y": [], "labels": []}
        return {
            "keys": self.keys,
            "x": np.round(self.xy[:, 0], 4).tolist(),
            "y": np.round(self.xy[:, 1], 4).tolist(),
            "labels": self.labels.tolist(),
        }
