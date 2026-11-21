"""The training job: graph -> fastgraphml model -> embeddings -> store.

This is the only file which uses fastgraphml. Everything else in the backend just
reads what this job wrote.
"""

from __future__ import annotations

import contextlib
import re
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from . import config


class Job:
    """State of the (single) training job, polled by the web UI."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error
        self.step = ""
        self.epoch = 0
        self.epochs = 0
        self.metrics: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "step": self.step,
            "epoch": self.epoch,
            "epochs": self.epochs,
            "metrics": self.metrics,
            "error": self.error,
        }


class EpochWatcher:
    """fastgraphml reports its progress with print(). This stand-in for sys.stdout
    passes everything on and picks the epoch number out of 'Epoch: 003, ...'."""

    def __init__(self, job: Job, stream: Any):
        self.job, self.stream = job, stream

    def write(self, text: str) -> int:
        match = re.search(r"Epoch: (\d+)", text)
        if match:
            self.job.epoch = int(match.group(1))
        return int(self.stream.write(text))

    def flush(self) -> None:
        self.stream.flush()


def whiten(emb: np.ndarray, dimensions: int = 32) -> np.ndarray:
    """Post-processing for similarity search.

    After a short unsupervised training most of the variance of the embeddings
    sits in one or two directions, so every cosine similarity is close to 1 and
    the ranking is dominated by that direction. PCA whitening keeps the strongest
    `dimensions` directions and gives each of them the same weight.
    """
    from sklearn.decomposition import PCA

    dimensions = min(dimensions, emb.shape[1], emb.shape[0] - 1)
    whitened = PCA(n_components=dimensions, whiten=True, random_state=7).fit_transform(
        emb
    )
    return np.asarray(whitened, dtype=np.float32)


def project_2d(emb: np.ndarray) -> np.ndarray:
    """2D positions for the map: UMAP if it is installed, otherwise PCA."""
    try:
        import umap

        xy = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=7).fit_transform(emb)
    except ImportError:
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2).fit_transform(emb)
    xy = np.asarray(xy, dtype=np.float32)
    xy -= xy.min(axis=0)
    xy /= np.maximum(xy.max(axis=0), 1e-9)  # scale to the unit square
    return xy


def train(
    job: Job, store: Any, model_name: str, epochs: int, embedding_size: int
) -> None:
    """Runs in a background thread, started by POST /api/train."""
    from fastgraphml import GAT, SAGE

    try:
        started = time.time()
        job.epochs, job.epoch, job.metrics, job.error = epochs, 0, {}, None
        model_cls = {"SAGE": SAGE, "GAT": GAT}[model_name]
        import torch_geometric.transforms as T

        # the features are word counts: scale every row to sum 1 before training
        params = dict(
            embedding_size=embedding_size,
            sizes=[10, 10],
            batch_size=512,
            transform=T.NormalizeFeatures(),
        )

        job.step = "Loading the graph"
        if config.mode() == "arangodb":
            # 1. the low-code part: ArangoDB graph + metagraph -> model
            model = model_cls(store.db, config.GRAPH, config.METAGRAPH, **params)
        else:
            from torch_geometric.datasets import Amazon

            data = Amazon(str(config.CACHE_DIR / "amazon"), "Computers")[0]
            model = model_cls(pyg_graph=data, **params)
            store.set_graph(data.y.numpy(), data.edge_index.numpy())

        # 2. train, the epoch number is picked out of the printed progress
        job.step = "Training"
        with contextlib.redirect_stdout(EpochWatcher(job, sys.stdout)):
            model._train(epochs=epochs, lr=0.005, ckp_path=None, best_model_path=None)

        # 3. embeddings + the node classification check fastgraphml offers
        job.step = "Computing embeddings"
        emb = model.get_embeddings()
        val_acc, test_acc = model.val()
        emb = whiten(emb)

        job.step = "Laying out the map"
        xy = project_2d(emb)

        job.step = "Storing"
        if config.mode() == "arangodb":
            # 4. one document per product: same _key as the product, embedding,
            #    label, class_name and source_id
            keys = model.graph_util.node_keys.get(None)
            if not keys:
                raise RuntimeError(
                    "The ArangoDB keys of the products are unknown, so the embeddings "
                    "cannot be linked to them. Upgrade adbpyg-adapter to 1.1 or newer."
                )
            model.graph_util.store_embeddings(
                emb,
                collection_name=config.EMBEDDINGS,
                batch_size=500,
                class_mapping=config.CATEGORIES,
            )
            store.save_map([str(k) for k in keys], xy)
        else:
            store.set_embeddings(emb, xy)

        job.metrics = {
            "model": model_name,
            "embedding_size": int(emb.shape[1]),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "seconds": round(time.time() - started, 1),
        }
        job.step, job.state = "Done", "done"
    except Exception as e:  # shown in the UI
        traceback.print_exc()
        job.error, job.state = f"{type(e).__name__}: {e}", "error"
