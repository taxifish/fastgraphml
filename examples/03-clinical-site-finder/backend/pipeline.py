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


def project_2d(emb: np.ndarray) -> np.ndarray:
    """2D positions for the map: UMAP if it is installed, otherwise PCA."""
    try:
        import umap

        xy = umap.UMAP(n_neighbors=15, min_dist=0.4, random_state=7).fit_transform(emb)
    except ImportError:
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2).fit_transform(emb)
    xy = np.asarray(xy, dtype=np.float32)
    xy -= xy.min(axis=0)
    xy /= np.maximum(xy.max(axis=0), 1e-9)  # scale to the unit square
    return xy


def train(job: Job, store: Any, epochs: int, embedding_size: int) -> None:
    """Runs in a background thread, started by POST /api/train."""
    import torch_geometric.transforms as T

    from fastgraphml import METAPATH2VEC

    try:
        started = time.time()
        job.epochs, job.epoch, job.metrics, job.error = epochs, 0, {}, None
        params = dict(
            metapaths=config.METAPATH,
            key_node=config.SITES,
            embedding_size=embedding_size,
            walk_length=20,
            context_size=7,
            walks_per_node=10,
            num_negative_samples=5,
            sparse=True,
            # adds the reversed edge types the metapath walks back on
            transform=T.ToUndirected(),
        )

        job.step = "Loading the graph"
        if config.mode() == "arangodb":
            # 1. the low-code part: ArangoDB graph + metagraph -> model
            db = store.db
            counts = {
                name: db.collection(name).count()
                for name in (config.SITES, config.TRIALS, config.INDICATIONS)
            }
            model = METAPATH2VEC(
                db, config.GRAPH, config.METAGRAPH, num_nodes_dict=counts, **params
            )
        else:
            from . import data

            model = METAPATH2VEC(pyg_graph=data.to_pyg(store.records), **params)

        # 2. train, the epoch number is picked out of the printed progress
        job.step = "Training"
        with contextlib.redirect_stdout(EpochWatcher(job, sys.stdout)):
            model._train(epochs=epochs, lr=0.03, ckp_path=None, best_model_path=None)

        # 3. METAPATH2VEC returns one array per node type of the metapath;
        #    the app only needs the sites
        job.step = "Computing embeddings"
        emb = model.get_embeddings()[config.SITES]
        val_acc, test_acc = model.val()

        job.step = "Laying out the map"
        xy = project_2d(emb)

        job.step = "Storing"
        if config.mode() == "arangodb":
            keys = model.graph_util.node_keys.get(config.SITES)
            if not keys:
                raise RuntimeError(
                    "The ArangoDB keys of the sites are unknown, so the embeddings "
                    "cannot be linked to them. Upgrade adbpyg-adapter to 1.1 or newer."
                )
            # 4. one document per site: same _key as the site, embedding, label,
            #    class_name and source_id
            model.graph_util.store_embeddings(
                emb,
                collection_name=config.EMBEDDINGS,
                node_type=config.SITES,
                class_mapping=config.AREAS,
            )
            store.save_map([str(k) for k in keys], xy)
        else:
            store.set_embeddings(emb, xy)

        job.metrics = {
            "embedding_size": int(emb.shape[1]),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "seconds": round(time.time() - started, 1),
        }
        job.step, job.state = "Done", "done"
    except Exception as e:  # shown in the UI
        traceback.print_exc()
        job.error, job.state = f"{type(e).__name__}: {e}", "error"
