"""Trains the recommender model in the terminal and stores the embeddings.

    python -m backend.train --epochs 10 --model SAGE

Does the same as the Train button of the web page, but shows the whole output,
which is what you want when something fails.
"""

from __future__ import annotations

import argparse
import sys

from . import config, pipeline
from .store import ArangoStore, MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", choices=["SAGE", "GAT"], default="SAGE")
    parser.add_argument("--embedding-size", type=int, default=64)
    args = parser.parse_args()

    store = (
        MemoryStore() if config.mode() == "offline" else ArangoStore(config.connect())
    )
    if config.mode() == "arangodb" and not store.has_graph():
        sys.exit(
            "The product graph is not in the database: python -m backend.load_data"
        )
    job = pipeline.Job()
    pipeline.train(job, store, args.model, args.epochs, args.embedding_size)
    if job.state != "done":
        sys.exit(f"Training failed: {job.error}")
    print(f"Done: {job.metrics}")


if __name__ == "__main__":
    main()
