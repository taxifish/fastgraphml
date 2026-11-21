"""Settings of the product recommender, read from the environment / .env file."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = EXAMPLE_DIR / ".cache"

# The Amazon Computer Products graph, as loaded by arango-datasets
GRAPH = "product_graph"
PRODUCTS = "Computer_Products"  # vertex collection
BOUGHT_TOGETHER = "bought_together"  # edge collection
EMBEDDINGS = "product_embeddings"  # collection written by fastgraphml

# Which ArangoDB attributes become PyG node features (x) and labels (y)
METAGRAPH = {
    "vertexCollections": {PRODUCTS: {"x": "features", "y": "label"}},
    "edgeCollections": {BOUGHT_TOGETHER: {}},
}

CATEGORIES = {
    0: "Desktops",
    1: "Data Storage",
    2: "Laptops",
    3: "Monitors",
    4: "Computer Components",
    5: "Video Projectors",
    6: "Routers",
    7: "Tablets",
    8: "Networking Products",
    9: "Webcams",
}


def load_env() -> None:
    """Reads KEY=VALUE lines from the first .env found in this example's folder or
    in the examples folder. Existing environment variables win."""
    for folder in (EXAMPLE_DIR, EXAMPLE_DIR.parent):
        env_file = folder / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return


def mode() -> str:
    """'arangodb' (default) or 'offline' (PyG dataset, everything in memory)."""
    load_env()
    return os.environ.get("FGML_MODE", "arangodb").lower()


def connect() -> Any:
    """Connects to the one database the examples are allowed to use."""
    from arango import ArangoClient

    load_env()
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    return ArangoClient(hosts=os.environ.get("ARANGO_URL", "http://localhost:8529")).db(
        os.environ.get("ARANGO_DB", "fastgraphml_examples"),
        os.environ.get("ARANGO_USER", "root"),
        os.environ.get("ARANGO_PASSWORD", ""),
        verify=True,
    )
