"""Settings of the clinical site finder, read from the environment / .env file."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = EXAMPLE_DIR / ".cache"

# The clinical trial graph, as written by backend.load_data
GRAPH = "clinical_trials"
SITES, TRIALS, INDICATIONS, SPONSORS = "site", "trial", "indication", "sponsor"
CONDUCTS, STUDIES, FUNDS = "conducts", "studies", "funds"  # edge collections
EMBEDDINGS = "site_embeddings"  # collection written by fastgraphml

# The part of the graph the model learns from. Sponsors are in the database, but
# not in the metagraph: they are not exported to PyG.
# A site has numeric features (x) and its primary therapeutic area as label (y).
METAGRAPH = {
    "vertexCollections": {
        SITES: {"x": "features", "y": "area"},
        TRIALS: {},
        INDICATIONS: {},
    },
    "edgeCollections": {CONDUCTS: {}, STUDIES: {}},
}

# The random walks of METAPATH2VEC follow this path over and over:
# site -> trial -> indication -> trial -> site. The rev_ edge types are the
# reversed edges added by the ToUndirected transform.
METAPATH = [
    (SITES, CONDUCTS, TRIALS),
    (TRIALS, STUDIES, INDICATIONS),
    (INDICATIONS, "rev_" + STUDIES, TRIALS),
    (TRIALS, "rev_" + CONDUCTS, SITES),
]

AREAS = {
    0: "Oncology",
    1: "Cardiology",
    2: "Neurology",
    3: "Immunology",
    4: "Infectious Disease",
    5: "Endocrinology",
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


def tls_verify() -> Any:
    """ARANGO_VERIFY: unset = normal certificate check, 'false' = no check (for a
    self-signed certificate on your own machine), or the path of a CA bundle."""
    value = os.environ.get("ARANGO_VERIFY", "").strip()
    if not value:
        return None
    if value.lower() in ("false", "0", "no", "off"):
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return False
    return value


def mode() -> str:
    """'arangodb' (default) or 'offline' (PyG dataset, everything in memory)."""
    load_env()
    return os.environ.get("FGML_MODE", "arangodb").lower()


def connect() -> Any:
    """Connects to the one database the examples are allowed to use."""
    from arango import ArangoClient

    load_env()
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    client = ArangoClient(
        hosts=os.environ.get("ARANGO_URL", "http://localhost:8529"),
        verify_override=tls_verify(),
    )
    return client.db(
        os.environ.get("ARANGO_DB", "fastgraphml_examples"),
        os.environ.get("ARANGO_USER", "root"),
        os.environ.get("ARANGO_PASSWORD", ""),
        verify=True,
    )
