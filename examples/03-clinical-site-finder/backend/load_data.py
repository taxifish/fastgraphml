"""Writes the synthetic clinical trial graph to the configured database.

    python -m backend.load_data            # create what is missing
    python -m backend.load_data --replace  # empty the collections first

Creates four vertex collections (site, trial, indication, sponsor), three edge
collections (conducts, studies, funds) and the clinical_trials graph. Nothing
outside these collections is touched.
"""

from __future__ import annotations

import sys

from . import config, data

VERTICES = [config.SITES, config.TRIALS, config.INDICATIONS, config.SPONSORS]
EDGES = {
    config.CONDUCTS: (config.SITES, config.TRIALS),
    config.STUDIES: (config.TRIALS, config.INDICATIONS),
    config.FUNDS: (config.SPONSORS, config.TRIALS),
}


def main(replace: bool = False) -> None:
    db = config.connect()
    records = data.generate()

    for name in VERTICES + list(EDGES):
        if not db.has_collection(name):
            db.create_collection(name, edge=name in EDGES)
        collection = db.collection(name)
        if replace:
            collection.truncate()
        elif collection.count() > 0:
            print(f"{name}: already has {collection.count()} documents, skipped")
            continue
        docs = records[name]
        for start in range(0, len(docs), 1000):
            collection.import_bulk(docs[start : start + 1000], on_duplicate="replace")
        print(f"{name}: {collection.count()} documents")

    if not db.has_graph(config.GRAPH):
        db.create_graph(
            config.GRAPH,
            edge_definitions=[
                {
                    "edge_collection": edge,
                    "from_vertex_collections": [src],
                    "to_vertex_collections": [dst],
                }
                for edge, (src, dst) in EDGES.items()
            ],
        )
        print(f"graph {config.GRAPH}: created")
    print(
        f"Done. Database '{db.name}' is ready, start the backend and train the model."
    )


if __name__ == "__main__":
    main(replace="--replace" in sys.argv)
