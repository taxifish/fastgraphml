from typing import Any

from fastgraphml.graph_embeddings.models.dmgi import DMGI
from fastgraphml.graph_embeddings.models.gat import GAT
from fastgraphml.graph_embeddings.models.graph_sage import SAGE
from fastgraphml.graph_embeddings.models.metapath2vec import METAPATH2VEC

__all__ = ["DMGI", "GAT", "SAGE", "METAPATH2VEC", "Datasets"]


def __getattr__(name: str) -> Any:
    # ``arango_datasets`` is an optional dependency (``pip install
    # fastgraphml[arango]``), so it is only imported when it is asked for.
    if name == "Datasets":
        try:
            from arango_datasets.datasets import Datasets
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "`Datasets` needs the arango-datasets package: "
                "pip install 'fastgraphml[arango]'"
            ) from e
        return Datasets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
