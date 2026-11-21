from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt


def similarity_search(
    graph_emb: Union[npt.NDArray[np.float64], Any],
    top_k_nbors: int = 10,
    nlist: int = 10,
    search_type: str = "exact",
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Performs similarity search in sets of vectors of any size, up to ones that
    possibly do not fit in RAM using FAISS.

    <https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/>.  # noqa: E501

    returns dist, nbors
        nbors: node ids with similar embeddings
        dist: corresponding similarity scores

    The embeddings passed in are not modified.

    :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
    n: number of nodes in graph
    embedding_size: Length of the node embeddings when they are mapped to
        d-dimensional euclidean space.
    :top_k_nbors (type: int): Returns top-k nearest neighbors of all embeddings.
    :nlist (type: int): Number of clusters to be generated.
    :search_type (type: str): We support two types of search for now:
    1. exact search: For precise similarity search but at the cost of scalability.
    2. approx search: For scalable similarity search but at the cost of some precision
        loss.
    """
    if search_type not in ("exact", "approx"):
        raise ValueError("Pass search type either exact or approx.")

    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "Similarity search needs FAISS: pip install 'fastgraphml[search]'"
        ) from e

    # FAISS needs C-contiguous float32 and normalises in place, so work on a copy
    emb = np.array(graph_emb, dtype=np.float32, order="C", copy=True)
    emb_dim = emb.shape[1]
    # L2 normalization of embeddings, inner product is then the cosine similarity
    faiss.normalize_L2(emb)

    # exact search
    if search_type == "exact":
        # inner product search index
        search_index: Any = faiss.IndexFlatIP(emb_dim)
    # approx search
    else:
        # assign embeddings to specific cluster using Inner product
        quantiser = faiss.IndexFlatIP(emb_dim)
        # creates partitioned search index
        search_index = faiss.IndexIVFFlat(
            quantiser, emb_dim, nlist, faiss.METRIC_INNER_PRODUCT
        )
        # train the index
        search_index.train(emb)

    search_index.add(emb)
    # nbors: top k node ids with highest cosine sim
    # dist: cosine similarities of top k nodes
    dist, nbors = search_index.search(emb, k=top_k_nbors + 1)

    return dist, nbors
