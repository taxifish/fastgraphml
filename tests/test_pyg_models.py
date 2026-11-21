"""Tests which need neither an ArangoDB instance nor a dataset download."""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch_geometric.transforms as T
import torch_geometric.typing as pyg_typing
from torch_geometric.data import Data, HeteroData

from fastgraphml.graph_embeddings import DMGI, GAT, METAPATH2VEC, SAGE, downstream_tasks

NUM_NODES = 120
NO_CKP: Dict[str, Any] = {"ckp_path": None, "best_model_path": None}


def homo_graph(labels: bool = True) -> Data:
    torch.manual_seed(0)
    src = torch.randint(0, NUM_NODES, (600,))
    dst = torch.randint(0, NUM_NODES, (600,))
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    data = Data(x=torch.randn(NUM_NODES, 16), edge_index=edge_index)
    if labels:
        data.y = torch.randint(0, 3, (NUM_NODES,))
    return data


def hetero_graph(labels: bool = True) -> HeteroData:
    torch.manual_seed(0)
    data = HeteroData()
    data["movie"].x = torch.randn(80, 16)
    data["actor"].x = torch.randn(100, 16)
    if labels:
        data["movie"].y = torch.randint(0, 3, (80,))
    edge_index = torch.stack(
        [torch.randint(0, 80, (500,)), torch.randint(0, 100, (500,))]
    )
    data["movie", "to", "actor"].edge_index = edge_index
    data["actor", "to", "movie"].edge_index = edge_index.flip(0)
    return data


class FakeCollection:
    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []

    def import_bulk(self, docs: List[Dict[str, Any]], **kwargs: Any) -> None:
        self.docs.extend(docs)


class FakeDatabase:
    """The few python-arango methods store_embeddings uses."""

    def __init__(self) -> None:
        self.collections: Dict[str, FakeCollection] = {}
        self.create_kwargs: Dict[str, Any] = {}

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, **kwargs: Any) -> None:
        self.create_kwargs = kwargs
        self.collections[name] = FakeCollection()

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


@pytest.mark.parametrize("model_cls", [SAGE, GAT])
def test_homogeneous_models(model_cls: Any, tmp_path: Any) -> None:
    ckp, best = str(tmp_path / "ckp.pt"), str(tmp_path / "best.pt")
    model = model_cls(pyg_graph=homo_graph(), embedding_size=8, device="cpu")
    model._train(epochs=2, ckp_path=ckp, best_model_path=best)
    embeddings = model.get_embeddings()
    # embedding_size dimensions, also for GAT with more than one head
    assert embeddings.shape == (NUM_NODES, 8)
    assert np.isfinite(embeddings).all()
    # the last checkpoint is the one of the last epoch
    assert torch.load(ckp)["epoch"] == 2
    assert (tmp_path / "best.pt").exists()
    val_acc, test_acc = model.val()
    assert 0.0 <= val_acc <= 1.0 and 0.0 <= test_acc <= 1.0


@pytest.mark.parametrize("model_cls", [SAGE, GAT])
def test_homogeneous_models_without_labels(model_cls: Any, tmp_path: Any) -> None:
    best = str(tmp_path / "best.pt")
    model = model_cls(pyg_graph=homo_graph(labels=False), embedding_size=8)
    model.fit(epochs=2, ckp_path=None, best_model_path=best)
    assert model.get_embeddings().shape == (NUM_NODES, 8)
    # without labels the best model is the one with the lowest loss
    assert (tmp_path / "best.pt").exists()
    with pytest.raises(Exception):
        model.val()


def test_training_without_neighbor_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    # neither pyg-lib nor torch-sparse installed: full graph training
    monkeypatch.setattr(pyg_typing, "WITH_PYG_LIB", False)
    monkeypatch.setattr(pyg_typing, "WITH_TORCH_SPARSE", False)
    with pytest.warns(UserWarning, match="pyg-lib"):
        model = SAGE(pyg_graph=homo_graph(), embedding_size=8)
    assert model.train_loader is None
    model._train(epochs=2, **NO_CKP)
    assert model.get_embeddings().shape == (NUM_NODES, 8)


def test_integer_features() -> None:
    # word counts stored as integers, as they come out of ArangoDB
    graph = homo_graph()
    graph.x = torch.randint(0, 3, (NUM_NODES, 16))
    model = SAGE(pyg_graph=graph, embedding_size=8, transform=T.NormalizeFeatures())
    assert model.G.x.is_floating_point()
    model._train(epochs=1, **NO_CKP)
    assert np.isfinite(model.get_embeddings()).all()

    hetero = hetero_graph()
    hetero["movie"].x = torch.randint(0, 3, (80, 16))
    model = DMGI(
        pyg_graph=hetero,
        metapaths=[[("movie", "actor"), ("actor", "movie")]],
        key_node="movie",
        embedding_size=4,
        transform=T.NormalizeFeatures(),
    )
    model._train(epochs=1, **NO_CKP)
    assert np.isfinite(model.get_embeddings()["movie"]).all()


def test_wrong_inputs() -> None:
    with pytest.raises(Exception):
        SAGE()
    with pytest.raises(TypeError):
        SAGE(database="not a database", arango_graph="g", metagraph={})
    with pytest.raises(ValueError):
        SAGE(pyg_graph=homo_graph(), num_layers=3)
    graph = homo_graph()
    del graph.x
    with pytest.raises(ValueError):
        SAGE(pyg_graph=graph)


@pytest.mark.parametrize("labels", [True, False])
def test_m2v(labels: bool) -> None:
    metapaths = [("movie", "to", "actor"), ("actor", "to", "movie")]
    model = METAPATH2VEC(
        pyg_graph=hetero_graph(labels),
        metapaths=metapaths,
        key_node="movie",
        embedding_size=4,
        sparse=True,
    )
    with pytest.raises(Exception):
        model.get_embeddings()
    model._train(epochs=2, **NO_CKP)
    embeddings = model.get_embeddings()
    assert embeddings["movie"].shape == (80, 4)
    assert embeddings["actor"].shape == (100, 4)


@pytest.mark.parametrize("labels", [True, False])
def test_dmgi(labels: bool) -> None:
    metapaths = [[("movie", "actor"), ("actor", "movie")]]  # MAM
    model = DMGI(
        pyg_graph=hetero_graph(labels),
        metapaths=metapaths,
        key_node="movie",
        embedding_size=4,
    )
    model._train(epochs=2, **NO_CKP)
    embeddings = model.get_embeddings()
    assert embeddings["movie"].shape == (80, 4)


def test_similarity_search() -> None:
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(200, 8))  # float64 on purpose
    original = embeddings.copy()
    for search_type in ("exact", "approx"):
        dist, nbors = downstream_tasks.similarity_search(
            embeddings, top_k_nbors=5, nlist=4, search_type=search_type
        )
        assert dist.shape == nbors.shape == (200, 6)
    # the most similar node of a node is the node itself (exact search)
    dist, nbors = downstream_tasks.similarity_search(embeddings, top_k_nbors=5)
    assert (nbors[:, 0] == np.arange(200)).all()
    # the embeddings are not normalised in place
    assert np.array_equal(embeddings, original)
    with pytest.raises(Exception):
        downstream_tasks.similarity_search(embeddings, search_type="wrong")


def test_visualize_embeddings(tmp_path: Any) -> None:
    pytest.importorskip("umap")
    import matplotlib

    matplotlib.use("Agg")
    graph = homo_graph()
    embeddings = np.random.default_rng(0).normal(size=(NUM_NODES, 8))
    path = tmp_path / "umap.png"
    downstream_tasks.visualize_embeddings(
        graph,
        embeddings,
        class_mapping={0: "a", 1: "b", 2: "c"},
        emb_percent=0.5,
        save_path=str(path),
    )
    assert path.exists()


def test_store_embeddings() -> None:
    pytest.importorskip("faiss")
    model = SAGE(pyg_graph=homo_graph(), embedding_size=8)
    embeddings = model.get_embeddings()
    db = FakeDatabase()
    model.graph_util.database = db

    # PyG graph: the node index is the key
    model.graph_util.store_embeddings(
        embeddings,
        collection_name="emb",
        batch_size=50,
        class_mapping={0: "a", 1: "b", 2: "c"},
        nearest_nbors_search=True,
        top_k_nbors=3,
    )
    docs = db["emb"].docs
    assert db.create_kwargs == {}
    assert [doc["_key"] for doc in docs] == [str(i) for i in range(NUM_NODES)]
    assert len(docs[0]["embedding"]) == 8
    assert docs[0]["class_name"] in ("a", "b", "c")
    # top_k_nbors is used (the node itself + 3 neighbors)
    assert len(docs[0]["similar_nodes"]) == 4
    # the embeddings are stored as they are, i.e. not normalised
    assert np.allclose(docs[5]["embedding"], embeddings[5])

    # graph read from ArangoDB: the documents are linked to their vertices
    model.graph_util.node_keys[None] = [f"k{i}" for i in range(NUM_NODES)]
    model.graph_util.metagraph = {
        "vertexCollections": {"Products": {"x": "features"}},
        "edgeCollections": {"bought_together": {}},
    }
    model.graph_util.store_embeddings(
        embeddings,
        collection_name="emb2",
        nearest_nbors_search=True,
        replication_factor=2,
    )
    docs = db["emb2"].docs
    assert db.create_kwargs == {"replication_factor": 2}
    assert docs[7]["_key"] == "k7"
    assert docs[7]["source_id"] == "Products/k7"
    assert docs[7]["similar_nodes"][0] == "k7"
