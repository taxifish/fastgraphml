import shutil
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.typing as pyg_typing
from arango.database import Database
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling

from ..utils import GraphUtils, check_inputs, has_labels, resolve_device


def save_checkpoints(
    state: Dict[str, Any],
    is_best: bool,
    ckp_path: Optional[str],
    best_model_path: Optional[str],
) -> None:
    """Saves the latest checkpoint and, if it is the best model so far, a copy of it.
    Nothing is written for a path which is None."""
    if ckp_path is not None:
        torch.save(state, ckp_path)
        if is_best and best_model_path is not None:
            # copy best checkpoint file to best model path
            shutil.copyfile(ckp_path, best_model_path)
    elif is_best and best_model_path is not None:
        torch.save(state, best_model_path)


def node_classification_accuracy(z: Tensor, store: Any) -> Tuple[float, float]:
    """Trains a logistic regression on the embeddings of the training nodes and
    returns its accuracy on the validation and on the test nodes."""
    emb = z.detach().cpu().numpy()
    y = store.y.detach().cpu().numpy()
    train_mask = store.train_mask.cpu().numpy()
    val_mask = store.val_mask.cpu().numpy()
    test_mask = store.test_mask.cpu().numpy()
    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    clf.fit(emb[train_mask], y[train_mask])
    val_acc = float(clf.score(emb[val_mask], y[val_mask]))
    test_acc = float(clf.score(emb[test_mask], y[test_mask]))
    return val_acc, test_acc


class TrainingLog:
    """Prints the progress of a training run and saves the checkpoints.

    The best model is the one with the highest validation accuracy. For graphs
    without labels it is the one with the lowest training loss.
    """

    def __init__(self, ckp_path: Optional[str], best_model_path: Optional[str]):
        self.ckp_path = ckp_path
        self.best_model_path = best_model_path
        self.best_acc = 0.0
        self.best_loss = float("inf")

    def epoch_end(
        self,
        epoch: int,
        loss: float,
        accs: Optional[Tuple[float, float]],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if accs is not None:
            val_acc, test_acc = accs
            print(
                f"Epoch: {epoch:03d}, Train_Loss: {loss:.4f}, "
                f"Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )
        else:
            print(f"Epoch: {epoch:03d}, Train_Loss: {loss:.4f}")

        # create checkpoint variable and add crucial model information
        model_checkpoint = {
            "epoch": epoch,
            "best_acc": accs[0] if accs is not None else None,
            "loss": loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if accs is not None:
            is_best = accs[0] > self.best_acc
            if is_best:
                print(
                    "Val Acc increased ({:.5f} --> {:.5f}).  Saving model ...".format(
                        self.best_acc, accs[0]
                    )
                )
                self.best_acc = accs[0]
        else:
            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss

        # save current/last checkpoint (used for resuming training) and
        # the best model
        save_checkpoints(model_checkpoint, is_best, self.ckp_path, self.best_model_path)


class OneEdgePerNode(torch.utils.data.Sampler[int]):
    """Samples, in every epoch, one random outgoing edge of every node.

    An epoch then has (at most) as many positive examples as the graph has nodes,
    as in the GraphSage paper, instead of one per edge, which would make epochs
    on dense graphs very long.
    """

    def __init__(self, edge_index: Tensor, num_nodes: int, shuffle: bool = True):
        src = edge_index[0].cpu()
        # edge ids sorted by source node, and where the edges of each node start
        self.perm = torch.argsort(src, stable=True)
        self.degree = torch.bincount(src, minlength=num_nodes)
        self.start = torch.cumsum(self.degree, dim=0) - self.degree
        self.nodes = torch.nonzero(self.degree > 0).view(-1)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return int(self.nodes.numel())

    def __iter__(self) -> Any:
        nodes = self.nodes
        if self.shuffle:
            nodes = nodes[torch.randperm(nodes.numel())]
        degree = self.degree[nodes]
        offset = (torch.rand(nodes.numel()) * degree).long().clamp(max=degree - 1)
        return iter(self.perm[self.start[nodes] + offset].tolist())


class HomogeneousEmbeddingModel(nn.Module):
    """Unsupervised training of a GNN encoder on a homogeneous graph, shared by
    the SAGE and the GAT model.

    The encoder is trained with the unsupervised GraphSage loss: the embeddings of
    two connected nodes should be similar, the embeddings of two random nodes
    should not. Subclasses create the layers and implement `encode`.
    """

    def __init__(
        self,
        database: Database = None,
        arango_graph: Optional[str] = None,
        metagraph: Union[Dict[str, object], None] = None,
        pyg_graph: Data = None,
        embedding_size: int = 64,
        num_layers: int = 2,
        sizes: List[int] = [10, 10],
        batch_size: int = 256,
        dropout_perc: float = 0.5,
        shuffle: bool = True,
        transform: Optional[List[Callable[..., Any]]] = None,
        num_val: float = 0.1,
        num_test: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        check_inputs(database, arango_graph, metagraph, pyg_graph)

        if len(sizes) != num_layers:
            raise ValueError(
                f"sizes needs one entry per layer: got {len(sizes)} sizes "
                f"for {num_layers} layers"
            )

        self.device = resolve_device(device)
        # arango to PyG
        self.graph_util = GraphUtils(
            arango_graph, metagraph, database, pyg_graph, num_val, num_test, transform
        )
        # get PyG graph
        G = self.graph_util.graph
        if getattr(G, "x", None) is None:
            raise ValueError(
                f"{type(self).__name__} needs node features (x). For graphs "
                "without features use METAPATH2VEC, or create features first, "
                "e.g. with torch_geometric.transforms.LocalDegreeProfile."
            )
        self.G = G
        self.in_channels = G.num_node_features
        self.hidden_channels = embedding_size
        # the graph stays on the CPU, mini-batches are moved to the device
        self.x = G.x.float().cpu()
        self.edge_index = G.edge_index.cpu()
        self.num_nodes = G.num_nodes
        self.num_layers = num_layers
        self.sizes = sizes
        self.batch_size = batch_size
        self.dropout_perc = dropout_perc
        self.shuffle = shuffle
        self.has_labels = has_labels(G)
        # create train loader
        self.train_loader = self._create_loader()

    def _create_loader(self) -> Optional[LinkNeighborLoader]:
        """In every epoch one random edge per node is a positive example. For each
        of them the loader samples a random pair of nodes (negative example), and
        the neighborhood of all these nodes.

        Neighborhood sampling needs pyg-lib (or torch-sparse). Without them the
        model is trained on the full graph, which is fine for small graphs only.
        """
        if not (pyg_typing.WITH_PYG_LIB or pyg_typing.WITH_TORCH_SPARSE):
            warnings.warn(
                "Neighborhood sampling needs pyg-lib (or torch-sparse), none of "
                "them is installed: training on the full graph instead, which "
                "does not scale to large graphs. See the installation section of "
                "the README."
            )
            return None
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)
        return LinkNeighborLoader(
            data,
            num_neighbors=self.sizes,
            batch_size=self.batch_size,
            sampler=OneEdgePerNode(self.edge_index, self.num_nodes, self.shuffle),
            neg_sampling_ratio=1.0,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encode(x, edge_index)

    # kept for backward compatibility
    def full_forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encode(x, edge_index)

    # save checkpoints whenever there is an increase in validation accuracy
    @staticmethod
    def save_checkpoints(
        state: Dict[str, Any],
        is_best: bool,
        ckp_path: Optional[str],
        best_model_path: Optional[str],
    ) -> None:
        save_checkpoints(state, is_best, ckp_path, best_model_path)

    @staticmethod
    def _link_loss(out: Tensor, edge_label_index: Tensor, edge_label: Tensor) -> Tensor:
        src, dst = edge_label_index
        logits = (out[src] * out[dst]).sum(-1)
        return F.binary_cross_entropy_with_logits(logits, edge_label)

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Trains the model for one epoch, returns the mean loss per example."""
        self.train()
        if self.train_loader is None:
            # full graph training: all the edges and as many random node pairs
            x = self.x.to(self.device)
            edge_index = self.edge_index.to(self.device)
            neg_edge_index = negative_sampling(edge_index, num_nodes=self.num_nodes)
            edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(edge_index.size(1), device=self.device),
                    torch.zeros(neg_edge_index.size(1), device=self.device),
                ]
            )
            optimizer.zero_grad()
            loss = self._link_loss(self(x, edge_index), edge_label_index, edge_label)
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        total_loss, total_examples = 0.0, 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            out = self(batch.x, batch.edge_index)
            loss = self._link_loss(out, batch.edge_label_index, batch.edge_label)
            loss.backward()
            optimizer.step()
            num_examples = batch.edge_label.numel()
            total_loss += float(loss.detach()) * num_examples
            total_examples += num_examples
        return total_loss / max(total_examples, 1)

    def _train(
        self,
        ckp_path: Optional[str] = "./latest_model_checkpoint.pt",
        best_model_path: Optional[str] = "./best_model.pt",
        epochs: int = 50,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Train GraphML model.

        :ckp_path (type: str): Path to save model's latest checkpoints
            (i.e. at each epoch). Pytorch models are saved with .pt file extension.
            By default it saves model in cwd. Pass None to not save it.
        :best_model_path (type: str): Path to save model whenever there is an increase
            in validation accuracy (or, for graphs without labels, a decrease in
            training loss). By default it saves model in cwd. Pass None to not
            save it.
        :epochs (type: int): Number of times training data go through the model.
        :lr (type: float): Learning rate.
        :**kwargs: Additional arguments for the Adam optimizer for
            e.g. weight_decay, betas, etc.
        """

        model = self.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr, **kwargs)
        log = TrainingLog(ckp_path, best_model_path)
        print("Training started .........")
        for epoch in range(1, epochs + 1):
            loss = self._train_epoch(optimizer)
            # validate model
            accs = self.val() if self.has_labels else None
            log.epoch_end(epoch, loss, accs, model, optimizer)

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Train GraphML model, same as `_train`."""
        self._train(*args, **kwargs)

    @torch.no_grad()
    def _embed(self) -> Tensor:
        self.eval()
        self.to(self.device)
        out: Tensor = self(self.x.to(self.device), self.edge_index.to(self.device))
        return out.cpu()

    @torch.no_grad()
    def val(
        self,
    ) -> Tuple[float, float]:
        """Tests the performance of a generated graph embeddings using Node Classification as a downstream task.

        returns validation and test accuracy.

        """  # noqa: E501
        if not self.has_labels:
            raise Exception("The graph has no labels (y) to validate the model with.")
        return node_classification_accuracy(self._embed(), self.G)

    @torch.no_grad()
    def get_embeddings(
        self,
    ) -> Union[Tensor, Any]:
        """Returns Graph Embeddings of size (n, embedding_size),
        n: number of nodes in graph.
        embedding_size: Length of the node embeddings when they are mapped to
            d-dimensional euclidean space.

        """
        return self._embed().numpy()
