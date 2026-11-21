from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from arango.database import Database
from torch_geometric.data import Data
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.typing import EdgeType

from ..utils import GraphUtils, check_inputs, has_labels, resolve_device
from .base import TrainingLog, node_classification_accuracy, save_checkpoints


class METAPATH2VEC:
    """metapath2vec model modified from '<https://github.com/pyg-team/pytorch_geometric/blob/6267de93c6b04f46a306aa58e414de330ef9bb10/examples/hetero/metapath2vec.py>'.  # noqa: E501

    :database (type: Database): A python-arango database instance.
    :arango_graph (type: str): The name of ArangoDB graph which we want to export to PyG
    :metagraph (type: Dict): It exports ArangoDB graphs to PyG data objects. We define
        metagraph as a dictionary defining vertex & edge collections to import to PyG,
        along with collection-level specifications to indicate which ArangoDB attributes
        will become PyG features/labels. It also supports different encoders such as
        identity and categorical encoder on database attributes. Detailed information
        regarding different use cases and metagraph definitons can be found on
        adbpyg_adapter github page i.e <https://github.com/arangoml/pyg-adapter>.
    :metapaths (type: list[Tuple(str,str,str)]): The metapath defined as
        (src_node_type, rel_type, dst_node_type) tuples. M2V uses metapaths to
        perform random walks on the graph and then uses skip-grapm to compute
        graph embeddings.
    :key_node (type: str): Node type on which we want to test the performance of
        generated graph embeddings. Performance is tested using node classification task,
        if this node type has labels (y), otherwise the validation is skipped.
    :pyg_graph (type: PyG data object): It generates graph embeddings using PyG graphs
        (via PyG data objects) directy rather than ArangoDB graphs. When generating graph
        embeddings via PyG graphs, database=arango_graph=metagraph=None.
    :embedding_size (type: int): Length of the node embeddings when they are mapped to
        d-dimensional euclidean space.
    :walk_length (type: int): The walk length.
    :context_size (type: int): The actual context size which is considered for positive
        samples. This parameter increases the effective sampling rate by reusing samples
        across different source nodes.
    :walks_per_node (type: float): The number of walks to sample for each node.
    :num_negative_samples (type: bool): The number of negative samples to use for each
        positive sample.
    :num_nodes_dict (type: Dict): Dictionary holding the number of nodes for each
        node type.
    :sparse (type: bool): If set to True, gradients w.r.t. to the weight matrix will be
        sparse.
    :transform (type: torch_geometric.transforms): It is used to transform PyG data
        objects. Various transformation methods can be chained together using Compose.
               for e.g. transform = T.Compose([
                        T.NormalizeFeatures(),
                        T.RandomNodeSplit(num_val=0.2, num_test=0.1)]).
    :num_val (type: float): Percentage of nodes selected for validation set.
    :num_test (type: float): Percentage of nodes selected for test set.
    :shuffle (type: bool): If set to True, it shuffles data before training.
    :device (type: str): Device to train on, e.g. 'cpu' or 'cuda'. By default the GPU
        is used if there is one.

    Note: After selecting the percentage for validation and test nodes,
    rest percentage of the nodes are considered as training nodes.

    :Detailed information about Metapath2Vec args can be found in
    <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.MetaPath2Vec>  # noqua: E501
    """

    def __init__(
        self,
        database: Database = None,
        arango_graph: Optional[str] = None,
        metagraph: Union[Dict[str, object], None] = None,
        metapaths: Optional[List[EdgeType]] = None,
        key_node: Union[str, None] = None,
        pyg_graph: Data = None,
        embedding_size: int = 64,
        walk_length: int = 5,
        context_size: int = 6,
        walks_per_node: int = 5,
        num_negative_samples: int = 5,
        num_nodes_dict: Optional[Dict[str, int]] = None,
        sparse: bool = False,
        batch_size: int = 64,
        transform: Optional[List[Callable[..., Any]]] = None,
        num_val: float = 0.1,
        num_test: float = 0.1,
        shuffle: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        check_inputs(database, arango_graph, metagraph, pyg_graph)
        if not metapaths:
            raise ValueError("Pass the metapaths to perform the random walks on.")
        if key_node is None:
            # any node type of the metapath is fine if there are no labels
            key_node = metapaths[0][0]

        self.device = resolve_device(device)
        # arango tp pyg
        self.graph_util = GraphUtils(
            arango_graph,
            metagraph,
            database,
            pyg_graph,
            num_val,
            num_test,
            transform,
            key_node,
        )
        # get PyG graph
        G = self.graph_util.graph
        self.G = G
        self.metadata = G.metadata()
        self.key_node = key_node
        self.has_labels = has_labels(G[key_node])
        # metapath2vec attributes
        self.edge_index_dict = self.G.edge_index_dict
        self.embeddin_size = embedding_size
        self.metapath = metapaths
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_node_dict = num_nodes_dict
        self.sparse = sparse
        # loader attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # trained model, set by _train
        self.m2v_model: Optional[MetaPath2Vec] = None

    # save checkpoints whenever there is an increase in validation accuracy
    @staticmethod
    def save_checkpoints(
        state: Dict[str, Any],
        is_best: bool,
        ckp_path: Optional[str],
        best_model_path: Optional[str],
    ) -> None:
        save_checkpoints(state, is_best, ckp_path, best_model_path)

    def _train(
        self,
        ckp_path: Optional[str] = "./latest_model_checkpoint.pt",
        best_model_path: Optional[str] = "./best_model.pt",
        epochs: int = 50,
        lr: float = 0.03,
        log_steps: int = 100,
        eval_steps: int = 2000,
        **kwargs: Any,
    ) -> None:
        """Train GraphML model.

        :ckp_path (type: str): Path to save model's latest checkpoints
            (i.e. at each epoch). Pytorch models are saved with .pt file extension.
            By default it saves model in cwd. Pass None to not save it.
        :best_model_path (type: str): Path to save model whenever there is an increase
            in validation accuracy (or, if the key node has no labels, a decrease in
            training loss). By default it saves model in cwd. Pass None to not
            save it.
        :epochs (type: int): Number of times training data go through the model.
        :lr (type: float): Learning rate.
        : log_steps (type: int): Logs epochs and loss.
        : eval_steps (type: int): Evaluate model performance using validation and
            test data.
        :**kwargs: Additional arguments for the Adam optimizer for
            e.g. weight_decay, betas, etc.
        """
        device = self.device
        # model
        model = MetaPath2Vec(
            edge_index_dict=self.edge_index_dict,
            embedding_dim=self.embeddin_size,
            metapath=self.metapath,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            num_nodes_dict=self.num_node_dict,
            sparse=self.sparse,
        )

        # transfer model to gpu
        model = model.to(device)
        self.m2v_model = model
        # returns data loader that creates both positive and negative
        # random walks on the heterogeneous graph.
        loader = model.loader(batch_size=self.batch_size, shuffle=self.shuffle)
        optimizer: torch.optim.Optimizer
        if self.sparse is True:
            optimizer = torch.optim.SparseAdam(
                list(model.parameters()), lr=lr, **kwargs
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
        log = TrainingLog(ckp_path, best_model_path)
        print("Training started .........")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            epoch_loss = 0.0
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()

                loss = model.loss(pos_rw.to(device), neg_rw.to(device))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                epoch_loss += loss.item()
                if (i + 1) % log_steps == 0:
                    print(
                        (
                            f"Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, "
                            f"Loss: {total_loss / log_steps:.4f}"
                        )
                    )
                    total_loss = 0.0

                if self.has_labels and (i + 1) % eval_steps == 0:
                    val_acc, test_acc = self.val(model)
                    model.train()
                    print(
                        f"Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, "
                        f"Val_Acc: {val_acc:.4f}, Test_Acc: {test_acc:.4f}"
                    )

            # validate model
            accs = self.val(model) if self.has_labels else None
            log.epoch_end(
                epoch, epoch_loss / max(len(loader), 1), accs, model, optimizer
            )

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Train GraphML model, same as `_train`."""
        self._train(*args, **kwargs)

    @torch.no_grad()
    def val(self, model: Any = None) -> Tuple[float, float]:
        """Tests the performance of a generated graph embeddings using Node
        Classification as a downstream task. Performance is tested on a node type
        (associated with labels) mentioned using key_node argument.

        returns validation and test accuracy on key node.

        model: Graph embedding model, by default the trained model.
        """
        if model is None:
            model = self._trained_model()
        if not self.has_labels:
            raise Exception(
                f"The {self.key_node} nodes have no labels (y) to validate the "
                "model with."
            )
        model.eval()
        z = model(self.key_node)
        return node_classification_accuracy(z, self.G[self.key_node])

    def _trained_model(self) -> MetaPath2Vec:
        if self.m2v_model is None:
            raise Exception("Train the model first: model._train()")
        return self.m2v_model

    @torch.no_grad()
    def get_embeddings(
        self,
    ) -> Dict[str, Any]:
        """Returns Graph Embeddings as a dictionary {node_type1:emb1, node_type2:emb2,

        ....} where embeddings for each node type is present, if that node type is used
        in metapath.

        Embeddings size: (n, embedding_size), where
        n: number of nodes present for specific type inside graph.
        embedding_size: Length of the node embeddings when they are mapped to
            d-dimensional euclidean space.
        """
        model = self._trained_model()
        model.eval()
        emb = {}
        metapath_nodes = []
        for src_node, _, dest_node in self.metapath:
            if src_node not in metapath_nodes:
                metapath_nodes.append(src_node)
            if dest_node not in metapath_nodes:
                metapath_nodes.append(dest_node)

        for node in metapath_nodes:
            emb[node] = model(node).detach().cpu().numpy()

        return emb
