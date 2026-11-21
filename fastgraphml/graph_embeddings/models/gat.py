from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from arango.database import Database
from torch import Tensor
from torch.nn import Linear as Lin
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from .base import HomogeneousEmbeddingModel


class GAT(HomogeneousEmbeddingModel):
    """GATCONV model modified from '<https://github.com/pyg-team/pytorch_geometric/blob/6267de93c6b04f46a306aa58e414de330ef9bb10/examples/gat.py>'.  # noqa: E501

    :database (type: Database): A python-arango database instance.
    :arango_graph (type: str): The name of ArangoDB graph which we want to export to PyG
    :metagraph (type: dict): It exports ArangoDB graphs to PyG data objects. We define
        metagraph as a dictionary defining vertex & edge collections to import to PyG,
        along with collection-level specifications to indicate which ArangoDB attributes
        will become PyG features/labels. It also supports different encoders such as
        identity and categorical encoder on database attributes. Detailed information
        regarding different use cases and metagraph definitons can be found on
        adbpyg_adapter github page i.e <https://github.com/arangoml/pyg-adapter>.
    :pyg_graph (type: PyG data object): It generates graph embeddings using PyG graphs
        (via PyG data objects) directy rather than ArangoDB graphs.
        When generating graph embeddings via PyG graphs,
        database=arango_graph=metagraph=None.
    :embedding_size (type: int): Length of the node embeddings when they are mapped to
        d-dimensional euclidean space.
    :heads (type: int): Number of attention heads. Model learns to give attention to
        only important nodes in node's neighborhood. The heads of the last layer are
        averaged, so that the embeddings have embedding_size dimensions.
    :num_layers (type: int): Number of GAT Layers.
    :sizes (type: [int, int]): Number of neighbors to select at each layer for every
        node (uniform random sampling) in order to perform neighborhood sampling.
    :batch_size (type: int): Number of nodes to be present inside batch along with
        their neighborhood. Used while performing neighborhood sampling.
    :dropout_perc (type: float): Handles overfitting inside the model
    :shuffle (type: bool): If set to True, it shuffles data before performing
        neighborhood sampling.
    :transform (type: torch_geometric.transforms): It is used to transform PyG data
        objects. Various transformation methods can be chained together using Compose.
               for e.g. transform = T.Compose([
                        T.NormalizeFeatures(),
                        T.RandomNodeSplit(num_val=0.2, num_test=0.1)])
    :device (type: str): Device to train on, e.g. 'cpu' or 'cuda'. By default the GPU
        is used if there is one.
    :num_val (type: float): Percentage of nodes selected for validation set.
    :num_test (type: float): Percentage of nodes selected for test set.

    Note: After selecting the percentage for validation and test nodes,
    rest percentage of the nodes are considered as training nodes.

    :**kwargs: Additional arguments of the GATConv class. For more arguments please
    refer the following link
     <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv>  # noqa: E501
    """

    def __init__(
        self,
        database: Database = None,
        arango_graph: Optional[str] = None,
        metagraph: Union[Dict[str, object], None] = None,
        pyg_graph: Data = None,
        embedding_size: int = 64,
        heads: int = 2,
        num_layers: int = 2,
        sizes: List[int] = [10, 10],
        batch_size: int = 256,
        dropout_perc: float = 0.5,
        shuffle: bool = True,
        transform: Optional[List[Callable[..., Any]]] = None,
        num_val: float = 0.1,
        num_test: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            database,
            arango_graph,
            metagraph,
            pyg_graph,
            embedding_size,
            num_layers,
            sizes,
            batch_size,
            dropout_perc,
            shuffle,
            transform,
            num_val,
            num_test,
            device,
        )
        self.heads = heads
        # the heads are concatenated in all layers but the last one
        kwargs.pop("concat", None)

        # defining GAT layers and skip connections
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.in_channels if i == 0 else heads * self.hidden_channels
            last = i == num_layers - 1
            self.convs.append(
                GATConv(
                    in_channels, self.hidden_channels, heads, concat=not last, **kwargs
                )
            )
            out_channels = (
                self.hidden_channels if last else heads * self.hidden_channels
            )
            self.skips.append(Lin(in_channels, out_channels))

    # encoder
    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) + self.skips[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_perc, training=self.training)
        return x
