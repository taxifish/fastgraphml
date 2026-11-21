# utils file
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch_geometric.transforms as T
from arango.database import Database
from rich.progress import track
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from .downstream_tasks.similarity_search import similarity_search


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Returns the requested device, or the GPU if there is one, else the CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_inputs(
    database: Any, arango_graph: Any, metagraph: Any, pyg_graph: Any
) -> None:
    """Validates the graph source arguments shared by all the models."""
    if (
        database is not None or arango_graph is not None or metagraph is not None
    ) and pyg_graph is not None:
        msg = "when generating graph embeddings via PyG data objects, database=arango_graph=metagraph=None and vice versa"  # noqa: E501
        raise Exception(msg)

    if database is None and pyg_graph is None:
        raise Exception("Pass either an ArangoDB database or a PyG data object.")

    if database is not None:
        if not issubclass(type(database), Database):
            msg = "**db** parameter must inherit from arango.database.Database"
            raise TypeError(msg)


def has_labels(store: Any) -> bool:
    """True if the node store has labels and train/val/test masks, i.e. if the
    embeddings can be evaluated with node classification."""
    return all(
        getattr(store, attr, None) is not None
        for attr in ("y", "train_mask", "val_mask", "test_mask")
    )


# various Graph ML utilites
class GraphUtils:
    """Various graph utility hepler methods."""

    def __init__(
        self,
        arango_graph: Optional[str],
        metagraph: Union[Dict[str, object], None],
        database: Database,
        pyg_graph: Union[Data, HeteroData, None],
        num_val: float,
        num_test: float,
        transform: Any,
        key_node: Union[str, None] = None,
        metapaths: Optional[List[EdgeType]] = None,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.key_node = key_node
        self.metapaths = metapaths
        self.database = database
        self.metagraph = metagraph
        self.transform = transform
        # ArangoDB `_key` of every node, in PyG node order. Only known when the
        # graph was exported from ArangoDB. The key is the node type for
        # heterogeneous graphs and None for homogeneous graphs.
        self.node_keys: Dict[Optional[str], List[str]] = {}

        if pyg_graph is None:
            self.graph = self.arango_to_pyg(arango_graph, metagraph)
        else:
            self.graph = self.pyg_preprocess(pyg_graph)
        print(self.graph_stats())

    def _label_store(self, pyg_data: Union[Data, HeteroData]) -> Any:
        """Node store holding the labels (the key node for hetero graphs)."""
        return pyg_data if self.key_node is None else pyg_data[self.key_node]

    def pyg_preprocess(self, pyg_data: Union[Data, HeteroData]) -> Any:
        """Takes PyG data object and preprocess it.
        By default it performs following preporocessing:
        1. Perform Random Node Split i.e. splitting data into train, val and test set
           (only if the graph has labels and is not split already).
        2. AddMetaPaths (only if metapaths are passed, i.e. for DMGI).
        3. Applies the user defined transform.

        :pyg_data (tpye: PyG data object): PyG data object
        """
        store = self._label_store(pyg_data)
        has_y = getattr(store, "y", None) is not None
        if has_y and getattr(store, "train_mask", None) is None:
            pyg_data = T.RandomNodeSplit(num_val=self.num_val, num_test=self.num_test)(
                pyg_data
            )

        if self.key_node is not None and self.metapaths is not None:
            # parallel edges make the sparse matrix product of AddMetaPaths
            # invalid (it can even crash), and do not change its result
            pyg_data = T.RemoveDuplicatedEdges()(pyg_data)
            pyg_data = T.AddMetaPaths(
                self.metapaths,
                drop_orig_edge_types=True,
                drop_unconnected_node_types=True,
            )(pyg_data)

        if self.transform is not None:
            pyg_data = self.transform(pyg_data)

        return pyg_data

    def arango_to_pyg(
        self, arango_graph: Optional[str], metagraph: Union[Dict[str, object], None]
    ) -> Any:
        """Exports ArangoDB graph to PyG data object using ArangoDB PyG Adapter.

        :arango_graph (type: str): The name of ArangoDB graph which we want to
        export to PyG.
        :metagraph (type: dict): It exports ArangoDB graphs to PyG data objects.
            We define metagraph as a dictionary defining vertex & edge collections
            to import to PyG, along with collection-level specifications to
            indicate which ArangoDB attributes will become PyG features/labels.
            It also supports different encoders such as identity and categorical
            encoder on database attributes. Detailed information regarding different
            use cases and metagraph definitons can be found on adbpyg_adapter github
            page i.e <https://github.com/arangoml/pyg-adapter>.
        """
        try:
            from adbpyg_adapter import ADBPyG_Adapter
        except ImportError as e:
            raise ImportError(
                "Reading graphs from ArangoDB needs the adbpyg-adapter package: "
                "pip install 'fastgraphml[arango]'"
            ) from e

        adbpyg = ADBPyG_Adapter(self.database)
        try:
            # keep the ArangoDB _key of every node, so that the embeddings can be
            # linked back to the documents they belong to (see store_embeddings)
            pyg_obj = adbpyg.arangodb_to_pyg(
                arango_graph, metagraph, preserve_adb_keys=True
            )
        except TypeError:  # adbpyg-adapter < 1.1.0
            pyg_obj = adbpyg.arangodb_to_pyg(arango_graph, metagraph)
        self._pop_adb_keys(pyg_obj)
        pyg_obj = self.pyg_preprocess(pyg_obj)
        return pyg_obj

    def _pop_adb_keys(self, pyg_obj: Union[Data, HeteroData]) -> None:
        """Moves the ArangoDB keys stored by the adapter out of the PyG graph (they
        are plain python lists which PyG transforms and loaders do not expect)."""
        if isinstance(pyg_obj, HeteroData):
            for node_type in pyg_obj.node_types:
                if "_key" in pyg_obj[node_type]:
                    self.node_keys[node_type] = list(pyg_obj[node_type]["_key"])
                    del pyg_obj[node_type]["_key"]
            for edge_type in pyg_obj.edge_types:
                if "_key" in pyg_obj[edge_type]:
                    del pyg_obj[edge_type]["_key"]
        else:
            if "_v_key" in pyg_obj:
                self.node_keys[None] = list(pyg_obj["_v_key"])
                del pyg_obj["_v_key"]
            if "_e_key" in pyg_obj:
                del pyg_obj["_e_key"]

    # print graph statistics information about input graph for
    # e.g. num of nodes, edges, etc.
    def graph_stats(
        self,
    ) -> Dict[str, Any]:
        g_info: Dict[str, Any] = {}
        if self.key_node is None:
            print("Homogeneous Graph Detected ........ \n")
            g_info["Nodes"] = self.graph.num_nodes
            g_info["Node_feature_size"] = self.graph.num_node_features
            if getattr(self.graph, "y", None) is not None:
                g_info["Number_of_classes"] = self.graph.y.unique().size(0)
            else:
                g_info["Number_of_classes"] = None
            g_info["Edges"] = self.graph.num_edges
            if "edge_attr" in self.graph:
                e_size = self.graph.edge_attr.size(1)
                g_info["Edge_feature_fize"] = e_size
            else:
                g_info["Edge_feature_fize"] = None
            if self.graph.is_directed():
                g_info["Graph Directionality"] = "Directed"
            else:
                g_info["Graph Directionality"] = "Undirected"
            g_info["Average node degree"] = (
                f"{self.graph.num_edges / self.graph.num_nodes:.2f}"
            )
            if has_labels(self.graph):
                g_info["Number of training nodes"] = self.graph.train_mask.sum().item()
                g_info["Number of val nodes"] = self.graph.val_mask.sum().item()
                g_info["Number of test nodes"] = self.graph.test_mask.sum().item()
            g_info["Has isolated nodes"] = self.graph.has_isolated_nodes()
            return g_info
        else:
            print("Heterogeneous Graph Detected .......... \n")
            node_types = self.graph.metadata()[0]
            edge_types = self.graph.metadata()[1]

            # adding node information
            node_data = {}
            for node in node_types:
                total_nodes = self.graph[node].num_nodes
                if total_nodes is None:
                    # counting documents in ArangoDB collections and adding it
                    # to num_nodes attribute
                    total_nodes = self.database.collection(node).count()
                    self.graph[node].num_nodes = total_nodes
                if has_labels(self.graph[node]):
                    node_data["Number of " + node + " nodes"] = total_nodes
                    node_data["Number of train " + node + " nodes"] = (
                        self.graph[node].train_mask.sum().item()
                    )
                    node_data["Number of val " + node + " nodes"] = (
                        self.graph[node].val_mask.sum().item()
                    )
                    node_data["Number of test " + node + " nodes"] = (
                        self.graph[node].test_mask.sum().item()
                    )
                    node_data["Number of classes in " + node + " nodes"] = (
                        self.graph[node].y.unique().size(0)
                    )
                else:
                    node_data["number of " + node + " nodes"] = total_nodes

            # add total nodes
            g_info["Nodes"] = self.graph.num_nodes
            # add total edges
            g_info["Edges"] = self.graph.num_edges
            # add all node types
            g_info["node_types"] = node_types
            # add all edge_types
            g_info["edge_types"] = edge_types
            # check graph directionality
            if self.graph.is_directed():
                g_info["Graph Directionality"] = "Directed"
            else:
                g_info["Graph Directionality"] = "Undirected"
            # check for isolated nodes
            g_info["Has isolated nodes"] = self.graph.has_isolated_nodes()
            g_info["node stats"] = node_data
            return g_info

    def _source_collection(self, node_type: Optional[str]) -> Optional[str]:
        """Name of the ArangoDB vertex collection the embeddings belong to."""
        if node_type is not None:
            return node_type
        if self.metagraph is not None:
            v_cols = list(self.metagraph.get("vertexCollections", {}))  # type: ignore
            if len(v_cols) == 1:
                return str(v_cols[0])
        return None

    # store graph embeddings inside arangodb
    def store_embeddings(
        self,
        graph_emb: Union[npt.NDArray[np.float64], Any],
        collection_name: str,
        batch_size: int = 100,
        class_mapping: Optional[Dict[int, str]] = None,
        node_type: Optional[str] = None,
        nearest_nbors_search: bool = False,
        top_k_nbors: int = 10,
        nlist: int = 10,
        search_type: str = "exact",
        replication_factor: Optional[int] = None,
    ) -> None:
        """Store generated graph embeddings inside ArangoDB.

        If the graph was read from ArangoDB, every embedding document gets the
        same `_key` as the vertex it belongs to, plus a `source_id` attribute with
        the `_id` of that vertex. For PyG graphs the `_key` is the PyG node index.
        Running the method again replaces the documents stored before.

        :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
        n: number of nodes in graph
        embedding_size: Length of the node embeddings when they are mapped to
            d-dimensional euclidean space.
        :collection_name (type: str): Name of the document collection where we want to
            store graph embeddings.
        :batch_size (type: int): Batch size.
        :class_mapping (type: dict): It is a dictionary where class names are mapped
            to integer labels.
        If class_mappings are not provided the method uses integers labels as legend
        inside the figure. for e.g.
        {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',
        4: 'Computer Components', 5: 'Video Projectors',6: 'Routers',7: 'Tablets',
        8: 'Networking Products',9: 'Webcams'} # amazon computer dataset.
        :node_type (type: str):  Node type for which we want to store embeddings.
            Used to store Hetero graph embeddings.
        :nearest_nbors_search (type: bool): If nearest_nbors_search=True,
            store_embeddings method saves generated Graph Embeddings in ArangoDB
            along with top_k nearest neighbors (node ids with similar embeddings)
            and their corresponding similarity scores (i.e. cosine distance).
        :top_k_nbors (type: int): Returns top-k nearest neighbors of all embeddings.
        :nlist (type: int): Number of clusters to be generated.
        :search_type (type: str): We support two types of search for now:
        1. exact search: For precise similarity search but at the cost of scalability.
        2. approx search: For scalable similarity search but at the cost of some
            precision loss.
        :replication_factor (type: int): Replication factor of the collection, if it
            has to be created. By default the database default is used.
        """

        if collection_name is None:
            raise Exception("Pass arangodb collection name to store embeddings.")
        if self.database is None:
            raise Exception(
                "No ArangoDB database: set model.graph_util.database to a "
                "python-arango database before storing embeddings."
            )
        if self.key_node is not None and node_type is None:
            node_type = self.key_node

        # create document collection with name "collection_name" in arangodb
        if not self.database.has_collection(collection_name):
            if replication_factor is None:
                self.database.create_collection(collection_name)
            else:
                self.database.create_collection(
                    collection_name, replication_factor=replication_factor
                )

        emb_collection = self.database[collection_name]

        if nearest_nbors_search is True:
            dist, nbors = similarity_search(
                graph_emb, top_k_nbors=top_k_nbors, nlist=nlist, search_type=search_type
            )

        # labels of the nodes, if there are any
        store = self.graph if node_type is None else self.graph[node_type]
        y = getattr(store, "y", None)
        labels = y.cpu().tolist() if y is not None else None

        keys = self.node_keys.get(node_type)
        if keys is not None and len(keys) != graph_emb.shape[0]:
            keys = None
        source = self._source_collection(node_type)

        batch = []
        for idx in track(range(graph_emb.shape[0])):
            insert_doc: Dict[str, Any] = {}
            if keys is not None:
                insert_doc["_key"] = str(keys[idx])
                if source is not None:
                    insert_doc["source_id"] = source + "/" + str(keys[idx])
            else:
                insert_doc["_key"] = str(idx)
            insert_doc["embedding"] = graph_emb[idx].tolist()
            # add labels and class names
            if labels is not None:
                insert_doc["label"] = labels[idx]
                if class_mapping is not None:
                    insert_doc["class_name"] = class_mapping[labels[idx]]
            # add top-k nearest neighbours
            if nearest_nbors_search is True:
                insert_doc["cosine_sim"] = dist[idx].tolist()
                if keys is not None:
                    insert_doc["similar_nodes"] = [
                        str(keys[n]) for n in nbors[idx].tolist() if n >= 0
                    ]
                else:
                    insert_doc["similar_nodes"] = nbors[idx].tolist()
            batch.append(insert_doc)
            if len(batch) == batch_size:
                emb_collection.import_bulk(batch, on_duplicate="replace")
                batch = []
        if len(batch) > 0:
            emb_collection.import_bulk(batch, on_duplicate="replace")
