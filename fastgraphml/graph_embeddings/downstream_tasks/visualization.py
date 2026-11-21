import math
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
from torch_geometric.data import Data


def visualize_embeddings(
    graph: Data,
    graph_emb: Union[npt.NDArray[np.float64], Any],
    class_mapping: Optional[Dict[int, str]] = None,
    node_type: Optional[str] = None,
    emb_percent: float = 0.1,
    save_path: Optional[str] = "./umap_embd_visualization.png",
) -> None:
    """Performs dimensionality reduction (2D) and visualization of graph embeddings
    using U-Map.

    :graph (type: PyG data object): PyG data object which can be accessed using model.G
        inside the framework.
    :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
    n: number of nodes in graph,
    embedding_size: Length of the node embeddings when they are mapped to d-dimensional
        euclidean space.
    :class_mapping (type: dict): It is a dictionary where class names are mapped to
        integer labels. If class_mappings are not provided the method uses integers
        labels as legend inside the figure. for e.g.
        {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',
        4: 'Computer Components', 5: 'Video Projectors',6: 'Routers',7: 'Tablets',
        8: 'Networking Products',9: 'Webcams'} # amazon computer dataset.
    :node_type (type: str):  Node type for which we want to retreive embeddings to
         perform visualization. Used to store Hetero graph embeddings.
    :emb_perc (type: float): Percentage of embeddings to visualize, 0.1 means 10% of
         data will be visualized.
    :save_path (type: str): Where the figure is saved. Pass None to not save it.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import umap
    except ImportError as e:
        raise ImportError(
            "Visualisation needs umap-learn, matplotlib and seaborn: "
            "pip install 'fastgraphml[viz]'"
        ) from e

    store = graph[node_type] if node_type is not None else graph
    y = getattr(store, "y", None)

    num_nodes = math.floor(graph_emb.shape[0] * emb_percent)
    graph_emb = graph_emb[:num_nodes]
    umap_embd = umap.UMAP().fit_transform(graph_emb)
    plt.figure(figsize=(8, 8))

    if y is None:
        # graph without labels: nothing to colour the points with
        sns.scatterplot(x=umap_embd.T[0], y=umap_embd.T[1])
    else:
        labels = y.cpu().numpy()[:num_nodes]
        if class_mapping is not None:
            hue = np.array([class_mapping[label] for label in labels.tolist()])
        else:
            hue = labels
        palette = {name: f"C{n}" for n, name in enumerate(sorted(set(hue.tolist())))}
        sns.scatterplot(x=umap_embd.T[0], y=umap_embd.T[1], hue=hue, palette=palette)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
