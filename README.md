# fastgraphml
[![build](https://github.com/arangoml/fastgraphml/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/arangoml/fastgraphml/actions/workflows/build.yml)
[![Last commit](https://img.shields.io/github/last-commit/arangoml/fastgraphml)](https://github.com/arangoml/fastgraphml/commits/main)
[![Downloads](https://static.pepy.tech/personalized-badge/fastgraphml?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://pepy.tech/project/fastgraphml)
[![Coverage Status](https://coveralls.io/repos/github/arangoml/fastgraphml/badge.svg?branch=main)](https://coveralls.io/github/arangoml/fastgraphml?branch=main)

[![Code style: black](https://img.shields.io/static/v1?style=for-the-badge&label=code%20style&message=black&color=black)](https://github.com/psf/black)

*Democratizing Graph Machine Learning with Low-Code*

Given an input graph it generates Graph Embeddings using Low-Code framework built on top of [PyG](https://pytorch-geometric.readthedocs.io/en/latest/). The package supports training on both GPU and CPU enabled machines. Training jobs on GPUs results in much faster execution and increased in performance when it comes to handling large graphs as compared to CPUs. In addition, the framework provides tight integration with  [ArangoDB](https://www.arangodb.com/) which is a scalable, fully managed graph database, document store and search engine in one place. Once Graph Embeddings are generated, they can be used for various downstream machine learning tasks like Node Classification, Link Prediction, Visualisation, Community Detection, Similartiy Search, Recommendation, etc. 

## Installation

Python 3.10 or newer is required. The library is tested with PyTorch 2.12 and 2.15, PyG 2.8 and NumPy 2.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA version (run `nvidia-smi` to find it).
2. Install the PyG extension matching your PyTorch and CUDA version, see the [PyG installation notes](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). For example, for PyTorch 2.12 on CPU:
    ```
    pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
    ```
    `pyg-lib` provides the neighborhood sampling used to train `SAGE` and `GAT` on mini-batches. Without it (and without `torch-sparse`) these models are trained on the full graph, which only works for small graphs. Wheels are published for specific PyTorch versions, so pick a PyTorch version they exist for. `torch-cluster` is not needed anymore.
3. Install fastgraphml:
    ```
    pip install "fastgraphml[all] @ git+https://github.com/taxifish/fastgraphml"
    ```

The extras are optional, leave out what you do not need:

Extra | Installs | Needed for
----- | -------- | ----------
`arango` | `adbpyg-adapter`, `arango-datasets` | reading graphs from [ArangoDB](https://www.arangodb.com/), example datasets
`search` | `faiss-cpu` | similarity search
`viz` | `umap-learn`, `matplotlib`, `seaborn` | visualisation of embeddings
`all` | all of the above |

Note: `adbpyg-adapter` needs `torch-sparse` and `torch-scatter`, install them from the same wheel index as `pyg-lib` first (`pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html`), otherwise pip compiles them from source. It also requires `python-arango` 7.x.

### Changes in 0.1.0

* `SAGE` and `GAT` use PyG's `LinkNeighborLoader` instead of the deprecated `NeighborSampler` and `torch-cluster`.
* Labels are optional for all models: without labels (`y`) the validation is skipped and the best model is the one with the lowest training loss.
* `GAT` returns embeddings with `embedding_size` dimensions (before: `embedding_size * heads`).
* `_train(epochs=N)` trains for N epochs (before: N - 1). `fit()` is an alias of `_train()`, and `ckp_path=None` / `best_model_path=None` turn the checkpoints off.
* All models take a `device` argument.
* `store_embeddings` keeps the link to the source documents for graphs read from ArangoDB (same `_key`, plus a `source_id` attribute), uses its `top_k_nbors`, `nlist` and `search_type` arguments, does not hardcode the replication factor anymore and replaces existing documents when it is run again.
* `similarity_search` no longer normalises the embeddings passed to it in place.
* Parallel edges are removed before metapaths are added for `DMGI`.

## Quickstart: Graph Embedding Generation

### Use Case 1: Generates Graph Embeddings using the graphs stored inside ArangoDB:

#### Example Homogneous Graphs

```python
from fastgraphml.graph_embeddings import SAGE, GAT
from fastgraphml.graph_embeddings import downstream_tasks
from fastgraphml import Datasets 
from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient("http://127.0.0.1:8529")
db = client.db('_system', username='root', password='')

# Loading Amazon Computer Products dataset into ArangoDB
Datasets(db).load("AMAZON_COMPUTER_PRODUCTS")

# Optionally use arangodb graph
# arango_graph = db.graph('product_graph')

# metadata information of arango_graph
metagraph = {
    "vertexCollections": {
        "Computer_Products": {"x": "features", "y": "label"},
    },
    "edgeCollections": {
        "bought_together": {},
    },
}

# generating graph embeddings with 3 lines of code
model = SAGE(db,'product_graph', metagraph, embedding_size=64) # define graph embedding model
model._train(epochs=10) # train
embeddings = model.get_embeddings() # get embeddings
```

#### Example Heterogeneous Graphs

```python
from fastgraphml.graph_embeddings import METAPATH2VEC, DMGI
from fastgraphml.graph_embeddings import downstream_tasks 
from fastgraphml import Datasets 

from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient("http://127.0.0.1:8529")
db = client.db('_system', username='root')

# Loading IMDB Dataset into ArangoDB
Datasets(db).load("IMDB_X")

# Optionally use ArangoDB Graph
# arango_graph = db.graph("IMDB")

metagraph = {
    "vertexCollections": {
    
        "movie": { "x": "x", "y": "y"},  
        "director": {"x": "x"},
        "actor": {"x": "x"},
    },
    "edgeCollections": {
        "to": {},
    },
}
metapaths = [('movie', 'to','actor'),
             ('actor', 'to', 'movie'), ] # MAM # co-actor relationship

# generating graph embeddings with 3 lines of code
model = METAPATH2VEC(db, "IMDB_X", metagraph, metapaths, key_node='movie', embedding_size=128,
                     walk_length=5, context_size=6, walks_per_node=5, num_negative_samples=5,
                     sparse=True) # define model
model._train(epochs=10, lr=0.03) # train
embeddings = model.get_embeddings() # get embeddings
```

### Use Case 2: Generates Graph Embeddings using PyG graphs:

```python
from fastgraphml.graph_embeddings import SAGE, GAT
from fastgraphml.graph_embeddings import downstream_tasks 
from torch_geometric.datasets import Planetoid

# load pyg dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# generating graph embeddings with 3 lines of code
model = SAGE(pyg_graph=data, embedding_size=64) # define graph embedding model
model._train(epochs=10) # train
embeddings = model.get_embeddings() # get embeddings
```
## Models Supported

Model         | Homogeneous   | Heterogeneous | Node Features
------------- | ------------- | ------------- | ------------- 
[GraphSage](https://arxiv.org/abs/1706.02216)     | ✔️             |               | ✔️ 
[GAT](https://arxiv.org/abs/1710.10903)           | ✔️             |               | ✔️ 
[Metapath2Vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)  |               | ✔️             |  
[DMGI](https://arxiv.org/pdf/1911.06750.pdf)          |               | ✔️             | ✔️ 



## Quickstart: Downstream Tasks
In addition, the library also provides various low-code helper methods to carry out number of downstream tasks such as visualisation, similarity search (recommendation) , and link prediction (to be added soon).

### Downstream Task 1: Graph Embedding Visualisation
This method helps in visualization of generated Graph Embeddings by reducing them 2 dimensions using U-Map.
#### Example
```python
# amazon computers dataset
class_names = {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',4: 'Computer Components',
 5: 'Video Projectors',6: 'Routers',7: 'Tablets',8: 'Networking Products',9: 'Webcams'}
# with one line of code
downstream_tasks.visualize_embeddings(model.G, embeddings, class_mapping=class_names, emb_percent=0.1,
                                      save_path="./umap_embd_visualization.png") # model.G is PyG data object
```
### Downstream Task 2: Scalable Similarity Search with Faiss
[Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) is a tool developed by Facebook that performs similarity search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
We support two types of search for now:
1. exact search: For precise similarity search but at the cost of scalability.
2. approx search: For scalable similarity search but at the cost of some precision loss.
#### Example 1
```python
downstream_tasks.similarity_search(embeddings, top_k_nbors=10, nlist=10, search_type='exact')
```
#### Example 2
If nearest_nbors_search=True, store_embeddings method saves generated Graph Embeddings in ArangoDB along with top_k nearest neighbors (node ids with similar embeddings) and their corresponding similarity scores (i.e. cosine distance). 
```python
model.graph_util.store_embeddings(embeddings, collection_name='emb', batch_size=100, class_mapping=None, 
        nearest_nbors_search=False, top_k_nbors=10, nlist=10, search_type='exact')
```
If the graph was read from ArangoDB, every document of the `emb` collection has the same `_key` as the vertex it belongs to and a `source_id` attribute with the `_id` of that vertex, so embeddings can be joined with their vertices in AQL:
```
FOR e IN emb
    LET vertex = DOCUMENT(e.source_id)
    RETURN { vertex, embedding: e.embedding }
```

