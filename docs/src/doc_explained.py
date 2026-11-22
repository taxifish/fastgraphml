"""Guide 1: fastgraphml explained."""

from __future__ import annotations

import diagrams
from common import code, fig, out, table

TITLE = "fastgraphml explained"

SECTIONS = [
    "What fastgraphml is for",
    "Graph embeddings in one page",
    "The three lines",
    "Step 1: getting the graph in",
    "Step 2: the four models",
    "Step 3: what training does",
    "Step 4: using the embeddings",
    "Installation and package layout",
    "Good practice and limits",
    "Reference",
]


def h2(n: int) -> str:
    return f'<h2><span class="num">{n}</span>{SECTIONS[n - 1]}</h2>'


def build() -> str:
    toc = "".join(
        f'<li><span class="num">{i}</span>{name}</li>'
        for i, name in enumerate(SECTIONS, 1)
    )
    parts = [f"""
<section class="first">
<h1>{TITLE}</h1>
<p class="lead">What the library is for, how it turns a graph into embeddings step by step,
and what happens inside each of its four models.</p>
<p class="meta">Describes fastgraphml 0.1.0 (github.com/taxifish/fastgraphml),
tested with PyTorch 2.12 and 2.15, PyG 2.8, Python 3.13. A second guide, Getting started with
fastgraphml, covers installation, first steps and the three example projects.</p>
<ul class="toc">{toc}</ul>

{h2(1)}
<p>fastgraphml is a small Python library that produces <b>graph embeddings</b> with very little
code. You give it a graph, either one stored in ArangoDB or a PyTorch Geometric (PyG) object,
choose one of four models, train it, and get one vector per node back. Those vectors can be
stored in ArangoDB next to the data they describe and used for similarity search,
recommendations, classification or visualisation.</p>
<p>The library does not invent new models. Its models are the reference implementations from
PyG. What it adds is everything around them: exporting the graph from the database, splitting
the nodes, building data loaders, the training loop, validation, checkpoints, and writing the
result back. That is the part that usually takes a few hundred lines per project.</p>
{fig(diagrams.big_picture(), "The whole library in one picture. Everything left of the model is done by the constructor, everything right of it by three methods.")}
<p>It is a good fit when you have a graph in ArangoDB and want embeddings without building a
training pipeline, when you want a baseline quickly, or when you want to compare a few
standard models on your data. It is not a framework for designing your own architectures: for
that, use PyG directly.</p>
</section>
"""]

    parts.append(f"""
<section>
{h2(2)}
<p>An embedding is a list of numbers, typically 64 to 256 of them, that stands for one node.
The model chooses the numbers so that the geometry of the vectors reflects the structure of the
graph: nodes that sit in similar surroundings get vectors that point in similar directions.</p>
{fig(diagrams.embedding_idea(), "From a graph to vectors to a map. The two groups of nodes end up in two regions of the vector space.")}
<p>Why this is useful: once every node is a vector, questions about the graph become questions
about distances, and those are cheap to answer.</p>
{table(["Task", "With embeddings it becomes", "Example"], [
    ["Similarity search", "nearest neighbours of a vector", "products like this one"],
    ["Recommendation", "nearest neighbours that are not already connected", "sites that resemble the experienced ones"],
    ["Node classification", "a simple classifier on the vectors", "predict the category of a product"],
    ["Visualisation", "projection of the vectors to 2D", "a map of all nodes, coloured by class"],
    ["Clustering", "k-means or similar on the vectors", "communities"],
], ["24%", "40%", "36%"])}
<p>All four models in fastgraphml are trained <b>unsupervised</b>: they learn from the
structure of the graph (and, for three of them, from node features), not from labels. Labels,
if the graph has them, are only used to measure how good the embeddings are.</p>
<div class="note info"><b>How quality is measured.</b> After every epoch fastgraphml trains a
logistic regression on the embeddings of the training nodes and reports its accuracy on
validation and test nodes. If a linear classifier can read the labels off the embeddings, the
embeddings have captured something meaningful. Without labels this check is skipped.</div>

{h2(3)}
<p>Every use of the library has the same shape: create a model, train it, read the
embeddings.</p>
{code('''
from fastgraphml.graph_embeddings import SAGE

model = SAGE(db, "product_graph", metagraph, embedding_size=64)   # 1  graph + model
model._train(epochs=10)                                           # 2  train
embeddings = model.get_embeddings()                               # 3  numpy array, nodes x 64
''')}
{table(["Line", "What it does"], [
    ["<code>SAGE(...)</code>", "Checks the arguments, exports the graph from ArangoDB (or takes your PyG object), splits the nodes into train, validation and test sets if there are labels, applies an optional transform, prints statistics of the graph, and builds the neural network and its data loader."],
    ["<code>_train(...)</code>", "Runs the training loop: mini-batches, loss, optimizer steps, validation after every epoch, checkpoints. <code>fit()</code> is another name for the same method."],
    ["<code>get_embeddings()</code>", "Runs the trained model over the whole graph once and returns the vectors."],
], ["24%", "76%"])}
<p>The four model classes, <code>SAGE</code>, <code>GAT</code>, <code>METAPATH2VEC</code> and
<code>DMGI</code>, all follow this pattern. The next sections go through the three lines in
detail.</p>
</section>
""")

    parts.append(f"""
<section>
{h2(4)}
<p>A model accepts a graph in one of two ways. Use one or the other, never both.</p>
<div class="two">
<div><h3>From ArangoDB</h3>
{code('''
from arango import ArangoClient

db = ArangoClient("http://localhost:8529").db(
    "shop", username="ml", password="...")

model = SAGE(db, "product_graph", metagraph)
''')}</div>
<div><h3>From a PyG object</h3>
{code('''
from torch_geometric.datasets import Planetoid

data = Planetoid("/tmp/Cora", "Cora")[0]

model = SAGE(pyg_graph=data)
''')}</div>
</div>
<h3>The metagraph</h3>
<p>ArangoDB stores documents with arbitrary attributes; a neural network needs tensors. The
metagraph is a dictionary that says which collections to read and which attributes become
which tensor.</p>
{fig(diagrams.metagraph(), "The metagraph maps collections and attributes to the tensors of a PyG graph. Numbers from the Amazon Computers graph.")}
{table(["PyG name", "Meaning", "Requirements"], [
    ["<code>x</code>", "Node features, the input of the network", "An attribute holding a list of numbers of the same length in every document. Needed by SAGE, GAT and DMGI."],
    ["<code>y</code>", "Node labels", "An integer class per document. Optional: only used for validation."],
    ["edge collections", "Become <code>edge_index</code>", "<code>{}</code> is enough; edge attributes are not used by the models."],
], ["18%", "32%", "50%"])}
<p>Three things are worth knowing about this step:</p>
<ul>
<li><b>The metagraph decides what is read.</b> The graph name passed as second argument only
labels the export in the log. Collections that are not in the metagraph are never touched, so
a metagraph is also a way to train on a part of a larger graph.</li>
<li><b>Keys are kept.</b> The adapter is asked to keep the <code>_key</code> of every vertex.
fastgraphml stores them in <code>model.graph_util.node_keys</code>, in the order of the
embedding rows, so that row 17 of the result can be traced back to its document.</li>
<li><b>Heterogeneous graphs</b> name several vertex collections. Each becomes a node type, and
every combination (from collection, edge collection, to collection) an edge type such as
<code>("movie", "to", "actor")</code>.</li>
</ul>
{code('''
metagraph = {
    "vertexCollections": {
        "movie": {"x": "x", "y": "y"},     # features and labels
        "director": {"x": "x"},            # features only
        "actor": {"x": "x"},
    },
    "edgeCollections": {"to": {},
}
''')}
<h3>What else the constructor does</h3>
<ul>
<li><b>Split.</b> If the graph has labels but no <code>train_mask</code>, the nodes are split
at random; <code>num_val=0.1, num_test=0.1</code> are the default shares.</li>
<li><b>Transform.</b> Any <code>torch_geometric.transforms</code> object passed as
<code>transform</code> is applied, for example <code>T.NormalizeFeatures()</code> or
<code>T.ToUndirected()</code>.</li>
<li><b>Statistics.</b> A summary of the graph is printed, which is a quick sanity check that
the export read what you meant:</li>
</ul>
{out('''
Homogeneous Graph Detected ........
{'Nodes': 2708, 'Node_feature_size': 1433, 'Number_of_classes': 7, 'Edges': 10556,
 'Graph Directionality': 'Undirected', 'Average node degree': '3.90',
 'Number of training nodes': 140, 'Number of val nodes': 500,
 'Number of test nodes': 1000, 'Has isolated nodes': False}
''')}
</section>
""")

    parts.append(f"""
<section>
{h2(5)}
{table(["Model", "Graph", "Needs features", "Learns from", "Embeds"], [
    ["<b>SAGE</b> (GraphSAGE)", "homogeneous", "yes", "features of a node and of its sampled neighbours", "all nodes"],
    ["<b>GAT</b>", "homogeneous", "yes", "the same, with learned attention weights per neighbour", "all nodes"],
    ["<b>METAPATH2VEC</b>", "heterogeneous", "no", "random walks along a metapath", "node types on the metapath"],
    ["<b>DMGI</b>", "heterogeneous", "yes, on the key node", "several relations between nodes of one type", "the key node type"],
], ["22%", "15%", "15%", "32%", "16%"])}
<p><b>Homogeneous</b> means one kind of node and one kind of edge (products bought together).
<b>Heterogeneous</b> means several kinds (sites, trials, indications).</p>

<h3>SAGE</h3>
<p>GraphSAGE computes the vector of a node from the node's own features and the average of
its neighbours' vectors, layer by layer. With two layers, a node sees its neighbours and their
neighbours. To keep this affordable on large graphs it does not use all neighbours but a random
sample: <code>sizes=[10, 10]</code> means ten neighbours in the first hop and ten for each of
those in the second.</p>
{fig(diagrams.sage(), "GraphSAGE: sample a neighbourhood, average it, combine it with the node itself. The loss compares pairs of nodes.")}
<p>Training is unsupervised. In every epoch each node is paired once with one of its
neighbours (a positive example) and a random pair of nodes is drawn for each of them
(a negative example). The loss is a binary cross-entropy on the dot product of the two
vectors: high for connected pairs, low for random ones.</p>
{code('''
model = SAGE(db, "product_graph", metagraph,
             embedding_size=64,     # length of the vectors
             num_layers=2,          # how many hops a node sees
             sizes=[10, 10],        # neighbours sampled per hop, one entry per layer
             batch_size=256, dropout_perc=0.5)
''')}
<h3>GAT</h3>
<p>The Graph Attention Network has the same structure and the same training as SAGE, with one
change: instead of a plain average, every neighbour gets a learned weight, so the model can
attend to some neighbours more than to others. <code>heads</code> sets how many independent
sets of weights are learned. The heads of the last layer are averaged, so the embeddings have
<code>embedding_size</code> dimensions whatever the number of heads. Each layer also has a skip
connection.</p>
{code('''
model = GAT(db, "product_graph", metagraph, embedding_size=64, heads=2)
''')}
<p>GAT is worth trying when neighbours differ a lot in relevance. It is slower than SAGE and
has more parameters.</p>
</section>

<section>
<h3>METAPATH2VEC</h3>
<p>Many graphs have no usable features: a site is just a name. METAPATH2VEC learns from
structure alone. It generates random walks that follow a fixed sequence of node and edge
types, the <b>metapath</b>, and trains vectors the way word2vec does for words in sentences:
nodes that occur close to each other on the walks get similar vectors.</p>
{fig(diagrams.metapath(), "A metapath author, paper, conference, paper, author (the DBLP example of the notebooks).")}
{code('''
metapaths = [("author", "to", "paper"),          # one walk pattern, written as a list of
             ("paper", "to", "conference"),      # (source type, edge type, target type)
             ("conference", "to", "paper"),      # steps that ends where it started
             ("paper", "to", "author")]

model = METAPATH2VEC(db, "DBLP", metagraph, metapaths, key_node="author",
                     embedding_size=128, walk_length=5, context_size=6,
                     walks_per_node=10, num_negative_samples=10, sparse=True)
model._train(epochs=15, lr=0.03)
embeddings = model.get_embeddings()       # {"author": array, "paper": array, "conference": array}
''')}
{table(["Parameter", "Meaning"], [
    ["<code>metapaths</code>", "The steps of the walk. The choice of metapath decides what similar means."],
    ["<code>key_node</code>", "The node type whose labels are used for validation."],
    ["<code>walk_length</code>, <code>walks_per_node</code>", "How long the walks are and how many start at each node."],
    ["<code>context_size</code>", "How many steps apart two nodes on a walk may be to count as a positive pair."],
    ["<code>num_negative_samples</code>", "Random nodes drawn per positive pair."],
    ["<code>sparse</code>", "Sparse gradients for the embedding table; uses the SparseAdam optimizer."],
], ["36%", "64%"])}
<p>The model is a lookup table with one row per node. That makes it fast and feature-free, and
it means it cannot produce a vector for a node it has not seen during training.</p>

<h3>DMGI</h3>
<p>Deep Multiplex Graph Infomax looks at one node type, the key node, through several
relations at once. Each metapath, for example movie, actor, movie and movie, director, movie,
is first turned into a direct edge type between key nodes. A graph convolution per relation
produces relation-specific embeddings, and a consensus embedding is trained to agree with all
of them.</p>
{fig(diagrams.dmgi(), "DMGI: one small network per relation, one consensus vector per node.")}
{code('''
metapaths = [[("movie", "actor"), ("actor", "movie")],          # each metapath is a list of
             [("movie", "director"), ("director", "movie")]]    # (source type, target type) steps

model = DMGI(db, "IMDB_X", metagraph, metapaths, key_node="movie", embedding_size=64)
model._train(epochs=100, lr=0.0005)
embeddings = model.get_embeddings()       # {"movie": array}
''')}
<p>DMGI trains on the full graph in every step, without sampling, and returns embeddings for
the key node type only. Duplicate edges are removed before the relations are computed.</p>
</section>
""")

    parts.append(f"""
<section>
{h2(6)}
{fig(diagrams.class_anatomy(), "Left: what the constructor does. Right: what one call of _train does.")}
{code('''
model._train(
    epochs=50,                                   # passes over the data
    lr=0.001,                                    # learning rate of the Adam optimizer
    ckp_path="./latest_model_checkpoint.pt",     # written after every epoch, None to skip
    best_model_path="./best_model.pt",           # written when the model improves, None to skip
    weight_decay=5e-4,                           # any other argument goes to the optimizer
)
''')}
<p>While it runs, one line per epoch is printed (five epochs of SAGE on the Cora citation
graph):</p>
{out('''
Training started .........
Epoch: 001, Train_Loss: 0.6847, Val: 0.6480, Test: 0.6420
Val Acc increased (0.00000 --> 0.64800).  Saving model ...
Epoch: 002, Train_Loss: 0.6278, Val: 0.7100, Test: 0.7520
Epoch: 003, Train_Loss: 0.5452, Val: 0.7320, Test: 0.7810
Epoch: 004, Train_Loss: 0.5169, Val: 0.7480, Test: 0.7760
Epoch: 005, Train_Loss: 0.4954, Val: 0.7600, Test: 0.7720
''')}
{table(["Column", "Meaning"], [
    ["<code>Train_Loss</code>", "Mean loss of the epoch. It should fall. If it grows or turns into nan, lower the learning rate or normalise the features."],
    ["<code>Val</code>, <code>Test</code>", "Accuracy of a logistic regression trained on the embeddings of the training nodes, on validation and test nodes. Only printed if the graph has labels."],
], ["24%", "76%"])}
<h3>Which model is the best one</h3>
<p>With labels, the best model is the one with the highest validation accuracy. Without
labels, it is the one with the lowest training loss. A checkpoint file holds the epoch, the
weights (<code>state_dict</code>) and the optimizer state, and can be loaded with
<code>torch.load</code>.</p>
<h3>Where it runs</h3>
<p>By default on the GPU if PyTorch finds one, otherwise on the CPU. Pass
<code>device="cpu"</code> or <code>device="cuda:1"</code> to the constructor to choose. SAGE
and GAT keep the graph in main memory and move one mini-batch at a time to the device.</p>
<div class="note"><b>Mini-batches need pyg-lib.</b> Neighbourhood sampling is implemented in
the pyg-lib (or torch-sparse) extension. If neither is installed, SAGE and GAT print a warning
and train on the full graph in one step per epoch. That works for graphs with a few thousand
nodes and does not scale beyond.</div>
</section>
""")

    parts.append(f"""
<section>
{h2(7)}
<h3>Reading them</h3>
{code('''
embeddings = model.get_embeddings()
# SAGE, GAT:            numpy array of shape (number of nodes, embedding_size)
# METAPATH2VEC, DMGI:   dictionary {node type: numpy array}

val_acc, test_acc = model.val()            # the node classification check, if there are labels
keys = model.graph_util.node_keys          # ArangoDB keys per node type (None for homogeneous)
graph = model.G                            # the PyG graph the model was trained on
''')}
<h3>Storing them in ArangoDB</h3>
{code('''
model.graph_util.store_embeddings(
    embeddings,                              # for hetero models: embeddings["movie"]
    collection_name="product_embeddings",    # created if it does not exist
    class_mapping={0: "Desktops", 1: "Data Storage"},   # optional names for the labels
    node_type=None,                          # the node type, for heterogeneous graphs
    nearest_nbors_search=True, top_k_nbors=10,   # also store each node's nearest neighbours
)
''')}
{fig(diagrams.store(), "One document per node. For a graph read from ArangoDB the document shares its _key with the vertex it describes.")}
<p>Running the method again replaces the documents. With <code>nearest_nbors_search=True</code>
each document also gets <code>similar_nodes</code> (the keys of the most similar nodes) and
<code>cosine_sim</code> (their scores), so a recommendation is a single document read.</p>
<p>Because embedding and vertex share a key, similarity can also be computed in AQL at query
time:</p>
{code('''
LET q = DOCUMENT("product_embeddings/8515").embedding
FOR e IN product_embeddings
    FILTER e._key != "8515"
    LET score = COSINE_SIMILARITY(e.embedding, q)
    SORT score DESC
    LIMIT 10
    RETURN { product: DOCUMENT(e.source_id), score }
''', "sql")}
<h3>Similarity search</h3>
{code('''
from fastgraphml.graph_embeddings import downstream_tasks

dist, nbors = downstream_tasks.similarity_search(embeddings, top_k_nbors=10,
                                                 search_type="exact")   # or "approx"
nbors[0]    # row numbers of the nodes most similar to node 0 (node 0 itself comes first)
dist[0]     # their cosine similarities
''')}
<p>This uses FAISS. <code>"exact"</code> compares every pair, <code>"approx"</code> first
groups the vectors into <code>nlist</code> clusters and searches only the nearest ones, which
is faster on large sets at a small cost in accuracy.</p>
<h3>Visualisation</h3>
{code('''
downstream_tasks.visualize_embeddings(model.G, embeddings, class_mapping=class_names,
                                      emb_percent=0.1, save_path="./umap.png")
''')}
<p>Projects the first <code>emb_percent</code> of the embeddings to two dimensions with UMAP
and colours the points by label. Well separated colours are a quick visual confirmation that
the embeddings carry the class structure.</p>
</section>
""")

    parts.append(f"""
<section>
{h2(8)}
{fig(diagrams.stack(), "fastgraphml sits on PyG and PyTorch. Three optional extras add the ArangoDB adapter, FAISS and UMAP.")}
{code('''
pip install torch
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
pip install "fastgraphml[all] @ git+https://github.com/taxifish/fastgraphml"
''', "bash")}
{table(["Extra", "Installs", "Needed for"], [
    ["<code>arango</code>", "adbpyg-adapter, arango-datasets", "reading graphs from ArangoDB, demo datasets"],
    ["<code>search</code>", "faiss-cpu", "<code>similarity_search</code>, nearest neighbours in <code>store_embeddings</code>"],
    ["<code>viz</code>", "umap-learn, matplotlib, seaborn", "<code>visualize_embeddings</code>"],
    ["<code>all</code>", "all of the above", ""],
], ["16%", "36%", "48%"])}
<p>The optional packages are imported when first used. Without the <code>arango</code> extra
the models still work with <code>pyg_graph=...</code>.</p>
<h3>Package layout</h3>
{table(["Module", "Content"], [
    ["<code>graph_embeddings/models/base.py</code>", "Shared training code of SAGE and GAT, checkpoints, the validation routine"],
    ["<code>graph_embeddings/models/graph_sage.py</code>, <code>gat.py</code>", "The layers of the two homogeneous models"],
    ["<code>graph_embeddings/models/metapath2vec.py</code>, <code>dmgi.py</code>", "The two heterogeneous models"],
    ["<code>graph_embeddings/utils.py</code>", "<code>GraphUtils</code>: export from ArangoDB, preprocessing, statistics, <code>store_embeddings</code>"],
    ["<code>graph_embeddings/downstream_tasks/</code>", "<code>similarity_search</code>, <code>visualize_embeddings</code>"],
], ["46%", "54%"])}

{h2(9)}
<ul>
<li><b>Normalise features that are counts.</b> On the Amazon Computers graph, SAGE with a
learning rate of 0.005 diverged on the raw word counts (3% accuracy) and reached 68% after
ten epochs with <code>transform=T.NormalizeFeatures()</code>.</li>
<li><b>Look at the spread of the embeddings before ranking with them.</b> After a short
training on that graph, 93% of the variance of the SAGE embeddings lay in one direction, and
all nearest-neighbour similarities were above 0.99. PCA whitening of the embeddings fixed the
ranking (see the product recommender example). METAPATH2VEC embeddings did not show this.</li>
<li><b>Labels measure, they do not teach.</b> A high validation accuracy means the embeddings
separate the classes. It says nothing about classes you did not label.</li>
<li><b>The metapath is a modelling decision.</b> Author, paper, author makes co-authors
similar; author, paper, conference, paper, author makes authors of the same community similar.
Only one metapath per METAPATH2VEC model is used.</li>
<li><b>Embeddings belong to one training run.</b> Vectors from two runs are not comparable,
even with the same settings. After retraining, replace all stored embeddings.</li>
</ul>
{table(["Limit", "Consequence"], [
    ["<code>get_embeddings</code> of SAGE and GAT runs over the whole graph at once", "Memory grows with the number of edges; very large graphs need a GPU with enough memory or a different tool"],
    ["METAPATH2VEC and DMGI learn one vector per node", "New nodes need a new training"],
    ["DMGI trains on the full graph", "Suited to graphs up to roughly a hundred thousand key nodes"],
    ["Edge attributes and edge weights are ignored", "All edges count the same"],
    ["Progress is reported with print", "Applications that want a progress bar have to read the output (the examples show how)"],
], ["48%", "52%"])}
<div class="note"><b>Tested and not tested.</b> The models, training, similarity search and
visualisation have been run on PyG graphs (Cora, Amazon Computers, synthetic heterogeneous
graphs). Reading from and writing to ArangoDB is covered by tests that need a running
database; they have not been run for this version yet.</div>
</section>
""")

    common_rows = [
        [
            "<code>database</code>, <code>arango_graph</code>, <code>metagraph</code>",
            "",
            "ArangoDB route: python-arango database, a name for the export, the metagraph",
        ],
        [
            "<code>pyg_graph</code>",
            "",
            "PyG route: a <code>Data</code> or <code>HeteroData</code> object",
        ],
        ["<code>embedding_size</code>", "64", "Length of the vectors"],
        [
            "<code>transform</code>",
            "None",
            "A <code>torch_geometric.transforms</code> object applied to the graph",
        ],
        [
            "<code>num_val</code>, <code>num_test</code>",
            "0.1",
            "Shares of the nodes used for validation and test, if a split has to be made",
        ],
        ["<code>device</code>", "auto", '<code>"cpu"</code>, <code>"cuda"</code>, ...'],
    ]
    parts.append(f"""
<section>
{h2(10)}
<h3>Arguments shared by all models</h3>
{table(["Argument", "Default", "Meaning"], common_rows, ["38%", "10%", "52%"])}
<h3>SAGE and GAT</h3>
{table(["Argument", "Default", "Meaning"], [
    ["<code>num_layers</code>", "2", "Number of layers, i.e. hops"],
    ["<code>sizes</code>", "[10, 10]", "Neighbours sampled per hop, one entry per layer; -1 takes all"],
    ["<code>batch_size</code>", "256", "Positive pairs per mini-batch"],
    ["<code>dropout_perc</code>", "0.5", "Dropout between layers"],
    ["<code>shuffle</code>", "True", "Shuffle the nodes every epoch"],
    ["<code>heads</code> (GAT)", "2", "Attention heads"],
    ["other keyword arguments", "", "Passed to PyG's <code>SAGEConv</code> or <code>GATConv</code>"],
], ["38%", "10%", "52%"])}
<h3>METAPATH2VEC</h3>
{table(["Argument", "Default", "Meaning"], [
    ["<code>metapaths</code>", "required", "List of (source type, edge type, target type) steps"],
    ["<code>key_node</code>", "first type", "Node type used for validation"],
    ["<code>walk_length</code>, <code>context_size</code>", "5, 6", "Length of a walk, size of the context window"],
    ["<code>walks_per_node</code>", "5", "Walks started at each node"],
    ["<code>num_negative_samples</code>", "5", "Negative samples per positive pair"],
    ["<code>num_nodes_dict</code>", "None", "Number of nodes per type, if it cannot be read from the edges"],
    ["<code>sparse</code>, <code>batch_size</code>, <code>shuffle</code>", "False, 64, True", "Sparse gradients, start nodes per batch, shuffling"],
], ["38%", "14%", "48%"])}
<h3>DMGI</h3>
{table(["Argument", "Default", "Meaning"], [
    ["<code>metapaths</code>", "required", "List of metapaths, each a list of (source type, target type) steps"],
    ["<code>key_node</code>", "required", "The node type to embed; it needs features"],
    ["<code>dropout_perc</code>", "0.5", "Dropout on the input features"],
], ["38%", "14%", "48%"])}
<h3>Methods</h3>
{table(["Call", "Returns"], [
    ["<code>model._train(epochs, lr, ckp_path, best_model_path, **optimizer)</code> or <code>model.fit(...)</code>", "nothing; trains the model. METAPATH2VEC also takes <code>log_steps</code> and <code>eval_steps</code>"],
    ["<code>model.get_embeddings()</code>", "array, or dictionary of arrays per node type"],
    ["<code>model.val()</code>", "(validation accuracy, test accuracy); raises if there are no labels"],
    ["<code>model.graph_util.store_embeddings(emb, collection_name, ...)</code>", "nothing; writes to ArangoDB"],
    ["<code>model.graph_util.graph_stats()</code>", "dictionary with the statistics printed at start"],
    ["<code>downstream_tasks.similarity_search(emb, top_k_nbors, nlist, search_type)</code>", "(similarities, neighbour indices)"],
    ["<code>downstream_tasks.visualize_embeddings(graph, emb, class_mapping, node_type, emb_percent, save_path)</code>", "nothing; draws and saves a UMAP plot"],
], ["60%", "40%"])}
</section>
""")
    return "".join(parts)
