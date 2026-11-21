"""Guide 2: getting started, and the three example projects."""

from __future__ import annotations

import base64
from pathlib import Path

import diagrams
from common import code, fig, out, table

TITLE = "Getting started with fastgraphml"
ASSETS = Path(__file__).resolve().parents[2] / "examples" / "docs-assets"

SECTIONS = [
    "Install",
    "First embeddings in five minutes",
    "The pieces you will use",
    "Your own graph in ArangoDB",
    "Choosing a model and settings",
    "The example projects",
    "Example 1: embedding pipeline CLI",
    "Example 2: product recommender",
    "Example 3: clinical site finder",
    "Troubleshooting",
]


def h2(n: int) -> str:
    return f'<h2><span class="num">{n}</span>{SECTIONS[n - 1]}</h2>'


def image(name: str, caption: str) -> str:
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return fig(f'<img src="data:image/png;base64,{data}" alt="">', caption)


def build() -> str:
    toc = "".join(
        f'<li><span class="num">{i}</span>{name}</li>'
        for i, name in enumerate(SECTIONS, 1)
    )
    parts = [f"""
<section class="first">
<h1>{TITLE}</h1>
<p class="lead">From an empty Python environment to embeddings of your own graph, followed by
a tour of the three example projects in the repository.</p>
<p class="meta">Describes fastgraphml 0.1.0 (branch examples-and-docs of taxifish/fastgraphml).
The companion guide, fastgraphml explained, describes how the models work.</p>
<ul class="toc">{toc}</ul>

{h2(1)}
<p>You need Python 3.10 or newer. A GPU is optional: everything in this guide runs on a laptop
CPU. ArangoDB is only needed from section 4 on.</p>
{fig(diagrams.stack(), "What gets installed. The extras on the right are optional.")}
<ol class="steps">
<li><b>Create an environment and install PyTorch.</b>
{code('''
git clone https://github.com/taxifish/fastgraphml && cd fastgraphml
git checkout examples-and-docs
python -m venv .venv && source .venv/bin/activate
pip install torch
''', "bash")}</li>
<li><b>Install the PyG extensions built for your PyTorch version.</b> They provide the
neighbourhood sampling SAGE and GAT use for mini-batches, and the adapter lists two of them as
requirements.
{code('''
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html
''', "bash")}
If no wheels exist for your PyTorch version or platform, see Troubleshooting.</li>
<li><b>Install fastgraphml and the packages of the examples.</b>
{code('''
pip install -e ".[all]"
pip install -r examples/requirements.txt
''', "bash")}</li>
<li><b>Check.</b>
{code('''
python examples/01-embedding-cli/fgml.py check
''', "bash")}
It lists the versions, says whether neighbourhood sampling and the optional packages are
available, and tries to connect to the database (which fails until section 4; that is fine).</li>
</ol>
</section>
"""]

    parts.append(f"""
<section>
{h2(2)}
<p>No database yet. PyG ships small benchmark graphs; Cora is a citation graph with 2,708
papers, 5,278 citations between them, a bag-of-words feature vector per paper and one of seven
topics as label.</p>
{code('''
from torch_geometric.datasets import Planetoid
from fastgraphml.graph_embeddings import SAGE, downstream_tasks

data = Planetoid(root="/tmp/Cora", name="Cora")[0]      # downloads on first use

model = SAGE(pyg_graph=data, embedding_size=64)         # 1  graph + model
model._train(epochs=5)                                  # 2  train
embeddings = model.get_embeddings()                     # 3  one vector per paper

print(embeddings.shape)                                 # (2708, 64)
''')}
<p>The constructor prints what it found, training prints one line per epoch:</p>
{out('''
Homogeneous Graph Detected ........
{'Nodes': 2708, 'Node_feature_size': 1433, 'Number_of_classes': 7, 'Edges': 10556, ...}
Training started .........
Epoch: 001, Train_Loss: 0.6847, Val: 0.6480, Test: 0.6420
Epoch: 002, Train_Loss: 0.6278, Val: 0.7100, Test: 0.7520
...
Epoch: 005, Train_Loss: 0.4954, Val: 0.7600, Test: 0.7720
''')}
<p><code>Val</code> and <code>Test</code> answer the question "are these embeddings any
good?": a simple classifier that only sees the embeddings predicts the topic of unseen papers
correctly 77% of the time, after five epochs and a few seconds. The model itself never saw the
topics.</p>
<h3>Ask the embeddings a question</h3>
{code('''
dist, nbors = downstream_tasks.similarity_search(embeddings, top_k_nbors=5)

print(nbors[0])            # [   0 2582 1862  926 1405  633]
print(dist[0].round(3))    # [1.    0.967 0.962 0.949 0.926 0.919]
''')}
<p>The papers most similar to paper 0 are papers 2582, 1862 and so on (paper 0 itself always
comes first, with similarity 1). That is a recommender in two lines.</p>
<h3>Look at them</h3>
{code('''
downstream_tasks.visualize_embeddings(data, embeddings, emb_percent=1.0, save_path="cora.png")
''')}
<p>This needs the <code>viz</code> extra. The picture shows all papers as dots, coloured by topic. Papers of one topic form groups.</p>
<div class="note info"><b>Two files appear in the working directory</b> during training:
<code>latest_model_checkpoint.pt</code> and <code>best_model.pt</code>. Pass
<code>ckp_path=None, best_model_path=None</code> to <code>_train</code> if you do not want
them.</div>

{h2(3)}
{table(["Piece", "What it is", "You use it to"], [
    ["<code>SAGE</code>, <code>GAT</code>, <code>METAPATH2VEC</code>, <code>DMGI</code>", "The model classes", "pick how embeddings are learned"],
    ["<code>metagraph</code>", "A dictionary", "say which ArangoDB collections and attributes to read"],
    ["<code>model._train()</code> / <code>model.fit()</code>", "The training loop", "train; set epochs and learning rate"],
    ["<code>model.get_embeddings()</code>", "Array, or dictionary of arrays", "get the result"],
    ["<code>model.val()</code>", "Two accuracies", "judge the result, if the graph has labels"],
    ["<code>model.G</code>", "The PyG graph", "look at the data the model saw"],
    ["<code>model.graph_util</code>", "Helper object", "<code>store_embeddings</code>, <code>node_keys</code>, <code>graph_stats</code>"],
    ["<code>downstream_tasks</code>", "Two functions", "<code>similarity_search</code>, <code>visualize_embeddings</code>"],
], ["36%", "26%", "38%"])}
<p>Three rules cover most surprises:</p>
<ul>
<li><b>A graph comes either from ArangoDB or from PyG.</b> Pass <code>database</code>,
<code>arango_graph</code> and <code>metagraph</code>, or pass <code>pyg_graph</code>.</li>
<li><b>Homogeneous models return an array, heterogeneous models a dictionary</b> with one
array per node type.</li>
<li><b>Labels are optional.</b> Without them nothing is validated and the lowest loss decides
which checkpoint is the best.</li>
</ul>
</section>
""")

    parts.append(f"""
<section>
{h2(4)}
<ol class="steps">
<li><b>Create a database and a user for the experiments</b>, in <code>arangosh</code> as root
or in the web interface. The examples of this repository use exactly this setup and never
touch another database.
{code('''
db._createDatabase("fastgraphml_examples");
const users = require("@arangodb/users");
users.save("fastgraphml", "choose-a-password");
users.grantDatabase("fastgraphml", "fastgraphml_examples", "rw");
''', "javascript")}</li>
<li><b>Connect</b> with python-arango.
{code('''
from arango import ArangoClient

db = ArangoClient("http://localhost:8529").db(
    "fastgraphml_examples", username="fastgraphml", password="choose-a-password")
''')}</li>
<li><b>Get a graph in.</b> To follow along, load the demo graph of 13,752 computer products
and their co-purchases. For your own data, skip this.
{code('''
from fastgraphml import Datasets

Datasets(db).load("AMAZON_COMPUTER_PRODUCTS")
''')}</li>
<li><b>Write the metagraph.</b> Open a document of each vertex collection and find the
attribute with the numeric features and, if there is one, the class label. The CLI example
does this for you: <code>python examples/01-embedding-cli/fgml.py inspect</code> prints the
collections, their attributes and a proposed metagraph.
{code('''
metagraph = {
    "vertexCollections": {"Computer_Products": {"x": "features", "y": "label"}},
    "edgeCollections": {"bought_together": {}},
}
''')}</li>
<li><b>Train.</b> The features of this graph are word counts, so they are normalised first.
{code('''
import torch_geometric.transforms as T
from fastgraphml.graph_embeddings import SAGE

model = SAGE(db, "product_graph", metagraph, embedding_size=64,
             batch_size=512, transform=T.NormalizeFeatures())
model._train(epochs=10, lr=0.005)
embeddings = model.get_embeddings()
''')}</li>
<li><b>Store</b> the embeddings where the rest of your application can read them.
{code('''
model.graph_util.store_embeddings(embeddings, collection_name="product_embeddings")
''')}</li>
<li><b>Query</b> in AQL. Every embedding document has the <code>_key</code> of its product
and a <code>source_id</code> pointing at it.
{code('''
LET q = DOCUMENT("product_embeddings", @key).embedding
FOR e IN product_embeddings
    FILTER e._key != @key
    LET score = COSINE_SIMILARITY(e.embedding, q)
    SORT score DESC
    LIMIT 10
    RETURN { product: DOCUMENT(e.source_id), score }
''', "sql")}</li>
</ol>
<div class="note"><b>Not yet verified against a server.</b> Steps 2 to 7 follow the
documented behaviour of python-arango, the adapter and AQL, and the same code is covered by
the repository's database tests, but those have not been run for this version. If a step
fails, the error message is the most useful thing to report.</div>
<h3>What if my vertices have no features?</h3>
<p>SAGE, GAT and DMGI need a list of numbers per node. Three options, from simple to
better:</p>
<ul>
<li>Use <b>METAPATH2VEC</b>, which needs none (example 3).</li>
<li>Build features from what you have: counts, amounts, one-hot encoded categories, text
embeddings of a description. Store them as a list attribute.</li>
<li>Let PyG compute structural features with a transform, e.g.
<code>transform=T.LocalDegreeProfile()</code> on a PyG graph.</li>
</ul>

{h2(5)}
{table(["Your graph", "Start with", "Then try"], [
    ["One node type, features available", "<code>SAGE</code>", "<code>GAT</code> if neighbours differ much in relevance"],
    ["Several node types, few or no features", "<code>METAPATH2VEC</code>", "another metapath"],
    ["Several node types, features on the type you care about, several relations between its nodes", "<code>DMGI</code>", "<code>METAPATH2VEC</code> for comparison"],
], ["48%", "22%", "30%"])}
{table(["Setting", "Sensible start", "Change it when"], [
    ["<code>embedding_size</code>", "64", "128 or 256 for large graphs with many classes; smaller is faster to search"],
    ["<code>epochs</code>", "10 for SAGE and GAT, 15 for METAPATH2VEC, 100 for DMGI", "the validation accuracy or the loss still improves at the end"],
    ["<code>lr</code>", "0.001 to 0.005 (SAGE, GAT), 0.03 (METAPATH2VEC), 0.0005 (DMGI)", "lower it if the loss jumps or becomes nan"],
    ["<code>sizes</code>", "[10, 10]", "smaller for speed, larger for very sparse graphs"],
    ["<code>batch_size</code>", "256 to 512", "larger on a GPU"],
    ["<code>walk_length</code>, <code>walks_per_node</code>", "20 and 10 for small graphs", "more walks if nodes have few edges"],
], ["26%", "36%", "38%"])}
<p>Judge a setting by <code>model.val()</code> if you have labels, and always by looking at a
few nearest-neighbour lists of nodes you know.</p>
</section>
""")

    parts.append(f"""
<section>
{h2(6)}
<p>The folder <code>examples/</code> holds three projects that use fastgraphml the way an
application would. Each has its own README with the full documentation; this guide summarises
them.</p>
{table(["Project", "Use case", "Model", "Interface"], [
    ["<code>01-embedding-cli</code>", "Run embedding jobs for any graph from a config file", "all four", "command line"],
    ["<code>02-product-recommender</code>", "Recommend products on a co-purchase graph", "SAGE or GAT", "FastAPI + React"],
    ["<code>03-clinical-site-finder</code>", "Find trial sites that resemble the experienced ones", "METAPATH2VEC", "FastAPI + React"],
], ["28%", "40%", "16%", "16%"])}
{fig(diagrams.examples_flow(), "The flow all three projects share: train from the graph, store the embeddings, answer questions with AQL.")}
<h3>Two modes</h3>
<p><b>ArangoDB mode</b> is the default: graph and embeddings live in the database configured
in <code>examples/.env</code> (copy <code>examples/.env.example</code> and fill in the password
of the user created in section 4).</p>
<p><b>Offline mode</b> needs no database: the graph comes from a PyG dataset or a generator
and everything is kept in memory. Start the web apps with <code>FGML_MODE=offline</code>, or
use the <code>*_offline.json</code> configs of the CLI.</p>
<div class="note"><b>Tested so far:</b> the offline mode of all three projects, end to end,
including both web interfaces. The ArangoDB mode was written without access to a server and has
not been run yet.</div>
<h3>What the examples create in the database</h3>
{table(["Collections", "Created by"], [
    ["<code>Computer_Products</code>, <code>bought_together</code>", "<code>fgml.py load-demo</code> or <code>python -m backend.load_data</code> in project 02"],
    ["<code>product_embeddings</code>", "training in projects 01 and 02"],
    ["<code>site</code>, <code>trial</code>, <code>indication</code>, <code>sponsor</code>, <code>conducts</code>, <code>studies</code>, <code>funds</code>", "<code>python -m backend.load_data</code> in project 03"],
    ["<code>site_embeddings</code>", "training in project 03"],
], ["56%", "44%"])}
</section>
""")

    parts.append(f"""
<section>
{h2(7)}
<h3>Use case</h3>
<p>A team keeps several graphs in ArangoDB and needs fresh embeddings for each of them
regularly, because other services read them. Nobody wants a notebook per graph. The CLI turns
an embedding job into a JSON file and one command, which a scheduler can run.</p>
{table(["Command", "Purpose"], [
    ["<code>check</code>", "Versions, optional packages, database connection"],
    ["<code>inspect</code>", "Collections and attributes of the database, and a proposed metagraph"],
    ["<code>load-demo</code>", "Load a demo dataset"],
    ["<code>run CONFIG</code>", "Train, evaluate, save a run folder, store the embeddings"],
    ["<code>similar RUN KEY</code>", "Nearest neighbours of a node, from the files or with <code>--db</code> from ArangoDB"],
    ["<code>export RUN</code>", "Embeddings as CSV or JSON lines"],
], ["24%", "76%"])}
<h3>Try it</h3>
{code('''
cd examples/01-embedding-cli
python fgml.py run configs/cora_offline.json
python fgml.py similar runs/cora 42 -k 5
''', "bash")}
{out('''
    Most similar to nodes/42 (Rule Learning)
 #   key    cosine similarity   class
 1   1372              0.9457   Rule Learning
 2   447               0.9389   Rule Learning
 3   56                0.9362   Rule Learning
 4   87                0.9334   Rule Learning
 5   1362              0.9286   Rule Learning
''')}
<h3>How it works</h3>
{fig(diagrams.cli(), "One config file drives the whole job. The run folder and the database both receive the result.")}
<p>A config has four sections. This one trains GraphSAGE on the product graph in ArangoDB
and stores the result with the ten nearest neighbours of every product:</p>
{code('''
{
  "name": "amazon",
  "source": {
    "type": "arangodb",
    "graph": "product_graph",
    "metagraph": {
      "vertexCollections": { "Computer_Products": { "x": "features", "y": "label" } },
      "edgeCollections": { "bought_together": {} }
    }
  },
  "model": {
    "name": "SAGE",
    "params": { "embedding_size": 64, "batch_size": 512, "transform": ["NormalizeFeatures"] }
  },
  "train": { "epochs": 10, "lr": 0.005 },
  "store": { "collection": "product_embeddings", "nearest_neighbors": 10 }
}
''', "json")}
<p>The fastgraphml part of the tool is short. The model is built from the config by name, and
<code>model_inputs</code> returns either the three ArangoDB arguments or
<code>pyg_graph</code>:</p>
{code('''
def build_model(config):
    name = config["model"]["name"].upper()             # SAGE, GAT, METAPATH2VEC or DMGI
    params = dict(config["model"].get("params", {}))
    ...
    return getattr(fastgraphml, name)(**model_inputs(config["source"]), **params)

model = build_model(config)
model._train(ckp_path=None, best_model_path=str(run_dir / "best_model.pt"), **train)
if model.has_labels:
    val_acc, test_acc = model.val()
''')}
<p>One helper, <code>node_sets</code>, hides the difference between the model families: it
turns the array of SAGE or GAT and the dictionary of METAPATH2VEC or DMGI into the same
structure, with the ArangoDB keys from <code>model.graph_util.node_keys</code> next to each
array. Everything after that (saving, storing, searching, exporting) works on that
structure.</p>
</section>
""")

    parts.append(f"""
<section>
{h2(8)}
<h3>Use case</h3>
<p>A shop knows which products are bought together: 13,752 products, 245,861 co-purchase
edges. The edges already give a recommendation list, but only products that were literally
bought with this one. A product bought in the same kind of company never shows up. GraphSAGE
embeddings close that gap, and the app shows both answers side by side.</p>
{image("product-recommender.png", "The product recommender. Left: catalogue. Centre: all products, placed by their embeddings and coloured by category. Right: recommendations (nearest embeddings) and direct co-purchases of the selected product.")}
{code('''
cd examples/02-product-recommender
FGML_MODE=offline uvicorn backend.app:app --port 8000    # without FGML_MODE: ArangoDB
cd frontend && npm install && npm run dev                # second terminal, port 5173
''', "bash")}
<h3>Architecture</h3>
{fig(diagrams.recommender(), "Training writes to the database, the API only reads from it. After a restart the app works without retraining.")}
<h3>The training job</h3>
<p><code>backend/pipeline.py</code> is the only file that uses fastgraphml. Shortened:</p>
{code('''
model = SAGE(db, "product_graph", METAGRAPH, embedding_size=64, sizes=[10, 10],
             batch_size=512, transform=T.NormalizeFeatures())
model._train(epochs=10, lr=0.005, ckp_path=None, best_model_path=None)

emb = model.get_embeddings()
val_acc, test_acc = model.val()          # about 0.70 after ten epochs, ten categories
emb = whiten(emb)                        # PCA whitening, 32 dimensions

model.graph_util.store_embeddings(emb, collection_name="product_embeddings",
                                  class_mapping=CATEGORIES)
''')}
<p>fastgraphml prints its progress. To drive the progress bar of the web page, the job
replaces <code>sys.stdout</code> during training with a small object that passes the text on
and reads the epoch number from it.</p>
<h3>The recommendation query</h3>
{code('''
LET q = DOCUMENT(CONCAT(@emb, "/", @key)).embedding
LET neighbours = (FOR v IN 1..1 ANY @start @@edges RETURN DISTINCT v._key)
FOR e IN @@emb
    FILTER e._key != @key
    LET score = COSINE_SIMILARITY(e.embedding, q)
    SORT score DESC
    LIMIT @k
    RETURN { key: e._key, label: e.label, score: score, bought_together: e._key IN neighbours }
''', "sql")}
<p>One query ranks all products by similarity and marks those that are also direct
neighbours; the page tags the others as new finds.</p>
<h3>What building it showed</h3>
<ul>
<li>Without feature normalisation the training diverged; with it the accuracy was 68 to 72%
after ten epochs.</li>
<li>The raw embeddings were nearly one-dimensional (93% of the variance in one direction), so
all similarities were close to 1. PCA whitening spread them out and raised the share of
same-category products among the twelve nearest neighbours from 72% to 76%.</li>
<li>Direct neighbours are often not the nearest vectors. A production recommender would blend
both signals.</li>
</ul>
</section>
""")

    parts.append(f"""
<section>
{h2(9)}
<h3>Use case</h3>
<p>Feasibility teams choose investigator sites for a new clinical trial. Sites that already
ran trials in the same indication are easy to find with a query, and they are the ones
everybody asks. The interesting candidates are sites with no trial in the indication whose
history is otherwise close to the experienced ones. A query on experience cannot find them; a
similarity search on embeddings can. All data in this example is synthetic.</p>
{image("clinical-site-finder.png", "The site finder. The sentence at the top is the query. Left list: sites with experience. Right list: lookalikes without any trial in the indication. The map shows all sites by embedding, coloured by therapeutic area.")}
{code('''
cd examples/03-clinical-site-finder
FGML_MODE=offline uvicorn backend.app:app --port 8001    # without FGML_MODE: ArangoDB
cd frontend && npm install && npm run dev                # second terminal, port 5174
''', "bash")}
<h3>The graph</h3>
{fig(diagrams.clinical_schema(), "Four vertex collections, three edge collections. The metagraph leaves the sponsors out.")}
<h3>The model</h3>
<p>Sites have hardly any features, and what matters is two steps away: a site runs trials,
trials study indications. METAPATH2VEC walks exactly that path.</p>
{fig(diagrams.metapath("site", "trial", "indication", ("conducts", "studies", "rev_studies", "rev_conducts"), "Two sites that run trials in the same indications share many walks."), "The metapath of the site finder. The rev_ edge types are reversed copies added by a transform.")}
{code('''
METAGRAPH = {
    "vertexCollections": {"site": {"x": "features", "y": "area"}, "trial": {}, "indication": {}},
    "edgeCollections": {"conducts": {}, "studies": {}},
}
METAPATH = [("site", "conducts", "trial"), ("trial", "studies", "indication"),
            ("indication", "rev_studies", "trial"), ("trial", "rev_conducts", "site")]

model = METAPATH2VEC(db, "clinical_trials", METAGRAPH, metapaths=METAPATH, key_node="site",
                     embedding_size=64, walk_length=20, context_size=7, walks_per_node=10,
                     num_negative_samples=5, sparse=True,
                     transform=T.ToUndirected(),      # adds the rev_ edge types
                     num_nodes_dict=counts)           # documents per collection
model._train(epochs=15, lr=0.03, ckp_path=None, best_model_path=None)

emb = model.get_embeddings()["site"]
model.graph_util.store_embeddings(emb, collection_name="site_embeddings",
                                  node_type="site", class_mapping=AREAS)
''')}
<p>The label <code>y</code> is the primary therapeutic area of a site. It is used for
validation only: a classifier on the embeddings predicts it with 96 to 98% accuracy on
held-out sites, and the six areas are the six clusters on the map.</p>
<h3>Answering the question</h3>
{fig(diagrams.recommend_flow(), "GET /api/recommend: a graph traversal finds the experienced sites, their embeddings define what to look for, a similarity ranking finds the rest.")}
{code('''
FOR t IN 1..1 INBOUND @indication @@studies          -- step 1: who has experience?
    FOR s IN 1..1 INBOUND t @@conducts
        COLLECT site = s._key WITH COUNT INTO n
        RETURN [site, n]
''', "sql")}
{code('''
centroid = sum(experience[key] * vector / norm(vector)   # step 2: what do they look like?
               for key, vector in vectors.items())
''')}
{code('''
FOR e IN @@emb                                           -- step 3: who else looks like that?
    LET s = DOCUMENT(CONCAT(@sites, "/", e._key))
    FILTER @country == null OR s.country == @country
    LET score = COSINE_SIMILARITY(e.embedding, @vector)
    SORT score DESC
    RETURN { site: s, score: score }
''', "sql")}
<h3>Does it work?</h3>
<p>On synthetic data the truth is known. For Heart Failure all ten lookalikes are cardiology
sites without a single Heart Failure trial. Over all 30 indications, at least nine of the ten
lookalikes belong to the therapeutic area of the indication in 24 cases; the weakest case is
five of ten. On real data the same check needs a holdout: hide some sites' trials in an
indication, train, and see where those sites rank.</p>
</section>
""")

    parts.append(f"""
<section>
{h2(10)}
{table(["What you see", "Why", "What to do"], [
    ["Warning: neighbourhood sampling needs pyg-lib", "Neither pyg-lib nor torch-sparse is installed", "Fine for small graphs. Otherwise install pyg-lib from data.pyg.org for your PyTorch version"],
    ["pip compiles torch-scatter or torch-sparse for minutes, or fails", "No prebuilt wheel for your PyTorch version or platform", "Use a PyTorch version listed on data.pyg.org/whl, or install without them: <code>pip install -e \".[search,viz]\"</code>, <code>pip install --no-deps adbpyg-adapter arango-datasets</code>, <code>pip install pandas \"python-arango~=7.6\"</code>"],
    ["pip reports a conflict about python-arango", "adbpyg-adapter 1.1.4 requires python-arango 7.x", "Keep python-arango below 8 in environments that use the adapter"],
    ["ValueError: SAGE needs node features (x)", "The metagraph has no <code>x</code>, or the attribute is missing", "Add features or use METAPATH2VEC"],
    ["ImportError mentioning fastgraphml[arango], [search] or [viz]", "An optional package is missing", "Install the extra named in the message"],
    ["Loss is nan, or accuracy near zero", "Unscaled features with a high learning rate", "<code>transform=T.NormalizeFeatures()</code>, lower <code>lr</code>"],
    ["All cosine similarities are close to 1", "Embeddings dominated by one direction", "Train longer, or whiten the embeddings (example 2)"],
    ["RuntimeError: the ArangoDB keys ... are unknown", "The adapter did not return document keys", "Upgrade adbpyg-adapter to 1.1 or newer"],
    ["The web app says it cannot connect to ArangoDB", "Wrong <code>examples/.env</code>, or the database is down", "Check the four ARANGO_ variables, or start with <code>FGML_MODE=offline</code>"],
    ["The web page is empty at port 8000 or 8001", "The frontend was not built", "Run <code>npm run dev</code> and open port 5173 or 5174, or run <code>npm run build</code>"],
], ["30%", "30%", "40%"])}
<h3>Where to go next</h3>
<ul>
<li>The README of each example has the full API, file-by-file explanations and limits.</li>
<li>The two notebooks in <code>examples/</code> walk through the Amazon Computers and the DBLP
graph cell by cell.</li>
<li>For how the four models work inside, read the companion guide, fastgraphml explained.</li>
<li>For large embedding collections, look at ArangoDB's vector index (version 3.12.4 and
newer), which replaces the full scan of the similarity queries shown here. The examples do not
use it.</li>
</ul>
</section>
""")
    return "".join(parts)
