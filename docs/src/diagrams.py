"""The diagrams of the two guides, drawn with common.Diagram."""

from __future__ import annotations

import math
from typing import List, Tuple

from common import (
    AMBER,
    AMBER_SOFT,
    BLUE,
    BLUE_SOFT,
    GREEN,
    GREEN_SOFT,
    GREY_SOFT,
    INK,
    LINE,
    MUTED,
    ROSE,
    ROSE_SOFT,
    Diagram,
)


def big_picture() -> str:
    d = Diagram(680, 240)
    d.box(
        "adb",
        6,
        26,
        104,
        64,
        "ArangoDB",
        "vertex and edge\ncollections",
        "green",
        db=True,
    )
    d.box(
        "adapter",
        142,
        28,
        118,
        60,
        "adbpyg-adapter",
        "reads what the\nmetagraph names",
        "soft",
    )
    d.box("pyg", 292, 28, 104, 60, "PyG graph", "x, y, edge_index", "blue")
    d.box("model", 428, 28, 118, 60, "Model", "SAGE, GAT,\nMETAPATH2VEC, DMGI", "blue")
    d.box("emb", 578, 28, 96, 60, "Embeddings", "one vector\nper node", "amber")
    d.arrow("adb", "adapter")
    d.arrow("adapter", "pyg")
    d.arrow("pyg", "model")
    d.arrow("model", "emb", "_train()")
    d.box("own", 280, 150, 128, 50, "Your own PyG graph", "pyg_graph=data", "plain")
    d.arrow("own", "pyg", sides="tb")
    d.box(
        "store", 420, 150, 84, 62, "ArangoDB", "store_\nembeddings()", "green", db=True
    )
    d.box("search", 514, 152, 76, 58, "Search", "similarity_\nsearch()", "plain")
    d.box("viz", 600, 152, 74, 58, "Map", "visualize_\nembeddings()", "plain")
    for target in ("store", "search", "viz"):
        d.arrow("emb", target, sides="bt")
    d.text(6, 14, "1  Get the graph in", 9.5, BLUE, weight=600)
    d.text(428, 14, "2  Train", 9.5, BLUE, weight=600)
    d.text(420, 232, "3  Use the embeddings", 9.5, BLUE, weight=600)
    return d.svg()


def _cluster(
    d: Diagram, cx: float, cy: float, color: str, labels: str, r: float = 30
) -> List[Tuple[float, float]]:
    pts = []
    for i, label in enumerate(labels):
        angle = 2 * math.pi * i / len(labels) + 0.5
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (i + j) % 2 or j == i + 1:
                d.edge(*pts[i], *pts[j], color=MUTED, width=0.9)
    for (x, y), label in zip(pts, labels):
        d.circle(x, y, 8.5, color, label=label)
    return pts


def embedding_idea() -> str:
    d = Diagram(680, 196)
    d.text(6, 14, "A graph", 9.5, BLUE, weight=600)
    left = _cluster(d, 60, 100, GREEN, "ABC")
    right = _cluster(d, 168, 110, AMBER, "DEFG", r=32)
    d.edge(*left[0], *right[2], color=MUTED, width=0.9)
    for pts, color, labels in ((left, GREEN, "ABC"), (right, AMBER, "DEFG")):
        for (x, y), label in zip(pts, labels):
            d.circle(x, y, 8.5, color, label=label)
    d.line(226, 100, 262, 100, "model")

    d.text(272, 14, "One vector per node", 9.5, BLUE, weight=600)
    rows = [
        ("A", " 0.81  0.12 -0.33 ...", GREEN),
        ("B", " 0.77  0.09 -0.41 ...", GREEN),
        ("C", " 0.84  0.20 -0.29 ...", GREEN),
        ("D", "-0.42  0.66  0.51 ...", AMBER),
        ("E", "-0.38  0.71  0.47 ...", AMBER),
        ("F", "-0.45  0.59  0.55 ...", AMBER),
        ("G", "-0.36  0.73  0.44 ...", AMBER),
    ]
    for i, (name, vector, color) in enumerate(rows):
        y = 36 + i * 20
        d.circle(282, y, 7, color, label=name)
        d.text(296, y + 3, vector, 8.6, INK, mono=True)
    d.line(432, 100, 468, 100, "2D map")

    d.text(480, 14, "Similar neighbourhoods, close vectors", 9.5, BLUE, weight=600)
    d.rect(480, 26, 194, 150, fill=GREY_SOFT, stroke=LINE)
    for (x, y), label in zip([(520, 130), (536, 146), (510, 150)], "ABC"):
        d.circle(x, y, 8.5, GREEN, label=label)
    for (x, y), label in zip([(620, 60), (640, 74), (612, 82), (634, 50)], "DEFG"):
        d.circle(x, y, 8.5, AMBER, label=label)
    d.text(
        6,
        190,
        "Nodes that are connected to similar nodes get similar vectors, "
        "whether or not they are connected to each other.",
        8.8,
        MUTED,
    )
    return d.svg()


def metagraph() -> str:
    d = Diagram(680, 210)
    d.text(6, 14, "Documents in ArangoDB", 9.5, GREEN, weight=600)
    d.rect(6, 24, 222, 96, fill=GREEN_SOFT, stroke=GREEN)
    d.text(
        14,
        40,
        'Computer_Products/8515\n{ "features": [0, 1, 0, ...],\n  "label": 4 }',
        8.3,
        INK,
        mono=True,
    )
    d.rect(6, 130, 222, 62, fill=GREEN_SOFT, stroke=GREEN)
    d.text(
        14,
        146,
        'bought_together/...\n{ "_from": "Computer_Products/8515",\n'
        '  "_to": "Computer_Products/970" }',
        8.3,
        INK,
        mono=True,
    )

    d.text(262, 14, "The metagraph", 9.5, BLUE, weight=600)
    d.rect(262, 24, 196, 168, fill=BLUE_SOFT, stroke=BLUE)
    d.text(
        270,
        42,
        '{\n "vertexCollections": {\n  "Computer_Products": {\n   "x": "features",\n'
        '   "y": "label"\n  }\n },\n "edgeCollections": {\n  "bought_together": {}\n }\n}',
        8.3,
        INK,
        mono=True,
    )

    d.text(492, 14, "The PyG graph", 9.5, AMBER, weight=600)
    for i, (name, shape, what) in enumerate(
        [
            ("x", "[13752, 767]", "node features"),
            ("y", "[13752]", "labels, for validation"),
            ("edge_index", "[2, 491722]", "who is connected"),
        ]
    ):
        y = 24 + i * 58
        d.rect(492, y, 182, 50, fill=AMBER_SOFT, stroke=AMBER)
        d.text(500, y + 20, name, 10, INK, weight=600, mono=True)
        d.text(666, y + 20, shape, 8.3, MUTED, anchor="end", mono=True)
        d.text(500, y + 38, what, 8.8, MUTED)
    d.line(228, 60, 262, 76, head=False, color=MUTED)
    d.line(228, 160, 262, 150, head=False, color=MUTED)
    d.line(458, 76, 492, 49, color=MUTED)
    d.line(458, 88, 492, 107, color=MUTED)
    d.line(458, 150, 492, 165, color=MUTED)
    return d.svg()


def class_anatomy() -> str:
    d = Diagram(680, 330)
    d.text(
        6,
        14,
        "model = SAGE(db, graph, metagraph, ...)",
        9.5,
        BLUE,
        weight=600,
        mono=True,
    )
    steps = [
        ("check", "Check the arguments", "database or pyg_graph, not both"),
        (
            "export",
            "Export the graph",
            "adapter: ArangoDB to PyG, _key of every node kept",
        ),
        (
            "split",
            "Split the nodes",
            "train / val / test masks, only if there are labels",
        ),
        ("transform", "Apply your transform", "e.g. NormalizeFeatures, ToUndirected"),
        (
            "stats",
            "Print the graph statistics",
            "nodes, edges, classes, isolated nodes",
        ),
        ("build", "Build layers and loader", "the model is now ready to train"),
    ]
    for i, (key, title, sub) in enumerate(steps):
        d.box(key, 6, 26 + i * 50, 300, 38, title, sub, "soft" if i < 5 else "blue")
        if i:
            d.arrow(steps[i - 1][0], key, sides="bt")

    d.text(352, 14, "model._train(epochs=10)", 9.5, BLUE, weight=600, mono=True)
    d.rect(352, 26, 322, 216, stroke=BLUE, dashed=True)
    d.text(362, 42, "repeated for every epoch", 8.6, BLUE)
    loop = [
        (
            "batch",
            "Sample a mini-batch",
            "positive and negative node pairs + neighbourhoods",
        ),
        ("encode", "Encode", "the GNN turns features into embeddings"),
        (
            "loss",
            "Loss and optimizer step",
            "pull connected pairs together, push random pairs apart",
        ),
        (
            "val",
            "Validate",
            "logistic regression on the embeddings, if there are labels",
        ),
    ]
    for i, (key, title, sub) in enumerate(loop):
        d.box(key, 364, 50 + i * 48, 298, 36, title, sub, "plain")
        if i:
            d.arrow(loop[i - 1][0], key, sides="bt")
    d.box(
        "ckp",
        364,
        258,
        298,
        38,
        "Checkpoint",
        "latest model every epoch, best model on improvement",
        "amber",
    )
    d.arrow("val", "ckp", sides="bt")
    d.arrow("build", "batch", sides="rl", color=BLUE)
    d.text(352, 312, "Afterwards:", 8.8, MUTED)
    d.text(
        410,
        312,
        "model.get_embeddings()   model.val()\nmodel.graph_util.store_embeddings(...)",
        8,
        INK,
        mono=True,
    )
    return d.svg()


def sage() -> str:
    d = Diagram(680, 214)
    d.text(6, 14, "Neighbourhood sampling: sizes=[3, 2]", 9.5, BLUE, weight=600)
    cx, cy = 150, 104
    first = [
        (cx + 54 * math.cos(a), cy + 54 * math.sin(a))
        for a in (-2.2, -0.5, 1.2, 2.6, 0.35)
    ]
    sampled = [0, 1, 2]
    for i, (x, y) in enumerate(first):
        on = i in sampled
        d.edge(
            cx,
            cy,
            x,
            y,
            color=BLUE if on else LINE,
            width=1.4 if on else 1,
            dashed=not on,
        )
        if on:
            for k in (-0.5, 0.5):
                angle = math.atan2(y - cy, x - cx) + k
                x2, y2 = x + 30 * math.cos(angle), y + 30 * math.sin(angle)
                d.edge(x, y, x2, y2, color=BLUE, width=1)
                d.circle(x2, y2, 6, BLUE_SOFT, BLUE)
    for i, (x, y) in enumerate(first):
        on = i in sampled
        d.circle(x, y, 8, BLUE if on else "#ffffff", BLUE if on else LINE)
    d.circle(cx, cy, 11, INK, label="v")
    d.text(6, 208, "Dashed: neighbours that were not drawn this time", 8.6, MUTED)

    d.text(330, 14, "What a layer does", 9.5, BLUE, weight=600)
    d.box("self", 330, 30, 146, 40, "own vector", "of node v", "plain")
    d.box("mean", 330, 86, 146, 40, "mean", "of the sampled neighbours", "plain")
    d.box("new", 516, 56, 150, 46, "new vector of v", "W1 * own + W2 * mean", "blue")
    d.arrow("self", "new")
    d.arrow("mean", "new")

    d.text(330, 152, "What the loss asks for", 9.5, BLUE, weight=600)
    d.circle(344, 178, 8, INK, label="v")
    d.circle(392, 178, 8, GREEN, label="u")
    d.edge(352, 178, 384, 178, color=GREEN, width=1.6)
    d.text(408, 175, "connected pair:", 8.8, INK, weight=600)
    d.text(408, 187, "vectors should agree", 8.8, MUTED)
    d.circle(530, 178, 8, INK, label="v")
    d.circle(578, 178, 8, ROSE, label="w")
    d.edge(538, 178, 570, 178, color=ROSE, width=1.2, dashed=True)
    d.text(594, 175, "random pair:", 8.8, INK, weight=600)
    d.text(594, 187, "should not", 8.8, MUTED)
    return d.svg()


def metapath(
    a: str = "author",
    b: str = "paper",
    c: str = "conference",
    rel: Tuple[str, str, str, str] = (
        "writes",
        "published at",
        "publishes",
        "written by",
    ),
    note: str = "Two authors who publish at the same conferences share many walks.",
) -> str:
    d = Diagram(680, 210)
    d.text(6, 14, "A heterogeneous graph", 9.5, BLUE, weight=600)
    aa = [(40, 60), (40, 120), (40, 180)]
    bb = [(140, 44), (140, 96), (140, 150), (140, 194)]
    cc = [(240, 80), (240, 160)]
    edges_ab = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
    edges_bc = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)]
    walk_ab, walk_bc = {(0, 1), (1, 2)}, {(1, 0), (2, 0)}
    for i, j in edges_ab:
        on = (i, j) in walk_ab
        d.edge(*aa[i], *bb[j], color=ROSE if on else LINE, width=2 if on else 1)
    for i, j in edges_bc:
        on = (i, j) in walk_bc
        d.edge(*bb[i], *cc[j], color=ROSE if on else LINE, width=2 if on else 1)
    for x, y in aa:
        d.circle(x, y, 9, GREEN)
    for x, y in bb:
        d.circle(x, y, 7, "#ffffff", INK)
    for x, y in cc:
        d.circle(x, y, 9, AMBER)
    d.text(40, 36, a, 8.8, GREEN, anchor="middle", weight=600)
    d.text(140, 26, b, 8.8, INK, anchor="middle", weight=600)
    d.text(240, 56, c, 8.8, AMBER, anchor="middle", weight=600)

    d.text(
        310,
        14,
        "The metapath: the pattern every random walk follows",
        9.5,
        BLUE,
        weight=600,
    )
    xs = [326, 406, 496, 586, 666]
    names = [a, b, c, b, a]
    fills = [GREEN, "#ffffff", AMBER, "#ffffff", GREEN]
    for i in range(4):
        d.line(xs[i] + 10, 70, xs[i + 1] - 10, 70, rel[i], color=ROSE, label_dy=-8)
    for x, name, fill in zip(xs, names, fills):
        d.circle(x, 70, 9, fill, INK if fill == "#ffffff" else "none")
        d.text(x, 94, name, 8.4, MUTED, anchor="middle")
    d.text(
        310,
        128,
        "Skip-gram training, as in word2vec: nodes that appear within",
        8.8,
        INK,
    )
    d.text(
        310,
        141,
        "context_size steps of each other on a walk get similar vectors.",
        8.8,
        INK,
    )
    d.text(310, 166, note, 8.8, MUTED)
    d.text(310, 190, "Red: one walk through the graph on the left.", 8.8, ROSE)
    return d.svg()


def dmgi() -> str:
    d = Diagram(680, 190)
    d.box("g", 6, 62, 120, 60, "Hetero graph", "key node: movie", "soft")
    d.box(
        "m0",
        160,
        20,
        150,
        50,
        "movie - actor - movie",
        "AddMetaPaths: metapath_0",
        "plain",
    )
    d.box(
        "m1",
        160,
        114,
        150,
        50,
        "movie - director - movie",
        "AddMetaPaths: metapath_1",
        "plain",
    )
    d.box("g0", 344, 20, 110, 50, "GCN 0", "embedding per\nrelation", "blue")
    d.box("g1", 344, 114, 110, 50, "GCN 1", "embedding per\nrelation", "blue")
    d.box(
        "z",
        494,
        62,
        180,
        60,
        "Consensus embedding Z",
        "one vector per key node,\nclose to all relation embeddings",
        "amber",
    )
    d.arrow("g", "m0")
    d.arrow("g", "m1")
    d.arrow("m0", "g0")
    d.arrow("m1", "g1")
    d.arrow("g0", "z")
    d.arrow("g1", "z")
    d.text(
        344,
        184,
        "Each GCN is trained with Deep Graph Infomax: tell real node features from shuffled ones.",
        8.6,
        MUTED,
        anchor="middle",
    )
    return d.svg()


def store() -> str:
    d = Diagram(680, 176)
    d.text(6, 14, "In memory", 9.5, BLUE, weight=600)
    d.rect(6, 24, 196, 60, fill=AMBER_SOFT, stroke=AMBER)
    d.text(14, 42, "embeddings[i]", 9, INK, weight=600, mono=True)
    d.text(14, 58, "row i of the array", 8.8, MUTED)
    d.text(14, 72, "[0.113, -0.482, 0.071, ...]", 8.3, INK, mono=True)
    d.rect(6, 98, 196, 60, fill=BLUE_SOFT, stroke=BLUE)
    d.text(14, 116, "graph_util.node_keys[i]", 9, INK, weight=600, mono=True)
    d.text(14, 132, "the _key the adapter kept", 8.8, MUTED)
    d.text(14, 146, '"8515"', 8.3, INK, mono=True)

    d.text(250, 14, "Embedding collection", 9.5, GREEN, weight=600)
    d.rect(250, 24, 220, 134, fill=GREEN_SOFT, stroke=GREEN)
    d.text(
        258,
        42,
        'product_embeddings/8515\n{\n "_key": "8515",\n "source_id":\n   "Computer_Products/8515",\n'
        ' "embedding": [0.113, ...],\n "label": 4,\n "class_name": "Computer Components"\n}',
        8.1,
        INK,
        mono=True,
    )
    d.text(516, 14, "Source vertex", 9.5, GREEN, weight=600)
    d.rect(516, 24, 158, 76, fill=GREEN_SOFT, stroke=GREEN)
    d.text(
        524,
        42,
        'Computer_Products/8515\n{\n "features": [...],\n "label": 4\n}',
        8.1,
        INK,
        mono=True,
    )
    d.line(202, 54, 250, 96, color=MUTED)
    d.line(202, 128, 250, 60, color=MUTED)
    d.line(470, 50, 516, 50, "same _key", color=GREEN, label_dy=-6)
    d.text(
        516,
        124,
        "DOCUMENT(e.source_id) joins an\nembedding with its vertex in AQL.",
        8.8,
        MUTED,
    )
    return d.svg()


def stack() -> str:
    d = Diagram(680, 150)
    d.box(
        "app",
        6,
        6,
        440,
        30,
        "Your code, a notebook or one of the example projects",
        "",
        "plain",
    )
    d.box("fg", 6, 42, 440, 30, "fastgraphml", "", "blue")
    d.box("pyg", 6, 78, 300, 30, "torch-geometric", "", "soft")
    d.box("lib", 312, 78, 134, 30, "pyg-lib (optional)", "", "soft")
    d.box("torch", 6, 114, 440, 30, "PyTorch", "", "soft")
    d.box(
        "ar",
        470,
        6,
        204,
        44,
        "extra: arango",
        "adbpyg-adapter, arango-datasets",
        "green",
    )
    d.box("se", 470, 56, 204, 40, "extra: search", "faiss-cpu", "amber")
    d.box(
        "vi",
        470,
        102,
        204,
        42,
        "extra: viz",
        "umap-learn, matplotlib, seaborn",
        "amber",
    )
    return d.svg()


def examples_flow() -> str:
    d = Diagram(680, 150)
    d.box("adb", 6, 30, 104, 64, "ArangoDB", "graph", "green", db=True)
    d.box("model", 150, 32, 118, 60, "fastgraphml", "model + _train()", "blue")
    d.box(
        "emb", 308, 30, 116, 64, "ArangoDB", "embedding\ncollection", "green", db=True
    )
    d.box("api", 464, 32, 100, 60, "CLI or REST API", "AQL queries", "plain")
    d.box("ui", 596, 32, 78, 60, "Terminal", "or React app", "plain")
    d.arrow("adb", "model", "metagraph", label_dy=-8)
    d.arrow("model", "emb", "store", label_dy=-8)
    d.arrow("emb", "api", "similarity", label_dy=-8)
    d.arrow("api", "ui")
    d.line(58, 94, 58, 124, head=False)
    d.line(58, 124, 514, 124, head=False)
    d.line(514, 124, 514, 92)
    d.text(
        286,
        138,
        "traversals: neighbours, trial histories, counts",
        8.4,
        MUTED,
        anchor="middle",
    )
    d.text(
        6,
        14,
        "Offline mode replaces both databases with a PyG dataset and numpy arrays.",
        8.8,
        MUTED,
    )
    return d.svg()


def cli() -> str:
    d = Diagram(680, 196)
    d.box("cfg", 6, 70, 96, 50, "config.json", "source, model,\ntrain, store", "amber")
    d.box("adb", 136, 16, 130, 46, "source: arangodb", "database + metagraph", "green")
    d.box("pyg", 136, 128, 130, 46, "source: pyg", "torch_geometric dataset", "plain")
    d.box(
        "build",
        300,
        70,
        110,
        50,
        "build_model()",
        "SAGE, GAT,\nMETAPATH2VEC, DMGI",
        "blue",
    )
    d.box(
        "train",
        440,
        70,
        100,
        50,
        "_train() + val()",
        "epochs, lr from\nthe train section",
        "blue",
    )
    d.box(
        "run",
        574,
        12,
        100,
        62,
        "runs/NAME/",
        "embeddings.npz\nkeys.json, run.json",
        "amber",
    )
    d.box("db", 574, 116, 100, 66, "ArangoDB", "store_\nembeddings()", "green", db=True)
    d.arrow("cfg", "adb")
    d.arrow("cfg", "pyg")
    d.arrow("adb", "build")
    d.arrow("pyg", "build")
    d.arrow("build", "train")
    d.arrow("train", "run")
    d.arrow("train", "db")
    d.text(574, 96, "similar, export", 8.4, MUTED)
    d.text(574, 194, "similar --db", 8.4, MUTED)
    return d.svg()


def recommender() -> str:
    d = Diagram(680, 200)
    d.rect(6, 24, 150, 160, stroke=LINE, dashed=True)
    d.text(14, 40, "Browser", 9, MUTED, weight=600)
    d.box(
        "ui",
        20,
        66,
        122,
        80,
        "React app",
        "catalogue\nembedding map\nresult lists",
        "plain",
    )
    d.rect(190, 24, 280, 160, stroke=LINE, dashed=True)
    d.text(198, 40, "FastAPI backend", 9, MUTED, weight=600)
    d.box("api", 204, 80, 96, 52, "app.py", "REST endpoints", "plain")
    d.box("job", 330, 50, 126, 46, "pipeline.py", "training job, fastgraphml", "blue")
    d.box("st", 330, 120, 126, 46, "store.py", "AQL reads", "plain")
    d.box(
        "db",
        520,
        44,
        154,
        128,
        "ArangoDB",
        "Computer_Products\nbought_together\nproduct_embeddings",
        "green",
        db=True,
    )
    d.arrow("ui", "api", "/api")
    d.arrow("api", "job", sides="rl")
    d.arrow("api", "st", sides="rl")
    d.line(456, 73, 520, 73, "read, store", color=BLUE, label_dy=-6)
    d.line(456, 143, 520, 143, "AQL", label_dy=-6)
    return d.svg()


def clinical_schema() -> str:
    d = Diagram(680, 150)
    d.box("sp", 6, 46, 110, 54, "sponsor", "30 documents", "soft")
    d.box("tr", 200, 46, 120, 54, "trial", "1,400 documents\nphase, year", "plain")
    d.box("site", 404, 6, 130, 54, "site", "480 documents\nfeatures, area", "green")
    d.box("ind", 404, 90, 130, 54, "indication", "30 documents\nin 6 areas", "amber")
    d.arrow("sp", "tr", "funds")
    d.arrow("site", "tr", "conducts", sides="lr", label_dy=-22, label_dx=18)
    d.arrow("tr", "ind", "studies", sides="rl", label_dy=24, label_dx=-18)
    d.rect(190, 0, 354, 150, stroke=BLUE, dashed=True)
    d.text(
        556,
        70,
        "Inside the frame: the part\nthe metagraph names, i.e.\nwhat the model learns from.",
        8.8,
        BLUE,
    )
    return d.svg()


def recommend_flow() -> str:
    d = Diagram(680, 176)
    d.box("q", 6, 60, 96, 50, "Question", "indication,\ncountry", "amber")
    d.box(
        "exp",
        132,
        60,
        124,
        50,
        "1  Experience",
        "traversal: sites with\ntrials in the indication",
        "green",
    )
    d.box(
        "vec",
        286,
        60,
        110,
        50,
        "2  Centroid",
        "weighted average of\ntheir embeddings",
        "blue",
    )
    d.box(
        "rank", 426, 60, 104, 50, "3  Rank", "cosine similarity\nto every site", "blue"
    )
    d.box(
        "e",
        560,
        10,
        114,
        56,
        "Experienced sites",
        "in step 1,\nordered by trials",
        "green",
    )
    d.box(
        "l",
        560,
        106,
        114,
        60,
        "Lookalike sites",
        "not in step 1,\nordered by similarity",
        "rose",
    )
    d.arrow("q", "exp")
    d.arrow("exp", "vec")
    d.arrow("vec", "rank")
    d.arrow("rank", "e")
    d.arrow("rank", "l")
    d.text(132, 130, "AQL on the graph", 8.4, MUTED)
    d.text(286, 130, "numpy", 8.4, MUTED)
    d.text(426, 130, "AQL on the embeddings", 8.4, MUTED)
    return d.svg()
