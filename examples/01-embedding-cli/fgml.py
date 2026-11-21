#!/usr/bin/env python
"""fgml - a command line pipeline around fastgraphml.

One JSON config describes a whole embedding job: where the graph comes from
(ArangoDB or a PyG dataset), which model to train, and where the embeddings go.

    python fgml.py check                         # environment and database
    python fgml.py inspect                       # what is in the database?
    python fgml.py run configs/cora_offline.json # train + evaluate + save (+ store)
    python fgml.py similar runs/cora 42          # nearest neighbours of node 42
    python fgml.py export runs/cora --format csv # embeddings for other tools

See README.md in this folder for the full walk-through.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Train, store and query graph embeddings with fastgraphml.",
)
console = Console()

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# configuration and connection
# --------------------------------------------------------------------------- #
def load_env() -> None:
    """Reads KEY=VALUE lines from the first .env found in the current folder,
    this folder or the examples folder. Existing environment variables win."""
    for folder in (Path.cwd(), HERE, HERE.parent):
        env_file = folder / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return


def connect() -> Any:
    """Connects to the ArangoDB database configured in the environment."""
    from arango import ArangoClient

    load_env()
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    url = os.environ.get("ARANGO_URL", "http://localhost:8529")
    name = os.environ.get("ARANGO_DB", "fastgraphml_examples")
    user = os.environ.get("ARANGO_USER", "root")
    password = os.environ.get("ARANGO_PASSWORD", "")
    try:
        return ArangoClient(hosts=url).db(name, user, password, verify=True)
    except Exception as e:
        console.print(f"[red]Could not connect to database '{name}' at {url}:[/] {e}")
        console.print("Check ARANGO_URL, ARANGO_DB, ARANGO_USER and ARANGO_PASSWORD.")
        raise typer.Exit(1)


def read_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        console.print(f"[red]No such config file:[/] {path}")
        raise typer.Exit(1)
    config: Dict[str, Any] = json.loads(path.read_text())
    for section in ("source", "model"):
        if section not in config:
            console.print(f"[red]The config needs a '{section}' section.[/]")
            raise typer.Exit(1)
    return config


# --------------------------------------------------------------------------- #
# the pipeline: source -> model -> embeddings
# --------------------------------------------------------------------------- #
def model_inputs(source: Dict[str, Any]) -> Dict[str, Any]:
    """Turns the 'source' section of a config into fastgraphml model arguments."""
    kind = source.get("type", "arangodb")
    if kind == "arangodb":
        return {
            "database": connect(),
            "arango_graph": source["graph"],
            "metagraph": source["metagraph"],
        }
    if kind == "pyg":
        import torch_geometric.datasets as pyg_datasets

        root = Path(source.get("root", HERE / ".data")) / source["dataset"]
        dataset = getattr(pyg_datasets, source["dataset"])(
            root=str(root), **source.get("args", {})
        )
        return {"pyg_graph": dataset[0]}
    console.print(f"[red]Unknown source type '{kind}'[/] (use 'arangodb' or 'pyg').")
    raise typer.Exit(1)


def build_model(config: Dict[str, Any]) -> Any:
    import fastgraphml

    name = config["model"]["name"].upper()
    if name not in ("SAGE", "GAT", "METAPATH2VEC", "DMGI"):
        console.print(f"[red]Unknown model '{name}'.[/]")
        raise typer.Exit(1)
    params = dict(config["model"].get("params", {}))
    if "transform" in params:
        # names of torch_geometric.transforms classes, e.g. ["NormalizeFeatures"]
        import torch_geometric.transforms as T

        params["transform"] = T.Compose([getattr(T, t)() for t in params["transform"]])
    if "metapaths" in params:
        # JSON has no tuples: METAPATH2VEC wants a list of (src, rel, dst) tuples,
        # DMGI a list of metapaths, each one a list of (src, dst) tuples.
        if name == "DMGI":
            params["metapaths"] = [[tuple(e) for e in mp] for mp in params["metapaths"]]
        else:
            params["metapaths"] = [tuple(e) for e in params["metapaths"]]
    return getattr(fastgraphml, name)(**model_inputs(config["source"]), **params)


def node_sets(model: Any, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Embeddings, keys and labels per node set.

    Homogeneous models return one array, heterogeneous models one per node type.
    The keys are the ArangoDB _key of the nodes if the graph came from ArangoDB,
    otherwise the PyG node index.
    """
    embeddings = model.get_embeddings()
    util = model.graph_util
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(embeddings, dict):
        for node_type, emb in embeddings.items():
            keys = util.node_keys.get(node_type) or [str(i) for i in range(len(emb))]
            y = getattr(model.G[node_type], "y", None)
            result[node_type] = {
                "emb": emb,
                "keys": [str(k) for k in keys],
                "labels": y.cpu().tolist() if y is not None else None,
                "node_type": node_type,
            }
    else:
        name = "nodes"
        if config["source"].get("type", "arangodb") == "arangodb":
            vertex_collections = list(
                config["source"]["metagraph"]["vertexCollections"]
            )
            if len(vertex_collections) == 1:
                name = vertex_collections[0]
        keys = util.node_keys.get(None) or [str(i) for i in range(len(embeddings))]
        y = getattr(model.G, "y", None)
        result[name] = {
            "emb": embeddings,
            "keys": [str(k) for k in keys],
            "labels": y.cpu().tolist() if y is not None else None,
            "node_type": None,
        }
    return result


def load_run(run_dir: Path, node_set: Optional[str]) -> Tuple[Dict[str, Any], str]:
    """Loads one node set of a finished run from disk."""
    meta_file = run_dir / "run.json"
    if not meta_file.is_file():
        console.print(f"[red]{run_dir} is not a run folder (no run.json).[/]")
        raise typer.Exit(1)
    meta = json.loads(meta_file.read_text())
    names = list(meta["node_sets"])
    if node_set is None:
        node_set = meta.get("key_node") if meta.get("key_node") in names else names[0]
    if node_set not in names:
        console.print(
            f"[red]No node set '{node_set}'[/] in this run, there is: {names}"
        )
        raise typer.Exit(1)
    with np.load(run_dir / "embeddings.npz") as data:
        emb = data[node_set]
    keys = json.loads((run_dir / "keys.json").read_text())[node_set]
    labels = json.loads((run_dir / "labels.json").read_text()).get(node_set)
    return {"emb": emb, "keys": keys, "labels": labels, "meta": meta}, node_set


def class_name(meta: Dict[str, Any], label: Optional[int]) -> str:
    if label is None:
        return ""
    return str(meta.get("class_names", {}).get(str(label), label))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def check() -> None:
    """Shows the installed versions and tests the database connection."""
    import importlib

    import torch
    import torch_geometric
    import torch_geometric.typing as pyg_typing

    import fastgraphml  # noqa: F401

    table = Table(title="Environment", show_header=False)
    table.add_row("torch", torch.__version__)
    table.add_row("torch-geometric", torch_geometric.__version__)
    sampler = pyg_typing.WITH_PYG_LIB or pyg_typing.WITH_TORCH_SPARSE
    table.add_row(
        "neighborhood sampling",
        "[green]available[/]" if sampler else "[yellow]missing: full graph training[/]",
    )
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    table.add_row("device", gpu)
    for module, purpose in (
        ("adbpyg_adapter", "reading ArangoDB graphs"),
        ("faiss", "similarity search"),
        ("umap", "visualisation"),
    ):
        found = importlib.util.find_spec(module) is not None
        state = "[green]installed[/]" if found else "[yellow]not installed[/]"
        table.add_row(f"{module} ({purpose})", state)
    console.print(table)

    db = connect()
    console.print(
        f"[green]Connected[/] to database '{db.name}', ArangoDB {db.version()}, "
        f"{len(db.graphs())} graph(s)."
    )


@app.command()
def inspect(
    graph: Optional[str] = typer.Option(None, help="Only show this graph."),
) -> None:
    """Lists the graphs of the database and proposes a metagraph for each.

    Attributes holding a list of numbers can be node features (x), attributes
    holding an integer can be labels (y).
    """
    db = connect()
    graphs = [g for g in db.graphs() if graph is None or g["name"] == graph]
    if not graphs:
        console.print("[yellow]No graph found.[/] Create one or run `load-demo`.")
        raise typer.Exit(1)

    for g in graphs:
        vertex_collections: List[str] = list(g["orphan_collections"])
        table = Table(title=f"Graph '{g['name']}': edges")
        for column in ("edge collection", "from", "to", "edges"):
            table.add_column(column)
        for definition in g["edge_definitions"]:
            froms = definition["from_vertex_collections"]
            tos = definition["to_vertex_collections"]
            for name in froms + tos:
                if name not in vertex_collections:
                    vertex_collections.append(name)
            count = db.collection(definition["edge_collection"]).count()
            table.add_row(
                definition["edge_collection"],
                ", ".join(froms),
                ", ".join(tos),
                str(count),
            )
        console.print(table)

        metagraph: Dict[str, Any] = {
            "vertexCollections": {},
            "edgeCollections": {
                d["edge_collection"]: {} for d in g["edge_definitions"]
            },
        }
        table = Table(title=f"Graph '{g['name']}': vertices")
        for column in ("vertex collection", "documents", "attribute", "looks like"):
            table.add_column(column)
        for name in vertex_collections:
            collection = db.collection(name)
            cursor = db.aql.execute(
                "FOR d IN @@c LIMIT 1 RETURN d", bind_vars={"@c": name}
            )
            sample = next(iter(cursor), {})
            mapping: Dict[str, str] = {}
            rows = []
            for attribute, value in sample.items():
                if attribute.startswith("_"):
                    continue
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(v, (int, float)) for v in value)
                ):
                    rows.append((attribute, f"features, {len(value)} numbers -> x"))
                    mapping.setdefault("x", attribute)
                elif isinstance(value, int) and not isinstance(value, bool):
                    rows.append((attribute, "integer -> y (if it is a class label)"))
                    mapping.setdefault("y", attribute)
                else:
                    rows.append((attribute, type(value).__name__))
            metagraph["vertexCollections"][name] = mapping
            count = str(collection.count())
            for i, (attribute, kind) in enumerate(rows or [("", "no attributes")]):
                table.add_row(
                    name if i == 0 else "", count if i == 0 else "", attribute, kind
                )
        console.print(table)
        console.print("Proposed metagraph (check the x and y guesses):")
        console.print_json(json.dumps(metagraph))


@app.command("load-demo")
def load_demo(
    dataset: str = typer.Argument(
        "AMAZON_COMPUTER_PRODUCTS", help="Name of an arango-datasets dataset."
    ),
) -> None:
    """Loads a demo dataset into the database (needs the arango-datasets package)."""
    from fastgraphml import Datasets

    db = connect()
    Datasets(db).load(dataset)
    console.print(f"[green]Loaded {dataset}.[/] Run `inspect` to see the graph.")


@app.command()
def run(
    config_file: Path = typer.Argument(..., help="JSON config of the job."),
    out: Optional[Path] = typer.Option(None, help="Run folder, default runs/<name>."),
    epochs: Optional[int] = typer.Option(None, help="Overrides train.epochs."),
    no_store: bool = typer.Option(False, help="Do not write to ArangoDB."),
) -> None:
    """Trains a model, evaluates it, saves the embeddings and stores them."""
    config = read_config(config_file)
    run_dir = out or HERE / "runs" / config.get("name", config_file.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = dict(config.get("train", {}))
    if epochs is not None:
        train["epochs"] = epochs

    console.rule(f"1/4 Loading the graph and building {config['model']['name']}")
    started = time.time()
    model = build_model(config)

    console.rule("2/4 Training")
    model._train(ckp_path=None, best_model_path=str(run_dir / "best_model.pt"), **train)
    metrics: Dict[str, Any] = {"train_seconds": round(time.time() - started, 1)}
    if getattr(model, "has_labels", False):
        val_acc, test_acc = model.val()
        metrics.update(val_accuracy=round(val_acc, 4), test_accuracy=round(test_acc, 4))

    console.rule("3/4 Saving the embeddings")
    sets = node_sets(model, config)
    np.savez_compressed(
        run_dir / "embeddings.npz", **{n: s["emb"] for n, s in sets.items()}
    )
    (run_dir / "keys.json").write_text(
        json.dumps({n: s["keys"] for n, s in sets.items()})
    )
    (run_dir / "labels.json").write_text(
        json.dumps({n: s["labels"] for n, s in sets.items()})
    )
    store = config.get("store")
    meta = {
        "name": config.get("name", config_file.stem),
        "config": config,
        "metrics": metrics,
        "key_node": config["model"].get("params", {}).get("key_node"),
        "class_names": (store or {}).get("class_names", config.get("class_names", {})),
        "node_sets": {
            n: {"nodes": int(s["emb"].shape[0]), "dimensions": int(s["emb"].shape[1])}
            for n, s in sets.items()
        },
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))

    console.rule("4/4 Storing in ArangoDB")
    is_arango = config["source"].get("type", "arangodb") == "arangodb"
    if store and is_arango and not no_store:
        node_type = store.get("node_type") or meta["key_node"]
        name = node_type if node_type in sets else next(iter(sets))
        class_names = {int(k): v for k, v in meta["class_names"].items()} or None
        neighbors = int(store.get("nearest_neighbors", 0))
        model.graph_util.store_embeddings(
            sets[name]["emb"],
            collection_name=store["collection"],
            class_mapping=class_names,
            node_type=sets[name]["node_type"],
            nearest_nbors_search=neighbors > 0,
            top_k_nbors=max(neighbors, 1),
        )
        console.print(
            f"Stored {len(sets[name]['keys'])} documents in '{store['collection']}'."
        )
    else:
        console.print("Skipped (offline source, no 'store' section or --no-store).")

    table = Table(title=f"Run saved in {run_dir}")
    table.add_column("node set")
    table.add_column("nodes", justify="right")
    table.add_column("dimensions", justify="right")
    for n, info in meta["node_sets"].items():
        table.add_row(n, str(info["nodes"]), str(info["dimensions"]))
    console.print(table)
    console.print(metrics)


@app.command()
def similar(
    run_dir: Path = typer.Argument(..., help="Run folder created by `run`."),
    key: str = typer.Argument(..., help="Key of the node to find neighbours for."),
    k: int = typer.Option(10, "-k", help="Number of neighbours."),
    node_set: Optional[str] = typer.Option(None, help="Node set of a hetero run."),
    db: bool = typer.Option(False, "--db", help="Query ArangoDB instead of the files."),
) -> None:
    """Shows the nodes with the most similar embeddings (cosine similarity)."""
    data, node_set = load_run(run_dir, node_set)
    meta = data["meta"]
    rows: List[Tuple[str, float, str]] = []

    if db:
        store = meta["config"].get("store")
        if not store:
            console.print(
                "[red]This run has no 'store' section, nothing is in ArangoDB.[/]"
            )
            raise typer.Exit(1)
        # brute force cosine similarity in AQL, works on every ArangoDB version;
        # see the README for the vector index variant for large collections
        query = """
        LET q = DOCUMENT(@@emb, @key).embedding
        FOR e IN @@emb
            FILTER e._key != @key
            LET score = COSINE_SIMILARITY(e.embedding, q)
            SORT score DESC
            LIMIT @k
            RETURN { key: e._key, score: score, label: e.label }
        """
        cursor = connect().aql.execute(
            query, bind_vars={"@emb": store["collection"], "key": key, "k": k}
        )
        rows = [
            (d["key"], d["score"], class_name(meta, d.get("label"))) for d in cursor
        ]
    else:
        if key not in data["keys"]:
            console.print(f"[red]No node with key '{key}'[/] in node set '{node_set}'.")
            raise typer.Exit(1)
        emb = data["emb"].astype(np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        index = data["keys"].index(key)
        scores = emb @ emb[index]
        scores[index] = -np.inf
        labels = data["labels"]
        for i in np.argsort(-scores)[:k]:
            label = labels[i] if labels else None
            rows.append((data["keys"][i], float(scores[i]), class_name(meta, label)))

    own_label = data["labels"][data["keys"].index(key)] if data["labels"] else None
    title = f"Most similar to {node_set}/{key}"
    if own_label is not None:
        title += f" ({class_name(meta, own_label)})"
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("key")
    table.add_column("cosine similarity", justify="right")
    table.add_column("class")
    for rank, (other, score, name) in enumerate(rows, start=1):
        table.add_row(str(rank), other, f"{score:.4f}", name)
    console.print(table)


@app.command()
def export(
    run_dir: Path = typer.Argument(..., help="Run folder created by `run`."),
    fmt: str = typer.Option("csv", "--format", help="csv or jsonl."),
    node_set: Optional[str] = typer.Option(None, help="Node set of a hetero run."),
    out: Optional[Path] = typer.Option(None, help="Output file."),
) -> None:
    """Writes the embeddings of a run to a CSV or JSON lines file."""
    data, node_set = load_run(run_dir, node_set)
    if fmt not in ("csv", "jsonl"):
        console.print("[red]--format must be csv or jsonl.[/]")
        raise typer.Exit(1)
    out = out or run_dir / f"{node_set}.{fmt}"
    labels = data["labels"]
    with out.open("w", newline="") as f:
        if fmt == "csv":
            writer = csv.writer(f)
            dims = data["emb"].shape[1]
            writer.writerow(["key", "label"] + [f"e{i}" for i in range(dims)])
            for i, key in enumerate(data["keys"]):
                label = labels[i] if labels else ""
                writer.writerow([key, label] + [f"{v:.6f}" for v in data["emb"][i]])
        else:
            for i, key in enumerate(data["keys"]):
                doc = {
                    "key": key,
                    "embedding": [round(float(v), 6) for v in data["emb"][i]],
                }
                if labels:
                    doc["label"] = labels[i]
                f.write(json.dumps(doc) + "\n")
    console.print(f"Wrote {len(data['keys'])} rows to {out}")


if __name__ == "__main__":
    app()
