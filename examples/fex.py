#!/usr/bin/env python
"""fex - one command for the fastgraphml example projects.

    fex prepare            create the environment, install everything, build the UIs
    fex status             what is installed, configured, loaded and running
    fex env                show the database settings; `fex env set KEY=VALUE` changes them
    fex db                 what the examples have in the database
    fex db init            create the database and user named in examples/.env
    fex db load ex2        load the data of an example (ex1, ex2, ex3 or all)
    fex db clean ex2       drop the collections of an example
    fex train ex3          train and store embeddings without the web page
    fex run ex2            run an example (ex1: the pipeline, ex2 and ex3: the web app)
    fex link               put `fex` on your PATH

This file only needs the Python standard library, so that `fex prepare` works in an
empty environment. Run it through the `fex` launcher in the repository root, which
picks the Python of the virtual environment once there is one.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parent
VENV = Path(os.environ.get("FEX_VENV", ROOT / ".venv"))
VENV_PYTHON = VENV / "bin" / "python"
ENV_FILE = EXAMPLES / ".env"

# What each example is, and which collections belong to it.
PROJECTS: Dict[str, Dict[str, Any]] = {
    "ex1": {
        "title": "Embedding pipeline CLI",
        "dir": EXAMPLES / "01-embedding-cli",
        "data": ["Computer_Products", "bought_together"],
        "embeddings": ["cli_product_embeddings"],
    },
    "ex2": {
        "title": "Product recommender",
        "dir": EXAMPLES / "02-product-recommender",
        "data": ["Computer_Products", "bought_together"],
        "embeddings": ["product_embeddings"],
        "port": 8000,
        "dev_port": 5173,
    },
    "ex3": {
        "title": "Clinical site finder",
        "dir": EXAMPLES / "03-clinical-site-finder",
        "data": [
            "site",
            "trial",
            "indication",
            "sponsor",
            "conducts",
            "studies",
            "funds",
        ],
        "embeddings": ["site_embeddings"],
        "port": 8001,
        "dev_port": 5174,
    },
}
ENV_DEFAULTS = {
    "ARANGO_URL": "http://localhost:8529",
    "ARANGO_DB": "fastgraphml_examples",
    "ARANGO_USER": "fastgraphml",
    "ARANGO_PASSWORD": "",
    "ARANGO_VERIFY": "",
}

# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def heading(text: str) -> None:
    print("\n" + paint(text, "1"))


def row(state: str, label: str, detail: str = "") -> None:
    """One status line. state: ok, warn, fail or info."""
    mark = {
        "ok": paint("ok  ", "32"),
        "warn": paint("warn", "33"),
        "fail": paint("FAIL", "31"),
        "info": paint("    ", "0"),
    }[state]
    print(f"  {mark}  {label:<28} {detail}")


def die(message: str) -> "None":
    print(paint("error: ", "31") + message, file=sys.stderr)
    sys.exit(1)


def sh(
    command: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None
) -> None:
    """Runs a command, shows it, and stops fex if it fails."""
    print(paint("  $ " + " ".join(str(c) for c in command), "2"))
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        die(f"command failed with exit code {result.returncode}")


# --------------------------------------------------------------------------- #
# settings (examples/.env)
# --------------------------------------------------------------------------- #
def read_env() -> Dict[str, str]:
    """Settings from examples/.env; real environment variables win, as in the examples."""
    values = dict(ENV_DEFAULTS)
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    for key in list(values):
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def write_env(changes: Dict[str, str]) -> None:
    """Sets KEY=VALUE lines in examples/.env, keeping comments and other lines."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.is_file() else []
    done = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip().lstrip("# ").strip()
        if "=" in line and key in changes:
            lines[i] = f"{key}={changes[key]}"
            done.add(key)
    lines += [f"{key}={value}" for key, value in changes.items() if key not in done]
    ENV_FILE.write_text("\n".join(lines) + "\n")


def masked(settings: Dict[str, str]) -> Dict[str, str]:
    shown = dict(settings)
    shown["ARANGO_PASSWORD"] = "(set)" if settings["ARANGO_PASSWORD"] else "(empty)"
    shown["ARANGO_VERIFY"] = (
        settings["ARANGO_VERIFY"] or "(default: check certificates)"
    )
    return shown


# --------------------------------------------------------------------------- #
# environment checks
# --------------------------------------------------------------------------- #
def in_venv() -> bool:
    return VENV_PYTHON.exists() and Path(sys.prefix).resolve() == VENV.resolve()


def venv_has(*modules: str) -> bool:
    """Can the virtual environment import all of these modules?"""
    if not VENV_PYTHON.exists():
        return False
    check = "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in sys.argv[1:]))"
    return subprocess.run([str(VENV_PYTHON), "-c", check, *modules]).returncode == 0


PROBE = """
import importlib.metadata as md, importlib.util as util, json, sys
packages = {"torch": "torch", "torch-geometric": "torch_geometric", "fastgraphml": "fastgraphml",
            "adbpyg-adapter": "adbpyg_adapter", "python-arango": "arango", "faiss-cpu": "faiss",
            "umap-learn": "umap", "fastapi": "fastapi", "typer": "typer"}
found = {}
for dist, module in packages.items():
    if util.find_spec(module) is not None:
        try:
            found[dist] = md.version(dist)
        except md.PackageNotFoundError:
            found[dist] = "installed"
sampling = False
if "torch-geometric" in found:
    import torch_geometric.typing as t
    sampling = bool(t.WITH_PYG_LIB or t.WITH_TORCH_SPARSE)
print(json.dumps({"python": sys.version.split()[0], "packages": found, "sampling": sampling}))
"""


def probe() -> Dict[str, Any]:
    """Versions of what is installed in the virtual environment."""
    import json

    result = subprocess.run(
        [str(VENV_PYTHON), "-W", "ignore", "-c", PROBE], capture_output=True, text=True
    )
    try:
        return dict(json.loads(result.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return {"python": "?", "packages": {}, "sampling": False}


def frontend_built(project: Dict[str, Any]) -> bool:
    return (project["dir"] / "frontend" / "dist" / "index.html").is_file()


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def need_venv() -> None:
    if not VENV_PYTHON.exists():
        die("there is no environment yet, run `fex prepare` first")
    if not in_venv():  # started with another Python: hand over to the environment's
        os.execv(
            str(VENV_PYTHON),
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
def tls_verify(settings: Dict[str, str]) -> Any:
    value = settings["ARANGO_VERIFY"].strip()
    if not value:
        return None
    if value.lower() in ("false", "0", "no", "off"):
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return False
    return value


def client(settings: Dict[str, str]) -> Any:
    import logging

    from arango import ArangoClient

    logging.getLogger("urllib3").setLevel(logging.ERROR)
    return ArangoClient(
        hosts=settings["ARANGO_URL"], verify_override=tls_verify(settings)
    )


def connect(settings: Dict[str, str]) -> Tuple[Any, Optional[str]]:
    """(database, None) or (None, why it failed)."""
    try:
        db = client(settings).db(
            settings["ARANGO_DB"],
            settings["ARANGO_USER"],
            settings["ARANGO_PASSWORD"],
            verify=True,
        )
        return db, None
    except ImportError:
        return None, "python-arango is not installed, run `fex prepare`"
    except Exception as e:
        hint = ""
        if settings["ARANGO_URL"].startswith("https") and not settings["ARANGO_VERIFY"]:
            hint = " (self-signed certificate? try `fex env set ARANGO_VERIFY=false`)"
        return None, f"{e}{hint}"


def must_connect() -> Any:
    settings = read_env()
    db, error = connect(settings)
    if db is None:
        die(
            f"cannot connect to database '{settings['ARANGO_DB']}' at "
            f"{settings['ARANGO_URL']} as '{settings['ARANGO_USER']}': {error}\n"
            "       `fex env` shows the settings, `fex db init` creates database and user."
        )
    return db


def count(db: Any, name: str) -> Optional[int]:
    return db.collection(name).count() if db.has_collection(name) else None


def data_rows(db: Any, keys: List[str]) -> bool:
    """Prints the collections of the given examples, returns True if all data is there."""
    complete = True
    seen = set()
    for key in keys:
        project = PROJECTS[key]
        heading(f"{key}  {project['title']}")
        for kind in ("data", "embeddings"):
            for name in project[kind]:
                n = count(db, name)
                shared = "  (same as above)" if name in seen and kind == "data" else ""
                if n:
                    plural = "" if n == 1 else "s"
                    row("ok", name, f"{n:,} document{plural}{shared}")
                elif kind == "data":
                    complete = False
                    row("warn", name, f"missing, run `fex db load {key}`")
                else:
                    how = "fex run ex1" if key == "ex1" else f"fex train {key}"
                    row("info", name, f"no embeddings yet, run `{how}`")
                seen.add(name)
    return complete


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_prepare(args: argparse.Namespace) -> None:
    """Creates the environment, installs the packages, creates .env, builds the UIs.
    Every step is skipped if it is already done, so the command can be repeated."""
    heading("1/5  Python environment")
    if not VENV_PYTHON.exists():
        python = args.python or sys.executable
        version = subprocess.run(
            [python, "-c", "import sys; print(sys.version_info >= (3, 10))"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if version != "True":
            die(f"{python} is older than Python 3.10, pass a newer one with --python")
        sh([python, "-m", "venv", str(VENV)])
        sh([str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    row("ok", "virtual environment", str(VENV))
    pip = [str(VENV_PYTHON), "-m", "pip", "install"]

    heading("2/5  PyTorch and fastgraphml")
    if not venv_has("torch"):
        sh(pip + ["torch"])
    torch = probe()["packages"].get("torch", "")
    row("ok", "torch", torch)
    if args.extensions and not venv_has("pyg_lib"):
        torch = torch.split("+")[0]
        sh(
            pip
            + [
                "pyg-lib",
                "torch-scatter",
                "torch-sparse",
                "-f",
                f"https://data.pyg.org/whl/torch-{torch}+cpu.html",
            ]
        )
    if not venv_has("fastgraphml", "torch_geometric", "faiss", "umap"):
        sh(pip + ["-e", f"{ROOT}[search,viz]"])
    row(
        "ok",
        "fastgraphml",
        "installed from this repository, with search and viz extras",
    )

    heading("3/5  ArangoDB adapter and example packages")
    if not venv_has("adbpyg_adapter", "arango_datasets", "pandas", "arango"):
        # The adapter lists torch-scatter and torch-sparse as requirements but never
        # imports them, so it is installed without them. It needs python-arango 7.
        sh(pip + ["--no-deps", "adbpyg-adapter", "arango-datasets"])
        sh(pip + ["pandas", "python-arango~=7.6"])
    row(
        "ok",
        "adbpyg-adapter",
        "pip may warn about torch-scatter/torch-sparse: not needed",
    )
    if not venv_has("typer", "fastapi", "uvicorn", "httpx"):
        sh(pip + ["-r", str(EXAMPLES / "requirements.txt")])
    row("ok", "typer, fastapi, uvicorn", "")

    heading("4/5  Settings")
    if not ENV_FILE.is_file():
        shutil.copy(EXAMPLES / ".env.example", ENV_FILE)
        row(
            "warn",
            "examples/.env",
            "created from .env.example: set the password with "
            "`fex env set ARANGO_PASSWORD=...`",
        )
    else:
        row("ok", "examples/.env", "exists, left as it is")

    heading("5/5  Web interfaces")
    npm = shutil.which("npm")
    if args.skip_frontend:
        row("info", "frontends", "skipped (--skip-frontend)")
    elif npm is None:
        row(
            "warn",
            "frontends",
            "npm not found: install Node.js, then run `fex prepare` again",
        )
    else:
        for key in ("ex2", "ex3"):
            frontend = PROJECTS[key]["dir"] / "frontend"
            if frontend_built(PROJECTS[key]) and not args.rebuild:
                row(
                    "ok",
                    f"{key} frontend",
                    "already built (use --rebuild to build again)",
                )
                continue
            if not (frontend / "node_modules").is_dir():
                sh([npm, "install", "--no-audit", "--no-fund"], cwd=frontend)
            sh([npm, "run", "build"], cwd=frontend)
            row("ok", f"{key} frontend", "built")

    heading("Next")
    print(
        "  fex status            check everything\n"
        "  fex db init           create the database and user from examples/.env (optional)\n"
        "  fex db load all       load the example data\n"
        "  fex run ex2           start an example   (add --offline to try it without ArangoDB)"
    )


def cmd_status(args: argparse.Namespace) -> None:
    if VENV_PYTHON.exists() and not in_venv() and venv_has("arango"):
        need_venv()  # continue inside the environment, where python-arango is

    heading("Environment")
    if not VENV_PYTHON.exists():
        row("fail", "virtual environment", f"{VENV} missing, run `fex prepare`")
    else:
        found = probe()
        row("ok", "virtual environment", f"{VENV}  (Python {found['python']})")
        for dist, label, needed in (
            ("torch", "torch", True),
            ("torch-geometric", "torch-geometric", True),
            ("fastgraphml", "fastgraphml", True),
            ("adbpyg-adapter", "adbpyg-adapter", True),
            ("python-arango", "python-arango", True),
            ("faiss-cpu", "faiss (search)", False),
            ("umap-learn", "umap-learn (maps)", False),
            ("fastapi", "fastapi (web apps)", True),
            ("typer", "typer (ex1)", True),
        ):
            if dist in found["packages"]:
                row("ok", label, found["packages"][dist])
            else:
                row(
                    "fail" if needed else "warn",
                    label,
                    "not installed, run `fex prepare`",
                )
        if found["sampling"]:
            row("ok", "neighbourhood sampling", "pyg-lib or torch-sparse found")
        else:
            row(
                "info",
                "neighbourhood sampling",
                "not installed: SAGE and GAT train on the "
                "full graph, fine for these examples",
            )
    node = shutil.which("node")
    row(
        "ok" if node else "warn",
        "node / npm",
        (
            subprocess.run(
                ["node", "--version"], capture_output=True, text=True
            ).stdout.strip()
            if node
            else "not found: needed to build the web interfaces"
        ),
    )
    for key in ("ex2", "ex3"):
        built = frontend_built(PROJECTS[key])
        row(
            "ok" if built else "warn",
            f"{key} frontend",
            "built" if built else "not built, run `fex prepare`",
        )

    heading("Settings (examples/.env)")
    settings = read_env()
    if not ENV_FILE.is_file():
        row("warn", "examples/.env", "missing, run `fex prepare`; showing the defaults")
    for key, value in masked(settings).items():
        row("info", key, value)

    heading("Database")
    if not (in_venv() and venv_has("arango")):
        row("fail", "connection", "python-arango is not installed, run `fex prepare`")
    else:
        db, error = connect(settings)
        if db is None:
            row("fail", "connection", str(error))
        else:
            row("ok", "connection", f"ArangoDB {db.version()}, database '{db.name}'")
            data_rows(db, list(PROJECTS))

    heading("Running")
    for key in ("ex2", "ex3"):
        port = PROJECTS[key]["port"]
        if port_open(port):
            row("ok", f"{key} on port {port}", f"http://localhost:{port}")
        else:
            row(
                "info",
                f"{key} on port {port}",
                f"not running, start it with `fex run {key}`",
            )
    print()


def cmd_env(args: argparse.Namespace) -> None:
    if args.action == "set":
        changes = {}
        for pair in args.pairs:
            if "=" not in pair:
                die(f"expected KEY=VALUE, got '{pair}'")
            key, value = pair.split("=", 1)
            if key not in ENV_DEFAULTS and key != "FGML_MODE":
                die(f"unknown setting '{key}', known: {', '.join(ENV_DEFAULTS)}")
            changes[key] = value
        if not changes:
            die("nothing to set, e.g. `fex env set ARANGO_VERIFY=false`")
        if not ENV_FILE.is_file():
            shutil.copy(EXAMPLES / ".env.example", ENV_FILE)
        write_env(changes)
    heading(f"Settings ({ENV_FILE})")
    for key, value in masked(read_env()).items():
        row("info", key, value)
    print()


def cmd_db(args: argparse.Namespace) -> None:
    need_venv()
    keys = list(PROJECTS) if args.target in (None, "all") else [args.target]
    settings = read_env()

    if args.action == "init":
        # needs an administrator; the password is asked for and not stored anywhere
        admin = args.admin_user
        password = getpass.getpass(f"Password of ArangoDB user '{admin}': ")
        try:
            system = client(settings).db("_system", admin, password, verify=True)
        except Exception as e:
            die(f"cannot connect to {settings['ARANGO_URL']} as '{admin}': {e}")
        name, user = settings["ARANGO_DB"], settings["ARANGO_USER"]
        if system.has_database(name):
            row("ok", f"database {name}", "exists already")
        else:
            system.create_database(name)
            row("ok", f"database {name}", "created")
        if user != admin:
            if system.has_user(user):
                row("ok", f"user {user}", "exists already (password not changed)")
            else:
                system.create_user(user, settings["ARANGO_PASSWORD"], active=True)
                row("ok", f"user {user}", "created with ARANGO_PASSWORD")
            system.update_permission(user, "rw", database=name)
            row("ok", "permission", f"{user} can read and write {name}, nothing else")
        return

    db = must_connect()
    if args.action == "load":
        python = str(VENV_PYTHON)
        if any(k in keys for k in ("ex1", "ex2")):
            heading("Amazon Computer Products (ex1, ex2)")
            sh(
                [python, "-W", "ignore", "-m", "backend.load_data"],
                cwd=PROJECTS["ex2"]["dir"],
            )
        if "ex3" in keys:
            heading("Clinical trial graph (ex3)")
            sh(
                [python, "-W", "ignore", "-m", "backend.load_data"],
                cwd=PROJECTS["ex3"]["dir"],
            )
        data_rows(db, keys)
    elif args.action == "clean":
        names: List[str] = []
        for key in keys:
            kinds = ("embeddings",) if args.embeddings_only else ("data", "embeddings")
            names += [n for kind in kinds for n in PROJECTS[key][kind]]
        # ex1 and ex2 share the product graph: keep it while the other one is not cleaned
        if not args.embeddings_only and ("ex1" in keys) != ("ex2" in keys):
            names = [n for n in names if n not in PROJECTS["ex2"]["data"]]
            print(
                "  The product graph is shared by ex1 and ex2 and is kept. "
                "Clean both (or `all`) to drop it."
            )
        names = [n for n in dict.fromkeys(names) if db.has_collection(n)]
        if not names:
            print("  Nothing to drop.")
            return
        print(f"  Collections to drop from '{db.name}': " + ", ".join(names))
        if (
            not args.yes
            and input("  Type 'yes' to drop them: ").strip().lower() != "yes"
        ):
            print("  Nothing dropped.")
            return
        for graph in db.graphs():  # named graphs built on these collections go first
            if any(d["edge_collection"] in names for d in graph["edge_definitions"]):
                db.delete_graph(graph["name"], drop_collections=False)
                row("ok", f"graph {graph['name']}", "removed")
        for name in names:
            db.delete_collection(name)
            row("ok", name, "dropped")
    else:
        row("ok", "connection", f"ArangoDB {db.version()}, database '{db.name}'")
        data_rows(db, keys)
    print()


def example_env(offline: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if offline:
        env["FGML_MODE"] = "offline"
    return env


def cmd_train(args: argparse.Namespace) -> None:
    """Trains in the terminal, with the full error output if something fails."""
    need_venv()
    if args.example == "ex1":
        die("ex1 trains as part of its pipeline: `fex run ex1`")
    command = [str(VENV_PYTHON), "-W", "ignore", "-m", "backend.train"]
    if args.epochs:
        command += ["--epochs", str(args.epochs)]
    if args.model and args.example == "ex2":
        command += ["--model", args.model]
    sh(command, cwd=PROJECTS[args.example]["dir"], env=example_env(args.offline))


def cmd_run(args: argparse.Namespace) -> None:
    need_venv()
    project = PROJECTS[args.example]
    env = example_env(args.offline)
    python = str(VENV_PYTHON)

    if args.example == "ex1":
        # with arguments: any fgml.py command; without: the whole pipeline
        config = (
            "configs/cora_offline.json"
            if args.offline
            else "configs/amazon_arangodb.json"
        )
        if args.args and not args.args[0].startswith("-"):
            extra = args.args  # a whole fgml.py command
        else:
            extra = [
                "run",
                config,
                *args.args,
            ]  # the pipeline, plus options such as --epochs
        result = subprocess.run(
            [python, "-W", "ignore", "fgml.py", *extra], cwd=project["dir"], env=env
        )
        sys.exit(result.returncode)

    port = args.port or project["port"]
    if port_open(port):
        die(
            f"port {port} is in use (is {args.example} running already?), choose another with --port"
        )
    children: List[subprocess.Popen] = []
    url = f"http://localhost:{port}"
    if args.dev:
        npm = shutil.which("npm") or die("npm not found")
        frontend = project["dir"] / "frontend"
        if not (frontend / "node_modules").is_dir():
            sh([str(npm), "install", "--no-audit", "--no-fund"], cwd=frontend)
        children.append(subprocess.Popen([str(npm), "run", "dev"], cwd=frontend))
        url = f"http://localhost:{project['dev_port']}"
        if port != project["port"]:
            print(
                f"  note: the dev server forwards /api to port {project['port']}, not {port}"
            )
    elif not frontend_built(project):
        die(
            f"the web interface of {args.example} is not built, run `fex prepare` "
            "(or use --dev to run it with the Vite dev server)"
        )

    mode = "offline data" if args.offline else f"ArangoDB {read_env()['ARANGO_DB']}"
    heading(f"{project['title']}  ({mode})")
    print(f"  {url}      stop with Ctrl+C\n")
    if args.open:
        threading.Timer(2.5, webbrowser.open, [url]).start()
    try:
        subprocess.run(
            [
                python,
                "-W",
                "ignore",
                "-m",
                "uvicorn",
                "backend.app:app",
                "--port",
                str(port),
            ],
            cwd=project["dir"],
            env=env,
        )
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            child.terminate()


def cmd_link(args: argparse.Namespace) -> None:
    """Symlinks the launcher into a folder of the PATH, so that `fex` works anywhere."""
    launcher = ROOT / "fex"
    on_path = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    candidates = [Path.home() / ".local" / "bin", Path.home() / "bin"]
    target_dir = (
        Path(args.dir).expanduser()
        if args.dir
        else next((c for c in candidates if c in on_path and c.is_dir()), None)
    )
    if target_dir is None:
        die(
            "neither ~/.local/bin nor ~/bin is on your PATH: pass a folder with --dir, "
            f"or add `alias fex={launcher}` to your shell profile"
        )
    target = target_dir / "fex"
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == launcher.resolve():
            print(f"  {target} already points to {launcher}")
            return
        die(f"{target} exists and is something else, remove it first")
    target.symlink_to(launcher)
    print(f"  {target} -> {launcher}")
    if target_dir not in on_path:
        print(f"  note: {target_dir} is not on your PATH")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fex",
        description="Prepare, inspect and run the fastgraphml example projects.",
        epilog="Examples: ex1 = embedding pipeline CLI, ex2 = product recommender, "
        "ex3 = clinical site finder.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser(
        "prepare", help="create the environment, install packages, build the UIs"
    )
    p.add_argument(
        "--python", help="Python to create the environment with (default: this one)"
    )
    p.add_argument(
        "--extensions",
        action="store_true",
        help="also install pyg-lib, torch-scatter and torch-sparse wheels",
    )
    p.add_argument(
        "--skip-frontend", action="store_true", help="do not build the web interfaces"
    )
    p.add_argument(
        "--rebuild", action="store_true", help="build the web interfaces again"
    )
    p.set_defaults(run=cmd_prepare)

    p = sub.add_parser(
        "status", help="what is installed, configured, loaded and running"
    )
    p.set_defaults(run=cmd_status)

    p = sub.add_parser("env", help="show or change the settings in examples/.env")
    p.add_argument("action", nargs="?", choices=["show", "set"], default="show")
    p.add_argument("pairs", nargs="*", metavar="KEY=VALUE")
    p.set_defaults(run=cmd_env)

    p = sub.add_parser("db", help="show, create, load or clean the example data")
    p.add_argument(
        "action",
        nargs="?",
        choices=["status", "init", "load", "clean"],
        default="status",
    )
    p.add_argument(
        "target",
        nargs="?",
        choices=[*PROJECTS, "all"],
        help="which example (default: all)",
    )
    p.add_argument(
        "--admin-user", default="root", help="administrator for `init` (default: root)"
    )
    p.add_argument(
        "--embeddings-only",
        action="store_true",
        help="`clean` drops only the embedding collections",
    )
    p.add_argument("--yes", action="store_true", help="`clean` without asking")
    p.set_defaults(run=cmd_db)

    p = sub.add_parser(
        "train", help="train and store embeddings of ex2 or ex3 in the terminal"
    )
    p.add_argument("example", choices=list(PROJECTS))
    p.add_argument("--epochs", type=int)
    p.add_argument("--model", choices=["SAGE", "GAT"], help="ex2 only")
    p.add_argument("--offline", action="store_true", help="without ArangoDB")
    p.set_defaults(run=cmd_train)

    p = sub.add_parser(
        "run",
        help="run an example",
        epilog="ex1 also takes any fgml.py command: `fex run ex1 similar runs/cora 42 -k 5`",
    )
    p.add_argument("example", choices=list(PROJECTS))
    p.add_argument("--offline", action="store_true", help="without ArangoDB")
    p.add_argument(
        "--port", type=int, help="port of the web app (ex2: 8000, ex3: 8001)"
    )
    p.add_argument("--dev", action="store_true", help="also start the Vite dev server")
    p.add_argument("--open", action="store_true", help="open the browser")
    p.set_defaults(run=cmd_run)

    p = sub.add_parser("link", help="put `fex` on your PATH")
    p.add_argument("--dir", help="folder for the link (default: ~/.local/bin or ~/bin)")
    p.set_defaults(run=cmd_link)

    args, extra = parser.parse_known_args()
    if not args.command:
        parser.print_help()
        return
    if extra and not (args.command == "run" and args.example == "ex1"):
        parser.error("unrecognized arguments: " + " ".join(extra))
    args.args = extra
    args.run(args)


if __name__ == "__main__":
    main()
