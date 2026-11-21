"""Offline tests of the site finder: no database and no training needed.

FGML_MODE=offline pytest examples/03-clinical-site-finder/tests
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ["FGML_MODE"] = "offline"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import app as app_module  # noqa: E402
from backend import config, data  # noqa: E402
from backend.store import MemoryStore  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    store = MemoryStore()
    # stand-in embeddings: one direction per primary area, plus noise
    rng = np.random.default_rng(0)
    areas = np.array([store.site_docs[k]["area"] for k in store.keys])
    emb = np.eye(len(config.AREAS))[areas] + rng.normal(
        0, 0.2, (len(areas), len(config.AREAS))
    )
    store.set_embeddings(emb, rng.random((len(areas), 2)))
    monkeypatch.setattr(app_module, "_store", store)
    return TestClient(app_module.app)


def test_generator_is_deterministic():
    first, second = data.generate(), data.generate()
    assert first["conducts"] == second["conducts"]
    assert len(first["site"]) == data.NUM_SITES
    # every site takes part in at least one trial
    assert {e["_from"] for e in first["conducts"]} == {
        f"site/{s['_key']}" for s in first["site"]
    }
    # generator internals do not leak into the documents
    assert all(not k.startswith("_") or k == "_key" for k in first["site"][0])


def test_pyg_graph():
    graph = data.to_pyg(data.generate())
    assert graph["site"].x.shape == (data.NUM_SITES, 3)
    assert graph["site", "conducts", "trial"].edge_index.shape[0] == 2
    assert int(graph["trial", "studies", "indication"].edge_index[1].max()) == 29


def test_recommend_splits_experienced_and_lookalikes(client):
    result = client.get("/api/recommend", params={"indication": "I06", "k": 10}).json()
    assert result["indication"]["name"] == "Heart Failure"
    assert all(site["trials"] > 0 for site in result["experienced"])
    assert all(site["trials"] == 0 for site in result["lookalikes"])
    # the lookalikes come from the therapeutic area of the indication
    assert sum(s["area_name"] == "Cardiology" for s in result["lookalikes"]) >= 8
    scores = [s["score"] for s in result["lookalikes"]]
    assert scores == sorted(scores, reverse=True)


def test_country_filter_and_site_profile(client):
    result = client.get(
        "/api/recommend", params={"indication": "I06", "country": "Germany"}
    ).json()
    sites = result["experienced"] + result["lookalikes"]
    assert sites and all(site["country"] == "Germany" for site in sites)

    profile = client.get(f"/api/sites/{sites[0]['key']}").json()
    assert profile["trial_count"] == sum(i["trials"] for i in profile["indications"])
    similar = client.get(
        f"/api/sites/{sites[0]['key']}/similar", params={"k": 5}
    ).json()
    assert len(similar) == 5 and sites[0]["key"] not in [s["key"] for s in similar]


def test_errors(client):
    assert (
        client.get("/api/recommend", params={"indication": "nope"}).status_code == 404
    )
    assert client.get("/api/sites/nope").status_code == 404
