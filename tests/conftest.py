from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="fastgraphml")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="")


@pytest.fixture(scope="session")
def db(request: pytest.FixtureRequest) -> Any:
    """ArangoDB database loaded with the Cora and the IMDB graph.

    The tests using it are skipped if the ArangoDB extras are not installed or if
    there is no ArangoDB instance to connect to.
    """
    pytest.importorskip("adbpyg_adapter")
    from adbpyg_adapter import ADBPyG_Adapter
    from arango import ArangoClient
    from arango.http import DefaultHTTPClient
    from torch_geometric.datasets import IMDB, Planetoid

    config = request.config
    con = {
        "url": config.getoption("url"),
        "username": config.getoption("username"),
        "password": config.getoption("password"),
        "dbName": config.getoption("dbName"),
    }

    class NoTimeoutHTTPClient(DefaultHTTPClient):
        REQUEST_TIMEOUT = None

    try:
        client = ArangoClient(hosts=con["url"], http_client=NoTimeoutHTTPClient())
        sys_db = client.db("_system", con["username"], con["password"], verify=True)
        if not sys_db.has_database(con["dbName"]):
            sys_db.create_database(con["dbName"])
        database = client.db(
            con["dbName"], con["username"], con["password"], verify=True
        )
    except Exception as e:
        pytest.skip(f"no ArangoDB instance at {con['url']}: {e}")

    adbpyg = ADBPyG_Adapter(database)
    # loading Cora dataset (Homogeneous)
    dataset_cora = Planetoid("./", "Cora")[0]
    database.delete_graph("cora", drop_collections=True, ignore_missing=True)
    adbpyg.pyg_to_arangodb("cora", dataset_cora)

    # loading IMDB dataset (Heterogeneous)
    dataset_imdb = IMDB("./imdb")[0]
    database.delete_graph("imdb", drop_collections=True, ignore_missing=True)
    adbpyg.pyg_to_arangodb("imdb", dataset_imdb)

    return database
