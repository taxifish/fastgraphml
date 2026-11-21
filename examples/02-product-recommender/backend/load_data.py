"""Loads the Amazon Computer Products graph into the configured database.

    python -m backend.load_data

Uses the arango-datasets package (pip install 'fastgraphml[arango]'). It creates
the Computer_Products and bought_together collections and the product_graph graph.
"""

from __future__ import annotations

from . import config


def main() -> None:
    from fastgraphml import Datasets

    db = config.connect()
    if (
        db.has_collection(config.PRODUCTS)
        and db.collection(config.PRODUCTS).count() > 0
    ):
        print(
            f"'{config.PRODUCTS}' already has documents in '{db.name}', nothing to do."
        )
        return
    Datasets(db).load("AMAZON_COMPUTER_PRODUCTS")
    print(
        f"Loaded {db.collection(config.PRODUCTS).count()} products and "
        f"{db.collection(config.BOUGHT_TOGETHER).count()} co-purchase edges into '{db.name}'."
    )


if __name__ == "__main__":
    main()
