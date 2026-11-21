"""Where the API reads sites, trials and embeddings from.

Two implementations with the same methods:

* ArangoStore  - the real thing: AQL traversals over the clinical_trials graph and
                 cosine similarity over the embedding collection fastgraphml wrote.
* MemoryStore  - offline mode: the generated records and numpy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from . import config, data

SITE_FIELDS = [
    "name",
    "city",
    "country",
    "kind",
    "area",
    "area_name",
    "secondary_area_name",
    "capacity",
    "startup_days",
    "enrollment_rate",
]


def site_view(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {"key": doc["_key"], **{f: doc.get(f) for f in SITE_FIELDS}}


def profile(site: Dict[str, Any], trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Site details for the UI: what it has worked on, and with whom."""
    by_indication = Counter((t["indication"], t["area"]) for t in trials)
    by_sponsor = Counter(t["sponsor"] for t in trials if t.get("sponsor"))
    return {
        **site,
        "trial_count": len(trials),
        "indications": [
            {"name": name, "area": area, "trials": n}
            for (name, area), n in by_indication.most_common()
        ],
        "sponsors": [
            {"name": name, "trials": n} for name, n in by_sponsor.most_common(5)
        ],
        "recent_trials": sorted(trials, key=lambda t: (-t["year"], t["key"]))[:8],
    }


class ArangoStore:
    def __init__(self, db: Any):
        self.db = db

    def _aql(self, query: str, **bind_vars: Any) -> List[Any]:
        return list(self.db.aql.execute(query, bind_vars=bind_vars))

    # ---- graph data ------------------------------------------------------- #
    def has_graph(self) -> bool:
        return (
            all(
                self.db.has_collection(name)
                for name in (
                    config.SITES,
                    config.TRIALS,
                    config.INDICATIONS,
                    config.CONDUCTS,
                    config.STUDIES,
                )
            )
            and self.db.collection(config.SITES).count() > 0
        )

    def counts(self) -> Dict[str, int]:
        if not self.has_graph():
            return {}
        return {
            "sites": self.db.collection(config.SITES).count(),
            "trials": self.db.collection(config.TRIALS).count(),
            "indications": self.db.collection(config.INDICATIONS).count(),
        }

    def facets(self) -> Dict[str, Any]:
        indications = self._aql(
            "FOR i IN @@c SORT i.area_id, i.name "
            "RETURN { key: i._key, name: i.name, area: i.area }",
            **{"@c": config.INDICATIONS},
        )
        countries = self._aql(
            "FOR s IN @@c COLLECT country = s.country SORT country RETURN country",
            **{"@c": config.SITES},
        )
        return {"indications": indications, "countries": countries}

    def sites(
        self, country: Optional[str], q: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        rows = self._aql(
            """
            FOR s IN @@sites
                FILTER @country == null OR s.country == @country
                FILTER @q == null OR CONTAINS(LOWER(CONCAT(s.name, " ", s.city)), LOWER(@q))
                SORT s.name, s.city
                LIMIT @limit
                RETURN s
            """,
            **{
                "@sites": config.SITES,
                "country": country,
                "q": q or None,
                "limit": limit,
            },
        )
        return [site_view(r) for r in rows]

    def site(self, key: str) -> Optional[Dict[str, Any]]:
        """One traversal: site -> its trials -> indication and sponsor of each."""
        rows = self._aql(
            """
            LET s = DOCUMENT(CONCAT(@sites, "/", @key))
            FILTER s != null
            LET trials = (
                FOR t IN 1..1 OUTBOUND s @@conducts
                    LET i = FIRST(FOR i IN 1..1 OUTBOUND t @@studies RETURN i)
                    LET p = FIRST(FOR p IN 1..1 INBOUND t @@funds RETURN p)
                    RETURN { key: t._key, title: t.title, phase: t.phase, year: t.year,
                             indication: i.name, area: i.area, sponsor: p.name }
            )
            RETURN { site: s, trials: trials }
            """,
            **{
                "sites": config.SITES,
                "key": key,
                "@conducts": config.CONDUCTS,
                "@studies": config.STUDIES,
                "@funds": config.FUNDS,
            },
        )
        if not rows:
            return None
        return profile(site_view(rows[0]["site"]), rows[0]["trials"])

    def indication(self, key: str) -> Optional[Dict[str, Any]]:
        doc = self.db.collection(config.INDICATIONS).get(key)
        return (
            {"key": doc["_key"], "name": doc["name"], "area": doc["area"]}
            if doc
            else None
        )

    def experience(self, indication_key: str) -> Dict[str, int]:
        """Number of trials in the indication, per site that ran at least one."""
        rows = self._aql(
            """
            FOR t IN 1..1 INBOUND @indication @@studies
                FOR s IN 1..1 INBOUND t @@conducts
                    COLLECT site = s._key WITH COUNT INTO n
                    RETURN [site, n]
            """,
            **{
                "indication": f"{config.INDICATIONS}/{indication_key}",
                "@studies": config.STUDIES,
                "@conducts": config.CONDUCTS,
            },
        )
        return {site: n for site, n in rows}

    # ---- embeddings ------------------------------------------------------- #
    def has_embeddings(self) -> bool:
        return bool(
            self.db.has_collection(config.EMBEDDINGS)
            and self.db.collection(config.EMBEDDINGS).count() > 0
        )

    def vectors(self, keys: List[str]) -> Dict[str, np.ndarray]:
        rows = self._aql(
            "FOR e IN @@emb FILTER e._key IN @keys RETURN [e._key, e.embedding]",
            **{"@emb": config.EMBEDDINGS, "keys": keys},
        )
        return {key: np.asarray(vector, dtype=np.float32) for key, vector in rows}

    def rank(
        self, vector: np.ndarray, country: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Sites ordered by the cosine similarity of their embedding to `vector`.
        The embedding documents have the same _key as their site."""
        rows = self._aql(
            """
            FOR e IN @@emb
                LET s = DOCUMENT(CONCAT(@sites, "/", e._key))
                FILTER s != null
                FILTER @country == null OR s.country == @country
                LET score = COSINE_SIMILARITY(e.embedding, @vector)
                SORT score DESC
                LIMIT @limit
                RETURN { site: s, score: score }
            """,
            **{
                "@emb": config.EMBEDDINGS,
                "sites": config.SITES,
                "country": country,
                "vector": [float(v) for v in vector],
                "limit": limit,
            },
        )
        return [{**site_view(r["site"]), "score": r["score"]} for r in rows]

    def save_map(self, keys: List[str], xy: np.ndarray) -> None:
        docs = [
            {"_key": key, "xy": [round(float(x), 4), round(float(y), 4)]}
            for key, (x, y) in zip(keys, xy)
        ]
        self.db.collection(config.EMBEDDINGS).import_bulk(docs, on_duplicate="update")

    def map_points(self) -> List[Dict[str, Any]]:
        return self._aql(
            """
            FOR e IN @@emb
                FILTER e.xy != null
                LET s = DOCUMENT(CONCAT(@sites, "/", e._key))
                FILTER s != null
                RETURN { key: s._key, x: e.xy[0], y: e.xy[1], area: s.area,
                         name: s.name, city: s.city, country: s.country }
            """,
            **{"@emb": config.EMBEDDINGS, "sites": config.SITES},
        )


class MemoryStore:
    """Offline mode: the generated records in dictionaries, embeddings in numpy."""

    def __init__(self) -> None:
        self.records = data.generate()
        self.site_docs = {s["_key"]: s for s in self.records["site"]}
        self.keys = list(self.site_docs)
        self.index = {k: i for i, k in enumerate(self.keys)}
        indications = {i["_key"]: i for i in self.records["indication"]}
        sponsors = {p["_key"]: p for p in self.records["sponsor"]}
        trial_docs = {t["_key"]: t for t in self.records["trial"]}
        self.indication_docs = indications

        key = lambda ref: ref.split("/")[1]  # noqa: E731
        trial_indication = {
            key(e["_from"]): indications[key(e["_to"])] for e in self.records["studies"]
        }
        trial_sponsor = {
            key(e["_to"]): sponsors[key(e["_from"])] for e in self.records["funds"]
        }
        self.site_trials: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.indication_sites: Dict[str, Counter] = defaultdict(Counter)
        for edge in self.records["conducts"]:
            site_key, trial_key = key(edge["_from"]), key(edge["_to"])
            trial, indication = trial_docs[trial_key], trial_indication[trial_key]
            self.site_trials[site_key].append(
                {
                    "key": trial_key,
                    "title": trial["title"],
                    "phase": trial["phase"],
                    "year": trial["year"],
                    "indication": indication["name"],
                    "area": indication["area"],
                    "sponsor": trial_sponsor[trial_key]["name"],
                }
            )
            self.indication_sites[indication["_key"]][site_key] += 1

        self.emb: Optional[np.ndarray] = None  # L2 normalised, in the order of keys
        self.xy: Optional[np.ndarray] = None
        if self._file.is_file():
            with np.load(self._file) as cached:
                if len(cached["emb"]) == len(self.keys):
                    self.emb, self.xy = cached["emb"], cached["xy"]

    @property
    def _file(self) -> Any:
        return config.CACHE_DIR / "sites.npz"

    def set_embeddings(self, emb: np.ndarray, xy: np.ndarray) -> None:
        emb = emb.astype(np.float32)
        self.emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        self.xy = xy.astype(np.float32)
        config.CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(self._file, emb=self.emb, xy=self.xy)

    # ---- same methods as ArangoStore ---------------------------------------- #
    def has_graph(self) -> bool:
        return True

    def counts(self) -> Dict[str, int]:
        return {
            "sites": len(self.keys),
            "trials": len(self.records["trial"]),
            "indications": len(self.records["indication"]),
        }

    def facets(self) -> Dict[str, Any]:
        return {
            "indications": [
                {"key": i["_key"], "name": i["name"], "area": i["area"]}
                for i in self.records["indication"]
            ],
            "countries": sorted({s["country"] for s in self.records["site"]}),
        }

    def sites(
        self, country: Optional[str], q: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        found = [
            s
            for s in self.records["site"]
            if (country is None or s["country"] == country)
            and (not q or q.lower() in f"{s['name']} {s['city']}".lower())
        ]
        found.sort(key=lambda s: (s["name"], s["city"]))
        return [site_view(s) for s in found[:limit]]

    def site(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self.site_docs:
            return None
        return profile(site_view(self.site_docs[key]), self.site_trials[key])

    def indication(self, key: str) -> Optional[Dict[str, Any]]:
        doc = self.indication_docs.get(key)
        return (
            {"key": doc["_key"], "name": doc["name"], "area": doc["area"]}
            if doc
            else None
        )

    def experience(self, indication_key: str) -> Dict[str, int]:
        return dict(self.indication_sites[indication_key])

    def has_embeddings(self) -> bool:
        return self.emb is not None

    def vectors(self, keys: List[str]) -> Dict[str, np.ndarray]:
        assert self.emb is not None
        return {k: self.emb[self.index[k]] for k in keys if k in self.index}

    def rank(
        self, vector: np.ndarray, country: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        assert self.emb is not None
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.emb @ vector
        ranked = []
        for i in np.argsort(-scores):
            doc = self.site_docs[self.keys[int(i)]]
            if country is None or doc["country"] == country:
                ranked.append({**site_view(doc), "score": float(scores[i])})
                if len(ranked) == limit:
                    break
        return ranked

    def map_points(self) -> List[Dict[str, Any]]:
        if self.xy is None:
            return []
        return [
            {
                "key": k,
                "x": round(float(self.xy[i, 0]), 4),
                "y": round(float(self.xy[i, 1]), 4),
                "area": self.site_docs[k]["area"],
                "name": self.site_docs[k]["name"],
                "city": self.site_docs[k]["city"],
                "country": self.site_docs[k]["country"],
            }
            for i, k in enumerate(self.keys)
        ]
