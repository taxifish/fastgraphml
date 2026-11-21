"""Synthetic clinical trial landscape.

Everything here is made up: sites, sponsors and trials are generated from a fixed
random seed, so every run produces the same data. The generator builds in the
structure a real landscape has, which is what the embeddings should discover:

* a site has a primary (and often a secondary) therapeutic area,
* inside its primary area it is specialised in two or three indications,
* a trial runs in a few countries and picks sites that fit its indication.

The same records feed both modes of the example: `to_pyg` turns them into a PyG
graph (offline), `backend.load_data` writes them to ArangoDB.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

SEED = 7
NUM_SITES = 480
NUM_TRIALS = 1400

AREAS = [
    "Oncology",
    "Cardiology",
    "Neurology",
    "Immunology",
    "Infectious Disease",
    "Endocrinology",
]

INDICATIONS = {
    "Oncology": [
        "Breast Cancer",
        "Non-Small Cell Lung Cancer",
        "Colorectal Cancer",
        "Multiple Myeloma",
        "Prostate Cancer",
    ],
    "Cardiology": [
        "Heart Failure",
        "Atrial Fibrillation",
        "Hypertension",
        "Coronary Artery Disease",
        "Hyperlipidemia",
    ],
    "Neurology": [
        "Alzheimer's Disease",
        "Multiple Sclerosis",
        "Parkinson's Disease",
        "Migraine",
        "Epilepsy",
    ],
    "Immunology": [
        "Rheumatoid Arthritis",
        "Psoriasis",
        "Crohn's Disease",
        "Ulcerative Colitis",
        "Atopic Dermatitis",
    ],
    "Infectious Disease": ["HIV", "Hepatitis B", "Influenza", "COVID-19", "RSV"],
    "Endocrinology": [
        "Type 2 Diabetes",
        "Obesity",
        "Type 1 Diabetes",
        "Osteoporosis",
        "Hypothyroidism",
    ],
}

CITIES = {
    "United States": ["Boston", "Houston", "Chicago", "San Diego", "Atlanta"],
    "United Kingdom": ["London", "Manchester", "Glasgow", "Leeds"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Cologne"],
    "France": ["Paris", "Lyon", "Marseille", "Toulouse"],
    "Spain": ["Madrid", "Barcelona", "Valencia"],
    "Italy": ["Milan", "Rome", "Bologna"],
    "Poland": ["Warsaw", "Krakow", "Gdansk"],
    "Canada": ["Toronto", "Montreal", "Vancouver"],
    "Japan": ["Tokyo", "Osaka", "Nagoya"],
    "Australia": ["Sydney", "Melbourne", "Brisbane"],
    "Brazil": ["Sao Paulo", "Rio de Janeiro", "Curitiba"],
    "South Korea": ["Seoul", "Busan"],
}

KINDS = [
    "University Hospital",
    "Medical Center",
    "Clinical Research Center",
    "General Hospital",
    "Specialty Clinic",
    "Research Institute",
]

EPONYMS = [
    "Riverside",
    "Northgate",
    "Lakeside",
    "Parkview",
    "Central",
    "Harbor",
    "Westfield",
    "Eastbrook",
    "Highland",
    "Meadow",
    "Cedar",
    "Summit",
    "Bridgewater",
    "Oakhill",
    "Stonegate",
    "Greenfield",
    "Kingsway",
    "Fairview",
    "Silverlake",
    "Redwood",
    "Brookside",
    "Ashford",
    "Millbrook",
    "Hillcrest",
]

SPONSOR_STEMS = [
    "Northwind",
    "Halcyon",
    "Verano",
    "Kestrel",
    "Lumen",
    "Arbor",
    "Solace",
    "Meridian",
    "Corvid",
    "Tessera",
    "Aurelia",
    "Boreal",
    "Calyx",
    "Dorian",
    "Elara",
    "Fathom",
    "Galen",
    "Helix",
    "Ione",
    "Juniper",
    "Kepler",
    "Lyra",
    "Mistral",
    "Nimbus",
    "Orchid",
    "Pallas",
    "Quill",
    "Rowan",
    "Sable",
    "Tindra",
]
SPONSOR_KINDS = ["Therapeutics", "Pharma", "Biosciences", "Biotech"]

PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]


def generate() -> Dict[str, List[Dict[str, Any]]]:
    """Returns the records of all collections: site, trial, indication, sponsor
    (vertices) and conducts, studies, funds (edges, with _from and _to)."""
    rng = np.random.default_rng(SEED)

    indications = []
    for area_id, area in enumerate(AREAS):
        for name in INDICATIONS[area]:
            indications.append(
                {
                    "_key": f"I{len(indications) + 1:02d}",
                    "name": name,
                    "area": area,
                    "area_id": area_id,
                }
            )
    by_area: Dict[int, List[int]] = {a: [] for a in range(len(AREAS))}
    for i, indication in enumerate(indications):
        by_area[indication["area_id"]].append(i)

    places = [(country, city) for country, cities in CITIES.items() for city in cities]
    sites, used_names = [], set()
    for n in range(NUM_SITES):
        country, city = places[rng.integers(len(places))]
        while True:
            name = f"{EPONYMS[rng.integers(len(EPONYMS))]} {KINDS[rng.integers(len(KINDS))]}"
            if (name, city) not in used_names:
                used_names.add((name, city))
                break
        primary = int(rng.integers(len(AREAS)))
        secondary = None
        if rng.random() < 0.6:
            secondary = int(rng.choice([a for a in range(len(AREAS)) if a != primary]))
        specialties = rng.choice(
            by_area[primary], size=int(rng.integers(2, 4)), replace=False
        )
        capacity = float(np.clip(rng.gamma(4.0, 1.5), 1.5, 20))  # trials per year
        startup_days = float(np.clip(rng.normal(95, 30), 30, 200))
        enrollment_rate = float(
            np.clip(rng.gamma(3.0, 0.6), 0.2, 6)
        )  # patients / month
        sites.append(
            {
                "_key": f"S{n + 1:04d}",
                "name": name,
                "city": city,
                "country": country,
                "kind": name.split(" ", 1)[1],
                "area": primary,  # the label (y) used to validate the embeddings
                "area_name": AREAS[primary],
                "secondary_area_name": (
                    AREAS[secondary] if secondary is not None else None
                ),
                "capacity": round(capacity, 1),
                "startup_days": round(startup_days),
                "enrollment_rate": round(enrollment_rate, 2),
                # numeric features (x), scaled to 0..1
                "features": [
                    round(capacity / 20, 4),
                    round(startup_days / 200, 4),
                    round(enrollment_rate / 6, 4),
                ],
                # only used by the generator, never by the model
                "_secondary": secondary,
                "_specialties": [int(i) for i in specialties],
            }
        )

    sponsors = []
    for n, stem in enumerate(SPONSOR_STEMS):
        focus = rng.choice(len(AREAS), size=int(rng.integers(1, 3)), replace=False)
        sponsors.append(
            {
                "_key": f"P{n + 1:02d}",
                "name": f"{stem} {SPONSOR_KINDS[n % len(SPONSOR_KINDS)]}",
                "_focus": [int(a) for a in focus],
            }
        )

    countries = list(CITIES)
    site_country = np.array([countries.index(s["country"]) for s in sites])
    site_capacity = np.array([s["capacity"] for s in sites])
    trials, conducts, studies, funds = [], [], [], []
    for n in range(NUM_TRIALS):
        ind_id = int(rng.integers(len(indications)))
        indication = indications[ind_id]
        area = indication["area_id"]
        focused = [p for p in sponsors if area in p["_focus"]]
        pool = focused if focused and rng.random() < 0.8 else sponsors
        sponsor = pool[rng.integers(len(pool))]
        phase = PHASES[int(rng.choice(4, p=[0.2, 0.35, 0.35, 0.1]))]
        key = f"T{n + 1:05d}"
        trials.append(
            {
                "_key": key,
                "title": f"{phase} study in {indication['name']}",
                "phase": phase,
                "year": int(rng.integers(2015, 2026)),
            }
        )
        studies.append(
            {"_from": f"trial/{key}", "_to": f"indication/{indication['_key']}"}
        )
        funds.append({"_from": f"sponsor/{sponsor['_key']}", "_to": f"trial/{key}"})

        # which sites take part: fit to the indication, capacity, trial countries
        weight = np.full(NUM_SITES, 0.03)
        for s, site in enumerate(sites):
            if ind_id in site["_specialties"]:
                weight[s] = 6.0
            elif site["area"] == area:
                weight[s] = 2.0
            elif site["_secondary"] == area:
                weight[s] = 1.0
        trial_countries = rng.choice(
            len(countries), size=int(rng.integers(1, 4)), replace=False
        )
        weight *= np.where(np.isin(site_country, trial_countries), 4.0, 1.0)
        weight *= site_capacity
        num_sites = int(rng.integers(2, 11))
        chosen = rng.choice(
            NUM_SITES, size=num_sites, replace=False, p=weight / weight.sum()
        )
        for s in chosen:
            conducts.append(
                {
                    "_from": f"site/{sites[int(s)]['_key']}",
                    "_to": f"trial/{key}",
                    "enrolled": int(rng.integers(3, 60)),
                }
            )

    # every site takes part in at least one trial
    busy = {edge["_from"] for edge in conducts}
    for site in sites:
        if f"site/{site['_key']}" not in busy:
            fitting = [
                t
                for t, st in zip(trials, studies)
                if indications[int(st["_to"][-2:]) - 1]["area_id"] == site["area"]
            ]
            trial = fitting[rng.integers(len(fitting))]
            conducts.append(
                {
                    "_from": f"site/{site['_key']}",
                    "_to": f"trial/{trial['_key']}",
                    "enrolled": int(rng.integers(3, 60)),
                }
            )

    def public(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in r.items() if k == "_key" or not k.startswith("_")}
            for r in records
        ]

    return {
        "site": public(sites),
        "trial": trials,
        "indication": public(indications),
        "sponsor": public(sponsors),
        "conducts": conducts,
        "studies": studies,
        "funds": funds,
    }


def to_pyg(records: Dict[str, List[Dict[str, Any]]]) -> Any:
    """The part of the records the model learns from, as a PyG HeteroData object:
    site -conducts-> trial -studies-> indication."""
    import torch
    from torch_geometric.data import HeteroData

    index = {
        name: {r["_key"]: i for i, r in enumerate(records[name])}
        for name in ("site", "trial", "indication")
    }
    data = HeteroData()
    data["site"].x = torch.tensor(
        [s["features"] for s in records["site"]], dtype=torch.float
    )
    data["site"].y = torch.tensor(
        [s["area"] for s in records["site"]], dtype=torch.long
    )
    data["trial"].num_nodes = len(records["trial"])
    data["indication"].num_nodes = len(records["indication"])

    def edges(name: str, src: str, dst: str) -> Any:
        pairs = [
            (index[src][e["_from"].split("/")[1]], index[dst][e["_to"].split("/")[1]])
            for e in records[name]
        ]
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    data["site", "conducts", "trial"].edge_index = edges("conducts", "site", "trial")
    data["trial", "studies", "indication"].edge_index = edges(
        "studies", "trial", "indication"
    )
    return data
