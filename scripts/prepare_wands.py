"""Build the fixed WANDS evaluation pool from verified upstream TSVs."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

REVISION = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
UPSTREAM = f"https://raw.githubusercontent.com/wayfair/WANDS/{REVISION}/dataset"
DIGESTS = {
    "product.csv": "d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6",
    "query.csv": "63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95",
    "label.csv": "c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef",
}
GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ensure_sources(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in DIGESTS.items():
        path = cache / name
        if path.exists() and digest(path) == expected:
            continue
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            partial = Path(tmp) / name
            with (
                urllib.request.urlopen(f"{UPSTREAM}/{name}", timeout=90) as src,
                partial.open("wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            if digest(partial) != expected:
                raise ValueError(f"unexpected SHA-256 for upstream {name}")
            partial.replace(path)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source, delimiter="\t")


def select(ids, count, namespace):
    def priority(identifier):
        return hashlib.sha256(f"wands-v1/{namespace}/{identifier}".encode()).digest()

    return sorted(ids, key=priority)[:count]


def product(row):
    title = " ".join(row["product_name"].split())[:200]
    description = " ".join(row["product_description"].split()) or title
    category = re.sub(r"[^a-z0-9]+", "-", row["product_class"].lower()).strip("-") or "home"
    return dict(
        id="W-" + row["product_id"],
        title=title,
        description=description[:2000],
        category=category[:40],
        price_cents=0,
        currency="USD",
        in_stock=True,
    )


def prepare(cache, output):
    ensure_sources(cache)
    queries = {row["query_id"]: row["query"] for row in read_rows(cache / "query.csv")}
    query_ids = select(queries, 24, "query")
    judgments = {qid: {} for qid in query_ids}
    for row in read_rows(cache / "label.csv"):
        if row["query_id"] in judgments:
            bucket = judgments[row["query_id"]]
            pid, grade = row["product_id"], GRADES[row["label"]]
            if pid in bucket and bucket[pid] != grade:
                raise ValueError("conflicting upstream relevance judgments")
            bucket[pid] = grade
    catalog_ids = (row["product_id"] for row in read_rows(cache / "product.csv"))
    pool = set(select(catalog_ids, 128, "catalog"))
    for qid in query_ids:
        pool.update(select(judgments[qid], 32, f"judged/{qid}"))
    catalog = [
        product(row) for row in read_rows(cache / "product.csv") if row["product_id"] in pool
    ]
    if len(catalog) != len(pool):
        raise ValueError("missing or duplicate product in source catalog")
    queries_out = [
        dict(
            id=f"WQ-{qid}",
            split="evaluation",
            query=queries[qid],
            filters={},
            relevance={f"W-{pid}": grade for pid, grade in judgments[qid].items() if pid in pool},
        )
        for qid in query_ids
    ]
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        name="Wayfair WANDS fixed pool v1",
        synthetic=False,
        upstream=f"https://github.com/wayfair/WANDS/tree/{REVISION}",
        products=len(catalog),
        queries=len(query_ids),
        judged_pool_per_query=32,
        catalog_distractors_sampled=128,
        sha256={},
    )
    for name, rows in (("catalog.json", catalog), ("queries.json", queries_out)):
        path = output / name
        path.write_text(json.dumps(sorted(rows, key=lambda row: row["id"]), sort_keys=True) + "\n")
        manifest["sha256"][name] = digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/wands/upstream"))
    parser.add_argument("--output", type=Path, default=Path(".cache/wands/fixed-v1"))
    options = parser.parse_args()
    print(json.dumps(prepare(options.cache, options.output), indent=2))
