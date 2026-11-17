import hashlib
import json
import math
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from hybrid_search.domain import SearchRequest, fingerprint, parse_catalog
from hybrid_search.models import SPECS

MODES = ("fts", "dense", "hybrid", "rerank")


def metrics(ranking: list[str], relevance: dict[str, int], k: int) -> dict[str, float]:
    if k < 1 or len(set(ranking)) != len(ranking):
        raise ValueError("k must be positive and rankings must not contain duplicates")
    if any(type(grade) is not int or not 0 <= grade <= 3 for grade in relevance.values()):
        raise ValueError("relevance grades must be integers from 0 through 3")
    grades = [relevance.get(identifier, 0) for identifier in ranking[:k]]
    dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    relevant = sum(grade > 0 for grade in relevance.values())
    return {
        "ndcg": dcg / idcg if idcg else 0.0,
        "reciprocal_rank": next((1 / rank for rank, grade in enumerate(grades, 1) if grade), 0.0),
        "recall": sum(grade > 0 for grade in grades) / relevant if relevant else 0.0,
    }


def percentile(values, percent):
    return sorted(values)[math.ceil(len(values) * percent / 100) - 1]


def dataset_info(manifest_path, hashes):
    if manifest_path is None:
        return {"name": "Custom catalog", "synthetic": None}
    manifest = json.loads(manifest_path.read_text())
    if manifest["sha256"] != hashes:
        raise ValueError("dataset manifest hashes do not match the catalog and queries")
    return manifest


def evaluate(
    service,
    catalog_path: Path,
    queries_path: Path,
    split: str,
    k=5,
    candidates=30,
    *,
    passes=1,
    manifest_path=None,
    rerank_limit=None,
):
    if not 1 <= passes <= 20:
        raise ValueError("passes must be between 1 and 20")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (catalog_path, queries_path)
    }
    dataset = dataset_info(manifest_path, hashes)
    products = parse_catalog(json.loads(catalog_path.read_text()))
    by_id = {p.id: p for p in products}
    expected = fingerprint([p.model_dump() for p in products])
    queries = json.loads(queries_path.read_text())
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("duplicate query ID")
    selected = [q for q in queries if q["split"] == split]
    if not selected:
        raise ValueError("requested split is empty")
    for q in selected:
        request = SearchRequest(
            query=q["query"],
            filters=q["filters"],
            k=k,
            candidate_limit=candidates,
            rerank_limit=rerank_limit,
        )
        metrics([], q["relevance"], k)
        if any(
            identifier not in by_id or not request.filters.allows(by_id[identifier])
            for identifier in q["relevance"]
        ):
            raise ValueError("judgment references an absent or filtered-out product")
    service.models.warmup()
    samples = []
    for iteration in range(passes):
        for index, q in enumerate(selected):
            offset = (index + iteration) % len(MODES)
            for mode in MODES[offset:] + MODES[:offset]:
                response = service.search(
                    SearchRequest(
                        query=q["query"],
                        filters=q["filters"],
                        k=k,
                        candidate_limit=candidates,
                        rerank_limit=rerank_limit if mode == "rerank" else None,
                        mode=mode,
                    )
                )
                if response.catalog_fingerprint != expected:
                    raise ValueError(
                        "evaluation catalog differs from indexed snapshot; reindex first"
                    )
                ranking = [hit.product.id for hit in response.hits]
                samples.append(
                    dict(
                        iteration=iteration + 1,
                        query_id=q["id"],
                        mode=mode,
                        ranking=ranking,
                        relevant_count=sum(g > 0 for g in q["relevance"].values()),
                        metrics=metrics(ranking, q["relevance"], k),
                        elapsed_ms=response.elapsed_ms,
                        reranked_count=response.reranked_count,
                        timings_ms=response.timings_ms.model_dump(),
                    )
                )
    aggregates = {}
    for mode in MODES:
        mode_samples = [s for s in samples if s["mode"] == mode]
        aggregates[mode] = {
            **{
                name: statistics.mean(s["metrics"][name] for s in mode_samples)
                for name in ("ndcg", "reciprocal_rank", "recall")
            },
            "latency_p50_ms": percentile([s["elapsed_ms"] for s in mode_samples], 50),
            "latency_p95_ms": percentile([s["elapsed_ms"] for s in mode_samples], 95),
            "queries": len(selected),
            "measurements": len(mode_samples),
            "queries_without_relevant_products": len(
                {s["query_id"] for s in mode_samples if s["relevant_count"] == 0}
            ),
        }
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnx_threads": service.models.threads,
    }
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            environment[name] = path.read_text().strip()
    source = {}
    try:
        source = {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        source = {"git_revision": "unavailable in this deployment"}
    return dict(
        report_version=2,
        dataset=dataset,
        synthetic=dataset["synthetic"],
        passes=passes,
        created_at_utc=datetime.now(UTC).isoformat(),
        split=split,
        k=k,
        candidate_limit=candidates,
        rerank_depth=SearchRequest(
            query="budget", k=k, candidate_limit=candidates, rerank_limit=rerank_limit
        ).rerank_depth,
        rrf_constant=60,
        catalog_fingerprint=expected,
        dataset_sha256=hashes,
        models=SPECS,
        environment=environment,
        source=source,
        methodology="All selected queries included. Zero-relevance queries score zero. "
        "Sequential warmed CPU inference. Mode order rotates across queries and passes. "
        "Latency includes query encoding, retrieval, fusion and any reranking.",
        aggregates=aggregates,
        samples=samples,
    )
