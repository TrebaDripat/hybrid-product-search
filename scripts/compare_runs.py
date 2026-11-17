"""Compare two paired evaluation reports using the saved query measurements."""

import argparse
import json
import math
import statistics
from pathlib import Path

MODES = ("fts", "dense", "hybrid", "rerank")


def indexed_samples(report):
    samples = report["samples"]
    rows = {(s["mode"], s["query_id"], s["iteration"]): s for s in samples}
    queries = {s["query_id"] for s in samples}
    expected = {
        (mode, query, iteration)
        for mode in MODES
        for query in queries
        for iteration in range(1, report["passes"] + 1)
    }
    if not samples or len(rows) != len(samples) or rows.keys() != expected:
        raise ValueError("missing or duplicate query, mode or pass")
    return rows


def compare(before, after):
    for field in (
        "dataset_sha256",
        "catalog_fingerprint",
        "split",
        "k",
        "candidate_limit",
        "rrf_constant",
        "models",
        "passes",
        "environment",
    ):
        if before[field] != after[field]:
            raise ValueError(f"reports differ in {field}")
    left, right = indexed_samples(before), indexed_samples(after)
    if left.keys() != right.keys() or any(
        a["relevant_count"] != right[key]["relevant_count"] for key, a in left.items()
    ):
        raise ValueError("query IDs or counts of relevant products differ between runs")
    k = before["k"]
    lines = [
        f"| Mode | Run | nDCG@{k} | MRR@{k} | Recall@{k} | p50 ms | p95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        for name, rows in (("before", left), ("after", right)):
            samples = [s for key, s in rows.items() if key[0] == mode]
            quality = [
                statistics.mean(s["metrics"][metric] for s in samples)
                for metric in ("ndcg", "reciprocal_rank", "recall")
            ]
            latencies = sorted(s["elapsed_ms"] for s in samples)
            p50, p95 = [latencies[math.ceil(len(latencies) * p) - 1] for p in (0.50, 0.95)]
            scores = " | ".join(f"{value:.4f}" for value in quality)
            lines.append(f"| {mode} | {name} | {scores} | {p50:.3f} | {p95:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        print(compare(json.loads(args.before.read_text()), json.loads(args.after.read_text())))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Cannot compare runs: {exc}\n")
