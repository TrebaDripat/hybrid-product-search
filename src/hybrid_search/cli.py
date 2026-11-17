import argparse
import json
import os
from pathlib import Path

from hybrid_search.domain import SearchRequest, parse_catalog
from hybrid_search.evaluation import evaluate
from hybrid_search.models import OnnxModels
from hybrid_search.search import SearchService
from hybrid_search.store import Store


def main():
    parser = argparse.ArgumentParser(description="Synthetic E5/pgvector/BGE catalog search")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-models", help="download pinned models and verify CPU inference")
    index = commands.add_parser("index", help="atomically replace the catalog snapshot")
    index.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", choices=("fts", "dense", "hybrid", "rerank"), default="rerank")
    search.add_argument("--k", type=int, default=5)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    evaluation.add_argument("--queries", type=Path, default=Path("data/queries.json"))
    evaluation.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    evaluation.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("hybrid_search.api:create_app", factory=True, host=args.host, port=args.port)
        return
    models = OnnxModels(threads=int(os.environ.get("INFERENCE_THREADS", "2")))
    if args.command == "download-models":
        models.warmup()
        print(json.dumps({"model_inference": "passed", "embedding_signature": models.signature}))
        return
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        parser.error("set DATABASE_URL to the demo PostgreSQL database")
    products = None
    if args.command == "index":
        products = parse_catalog(json.loads(args.catalog.read_text()))
        Store.initialize(dsn)
    store = Store(dsn)
    try:
        if args.command == "index":
            result = store.index(products, models)
        elif args.command == "search":
            result = (
                SearchService(store, models)
                .search(SearchRequest(query=args.query, mode=args.mode, k=args.k))
                .model_dump()
            )
        else:
            result = evaluate(SearchService(store, models), args.catalog, args.queries, args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        store.close()
