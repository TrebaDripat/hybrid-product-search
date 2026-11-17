#!/usr/bin/env python3
"""Verify real E5/BGE inference in an isolated Docker Compose project."""

import argparse
import json
import os
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-ca", type=Path, help="optional trusted CA bundle for build downloads"
    )
    parser.add_argument(
        "--model-cache", type=Path, help="reuse a populated HF_HOME read-only/offline"
    )
    parser.add_argument("--browser", action="store_true", help="run the Playwright explorer checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = "hybrid-smoke-" + uuid.uuid4().hex[:10]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ, SEARCH_PORT=str(port))
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="hybrid-compose-") as directory:
        command = ["docker", "compose", "-p", project, "-f", str(root / "compose.yaml")]
        override = {"services": {}}
        if args.build_ca:
            ca = str(args.build_ca.resolve(strict=True))
            override["secrets"] = {"build_ca": {"file": ca}}
            for name in ("api", "index"):
                override["services"][name] = {"build": {"secrets": ["build_ca"]}}
        if args.model_cache:
            cache = str(args.model_cache.resolve(strict=True))
            for name in ("api", "index"):
                override["services"].setdefault(name, {}).update(
                    {
                        "volumes": [
                            {
                                "type": "bind",
                                "source": cache,
                                "target": "/models",
                                "read_only": True,
                            }
                        ],
                        "environment": {"HF_HUB_OFFLINE": "1"},
                    }
                )
        if override["services"]:
            path = Path(directory) / "override.json"
            path.write_text(json.dumps(override))
            command += ["-f", str(path)]

        def compose(*arguments, capture=False, timeout=900):
            return subprocess.run(
                command + list(arguments),
                cwd=root,
                env=environment,
                text=True,
                capture_output=capture,
                check=True,
                timeout=timeout,
            )

        def get(path):
            with urllib.request.urlopen(base + path, timeout=10) as response:
                return json.load(response)

        def search(payload):
            request = urllib.request.Request(
                base + "/search",
                json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)

        try:
            compose("build", "api")
            compose("up", "--detach", "--no-build", "--wait", "--wait-timeout", "600")
            health = get("/health")
            assert health["product_count"] == 30
            if args.browser:
                from check_explorer import check_explorer

                check_explorer(base)
            assert "/search" in get("/openapi.json")["paths"]
            assert get("/models")["embedding"]["dimensions"] == 384
            response = search({"query": "noise cancelling headphones", "mode": "rerank"})
            assert response["hits"][0]["product"]["id"] == "AUD-ANC"
            assert response["hits"][0]["rerank_logit"] is not None
            for mode in ("fts", "dense", "hybrid", "rerank"):
                response = search(
                    {
                        "query": "coffee",
                        "mode": mode,
                        "filters": {
                            "category": "kitchen",
                            "max_price_cents": 5000,
                            "in_stock": True,
                        },
                    }
                )
                assert response["hits"]
                assert all(
                    h["product"]["category"] == "kitchen"
                    and h["product"]["price_cents"] <= 5000
                    and h["product"]["in_stock"]
                    for h in response["hits"]
                )
            assert search({"query": "coffee", "filters": {"max_price_cents": 0}})["hits"] == []
            try:
                search({"query": " "})
                raise AssertionError("invalid input was accepted")
            except urllib.error.HTTPError as error:
                assert error.code == 422
            indexed = json.loads(
                compose("run", "--rm", "--no-deps", "-T", "index", capture=True).stdout
            )
            assert indexed["embedded"] == 0 and indexed["updated"] == 0 and indexed["deleted"] == 0
            compose("stop")
            compose("up", "--detach", "--no-build", "--wait", "--wait-timeout", "600")
            assert get("/health")["catalog_fingerprint"] == health["catalog_fingerprint"]
            assert (
                search({"query": "noise cancelling headphones"})["hits"][0]["product"]["id"]
                == "AUD-ANC"
            )
            print(
                json.dumps(
                    dict(
                        compose_smoke="passed",
                        real_models=True,
                        products=30,
                        modes_checked=4,
                        unchanged_products_reembedded=0,
                        persisted_after_restart=True,
                    ),
                    indent=2,
                )
            )
        except BaseException:
            subprocess.run(
                command + ["logs", "--no-color", "--tail", "60"], env=environment, timeout=30
            )
            raise
        finally:
            compose("down", "--volumes", "--remove-orphans", timeout=90)


if __name__ == "__main__":
    main()
