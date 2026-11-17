# Running and updating the demo

```sh
docker compose up --build --detach --wait --wait-timeout 600
curl http://127.0.0.1:8080/health
```

The first start downloads pinned E5 and BGE ONNX exports into a named model cache.
Allow several minutes and reserve several gigabytes for dependencies, model cache
and images. No GPU or paid API is required. Docker builds locked Python
dependencies; inference uses two CPU threads by default. PostgreSQL starts first,
the one-shot indexer commits a complete catalog, and the API becomes ready after
warming both models. Postgres is not published to the host; the API is published
only on loopback. Open `http://127.0.0.1:8080/docs` for the interactive API.

The disposable database password in Compose is public demo configuration, not a
production credential. The application runs as UID 10001 with a read-only root
filesystem. `postgres-data` stores catalog rows/vectors; `model-cache` stores model
artifacts. Different model revisions require reindexing. A mismatched signature
returns 503 rather than silently mixing embedding spaces.

```sh
# Re-run indexing against the packaged catalog; unchanged vectors are reused.
docker compose run --rm index
# Stop the demo and keep both volumes.
docker compose down
```

`docker compose down --volumes` deletes this project's stored demo data and model
cache. For a modified catalog, rebuild the image and rerun the indexer. Absent IDs
are deleted when publishing a new snapshot. Back up PostgreSQL before replacing
non-disposable data; there is no public indexing endpoint. API callers can only
read. The schema is this project's initial version, not a production migration
framework. Read-only search uses a repeatable-read snapshot and five-second SQL
statement timeouts. CPU inference is serialized and at most four API requests are
admitted; this is a small local demonstration, not a horizontally scaled service.

## Reproduce verification

```sh
uv sync --frozen
uv run ruff check src tests scripts
uv run pytest -q                 # database cases skip without TEST_DATABASE_URL
python3 scripts/compose_smoke.py # complete real-model scenario in a unique project
```

Set `TEST_DATABASE_URL` to an **exclusively disposable** database to run the
PostgreSQL tests: the fixtures truncate the demo's tables. CI always supplies one
and runs all tests. Fixture models in those tests verify contracts, filtering,
atomicity and snapshots; the separate Compose job runs the actual ONNX models.
The Compose smoke checks all four modes, filters, invalid input, unchanged-index
reuse and persisted results after restart. It removes only its unique project's
containers/volumes. Images remain cached.

For a custom TLS proxy, a trusted CA bundle can be mounted temporarily during the
build. Download models on the host with that CA and reuse the populated cache
read-only/offline during the container check:

```sh
HF_HOME=/path/to/model-cache REQUESTS_CA_BUNDLE=/path/to/trusted-ca.pem \
  HF_HUB_DISABLE_XET=1 uv run hybrid-search download-models
python3 scripts/compose_smoke.py --build-ca /path/to/trusted-ca.pem \
  --model-cache /path/to/model-cache
```

Neither custom CA contents nor a developer's model-cache path are baked into the
image. TLS verification is never disabled. The optional offline cache must
contain both exact pinned model revisions; a missing artifact fails explicitly.
Use a dedicated public-model cache readable by container UID 10001.
