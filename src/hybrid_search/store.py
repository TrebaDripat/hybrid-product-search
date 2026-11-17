import re
from importlib.resources import files

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from hybrid_search.domain import Product, fingerprint

FIELDS = "id,title,description,category,price_cents,currency,in_stock"
INDEX_LOCK = 740551
WEB_OPERATORS = re.compile(r'"|(?:^|\s)-\S|\bOR\b', re.IGNORECASE)


class CatalogNotReady(RuntimeError):
    pass


def validate_vectors(values, count):
    vectors = np.asarray(values, dtype=np.float32)
    if (
        vectors.shape != (count, 384)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    ):
        raise ValueError("expected finite, normalized 384-dimensional E5 vectors")
    return vectors


class Store:
    def __init__(self, dsn: str):
        self.pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=4,
            timeout=5,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=register_vector,
        )
        self.pool.open(wait=True, timeout=10)

    @staticmethod
    def initialize(dsn: str):
        with psycopg.connect(dsn) as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (INDEX_LOCK,))
            connection.execute(files("hybrid_search").joinpath("schema.sql").read_text())

    def close(self):
        self.pool.close()

    def state(self, signature):
        with self.pool.connection() as connection:
            return self._state(connection, signature)

    @staticmethod
    def _state(connection, signature):
        state = connection.execute("SELECT * FROM catalog_state WHERE singleton").fetchone()
        if state is None:
            raise CatalogNotReady("index the catalog before serving searches")
        if state["embedding_signature"] != signature:
            raise CatalogNotReady("embedding model changed; reindex the catalog")
        state.pop("singleton")
        return state

    def index(self, products: list[Product], models):
        if not products or len({p.id for p in products}) != len(products):
            raise ValueError("catalog must be nonempty and product IDs unique")
        products = sorted(products, key=lambda p: p.id)
        catalog_hash = fingerprint([p.model_dump() for p in products])
        with self.pool.connection() as connection, connection.transaction():
            # Serialize complete snapshots, including embedding generation. Readers
            # keep seeing the previous committed catalog while indexing runs.
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (INDEX_LOCK,))
            old = {
                row["id"]: row
                for row in connection.execute(
                    "SELECT id,embedding_hash,content_hash,embedding FROM products"
                )
            }
            embedding_hashes = {
                p.id: fingerprint([models.signature, p.document()]) for p in products
            }
            changed = [
                p
                for p in products
                if old.get(p.id, {}).get("embedding_hash") != embedding_hashes[p.id]
            ]
            vectors = validate_vectors(
                models.passages([p.document() for p in changed]), len(changed)
            )
            fresh = dict(zip((p.id for p in changed), vectors, strict=True))
            updated = 0
            for p in products:
                content_hash = fingerprint(p.model_dump())
                if p.id not in fresh and old[p.id]["content_hash"] == content_hash:
                    continue
                vector = fresh[p.id] if p.id in fresh else old[p.id]["embedding"]
                connection.execute(
                    """INSERT INTO products(id,title,description,category,price_cents,currency,
                       in_stock,embedding,embedding_hash,content_hash)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                       description=excluded.description,category=excluded.category,
                       price_cents=excluded.price_cents,currency=excluded.currency,
                       in_stock=excluded.in_stock,embedding=excluded.embedding,
                       embedding_hash=excluded.embedding_hash,content_hash=excluded.content_hash""",
                    (
                        p.id,
                        p.title,
                        p.description,
                        p.category,
                        p.price_cents,
                        p.currency,
                        p.in_stock,
                        vector,
                        embedding_hashes[p.id],
                        content_hash,
                    ),
                )
                updated += 1
            deleted = connection.execute(
                "DELETE FROM products WHERE NOT (id = ANY(%s))", ([p.id for p in products],)
            ).rowcount
            connection.execute(
                """INSERT INTO catalog_state(singleton,embedding_signature,catalog_fingerprint,
                   product_count) VALUES(true,%s,%s,%s) ON CONFLICT(singleton) DO UPDATE SET
                   embedding_signature=excluded.embedding_signature,
                   catalog_fingerprint=excluded.catalog_fingerprint,
                   product_count=excluded.product_count,indexed_at=now()""",
                (models.signature, catalog_hash, len(products)),
            )
        return dict(
            products=len(products),
            embedded=len(changed),
            updated=updated,
            deleted=deleted,
            catalog_fingerprint=catalog_hash,
        )

    def candidates(self, request, vector, signature):
        conditions, params = [], {"query": request.query, "limit": request.candidate_limit}
        for field, operator, column in (
            ("category", "=", "category"),
            ("in_stock", "=", "in_stock"),
            ("min_price_cents", ">=", "price_cents"),
            ("max_price_cents", "<=", "price_cents"),
        ):
            value = getattr(request.filters, field)
            if value is not None:
                conditions.append(f"{column} {operator} %({field})s")
                params[field] = value
        where = " AND ".join(conditions) or "true"
        lexical, dense = [], []
        with self.pool.connection() as connection, connection.transaction():
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            connection.execute("SET LOCAL statement_timeout = '5s'")
            connection.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
            state = self._state(connection, signature)
            if request.mode != "dense":
                params["plain"] = not WEB_OPERATORS.search(request.query)
                lexical = connection.execute(
                    f"""WITH input AS (
                        SELECT websearch_to_tsquery('english', %(query)s) AS strict,
                               plainto_tsquery('english', %(query)s) AS all_terms,
                               tsvector_to_array(to_tsvector('english', %(query)s)) AS terms
                    ), q AS (
                        SELECT strict, terms, CASE WHEN %(plain)s THEN
                            replace(all_terms::text, ' & ', ' | ')::tsquery
                            ELSE strict END AS matching FROM input
                    ) SELECT {FIELDS}, ts_rank_cd(searchable, q.matching, 32) AS score,
                        CASE WHEN id=upper(%(query)s) OR searchable @@ q.strict THEN 1.0
                            ELSE (SELECT count(*)::float8 FROM unnest(q.terms) AS t(term)
                                  WHERE t.term = ANY(tsvector_to_array(searchable)))
                                 / greatest(cardinality(q.terms), 1) END AS lexical_weight
                    FROM products CROSS JOIN q
                    WHERE {where} AND (id=upper(%(query)s) OR searchable @@ q.matching)
                    ORDER BY (id=upper(%(query)s)) DESC, (searchable @@ q.strict) DESC,
                             lexical_weight DESC, score DESC, id LIMIT %(limit)s""",
                    params,
                ).fetchall()
            if request.mode != "fts":
                params["vector"] = validate_vectors([vector], 1)[0]
                dense = connection.execute(
                    f"""SELECT {FIELDS},1-(embedding <=> %(vector)s) AS score FROM products
                    WHERE {where} ORDER BY embedding <=> %(vector)s,id LIMIT %(limit)s""",
                    params,
                ).fetchall()
        return lexical, dense, state
