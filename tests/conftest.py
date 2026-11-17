import hashlib
import os

import numpy as np
import pytest

from hybrid_search.store import Store


class FixtureModels:
    """Only for mechanical tests, never evaluation or a production fallback."""

    signature = "test-fixture-v1"

    def __init__(self):
        self.embedded = 0

    def passages(self, texts):
        self.embedded += len(texts)
        result = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in enumerate(texts):
            result[row, hashlib.sha256(text.encode()).digest()[0]] = 1
        return result

    def query(self, text):
        return self.passages([text])[0]

    def rerank(self, query, documents):
        return [float(len(set(query.lower().split()) & set(d.lower().split()))) for d in documents]


@pytest.fixture
def models():
    return FixtureModels()


@pytest.fixture
def store():
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set TEST_DATABASE_URL to an exclusively disposable PostgreSQL database")
    Store.initialize(dsn)
    database = Store(dsn)
    with database.pool.connection() as connection:
        connection.execute("TRUNCATE products,catalog_state")
    try:
        yield database
    finally:
        database.close()
