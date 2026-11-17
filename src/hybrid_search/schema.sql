CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS products (
    id text PRIMARY KEY,
    title text NOT NULL,
    description text NOT NULL,
    category text NOT NULL,
    price_cents integer NOT NULL CHECK (price_cents >= 0),
    currency text NOT NULL CHECK (currency = 'USD'),
    in_stock boolean NOT NULL,
    embedding vector(384) NOT NULL,
    embedding_hash text NOT NULL,
    content_hash text NOT NULL,
    searchable tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', id || ' ' || title), 'A') ||
        setweight(to_tsvector('english', description || ' ' || category), 'B')
    ) STORED
);
CREATE INDEX IF NOT EXISTS products_fts ON products USING gin(searchable);
CREATE INDEX IF NOT EXISTS products_cosine ON products USING hnsw(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS products_filters ON products(category, in_stock, price_cents);
CREATE TABLE IF NOT EXISTS catalog_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    embedding_signature text NOT NULL,
    catalog_fingerprint text NOT NULL,
    product_count integer NOT NULL,
    indexed_at timestamptz NOT NULL DEFAULT now()
);
