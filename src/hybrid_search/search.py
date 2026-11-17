import math
import time

from pydantic import BaseModel

from hybrid_search.domain import Mode, Product, SearchRequest


class Hit(BaseModel):
    product: Product
    lexical_rank: int | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_weight: float | None = None
    cosine_similarity: float | None = None
    rrf_score: float | None = None
    rerank_logit: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: Mode
    catalog_fingerprint: str
    embedding_signature: str
    candidate_count: int
    elapsed_ms: float
    hits: list[Hit]


def fuse(lexical, dense):
    hits = {}
    for branch, rows in (("lexical", lexical), ("dense", dense)):
        for rank, row in enumerate(rows, 1):
            identifier = row["id"]
            if identifier not in hits:
                product = Product.model_validate({name: row[name] for name in Product.model_fields})
                hits[identifier] = Hit(product=product, rrf_score=0)
            hit = hits[identifier]
            weight = row.get("lexical_weight", 1.0) if branch == "lexical" else 1.0
            hit.rrf_score += weight / (60 + rank)
            if branch == "lexical":
                hit.lexical_rank, hit.lexical_score = rank, float(row["score"])
                hit.lexical_weight = weight
            else:
                hit.dense_rank, hit.cosine_similarity = rank, float(row["score"])
    return sorted(hits.values(), key=lambda h: (-h.rrf_score, h.product.id))


class SearchService:
    def __init__(self, store, models):
        self.store, self.models = store, models

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        vector = None if request.mode == "fts" else self.models.query(request.query)
        lexical, dense, state = self.store.candidates(request, vector, self.models.signature)
        hits = fuse(lexical, dense)
        # For a single branch preserve its rank and do not mislabel it as RRF.
        if request.mode in ("fts", "dense"):
            hits.sort(key=lambda h: h.lexical_rank if request.mode == "fts" else h.dense_rank)
            for hit in hits:
                hit.rrf_score = None
        else:
            # A request for a complete, eligible catalog ID is navigational.
            hits.sort(key=lambda h: h.product.id != request.query.upper())
            hits = hits[: request.candidate_limit]
        if request.mode == "rerank" and hits:
            scores = self.models.rerank(request.query, [h.product.document() for h in hits])
            if len(scores) != len(hits) or not all(math.isfinite(x) for x in scores):
                raise ValueError("invalid reranker scores")
            for hit, score in zip(hits, scores, strict=True):
                hit.rerank_logit = score
            hits.sort(
                key=lambda h: (
                    h.product.id != request.query.upper(),
                    -h.rerank_logit,
                    -h.rrf_score,
                    h.product.id,
                )
            )
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            catalog_fingerprint=state["catalog_fingerprint"],
            embedding_signature=state["embedding_signature"],
            candidate_count=len(hits),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            hits=hits[: request.k],
        )
