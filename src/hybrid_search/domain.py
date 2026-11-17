import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Product(Contract):
    id: str = Field(pattern=r"^[A-Z0-9-]{1,64}$")
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    category: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    price_cents: int = Field(ge=0, strict=True)
    currency: Literal["USD"] = "USD"
    in_stock: bool

    def document(self) -> str:
        return f"{self.id}. {self.title}. {self.description} Category: {self.category}."


class Filters(Contract):
    category: str | None = Field(default=None, pattern=r"^[a-z0-9_-]{1,40}$")
    min_price_cents: int | None = Field(default=None, ge=0, strict=True)
    max_price_cents: int | None = Field(default=None, ge=0, strict=True)
    in_stock: bool | None = None

    @model_validator(mode="after")
    def price_order(self):
        if (
            self.min_price_cents is not None
            and self.max_price_cents is not None
            and self.min_price_cents > self.max_price_cents
        ):
            raise ValueError("min_price_cents must not exceed max_price_cents")
        return self

    def allows(self, product: Product) -> bool:
        return (
            (self.category is None or product.category == self.category)
            and (self.min_price_cents is None or product.price_cents >= self.min_price_cents)
            and (self.max_price_cents is None or product.price_cents <= self.max_price_cents)
            and (self.in_stock is None or product.in_stock == self.in_stock)
        )


Mode = Literal["fts", "dense", "hybrid", "rerank"]


class SearchRequest(Contract):
    query: str = Field(min_length=1, max_length=300)
    mode: Mode = "rerank"
    k: int = Field(default=5, ge=1, le=20, strict=True)
    candidate_limit: int = Field(default=30, ge=1, le=100, strict=True)
    filters: Filters = Field(default_factory=Filters)

    @model_validator(mode="after")
    def enough_candidates(self):
        if self.candidate_limit < self.k:
            raise ValueError("candidate_limit must be at least k")
        return self


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_catalog(raw: list[dict]) -> list[Product]:
    products = [Product.model_validate(p) for p in raw]
    if not products:
        raise ValueError("refusing to publish an empty catalog")
    if len({p.id for p in products}) != len(products):
        raise ValueError("duplicate product ID")
    return sorted(products, key=lambda p: p.id)
