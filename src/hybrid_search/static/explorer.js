"use strict";

const $ = (id) => document.getElementById(id);
const modes = new Map([
  ["fts", "Full text"],
  ["dense", "E5 semantic"],
  ["hybrid", "Hybrid RRF"],
  ["rerank", "BGE · up to 8"],
  ["rerank-full", "BGE · up to 30"],
]);
const stages = {
  encode: "Query encoding",
  retrieval: "Candidate retrieval",
  fusion: "Rank fusion",
  rerank: "BGE scoring",
};
const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const ms = (n) => `${n.toLocaleString("en-US", { maximumFractionDigits: 1 })} ms`;
let active;

function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

function disclosure(title, entries, description) {
  const block = element("details");
  block.append(element("summary", title));
  if (description) block.append(element("p", description));
  const list = element("dl");
  for (const [key, value] of entries) {
    if (value != null) list.append(element("dt", key), element("dd", value));
  }
  block.append(list);
  return block;
}

function panel(response, label, reference) {
  const article = element("article", undefined, "panel");
  const head = element("header", undefined, "panel-head");
  const line = element("div", undefined, "panel-line");
  line.append(element("h2", label), element("span", ms(response.elapsed_ms), "duration"));
  const count =
    `${response.candidate_count} candidates` +
    (response.reranked_count ? ` · ${response.reranked_count} scored by BGE` : "");
  head.append(
    line,
    element("p", count, "counts"),
    disclosure(
      "Stage timings",
      Object.entries(stages).map(([key, name]) => [name, ms(response.timings_ms[key])]),
    ),
  );
  article.append(head);
  const list = element("ol");
  response.hits.forEach((hit, index) => {
    const product = hit.product;
    const row = element("li");
    const number = element("span", String(index + 1).padStart(2, "0"), "rank");
    number.setAttribute("aria-hidden", "true");
    const body = element("div");
    const title = element("div", undefined, "item-head");
    title.append(
      element("h3", product.title),
      element("span", money.format(product.price_cents / 100), "price"),
    );
    const meta = element("div", undefined, "meta");
    meta.append(
      element("span", product.id),
      element("span", product.in_stock ? "In stock" : "Out of stock"),
    );
    if (reference) {
      const previous = reference.indexOf(product.id);
      const shift = previous - index;
      if (previous < 0 || shift !== 0) {
        const change = previous < 0 ? "New in top 5" : `${shift > 0 ? "↑" : "↓"}${Math.abs(shift)}`;
        const badge = element("span", change, "delta");
        badge.title =
          previous < 0
            ? "Outside the first ranking’s top five"
            : `Position ${previous + 1} in the first ranking`;
        meta.append(badge);
      }
    }
    const fmt = (value) => (value == null ? null : value.toFixed(4));
    body.append(
      title,
      meta,
      disclosure(
        "Ranking details",
        [
          ["Category", product.category],
          ["FTS position", hit.lexical_rank],
          ["FTS score", fmt(hit.lexical_score)],
          ["Lexical weight", fmt(hit.lexical_weight)],
          ["Dense position", hit.dense_rank],
          ["Cosine similarity", fmt(hit.cosine_similarity)],
          ["RRF score", fmt(hit.rrf_score)],
          ["BGE logit", fmt(hit.rerank_logit)],
        ],
        product.description,
      ),
    );
    row.append(number, body);
    list.append(row);
  });
  if (response.hits.length) article.append(list);
  else article.append(element("p", "No matching products. Adjust the query or filters.", "empty"));
  return article;
}

async function request(mode, query, filters, signal) {
  const body = { query, filters, k: 5, mode: mode === "rerank-full" ? "rerank" : mode };
  if (mode === "rerank-full") body.rerank_limit = 30;
  const response = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    if (response.status === 422)
      throw Error("Check the query, category and price, then search again.");
    if (response.status === 503)
      throw Error("Search is temporarily unavailable. Try again shortly.");
    throw Error(`Search returned HTTP ${response.status}.`);
  }
  return response.json();
}

async function compare(event) {
  event?.preventDefault();
  active?.abort();
  const controller = new AbortController();
  active = controller;
  const query = $("query").value.trim();
  const first = $("first").value;
  const second = $("second").value;
  const filters = {};
  if ($("category").value.trim()) filters.category = $("category").value.trim();
  if ($("price").value !== "") filters.max_price_cents = Math.round(Number($("price").value) * 100);
  if ($("stock").checked) filters.in_stock = true;
  const results = $("results");
  const status = $("status");
  results.setAttribute("aria-busy", "true");
  results.replaceChildren();
  status.className = "";
  status.textContent = "Searching the catalog…";
  try {
    const left = await request(first, query, filters, controller.signal);
    let right = null;
    if (second) {
      right = second === first ? left : await request(second, query, filters, controller.signal);
    }
    if (active !== controller) return;
    if (right && left.catalog_fingerprint !== right.catalog_fingerprint) {
      throw Error("The catalog changed between searches. Search again to compare one snapshot.");
    }
    results.className = right ? "" : "single";
    results.append(panel(left, modes.get(first)));
    if (right)
      results.append(
        panel(
          right,
          modes.get(second),
          left.hits.map((h) => h.product.id),
        ),
      );
    status.textContent = `Results for “${query}”`;
  } catch (error) {
    if (active !== controller) return;
    status.className = "error";
    status.textContent = error.message || "Search could not complete.";
  } finally {
    if (active === controller) results.setAttribute("aria-busy", "false");
  }
}

for (const [key, value] of modes) {
  $("first").add(new Option(value, key));
  $("second").add(new Option(value, key));
}
$("second").add(new Option("Off", ""));
$("first").value = "fts";
$("second").value = "rerank";
$("form").addEventListener("submit", compare);
fetch("/health")
  .then((r) => (r.ok ? r.json() : null))
  .then((s) => {
    if (s) $("catalog").textContent = `${s.product_count} products indexed`;
  })
  .catch(() => {});
compare();
