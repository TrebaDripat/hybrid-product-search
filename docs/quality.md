# Paired search measurements

The lexical change recovered results on eight previously empty WANDS queries. The hybrid ranker improved from 0.7794 to 0.8041 nDCG@5. BGE at a depth of 30 achieved the highest nDCG@5 on the external pool. Scoring eight pairs gave a lower p95 at a measurable cost in ordering quality.

## Wayfair WANDS

The same 24 queries searched the same pool of 875 products. Products and queries were sampled by identifier independently of grade, then every available upstream judgment for each query within the pool was retained. Every mode used up to 30 candidates per branch and returned five hits. [Pool selection and license](wands.md)

| Mode | nDCG@5 | MRR@5 | Recall@5 | CPU p95 |
| --- | ---: | ---: | ---: | ---: |
| Original FTS | 0.3355 | 0.5833 | 0.0395 | 13.83 ms |
| Current FTS | 0.7331 | 0.8938 | 0.1384 | 15.68 ms |
| Original E5 | 0.7768 | 0.9583 | 0.1394 | 25.31 ms |
| Current E5 | 0.7768 | 0.9583 | 0.1394 | 14.44 ms |
| Original hybrid | 0.7794 | 0.9375 | 0.1355 | 17.62 ms |
| Current hybrid | 0.8041 | 0.9583 | 0.1434 | 22.09 ms |
| Original BGE, up to 30 | 0.8079 | 0.9514 | 0.1485 | 6.21 s |
| Current BGE, up to 8 (default) | 0.7925 | 0.9583 | 0.1470 | 1.84 s |
| Current BGE, up to 30 | 0.8266 | 0.9514 | 0.1556 | 5.35 s |

FTS improved nDCG@5 on 19 queries, tied on three and regressed on two. For `town & country living curtains` (`WQ-467`), its score went from an empty ranking to 1.0000. For `monthly calendar` (`WQ-63`), the new ordering dropped from 1.0000 to 0.7995. Hybrid improved on seven queries, tied on eleven and regressed on six.

The candidate budget mattered for `emma headboard` (`WQ-481`). Increasing the current BGE depth from eight to 30 raised that query’s score from 0.2140 to 0.8688. On `queen wingback chair` (`WQ-143`), the same change lowered it from 0.5148 to 0.3156. The current hybrid mode returned its five hits in 22 ms at p95 and beat the default eight-pair BGE on mean nDCG. [Original per-query report](../reports/quality-v2/before-wands.json) · [Current default](../reports/quality-v2/after-wands.json) · [Current full-depth control](../reports/quality-v2/full-wands.json)

## Synthetic demonstration catalog

The 18 held-out queries searched 30 English products. The pool includes metadata filters and one query without a relevant product. Three passes per method gave 54 latency observations per mode.

| Mode | Original nDCG@5 | Current nDCG@5 | Original p95 | Current p95 |
| --- | ---: | ---: | ---: | ---: |
| FTS | 0.1111 | 0.8889 | 3.00 ms | 4.37 ms |
| E5 | 0.9429 | 0.9429 | 10.28 ms | 7.91 ms |
| Hybrid | 0.9429 | 0.9444 | 9.39 ms | 8.52 ms |
| BGE, default budget | 0.9417 | 0.9417 | 739.93 ms | 192.10 ms |

Partial matching reduced empty FTS results from 16 queries to two. Hybrid reached the 17/18 ceiling for this fixture. BGE kept the same mean quality after reducing the default budget. [Original per-query report](../reports/quality-v2/before-demo.json) · [Current per-query report](../reports/quality-v2/after-demo.json)

## Reproduce and interpret

Baseline `86585e3` adds the evaluation protocol to the original search implementation. Current results were measured from `b20411c`. Both checkouts were clean and used the same quantized E5/BGE revisions, data hashes and PostgreSQL snapshot for each dataset. Both used two ONNX threads on a Linux host with a four-CPU cgroup quota and 16 GiB memory limit. The WANDS runs used one sequential, warmed pass per mode. The p95 there is the 23rd fastest observation out of 24. E5 retrieval was unchanged and produced identical quality. Its timing variation illustrates the uncertainty of short single-host runs.

The WANDS pool is a subset of the full 42,994-product collection. Unjudged products receive zero gain, and Recall@5 uses relevant products inside the sample. The pool, small query count and single-user timing limit comparisons to these runs. The retained data includes all five runs with rankings and full precision.

```sh
# Recalculate the table from the retained samples with just Python.
python3 scripts/compare_runs.py reports/quality-v2/before-wands.json reports/quality-v2/after-wands.json
python3 scripts/compare_runs.py reports/quality-v2/after-wands.json reports/quality-v2/full-wands.json
```

To rerun inference, generate and index the fixed WANDS pool in a separate database. Run the evaluation at its default budget, then repeat with `--rerank-limit 30` and a different output file. The synthetic baseline can be rerun from `86585e3` with `--passes 3`. [Dataset commands](wands.md) · [Metric definitions](evaluation.md) · [Original demonstration report](results.md)
