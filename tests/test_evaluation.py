import math

import pytest

from hybrid_search.evaluation import metrics


def test_metrics_match_hand_calculated_graded_example():
    actual = metrics(["irrelevant", "strong", "weak"], {"strong": 3, "weak": 1}, 3)
    dcg = 7 / math.log2(3) + 1 / math.log2(4)
    idcg = 7 + 1 / math.log2(3)
    assert actual == pytest.approx({"ndcg": dcg / idcg, "reciprocal_rank": 0.5, "recall": 1})
    assert metrics(["strong"], {"strong": 3, "weak": 1}, 1)["recall"] == 0.5


def test_metrics_keep_empty_and_unanswerable_queries_explicit():
    assert metrics([], {"a": 3}, 5) == dict(ndcg=0, reciprocal_rank=0, recall=0)
    assert metrics(["a"], {}, 5) == dict(ndcg=0, reciprocal_rank=0, recall=0)
    with pytest.raises(ValueError, match="duplicates"):
        metrics(["a", "a"], {"a": 3}, 5)
    with pytest.raises(ValueError, match="grades"):
        metrics(["a"], {"a": -1}, 5)
