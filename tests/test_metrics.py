import math

import pytest

from kazrag import metrics


def test_normalize_strips_punctuation_case_and_space():
    assert metrics.normalize_answer("  Ұлтабарға! ") == "ұлтабарға"
    assert metrics.normalize_answer("1991 ж.,  16 желтоқсан") == "1991 ж 16 желтоқсан"


def test_exact_match_takes_best_gold():
    assert metrics.exact_match("ұлтабарға", ["Асқазанға", "Ұлтабарға."]) == 1.0
    assert metrics.exact_match("ұлтабар", ["Ұлтабарға"]) == 0.0


def test_f1_partial_overlap():
    # pred 2 tokens, gold 3 tokens, 2 shared -> P=1, R=2/3
    assert metrics.token_f1("Абай Құнанбайұлы", "Абай Құнанбайұлы ақын") == pytest.approx(0.8)
    assert metrics.max_f1("", ["x"]) == 0.0
    assert metrics.token_f1("", "") == 1.0


def test_ranking_metrics():
    ranked = ["d3", "d1", "d9", "d2"]
    rel = {"d1", "d2"}
    assert metrics.mrr_at_k(ranked, rel, 10) == 0.5
    assert metrics.hit_at_k(ranked, rel, 1) == 0.0
    assert metrics.hit_at_k(ranked, rel, 2) == 1.0
    assert metrics.recall_at_k(ranked, rel, 2) == 0.5
    assert metrics.recall_at_k(ranked, rel, 4) == 1.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert metrics.ndcg_at_k(ranked, rel, 10) == pytest.approx(dcg / idcg)
    assert metrics.ndcg_at_k(ranked, set(), 10) == 0.0
