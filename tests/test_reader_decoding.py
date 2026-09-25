import math

import numpy as np

from kazrag.reader import Span, best_span


def test_best_span_respects_order_and_context():
    start = np.array([9.0, 0.0, 1.0, 5.0, 0.0, 0.0])
    end = np.array([9.0, 0.0, 6.0, 0.0, 2.0, 0.0])
    ctx = np.array([False, False, True, True, True, True])  # index 0 is CLS, 1 is question
    # Best independent argmaxes would be start=3, end=2 (invalid); the valid best is 3..4.
    s, e, score = best_span(start, end, ctx)
    assert (s, e) == (3, 4)
    assert score == 7.0


def test_best_span_limits_length():
    start = np.array([0.0, 5.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, 0.0, 0.0, 5.0])
    ctx = np.ones(5, dtype=bool)
    s, e, _ = best_span(start, end, ctx, max_answer_tokens=2)
    assert e - s + 1 <= 2


def test_best_span_empty_context():
    assert best_span(np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool)) is None


def test_confidence_is_sigmoid_of_margin():
    span = Span(0, "x", 0, 1, score=3.0, null_score=1.0)
    assert math.isclose(span.confidence, 1 / (1 + math.exp(-2.0)))
