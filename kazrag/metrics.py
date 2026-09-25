"""SQuAD-style answer metrics and ranking metrics for KazQAD."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

_PUNCT = re.compile(r"[^\w\s]|_")
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Kazakh has no articles, so SQuAD's article stripping is skipped. NFKC folds
    look-alike code points (e.g. Latin/Cyrillic compatibility forms) together.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(prediction: str, golds: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in golds))


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction: str, golds: Iterable[str]) -> float:
    return max((token_f1(prediction, g) for g in golds), default=0.0)


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top k."""
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1 if any relevant doc is in the top k. This is what an extractive reader needs."""
    return float(any(d in relevant for d in ranked[:k]))


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for rank, docid in enumerate(ranked[:k], start=1):
        if docid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
