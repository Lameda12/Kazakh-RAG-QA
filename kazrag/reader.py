"""Extractive reader with sliding windows and SQuAD2-style null scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kazrag.device import default_device

DEFAULT_READER = "deepset/xlm-roberta-large-squad2"


@dataclass(frozen=True, slots=True)
class Span:
    passage_idx: int
    text: str
    start_char: int
    end_char: int
    score: float  # start_logit + end_logit of the span
    null_score: float  # start_logit + end_logit of the [CLS] "no answer" position

    @property
    def confidence(self) -> float:
        """sigmoid(span - null): the model's odds that this span beats "no answer".

        A heuristic, not a calibrated probability. It is comparable across
        passages, unlike raw logits.
        """
        return 1.0 / (1.0 + math.exp(-(self.score - self.null_score)))


def best_span(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    context_mask: np.ndarray,
    max_answer_tokens: int = 30,
    n_best: int = 20,
) -> tuple[int, int, float] | None:
    """Best (start, end, score) with start <= end < start + max_answer_tokens, both in the context.

    Returns None when no valid span exists (e.g. an all-question window).
    """
    ctx = np.flatnonzero(context_mask)
    if ctx.size == 0:
        return None
    starts = ctx[np.argsort(start_logits[ctx])[::-1][:n_best]]
    ends = ctx[np.argsort(end_logits[ctx])[::-1][:n_best]]
    best = None
    for s in starts:
        for e in ends:
            if e < s or e - s + 1 > max_answer_tokens:
                continue
            score = float(start_logits[s] + end_logits[e])
            if best is None or score > best[2]:
                best = (int(s), int(e), score)
    return best


class ExtractiveReader:
    def __init__(
        self,
        model_name: str = DEFAULT_READER,
        device: str | None = None,
        max_length: int = 384,
        stride: int = 128,
        max_answer_tokens: int = 30,
        batch_size: int = 8,
    ):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise RuntimeError(f"{model_name} needs a fast tokenizer for offset mapping")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.max_answer_tokens = max_answer_tokens
        self.batch_size = batch_size

    def read(self, question: str, contexts: Sequence[str]) -> list[Span | None]:
        """Best span per context. Long contexts are split into overlapping windows."""
        if not contexts:
            return []
        question = self._clip_question(question)
        enc = self.tokenizer(
            [question] * len(contexts),
            list(contexts),
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="longest",
            return_tensors="np",
        )
        sample_map = enc.pop("overflow_to_sample_mapping")
        offsets = enc.pop("offset_mapping")
        model_inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

        start_all, end_all = [], []
        with self.torch.inference_mode():
            for lo in range(0, len(sample_map), self.batch_size):
                batch = {
                    k: self.torch.as_tensor(v[lo : lo + self.batch_size], device=self.device)
                    for k, v in model_inputs.items()
                }
                out = self.model(**batch)
                start_all.append(out.start_logits.float().cpu().numpy())
                end_all.append(out.end_logits.float().cpu().numpy())
        start_logits, end_logits = np.concatenate(start_all), np.concatenate(end_all)

        best: list[Span | None] = [None] * len(contexts)
        null: list[float] = [math.inf] * len(contexts)
        for w, p_idx in enumerate(sample_map):
            p_idx = int(p_idx)
            context_mask = np.array([sid == 1 for sid in enc.sequence_ids(w)])
            # The CLS position (index 0) is the SQuAD2 "no answer" slot.
            null[p_idx] = min(null[p_idx], float(start_logits[w, 0] + end_logits[w, 0]))
            found = best_span(start_logits[w], end_logits[w], context_mask, self.max_answer_tokens)
            if found is None:
                continue
            s, e, score = found
            if best[p_idx] is None or score > best[p_idx].score:
                start_char, end_char = int(offsets[w, s, 0]), int(offsets[w, e, 1])
                raw = contexts[p_idx][start_char:end_char]
                start_char += len(raw) - len(raw.lstrip())  # sentencepiece offsets can include the leading space
                end_char -= len(raw) - len(raw.rstrip())
                text = contexts[p_idx][start_char:end_char]
                best[p_idx] = Span(p_idx, text, start_char, end_char, score, 0.0)

        return [
            Span(s.passage_idx, s.text, s.start_char, s.end_char, s.score, null[i]) if s else None
            for i, s in enumerate(best)
        ]

    def _clip_question(self, question: str) -> str:
        return clip_question(self.tokenizer, question, self.max_length)


def clip_question(tokenizer, question: str, max_length: int) -> str:
    """only_second truncation cannot shorten the question, so cap it here (SQuAD-style max_query_length).

    A third of the window keeps room for the context plus the stride overlap.
    """
    max_tokens = min(64, max_length // 3)
    ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    return question if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])
