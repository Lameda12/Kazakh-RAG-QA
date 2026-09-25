"""Retrieve-then-read pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from kazrag.index import DenseRetriever, Hit
from kazrag.reader import ExtractiveReader, Span


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    confidence: float
    span: Span | None
    passage: Hit | None
    hits: list[Hit]

    @property
    def found(self) -> bool:
        return self.span is not None


class KazakhQA:
    def __init__(self, retriever: DenseRetriever, reader: ExtractiveReader):
        self.retriever = retriever
        self.reader = reader

    def answer(self, question: str, top_k: int = 5) -> Answer:
        question = question.strip()
        if not question:
            return Answer("", 0.0, None, None, [])
        hits = self.retriever.search([question], k=top_k)[0]
        return self.read(question, hits)

    def read(self, question: str, hits: list[Hit]) -> Answer:
        """Pick the span with the highest confidence across the retrieved passages.

        Confidence is sigmoid(span_logit - null_logit), which is comparable
        across passages; ties go to the higher-ranked passage. KazQAD questions
        are all answerable, so the best non-null span is always returned and
        the caller decides whether its confidence is high enough to show.
        """
        spans = self.reader.read(question, [h.content for h in hits])
        candidates = [(s.confidence, -h.rank, s, h) for s, h in zip(spans, hits, strict=True) if s and s.text]
        if not candidates:
            return Answer("", 0.0, None, None, hits)
        confidence, _, span, hit = max(candidates, key=lambda c: (c[0], c[1]))
        return Answer(span.text, confidence, span, hit, hits)
