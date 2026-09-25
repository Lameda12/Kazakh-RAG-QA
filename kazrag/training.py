"""Feature building and scoring for fine-tuning the extractive reader."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from kazrag import metrics
from kazrag.reader import ExtractiveReader, clip_question


def build_features(tokenizer, rows: Sequence[dict], max_length: int = 384, stride: int = 128) -> list[dict]:
    """SQuAD-style training features: one per sliding window, labelled with token start/end positions.

    Windows that do not contain the (first) gold answer are labelled with the
    CLS position, the SQuAD2 "no answer" target, so the model also learns the
    null score the reader uses for confidence.
    """
    features: list[dict] = []
    for lo in range(0, len(rows), 1000):
        batch = rows[lo : lo + 1000]
        # Same "title. context" text the reader sees at inference; offsets shift by the prefix.
        prefixes = [f"{r['title']}. " if r.get("title") else "" for r in batch]
        enc = tokenizer(
            [clip_question(tokenizer, r["question"], max_length) for r in batch],
            [pre + r["context"] for pre, r in zip(prefixes, batch, strict=True)],
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        for i, offsets in enumerate(enc["offset_mapping"]):
            sample = enc["overflow_to_sample_mapping"][i]
            row = batch[sample]
            start_char = len(prefixes[sample]) + row["answers"]["answer_start"][0]
            end_char = start_char + len(row["answers"]["text"][0])
            seq_ids = enc.sequence_ids(i)
            ctx = [t for t, sid in enumerate(seq_ids) if sid == 1]
            start_pos = end_pos = 0
            if ctx and offsets[ctx[0]][0] <= start_char and offsets[ctx[-1]][1] >= end_char:
                start_pos = next(t for t in ctx if offsets[t][1] > start_char)
                end_pos = next(t for t in reversed(ctx) if offsets[t][0] < end_char)
            features.append(
                {
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "start_positions": start_pos,
                    "end_positions": end_pos,
                }
            )
    return features


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features: list[dict]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict:
        return self.features[i]


def score_reader(reader: ExtractiveReader, rows: Sequence[dict]) -> dict:
    """EM/F1 (0-100) of the reader on gold passages, with the title prepended as at inference time."""
    em, f1 = [], []
    for row in rows:
        context = f"{row['title']}. {row['context']}" if row.get("title") else row["context"]
        span = reader.read(row["question"], [context])[0]
        pred = span.text if span else ""
        golds = row["answers"]["text"]
        em.append(metrics.exact_match(pred, golds))
        f1.append(metrics.max_f1(pred, golds))
    n = len(rows)
    return {"n": n, "EM": 100 * sum(em) / n if n else 0.0, "F1": 100 * sum(f1) / n if n else 0.0}
