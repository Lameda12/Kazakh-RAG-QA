#!/usr/bin/env python3
"""Evaluate the pipeline on KazQAD.

Tasks:
  retrieval  nDCG@10, MRR@10, Recall@k, Hit@k against the official qrels
  reader     EM/F1 when the reader is given the gold passage (reading comprehension)
  odqa       EM/F1 end to end: retrieve top-k from the index, then read

Examples:
    python scripts/eval.py --index index/ --tasks retrieval
    python scripts/eval.py --index <user>/kazqad-bge-m3-index --tasks all --limit 200
    python scripts/eval.py --tasks reader --reader deepset/xlm-roberta-base-squad2

Published KazQAD baselines (test): retrieval nDCG@10 0.389 / MRR 0.382,
reader EM 38.5 / F1 54.2, ODQA EM 17.8 / F1 28.7.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kazrag import metrics
from kazrag.data import load_qrels, load_questions, load_reading_comprehension, load_topics
from kazrag.index import DenseRetriever, resolve_index_dir
from kazrag.pipeline import KazakhQA
from kazrag.reader import DEFAULT_READER, ExtractiveReader

TASKS = ("retrieval", "reader", "odqa")
RETRIEVAL_KS = (1, 5, 10, 20, 100)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def eval_retrieval(retriever: DenseRetriever, split: str, limit: int, predictions: list) -> dict:
    topics, qrels = load_topics(split), load_qrels(split)
    qids = [q for q in topics if q in qrels][: limit or None]
    indexed = set(retriever.passages.column("docid").to_pylist())
    missing = sum(len(qrels[q] - indexed) for q in qids)
    if missing:
        log(f"warning: {missing} relevant passages are not in the index; recall is capped")

    scores: dict[str, list[float]] = {"nDCG@10": [], "MRR@10": []}
    for k in RETRIEVAL_KS:
        scores[f"Recall@{k}"], scores[f"Hit@{k}"] = [], []

    batch = 64
    for lo in range(0, len(qids), batch):
        chunk = qids[lo : lo + batch]
        for qid, hits in zip(chunk, retriever.search([topics[q] for q in chunk], k=max(RETRIEVAL_KS)), strict=True):
            ranked, rel = [h.docid for h in hits], qrels[qid]
            scores["nDCG@10"].append(metrics.ndcg_at_k(ranked, rel, 10))
            scores["MRR@10"].append(metrics.mrr_at_k(ranked, rel, 10))
            for k in RETRIEVAL_KS:
                scores[f"Recall@{k}"].append(metrics.recall_at_k(ranked, rel, k))
                scores[f"Hit@{k}"].append(metrics.hit_at_k(ranked, rel, k))
            predictions.append({"task": "retrieval", "qid": qid, "ranked": ranked[:10], "relevant": sorted(rel)})
        log(f"retrieval {min(lo + batch, len(qids))}/{len(qids)}")

    return {"n": len(qids), "relevant_not_indexed": missing, **{k: mean(v) for k, v in scores.items()}}


def eval_reader(reader: ExtractiveReader, split: str, limit: int, predictions: list) -> dict:
    rows = load_reading_comprehension(split)[: limit or None]
    em, f1 = [], []
    for i, row in enumerate(rows, 1):
        golds = row["answers"]["text"]
        context = f"{row['title']}. {row['context']}" if row.get("title") else row["context"]
        span = reader.read(row["question"], [context])[0]
        pred = span.text if span else ""
        em.append(metrics.exact_match(pred, golds))
        f1.append(metrics.max_f1(pred, golds))
        predictions.append({"task": "reader", "id": row["id"], "prediction": pred, "golds": golds})
        if i % 100 == 0 or i == len(rows):
            log(f"reader {i}/{len(rows)}  EM {100 * mean(em):.1f}  F1 {100 * mean(f1):.1f}")
    return {"n": len(rows), "EM": 100 * mean(em), "F1": 100 * mean(f1)}


def eval_odqa(qa: KazakhQA, split: str, top_k: int, limit: int, predictions: list) -> dict:
    questions = load_questions(split)[: limit or None]
    qrels = load_qrels(split)
    em, f1, hit = [], [], []
    for i, ex in enumerate(questions, 1):
        ans = qa.answer(ex.question, top_k=top_k)
        gold_docs = set(ex.docids) | qrels.get(ex.qid, set())
        em.append(metrics.exact_match(ans.text, ex.answers))
        f1.append(metrics.max_f1(ans.text, ex.answers))
        hit.append(metrics.hit_at_k([h.docid for h in ans.hits], gold_docs, top_k))
        predictions.append(
            {
                "task": "odqa",
                "qid": ex.qid,
                "question": ex.question,
                "prediction": ans.text,
                "confidence": ans.confidence,
                "golds": ex.answers,
                "source": ans.passage.docid if ans.passage else None,
                "retrieved": [h.docid for h in ans.hits],
            }
        )
        if i % 50 == 0 or i == len(questions):
            log(f"odqa {i}/{len(questions)}  EM {100 * mean(em):.1f}  F1 {100 * mean(f1):.1f}")
    return {"n": len(questions), "top_k": top_k, "EM": 100 * mean(em), "F1": 100 * mean(f1), f"Hit@{top_k}": mean(hit)}


def to_markdown(results: dict) -> str:
    lines = [
        f"KazQAD `{results['split']}` | embedder `{results.get('embedder', '-')}` "
        f"| reader `{results.get('reader_model', '-')}`",
        "",
    ]
    for task in TASKS:
        if task not in results:
            continue
        r = {k: v for k, v in results[task].items() if isinstance(v, float)}
        lines += [f"**{task}** (n={results[task]['n']})", "", "| " + " | ".join(r) + " |", "|" + "---|" * len(r)]
        lines += ["| " + " | ".join(f"{v:.1f}" if k in ("EM", "F1") else f"{v:.3f}" for k, v in r.items()) + " |", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", choices=[*TASKS, "all"], default=["all"])
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--index", default="index", help="index directory or HF dataset repo id (default: index/)")
    ap.add_argument("--reader", default=DEFAULT_READER)
    ap.add_argument("--top-k", type=int, default=5, help="passages read per question in odqa")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N items per task")
    ap.add_argument("--out", type=Path, help="write results JSON here (default: results/<split>-<time>.json)")
    ap.add_argument("--predictions", type=Path, help="also write per-item predictions as JSONL")
    args = ap.parse_args()

    tasks = TASKS if "all" in args.tasks else tuple(t for t in TASKS if t in args.tasks)
    results: dict = {"split": args.split, "limit": args.limit or None}
    predictions: list[dict] = []

    retriever = reader = None
    if {"retrieval", "odqa"} & set(tasks):
        retriever = DenseRetriever(resolve_index_dir(args.index))
        results |= {"embedder": retriever.spec.name, "index": retriever.meta}
    if {"reader", "odqa"} & set(tasks):
        reader = ExtractiveReader(args.reader)
        results["reader_model"] = args.reader

    t0 = time.perf_counter()
    if "retrieval" in tasks:
        results["retrieval"] = eval_retrieval(retriever, args.split, args.limit, predictions)
    if "reader" in tasks:
        results["reader"] = eval_reader(reader, args.split, args.limit, predictions)
    if "odqa" in tasks:
        results["odqa"] = eval_odqa(KazakhQA(retriever, reader), args.split, args.top_k, args.limit, predictions)
    results["seconds"] = round(time.perf_counter() - t0, 1)

    out = args.out or Path("results") / f"{args.split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions)

    print(to_markdown(results))
    log(f"results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
