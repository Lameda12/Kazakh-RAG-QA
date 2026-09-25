#!/usr/bin/env python3
"""Break down open-domain QA errors from the predictions `scripts/eval.py --predictions` writes.

Every question lands in exactly one bucket:
  correct         exact match with a gold answer
  partial         token overlap but no exact match (often a Kazakh suffix, e.g. "Ұлтабар" vs "Ұлтабарға")
  retrieval miss  no gold passage among the passages the reader saw: fix retrieval (reranker, hybrid search)
  reader miss     a gold passage was retrieved but the answer is wrong: fix the reader (fine-tune)
Reader misses are split by whether the answer was taken from a gold passage.

Also reports results per exam subject and accuracy by confidence band (whether the
confidence score can decide when to abstain), plus sample failures to read by hand.

    python scripts/eval.py --index index/ --tasks odqa --predictions results/preds.jsonl
    python scripts/error_analysis.py results/preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kazrag import metrics

SUBJECTS = {"kzh": "Kazakh history", "woh": "world history", "bio": "biology", "geo": "geography", "lit": "literature"}
BUCKETS = ("correct", "partial", "retrieval miss", "reader miss")
CONFIDENCE_BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def subject(qid: str) -> str:
    prefix = m.group(0) if (m := re.match(r"[a-z]+", qid)) else "?"
    return SUBJECTS.get(prefix, prefix)


def classify(rec: dict) -> str:
    if metrics.exact_match(rec["prediction"], rec["golds"]):
        return "correct"
    if metrics.max_f1(rec["prediction"], rec["golds"]) > 0:
        return "partial"
    if not set(rec["retrieved"]) & set(rec["gold_docs"]):
        return "retrieval miss"
    return "reader miss"


def load_records(path: Path, split: str) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        records = [r for line in fh if line.strip() and (r := json.loads(line)).get("task") == "odqa"]
    if records and "gold_docs" not in records[0]:
        # predictions written before eval.py recorded gold passages
        from kazrag.data import load_qrels, load_questions

        qrels = load_qrels(split)
        gold = {q.qid: set(q.docids) | qrels.get(q.qid, set()) for q in load_questions(split)}
        for r in records:
            r["gold_docs"] = sorted(gold.get(r["qid"], ()))
    return records


def analyze(records: list[dict]) -> dict:
    for r in records:
        r["bucket"] = classify(r)
        r["em"] = metrics.exact_match(r["prediction"], r["golds"])
        r["f1"] = metrics.max_f1(r["prediction"], r["golds"])

    n = len(records)
    buckets = Counter(r["bucket"] for r in records)
    reader_misses = [r for r in records if r["bucket"] == "reader miss"]
    wrong_passage = sum(r["source"] not in r["gold_docs"] for r in reader_misses)

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_subject[subject(r["qid"])].append(r)

    bands = []
    for lo, hi in CONFIDENCE_BANDS:
        rs = [r for r in records if lo <= r["confidence"] < hi]
        bands.append({"band": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": len(rs), "EM": _pct(rs, "em")})

    return {
        "n": n,
        "EM": _pct(records, "em"),
        "F1": _pct(records, "f1"),
        "buckets": {b: {"n": buckets[b], "share": 100 * buckets[b] / n if n else 0.0} for b in BUCKETS},
        "reader_miss_wrong_passage": wrong_passage,
        "by_subject": {
            s: {
                "n": len(rs),
                "EM": _pct(rs, "em"),
                "F1": _pct(rs, "f1"),
                "retrieval_miss": 100 * sum(r["bucket"] == "retrieval miss" for r in rs) / len(rs),
            }
            for s, rs in sorted(by_subject.items(), key=lambda kv: -len(kv[1]))
        },
        "confidence": bands,
    }


def _pct(rs: list[dict], key: str) -> float:
    return 100 * sum(r[key] for r in rs) / len(rs) if rs else 0.0


def to_markdown(report: dict, records: list[dict], samples: int, seed: int) -> str:
    out = [f"**{report['n']} questions**: EM {report['EM']:.1f}, F1 {report['F1']:.1f}", ""]
    out += ["| bucket | questions | share |", "|---|---|---|"]
    out += [f"| {b} | {v['n']} | {v['share']:.1f}% |" for b, v in report["buckets"].items()]
    misses = report["buckets"]["reader miss"]["n"]
    if misses:
        out += [
            "",
            f"Reader misses where the answer came from a non-gold passage: "
            f"{report['reader_miss_wrong_passage']} of {misses}.",
        ]
    out += ["", "| subject | questions | EM | F1 | retrieval miss |", "|---|---|---|---|---|"]
    out += [
        f"| {s} | {v['n']} | {v['EM']:.1f} | {v['F1']:.1f} | {v['retrieval_miss']:.1f}% |"
        for s, v in report["by_subject"].items()
    ]
    out += ["", "| confidence | questions | EM |", "|---|---|---|"]
    out += [f"| {b['band']} | {b['n']} | {b['EM']:.1f} |" for b in report["confidence"]]

    rng = random.Random(seed)
    for bucket in ("retrieval miss", "reader miss", "partial"):
        rs = [r for r in records if r["bucket"] == bucket]
        if not rs or not samples:
            continue
        out += ["", f"**Sample {bucket}** ({min(samples, len(rs))} of {len(rs)})", ""]
        for r in rng.sample(rs, min(samples, len(rs))):
            out.append(
                f"- `{r['qid']}` {r['question']}  \n  predicted: {r['prediction'] or '∅'} "
                f"(conf {r['confidence']:.2f}) · gold: {' | '.join(r['golds'])}"
            )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, help="JSONL written by scripts/eval.py --predictions")
    ap.add_argument("--split", default="test", help="split the predictions came from (for older files)")
    ap.add_argument("--samples", type=int, default=5, help="failures to print per bucket")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="also write the report as JSON")
    args = ap.parse_args()

    records = load_records(args.predictions, args.split)
    if not records:
        print(f"no odqa predictions in {args.predictions}; run eval.py with --tasks odqa", file=sys.stderr)
        return 1
    report = analyze(records)
    print(to_markdown(report, records, args.samples, args.seed))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
