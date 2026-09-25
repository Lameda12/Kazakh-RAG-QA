#!/usr/bin/env python3
"""Build the dense passage index over the KazQAD corpus.

Run this once on a GPU (Kaggle/Colab T4 is enough), then publish the result and
point the app at it with INDEX_REPO:

    python scripts/build_index.py --out index/
    python scripts/build_index.py --out index/ --push-to <user>/kazqad-bge-m3-index

Re-running with the same arguments resumes an interrupted encoding run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kazrag.data import iter_corpus, reading_comprehension_passages
from kazrag.index import DEFAULT_EMBEDDER, DEFAULT_FACTORY, PUBLISHED_FILES, EmbedderSpec, build_index

CARD = """---
license: cc-by-sa-4.0
language: [kk]
tags: [kazakh, retrieval, faiss]
---
# KazQAD dense index ({embedder})

FAISS `{factory}` index over {n:,} passages of the [KazQAD](https://github.com/IS2AI/KazQAD)
Kazakh Wikipedia corpus, built with `{embedder}` for
[Kazakh-RAG-QA](https://github.com/Lameda12/Kazakh-RAG-QA).

Passage text is from KazQAD (Yeshpanov et al., LREC-COLING 2024), CC BY-SA 4.0.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("index"), help="output directory (default: index/)")
    ap.add_argument(
        "--embedder", default=DEFAULT_EMBEDDER, help=f"sentence-transformers model (default: {DEFAULT_EMBEDDER})"
    )
    ap.add_argument(
        "--corpus",
        choices=["full", "rc"],
        default="full",
        help="full: ~824k Wikipedia passages; rc: ~4.5k passages with annotated answers (quick demo)",
    )
    ap.add_argument("--limit", type=int, default=0, help="index only the first N passages (smoke tests)")
    ap.add_argument(
        "--factory", default=DEFAULT_FACTORY, help=f"faiss index_factory string (default: {DEFAULT_FACTORY})"
    )
    ap.add_argument("--query-prefix", help="override the query prefix (default: inferred from --embedder)")
    ap.add_argument("--passage-prefix", help="override the passage prefix (default: inferred from --embedder)")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--push-to", metavar="REPO_ID", help="upload the index to this Hugging Face dataset repo")
    ap.add_argument("--private", action="store_true", help="create the pushed repo as private")
    args = ap.parse_args()

    if args.corpus == "full":
        print("loading KazQAD corpus (downloads ~131 MB on first run)...", file=sys.stderr)
        passages = list(iter_corpus())
    else:
        passages = reading_comprehension_passages()
    if args.limit:
        passages = passages[: args.limit]
    print(f"{len(passages):,} passages", file=sys.stderr)

    corpus_name = f"kazqad-v1.0-{args.corpus}" + (f"-first{args.limit}" if args.limit else "")
    spec = EmbedderSpec.for_model(args.embedder, args.max_seq_length)
    if args.query_prefix is not None or args.passage_prefix is not None:
        spec = replace(
            spec,
            query_prefix=spec.query_prefix if args.query_prefix is None else args.query_prefix,
            passage_prefix=spec.passage_prefix if args.passage_prefix is None else args.passage_prefix,
        )
    out = build_index(
        passages,
        args.out,
        spec,
        factory=args.factory,
        batch_size=args.batch_size,
        corpus_name=corpus_name,
        log=lambda msg: print(msg, file=sys.stderr),
    )

    if args.push_to:
        from huggingface_hub import HfApi

        (out / "README.md").write_text(CARD.format(embedder=spec.name, factory=args.factory, n=len(passages)))
        api = HfApi()
        api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.push_to, repo_type="dataset", folder_path=out, allow_patterns=list(PUBLISHED_FILES)
        )
        print(f"pushed to https://huggingface.co/datasets/{args.push_to}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
