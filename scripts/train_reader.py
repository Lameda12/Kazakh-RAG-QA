#!/usr/bin/env python3
"""Fine-tune the extractive reader on Kazakh QA, then average the runs into a model soup.

Trains one model per seed from the same starting checkpoint on KazQAD train
(repeated --kazqad-repeat times) plus the ~61.6k machine-translated NQ items,
scores each on KazQAD validation, then builds a uniform soup (all runs) and a
greedy soup (runs added best-first while validation F1 does not drop). The best
of {base, each run, both soups} by validation F1 is copied to <out>/final.

Each seed's run is resumable: re-running the same command after a Colab
disconnect skips finished seeds and resumes the current one from its last
checkpoint. Put --out on Google Drive for that to survive a runtime reset.

    python scripts/train_reader.py --out /content/drive/MyDrive/reader-runs
    python scripts/eval.py --tasks reader --reader /content/drive/MyDrive/reader-runs/final
    python scripts/train_reader.py --out ... --push-to <user>/xlm-roberta-base-kazqad-soup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kazrag.data import load_nq_translated, load_reading_comprehension
from kazrag.reader import ExtractiveReader
from kazrag.soup import greedy_soup, make_soup
from kazrag.training import FeatureDataset, build_features, score_reader

DEFAULT_BASE = "deepset/xlm-roberta-base-squad2"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cached_score(model_dir: Path | str, val_rows: list[dict], cache: Path) -> dict:
    """Validation EM/F1 for a model, cached in a JSON file so resumed runs don't re-score."""
    if cache.exists():
        return json.loads(cache.read_text())
    t0 = time.perf_counter()
    score = score_reader(ExtractiveReader(str(model_dir), batch_size=32), val_rows)
    score["seconds"] = round(time.perf_counter() - t0, 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(score, indent=2))
    return score


def train_one(args, features: list[dict], tokenizer, seed: int, run_dir: Path) -> None:
    import torch
    from transformers import AutoModelForQuestionAnswering, DataCollatorWithPadding, Trainer, TrainingArguments

    ckpt_dir = run_dir / "checkpoints"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        train_sampling_strategy="group_by_length",  # batches of similar length: far less padding
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        use_cpu=args.cpu,
    )
    trainer = Trainer(
        model=AutoModelForQuestionAnswering.from_pretrained(args.base),
        args=targs,
        train_dataset=FeatureDataset(features),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        log(f"seed {seed}: resuming from {checkpoints[-1].name}")
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(run_dir)
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # optimizer state is ~3x the model; not needed once done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"starting checkpoint for every run (default: {DEFAULT_BASE})")
    ap.add_argument("--out", type=Path, default=Path("runs/reader"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps per run (smoke tests)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--kazqad-repeat", type=int, default=2, help="upsample KazQAD train against the larger NQ set")
    ap.add_argument("--nq-limit", type=int, default=0, help="use only the first N translated NQ items (0 = all)")
    ap.add_argument("--val-limit", type=int, default=0, help="score on the first N validation rows (0 = all 764)")
    ap.add_argument("--save-steps", type=int, default=500, help="checkpoint interval for resuming a run")
    ap.add_argument("--cpu", action="store_true", help="force CPU training")
    ap.add_argument("--push-to", metavar="REPO_ID", help="upload <out>/final to this Hugging Face model repo")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    nq = load_nq_translated()
    train_rows = load_reading_comprehension("train") * args.kazqad_repeat + nq[: args.nq_limit or None]
    val_rows = load_reading_comprehension("validation")[: args.val_limit or None]
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    features = build_features(tokenizer, train_rows, args.max_length, args.stride)
    null = sum(f["start_positions"] == 0 for f in features)
    log(f"{len(train_rows):,} examples -> {len(features):,} windows ({null / len(features):.0%} without the answer)")
    log(f"validating on {len(val_rows)} KazQAD validation rows")

    scores: dict[str, dict] = {"base": cached_score(args.base, val_rows, args.out / "base-score.json")}
    log(f"base {args.base}: EM {scores['base']['EM']:.1f} F1 {scores['base']['F1']:.1f}")

    runs: list[tuple[Path, float]] = []
    for seed in args.seeds:
        run_dir = args.out / f"seed-{seed}"
        if not (run_dir / "config.json").exists():
            t0 = time.perf_counter()
            train_one(args, features, tokenizer, seed, run_dir)
            log(f"seed {seed}: trained in {(time.perf_counter() - t0) / 60:.1f} min")
        score = cached_score(run_dir, val_rows, run_dir / "val-score.json")
        scores[run_dir.name] = score
        runs.append((run_dir, score["F1"]))
        log(f"seed {seed}: EM {score['EM']:.1f} F1 {score['F1']:.1f}")

    candidates = {"base": args.base, **{p.name: p for p, _ in runs}}
    if len(runs) > 1:
        uniform = make_soup([p for p, _ in runs], args.out / "soup-uniform")
        scores["soup-uniform"] = cached_score(uniform, val_rows, uniform / "val-score.json")
        candidates["soup-uniform"] = uniform

        trial = args.out / "soup-trial"

        def soup_f1(paths: list[Path]) -> float:
            shutil.rmtree(trial, ignore_errors=True)
            return score_reader(ExtractiveReader(str(make_soup(paths, trial)), batch_size=32), val_rows)["F1"]

        ranked = sorted(runs, key=lambda r: r[1], reverse=True)
        members, _ = greedy_soup(ranked, soup_f1, log)
        shutil.rmtree(trial, ignore_errors=True)
        greedy = make_soup(members, args.out / "soup-greedy")
        scores["soup-greedy"] = cached_score(greedy, val_rows, greedy / "val-score.json")
        scores["soup-greedy"]["members"] = [p.name for p in members]
        candidates["soup-greedy"] = greedy

    best = max(scores, key=lambda name: scores[name]["F1"])
    final = args.out / "final"
    shutil.rmtree(final, ignore_errors=True)
    if best == "base":
        from transformers import AutoModelForQuestionAnswering

        AutoModelForQuestionAnswering.from_pretrained(args.base).save_pretrained(final)
        tokenizer.save_pretrained(final)
    else:
        shutil.copytree(candidates[best], final)
    summary = {
        "best": best,
        "base_model": args.base,
        "scores": scores,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("| model | EM | F1 |\n|---|---|---|")
    for name, s in scores.items():
        print(f"| {name}{' (best)' if name == best else ''} | {s['EM']:.1f} | {s['F1']:.1f} |")
    log(f"best on validation: {best} -> {final}")

    if args.push_to:
        from huggingface_hub import HfApi

        s = scores[best]
        (final / "README.md").write_text(
            "---\nlanguage: [kk]\nlicense: cc-by-sa-4.0\npipeline_tag: question-answering\n"
            f"base_model: {args.base}\ndatasets: [issai/kazqad]\n---\n"
            f"# Kazakh extractive QA reader ({best})\n\n"
            f"`{args.base}` fine-tuned on KazQAD train and the machine-translated NQ set; `{best}` scored best "
            f"on KazQAD validation: EM {s['EM']:.1f}, F1 {s['F1']:.1f} (base: EM {scores['base']['EM']:.1f}, "
            f"F1 {scores['base']['F1']:.1f}).\n\nBuilt with [Kazakh-RAG-QA](https://github.com/Lameda12/Kazakh-RAG-QA).\n"
        )
        api = HfApi()
        api.create_repo(args.push_to, exist_ok=True)
        api.upload_folder(repo_id=args.push_to, folder_path=final, ignore_patterns=["val-score.json"])
        log(f"pushed to https://huggingface.co/{args.push_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
