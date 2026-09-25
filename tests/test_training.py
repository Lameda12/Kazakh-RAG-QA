import importlib
import json
import sys
from pathlib import Path

from kazrag.data import load_nq_translated, load_reading_comprehension
from kazrag.metrics import normalize_answer
from kazrag.training import build_features


def test_nq_loader_reads_supplementary_file():
    rows = load_nq_translated()
    assert len(rows) == 5 and rows[0]["answers"]["answer_start"] == [0]


def test_features_label_the_answer_tokens(tiny_models):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_models["reader"])
    rows = load_reading_comprehension("train")
    # a small window forces several windows per row; only those containing the answer get non-CLS labels
    features = build_features(tok, rows, max_length=32, stride=8)
    labelled = [f for f in features if f["start_positions"] > 0]
    assert labelled, "at least one window must contain its answer"
    answers = {normalize_answer(a) for r in rows for a in r["answers"]["text"]}
    for f in labelled:
        assert f["start_positions"] <= f["end_positions"]
        span = tok.decode(f["input_ids"][f["start_positions"] : f["end_positions"] + 1])
        assert normalize_answer(span) in answers


def test_train_reader_end_to_end(tiny_models, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    train_reader = importlib.import_module("train_reader")
    out = tmp_path / "runs"
    argv = [
        "train_reader.py",
        "--base",
        tiny_models["reader"],
        "--out",
        str(out),
        "--seeds",
        "1",
        "2",
        "--max-steps",
        "2",
        "--batch-size",
        "4",
        "--max-length",
        "64",
        "--stride",
        "16",
        "--cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train_reader.main() == 0

    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["scores"]) == {"base", "seed-1", "seed-2", "soup-uniform", "soup-greedy"}
    assert (out / "final" / "config.json").exists()
    assert not (out / "seed-1" / "checkpoints").exists()  # optimizer checkpoints are cleaned up
    assert "| model | EM | F1 |" in capsys.readouterr().out

    # re-running skips finished seeds (no retraining) and reuses cached scores
    monkeypatch.setattr(train_reader, "train_one", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    assert train_reader.main() == 0
