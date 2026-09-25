import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
error_analysis = importlib.import_module("error_analysis")


def rec(qid, pred, golds, retrieved, gold_docs, source, conf=0.5):
    return {
        "task": "odqa",
        "qid": qid,
        "question": "?",
        "prediction": pred,
        "golds": golds,
        "retrieved": retrieved,
        "gold_docs": gold_docs,
        "source": source,
        "confidence": conf,
    }


RECORDS = [
    rec("bio0001bio", "Ұлтабарға", ["Ұлтабарға"], ["d1"], ["d1"], "d1", 0.9),  # correct
    rec("bio0002bio", "Қызылорда облысы", ["Қызылорда облысында"], ["d2"], ["d2"], "d2"),  # partial
    rec("kzh0001kzh", "Алматы", ["Астана"], ["d9"], ["d3"], "d9", 0.1),  # retrieval miss
    rec("kzh0002kzh", "1997", ["1991"], ["d4", "d8"], ["d4"], "d8", 0.3),  # reader miss, wrong passage
]


def test_buckets_subjects_and_confidence():
    report = error_analysis.analyze([dict(r) for r in RECORDS])
    assert {b: v["n"] for b, v in report["buckets"].items()} == {
        "correct": 1,
        "partial": 1,
        "retrieval miss": 1,
        "reader miss": 1,
    }
    assert report["reader_miss_wrong_passage"] == 1
    assert report["by_subject"]["biology"]["EM"] == 50.0
    assert report["by_subject"]["Kazakh history"]["retrieval_miss"] == 50.0
    top = next(b for b in report["confidence"] if b["band"].startswith("0.75"))
    assert top == {"band": "0.75-1.00", "n": 1, "EM": 100.0}


def test_cli_prints_report(tmp_path, monkeypatch, capsys):
    path = tmp_path / "preds.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in RECORDS), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["error_analysis.py", str(path), "--out", str(tmp_path / "r.json")])
    assert error_analysis.main() == 0
    out = capsys.readouterr().out
    assert "| retrieval miss | 1 | 25.0% |" in out and "Sample reader miss" in out
    assert json.loads((tmp_path / "r.json").read_text())["n"] == 4


def test_legacy_predictions_get_gold_docs_from_data(tmp_path):
    # fixture split: q1's gold passage is 1_1_1
    legacy = rec("q1", "Астана", ["Астана"], ["1_1_1"], [], "1_1_1")
    del legacy["gold_docs"]
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")
    assert error_analysis.load_records(path, "test")[0]["gold_docs"] == ["1_1_1"]
