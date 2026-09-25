"""End-to-end over tiny random models: exercises the real library APIs, not answer quality."""

import importlib
import sys

import pytest

from kazrag.data import iter_corpus
from kazrag.index import DenseRetriever, EmbedderSpec, build_index
from kazrag.pipeline import KazakhQA
from kazrag.reader import ExtractiveReader


@pytest.fixture(scope="module", params=["Flat", "SQfp16", "SQ8"])
def index_dir(request, tiny_models, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"index-{request.param}")
    spec = EmbedderSpec.for_model(tiny_models["embedder"], max_seq_length=128)
    return build_index(list(iter_corpus()), out, spec, factory=request.param, chunk_size=2, log=lambda _: None)


def test_index_roundtrip_and_self_retrieval(index_dir):
    retriever = DenseRetriever(index_dir)
    assert len(retriever) == 5
    # Querying with a passage's own content should rank it first, even with random weights.
    passages = list(iter_corpus())
    hits = retriever.search([p.content for p in passages], k=3)
    assert [h[0].docid for h in hits] == [p.docid for p in passages]
    assert all(h[0].rank == 1 and len(h) == 3 for h in hits)


def test_build_resumes_from_progress(tiny_models, tmp_path):
    spec = EmbedderSpec.for_model(tiny_models["embedder"], max_seq_length=128)
    passages = list(iter_corpus())
    build_index(passages, tmp_path, spec, factory="Flat", chunk_size=2, log=lambda _: None)
    logs = []
    build_index(passages, tmp_path, spec, factory="Flat", chunk_size=2, log=logs.append)
    assert not any(m.startswith("encoded") for m in logs)  # nothing re-encoded


def test_reader_returns_substrings_with_windows(tiny_models):
    # max_length forces several overlapping windows per context.
    reader = ExtractiveReader(tiny_models["reader"], device="cpu", max_length=32, stride=8)
    contexts = [p.content for p in iter_corpus()]
    spans = reader.read("Қазақстанның астанасы қай қала?", contexts)
    assert len(spans) == len(contexts)
    for span, ctx in zip(spans, contexts, strict=True):
        assert span is not None
        assert ctx[span.start_char : span.end_char] == span.text
        assert 0.0 <= span.confidence <= 1.0


def test_pipeline_answer(index_dir, tiny_models):
    qa = KazakhQA(DenseRetriever(index_dir), ExtractiveReader(tiny_models["reader"], device="cpu"))
    ans = qa.answer("Байқоңыр ғарыш айлағы қай облыста?", top_k=3)
    assert ans.found and len(ans.hits) == 3
    assert ans.text in ans.passage.content
    assert qa.answer("   ").hits == []


def test_eval_script(index_dir, tiny_models, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, "scripts")
    eval_script = importlib.import_module("eval")
    out = tmp_path / "res.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--index",
            str(index_dir),
            "--reader",
            tiny_models["reader"],
            "--out",
            str(out),
            "--predictions",
            str(tmp_path / "p.jsonl"),
        ],
    )
    assert eval_script.main() == 0
    import json

    res = json.loads(out.read_text())
    assert res["retrieval"]["n"] == 3 and 0 <= res["retrieval"]["nDCG@10"] <= 1
    assert res["retrieval"]["Recall@100"] == 1.0  # tiny corpus: everything is retrieved
    assert res["reader"]["n"] == 3 and res["odqa"]["n"] == 3
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 9
    assert "**odqa**" in capsys.readouterr().out


def test_app_ask(index_dir, tiny_models, monkeypatch):
    monkeypatch.delenv("INDEX_REPO", raising=False)
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("READER_MODEL", tiny_models["reader"])
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    assert not app.demo_mode
    answer_md, passages_html = app.ask("Абай қай жылы туған?", 2)
    assert "Сенімділік" in answer_md
    assert passages_html.count("<details") == 2
    assert app.ask("  ", 2)[0] == "Сұрақ енгізіңіз."


def test_reader_clips_long_questions(tiny_models):
    reader = ExtractiveReader(tiny_models["reader"], device="cpu", max_length=64, stride=16)
    long_question = " ".join(["Қазақстанның астанасы қай қала?"] * 60)
    [span] = reader.read(long_question, [next(iter_corpus()).content])
    assert span is not None
