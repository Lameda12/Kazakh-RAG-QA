"""Fixtures: a fake KazQAD release on disk and tiny random models, so tests run offline."""

import gzip
import json

import pytest

PASSAGES = [
    ("1_1_1", "Астана", "Астана — Қазақстан Республикасының астанасы. Қала Есіл өзенінің бойында орналасқан."),
    ("2_1_1", "Алматы", "Алматы — Қазақстандағы ең ірі қала. Ол 1997 жылға дейін ел астанасы болды."),
    ("3_1_1", "Байқоңыр", "Байқоңыр ғарыш айлағы Қазақстанның Қызылорда облысында орналасқан."),
    ("4_1_1", "Асқорыту", "Ұлтабарға бауырдан келетін өт өзегі және ұйқыбездің өзегі ашылады."),
    ("5_1_1", "Абай", "Абай Құнанбайұлы 1845 жылы Шыңғыстау баурайында дүниеге келген ақын."),
    ("6_1_1", "Кесте", "\n\n|}\n\n|}"),  # wiki-table debris, must be filtered
]
QUESTIONS = {
    "q1": ("Қазақстанның астанасы қай қала?", "1_1_1", "Астана"),
    "q2": ("Байқоңыр ғарыш айлағы қай облыста?", "3_1_1", "Қызылорда облысында"),
    "q3": ("Бауыр өзегі қандай бөлікке ашылады?", "4_1_1", "Ұлтабарға"),
}


@pytest.fixture(scope="session")
def kazqad_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("kazqad")
    corpus = root / "information-retrieval" / "corpus"
    corpus.mkdir(parents=True)
    rows = [{"docid": d, "title": t, "text": x} for d, t, x in PASSAGES]
    for part, chunk in zip((1, 2, 3), (rows[:2], rows[2:4], rows[4:]), strict=True):
        with gzip.open(corpus / f"kazqad-corpus-v1.0-kk-part-{part}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk)

    (root / "information-retrieval" / "topics").mkdir()
    (root / "information-retrieval" / "qrels").mkdir()
    (root / "reading-comprehension").mkdir()
    for split in ("train", "validation", "test"):
        topics = "".join(f"{q}\t{text}\n" for q, (text, _, _) in QUESTIONS.items())
        qrels = "".join(f"{q}\t0\t{d}\t1\n{q}\t0\t5_1_1\t0\n" for q, (_, d, _) in QUESTIONS.items())
        (root / f"information-retrieval/topics/kazqad-topics-v1.0-kk-{split}.tsv").write_text(topics, encoding="utf-8")
        (root / f"information-retrieval/qrels/kazqad-qrels-v1.0-{split}.tsv").write_text(qrels, encoding="utf-8")
        ctx = {d: (t, x) for d, t, x in PASSAGES}
        rc = []
        for q, (text, d, ans) in QUESTIONS.items():
            title, context = ctx[d]
            rc.append(
                {
                    "id": f"{q}#{d}",
                    "title": title,
                    "context": context,
                    "question": text,
                    "answers": {"text": [ans], "answer_start": [context.index(ans)]},
                }
            )
        path = root / f"reading-comprehension/kazqad-reading-comprehension-v1.0-kk-{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rc), encoding="utf-8")
    return root


@pytest.fixture(scope="session", autouse=True)
def _use_fake_kazqad(kazqad_root):
    # Session-scoped so module-scoped fixtures (index builds) never touch the network.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KAZQAD_ROOT", str(kazqad_root))
        yield


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """Random-weight XLM-R-shaped embedder and QA reader with a tokenizer trained on the fixture text."""
    torch = pytest.importorskip("torch")
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import (
        AutoModelForQuestionAnswering,
        PreTrainedTokenizerFast,
        XLMRobertaConfig,
        XLMRobertaModel,
    )

    torch.manual_seed(0)
    base = tmp_path_factory.mktemp("models")
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    corpus = [x for _, t, x in PASSAGES] + [q for q, _, _ in QUESTIONS.values()]
    tok.train_from_iterator(corpus, trainers.UnigramTrainer(vocab_size=300, special_tokens=specials, unk_token="<unk>"))
    tok.post_processor = processors.RobertaProcessing(
        ("</s>", tok.token_to_id("</s>")), ("<s>", tok.token_to_id("<s>"))
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        model_input_names=["input_ids", "attention_mask"],
    )
    config = XLMRobertaConfig(
        vocab_size=len(fast),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=520,
        pad_token_id=fast.pad_token_id,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
        type_vocab_size=1,
    )

    reader_dir, enc_dir, emb_dir = base / "reader", base / "encoder", base / "embedder"
    AutoModelForQuestionAnswering.from_config(config).save_pretrained(reader_dir)
    fast.save_pretrained(reader_dir)
    XLMRobertaModel(config).save_pretrained(enc_dir)
    fast.save_pretrained(enc_dir)
    word = Transformer(str(enc_dir), max_seq_length=128)
    SentenceTransformer(modules=[word, Pooling(word.get_embedding_dimension(), "mean")]).save(str(emb_dir))
    return {"reader": str(reader_dir), "embedder": str(emb_dir)}
