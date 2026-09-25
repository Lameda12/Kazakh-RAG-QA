"""Gradio front end for Kazakh open-domain QA.

Configuration (environment variables):
    INDEX_REPO    HF dataset repo with a prebuilt index (see scripts/build_index.py)
    INDEX_DIR     local index directory (default: index/)
    READER_MODEL  extractive QA model (default: deepset/xlm-roberta-large-squad2)

With neither index available, the app builds a small demo index over the ~4.5k
KazQAD passages that have annotated answers. That is slow on CPU and only
covers the dataset's own questions; publish a full index for real use.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from kazrag.data import reading_comprehension_passages
from kazrag.index import DEFAULT_EMBEDDER, DenseRetriever, EmbedderSpec, build_index, resolve_index_dir
from kazrag.pipeline import Answer, KazakhQA
from kazrag.reader import DEFAULT_READER, ExtractiveReader

LOW_CONFIDENCE = 0.5
DEMO_INDEX_DIR = Path("index-demo")


def load_index() -> tuple[Path, bool]:
    if repo := os.environ.get("INDEX_REPO"):
        return resolve_index_dir(repo), False
    local = Path(os.environ.get("INDEX_DIR", "index"))
    if (local / "meta.json").exists():
        return local, False
    if not (DEMO_INDEX_DIR / "meta.json").exists():
        print("No prebuilt index found; building the demo index over KazQAD answer passages...")
        build_index(
            reading_comprehension_passages(),
            DEMO_INDEX_DIR,
            EmbedderSpec.for_model(DEFAULT_EMBEDDER),
            corpus_name="kazqad-v1.0-rc",
        )
    return DEMO_INDEX_DIR, True


index_dir, demo_mode = load_index()
retriever = DenseRetriever(index_dir)
reader = ExtractiveReader(os.environ.get("READER_MODEL", DEFAULT_READER))
qa = KazakhQA(retriever, reader)
print(f"Ready: {len(retriever):,} passages, embedder={retriever.spec.name}, reader={reader.model_name}")


def render_answer(ans: Answer) -> str:
    if not ans.found:
        return "**Жауап табылмады.** Сұрақты басқаша қойып көріңіз."
    flag = " · ⚠️ сенімділігі төмен" if ans.confidence < LOW_CONFIDENCE else ""
    return (
        f"### {ans.text}\n\n"
        f"Сенімділік: **{ans.confidence:.0%}**{flag} · Дереккөз: #{ans.passage.rank} *{ans.passage.title}*"
    )


def render_passages(ans: Answer) -> str:
    cards = []
    for h in ans.hits:
        content = h.content
        if ans.passage and h.docid == ans.passage.docid and ans.span:
            s, e = ans.span.start_char, ans.span.end_char
            body = f"{html.escape(content[:s])}<mark>{html.escape(content[s:e])}</mark>{html.escape(content[e:])}"
        else:
            body = html.escape(content)
        cards.append(
            f"<details {'open' if h.rank == 1 or (ans.passage and h.docid == ans.passage.docid) else ''}>"
            f"<summary><b>{h.rank}. {html.escape(h.title)}</b> · ұқсастық {h.score:.3f} · "
            f"<code>{h.docid}</code></summary>"
            f"<p>{body}</p></details>"
        )
    return "\n".join(cards)


def ask(question: str, top_k: int) -> tuple[str, str]:
    if not question.strip():
        return "Сұрақ енгізіңіз.", ""
    try:
        ans = qa.answer(question, top_k=int(top_k))
    except Exception as exc:  # surface model/index failures in the UI instead of a blank output
        raise gr.Error(f"Қате: {exc}") from exc
    return render_answer(ans), render_passages(ans)


corpus_note = (
    f"⚠️ Демо режимі: тек {len(retriever):,} KazQAD үзіндісі индекстелген."
    if demo_mode
    else f"{len(retriever):,} Қазақша Википедия үзіндісі"
)

with gr.Blocks(title="Kazakh RAG QA") as demo:
    gr.Markdown(
        "# 🇰🇿 Қазақша Сұрақ-Жауап Жүйесі\n"
        f"`{retriever.spec.name}` → `FAISS` → `{reader.model_name}` · {corpus_note} · "
        "[KazQAD (ISSAI)](https://github.com/IS2AI/KazQAD)"
    )
    with gr.Row():
        q_input = gr.Textbox(label="Сұрақ (қазақша)", placeholder="Қазақстан туралы сұрақ...", lines=2, scale=4)
        topk = gr.Slider(1, 20, value=5, step=1, label="Оқылатын үзінділер (top-k)", scale=1)
    btn = gr.Button("Іздеу 🔍", variant="primary")
    answer_out = gr.Markdown()
    passages_out = gr.HTML()

    gr.Examples(
        [
            ["Бауыр өзегі қандай бөлікке ашылады?", 5],
            ["Қазақстанның астанасы қай қала?", 5],
            ["Байқоңыр ғарыш айлағы қай елде?", 5],
        ],
        inputs=[q_input, topk],
    )

    btn.click(ask, [q_input, topk], [answer_out, passages_out])
    q_input.submit(ask, [q_input, topk], [answer_out, passages_out])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="teal"))
