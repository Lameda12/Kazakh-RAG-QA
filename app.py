import gradio as gr
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# ── Load everything at startup ──────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")

print("Loading KazQAD...")
dataset = load_dataset("issai/kazqad", "kazqad", token=HF_TOKEN)

all_passages = []
for split in ["train", "validation", "test"]:
    for item in dataset[split]:
        all_passages.append(item["context"])

unique_passages = list(dict.fromkeys(all_passages))[:500]
print(f"Corpus: {len(unique_passages):,} passages")

print("Loading KazEmbed-v5...")
embedder = SentenceTransformer("Nurlykhan/kazembed-v5")

print("Encoding corpus...")
corpus_embeddings = embedder.encode(
    [f"passage: {p}" for p in unique_passages],
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
).astype(np.float32)

index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)
print(f"FAISS index built: {index.ntotal:,} vectors")

print("Loading QA reader...")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_model.eval()
print("✅ Ready!")

# ── RAG pipeline ─────────────────────────────────────────────
def kazakh_rag_qa(question, top_k=5):
    if not question.strip():
        return "Сұрақ енгізіңіз.", ""

    query_emb = embedder.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(query_emb, int(top_k))
    passages = [
        {"passage": unique_passages[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0]) if idx >= 0
    ]

    candidates = []
    for p in passages:
        inputs = qa_tokenizer(
            question, p["passage"],
            return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start = outputs.start_logits.argmax()
        end = outputs.end_logits.argmax() + 1
        tokens = inputs["input_ids"][0][start:end]
        answer = qa_tokenizer.decode(tokens, skip_special_tokens=True)
        score = float(outputs.start_logits.max() + outputs.end_logits.max())
        candidates.append({
            "answer": answer,
            "answer_score": score,
            "combined_score": score * p["score"],
            "passage": p["passage"],
            "retrieval_score": p["score"]
        })

    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    best = candidates[0]

    answer_md = f"**Жауап:** {best['answer']}\n\n*Сенімділік: {best['answer_score']:.1f} | Ұқсастық: {best['retrieval_score']:.3f}*"
    passages_md = ""
    for i, p in enumerate(passages, 1):
        passages_md += f"**{i}.** (score: {p['score']:.3f})\n{p['passage'][:400]}...\n\n---\n\n"

    return answer_md, passages_md

# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(title="🇰🇿 Kazakh RAG QA", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        "# 🇰🇿 Қазақша RAG Сұрақ-Жауап Жүйесі\n"
        "`kazembed-v5` · `FAISS` · `xlm-roberta` · `KazQAD (ISSAI)`"
    )
    with gr.Row():
        q_input = gr.Textbox(label="Сұрақ (қазақша)", placeholder="Қазақстан туралы сұрақ...", lines=2)
        topk = gr.Slider(1, 10, value=5, step=1, label="Top-K үзінділер")
    btn = gr.Button("Іздеу 🔍", variant="primary")
    answer_out = gr.Markdown(label="Жауап")
    passages_out = gr.Markdown(label="Үзінділер")

    gr.Examples([
        ["Қазақстанның астанасы қай қала?", 3],
        ["Қазақстан қай жылы тәуелсіздік алды?", 3],
        ["Байқоңыр ғарыш айлағы қай елде?", 3],
    ], inputs=[q_input, topk])

    btn.click(kazakh_rag_qa, [q_input, topk], [answer_out, passages_out])
    q_input.submit(kazakh_rag_qa, [q_input, topk], [answer_out, passages_out])

demo.launch()
