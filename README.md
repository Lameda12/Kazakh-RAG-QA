# 🇰🇿 Kazakh RAG Question Answering

A Retrieval-Augmented Generation (RAG) system for Kazakh-language QA.

## How it works
1. Your question is embedded using **KazEmbed-v5**
2. **FAISS** retrieves the top-K most similar passages from KazQAD
3. **RoBERTa** extracts the answer span from the best passage

## Dataset
[KazQAD](https://huggingface.co/datasets/issai/kazqad) by ISSAI, Nazarbayev University

## Stack
- `sentence-transformers` + `Nurlykhan/kazembed-v5`
- `faiss-cpu`
- `deepset/roberta-base-squad2`
- `Gradio`
