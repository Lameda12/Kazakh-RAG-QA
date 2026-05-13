---
license: mit
title: Kazakh RAG QA
sdk: gradio
emoji: 🌍
colorFrom: gray
colorTo: green
pinned: false
short_description: Ask questions in Kazakh, get answers instantly
sdk_version: 6.14.0
---
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
