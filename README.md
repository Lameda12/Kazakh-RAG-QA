# 🇰🇿 Kazakh RAG Question Answering

Open-domain question answering over Kazakh Wikipedia, evaluated on [KazQAD](https://github.com/IS2AI/KazQAD) (ISSAI, LREC-COLING 2024).

## How it works

1. The question is embedded with **`BAAI/bge-m3`**.
2. **FAISS** searches a prebuilt index over the full KazQAD corpus (823,889 passages after dropping wiki-table debris).
3. **`deepset/xlm-roberta-large-squad2`** reads the top-k passages with sliding windows and returns the best span, ranked by a SQuAD2 no-answer margin so scores are comparable across passages.

Why these models: on the full KazQAD corpus, zero-shot bge-m3 matches the best published Kazakh fine-tunes ([kaz-embed](https://github.com/ProgrammerXX1/kaz-embed), nDCG@10 0.362). kazembed-v5's own model card shows multilingual-e5-large ahead of it. The previous reader, `roberta-base-squad2`, was trained on English only.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the index. Needs a GPU for the full corpus (a Kaggle/Colab T4 works); resumable if interrupted.
python scripts/build_index.py --out index/ --push-to <hf-user>/kazqad-bge-m3-index

# 2. Run the app against it
INDEX_REPO=<hf-user>/kazqad-bge-m3-index python app.py   # or INDEX_DIR=index/
```

For a quick CPU test, run `python scripts/build_index.py --corpus rc` to index only the ~4.5k passages that have annotated answers. The app builds that demo index itself if it finds no index at all.

| Env var | Default | |
|---|---|---|
| `INDEX_REPO` | – | HF dataset repo with a published index |
| `INDEX_DIR` | `index/` | local index directory |
| `READER_MODEL` | `deepset/xlm-roberta-large-squad2` | try `deepset/xlm-roberta-base-squad2` for faster CPU inference |
| `KAZQAD_ROOT` | – | local copy of KazQAD's `data/` dir (skips downloads) |

KazQAD data is downloaded from ISSAI's public GitHub release (CC BY-SA 4.0), so no HF token is needed. The HF copies are gated.

## Evaluation

```bash
python scripts/eval.py --index index/ --tasks all            # full test split
python scripts/eval.py --index index/ --tasks odqa --limit 200 --predictions results/odqa.jsonl
```

| Task | Metric | KazQAD paper baseline | This repo |
|---|---|---|---|
| Retrieval | nDCG@10 / MRR | 0.389 / 0.382 | run `--tasks retrieval` |
| Reading comprehension (gold passage) | EM / F1 | 38.5 / 54.2 | run `--tasks reader` |
| Open-domain QA | EM / F1 | 17.8 / 28.7 | run `--tasks odqa` |

Results are written to `results/<split>-<time>.json`, and a markdown table is printed to stdout.

## Fine-tuning the reader

`scripts/train_reader.py` fine-tunes `deepset/xlm-roberta-base-squad2` on KazQAD train plus ~61.6k machine-translated NQ items, one run per seed, then averages the runs into a [model soup](https://arxiv.org/abs/2203.05482) (uniform and greedy). Whichever of the base model, each run and the soups scores best on KazQAD validation is saved to `<out>/final`. Runs resume after a disconnect when `--out` is on Google Drive.

```bash
python scripts/train_reader.py --out /content/drive/MyDrive/reader-runs
python scripts/eval.py --tasks reader --reader /content/drive/MyDrive/reader-runs/final
```

## Development

```bash
pip install -r requirements-dev.txt
pytest          # runs offline against a fake KazQAD fixture and tiny random models
ruff check . && ruff format --check .
```

## Deploying to Hugging Face Spaces

The Space needs the Spaces front matter (`sdk: gradio`, `sdk_version`) in its own README. Copy `app.py`, `kazrag/` and `requirements.txt`, set `sdk_version: 6.28.0`, and add `INDEX_REPO` as a Space variable. Estimated RAM on CPU: ~2.3 GB bge-m3 + ~2.2 GB reader + ~1.7 GB fp16 index + passages, roughly 7 GB, which fits the free CPU tier (16 GB).

## Citation

```bibtex
@inproceedings{yeshpanov-etal-2024-kazqad,
    title = "{K}az{QAD}: {K}azakh Open-Domain Question Answering Dataset",
    author = "Yeshpanov, Rustem and Efimov, Pavel and Boytsov, Leonid and Shalkarbayuli, Ardak and Braslavski, Pavel",
    booktitle = "Proceedings of LREC-COLING 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.lrec-main.843",
}
```
