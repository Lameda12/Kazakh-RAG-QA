"""Dense passage index: build once offline, load read-only at serve time.

Index directory layout:
    meta.json          embedder name, prefixes, faiss factory, corpus size
    passages.parquet   docid/title/text; row i is faiss id i
    index.faiss        the vector index
    embeddings.npy     (build scratch, resumable, not published)
    progress.json      (build scratch)
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kazrag.data import Passage
from kazrag.device import default_device

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_FACTORY = "SQfp16"  # exact inner-product scan at half the memory of Flat
PUBLISHED_FILES = ("meta.json", "passages.parquet", "index.faiss", "README.md")


@dataclass(frozen=True, slots=True)
class EmbedderSpec:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512

    @classmethod
    def for_model(cls, name: str, max_seq_length: int = 512) -> EmbedderSpec:
        """E5-family models (incl. kazembed-v5) need "query: "/"passage: " prefixes; bge-m3 needs none."""
        lowered = name.lower()
        if "e5" in lowered or "kazembed" in lowered:
            return cls(name, "query: ", "passage: ", max_seq_length)
        return cls(name, max_seq_length=max_seq_length)


@dataclass(frozen=True, slots=True)
class Hit:
    rank: int
    docid: str
    title: str
    text: str
    score: float

    @property
    def content(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


def load_embedder(spec: EmbedderSpec, device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or default_device()
    model_kwargs = {"dtype": torch.float16} if device.startswith(("cuda", "mps")) else {}
    model = SentenceTransformer(spec.name, device=device, model_kwargs=model_kwargs)
    # bge-m3 defaults to 8192 tokens; KazQAD passages are short (p95 ~970 chars).
    model.max_seq_length = spec.max_seq_length
    return model


def encode(model, texts: Sequence[str], prefix: str, batch_size: int = 32, pool=None) -> np.ndarray:
    return model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        pool=pool,
    ).astype(np.float32)


def default_search_params(factory: str) -> str:
    """Query-time knobs for approximate indexes; faiss defaults (nprobe=1) wreck recall."""
    params = []
    if "IVF" in factory:
        params.append("nprobe=64")
    if "HNSW" in factory:
        params.append("efSearch=128")
    return ",".join(params)


def build_index(
    passages: Sequence[Passage],
    out_dir: Path,
    spec: EmbedderSpec,
    *,
    factory: str = DEFAULT_FACTORY,
    search_params: str | None = None,
    batch_size: int = 32,
    chunk_size: int = 20_000,
    corpus_name: str = "kazqad-v1.0",
    model=None,
    device: str | None = None,
    devices: Sequence[str] | None = None,
    log=print,
) -> Path:
    """Encode passages and write a self-describing index directory.

    Encoding is checkpointed per chunk into embeddings.npy, so an interrupted
    Colab/Kaggle run resumes where it stopped when re-run with the same args.
    Once encoding is complete, re-running with a different `factory` only
    rebuilds the faiss index from embeddings.npy: no model is loaded.
    `devices` (e.g. ["cuda:0", "cuda:1"]) encodes on several GPUs at once.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(passages)
    if n == 0:
        raise ValueError("no passages to index")

    _write_passages(passages, out_dir / "passages.parquet")
    emb_path, progress_path = out_dir / "embeddings.npy", out_dir / "progress.json"
    progress = _read_progress(progress_path, spec, n)

    emb = None
    if progress["done"] and emb_path.exists():
        emb = np.lib.format.open_memmap(emb_path, mode="r+")
        if emb.shape[0] != n:
            emb, progress = None, _read_progress(None, spec, n)
    if emb is None:
        model = model or load_embedder(spec, device)
        dim = (getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension)()
        emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, dim))
    dim = emb.shape[1]

    start = progress["done"]
    if start < n:
        model = model or load_embedder(spec, device)
        pool = model.start_multi_process_pool(list(devices)) if devices and len(devices) > 1 else None
        log(
            f"encoding on {', '.join(devices) if pool else model.device}"
            + (f", resuming at {start:,}/{n:,}" if start else "")
        )
        try:
            t0 = time.perf_counter()
            for lo in range(start, n, chunk_size):
                hi = min(lo + chunk_size, n)
                texts = [p.content for p in passages[lo:hi]]
                emb[lo:hi] = encode(model, texts, spec.passage_prefix, batch_size, pool)
                emb.flush()
                progress["done"] = hi
                progress_path.write_text(json.dumps(progress))
                rate = (hi - start) / (time.perf_counter() - t0)
                log(f"encoded {hi:,}/{n:,} ({rate:.0f} passages/s, eta {(n - hi) / rate / 60:.1f} min)")
        finally:
            if pool is not None:
                model.stop_multi_process_pool(pool)
    else:
        log(f"all {n:,} passages already encoded; rebuilding the {factory} index from embeddings.npy")

    index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        sample = np.random.default_rng(0).choice(n, size=min(n, 200_000), replace=False)
        log(f"training {factory} on {len(sample):,} vectors")
        index.train(np.asarray(emb[np.sort(sample)], dtype=np.float32))
    for lo in range(0, n, chunk_size):
        index.add(np.asarray(emb[lo : lo + chunk_size], dtype=np.float32))
    faiss.write_index(index, str(out_dir / "index.faiss"))

    meta = {
        "embedder": spec.name,
        "query_prefix": spec.query_prefix,
        "passage_prefix": spec.passage_prefix,
        "max_seq_length": spec.max_seq_length,
        "dim": dim,
        "faiss_factory": factory,
        "search_params": default_search_params(factory) if search_params is None else search_params,
        "n_passages": n,
        "corpus": corpus_name,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    size_mb = (out_dir / "index.faiss").stat().st_size / 1e6
    log(f"index written to {out_dir} ({index.ntotal:,} vectors, {factory}, {size_mb:,.0f} MB)")
    return out_dir


def _write_passages(passages: Sequence[Passage], path: Path) -> None:
    table = pa.table(
        {
            "docid": [p.docid for p in passages],
            "title": [p.title for p in passages],
            "text": [p.text for p in passages],
        }
    )
    pq.write_table(table, path, compression="zstd")


def _read_progress(path: Path | None, spec: EmbedderSpec, n: int) -> dict:
    key = {"embedder": spec.name, "n": n, "max_seq_length": spec.max_seq_length, "passage_prefix": spec.passage_prefix}
    fresh = {**key, "done": 0}
    if path is None or not path.exists():
        return fresh
    progress = json.loads(path.read_text())
    # Anything that changes the vectors invalidates them. Keys missing from an
    # older progress.json are treated as matching so existing runs still resume.
    if any(progress.get(k, v) != v for k, v in key.items()):
        return fresh
    return {**progress, **key}


def resolve_index_dir(location: str) -> Path:
    """Accept a local directory or a Hugging Face dataset repo id holding a published index."""
    path = Path(location)
    if (path / "meta.json").exists():
        return path
    if path.exists() or location.count("/") != 1:
        raise FileNotFoundError(f"no index at {location!r} (expected meta.json)")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(location, repo_type="dataset", allow_patterns=list(PUBLISHED_FILES)))


class DenseRetriever:
    def __init__(self, index_dir: Path, model=None, device: str | None = None):
        self.meta = json.loads((index_dir / "meta.json").read_text())
        self.spec = EmbedderSpec(
            self.meta["embedder"],
            self.meta["query_prefix"],
            self.meta["passage_prefix"],
            self.meta["max_seq_length"],
        )
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        if params := self.meta.get("search_params", default_search_params(self.meta.get("faiss_factory", ""))):
            faiss.ParameterSpace().set_index_parameters(self.index, params)
        self.passages = pq.read_table(index_dir / "passages.parquet")
        if self.index.ntotal != self.passages.num_rows:
            raise ValueError(
                f"index has {self.index.ntotal} vectors but passages.parquet has {self.passages.num_rows} rows"
            )
        self.model = model or load_embedder(self.spec, device)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, queries: Sequence[str], k: int = 10, batch_size: int = 64) -> list[list[Hit]]:
        if not queries:
            return []
        q = encode(self.model, queries, self.spec.query_prefix, batch_size)
        scores, ids = self.index.search(q, min(k, len(self)))
        flat_ids = [int(i) for row in ids for i in row if i >= 0]
        rows = self.passages.take(pa.array(flat_ids, type=pa.int64())).to_pylist() if flat_ids else []
        lookup = dict(zip(flat_ids, rows, strict=True))

        results = []
        for row_scores, row_ids in zip(scores, ids, strict=True):
            hits = []
            for score, idx in zip(row_scores, row_ids, strict=True):
                if idx < 0:
                    continue
                p = lookup[int(idx)]
                hits.append(Hit(len(hits) + 1, p["docid"], p["title"], p["text"], float(score)))
            results.append(hits)
        return results
