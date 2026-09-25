"""KazQAD loaders.

Reads the official release from github.com/IS2AI/KazQAD (CC BY-SA 4.0). The
Hugging Face copies (issai/kazqad, issai/kazqad-retrieval) are gated, while the
GitHub files are public and identical in content, so no token is needed.

Set KAZQAD_ROOT to a local checkout of the repo's `data/` directory to skip
downloads entirely.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import time
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

GITHUB_ROOT = "https://raw.githubusercontent.com/IS2AI/KazQAD/main/data"
CACHE_DIR = Path(os.environ.get("KAZRAG_CACHE", Path.home() / ".cache" / "kazrag"))

CORPUS_PARTS = [f"information-retrieval/corpus/kazqad-corpus-v1.0-kk-part-{i}.jsonl.gz" for i in (1, 2, 3)]
Split = Literal["train", "validation", "test"]

# Passages whose body has fewer word characters than this are wiki-table debris
# ("\n\n|}"). At 5, 1,420 of 825,309 passages are dropped and none of them is
# judged relevant in any qrels split.
MIN_TEXT_WORD_CHARS = 5

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[\W_]+")


@dataclass(frozen=True, slots=True)
class Passage:
    docid: str
    title: str
    text: str

    @property
    def content(self) -> str:
        """Title + body, the string that gets embedded and read."""
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass(slots=True)
class QAExample:
    """One question with every gold answer across its annotated passages."""

    qid: str
    question: str
    answers: list[str] = field(default_factory=list)
    docids: list[str] = field(default_factory=list)


def resolve(relpath: str) -> Path:
    """Return a local path for a KazQAD data file, downloading it once if needed."""
    if root := os.environ.get("KAZQAD_ROOT"):
        path = Path(root) / relpath
        if not path.exists():
            raise FileNotFoundError(f"{path} not found under KAZQAD_ROOT={root}")
        return path

    path = CACHE_DIR / "kazqad" / relpath
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _download(f"{GITHUB_ROOT}/{relpath}", path)
    return path


def _download(url: str, dest: Path, attempts: int = 6) -> None:
    """Download with HTTP Range resume. Proxies can cut transfers short without
    raising, so completeness is checked against the server's size, and a partial
    file is never moved into the cache."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    total: int | None = None
    last_error: Exception | None = None
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored Range: start over
                    have = 0
                length = resp.headers.get("Content-Length")
                if length is not None:
                    total = have + int(length)
                with tmp.open("ab" if have else "wb") as out:
                    shutil.copyfileobj(resp, out)
        except OSError as exc:
            last_error = exc
        size = tmp.stat().st_size if tmp.exists() else 0
        if (total is not None and size == total) or (total is None and last_error is None):
            tmp.rename(dest)
            return
        time.sleep(min(2**attempt, 16))
    got = tmp.stat().st_size if tmp.exists() else 0
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} after {attempts} attempts: got {got} of {total} bytes ({last_error})")


def clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def iter_corpus(min_word_chars: int = MIN_TEXT_WORD_CHARS) -> Iterator[Passage]:
    """Yield the ~825k Kazakh Wikipedia passages in release order."""
    for part in CORPUS_PARTS:
        with gzip.open(resolve(part), "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                text = clean(row["text"])
                if len(_NON_WORD.sub("", text)) < min_word_chars:
                    continue
                yield Passage(row["docid"], clean(row["title"] or ""), text)


def load_topics(split: Split) -> dict[str, str]:
    """qid -> question for the retrieval task."""
    path = resolve(f"information-retrieval/topics/kazqad-topics-v1.0-kk-{split}.tsv")
    topics = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, question = line.split("\t", 1)
            topics[qid] = question.strip()
    return topics


def load_qrels(split: Split) -> dict[str, set[str]]:
    """qid -> docids judged relevant (grade 1). Grade-0 judgements are dropped."""
    path = resolve(f"information-retrieval/qrels/kazqad-qrels-v1.0-{split}.tsv")
    qrels: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, _, docid, grade = line.split("\t")
        if int(grade) > 0:
            qrels[qid].add(docid)
    return dict(qrels)


def load_reading_comprehension(split: Split) -> list[dict]:
    """Raw SQuAD-style rows: id ("<qid>#<docid>"), title, context, question, answers."""
    path = resolve(f"reading-comprehension/kazqad-reading-comprehension-v1.0-kk-{split}.jsonl")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_nq_translated() -> list[dict]:
    """~61.6k Natural Questions items machine-translated into Kazakh (KazQAD supplementary, training only).

    Same SQuAD-style schema as the reading-comprehension files; answer offsets
    are character-exact and no question overlaps the validation or test splits.
    """
    path = resolve("supplementary/nq-translate-kk/nq-reading-comprehension-translate-kk.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_questions(split: Split) -> list[QAExample]:
    """Group reading-comprehension rows by question for open-domain evaluation."""
    by_qid: dict[str, QAExample] = {}
    for row in load_reading_comprehension(split):
        qid, docid = row["id"].split("#", 1)
        ex = by_qid.setdefault(qid, QAExample(qid, row["question"].strip()))
        ex.docids.append(docid)
        for ans in row["answers"]["text"]:
            if ans not in ex.answers:
                ex.answers.append(ans)
    return list(by_qid.values())


def reading_comprehension_passages() -> list[Passage]:
    """Unique passages that carry an annotated answer, across all splits (~4.5k)."""
    seen: dict[str, Passage] = {}
    for split in ("train", "validation", "test"):
        for row in load_reading_comprehension(split):
            docid = row["id"].split("#", 1)[1]
            if docid not in seen:
                seen[docid] = Passage(docid, clean(row["title"] or ""), clean(row["context"]))
    return list(seen.values())
