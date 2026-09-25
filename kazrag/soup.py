"""Model soups: average the weights of fine-tuned runs that share an initialization.

Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time" (ICML 2022). The uniform
soup averages every run; the greedy soup adds runs in order of validation score
and keeps each one only if the held-out score does not drop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import torch


def average_state_dicts(state_dicts: Iterable[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Element-wise mean of floating-point tensors, accumulated in fp32 and cast back.

    Consumes the iterable one state dict at a time, so a generator of loaded
    checkpoints never holds more than one model plus the running sum. Integer
    buffers (e.g. position ids) are identical across runs and are taken from
    the first state dict.
    """
    total: dict[str, torch.Tensor] = {}
    first: Mapping[str, torch.Tensor] | None = None
    count = 0
    for sd in state_dicts:
        if first is None:
            first = sd
            total = {k: v.to(torch.float32).clone() for k, v in sd.items() if v.is_floating_point()}
        else:
            if sd.keys() != first.keys():
                raise ValueError(f"state dict {count} has different keys; soups need identical architectures")
            for k in total:
                total[k] += sd[k].to(torch.float32)
        count += 1
    if first is None:
        raise ValueError("nothing to average")
    return {k: (total[k] / count).to(v.dtype) if k in total else v.clone() for k, v in first.items()}


def make_soup(model_dirs: Sequence[Path], out_dir: Path) -> Path:
    """Average the QA models in `model_dirs` and save the result (with the first run's tokenizer) to `out_dir`."""
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    def load(d: Path):
        return AutoModelForQuestionAnswering.from_pretrained(d)

    soup = load(model_dirs[0])
    soup.load_state_dict(average_state_dicts(load(d).state_dict() for d in model_dirs))
    out_dir.mkdir(parents=True, exist_ok=True)
    soup.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_dirs[0]).save_pretrained(out_dir)
    return out_dir


def greedy_soup(
    ranked: Sequence[tuple[Path, float]],
    evaluate: Callable[[list[Path]], float],
    log: Callable[[str], None] = print,
) -> tuple[list[Path], float]:
    """Greedy soup over runs sorted best-first by their own validation score.

    `evaluate` scores the soup of the given runs (higher is better). A run joins
    only if the soup's score does not drop, so the result is never worse than
    the best single run on the validation set.
    """
    if not ranked:
        raise ValueError("no runs to soup")
    members, best = [ranked[0][0]], ranked[0][1]
    for path, _ in ranked[1:]:
        score = evaluate([*members, path])
        keep = score >= best
        log(f"greedy soup + {path.name}: {score:.2f} ({'kept' if keep else 'dropped'})")
        if keep:
            members, best = [*members, path], score
    return members, best
