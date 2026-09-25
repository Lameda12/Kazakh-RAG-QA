from pathlib import Path

import pytest
import torch

from kazrag.soup import average_state_dicts, greedy_soup


def test_average_state_dicts_means_floats_and_keeps_int_buffers():
    a = {"w": torch.tensor([1.0, 3.0], dtype=torch.float16), "ids": torch.tensor([0, 1])}
    b = {"w": torch.tensor([3.0, 5.0], dtype=torch.float16), "ids": torch.tensor([0, 1])}
    out = average_state_dicts(iter([a, b]))  # a generator must work: make_soup streams checkpoints
    assert out["w"].dtype == torch.float16
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0], dtype=torch.float16))
    assert torch.equal(out["ids"], torch.tensor([0, 1]))


def test_average_state_dicts_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="different keys"):
        average_state_dicts([{"w": torch.zeros(1)}, {"v": torch.zeros(1)}])
    with pytest.raises(ValueError, match="nothing"):
        average_state_dicts([])


def test_greedy_soup_keeps_only_non_worsening_runs():
    runs = [(Path("seed-1"), 60.0), (Path("seed-2"), 59.0), (Path("seed-3"), 58.0)]
    # adding seed-2 helps, adding seed-3 hurts
    scores = {("seed-1", "seed-2"): 61.0, ("seed-1", "seed-2", "seed-3"): 60.5}
    members, best = greedy_soup(runs, lambda paths: scores[tuple(p.name for p in paths)], log=lambda _: None)
    assert [p.name for p in members] == ["seed-1", "seed-2"]
    assert best == 61.0
