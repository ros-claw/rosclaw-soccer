import copy
from collections import OrderedDict

import pytest

from rosclaw_soccer.training.checkpoint_continuity import verify_checkpoint_continuity


def snapshots():
    torch = pytest.importorskip("torch")
    before = dict(
        model={"weight": torch.ones(3)},
        optimizer={
            "state": {0: {"step": torch.tensor(2.0)}},
            "param_groups": [{"lr": 1e-4, "params": [0]}],
        },
        rng_snapshot={"cpu_state": torch.arange(8, dtype=torch.uint8), "cuda": []},
    )
    return dict(copy.deepcopy(before), generation=2), before


def verify(a, b):
    verify_checkpoint_continuity(a, b, expected_preceding_generation=2)


def test_exact_chain_is_read_only():
    torch = pytest.importorskip("torch")
    a, b = snapshots()
    original = b["model"]["weight"].clone()
    verify(a, b)
    assert torch.equal(b["model"]["weight"], original)


@pytest.mark.parametrize(
    "kind",
    [
        "model",
        "dtype",
        "shape",
        "lr",
        "optimizer_step",
        "rng",
        "generation",
        "generation_bool",
        "missing",
        "extra",
        "container",
        "nonfinite",
        "unsupported",
        "mapping_key",
    ],
)
def test_silent_resets_and_tampering_are_rejected(kind):
    a, b = snapshots()
    if kind == "model":
        b["model"]["weight"][0] += 1
    elif kind == "dtype":
        b["model"]["weight"] = b["model"]["weight"].double()
    elif kind == "shape":
        b["model"]["weight"] = b["model"]["weight"][None]
    elif kind == "lr":
        b["optimizer"]["param_groups"][0]["lr"] = 1e-3
    elif kind == "optimizer_step":
        b["optimizer"]["state"][0]["step"].zero_()
    elif kind == "rng":
        b["rng_snapshot"]["cpu_state"][0] = 99
    elif kind == "generation":
        a["generation"] = 1
    elif kind == "generation_bool":
        a["generation"] = True
    elif kind == "missing":
        b.pop("optimizer")
    elif kind == "extra":
        b["unannounced_reset"] = True
    elif kind == "container":
        b["rng_snapshot"]["cuda"] = ()
    elif kind == "nonfinite":
        a["model"]["weight"][0] = float("inf")
        b["model"]["weight"][0] = float("inf")
    elif kind == "unsupported":
        a["model"]["extra"] = b["model"]["extra"] = object()
    else:
        a["optimizer"]["state"] = {True: {}}
        b["optimizer"]["state"] = {1: {}}
    with pytest.raises(ValueError):
        verify(a, b)


def test_cycles_fail_boundedly():
    a, b = snapshots()
    a["model"]["cycle"] = a["model"]
    b["model"]["cycle"] = b["model"]
    with pytest.raises(ValueError, match="bounds"):
        verify(a, b)


def test_real_torch_state_dict_and_metadata_are_checked():
    torch = pytest.importorskip("torch")
    a, b = snapshots()
    model = torch.nn.Linear(3, 2)
    b["model"] = model.state_dict()
    a["model"] = copy.deepcopy(b["model"])
    assert type(a["model"]) is OrderedDict
    verify(a, b)
    b["model"]._metadata[""]["version"] += 1
    with pytest.raises(ValueError):
        verify(a, b)
