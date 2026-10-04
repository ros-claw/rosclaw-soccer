import copy
import json

import pytest

from rosclaw_soccer.rsi.owned_smooth_preview import OwnedSmoothPreview
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview, make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("seed,std,rho", [(0, 0.1, 0.9), (2**32 - 1, 0.15, 0.95), (4, 0.01, 0)])
def test_owned_preview_is_original_json_and_hash_exact(smooth_parent, seed, std, rho):  # noqa: F811
    owned = OwnedSmoothPreview(smooth_parent)
    view = make_sampling_view(smooth_parent, seed=seed, std=std, rho=rho)
    expected, actual = make_preview(view), owned.preview(view)
    assert actual == expected
    assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False
    )
    assert actual["policy_hash"] == hash_json(
        {k: v for k, v in actual.items() if k != "policy_hash"}
    )


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "mean",
        "mean-numeric-type",
        "authority",
        "authority-zero",
        "source",
        "seed",
        "std",
        "rho",
        "extra",
    ],
)
def test_resealed_bad_sampling_data_is_not_grandfathered(smooth_parent, fault):  # noqa: F811
    owned = OwnedSmoothPreview(smooth_parent)
    view = make_sampling_view(smooth_parent, seed=4, std=0.1)
    if fault == "hash":
        view["model_hash"] = "wrong"
    elif fault == "mean":
        view["mean_model"]["residual_layers"][0]["bias"][0] += 0.1
    elif fault == "mean-numeric-type":
        # Python equality aliases 0 and False; canonical JSON must not.
        view["mean_model"]["hardware_authorized"] = 0
    elif fault.startswith("authority"):
        view["hardware_authorized"] = 0 if fault == "authority-zero" else True
    elif fault == "source":
        view["source_hash"] = "wrong"
    elif fault == "seed":
        view["seed"] = True
    elif fault == "std":
        view["std_raw"] = 0.001
    elif fault == "rho":
        view["rho"] = 0.999
    else:
        view["extra"] = "ignored?"
    if fault != "hash":
        view["model_hash"] = hash_json({k: v for k, v in view.items() if k != "model_hash"})
    with pytest.raises(ValueError, match="owned"):
        owned.preview(view)


def test_mean_ownership_and_source_drift_are_explicit(smooth_parent, tmp_path):  # noqa: F811
    owned = OwnedSmoothPreview(smooth_parent)
    view = make_sampling_view(copy.deepcopy(smooth_parent), seed=4, std=0.1)
    smooth_parent["residual_layers"][0]["bias"][0] += 0.1
    assert owned.preview(view) == make_preview(view)
    path = tmp_path / "source.py"
    path.write_text("original")
    owned._pins[str(path)] = hash_bytes(path.read_bytes())
    path.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        owned.preview(view)
