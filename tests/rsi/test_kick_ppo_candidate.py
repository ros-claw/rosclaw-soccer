from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_kick_prior_actor import tensors  # noqa: E402

from rosclaw_soccer.rsi.kick_ppo_candidate import admit_pilot, candidate_factory  # noqa: E402
from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor, TorchKickSession  # noqa: E402


def factory():
    return SimpleNamespace(ort_session=TorchKickSession(KickPriorActor(tensors())))


def state():
    return {k: v.clone() for k, v in factory().ort_session.actor.state_dict().items()}


def test_candidate_changes_only_private_readout():
    values = state()
    values["actor.6.bias"] += 0.001
    create = candidate_factory(factory, dict(actor=values))
    first, second = create(), create()
    assert first.ort_session.actor is not second.ort_session.actor
    for key, value in first.ort_session.actor.state_dict().items():
        assert torch.equal(value, values[key])
    with torch.no_grad():
        first.ort_session.actor.actor[6].bias.add_(1)
    assert torch.equal(second.ort_session.actor.actor[6].bias, values["actor.6.bias"])


@pytest.mark.parametrize("defect", ["features", "unchanged", "nonfinite", "shape"])
def test_candidate_frozen_contract_refuses_drift(defect):
    values = state()
    if defect == "features":
        values["actor.0.bias"] += 1
    elif defect == "nonfinite":
        values["actor.6.bias"][0] = float("nan")
    elif defect == "shape":
        values["actor.6.bias"] = values["actor.6.bias"][:1]
    with pytest.raises(ValueError):
        candidate_factory(factory, dict(actor=values))()


def test_candidate_hash_checked_before_interpretation(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("not a checkpoint report")
    with pytest.raises(ValueError, match="exact complete"):
        admit_pilot(path, "sha256:" + "0" * 64)
