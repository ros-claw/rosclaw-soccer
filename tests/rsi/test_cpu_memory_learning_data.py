import numpy as np
import pytest

from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts import rsi_cpu_memory_learning_data as data_module
from scripts.rsi_cpu_memory_learning_data import ordered_cpu_arrays


def values():
    return (
        np.zeros((270, 134)),
        np.repeat(np.arange(3), 90),
        [(np.zeros(12), -1.0) for _ in range(270)],
    )


def outcome():
    return dict(
        reward=0.0,
        high_quality=False,
        clean_foot_only=False,
        minimum_pelvis_z_m=0.7,
        forward_60_m=1.0,
        lateral_over_forward_60=0.2,
        maximum_lateral_excursion_m=1.0,
    )


def test_complete_failed_cpu_trajectory_kept_with_unchanged_terminal_objective():
    x, p, draws = values()
    arrays = ordered_cpu_arrays(x, p, draws, outcome=outcome(), std=0.1, group=2)
    assert arrays["observation"].shape == (270, 134)
    assert np.array_equal(arrays["trajectory_index"], np.full(270, 2))
    assert np.all(arrays["terminal_return"] == terminal_return(outcome()))
    x[:] = 1
    p[:] = 2
    assert not np.any(arrays["observation"])
    assert arrays["phase_index"][0] == 0


@pytest.mark.parametrize(
    "fault", ["short", "nan", "phase", "phase_float", "draws", "latent", "logp", "std", "group"]
)
def test_partial_nonfinite_or_relabelled_cpu_data_rejected(fault):
    x, p, draws = values()
    std = 0.1
    group = 0
    if fault == "short":
        x = x[:-1]
    elif fault == "nan":
        x[7, 3] = np.nan
    elif fault == "phase":
        p[3] = 3
    elif fault == "phase_float":
        p = p.astype(float)
    elif fault == "draws":
        draws.pop()
    elif fault == "latent":
        draws[4] = (np.zeros(11), -1.0)
    elif fault == "logp":
        draws[3] = (np.zeros(12), np.inf)
    elif fault == "std":
        std = 0.05
    else:
        group = True
    with pytest.raises(ValueError):
        ordered_cpu_arrays(x, p, draws, outcome=outcome(), std=std, group=group)


@pytest.mark.parametrize(
    "fault",
    [
        "course",
        "source",
        "model",
        "seed",
        "mean",
        "schema",
        "std",
        "rho",
        "training",
        "authority",
        "unsealed",
    ],
)
def test_wrong_cpu_sampling_provenance_rejected_before_dynamics_replay(
    tmp_path, monkeypatch, fault
):
    runner = tmp_path / "runner.py"
    runner.write_text("# fixture only\n")
    mean = {"model_hash": "fixture-mean"}
    view = dict(
        schema="soccer.rsi.smooth_memory_sampling.v1",
        mean_model=mean,
        seed=17,
        std_raw=0.1,
        rho=0.9,
        training_only=True,
    )
    view["model_hash"] = hash_json(view)
    expected_hash = view["model_hash"]
    raw = dict(
        seed=101,
        lane=2,
        execution_profile="taskspace_plus_motor",
        source_hash=hash_bytes(runner.read_bytes()),
        step_model_hash=expected_hash,
        executed_motor_policy={"step_motor_proof": {"model": view}},
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if fault == "course":
        raw["lane"] = 4
    elif fault == "source":
        raw["source_hash"] = "different"
    elif fault == "model":
        raw["step_model_hash"] = "different"
    elif fault == "seed":
        view["seed"] = 18
    elif fault == "mean":
        view["mean_model"] = {"other": True}
    elif fault == "schema":
        view["schema"] = "other"
    elif fault == "std":
        view["std_raw"] = 0.05
    elif fault == "rho":
        view["rho"] = 0.5
    elif fault == "training":
        view["training_only"] = False
    elif fault == "authority":
        raw["hardware_authorized"] = True
    else:
        view["unsigned_extra"] = 1
    monkeypatch.setattr(data_module, "_sealed", lambda _: raw)

    def should_not_audit(*_):
        pytest.fail("bad provenance reached physics replay")

    monkeypatch.setattr(data_module, "audit_cpu_transfer", should_not_audit)
    with pytest.raises(ValueError, match="declared CPU"):
        data_module.audit_cpu_learning_rollout(
            tmp_path,
            runner,
            mean_model=mean,
            expected_view_hash=expected_hash,
            expected_sampling_seed=17,
            course=(101, 2),
            group=0,
        )
