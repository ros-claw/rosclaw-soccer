"""Recording boundary and action conversion fixtures; not physical evidence."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    G1SonicRunupController,
    _sonic_control_parameters,
)
from rosclaw_soccer.rsi.recorded_entry_review import (
    recorded_target,
    replay_prefix,
    replay_suffix,
    validate_recording,
)


def recording(frames=20):
    shapes = dict(
        canonical_qpos=(43,),
        canonical_qvel=(41,),
        joint_target_rad=(29,),
        pd_requested_torque_nm=(10, 29),
        torque_nm=(10, 29),
        actual_actuator_force_nm=(10, 29),
        pelvis_z_per_substep_m=(10,),
        force_n=(6,),
        ball_position_after_step_m=(3,),
        target_body_canonical_qpos_after_substep=(10, 43),
        target_body_canonical_qvel_after_substep=(10, 41),
        ball_contact_force_by_body_n=(10, 8),
        neural_action_isaac=(29,),
        encoder_reference_poses=(10, 36),
        entry_stage=(),
        entry_command=(3,),
        entry_target_root_xy_m=(2,),
        entry_error_m=(),
    )
    return {
        k: np.zeros(
            (frames,) + s,
            dtype=np.int64
            if k == "entry_stage"
            else np.float32
            if k == "neural_action_isaac"
            else np.float64,
        )
        for k, s in shapes.items()
    }


def test_strict_recording_shapes_and_types_no_physics():
    data = recording()
    validate_recording(data, frames=20, nbody=8)
    data["extra"] = np.zeros(20)
    with pytest.raises(ValueError, match="18-field"):
        validate_recording(data, frames=20, nbody=8)


@pytest.mark.parametrize("kind", ["nan", "dtype", "missing_substep", "negative_force"])
def test_invalid_recording_is_fatal(kind):
    data = recording()
    if kind == "nan":
        data["canonical_qpos"][0, 0] = np.nan
    elif kind == "dtype":
        data["neural_action_isaac"] = data["neural_action_isaac"].astype(np.float64)
    elif kind == "missing_substep":
        data["torque_nm"] = data["torque_nm"][:, :9]
    else:
        data["ball_contact_force_by_body_n"][0, 0, 0] = -1
    with pytest.raises(ValueError):
        validate_recording(data, frames=20, nbody=8)


def test_target_is_exact_recorded_neural_action_permutation_and_scale():
    data = recording()
    data["neural_action_isaac"][0] = np.linspace(-1, 1, 29, dtype=np.float32)
    _, _, scale = _sonic_control_parameters(1.0, (1.0,) * 29)
    target = (
        G1SonicRunupController.default_angles
        + data["neural_action_isaac"][0][ISAACLAB_TO_MUJOCO] * scale
    )
    data["joint_target_rad"][0] = target
    np.testing.assert_array_equal(recorded_target(data, 0), target)
    data["joint_target_rad"][0, 0] += 0.001
    with pytest.raises(AssertionError):
        recorded_target(data, 0)


def test_native_body_clock_alignment_before_step():
    with pytest.raises(ValueError, match="new native body"):
        replay_prefix(
            SimpleNamespace(next_frame=1, model=SimpleNamespace(nbody=8)), recording(), None
        )
    with pytest.raises(ValueError, match="prefix20"):
        replay_suffix(
            SimpleNamespace(next_frame=0, model=SimpleNamespace(nbody=8)),
            recording(280),
            law=None,
            offset_root_m=(0, 0, 0),
            affordance=None,
            source_poses=np.zeros((1000, 36)),
            source_evidence_hash="unused",
            report={},
        )


@pytest.mark.parametrize("variant,candidate", [(0, None), (1, 1)])
def test_external_certified_native_recording(variant, candidate):
    """Explicit external-asset integration test: new recorded-control replay only."""
    import hashlib
    import json
    from pathlib import Path

    location = os.environ.get("ROSCLAW_RECORDED_REVIEW_FIXTURE")
    if location is None:
        pytest.skip("explicit external certified fixture required for native integration")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    root = Path(location)
    if digest(root / "complete.json") != os.environ["ROSCLAW_RECORDED_REVIEW_FIXTURE_SHA"]:
        raise ValueError("external fixture identity differs")
    declaration = json.loads((root / "declaration.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    check_pins(declaration["input_file_hashes"])
    choices = [Path(p) for p in declaration["input_file_hashes"]]
    worlds = [p for p in choices if p.suffix == ".mjb"]
    csvs = [p for p in choices if p.name == "BGI_Shooting_00001.csv"]
    affordances = [p for p in choices if p.name == "entry-affordance.json"]
    assert len(worlds) == len(csvs) == len(affordances) == 1
    pair = complete["results"][variant]
    nominal, teacher = pair["results"]
    folder = root / f"variant{variant}"

    def load(receipt):
        path = folder / receipt["artifact"]
        assert digest(path) == receipt["artifact_hash"]
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    actual = load(nominal)
    receipt = nominal if candidate is None else teacher["candidates"][candidate]
    report = nominal if candidate is None else receipt["report"]
    raw = {k: a[20:] for k, a in actual.items()} if candidate is None else load(receipt)
    prefix = {k: a[:20] for k, a in actual.items()}
    pins = {str(p): digest(p) for p in (worlds + csvs + affordances)}
    import mujoco

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.full_entry_forecast import coupled_entry_options
    from rosclaw_soccer.sim.contracts import hash_json

    stored = json.loads(affordances[0].read_text())
    fields = {
        k: stored[k]
        for k in (
            "ball_offset_xy_m",
            "historical_frame",
            "historical_forward_speed_mps",
            "historical_artifact_hash",
        )
    }
    fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
    affordance = SkillEntryAffordance(**fields)
    assert affordance.contract_hash == stored["contract_hash"]
    model = mujoco.MjModel.from_binary_path(str(worlds[0]))
    poses = np.loadtxt(csvs[0], delimiter=",", skiprows=1, dtype=np.float64)
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    body = G1TargetEpisode(
        model,
        world_hash=digest(worlds[0]),
        kp=kp,
        kd=kd,
        course=tuple(pair["course"]["course"]),
        defaults=G1SonicRunupController.default_angles.copy(),
        record_all_ball_contacts=True,
    )
    replay_prefix(body, prefix, affordance)
    option = coupled_entry_options()[0 if candidate is None else candidate]
    replay_suffix(
        body,
        raw,
        law=option.law,
        offset_root_m=option.offset_root_m,
        affordance=affordance,
        source_poses=poses,
        source_evidence_hash=hash_json(declaration),
        report=report,
    )
    assert body.attempted_steps == body.integrated_steps == 3000
    check_pins(pins)
    print(
        "CERTIFIED_RECORDED_NATIVE_REPLAY",
        variant,
        candidate,
        3000,
        "sha256:" + hashlib.sha256(raw["joint_target_rad"].tobytes()).hexdigest(),
    )


def test_external_completed_matched_velocity_case():
    """Replays one immutable completed case, not its still-running parent shard."""
    import json
    from pathlib import Path

    location = os.environ.get("ROSCLAW_MATCHED_REVIEW_FIXTURE")
    if location is None:
        pytest.skip("explicit completed matched-mode fixture required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = Path(location)
    if digest(folder / "complete.json") != os.environ["ROSCLAW_MATCHED_REVIEW_FIXTURE_SHA"]:
        raise ValueError("completed matched-mode case identity differs")
    row = json.loads((folder / "complete.json").read_text())
    chosen = row["candidates"][row["selected_index"]]
    assert chosen["entry_reference_mode"] == "matched_velocity"
    declaration = json.loads((folder.parent / "declaration.json").read_text())
    check_pins(declaration["input_file_hashes"])
    choices = [Path(p) for p in declaration["input_file_hashes"]]
    worlds = [p for p in choices if p.suffix == ".mjb"]
    csvs = [p for p in choices if p.name == "BGI_Shooting_00001.csv"]
    priors = [p for p in choices if p.name == "entry-affordance.json"]
    assert len(worlds) == len(csvs) == len(priors) == 1
    p = folder / row["artifact"]
    assert digest(p) == row["artifact_hash"]
    pins = {
        str(p): digest(p)
        for p in [folder / "complete.json", folder.parent / "declaration.json", p]
        + worlds
        + csvs
        + priors
    }
    with np.load(p, allow_pickle=False) as a:
        raw = {k: a[k] for k in a.files}
    validate_recording(raw, frames=300, nbody=46)
    import mujoco

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.full_entry_forecast import feedback_entry_laws
    from rosclaw_soccer.sim.contracts import hash_json

    stored = json.loads(priors[0].read_text())
    fields = {
        k: stored[k]
        for k in (
            "ball_offset_xy_m",
            "historical_frame",
            "historical_forward_speed_mps",
            "historical_artifact_hash",
        )
    }
    fields["ball_offset_xy_m"] = tuple(fields["ball_offset_xy_m"])
    affordance = SkillEntryAffordance(**fields)
    assert affordance.contract_hash == stored["contract_hash"]
    model = mujoco.MjModel.from_binary_path(str(worlds[0]))
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    body = G1TargetEpisode(
        model,
        world_hash=digest(worlds[0]),
        kp=kp,
        kd=kd,
        course=tuple(row["course"]["course"]),
        defaults=G1SonicRunupController.default_angles.copy(),
        record_all_ball_contacts=True,
    )
    replay_prefix(body, {k: a[:20] for k, a in raw.items()}, affordance)
    poses = np.loadtxt(csvs[0], delimiter=",", skiprows=1, dtype=np.float64)
    replay_suffix(
        body,
        {k: a[20:] for k, a in raw.items()},
        law=feedback_entry_laws()[chosen["law_index"]],
        offset_root_m=tuple(chosen["offset_root_m"]),
        affordance=affordance,
        source_poses=poses,
        source_evidence_hash=hash_json(declaration),
        report=row["report"],
        entry_reference_mode=chosen["entry_reference_mode"],
    )
    assert body.attempted_steps == body.integrated_steps == 3000
    check_pins(pins)
    print(
        "COMPLETED_MATCHED_VELOCITY_NATIVE_REPLAY",
        3000,
        row["report"]["entry_reference_adaptation"]["entry_intervals"],
    )
