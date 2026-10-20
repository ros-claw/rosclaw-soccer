"""Explicit checkpoint fixture: a task failure can still replay exactly."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("variant", ["initial", "trained"])
def test_external_actual_student_nn_and_native_replay(variant):
    location = os.environ.get("ROSCLAW_MOTOR_STUDENT_TRAIN_FIXTURE")
    if location is None:
        pytest.skip("explicit trained motor checkpoint fixture required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    root = Path(location)
    assert digest(root / "complete.json") == os.environ["ROSCLAW_MOTOR_STUDENT_TRAIN_SHA"]
    complete = json.loads((root / "complete.json").read_text())
    declaration = json.loads((root / "declaration.json").read_text())
    check_pins(declaration["input_file_hashes"])
    paths = [Path(p) for p in declaration["input_file_hashes"]]
    worlds = [p for p in paths if p.suffix == ".mjb"]
    priors = [p for p in paths if p.name == "entry-affordance.json"]
    assert len(worlds) == len(priors) == 1
    bank = Path(os.environ["ROSCLAW_MOTOR_STUDENT_BANK_FIXTURE"])
    assert digest(bank / "complete.json") == complete["bank_complete_hash"]
    bd = json.loads((bank / "declaration.json").read_text())
    record = bd["rows"][0]
    path = root / ("initial.pt" if variant == "initial" else complete["checkpoint"])
    assert (
        digest(path)
        == complete["initial_checkpoint_hash" if variant == "initial" else "checkpoint_hash"]
    )
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.neural_motor_student import create_motor_student
    from rosclaw_soccer.rsi.recorded_entry_review import replay_prefix
    from scripts.rsi_motor_student_rollout import (
        replay_student,
        rollout_student,
        validate_student_recording,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = create_motor_student()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    world = mujoco.MjModel.from_binary_path(str(worlds[0]))
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )

    def body():
        return G1TargetEpisode(
            world,
            world_hash=digest(worlds[0]),
            kp=kp,
            kd=kd,
            course=tuple(record["course"]["course"]),
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )

    actual = body()
    raw = Path(record["source_artifact"])
    assert digest(raw) == record["source_artifact_hash"]
    with np.load(raw, allow_pickle=False) as z:
        prefix = {k: z[k][:20].copy() for k in z.files}
    replay_prefix(actual, prefix, prior)
    trace = rollout_student(model, actual, prefix, progress={})
    validate_student_recording(trace, nbody=world.nbody)
    assert actual.integrated_steps == 3000
    verified = replay_student(model, body(), trace)
    assert (
        verified["additional_native_steps"] == 3000 and verified["student_actions_consumed"] == 280
    )
    assert verified["teacher_actions_consumed_after_boundary"] == 0
    assert verified["student_neural_math_and_native_physics_bit_exact"]
    print("DIRECT29_NN_NATIVE_VERIFIED", variant, 6000, flush=True)
    check_pins(declaration["input_file_hashes"])
