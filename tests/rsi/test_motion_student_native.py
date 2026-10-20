"""Opt-in real native+NN smoke; reproducibility is not successful football."""

import json
import os
import subprocess
from pathlib import Path

import pytest


def test_external_untrained_intention_and_frozen_muscle_native_replay():
    location = os.environ.get("ROSCLAW_MOTION_BANK_FIXTURE")
    if location is None:
        pytest.skip("explicit full40 exact neural motion-bank fixture required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        == os.environ["ROSCLAW_MOTION_EXPECTED_HEAD"]
    )
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    output = Path(os.environ["ROSCLAW_MOTION_SMOKE_OUTPUT"])
    assert not output.exists()
    runtime_pins = {
        str(p): digest(p)
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", Path.cwd() / "tests")
        for p in folder.rglob("*.py")
    }
    root = Path(location)
    assert digest(root / "complete.json") == os.environ["ROSCLAW_MOTION_BANK_SHA"]
    bc = json.loads((root / "complete.json").read_text())
    bd = json.loads((root / "declaration.json").read_text())
    assert bc["actions_byte_exact"] == 12000
    check_pins(bd["input_file_hashes"])
    paths = [Path(p) for p in bd["input_file_hashes"]]
    worlds = [p for p in paths if p.suffix == ".mjb"]
    priors = [p for p in paths if p.name == "entry-affordance.json"]
    encoders = [
        p for p in paths if p.name == "model_encoder.onnx" and p.parent.name == "low_latency"
    ]
    assert len(worlds) == len(priors) == len(encoders) == 1
    assert digest(root / bc["artifact"]) == bc["artifact_hash"]
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from rosclaw_soccer.rsi.motor_student_bank import training_observation_statistics
    from rosclaw_soccer.rsi.neural_motion_intention import create_motion_intention_student
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student

    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    with np.load(root / bc["artifact"], allow_pickle=False) as z:
        x, token, train, actor = (
            z[k].copy() for k in ("features", "motion_tokens", "train", "actor_eligible")
        )
    model = create_motion_intention_student()
    mean, std = training_observation_statistics(x, train)
    with torch.no_grad():
        model.observation_mean.copy_(torch.from_numpy(mean))
        model.observation_std.copy_(torch.from_numpy(std))
    model.initialize_training_mean(torch.from_numpy(token[train & actor].mean(axis=(0, 1))))
    model.eval()
    world = mujoco.MjModel.from_binary_path(str(worlds[0]))
    model_root = encoders[0].parent.parent
    nav = G1SonicNavigation(
        model_root,
        "tracker",
        SonicNavigationConfig(
            maximum_frames=300,
            model_variant="low_latency",
            inference_threads=1,
            record_default_planner_events=True,
        ),
    )
    item = bd["rows"][2]
    assert item["course"]["course_index"] == 2

    def body():
        return G1TargetEpisode(
            world,
            world_hash=digest(worlds[0]),
            kp=nav.backend.kp.copy(),
            kd=nav.backend.kd.copy(),
            course=tuple(item["course"]["course"]),
            defaults=G1SonicRunupController.default_angles.copy(),
            record_all_ball_contacts=True,
        )

    def codec():
        # Navigation's pure proposal backend also carries the same qualified
        # muscle decoder; no propose/reference method is used by the codec.
        unused = G1SonicNavigation(
            model_root,
            "tracker",
            SonicNavigationConfig(
                maximum_frames=300, model_variant="low_latency", inference_threads=1
            ),
        )
        return FoundationMotorCodec(unused.backend)

    prior = json.loads(priors[0].read_text())
    affordance = SkillEntryAffordance(
        tuple(prior["ball_offset_xy_m"]),
        prior["historical_frame"],
        prior["historical_forward_speed_mps"],
        prior["historical_artifact_hash"],
    )
    actual = body()
    prefix = nominal_prefix20(nav, actual, affordance)
    raw_path = Path(item["source_artifact"])
    assert digest(raw_path) == item["source_artifact_hash"]
    with np.load(raw_path, allow_pickle=False) as z:
        for k in prefix:
            np.testing.assert_array_equal(prefix[k], z[k][:20])
    raw = rollout_motion_student(model, codec(), actual, prefix, progress={})
    assert actual.integrated_steps == 3000
    proof = replay_motion_student(model, codec(), body(), raw)
    assert proof["additional_native_steps"] == 3000
    assert proof["teacher_encoder_calls"] == proof["teacher_planner_calls"] == 0
    assert proof["student_neural_and_muscle_native_bit_exact"]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "actual-native-motion.npz", **raw)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            optimizer_steps=0,
            seed=17,
            model_kind="motion_intention_fsq64",
            source_head=os.environ["ROSCLAW_MOTION_EXPECTED_HEAD"],
        ),
        output / "initial.pt",
    )
    write_once(
        output / "complete.json",
        dict(
            source_head=os.environ["ROSCLAW_MOTION_EXPECTED_HEAD"],
            runtime_file_hashes=runtime_pins,
            bank_hash=os.environ["ROSCLAW_MOTION_BANK_SHA"],
            actual_native_steps=3000,
            replay=proof,
            optimizer_steps=0,
            score=score_full_entry(world, {k: v[20:] for k, v in raw.items()}),
            artifact_hash=digest(output / "actual-native-motion.npz"),
            initial_checkpoint_hash=digest(output / "initial.pt"),
            development_only=True,
            publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    print(
        "MOTION_MUSCLE_NATIVE_DEVELOPMENT_ONLY",
        json.dumps(
            dict(
                proof=proof,
                score=score_full_entry(world, {k: v[20:] for k, v in raw.items()}),
                optimizer_steps=0,
            )
        ),
        flush=True,
    )
    check_pins(bd["input_file_hashes"])
    check_pins(runtime_pins)
