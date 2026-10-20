"""Declared successor: native exam after each isolated geometry learner closes."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "training", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "training-unit",
        "expected-training-source",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    a = p.parse_args()
    if (
        not a.indices
        or len(a.indices) != len(set(a.indices))
        or any(i not in range(40) for i in a.indices)
    ):
        raise ValueError("distinct original40 course indices required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit, load_frozen_experts
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    wait_for_terminal_collection(a.training_unit, wait=True)
    d = json.loads((a.training / "declaration.json").read_text())
    c = json.loads((a.training / "complete.json").read_text())
    if (
        c["status"] != "KINEMATIC_COLUMN_TRAINED_REQUIRES_NATIVE_EXAM"
        or c["source_head"] != a.expected_training_source
        or d["source_head"] != a.expected_training_source
        or c["optimizer_steps"] != 2000
        or c["new_native_steps"] != 0
        or d["parent_checkpoint_hash"] != spec["experts"][0]["checkpoint_hash"]
        or digest(a.training / "candidate.pt") != c["checkpoint_hash"]
    ):
        raise ValueError("closed frozen-column learner commitment differs")
    sources = [
        Path(k).parent.parent
        for k in d["input_file_hashes"]
        if k.endswith("/scripts/rsi_train_kinematic_motion_column.py")
    ]
    if (
        len(sources) != 1
        or subprocess.check_output(
            ["git", "-C", str(sources[0]), "rev-parse", "HEAD"], text=True
        ).strip()
        != a.expected_training_source
    ):
        raise ValueError("one original immutable learner source required")
    for folder in ("src", "scripts"):
        for f in (sources[0] / folder).rglob("*.py"):
            if digest(f) != digest(Path.cwd() / f.relative_to(sources[0])):
                raise ValueError("native exam cannot change training actor/helper code")
    pins.update(d["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.training.iterdir() if f.is_file()})
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.kinematic_column_native_exam.v1",
            source_head=a.expected_head,
            candidate_source=a.expected_training_source,
            candidate_hash=c["checkpoint_hash"],
            input_file_hashes=pins,
            indices=a.indices,
            no_parameter_updates=True,
            original_far_or_private_fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import mujoco
    import torch

    from rosclaw_soccer.rsi.kinematic_motion_residual import create_kinematic_motion_residual
    from rosclaw_soccer.rsi.measured_foot_kinematics import MeasuredFootKinematics
    from scripts.rsi_motion_ppo_pilot import episodes

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    a.mode, a.seed = "evaluate", 92029
    extractor = MeasuredFootKinematics(
        mujoco.MjModel.from_binary_path(str(a.world)), expected_world_hash=digest(a.world)
    )
    parent = load_frozen_experts(spec)[0]
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    model = create_kinematic_motion_residual(parent, extractor)
    payload = torch.load(a.training / "candidate.pt", map_location="cpu", weights_only=True)
    if (
        payload["source_head"] != a.expected_training_source
        or payload["model_kind"] != "kinematic_motion_residual64"
        or payload["optimizer_steps"] != 2000
        or payload["world_hash"] != digest(a.world)
        or payload["parent_checkpoint_hash"] != spec["experts"][0]["checkpoint_hash"]
        or payload["experience_bank_hash"] != c["experience_bank_hash"]
    ):
        raise ValueError("candidate training/geometry identity differs")
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.equal(v, before[k]) for k, v in parent.state_dict().items()):
        raise ValueError("candidate changed immutable predecessor")
    model.eval().requires_grad_(False)
    result = episodes(a, model, bd, pins)
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="KINEMATIC_COLUMN_NATIVE_EPISODES_AND_REPLAY_VERIFIED",
            source_head=a.expected_head,
            candidate_hash=c["checkpoint_hash"],
            results=result,
            new_native_steps=result["new_native_steps"],
            optimizer_steps=0,
            geometry_recomputed_from_current108_in_independent_replay=True,
            parent_unchanged=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
