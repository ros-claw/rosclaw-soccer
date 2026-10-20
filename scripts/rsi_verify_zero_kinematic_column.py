"""Native identity test: new zero geometry column must preserve original BC29."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "core-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-head", "expected-inputs-hash"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_portfolio_common import admit, load_frozen_experts
    from scripts.rsi_native_context_recovery_common import check_pins

    spec, bd, pins, cases = admit(a)
    a.output.mkdir(parents=True)
    a.world, a.model_root = Path(spec["world"]), Path(spec["model_root"])
    a.indices, a.mode, a.seed = [0, 10, 15, 19], "evaluate", 92029
    write_once(
        a.output / "declaration.json",
        dict(
            schema="soccer.rsi.zero_geometry_column_identity.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            indices=a.indices,
            optimizer_steps=0,
            baseline_checkpoint_hash=spec["experts"][0]["checkpoint_hash"],
            planned_new_native_steps=24000,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.kinematic_motion_residual import create_kinematic_motion_residual
    from rosclaw_soccer.rsi.measured_foot_kinematics import MeasuredFootKinematics
    from scripts.rsi_motion_ppo_pilot import episodes
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import digest

    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = mujoco.MjModel.from_binary_path(str(a.world))
    extractor = MeasuredFootKinematics(model, expected_world_hash=digest(a.world))
    parent = load_frozen_experts(spec)[0]
    student = create_kinematic_motion_residual(parent, extractor).eval().requires_grad_(False)
    result = episodes(a, student, bd, pins)
    for row in result["results"]:
        i = row["course_index"]
        with (
            np.load(cases[0][i][1], allow_pickle=False) as old,
            np.load(a.output / f"course{i:02d}/native.npz", allow_pickle=False) as new,
        ):
            for key in (
                *PHYSICAL_FIELDS,
                "student_features",
                "student_motion_token",
                "muscle_proprioception",
                "consumed_action_isaac",
                "hidden_after",
                "value_estimate",
            ):
                exact_array(new[key], old[key])
        for key, value in row["score"].items():
            if cases[0][i][0]["score"].get(key, value) != value:
                raise ValueError("unchanged baseline gate differs")
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="ZERO_KINEMATIC_COLUMN_NATIVE_PARENT_BIT_EXACT",
            source_head=a.expected_head,
            results=result,
            new_native_steps=result["new_native_steps"],
            optimizer_steps=0,
            zero_column_is_not_learned_skill=True,
            geometry_from_existing_current108_only=True,
            own_replay_recomputes_geometry_neural_muscle_native=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
