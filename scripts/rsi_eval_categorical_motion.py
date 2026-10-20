"""Original H300 native muscle execution and independent categorical replay."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "core-source", "training", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "expected-head",
        "expected-inputs-hash",
        "training-unit",
        "expected-training-source",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--variant", choices=("initial", "candidate"), required=True)
    p.add_argument("--indices", type=int, nargs="+", required=True)
    p.add_argument("--initial-probe", type=Path)
    p.add_argument("--initial-probe-unit")
    a = p.parse_args()
    if len(set(a.indices)) != len(a.indices) or any(i not in range(40) for i in a.indices):
        raise ValueError("unique original course indices required")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    wait_for_terminal_collection(a.training_unit, wait=True)
    td = json.loads((a.training / "declaration.json").read_text())
    tc = json.loads((a.training / "complete.json").read_text())
    if (
        tc["source_head"] != a.expected_training_source
        or td["source_head"] != a.expected_training_source
        or tc["status"] != "CATEGORICAL_MOTION_TRAINED_REQUIRES_NATIVE_EXAM"
        or tc["optimizer_steps"] != 1000
        or tc["validation_sequences"] != 0
    ):
        raise ValueError("closed fixed categorical training required")
    if a.variant == "candidate":
        if not a.initial_probe or not a.initial_probe_unit:
            raise ValueError("actual zero-head native parity prerequisite required")
        wait_for_terminal_collection(a.initial_probe_unit, wait=True)
        probe = json.loads((a.initial_probe / "complete.json").read_text())
        if (
            probe["variant"] != "initial"
            or probe["anchor_hash"] != tc["anchor_hash"]
            or probe["zero_head_native_identity_verified"] is not True
            or [r["course_index"] for r in probe["results"]] != [0, 10, 15, 16]
        ):
            raise ValueError("whole initial four-course parity proof required")
        pins[str(a.initial_probe / "complete.json")] = digest(a.initial_probe / "complete.json")
    for k, h in td["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting categorical training source")
        pins[k] = h
    for name in ("complete.json", "declaration.json"):
        pins[str(a.training / name)] = digest(a.training / name)
    checkpoint = a.training / (a.variant + ".pt")
    expected = tc["checkpoint_hash"] if a.variant == "candidate" else tc["initial_checkpoint_hash"]
    if digest(checkpoint) != expected:
        raise ValueError("trained checkpoint changed")
    pins[str(checkpoint)] = expected
    anchors = [Path(k) for k, h in pins.items() if h == tc["anchor_hash"] and k.endswith(".pt")]
    if len(anchors) != 1:
        raise ValueError("one explicit frozen parent required")
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            variant=a.variant,
            indices=a.indices,
            candidate_hash=expected,
            training_source=a.expected_training_source,
            anchor_hash=tc["anchor_hash"],
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        import torch

        from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion
        from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
        from scripts.rsi_online_kinematic_motion import load_model

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        parent = load_model(spec, anchors[0]).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in parent.state_dict().items()}
        model = create_categorical_motion(parent)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload["model_kind"] != "categorical_fsq32_motion64"
            or payload["anchor_hash"] != tc["anchor_hash"]
            or payload["source_head"] != a.expected_training_source
            or payload["optimizer_steps"] != (1000 if a.variant == "candidate" else 0)
            or payload["world_hash"] != digest(Path(spec["world"]))
        ):
            raise ValueError("categorical payload identity differs")
        model.load_state_dict(payload["state_dict"], strict=True)
        for k, v in parent.state_dict().items():
            exact_tensor_tree(v, before[k], path=k)
        rows = episodes(a, spec, bd, pins, model.eval().requires_grad_(False))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CATEGORICAL_NATIVE_AND_NEURAL_REPLAY_VERIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                candidate_hash=expected,
                anchor_hash=tc["anchor_hash"],
                variant=a.variant,
                results=rows,
                new_native_steps=6000 * len(rows),
                zero_head_native_identity_verified=a.variant == "initial",
                optimizer_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                automatic_retry_authorized=False,
                partial_charge_requires_child_receipts=True,
            ),
        )
        raise


def episodes(a, spec, bd, pins, model):
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import digest

    world = mujoco.MjModel.from_binary_path(spec["world"])
    backend = G1SonicRunupController(
        Path(spec["model_root"]),
        G1SonicRunupConfig(model_variant="low_latency"),
        inference_threads=1,
    )
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("original prefix prior required")
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("prefix contract differs")
    result = []
    for i in a.indices:
        record = bd["rows"][i]
        folder = a.output / f"course{i:02d}"
        folder.mkdir()

        def body(record=record):
            return G1TargetEpisode(
                world,
                world_hash=digest(Path(spec["world"])),
                kp=backend.kp.copy(),
                kd=backend.kd.copy(),
                course=tuple(record["course"]["course"]),
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )

        actual = body()
        nav = G1SonicNavigation(
            Path(spec["model_root"]),
            record["course"]["agent_id"],
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        prefix = nominal_prefix20(nav, actual, prior)
        del nav
        source = Path(record["source_artifact"])
        if digest(source) != record["source_artifact_hash"]:
            raise ValueError("prefix recording changed")
        with np.load(source, allow_pickle=False) as z:
            for k, v in prefix.items():
                exact_array(v, z[k][:20])
        model.calls = 0
        progress = {}
        try:
            trace = rollout_motion_student(
                model, FoundationMotorCodec(backend), actual, prefix, progress=progress
            )
        except Exception:
            rows = progress.get("rows", [])
            if rows:
                np.savez_compressed(
                    folder / "partial.npz",
                    **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
                )
            write_once(
                folder / "failure.json",
                dict(native_steps=actual.integrated_steps, attempted_steps=actual.attempted_steps),
            )
            raise
        np.savez_compressed(folder / "native.npz", **trace)
        model.calls = 0
        independent = body()
        replay = replay_motion_student(model, FoundationMotorCodec(backend), independent, trace)
        if actual.integrated_steps != 3000 or independent.integrated_steps != 3000:
            raise ValueError("whole native horizon required")
        if a.variant == "initial":
            memories = [
                Path(k)
                for k in pins
                if k.endswith(f"/round0/evaluate{i % 4}/course{i:02d}/native.npz")
            ]
            if len(memories) != 1:
                raise ValueError("one original successful training memory required for parity")
            with np.load(memories[0], allow_pickle=False) as z:
                for k in (
                    *PHYSICAL_FIELDS,
                    "student_motion_token",
                    "hidden_after",
                    "student_features",
                    "muscle_proprioception",
                    "consumed_action_isaac",
                    "value_estimate",
                ):
                    exact_array(trace[k], z[k])
        score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
        pd = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
        row = dict(
            course_index=i,
            split=record["split"],
            score=score,
            prefix_diagnostics=pd,
            native_hash=digest(folder / "native.npz"),
            native_steps=6000,
            replay=replay,
        )
        write_once(folder / "complete.json", row)
        result.append(row)
        print("CATEGORICAL_NATIVE_REPLAY", i, score["qualified"], flush=True)
    return result


if __name__ == "__main__":
    main()
