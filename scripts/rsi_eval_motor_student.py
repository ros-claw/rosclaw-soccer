"""Evaluate initial and trained recurrent29 controllers on all40 courses, SIM only."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("train-root", "bank", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "train-unit", "expected-training-hash", "expected-bank-hash"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean sealed evaluation source and unused output required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                args.train_unit,
                "-p",
                "MainPID",
                "-p",
                "SubState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != dict(MainPID="0", SubState="exited", Result="success", ExecMainStatus="0"):
        raise ValueError("training must actually have closed successfully")
    if (
        digest(args.train_root / "complete.json") != args.expected_training_hash
        or digest(args.bank / "complete.json") != args.expected_bank_hash
    ):
        raise ValueError("training/bank identity differs")
    trained = json.loads((args.train_root / "complete.json").read_text())
    td = json.loads((args.train_root / "declaration.json").read_text())
    bank = json.loads((args.bank / "declaration.json").read_text())
    if (
        trained["optimizer_steps"] != 3000
        or trained["bank_complete_hash"] != args.expected_bank_hash
        or trained["parameter_change_l2_squared"] <= 0
    ):
        raise ValueError("actual positive optimizer updates on this bank required")
    pins = dict(td["input_file_hashes"])
    pins.update(bank["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.bank.iterdir() if p.is_file()})
    pins.update({str(p): digest(p) for p in args.train_root.iterdir() if p.is_file()})
    for p in [args.world] + [
        args.model_root / s
        for s in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(p)) != digest(p):
            raise ValueError("original native world required")
    check_pins(pins)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
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
    from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from rosclaw_soccer.rsi.neural_motor_student import create_motor_student
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motor_student_rollout import rollout_student

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
    if len(choices) != 1:
        raise ValueError("one unchanged prefix affordance required")
    prior = json.loads(choices[0].read_text())
    affordance = SkillEntryAffordance(
        tuple(prior["ball_offset_xy_m"]),
        prior["historical_frame"],
        prior["historical_forward_speed_mps"],
        prior["historical_artifact_hash"],
    )
    if affordance.contract_hash != prior["contract_hash"]:
        raise ValueError("original prefix prior differs")
    checkpoints = [
        ("initial", args.train_root / "initial.pt", trained["initial_checkpoint_hash"]),
        ("trained", args.train_root / trained["checkpoint"], trained["checkpoint_hash"]),
    ]
    models = {}
    for variant, path, expected in checkpoints:
        if digest(path) != expected:
            raise ValueError("checkpoint bytes differ")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["optimizer_steps"] != (0 if variant == "initial" else 3000)
            or payload["seed"] != td["seed"]
            or payload["bank_hash"] != args.expected_bank_hash
            or payload["source_head"] != trained["source_head"]
            or payload["hidden_size"] != 128
            or payload["action_limit"] != 8
        ):
            raise ValueError("motor checkpoint architecture/provenance differs")
        model = create_motor_student(
            hidden_size=payload["hidden_size"], action_limit=payload["action_limit"]
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("nonfinite checkpoint tensors")
        models[variant] = model
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.recurrent_motor_physics40.v1",
            source_head=head,
            input_file_hashes=pins,
            bank_complete_hash=args.expected_bank_hash,
            training_complete_hash=args.expected_training_hash,
            seed=td["seed"],
            prefix_foundation_frames=20,
            student_control_frames=280,
            role="first_touch",
            teacher_fallback_after_frame20=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    world = mujoco.MjModel.from_binary_path(str(args.world))
    results = []
    for variant, _, weight_hash in checkpoints:
        policy = models[variant]
        for record in bank["rows"]:
            item = record["course"]
            index = item["course_index"]
            folder = args.output / f"{variant}-course{index:02d}"
            folder.mkdir()
            nav = G1SonicNavigation(
                args.model_root,
                item["agent_id"],
                SonicNavigationConfig(
                    maximum_frames=300,
                    model_variant="low_latency",
                    inference_threads=1,
                    record_default_planner_events=True,
                ),
            )
            body = G1TargetEpisode(
                world,
                world_hash=digest(args.world),
                kp=nav.backend.kp.copy(),
                kd=nav.backend.kd.copy(),
                course=tuple(item["course"]),
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            prefix = nominal_prefix20(nav, body, affordance)
            gold = Path(record["source_artifact"])
            if digest(gold) != record["source_artifact_hash"]:
                raise ValueError("teacher prefix source changed")
            with np.load(gold, allow_pickle=False) as raw:
                for k in prefix:
                    np.testing.assert_array_equal(prefix[k], raw[k][:20])
            np.savez_compressed(folder / "actual-prefix20.npz", **prefix)
            progress = {}
            try:
                trace = rollout_student(policy, body, prefix, progress=progress)
                if body.integrated_steps != body.attempted_steps or body.integrated_steps != 3000:
                    raise ValueError("full native H300 required")
                pd = diagnose_entry_prefix(world, prefix)
                score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
                score["original_suffix_qualified"] = score["qualified"]
                score["common_prefix_physical_safe"] = pd["prefix_physical_safe"]
                if not pd["prefix_physical_safe"]:
                    score["qualified"], score["cost"], score["physical_prefix_and_suffix_safe"] = (
                        False,
                        None,
                        False,
                    )
                artifact = folder / "native.npz"
                np.savez_compressed(artifact, **trace)
                result = dict(
                    course=item,
                    variant=variant,
                    status="COMPLETE",
                    split=record["split"],
                    score=score,
                    prefix_diagnostics=pd,
                    observed_contact_foot=observed_contact_foot(world, trace, global_start_frame=0),
                    causal_student_first_touch=bool(
                        score["qualified"] and pd["post_boundary_control_can_cause_first_touch"]
                    ),
                    native_steps=3000,
                    student_nn_calls=300,
                    student_actions_consumed=280,
                    teacher_actions_consumed_after_boundary=0,
                    weights_hash=weight_hash,
                    artifact=artifact.name,
                    artifact_hash=digest(artifact),
                    body=body.review(),
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
            except Exception as error:
                rows = progress.get("rows", [])
                if rows:
                    np.savez_compressed(
                        folder / "incomplete-control.npz",
                        **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
                    )
                np.savez_compressed(
                    folder / "last-native-state.npz",
                    qpos=body.data.qpos.copy(),
                    qvel=body.data.qvel.copy(),
                    time_sec=np.array([body.data.time]),
                )
                if "last_proposal" in progress:
                    np.savez_compressed(folder / "last-proposal.npz", **progress["last_proposal"])
                result = dict(
                    course=item,
                    variant=variant,
                    split=record["split"],
                    status="EXECUTION_FAILED",
                    error=repr(error),
                    attempted_native_steps=body.attempted_steps,
                    native_steps=body.integrated_steps,
                    score=None,
                    weights_hash=weight_hash,
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
            write_once(folder / "complete.json", result)
            results.append(result)
            print(
                "MOTOR_STUDENT_NATIVE",
                td["seed"],
                variant,
                index,
                result["status"],
                result["score"]["qualified"] if result["score"] is not None else False,
                flush=True,
            )
    check_pins(pins)
    write_once(
        args.output / "complete.json",
        dict(
            seed=td["seed"],
            results=results,
            new_native_steps=sum(r["native_steps"] for r in results),
            initial_qualified=sum(
                r["variant"] == "initial" and r["score"] is not None and r["score"]["qualified"]
                for r in results
            ),
            trained_qualified=sum(
                r["variant"] == "trained" and r["score"] is not None and r["score"]["qualified"]
                for r in results
            ),
            execution_failed=sum(r["status"] != "COMPLETE" for r in results),
            initial_causal_qualified=sum(
                r["variant"] == "initial" and r.get("causal_student_first_touch", False)
                for r in results
            ),
            trained_causal_qualified=sum(
                r["variant"] == "trained" and r.get("causal_student_first_touch", False)
                for r in results
            ),
            optimizer_updates_during_evaluation=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
