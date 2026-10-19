"""Whole18 one-factor approach probe; zero postcontact corrective interventions.

Reuse all original controls, including unsafe/late outcomes. This is public
mechanism diagnosis, not a new G0 bank, learned actor, teacher or Fresh exam.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_postcontact_release_transfer import paired_summary
from scripts.rsi_postcontact_release_transfer_inputs import prepare as transfer_prepare
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    fixed = dict(
        profile="constant-forward-approach18-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        new_rollouts=18,
        native_step_budget=54000,
        forecast_step_budget=0,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        baseline_rollouts_reused=18,
        constant_forward_approach=True,
        forward_cap_m_s=1.4,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("whole18 one-factor fixed1.4m/s approach diagnostic required")


def approach_review(trace, baseline):
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
    from rosclaw_soccer.rsi.constant_forward_approach import apply_constant_forward
    from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse

    # The raw *request* changes earlier than the physical projected command.
    # Compare every original bit BEFORE first actual command divergence;
    # at that frame only the PRE-action body must still be byte-identical.
    if set(trace) != set(baseline):
        raise ValueError("unchanged original trace fields required")
    for k in baseline:
        if (
            trace[k].shape != baseline[k].shape
            or trace[k].dtype != baseline[k].dtype
            or not np.isfinite(trace[k]).all()
        ):
            raise ValueError("complete finite original-shaped physical/NN trace required")
    command = trace["navigation_command"]
    different = np.any(command != baseline["navigation_command"], axis=(1, 2))
    positions = np.flatnonzero(different)
    first = int(positions[0]) if positions.size else 300
    for k in baseline:
        if k == "approach_requested_navigation_command":
            continue
        if trace[k][:first].tobytes() != baseline[k][:first].tobytes():
            raise ValueError(
                "physical/NN prefix changed before first navigation intervention: " + k
            )
    if first < 300:
        for k in (
            "canonical_qpos",
            "canonical_qvel",
            "ball_position_before_step_m",
            "ball_linear_velocity_before_step_m_s",
        ):
            if trace[k][first].tobytes() != baseline[k][first].tobytes():
                raise ValueError("same current body required at first changed command")
    if not np.all(command[:, :, 0] == 1.4):
        raise ValueError("same constant1.4 forward cap required over whole probe")
    contact = False
    profile, pulse = BoundedFirstTouchControl(), OriginalIkPlannerPulse(0.08)
    for f in range(300):
        gap = trace["ball_position_before_step_m"][f, 0] - trace["root_pose_xyzw_m"][f, 0, :3]
        raw, clipped = profile.navigation(
            gap_x=float(gap[0]),
            gap_y=float(gap[1]),
            ball_vx=float(trace["ball_linear_velocity_before_step_m_s"][f, 0, 0]),
            before_contact=not contact,
        )
        raw, clipped = apply_constant_forward(raw, clipped, before_contact=not contact, cap=1.4)
        residual, applied = pulse.compose(f, clipped)
        for key, expected in (
            ("approach_requested_navigation_command", raw),
            ("ik_planner_pulse_baseline_command", clipped),
            ("ik_planner_pulse_requested_residual", residual),
            ("ik_planner_pulse_applied_command", applied),
            ("navigation_command", applied),
        ):
            if trace[key][f, 0].tobytes() != np.asarray(expected).tobytes():
                raise ValueError("one-factor current-state navigation audit differs: " + key)
        contact = contact or bool(np.any(trace["force_n"][f] > 1))
    return dict(
        first_changed_projected_command_frame=None if first == 300 else first,
        before_first_changed_command_physical_and_nn_bits_matched=True,
        original_raw_request_not_claimed_unchanged=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = transfer_prepare(args.config, scope_validator=require_scope)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_constant_forward_approach.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded approach entry required")
    if args.preflight:
        print("CONSTANT_FORWARD18_READY_NO_NEW_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before sole full mean constructor required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.constant_forward18_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        jobs=[{k: v for k, v in j.items() if k != "baseline_trace"} for j in p.jobs],
        changed_factor="precontact_forward_request_only_same1.4_cap",
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    rows, native, replay, current, history = [], 0, None, None, {}
    try:
        import mujoco
        import torch

        from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
        from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
        from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
        from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
        from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        torch.set_num_threads(1)
        print("ALLOCATING_ORIGINAL_FULL_MEAN_FOR_CONSTANT_FORWARD18", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete original mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for index, job in enumerate(p.jobs):
            current = {k: v for k, v in job.items() if k != "baseline_trace"}
            write_once(p.root / f"case-{index}-started.json", current)
            case = p.cases[job["source_index"]]
            witness = case["receipt"]["witnesses"]
            commitment = load_json_artifact(Path(witness["commitment"]["path"]))
            report = load_json_artifact(Path(witness["report"]["path"]))["stripped_document"]
            seed = report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
            policy, decoder = factory.generate_and_bind_seed(seed=seed)
            replay = CausalImpactReplay(
                model,
                world_hash=p.world_hash,
                kp=np.asarray(case["physics"]["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(case["physics"]["canonical_pd_kd"], dtype=np.float64),
                model_root=p.model_root,
                foundation=foundation,
                expected_foundation_hash=commitment["foundation_hash"],
                policy=policy,
                decoder=decoder,
                expected_policy_hash=commitment["motor_policy_hash"],
                course=tuple(job["course"]),
                profile=BoundedFirstTouchControl(**p.declaration["candidate"]["parameters"]),
                heading=FootHeadingContactProbe(case["job"]["heading_enabled"]),
                impact_enabled=False,
                constant_forward_approach=True,
            )
            history = {}
            for frame in range(300):
                if (
                    time.time() >= p.cfg["deadline_unix"]
                    or native + replay.attempted_steps + 10 > 54000
                ):
                    raise RuntimeError("fixed approach18 deadline/native budget exhausted")
                result = replay.step(frame)
                for k, v in result.arrays.items():
                    history.setdefault(k, []).append(v.copy())
            trace = {k: np.asarray(v) for k, v in history.items()}
            file = p.root / f"case-{index}-trace.npz"
            with file.open("xb") as stream:
                np.savez_compressed(stream, **trace)
            review = replay.review()
            if (
                review["native_integrated_steps"] != 3000
                or review["forecast_integrated_steps"] != 0
                or review.get("constant_forward_approach") is not True
                or review.get("original_precontact_forward_law_unchanged") is not False
            ):
                raise ValueError("complete disclosed one-factor H300 body required")
            row = dict(
                current,
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_policy_hash=policy["policy_hash"],
                arrays=dict(path=str(file), sha256=digest(file)),
                outcome=label_first_touch(trace).to_dict(),
                causal_review=review,
                control_review=approach_review(trace, job["baseline_trace"]),
                new_optimizer_updates=0,
                teacher_qualified=False,
                training_authorized=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{index}-complete.json", row)
            rows.append(row)
            native += 3000
            print("CONSTANT_FORWARD_NATIVE_CASE_COMPLETE", index, row["outcome"], flush=True)
            replay, history = None, {}
            del trace, policy, decoder
            gc.collect()
        factory._stable()
        if native != 54000 or any(digest(n) != h for n, h in p.pins.items()):
            raise ValueError("whole18 original-source-pinned completion required")
        summary = paired_summary(rows)
        summary["approach_mechanism_locally_promising"] = summary.pop(
            "fixed_action_locally_promising"
        )
        result = dict(
            schema="soccer.rsi.constant_forward18_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_native_steps=native,
            forecast_steps=0,
            new_optimizer_updates=0,
            executed_control_foundation_pairs=5400,
            baseline_rollouts_reused=18,
            **summary,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if history:
            with (p.root / "failure-partial-trace.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in history.items()})
        if replay is not None:
            with (p.root / "failure-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=replay.data.qpos, qvel=replay.data.qvel, ctrl=replay.data.ctrl
                )
        write_once(
            p.root / "failure.json",
            dict(
                current=current,
                completed_cases=len(rows),
                completed_native_steps=native,
                current_attempted_steps=0 if replay is None else replay.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
