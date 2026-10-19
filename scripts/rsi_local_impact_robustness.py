"""Local declared course perturbations with paired current-body H300 physics.

Reuses the closed six source controls and old paired baseline outcomes. New
courses are public development only; even a complete local pass is not a
general teacher qualification or authorization to train/promote a student.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_early_neural_components import load_inputs
from scripts.rsi_early_impact_h300 import intervention_frames
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    fixed = dict(
        profile="local-early-impact-perturbation24-v1",
        mode="SIM_ONLY",
        source_cases=6,
        new_rollouts=24,
        new_native_step_budget=72000,
        forecast_step_budget=21600,
        combined_step_budget=93600,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        reuse_owned_early_updates=True,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        private_fresh_accessed=False,
        automatic_retry=False,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("fixed local24 paired SIM-only perturbation protocol required")


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_local_impact_robustness.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded local24 runner required")
    _, _, pins, complete, modelroot = load_inputs(p.cfg, path, p.source, p.root)
    if modelroot != p.model_root:
        raise ValueError("same whole source Foundation required")
    for name, h in pins.items():
        if name in p.pins and p.pins[name] != h:
            raise ValueError("conflicting local24 predecessor source")
        p.pins[name] = h

    def sealed(file, expected=None):
        h = digest(file)
        if file.is_symlink() or (expected is not None and expected != h):
            raise ValueError("ordinary exact predecessor bytes required")
        if str(file) in p.pins and p.pins[str(file)] != h:
            raise ValueError("predecessor changed during admission")
        d = json.loads(file.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole predecessor logical seal required")
        p.pins[str(file)] = h
        return d

    for kind, schema in (
        ("motor", "soccer.rsi.six_early_original_motor_complete.v1"),
        ("owned", "soccer.rsi.owned_selected_early_feedback_continuation_parity_complete.v1"),
    ):
        require_terminal(p.cfg[kind + "_unit"])
        root = Path(p.cfg[kind + "_root"])
        c = sealed(root / "complete.json", p.cfg[kind + "_complete_file_hash"])
        d = sealed(root / "declaration.json")
        if (
            c["schema"] != schema
            or c["declaration_hash"] != d["report_hash"]
            or len(c["rows"]) != 6
            or c["new_optimizer_updates"] != 0
            or any(
                c[k] is not False
                for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("whole closed motor and selected-nonzero parity proofs required")
        for name, h in d["input_file_hashes"].items():
            if digest(name) != h or (name in p.pins and p.pins[name] != h):
                raise ValueError("whole controller/selected-action proof chain changed")
            p.pins[name] = h
        for i, (row, case) in enumerate(zip(c["rows"], p.cases, strict=True)):
            if sealed(root / f"case-{i}-complete.json") != row or row["index"] != i:
                raise ValueError("all original ordered predecessor cases required")
            if kind == "motor":
                if (
                    row["source_case_hash"] != complete["rows"][i]["report_hash"]
                    or row["review"]["actual_physical_steps"] != 3000
                ):
                    raise ValueError("same independently checked complete motor body required")
            elif (
                row["source_receipt_hash"] != case["receipt"]["receipt_hash"]
                or row["both_selected_forecasts_match_closed_h300_gold_bit_exact"] is not True
                or len(row["predictions"]) != 2
            ):
                raise ValueError("all six selected-action forecasts must be qualified first")
        if kind == "owned" and (
            d["selected_h300_hash"] != complete["report_hash"]
            or c["all_six_selected_action_forecasts_match_closed_h300_gold"] is not True
            or c["combined_integrated_steps"] != 18480
            or c["branch_target_completed"] != 36
        ):
            raise ValueError("same complete selected nonzero parity result required")
    courses = []
    for case in p.cases:
        commitment = json.loads(
            Path(case["receipt"]["witnesses"]["commitment"]["path"]).read_text()
        )
        courses.append(tuple(commitment["course"]))
    p.plan = declare_local_impact_perturbations(tuple(courses))
    p.reused_h300 = complete
    return p


def paired_summary(rows):
    if (
        len(rows) != 12
        or any(type(r["index"]) is not int for r in rows)
        or [r["index"] for r in rows] != list(range(12))
    ):
        raise ValueError("all12 declared paired contexts required")
    for row in rows:
        if len(row["arms"]) != 2 or [a["arm"] for a in row["arms"]] != [
            "original_control",
            "prospective_early_impact",
        ]:
            raise ValueError("ordered original and impact arms required")
        a, b = row["arms"]
        if (
            any(a[k] != b[k] for k in ("source_receipt_hash", "original_policy_hash", "job"))
            or a["job"] != row["job"]
        ):
            raise ValueError("paired arms require identical course and original policy")
    before = [r["arms"][0]["outcome"] for r in rows]
    after = [r["arms"][1]["outcome"] for r in rows]
    for a, b in zip(before, after, strict=True):
        if any(type(v[k]) is not bool for v in (a, b) for k in ("task_pass", "safe")):
            raise ValueError("primitive paired task and safety outcomes required")
    rescue = sum(not a["task_pass"] and b["task_pass"] for a, b in zip(before, after, strict=True))
    regression = sum(
        a["task_pass"] and not b["task_pass"] for a, b in zip(before, after, strict=True)
    )
    unsafe = sum(a["safe"] and not b["safe"] for a, b in zip(before, after, strict=True))
    return dict(
        original_passes=sum(a["task_pass"] for a in before),
        candidate_passes=sum(b["task_pass"] for b in after),
        rescues=rescue,
        regressions=regression,
        added_unsafe=unsafe,
        local_reproducible_net_gain=rescue > regression and unsafe == 0,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if args.preflight:
        print("LOCAL24_ALL_COURSES_AND_PREDECESSORS_READY_NO_ALLOCATION_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available RAM before original full mean construction required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.local_impact_robustness_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        plan=p.plan,
        reused_h300_complete_hash=p.reused_h300["report_hash"],
        old_baseline_and_early_outcomes_reused=[
            dict(
                index=i,
                baseline=case["receipt"]["outcome"],
                early=p.reused_h300["rows"][i]["causal_rollout_outcome"],
            )
            for i, case in enumerate(p.cases)
        ],
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    rows = []
    native = forecasts = 0
    replay = current = None
    history = {}
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
        print("ALLOCATING_FIXED_ORIGINAL_MEAN", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same full original mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for index, job in enumerate(p.plan["jobs"]):
            case = p.cases[job["source_index"]]
            commitment = load_json_artifact(
                Path(case["receipt"]["witnesses"]["commitment"]["path"])
            )
            report = load_json_artifact(Path(case["receipt"]["witnesses"]["report"]["path"]))[
                "stripped_document"
            ]
            seed = report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
            arms = []
            original = None
            for arm, enabled in (("original_control", False), ("prospective_early_impact", True)):
                current = dict(index=index, arm=arm, job=job)
                write_once(p.root / f"pair-{index}-{arm}-started.json", current)
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
                    early_impact_enabled=enabled,
                    early_evidence_root=p.root / f"pair-{index}-preaction-evidence"
                    if enabled
                    else None,
                    reuse_owned_early_updates=enabled,
                )
                history = {}
                for frame in range(300):
                    if time.time() >= p.cfg["deadline_unix"]:
                        raise RuntimeError("local24 declared deadline exhausted")
                    result = replay.step(frame)
                    for k, v in result.arrays.items():
                        history.setdefault(k, []).append(v.copy())
                    fsteps = (
                        0 if replay.teacher is None else replay.teacher.forecaster.attempted_steps
                    )
                    if native + replay.attempted_steps > 72000 or forecasts + fsteps > 21600:
                        raise RuntimeError("local24 fixed physical budget exhausted")
                trace = {k: np.asarray(v) for k, v in history.items()}
                file = p.root / f"pair-{index}-{arm}-trace.npz"
                with file.open("xb") as stream:
                    np.savez_compressed(stream, **trace)
                review = replay.review()
                outcome = label_first_touch(trace).to_dict()
                requested = applied = None
                if enabled:
                    requested, applied = intervention_frames(trace)
                    require_prefix(trace, original, applied)
                else:
                    original = trace
                record = dict(
                    arm=arm,
                    job=job,
                    review=review,
                    outcome=outcome,
                    source_receipt_hash=case["receipt"]["receipt_hash"],
                    original_policy_hash=policy["policy_hash"],
                    owned_continuation_enabled=enabled,
                    first_requested_correction_frame=requested,
                    first_physically_applied_correction_frame=applied,
                    arrays=dict(path=str(file), sha256=digest(file)),
                    teacher_qualified=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                )
                record["report_hash"] = hash_json(record)
                write_once(p.root / f"pair-{index}-{arm}-complete.json", record)
                arms.append(record)
                native += review["native_integrated_steps"]
                forecasts += review["forecast_integrated_steps"]
                print("LOCAL_IMPACT_ARM_COMPLETE", index, arm, outcome, flush=True)
                replay = None
                history = {}
                del policy, decoder, trace
                gc.collect()
            row = dict(index=index, job=job, arms=arms, original_prefix_matched=True)
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"pair-{index}-complete.json", row)
            rows.append(row)
        factory._stable()
        if (
            native != 72000
            or forecasts > 21600
            or any(digest(name) != h for name, h in p.pins.items())
        ):
            raise ValueError("whole immutable local24 study and counters required")
        result = dict(
            schema="soccer.rsi.local_impact_robustness_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_native_steps=native,
            forecast_steps=forecasts,
            new_optimizer_updates=0,
            **paired_summary(rows),
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if history:
            with (p.root / "failure-partial-trace.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in history.items()})
        if replay is not None:
            with (p.root / "failure-body-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=replay.data.qpos, qvel=replay.data.qvel, ctrl=replay.data.ctrl
                )
        write_once(
            p.root / "failure.json",
            dict(
                current=current,
                completed_pairs=len(rows),
                completed_native_steps=native,
                completed_forecast_steps=forecasts,
                current_attempted_steps=0 if replay is None else replay.attempted_steps,
                current_forecast_attempts=0
                if replay is None or replay.teacher is None
                else replay.teacher.forecaster.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
