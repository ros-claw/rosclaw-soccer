"""Six complete causal SIM rollouts with prospective60ms impact correction.

Explicitly reuses six original controls. Not an independent controller audit,
new neural learning, the old native kernel, or teacher qualification.
"""

import argparse
import gc
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_early_impact_pilot_inputs import prepare
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_stage_release_contact_replay import digest


def intervention_frames(trace):
    requested = trace["early_impact_feedback_requested_offset_nm"]
    applied = trace["early_impact_feedback_applied_offset_nm"]
    if (
        requested.shape != (300, 1, 29)
        or applied.shape != (300, 1, 10, 29)
        or requested.dtype != np.float64
        or applied.dtype != np.float64
        or not np.isfinite(requested).all()
        or not np.isfinite(applied).all()
    ):
        raise ValueError("whole finite requested/actually applied action schedules required")
    req = np.flatnonzero(np.any(requested != 0, axis=(1, 2)))
    act = np.flatnonzero(np.any(applied != 0, axis=(1, 2, 3)))
    if len(req) and (len(req) > 3 or not np.array_equal(req, np.arange(req[0], req[-1] + 1))):
        raise ValueError("at most one contiguous60ms action allowed")
    if any(frame not in req for frame in act):
        raise ValueError("physical correction outside declared action interval")
    return (None if not len(req) else int(req[0]), None if not len(act) else int(act[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if Path(__file__).resolve() != p.source / "scripts/rsi_early_impact_h300.py":
        raise ValueError("exact frozen full-episode runner required")
    if args.preflight:
        print("SIX_PROSPECTIVE_H300_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before full original-model allocation required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.early_impact_h300_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        early_zero_readback_hash=p.early_zero["report_hash"],
        early_screen_readback_hash=p.early_screen["report_hash"],
        causal_off_calibration_hash=p.causal_calibration["report_hash"],
        original_cases_including_failures_retained=True,
        reused_original_controls=6,
        shared_frozen_controller_algorithms=True,
        independent_qualification_claimed=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    replay, current = None, None
    rows = []
    completed_steps = completed_forecasts = 0
    history = {}
    try:
        import torch

        torch.set_num_threads(1)
        print("ALLOCATING_FULL_FIXED_MEAN", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete original mean model required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        print("ALLOCATING_FULL_FOUNDATION", flush=True)
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for i, case in enumerate(p.cases):
            if time.time() >= p.cfg["deadline_unix"]:
                raise RuntimeError("fixed prospective pilot deadline exhausted")
            current, replay, history = i, None, {}
            write_once(p.root / f"case-{i}-started.json", dict(index=i, job=case["job"]))
            c = load_json_artifact(Path(case["receipt"]["witnesses"]["commitment"]["path"]))
            report = load_json_artifact(Path(case["receipt"]["witnesses"]["report"]["path"]))[
                "stripped_document"
            ]
            seed = report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
            policy, decoder = factory.generate_and_bind_seed(seed=seed)
            replay = CausalImpactReplay(
                model,
                world_hash=p.world_hash,
                kp=np.asarray(case["physics"]["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(case["physics"]["canonical_pd_kd"], dtype=np.float64),
                model_root=p.model_root,
                foundation=foundation,
                expected_foundation_hash=c["foundation_hash"],
                policy=policy,
                decoder=decoder,
                expected_policy_hash=c["motor_policy_hash"],
                course=tuple(c["course"]),
                profile=BoundedFirstTouchControl(**p.declaration["candidate"]["parameters"]),
                heading=FootHeadingContactProbe(case["job"]["heading_enabled"]),
                impact_enabled=False,
                early_impact_enabled=True,
                early_evidence_root=p.root / f"case-{i}-preaction-evidence",
            )
            for frame in range(300):
                if time.time() >= p.cfg["deadline_unix"]:
                    raise RuntimeError("fixed prospective pilot deadline exhausted")
                result = replay.step(frame)
                for key, value in result.arrays.items():
                    history.setdefault(key, []).append(value.copy())
            trace = {k: np.asarray(v) for k, v in history.items()}
            file = p.root / f"case-{i}-trace.npz"
            with file.open("xb") as stream:
                np.savez_compressed(stream, **trace)
            requested, applied = intervention_frames(trace)
            require_prefix(trace, case["trace"], applied)
            review = replay.review()
            if (
                review["native_integrated_steps"] != 3000
                or review["forecast_integrated_steps"] > 1800
            ):
                raise ValueError("one complete bounded prospective rollout required")
            outcome = label_first_touch(trace).to_dict()
            row = dict(
                index=i,
                job=case["job"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_outcome=case["receipt"]["outcome"],
                causal_rollout_outcome=outcome,
                first_requested_correction_frame=requested,
                first_physically_applied_correction_frame=applied,
                whole_preintervention_prefix_and_current_neural_bits_matched=True,
                arrays=dict(path=str(file), sha256=digest(file)),
                causal_review=review,
                new_neural_policy_trained=False,
                independent_physics_qualified=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{i}-complete.json", row)
            rows.append(row)
            completed_steps += replay.integrated_steps
            completed_forecasts += review["forecast_integrated_steps"]
            print("EARLY_H300_CASE_COMPLETE", i, outcome, flush=True)
            replay = None
            del policy, decoder, report, trace
            history = {}
            gc.collect()
        factory._stable()
        if (
            completed_steps != 18000
            or completed_forecasts > 10800
            or any(digest(name) != h for name, h in p.pins.items())
        ):
            raise ValueError("complete whole6 bounded pilot and immutable inputs required")
        before = [row["original_outcome"] for row in rows]
        after = [row["causal_rollout_outcome"] for row in rows]
        complete = dict(
            schema="soccer.rsi.early_impact_h300_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_causal_rollout_steps=completed_steps,
            actual_forecast_steps=completed_forecasts,
            reused_original_controls=6,
            original_passes=sum(o["task_pass"] for o in before),
            candidate_passes=sum(o["task_pass"] for o in after),
            rescues=sum(
                not a["task_pass"] and b["task_pass"] for a, b in zip(before, after, strict=True)
            ),
            regressions=sum(
                a["task_pass"] and not b["task_pass"] for a, b in zip(before, after, strict=True)
            ),
            added_unsafe=sum(
                a["safe"] and not b["safe"] for a, b in zip(before, after, strict=True)
            ),
            new_optimizer_updates=0,
            new_neural_policy_trained=False,
            independent_physics_qualified=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(p.root / "complete.json", complete)
    except BaseException as error:
        if replay is not None:
            with (p.root / "failure-last-causal-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    qpos=replay.data.qpos,
                    qvel=replay.data.qvel,
                    ctrl=replay.data.ctrl,
                    **{k: np.asarray(v) for k, v in history.items()},
                )
        failure = dict(
            schema="soccer.rsi.early_impact_h300_failure.v1",
            current_case=current,
            completed_cases=len(rows),
            completed_causal_steps=completed_steps,
            completed_forecast_steps=completed_forecasts,
            current_attempted_steps=0 if replay is None else replay.attempted_steps,
            current_integrated_steps=0 if replay is None else replay.integrated_steps,
            current_forecast_attempted_steps=0
            if replay is None
            else replay.teacher.forecaster.attempted_steps,
            error_type=type(error).__name__,
            error=str(error),
            automatic_retry_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        failure["report_hash"] = hash_json(failure)
        write_once(p.root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
