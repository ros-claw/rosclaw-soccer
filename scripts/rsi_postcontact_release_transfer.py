"""Fixed release action on all12 local contexts and six original retentions.

Baselines are reused, byte-bound and relabeled; these are not new randomized
worlds. The one-context action-authority discovery is consumed development,
not held-out proof. Every candidate gets its own full causal H300 body.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_release_trace import check_release_trace
from scripts.rsi_postcontact_release_transfer_inputs import prepare
from scripts.rsi_stage_release_contact_replay import digest


def paired_summary(rows):
    if (
        len(rows) != 18
        or [r["index"] for r in rows] != list(range(18))
        or [r["partition"] for r in rows]
        != ["CONSUMED_LOCAL_DEVELOPMENT"] * 12 + ["CONSUMED_ORIGINAL_RETENTION"] * 6
    ):
        raise ValueError("all12 local and six retention rows in fixed order required")
    result = {}
    for name, group in (("local", rows[:12]), ("retention", rows[12:])):
        for r in group:
            if any(
                type(r[side][k]) is not bool
                for side in ("baseline_outcome", "outcome")
                for k in ("task_pass", "safe")
            ):
                raise ValueError("primitive paired outcome booleans required")
        b, a = [r["baseline_outcome"] for r in group], [r["outcome"] for r in group]
        result[name] = dict(
            contexts=len(group),
            baseline_passes=sum(r["task_pass"] for r in b),
            candidate_passes=sum(r["task_pass"] for r in a),
            rescues=sum(not x["task_pass"] and y["task_pass"] for x, y in zip(b, a, strict=True)),
            regressions=sum(
                x["task_pass"] and not y["task_pass"] for x, y in zip(b, a, strict=True)
            ),
            added_unsafe=sum(x["safe"] and not y["safe"] for x, y in zip(b, a, strict=True)),
        )
    result["fixed_action_locally_promising"] = (
        result["local"]["rescues"] > result["local"]["regressions"]
        and result["local"]["added_unsafe"] == 0
        and result["retention"]["regressions"] == result["retention"]["added_unsafe"] == 0
    )
    # A promising development intervention still needs robustness and a causal
    # teacher policy before any student is trained. Never automatic admission.
    result.update(teacher_qualified=False, training_authorized=False, promotion_authorized=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_postcontact_release_transfer.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded release transfer entry required")
    if args.preflight:
        print(
            "RELEASE_TRANSFER18_WHOLE_CONSUMED_COHORT_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True
        )
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before sole original mean constructor required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.postcontact_release_transfer_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        local_complete_hash=p.local_complete_hash,
        release_physics_hash=p.release_physics_hash,
        jobs=[{k: v for k, v in j.items() if k != "baseline_trace"} for j in p.jobs],
        action_chosen_from_consumed_single_context=True,
        baseline_rollouts_reused=18,
        original_unique_courses=3,
        local_unique_courses=6,
        original_source_clusters=3,
        private_fresh_accessed=False,
        teacher_qualified=False,
        hardware_authorized=False,
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
        print("ALLOCATING_FULL_ORIGINAL_MEAN_FOR_RELEASE_TRANSFER", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete source mean required")
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
                early_impact_enabled=False,
                release_action_name=p.cfg["action_name"],
            )
            history = {}
            for frame in range(300):
                if (
                    time.time() >= p.cfg["deadline_unix"]
                    or native + replay.attempted_steps + 10 > 54000
                ):
                    raise RuntimeError("fixed transfer18 deadline/native budget exhausted")
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
            ):
                raise ValueError("whole H300 zero-forecast release intervention required")
            control = check_release_trace(trace, review, action_name=p.cfg["action_name"])
            require_prefix(
                trace, job["baseline_trace"], control["first_physically_applied_correction_frame"]
            )
            row = dict(
                current,
                action_name=p.cfg["action_name"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_policy_hash=policy["policy_hash"],
                arrays=dict(path=str(file), sha256=digest(file)),
                outcome=label_first_touch(trace).to_dict(),
                causal_review=review,
                control_review=control,
                original_prefix_and_unmodified_neural_bits_matched=True,
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
            print(
                "RELEASE_TRANSFER_NATIVE_CASE_COMPLETE",
                index,
                row["partition"],
                row["outcome"],
                flush=True,
            )
            replay, history = None, {}
            del trace, policy, decoder
            gc.collect()
        factory._stable()
        if native != 54000 or any(digest(n) != h for n, h in p.pins.items()):
            raise ValueError("whole18 immutable paired transfer cohort required")
        result = dict(
            schema="soccer.rsi.postcontact_release_transfer_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_native_steps=native,
            executed_control_foundation_pairs=5400,
            forecast_steps=0,
            new_optimizer_updates=0,
            baseline_rollouts_reused=18,
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
