"""160 new H300 bodies on predeclared public40; unqualified teacher data only.

Serial shared immutable Foundation weights, fresh body/decoder/history per arm.
No parallel-speedup, actor-training or whole-team gameplay claim.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.foot_heading_parity import require_heading_parity
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_exit_trace import check_exit_trace
from scripts.rsi_public_four_arm_inputs import prepare
from scripts.rsi_stage_release_contact_replay import digest


def summarize_pairs(pairs):
    if len(pairs) != 80:
        raise ValueError("all forty courses and both heading pairs required")
    for i, pair in enumerate(pairs):
        if pair["course_index"] != i // 2 or pair["heading_enabled"] is not bool(i % 2):
            raise ValueError("fixed original public course/heading order required")
    result = {}
    for name, selected in (
        ("all", pairs),
        ("heading_off", pairs[::2]),
        ("heading_on", pairs[1::2]),
    ):
        b, t = [p["baseline_outcome"] for p in selected], [p["teacher_outcome"] for p in selected]
        if any(type(r[k]) is not bool for r in (*b, *t) for k in ("task_pass", "safe")):
            raise ValueError("primitive measured task/safety required")
        result[name] = dict(
            contexts=len(selected),
            baseline_passes=sum(r["task_pass"] for r in b),
            teacher_passes=sum(r["task_pass"] for r in t),
            rescues=sum(not x["task_pass"] and y["task_pass"] for x, y in zip(b, t, strict=True)),
            regressions=sum(
                x["task_pass"] and not y["task_pass"] for x, y in zip(b, t, strict=True)
            ),
            added_unsafe=sum(x["safe"] and not y["safe"] for x, y in zip(b, t, strict=True)),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_collect_public_four_arm.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded public40 collection entry required")
    if args.preflight:
        print("PUBLIC40_ALL_COURSES_PREFLIGHTED_NO_NEW_BODIES", flush=True)
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
        schema="soccer.rsi.public_four_arm40_declaration.v1",
        config=p.cfg,
        plan=p.plan,
        input_file_hashes=p.pins,
        selected_owned_parity_hash=p.owned_parity_hash,
        complete14_bank_or_realtime_qualification_claimed=False,
        actor_privileged_current_state_teacher_only=True,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    rows, pairs, native, forecast = [], [], 0, 0
    replay, current, history = None, None, {}
    current_charged = False
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
        print("ALLOCATING_ORIGINAL_FULL_MEAN_FOR_PUBLIC40", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete original recurrent mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        anchor = p.cases[0]
        w = anchor["receipt"]["witnesses"]
        commitment = load_json_artifact(Path(w["commitment"]["path"]))
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(w["world"]["path"])
        for job in p.plan["records"]:
            originals = {}
            for ai, arm in enumerate(job["arms"]):
                current_charged = False
                ci = job["index"]
                teacher = arm["postcontact_exit_enabled"]
                heading = arm["heading_enabled"]
                current = dict(
                    course_index=ci,
                    arm_index=ai,
                    arm=arm,
                    course=job["course"],
                    sampling_seed=job["sampling_seed"],
                )
                write_once(p.root / f"course-{ci}-arm-{ai}-started.json", current)
                started = time.perf_counter()
                policy, decoder = factory.generate_and_bind_seed(seed=job["sampling_seed"])
                options = (
                    dict(
                        postcontact_exit_enabled=True,
                        postcontact_exit_evidence_root=p.root / f"course-{ci}-arm-{ai}-preaction",
                        reuse_owned_postcontact_updates=True,
                    )
                    if teacher
                    else {}
                )
                replay = CausalImpactReplay(
                    model,
                    world_hash=p.world_hash,
                    kp=np.asarray(anchor["physics"]["canonical_pd_kp"], dtype=np.float64),
                    kd=np.asarray(anchor["physics"]["canonical_pd_kd"], dtype=np.float64),
                    model_root=p.model_root,
                    foundation=foundation,
                    expected_foundation_hash=commitment["foundation_hash"],
                    policy=policy,
                    decoder=decoder,
                    expected_policy_hash=policy["policy_hash"],
                    course=tuple(job["course"]),
                    profile=BoundedFirstTouchControl(),
                    heading=FootHeadingContactProbe(heading),
                    impact_enabled=False,
                    **options,
                )
                bind_seconds = time.perf_counter() - started
                history = {}
                for f in range(300):
                    if (
                        time.time() >= p.cfg["deadline_unix"]
                        or native + replay.attempted_steps + 10 > 480000
                    ):
                        raise RuntimeError("declared public40 native/wall budget exhausted")
                    result = replay.step(f)
                    for k, v in result.arrays.items():
                        history.setdefault(k, []).append(v.copy())
                    used = (
                        0 if replay.teacher is None else replay.teacher.forecaster.attempted_steps
                    )
                    if forecast + used > 44800:
                        raise RuntimeError("declared public40 forecast budget exhausted")
                trace = {k: np.asarray(v) for k, v in history.items()}
                file = p.root / f"course-{ci}-arm-{ai}-trace.npz"
                with file.open("xb") as stream:
                    np.savez_compressed(stream, **trace)
                review = replay.review()
                if review["native_integrated_steps"] != 3000 or review[
                    "forecast_integrated_steps"
                ] > (560 if teacher else 0):
                    raise ValueError("whole public H300 body/forecast bounds required")
                control = None
                if teacher:
                    if review.get("postcontact_exit_owned_updates_reused") is not True:
                        raise ValueError("explicit declared owned continuation required")
                    control = check_exit_trace(trace, review)
                    old, old_row = originals[heading]
                    require_prefix(trace, old, control["first_physically_applied_correction_frame"])
                    if old_row["original_policy_hash"] != policy["policy_hash"]:
                        raise ValueError(
                            "all matched arms must use same newly sampled motor policy"
                        )
                elif heading:
                    off, off_row = originals[False]
                    if policy["policy_hash"] != off_row["original_policy_hash"]:
                        raise ValueError("same motor policy before heading treatment required")
                    # The enabled metadata differs intentionally from frame0;
                    # original physical/control/NN tensors must not differ.
                    require_heading_parity(trace, off, frames=30)
                row = dict(
                    current,
                    original_policy_hash=policy["policy_hash"],
                    source_anchor_receipt_hash=anchor["receipt"]["receipt_hash"],
                    arrays=dict(path=str(file), sha256=digest(file)),
                    outcome=label_first_touch(trace).to_dict(),
                    causal_review=review,
                    control_review=control,
                    bind_navigation_seconds=bind_seconds,
                    rollout_total_seconds=time.perf_counter() - started,
                    new_optimizer_updates=0,
                    teacher_qualified=False,
                    training_authorized=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                )
                row["report_hash"] = hash_json(row)
                write_once(p.root / f"course-{ci}-arm-{ai}-complete.json", row)
                rows.append(row)
                native += 3000
                forecast += review["forecast_integrated_steps"]
                current_charged = True
                if not teacher:
                    originals[heading] = (trace, row)
                else:
                    pairs.append(
                        dict(
                            course_index=ci,
                            heading_enabled=heading,
                            baseline_hash=old_row["report_hash"],
                            teacher_hash=row["report_hash"],
                            baseline_outcome=old_row["outcome"],
                            teacher_outcome=row["outcome"],
                        )
                    )
                print("PUBLIC40_NATIVE_BODY_COMPLETE", ci, ai, row["outcome"], flush=True)
                replay, history = None, {}
                del policy, decoder
                gc.collect()
        factory._stable()
        if len(rows) != 160 or native != 480000 or any(digest(n) != h for n, h in p.pins.items()):
            raise ValueError("all public40 original-source-pinned four-arm bodies required")
        result = dict(
            schema="soccer.rsi.public_four_arm40_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            pairs=pairs,
            summary=summarize_pairs(pairs),
            unique_initial_states=40,
            generator_seed_clusters=10,
            new_native_rollouts=160,
            new_native_steps=native,
            forecast_steps=forecast,
            executed_control_foundation_pairs=48000,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            private_fresh_accessed=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if history:
            with (p.root / "failure-partial-trace.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray(v) for k, v in history.items()})
        write_once(
            p.root / "failure.json",
            dict(
                current=current,
                completed_bodies=len(rows),
                completed_native_steps=native,
                completed_forecast_steps=forecast,
                current_attempted_steps=0
                if replay is None or current_charged
                else replay.attempted_steps,
                current_forecast_attempts=0
                if replay is None or replay.teacher is None or current_charged
                else replay.teacher.forecaster.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
