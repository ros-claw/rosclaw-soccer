"""Fourteen full current-body SIM trials on one consumed post-touch failure.

Fixed named Jacobian actions are research interventions, not a learned policy.
No short-horizon proxy selects the outcome: every action gets a complete H300
body with its own future Foundation/IK/RNN/planner feedback. No teacher or
training qualification is granted, even if this one context can be rescued.
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_release_inputs import prepare
from scripts.rsi_postcontact_release_trace import check_release_trace
from scripts.rsi_stage_release_contact_replay import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if (
        Path(__file__).resolve() != p.source / "scripts/rsi_postcontact_release_authority.py"
        or not time.time() < p.cfg["deadline_unix"] <= time.time() + 86400
    ):
        raise ValueError("exact frozen bounded release authority entry required")
    if args.preflight:
        print("RELEASE14_CLOSED_FAILURE_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available RAM before full original mean construction required")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.postcontact_release_authority_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        original_outcome=p.original_outcome,
        closed_local_pair_hash=p.local_pair_hash,
        course=list(p.course),
        original_source_receipt_hash=p.case["receipt"]["receipt_hash"],
        physical_contexts=1,
        all14_trials_including_failures_required=True,
        source_campaign_completion_not_required=True,
        future_source_body_or_contact_frame_passed_to_actor=False,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    rows = []
    native = 0
    current = replay = None
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
        print("ALLOCATING_FULL_ORIGINAL_MEAN_FOR_RELEASE_DIAGNOSTIC", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("same complete source mean required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        witness = p.case["receipt"]["witnesses"]
        model = mujoco.MjModel.from_binary_path(witness["world"]["path"])
        commitment = load_json_artifact(Path(witness["commitment"]["path"]))
        report = load_json_artifact(Path(witness["report"]["path"]))["stripped_document"]
        seed = report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
        for index, name in enumerate(p.cfg["action_names"]):
            current = dict(index=index, action_name=name)
            write_once(p.root / f"trial-{index}-started.json", current)
            policy, decoder = factory.generate_and_bind_seed(seed=seed)
            replay = CausalImpactReplay(
                model,
                world_hash=p.world_hash,
                kp=np.asarray(p.case["physics"]["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(p.case["physics"]["canonical_pd_kd"], dtype=np.float64),
                model_root=p.model_root,
                foundation=foundation,
                expected_foundation_hash=commitment["foundation_hash"],
                policy=policy,
                decoder=decoder,
                expected_policy_hash=commitment["motor_policy_hash"],
                course=p.course,
                profile=BoundedFirstTouchControl(**p.declaration["candidate"]["parameters"]),
                heading=FootHeadingContactProbe(p.case["job"]["heading_enabled"]),
                impact_enabled=False,
                early_impact_enabled=False,
                release_action_name=name,
            )
            history = {}
            for frame in range(300):
                if (
                    time.time() >= p.cfg["deadline_unix"]
                    or native + replay.attempted_steps + 10 > 42000
                ):
                    raise RuntimeError(
                        "fixed release authority deadline or native budget exhausted"
                    )
                result = replay.step(frame)
                for k, v in result.arrays.items():
                    history.setdefault(k, []).append(v.copy())
            trace = {k: np.asarray(v) for k, v in history.items()}
            file = p.root / f"trial-{index}-trace.npz"
            with file.open("xb") as stream:
                np.savez_compressed(stream, **trace)
            review = replay.review()
            if (
                review["native_integrated_steps"] != 3000
                or review["forecast_integrated_steps"] != 0
            ):
                raise ValueError("one complete3000-step zero-forecast physical trial required")
            control = check_release_trace(trace, review, action_name=name)
            require_prefix(
                trace, p.original_trace, control["first_physically_applied_correction_frame"]
            )
            outcome = label_first_touch(trace).to_dict()
            if name == "baseline" and outcome != p.original_outcome:
                raise ValueError("both complete baseline outcomes must match the original source")
            row = dict(
                index=index,
                action_name=name,
                original_policy_hash=policy["policy_hash"],
                source_receipt_hash=p.case["receipt"]["receipt_hash"],
                arrays=dict(path=str(file), sha256=digest(file)),
                control_review=control,
                causal_review=review,
                outcome=outcome,
                original_prefix_and_baseline_neural_bits_matched=True,
                new_optimizer_updates=0,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"trial-{index}-complete.json", row)
            rows.append(row)
            native += review["native_integrated_steps"]
            print("RELEASE_AUTHORITY_NATIVE_TRIAL_COMPLETE", index, name, outcome, flush=True)
            replay = None
            history = {}
            del policy, decoder, trace
            gc.collect()
        factory._stable()
        if (
            native != 42000
            or len(rows) != 14
            or any(digest(name) != h for name, h in p.pins.items())
        ):
            raise ValueError("whole original14 scope and immutable source inputs required")
        successful = [
            r["action_name"] for r in rows if r["outcome"]["task_pass"] and r["outcome"]["safe"]
        ]
        result = dict(
            schema="soccer.rsi.postcontact_release_authority_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_native_steps=native,
            executed_control_foundation_pairs=4200,
            forecast_steps=0,
            new_optimizer_updates=0,
            physical_contexts=1,
            successful_action_names=successful,
            consumed_context_rescued=bool(successful),
            unsafe_trials=sum(not r["outcome"]["safe"] for r in rows),
            both_original_zero_controls_matched=True,
            teacher_qualified=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
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
                completed_trials=len(rows),
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
