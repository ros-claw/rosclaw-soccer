"""Six full causal G1 rollouts against original C3, impact search OFF only."""

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
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "six-full-causal-replay-zero-impact-v1"
        or cfg.get("impact_enabled") is not False
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("full_causal_rollouts", 6),
                ("native_step_budget", 18000),
                ("forecast_step_budget", 0),
                ("official_horizon_frames", 300),
                ("post_contact_frames", 60),
                ("new_policy_candidates", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "teacher_qualified",
                "training_authorized",
                "private_fresh_accessed",
                "actor_future_event_input_allowed",
            )
        )
    ):
        raise ValueError("exact whole6 impact-OFF18000-step causal controller calibration required")


def compare_frame(actual, source, frame):
    if set(actual) != set(source):
        raise ValueError("all original source capture fields required in causal replay")
    errors = {}
    for key, a in actual.items():
        e = source[key][frame]
        if (
            a.shape != e.shape
            or a.dtype != e.dtype
            or not np.isfinite(a).all()
            or not np.isfinite(e).all()
        ):
            raise ValueError("whole finite typed causal capture required: " + key)
        exact = key.startswith("foundation_neural_") or a.dtype == np.bool_
        if exact and a.tobytes() != e.tobytes():
            raise ValueError("causal current neural capture bits differ: " + key)
        error = float(np.any(a != e)) if a.dtype == np.bool_ else float(np.max(abs(a - e)))
        if error > 1e-8:
            raise ValueError(f"causal source frame{frame} differs:{key}:maxabs{error}")
        errors[key] = error
    return errors


def compare_outcome(actual, original):
    if set(actual) != set(original):
        raise ValueError("whole original outcome fields required")
    for key, e in original.items():
        a = actual[key]
        if type(a) is not type(e) or (
            not np.isfinite([a, e]).all() or abs(a - e) > 1e-8 if type(a) is float else a != e
        ):
            raise ValueError("original outcome differs: " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = original_prepare(args.config, scope_validator=require_scope)
    if Path(__file__).resolve() != p.source / "scripts/rsi_calibrate_causal_impact_replay.py":
        raise ValueError("exact frozen causal calibration runner required")
    if args.preflight:
        print("WHOLE6_CAUSAL_REPLAY_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available before full fixed mean allocation required")
    import torch

    torch.set_num_threads(1)
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.full_causal_replay_calibration_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        original_cases=[c["receipt"]["receipt_hash"] for c in p.cases],
        original_task_and_courses_unchanged=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    replay, current, rows, completed_steps = None, None, [], 0
    try:
        print("ALLOCATING_FULL_FIXED_MEAN", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("original full frozen mean identity required")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        print("ALLOCATING_FULL_FOUNDATION", flush=True)
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        for i, case in enumerate(p.cases):
            if time.time() >= p.cfg["deadline_unix"]:
                raise RuntimeError("causal calibration deadline exceeded")
            current, replay = i, None
            write_once(
                p.root / f"case-{i}-started.json",
                dict(index=i, source_receipt_hash=case["receipt"]["receipt_hash"]),
            )
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
            )
            history = {k: [] for k in case["trace"]}
            maxima = {}
            for frame in range(300):
                if time.time() >= p.cfg["deadline_unix"]:
                    raise RuntimeError("causal calibration deadline exceeded")
                result = replay.step(frame)
                for key, value in result.arrays.items():
                    history[key].append(value.copy())
                # Save mismatch output, not just a failing comparison label.
                try:
                    errors = compare_frame(result.arrays, case["trace"], frame)
                except BaseException:
                    with (p.root / f"case-{i}-frame-{frame}-mismatch.npz").open("xb") as stream:
                        np.savez_compressed(stream, **result.arrays)
                    raise
                for key, error in errors.items():
                    maxima[key] = max(maxima.get(key, 0.0), error)
            out = {k: np.asarray(v) for k, v in history.items()}
            path = p.root / f"case-{i}-causal-trace.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **out)
            review = replay.review()
            outcome = label_first_touch(out).to_dict()
            compare_outcome(outcome, case["receipt"]["outcome"])
            result = dict(
                index=i,
                job=case["job"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_outcome=case["receipt"]["outcome"],
                causal_replay_outcome=outcome,
                maximum_absolute_errors=maxima,
                all300_frames_and_all_current_neural_bits_matched=True,
                arrays=dict(path=str(path), sha256=digest(path)),
                causal_review=review,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            result["report_hash"] = hash_json(result)
            write_once(p.root / f"case-{i}-complete.json", result)
            rows.append(result)
            completed_steps += replay.integrated_steps
            print("FULL_CAUSAL_CASE_COMPLETE", i, flush=True)
            replay = None
            del decoder, policy, history, out, report
            gc.collect()
        if completed_steps != 18000 or any(digest(n) != h for n, h in p.pins.items()):
            raise ValueError("complete bounded causal replay and unchanged sources required")
        result = dict(
            schema="soccer.rsi.full_causal_replay_calibration_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_native_steps=completed_steps,
            forecast_steps=0,
            all_six_source_outcomes_and_neural_bits_matched=True,
            impact_enabled_false_only=True,
            new_policy_candidates=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if replay is not None:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, replay.data, state, spec)
            with (p.root / "failure-last-causal-state.npz").open("xb") as stream:
                np.savez_compressed(stream, integration_state=state)
        failure = dict(
            schema="soccer.rsi.full_causal_replay_calibration_failure.v1",
            current_case=current,
            completed_native_steps=completed_steps,
            current_attempted_steps=0 if replay is None else replay.attempted_steps,
            current_integrated_steps=0 if replay is None else replay.integrated_steps,
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
