"""Whole-six40ms zero-feedback calibration; no action optimization or learning."""

import argparse
import contextlib
import gc
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_feedback_branch_inputs import prepare
from scripts.rsi_impact_torque_authority_screen import equal
from scripts.rsi_stage_release_contact_replay import digest


def tracker_signature(tracker):
    import torch

    def value(v):
        if isinstance(v, torch.Tensor):
            return dict(
                shape=list(v.shape),
                dtype=str(v.dtype),
                bytes=hash_bytes(v.detach().cpu().numpy().tobytes()),
            )
        if type(v) in (list, tuple):
            return [value(x) for x in v]
        if type(v) is dict:
            return {k: value(x) for k, x in v.items()}
        return v

    return hash_json(
        {k: value(v) for k, v in vars(tracker).items() if k not in ("model", "_torch")}
    )


def ik_signature(ik):
    from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash

    # The probe is a reviewed frozen slots dataclass, not a mutable object graph.
    fields = {k: v for k, v in vars(ik).items() if k not in ("model", "probe")}
    fields["probe_enabled"] = ik.probe.enabled
    return _numeric_graph_hash(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if Path(__file__).resolve() != p.source / "scripts/rsi_calibrate_feedback_branches.py":
        raise ValueError("exact frozen feedback calibration runner required")
    if args.preflight:
        print("WHOLE_SIX_ZERO_FEEDBACK_READY_NO_FULL_MODELS_OR_PHYSICS", flush=True)
        return
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if available < 56 * 1024**3:
        raise ValueError("56GiB available RAM required before full fixed mean allocation")
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.zero_feedback_calibration_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        restore_complete_hash=p.restore["report_hash"],
        native_complete_hash=p.complete["report_hash"],
        original_task_cases_including_failures_retained=True,
        future_source_targets_used_only_for_comparison=True,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    attempted = integrated = plans = currents = residuals = recurrent_completed = 0
    rows = []
    forecaster = None
    current_case = None

    def counters():
        fa = 0 if forecaster is None else forecaster.attempted_steps
        fi = 0 if forecaster is None else forecaster.integrated_steps
        return dict(
            recorded_attempted_steps=attempted,
            recorded_integrated_steps=integrated,
            forecast_attempted_steps=fa,
            forecast_integrated_steps=fi,
            combined_attempted_steps=attempted + fa,
            combined_integrated_steps=integrated + fi,
            planner_runs_attempted=plans,
            current_pairs_completed=currents,
            original_residual_decisions_completed=residuals,
            original_recurrent_updates_completed=recurrent_completed,
            branch_target_attempts=0 if forecaster is None else forecaster.next_target_attempts,
            branch_target_completed=0 if forecaster is None else forecaster.next_target_completed,
        )

    def bounded():
        if (
            time.time() >= p.cfg["deadline_unix"]
            or attempted > 18000
            or (forecaster is not None and forecaster.attempted_steps > 240)
            or plans > 60
            or residuals > 1800
        ):
            raise RuntimeError("fixed feedback diagnostic budget/deadline exhausted; no retry")

    try:
        import onnxruntime as ort
        import torch

        from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
        from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecaster
        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.providers.g1.sonic_causal_capture_state import (
            WIDTHS,
            restore_current_tracker,
        )
        from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch
        from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES, G1FeedbackBranchControl
        from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
        from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
        from rosclaw_soccer.rsi.foot_heading_evidence import FootHeadingCpuReconstruction
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments
        from rosclaw_soccer.sim.root_velocity_reference import root_velocity_world

        torch.set_num_threads(1)
        print("ALLOCATING_FULL_FIXED_MEAN", flush=True)
        mean = load_json_artifact(Path(p.cfg["mean_model_path"]))
        if mean["model_hash"] != p.cfg["mean_model_hash"]:
            raise ValueError("original whole mean model identity differs")
        factory = RecurrentSamplingEpisodeFactory(mean)
        del mean
        gc.collect()
        print("ALLOCATING_FROZEN_FOUNDATION_AND_PLANNER", flush=True)
        foundation = FrozenSonicG1Torch(p.model_root, variant="low_latency", device="cpu")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        planner = ort.InferenceSession(
            str(p.model_root / "planner_sonic.onnx"), options, providers=["CPUExecutionProvider"]
        )

        class CountedPlanner:
            def run(self, *a, **kw):
                nonlocal plans
                bounded()
                if plans >= 60:
                    raise RuntimeError("past planner budget exhausted")
                plans += 1
                return planner.run(*a, **kw)

        class ReferenceReady(Exception):
            pass

        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        physics = p.cases[0]["physics"]
        kp, kd = (
            np.asarray(physics[k], dtype=np.float64) for k in ("canonical_pd_kp", "canonical_pd_kd")
        )
        forecaster = G1FeedbackForecaster(
            model, world_hash=p.world_hash, kp=kp, kd=kd, max_forecast_steps=240
        )
        qi, vi, ai = forecaster._qi, forecaster._vi, forecaster._ai
        bq, bv, ball, bg, pelvis = (
            forecaster._bq,
            forecaster._bv,
            forecaster._ball,
            forecaster._bg,
            forecaster._pelvis,
        )
        if (
            model.geom_size[bg, 0] != 0.11
            or model.body_mass[ball] != 0.43
            or np.any(model.dof_damping[bv : bv + 6])
        ):
            raise ValueError("unchanged regulation ball dimensions/mass/damping required")
        geometry = [
            int(model.body(n).id)
            for n in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            )
        ]
        limits = model.jnt_range[[model.joint(n).id for n in G1_DDS_JOINT_NAMES]]

        def canonical(d):
            return (
                np.concatenate((d.qpos[:7], d.qpos[qi], d.qpos[bq : bq + 7])),
                np.concatenate((d.qvel[:6], d.qvel[vi], d.qvel[bv : bv + 6])),
            )

        for index, case in enumerate(p.cases):
            bounded()
            current_case = index
            write_once(p.root / f"case-{index}-started.json", dict(index=index, job=case["job"]))
            trace = case["trace"]
            commitment = json.loads(
                Path(case["receipt"]["witnesses"]["commitment"]["path"]).read_text()
            )
            if (
                commitment["foundation_hash"] != foundation.qualification.qualification_hash
                or case["physics"] != physics
            ):
                raise ValueError("same source Foundation/gains required in all cases")
            envelope = load_json_artifact(Path(case["receipt"]["witnesses"]["report"]["path"]))
            report = envelope["stripped_document"]
            seed = report["executed_motor_policy"]["step_motor_proof"]["model"]["seed"]
            policy, decoder = factory.generate_and_bind_seed(seed=seed)
            if policy["policy_hash"] != commitment["motor_policy_hash"]:
                raise ValueError("same original complete source policy required")
            cached = None

            def observe(tick, reference, stop=case["frame"] // 20 * 20):
                nonlocal cached
                if tick == stop:
                    cached = reference.copy()
                    raise ReferenceReady

            with contextlib.suppress(ReferenceReady):
                replay_locomotion_segments(
                    CountedPlanner(),
                    trace,
                    report["default_locomotion_planner_events"],
                    2,
                    reference_observer=observe,
                )
            if cached is None:
                raise ValueError("current reference must come from actual past plans")
            names = {"canonical_qpos", "canonical_qvel"} | {
                "foundation_neural_" + k for k in WIDTHS
            }
            tracker = restore_current_tracker(
                foundation,
                cached_reference=cached,
                prefix={k: trace[k][: case["frame"] + 1].copy() for k in names},
                frame=case["frame"],
            )
            currents += 1
            ik = FootHeadingCpuReconstruction(
                model, FootHeadingContactProbe(case["job"]["heading_enabled"])
            )
            d = mujoco.MjData(model)
            q, v = trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
            d.qpos[:7], d.qpos[qi], d.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:43]
            d.qvel[:6], d.qvel[vi], d.qvel[bv : bv + 6] = v[:6], v[6:35], v[35:41]
            mujoco.mj_forward(model, d)
            body = {k: [] for k in BODY_SHAPES}
            previous = np.zeros(12)
            predicted = None
            predictions = []
            for frame in range(300):
                bounded()
                q, v = canonical(d)
                equal(q, trace["canonical_qpos"][frame, 0], "original qpos")
                equal(v, trace["canonical_qvel"][frame, 0], "original qvel")
                ball_velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, ball, ball_velocity, 0)
                current = dict(
                    joint_position_rad=d.qpos[qi],
                    joint_velocity_rad_s=d.qvel[vi],
                    root_pose_xyzw_m=np.concatenate(
                        (d.xpos[pelvis], d.xquat[pelvis][[1, 2, 3, 0]])
                    ),
                    root_velocity_world=root_velocity_world(model, d, pelvis, "body-com"),
                    ball_position_before_step_m=d.xpos[ball],
                    ball_linear_velocity_before_step_m_s=ball_velocity[3:],
                    foot_geometry_position_before_step_m=d.xpos[geometry],
                )
                for key, value in current.items():
                    equal(value, trace[key][frame, 0], "original body " + key)
                    body[key].append(value.copy()[None])
                nav = trace["foundation_navigation_joint_target_rad"][frame, 0]
                delta = ik.before_step(d, frame, d.xpos, nav)
                equal(delta, trace["taskspace_joint_increment_rad"][frame, 0], "original IK")
                nominal = nav + delta
                previous = decoder.delta_at_frame(
                    policy,
                    body,
                    frame=frame,
                    nominal_target=nominal,
                    baseline=nominal[:12],
                    limits=limits[:12],
                    previous=previous,
                    previous_contact_forces=trace["force_n"][frame - 1, 0]
                    if frame
                    else np.zeros(6),
                )
                residuals += 1
                if frame >= 30:
                    recurrent_completed += 1
                equal(previous, trace["motor_delta_rad"][frame, 0], "original recurrent residual")
                equal(
                    decoder.hidden_state,
                    trace["recurrent_residual_hidden_state"][frame, 0],
                    "original recurrent hidden",
                )
                for key, value in decoder.sampled_transition.items():
                    equal(value, trace[key][frame, 0], "original sampling " + key)
                target = nominal.copy()
                target[:12] += previous
                equal(target, trace["joint_target_rad"][frame, 0], "original full target")
                if frame == case["frame"]:
                    live_hash = _numeric_graph_hash(decoder)
                    ik_hash = ik_signature(ik)
                    cached_hash = hash_json(
                        {
                            k: hash_bytes(getattr(d, k).tobytes())
                            for k in ("xpos", "xquat", "cvel", "qacc", "qfrc_constraint")
                        }
                    )
                    tracker_hash = tracker_signature(tracker)
                    for branch in range(2):
                        control = G1FeedbackBranchControl(
                            model,
                            frame=frame,
                            tracker=tracker,
                            decoder=decoder,
                            ik=ik,
                            policy=policy,
                            body_prefix={k: np.asarray(v) for k, v in body.items()},
                            previous_motor_delta_rad=previous,
                        )
                        out = forecaster.forecast(
                            d, target, control=control, first_offset_nm=np.zeros(29)
                        )
                        # Preserve raw successful integration even if comparison rejects it.
                        record = out.next_target_record
                        neural = control.tracker.neural_observation()
                        path = p.root / f"case-{index}-branch-{branch}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(
                                stream,
                                end_integration_state=out.end_integration_state,
                                **out.arrays,
                                **{"next_" + k: v for k, v in record.items()},
                                **{"neural_next_" + k: v.cpu().numpy() for k, v in neural.items()},
                            )
                        predictions.append(
                            dict(
                                path=str(path),
                                sha256=digest(path),
                                donor_hash=out.donor_integration_state_hash,
                            )
                        )
                        if predicted is None:
                            predicted = out
                        else:
                            if (
                                out.end_integration_state.tobytes()
                                != predicted.end_integration_state.tobytes()
                                or any(
                                    out.arrays[k].tobytes() != predicted.arrays[k].tobytes()
                                    for k in out.arrays
                                )
                                or out.donor_integration_state_hash
                                != predicted.donor_integration_state_hash
                                or any(
                                    record[k].tobytes() != predicted.next_target_record[k].tobytes()
                                    for k in record
                                )
                            ):
                                raise ValueError("independent zero feedback branches differ")
                        for k, source in (
                            ("navigation_target_rad", "foundation_navigation_joint_target_rad"),
                            ("taskspace_increment_rad", "taskspace_joint_increment_rad"),
                            ("motor_delta_rad", "motor_delta_rad"),
                            ("joint_target_rad", "joint_target_rad"),
                        ):
                            equal(record[k], trace[source][frame + 1, 0], "predicted next " + k)
                        for name in WIDTHS:
                            if not np.array_equal(
                                neural[name].cpu().numpy(),
                                trace["foundation_neural_" + name][frame + 1],
                            ):
                                raise ValueError(
                                    "predicted next actual neural capture differs: " + name
                                )
                    if (
                        _numeric_graph_hash(decoder) != live_hash
                        or ik_signature(ik) != ik_hash
                        or hash_json(
                            {
                                k: hash_bytes(getattr(d, k).tobytes())
                                for k in ("xpos", "xquat", "cvel", "qacc", "qfrc_constraint")
                            }
                        )
                        != cached_hash
                        or tracker_signature(tracker) != tracker_hash
                        or ik.next_frame != frame
                        or ik.pending is None
                        or ik.pending[0] != frame
                    ):
                        raise ValueError("feedback branch mutated original controller")
                peaks = np.zeros(6)
                for substep in range(10):
                    d.ctrl[ai] = trace["torque_nm"][frame, 0, substep]
                    attempted += 1
                    mujoco.mj_step(model, d)
                    integrated += 1
                    equal(
                        d.actuator_force[ai],
                        trace["actual_actuator_force_nm"][frame, 0, substep],
                        "actual original actuator",
                    )
                    equal(
                        d.xpos[pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, substep],
                        "actual original pelvis",
                    )
                    for c in range(d.ncon):
                        g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
                        if bg in (g1, g2):
                            b = int(model.geom_bodyid[g2 if g1 == bg else g1])
                            if b in forecaster._contact_bodies:
                                force = np.zeros(6)
                                mujoco.mj_contactForce(model, d, c, force)
                                channel = forecaster._contact_bodies.index(b)
                                peaks[channel] = max(
                                    peaks[channel], float(np.linalg.norm(force[:3]))
                                )
                    if predicted is not None and case["frame"] <= frame <= case["frame"] + 1:
                        slot = (frame - case["frame"]) * 10 + substep
                        q, v = canonical(d)
                        equal(
                            q,
                            predicted.arrays["canonical_qpos_after_substep"][slot],
                            "feedback qpos",
                        )
                        equal(
                            v,
                            predicted.arrays["canonical_qvel_after_substep"][slot],
                            "feedback qvel",
                        )
                        equal(
                            d.actuator_force[ai],
                            predicted.arrays["actual_actuator_force_nm"][slot],
                            "feedback actuator",
                        )
                        equal(
                            d.xpos[pelvis, 2],
                            predicted.arrays["pelvis_height_solver_epoch_m"][slot],
                            "feedback pelvis",
                        )
                        equal(
                            trace["pd_requested_torque_nm"][frame, 0, substep],
                            predicted.arrays["requested_pd_torque_nm"][slot],
                            "feedback requested PD",
                        )
                        world_force = np.zeros(3)
                        for w in read_geom_contact_wrenches(model, d, geom_id=bg):
                            if model.geom_bodyid[w.other_geom_id] != 0:
                                world_force += np.asarray(w.force_world_n)
                        equal(
                            world_force,
                            predicted.arrays["robot_force_on_ball_world_n"][slot],
                            "feedback true force",
                        )
                equal(peaks, trace["force_n"][frame, 0], "original full six contact channels")
                if frame == case["frame"]:
                    equal(
                        peaks,
                        predicted.arrays["first_control_peak_force_n"],
                        "branch first control contact peaks",
                    )
                equal(
                    d.xpos[ball],
                    trace["ball_position_after_step_m"][frame, 0],
                    "original after-frame ball",
                )
                ik.after_step(frame, peaks)
            row = dict(
                index=index,
                job=case["job"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_outcome=case["receipt"]["outcome"],
                retrospective_frame=case["frame"],
                predictions=predictions,
                all_original300_residual_ik_and_dynamics_frames_matched=True,
                both_zero_feedback40ms_branches_match_original=True,
                both_branch_end_states_and_arrays_bit_equal=True,
                next_target_decomposition_and_neural_capture_matched=True,
                original_controller_unchanged=True,
                new_closed_loop_policy_trial=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{index}-complete.json", row)
            rows.append(row)
            print("ZERO_FEEDBACK_CASE_COMPLETE", index, flush=True)
            del policy, decoder, tracker, d, ik
            gc.collect()
        bounded()
        factory._stable()
        if any(digest(name) != h for name, h in p.pins.items()):
            raise ValueError("original source/model/input changed")
        result = dict(
            schema="soccer.rsi.zero_feedback_calibration_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            **counters(),
            all_six_original_cases_retained=len(rows) == 6,
            all_six_zero_feedback_branches_qualified=True,
            new_closed_loop_policy_trials=0,
            new_optimizer_updates=0,
            candidate_actions_optimized=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(p.root / "complete.json", result)
    except BaseException as error:
        if forecaster is not None and forecaster.attempted_steps:
            with (p.root / "failure-last-forecast-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    qpos=forecaster._scratch.qpos,
                    qvel=forecaster._scratch.qvel,
                    qacc=forecaster._scratch.qacc,
                    ctrl=forecaster._scratch.ctrl,
                    cached_xpos=forecaster._scratch.xpos,
                    cached_xquat=forecaster._scratch.xquat,
                    cached_cvel=forecaster._scratch.cvel,
                )
        write_once(
            p.root / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                current_case=current_case,
                completed_cases=len(rows),
                **counters(),
                automatic_retry=False,
                constructor_parity_passes_not_instrumented_on_failure=True,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
