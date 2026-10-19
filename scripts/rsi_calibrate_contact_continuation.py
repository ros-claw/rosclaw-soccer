"""Whole-six held-target40ms bias calibration; no action selection or neural policy."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches
from rosclaw_soccer.physics.g1_contact_continuation import (
    G1ContactContinuationForecast,
    G1ContactContinuationForecaster,
)
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_impact_screen_inputs import prepare
from scripts.rsi_impact_torque_authority_screen import equal, repeat_equal
from scripts.rsi_stage_release_contact_replay import digest

BIAS_LIMITS = dict(
    maximum_ball_velocity_error_m_s=0.1,
    maximum_ball_position_error_m=0.01,
    maximum_joint_position_error_rad=0.05,
    maximum_pelvis_height_error_m=0.01,
    maximum_initial_contact_impulse_error_n_s=0.05,
)


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "whole-six-held-target40ms-bias-v1"
        or cfg.get("frame_selection") != "RETROSPECTIVE_FIRST_FOOT_CONTACT_CONTROL_FRAME"
        or cfg.get("prediction_input") != "CURRENT_STATE_AND_CURRENT_TARGET_ONLY"
        or cfg.get("oracle_input") != "FUTURE_TARGETS_OFFLINE_VALIDATION_ONLY"
        or cfg.get("bias_limits") != BIAS_LIMITS
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("horizon_substeps", 20),
                ("recorded_replay_substep_budget", 18000),
                ("held_target_forecast_substep_budget", 240),
                ("oracle_validation_substep_budget", 120),
                ("combined_substep_budget", 18360),
                ("new_closed_loop_policy_trials", 0),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
                ("candidate_actions_optimized", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "training_authorized",
                "teacher_qualified",
                "actor_future_event_input_allowed",
            )
        )
    ):
        raise ValueError("exact six-case18360-step calibration with frozen bias limits required")


def model_bias(held, oracle, *, first_contact_substep):
    if type(first_contact_substep) is not int or not 0 <= first_contact_substep < 10:
        raise ValueError("original contact must fall within first control frame")
    for name, shape in (
        ("canonical_qpos_after_substep", (20, 43)),
        ("canonical_qvel_after_substep", (20, 41)),
        ("pelvis_height_solver_epoch_m", (20,)),
        ("robot_force_on_ball_world_n", (20, 3)),
    ):
        for values in (held, oracle):
            a = values[name]
            if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError("whole finite40ms prediction arrays required")
    window = slice(first_contact_substep, first_contact_substep + 10)
    h_impulse = held["robot_force_on_ball_world_n"][window].sum(axis=0) * 0.002
    o_impulse = oracle["robot_force_on_ball_world_n"][window].sum(axis=0) * 0.002
    qerr = abs(held["canonical_qpos_after_substep"] - oracle["canonical_qpos_after_substep"])
    verr = abs(held["canonical_qvel_after_substep"] - oracle["canonical_qvel_after_substep"])
    errors = dict(
        maximum_ball_velocity_error_m_s=float(verr[:, 35:38].max()),
        maximum_ball_position_error_m=float(qerr[:, 36:39].max()),
        maximum_joint_position_error_rad=float(qerr[:, 7:36].max()),
        maximum_pelvis_height_error_m=float(
            abs(held["pelvis_height_solver_epoch_m"] - oracle["pelvis_height_solver_epoch_m"]).max()
        ),
        maximum_initial_contact_impulse_error_n_s=float(abs(h_impulse - o_impulse).max()),
    )
    return dict(
        errors=errors,
        held_robot_impulse_first10_contact_substeps_n_s=h_impulse.tolist(),
        oracle_robot_impulse_first10_contact_substeps_n_s=o_impulse.tolist(),
        held_end_ball_velocity_m_s=held["canonical_qvel_after_substep"][-1, 35:38].tolist(),
        oracle_end_ball_velocity_m_s=oracle["canonical_qvel_after_substep"][-1, 35:38].tolist(),
        frozen_limits_satisfied=all(errors[k] <= limit for k, limit in BIAS_LIMITS.items()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config, scope_validator=require_scope)
    prior_path = Path(p.cfg["prior_impact_readback"])
    if prior_path.is_symlink() or digest(prior_path) != p.cfg["prior_impact_readback_file_hash"]:
        raise ValueError("original whole impact readback bytes required")
    prior = json.loads(prior_path.read_text())
    if (
        prior["report_hash"] != p.cfg["prior_impact_readback_hash"]
        or prior["report_hash"] != hash_json({k: v for k, v in prior.items() if k != "report_hash"})
        or prior["schema"] != "soccer.rsi.impact_torque_screen_readback.v1"
        or prior["all78_saved_proxy_scores_recomputed"] is not True
        or len(prior["rows"]) != 6
        or any(
            prior[k] is not False
            for k in (
                "final_m0_improvement_proven",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or [r["job"] for r in prior["rows"]] != [c["job"] for c in p.cases]
        or [r["original_outcome"] for r in prior["rows"]]
        != [c["receipt"]["outcome"] for c in p.cases]
    ):
        raise ValueError("same original whole6 closed retrospective screen required")
    for name, h in prior["input_file_hashes"].items():
        if name in p.pins:
            if p.pins[name] != h:
                raise ValueError("prior/current scientific input conflict")
        elif digest(name) != h:
            raise ValueError("prior whole impact proof changed")
        p.pins[name] = h
    p.pins[str(prior_path)] = p.cfg["prior_impact_readback_file_hash"]
    if Path(__file__).resolve() != p.source / "scripts/rsi_calibrate_contact_continuation.py":
        raise ValueError("exact frozen calibration source required")
    if mujoco.__version__ != "3.12.0" or mujoco.get_mjcb_control() is not None:
        raise ValueError("original callback-free engine required")
    if args.preflight:
        print("WHOLE_SIX_BIAS_CALIBRATION_PREFLIGHT_NO_PHYSICS_ALLOCATED", flush=True)
        return
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.contact_continuation_calibration_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        native_complete_hash=p.complete["report_hash"],
        original_geometry_hash=p.geometry["report_hash"],
        future_targets_used_only_for_offline_oracle_validation=True,
        candidate_actions_optimized=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    attempted = integrated = 0
    held_workers, oracle_workers, rows = [], [], []
    current_case = None

    def counters():
        ha, hi = (
            sum(getattr(f, attr) for f in held_workers)
            for attr in ("attempted_steps", "integrated_steps")
        )
        oa, oi = (
            sum(getattr(f, attr) for f in oracle_workers)
            for attr in ("attempted_steps", "integrated_steps")
        )
        return dict(
            recorded_attempted_steps=attempted,
            recorded_integrated_steps=integrated,
            held_attempted_steps=ha,
            held_integrated_steps=hi,
            oracle_attempted_steps=oa,
            oracle_integrated_steps=oi,
            combined_attempted_steps=attempted + ha + oa,
            combined_integrated_steps=integrated + hi + oi,
        )

    def bounded():
        c = counters()
        if time.time() >= p.cfg["deadline_unix"] or any(
            c[k] > limit
            for k, limit in (
                ("recorded_attempted_steps", 18000),
                ("held_attempted_steps", 240),
                ("oracle_attempted_steps", 120),
                ("combined_attempted_steps", 18360),
            )
        ):
            raise RuntimeError("fixed calibration deadline/budget exhausted; no automatic retry")

    try:
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        if (
            (model.nq, model.nv, model.nu) != (43, 41, 29)
            or model.opt.timestep != 0.002
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
            or model.nplugin
        ):
            raise ValueError("original native canonical2ms Euler world required")
        qi = [int(model.joint(n).qposadr[0]) for n in G1_DDS_JOINT_NAMES]
        vi = [int(model.joint(n).dofadr[0]) for n in G1_DDS_JOINT_NAMES]
        ai = []
        for n in G1_DDS_JOINT_NAMES:
            ids = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == model.joint(n).id)
            )
            if len(ids) != 1:
                raise ValueError("one original actuator per canonical joint required")
            ai.append(int(ids[0]))
        ball, bg, pelvis = (
            int(model.body("ball").id),
            int(model.geom("ball_geom").id),
            int(model.body("pelvis").id),
        )
        bj = int(model.body_jntadr[ball])
        bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
        if (
            model.geom_size[bg, 0] != 0.11
            or model.body_mass[ball] != 0.43
            or np.any(model.dof_damping[bv : bv + 6])
        ):
            raise ValueError("unchanged original ball dimensions/mass/damping required")

        def canonical(data):
            return (
                np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7])),
                np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])),
            )

        for index, case in enumerate(p.cases):
            bounded()
            current_case = index
            write_once(p.root / f"case-{index}-started.json", dict(index=index, job=case["job"]))
            trace, physics = case["trace"], case["physics"]
            q, v = trace["canonical_qpos"], trace["canonical_qvel"]
            data = mujoco.MjData(model)
            data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = (
                q[0, 0, :7],
                q[0, 0, 7:36],
                q[0, 0, 36:43],
            )
            data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = (
                v[0, 0, :6],
                v[0, 0, 6:35],
                v[0, 0, 35:41],
            )
            mujoco.mj_forward(model, data)
            kwargs = dict(
                expected_world_hash=p.world_hash,
                kp=np.asarray(physics["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(physics["canonical_pd_kd"], dtype=np.float64),
            )
            held_f = G1ContactContinuationForecaster(model, **kwargs, max_forecast_steps=40)
            oracle_f = G1PDTorqueForecaster(model, **kwargs, max_forecast_steps=20)
            held_workers.append(held_f)
            oracle_workers.append(oracle_f)
            held = oracle = None
            eligibility = case["frame"] is not None and case["frame"] < 299
            artifacts, bias = {}, None
            for frame in range(300):
                bounded()
                aq, av = canonical(data)
                equal(aq, q[frame, 0], "whole recorded qpos")
                equal(av, v[frame, 0], "whole recorded qvel")
                if eligibility and frame == case["frame"]:
                    target = trace["joint_target_rad"][frame, 0].copy()
                    prediction = held_f.forecast(data, target, np.zeros(29))
                    repeated = held_f.forecast(data, target, np.zeros(29))
                    repeat_equal(prediction.first, repeated.first)
                    repeat_equal(prediction.second, repeated.second)
                    held = prediction.substep_arrays()
                    # Explicit OFFLINE oracle, isolated from the causal predictor.
                    first = oracle_f.forecast(data, target, np.zeros(29))
                    continuation = mujoco.MjData(model)
                    mujoco.mj_setState(
                        model,
                        continuation,
                        first.end_integration_state,
                        mujoco.mjtState.mjSTATE_INTEGRATION,
                    )
                    second = oracle_f.forecast(
                        continuation, trace["joint_target_rad"][frame + 1, 0].copy(), np.zeros(29)
                    )
                    oracle = G1ContactContinuationForecast(first, second).substep_arrays()
                    repeat_equal(prediction.first, first)
                    for name, value in (
                        ("held", held),
                        ("repeated", repeated.substep_arrays()),
                        ("oracle", oracle),
                    ):
                        path = p.root / f"case-{index}-{name}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, **value)
                        artifacts[name] = dict(path=str(path), sha256=digest(path))
                    bias = model_bias(
                        held,
                        oracle,
                        first_contact_substep=case["geometry"]["contact"]["first_foot_substep"]
                        - frame * 10,
                    )
                for substep in range(10):
                    data.ctrl[ai] = trace["torque_nm"][frame, 0, substep]
                    attempted += 1
                    mujoco.mj_step(model, data)
                    integrated += 1
                    equal(
                        data.actuator_force[ai],
                        trace["actual_actuator_force_nm"][frame, 0, substep],
                        "whole recorded actuator",
                    )
                    equal(
                        data.xpos[pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, substep],
                        "whole recorded pelvis",
                    )
                    if oracle is not None and case["frame"] <= frame <= case["frame"] + 1:
                        slot = (frame - case["frame"]) * 10 + substep
                        aq, av = canonical(data)
                        equal(aq, oracle["canonical_qpos_after_substep"][slot], "40ms oracle qpos")
                        equal(av, oracle["canonical_qvel_after_substep"][slot], "40ms oracle qvel")
                        equal(
                            data.actuator_force[ai],
                            oracle["actual_actuator_force_nm"][slot],
                            "40ms oracle actuator",
                        )
                        equal(
                            data.xpos[pelvis, 2],
                            oracle["pelvis_height_solver_epoch_m"][slot],
                            "40ms oracle pelvis",
                        )
                        equal(
                            trace["pd_requested_torque_nm"][frame, 0, substep],
                            oracle["requested_pd_torque_nm"][slot],
                            "40ms oracle requested PD",
                        )
                        force = np.zeros(3)
                        for w in read_geom_contact_wrenches(model, data, geom_id=bg):
                            if model.geom_bodyid[w.other_geom_id] != 0:
                                force += np.asarray(w.force_world_n)
                        equal(
                            force,
                            oracle["robot_force_on_ball_world_n"][slot],
                            "40ms oracle robot force",
                        )
                equal(
                    data.xpos[ball],
                    trace["ball_position_after_step_m"][frame, 0],
                    "whole recorded ball",
                )
            row = dict(
                index=index,
                job=case["job"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_outcome=case["receipt"]["outcome"],
                retrospective_frame=case["frame"],
                eligible=eligibility,
                ineligible_reason=None if eligibility else "NO_COMPLETE_TWO_FRAME_CONTACT_WINDOW",
                repeat_held_prediction_bit_equal=eligibility,
                first20ms_held_prediction_matches_original=eligibility,
                oracle40ms_matches_original_every_substep=eligibility,
                artifacts=artifacts,
                bias=bias,
                recorded_replay_substeps=3000,
                held_substeps=held_f.integrated_steps,
                oracle_substeps=oracle_f.integrated_steps,
                future_targets_not_input_to_held_predictor=True,
                trained_policy=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{index}-complete.json", row)
            rows.append(row)
            print(
                "CONTINUATION_CALIBRATION_COMPLETE",
                index,
                "BIAS_LIMITS",
                None if bias is None else bias["frozen_limits_satisfied"],
                flush=True,
            )
        bounded()
        if any(digest(path) != h for path, h in p.pins.items()):
            raise ValueError("source/original input bytes changed")
        report = dict(
            schema="soccer.rsi.contact_continuation_calibration_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            **counters(),
            all_six_source_cases_retained=len(rows) == 6,
            held_target_prediction_within_frozen_bias_limits_all_six=all(
                r["bias"] is not None and r["bias"]["frozen_limits_satisfied"] for r in rows
            ),
            new_closed_loop_policy_trials=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            candidate_actions_optimized=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        write_once(p.root / "complete.json", report)
    except BaseException as error:
        write_once(
            p.root / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                current_case=current_case,
                completed_cases=len(rows),
                **counters(),
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
