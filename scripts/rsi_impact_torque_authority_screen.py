"""Bounded retrospective torque authority diagnostic; never an online controller."""

import argparse
import dataclasses
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.impact_torque_screen import (
    impact_prediction_score,
    impact_torque_candidates,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_impact_screen_inputs import prepare
from scripts.rsi_stage_release_contact_replay import digest


def equal(a, b, name):
    if not np.allclose(a, b, rtol=0, atol=1e-8):
        raise ValueError("original physics/zero forecast differs: " + name)


def repeat_equal(first, second):
    for field in dataclasses.fields(first):
        a, b = getattr(first, field.name), getattr(second, field.name)
        same = (
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            if (isinstance(a, np.ndarray))
            else a == b
        )
        if not same:
            raise ValueError("candidate order contaminated zero forecast: " + field.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    p = prepare(args.config)
    if Path(__file__).resolve() != p.source / "scripts/rsi_impact_torque_authority_screen.py":
        raise ValueError("exact frozen runner required")
    if mujoco.__version__ != "3.12.0" or mujoco.get_mjcb_control() is not None:
        raise ValueError("original callback-free engine required")
    if args.preflight:
        print("WHOLE_SIX_AUTHENTICATED_ZERO_PHYSICS_ALLOCATIONS", flush=True)
        return
    p.root.mkdir()
    declaration = dict(
        schema="soccer.rsi.retrospective_impact_torque_screen_declaration.v1",
        config=p.cfg,
        input_file_hashes=p.pins,
        geometry_report_hash=p.geometry["report_hash"],
        stage_hash=p.stage["report_hash"],
        all_six_source_cases_retained=True,
        contact_time_is_retrospective_not_actor_feature=True,
        new_closed_loop_policy_trials=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        teacher_qualified=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(p.root / "declaration.json", declaration)
    attempted = integrated = 0
    forecasters, rows = [], []
    current_case = None

    def totals():
        fa = sum(f.attempted_steps for f in forecasters)
        fi = sum(f.integrated_steps for f in forecasters)
        return dict(
            recorded_attempted_steps=attempted,
            recorded_integrated_steps=integrated,
            forecast_attempted_steps=fa,
            forecast_integrated_steps=fi,
            combined_attempted_steps=attempted + fa,
            combined_integrated_steps=integrated + fi,
        )

    def bounded():
        t = totals()
        if (
            time.time() >= p.cfg["deadline_unix"]
            or t["recorded_attempted_steps"] > 18000
            or t["forecast_attempted_steps"] > 840
            or t["combined_attempted_steps"] > 18840
        ):
            raise RuntimeError("fixed diagnostic budget/deadline exhausted; no retry")

    try:
        model = mujoco.MjModel.from_binary_path(p.cases[0]["receipt"]["witnesses"]["world"]["path"])
        sampler = G1JointTaskKinematicSampler(model, expected_world_hash=p.world_hash)
        qi = [int(model.joint(n).qposadr[0]) for n in G1_DDS_JOINT_NAMES]
        vi = [int(model.joint(n).dofadr[0]) for n in G1_DDS_JOINT_NAMES]
        ai = []
        for name in G1_DDS_JOINT_NAMES:
            ids = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == model.joint(name).id)
            )
            if len(ids) != 1:
                raise ValueError("one original motor per joint required")
            ai.append(int(ids[0]))
        ball = int(model.body("ball").id)
        bj = int(model.body_jntadr[ball])
        bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
        pelvis = int(model.body("pelvis").id)
        if (
            model.geom("ball_geom").size[0] != 0.11
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
            forecaster = G1PDTorqueForecaster(
                model,
                expected_world_hash=p.world_hash,
                kp=np.asarray(physics["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(physics["canonical_pd_kd"], dtype=np.float64),
                max_forecast_steps=140,
            )
            forecasters.append(forecaster)
            predictions = []
            baseline = None
            for frame in range(300):
                bounded()
                aq, av = canonical(data)
                equal(aq, q[frame, 0], "whole recorded frame qpos")
                equal(av, v[frame, 0], "whole recorded frame qvel")
                if frame == case["frame"] and case["side"] is not None:
                    target = trace["joint_target_rad"][frame, 0].copy()
                    candidates = impact_torque_candidates(sampler.sample(data), case["side"])
                    if not 1 <= len(candidates) <= 13 or np.any(candidates[0][1]):
                        raise ValueError("bounded baseline-first candidate bank required")
                    for number, (name, offset) in enumerate(candidates):
                        bounded()
                        forecast = forecaster.forecast(data, target, offset)
                        if number == 0:
                            baseline = forecast
                        array_path = p.root / f"case-{index}-candidate-{number}.npz"
                        arrays = {
                            f.name: getattr(forecast, f.name)
                            for f in dataclasses.fields(forecast)
                            if isinstance(getattr(forecast, f.name), np.ndarray)
                        }
                        with array_path.open("xb") as stream:
                            np.savez_compressed(stream, offset_nm=offset, **arrays)
                        prediction = dict(
                            name=name,
                            candidate_index=number,
                            offset_nm=offset.tolist(),
                            donor_integration_state_hash=forecast.donor_integration_state_hash,
                            nonfoot_robot_contact_over_1n=forecast.nonfoot_robot_contact_over_1n,
                            score=impact_prediction_score(forecast, offset),
                            arrays=dict(path=str(array_path), sha256=digest(array_path)),
                        )
                        write_once(p.root / f"case-{index}-candidate-{number}.json", prediction)
                        predictions.append(prediction)
                    bounded()
                    repeated = forecaster.forecast(data, target, np.zeros(29))
                    repeat_equal(baseline, repeated)
                    equal(
                        baseline.requested_pd_torque_nm,
                        trace["pd_requested_torque_nm"][frame, 0],
                        "zero PD",
                    )
                    equal(baseline.applied_control_nm, trace["torque_nm"][frame, 0], "zero torque")
                for substep in range(10):
                    data.ctrl[ai] = trace["torque_nm"][frame, 0, substep]
                    attempted += 1
                    mujoco.mj_step(model, data)
                    integrated += 1
                    equal(
                        data.actuator_force[ai],
                        trace["actual_actuator_force_nm"][frame, 0, substep],
                        "recorded actuator force",
                    )
                    equal(
                        data.xpos[pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, substep],
                        "recorded pelvis",
                    )
                    if frame == case["frame"] and baseline is not None:
                        aq, av = canonical(data)
                        equal(aq, baseline.canonical_qpos_after_substep[substep], "zero qpos")
                        equal(av, baseline.canonical_qvel_after_substep[substep], "zero qvel")
                        equal(
                            data.actuator_force[ai],
                            baseline.actual_actuator_force_nm[substep],
                            "zero actuator",
                        )
                        equal(
                            data.xpos[pelvis, 2],
                            baseline.pelvis_height_solver_epoch_m[substep],
                            "zero pelvis",
                        )
                equal(
                    data.xpos[ball], trace["ball_position_after_step_m"][frame, 0], "recorded ball"
                )
            feasible = [r for r in predictions if r["score"]["feasible_predicted_impact"]]
            best = min(feasible, key=lambda r: r["score"]["proxy_cost"]) if feasible else None
            row = dict(
                index=index,
                job=case["job"],
                source_receipt_hash=case["receipt"]["receipt_hash"],
                original_outcome=case["receipt"]["outcome"],
                retrospective_frame=case["frame"],
                causal_swing_side=case["side"],
                ineligible_reason=case["ineligible_reason"],
                predictions=predictions,
                best_proxy_candidate_index=None if best is None else best["candidate_index"],
                repeat_zero_bit_equal=baseline is not None,
                zero_forecast_matches_original_ten_substeps=baseline is not None,
                recorded_replay_substeps=3000,
                forecast_substeps=forecaster.integrated_steps,
                live_action_applied=False,
                new_closed_loop_policy_trial=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            row["report_hash"] = hash_json(row)
            write_once(p.root / f"case-{index}-complete.json", row)
            rows.append(row)
            print(
                "IMPACT_CASE_COMPLETE",
                index,
                "FORECAST_STEPS",
                forecaster.integrated_steps,
                flush=True,
            )
        bounded()
        if any(digest(path) != h for path, h in p.pins.items()):
            raise ValueError("original frozen source/input changed")
        report = dict(
            schema="soccer.rsi.retrospective_impact_torque_screen_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            **totals(),
            all_six_source_cases_retained=len(rows) == 6,
            new_closed_loop_policy_trials=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            final_m0_improvement_proven=False,
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
                **totals(),
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
