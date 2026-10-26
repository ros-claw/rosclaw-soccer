"""Independently audit guarded rolling-ball runs and pre-actuation refusals."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--baseline-review", type=Path, required=True)
    parser.add_argument("--baseline-review-hash", required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_evaluate_guarded_aim import CASES
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or a.baseline_review.is_symlink()
        or digest(complete) != a.source_hash
        or digest(a.baseline_review) != a.baseline_review_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused isolated audit and exact source/baseline required")
    source, baseline = json.loads(complete.read_text()), json.loads(a.baseline_review.read_text())
    expected = [
        (case, arm, velocity)
        for case, _, velocity, is_new in CASES
        for arm in (("baseline", "guarded") if is_new else ("guarded",))
    ]
    if (
        source["status"] != "CFTRACK_GUARDED_AIM_VELOCITY_DEVELOPMENT_AND_PROSPECTIVE_RECORDED"
        or [(r["case"], r["arm"], tuple(r["ball_velocity_m_s"])) for r in source["records"]]
        != expected
        or not source["all_pair_initial_states_match"]
        or not source["cold_template_unchanged"]
        or source["new_neural_calls"] > 3600
        or source["new_native_steps"] > 36000
        or source["neural_optimizer_updates"] != 0
        or baseline["status"] != "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
    ):
        raise ValueError("all declared source cases and independently reviewed baseline required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    if pins.get(str(a.baseline_review)) != a.baseline_review_hash:
        raise ValueError("same independently reviewed original baseline required")
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(
        str(path) not in pins for path in (cfg_path, external / "policy/omnicontact/OmniContact.py")
    ):
        raise ValueError("source-bound original NN and motor configuration required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical physical scene required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            max_nn_readbacks=3600,
            max_native_replay_steps=36000,
            independent_initial_state_and_manual_motor_composition=True,
            reference_and_selector_math_shared=True,
            actual_old_targets_for_verification_only=True,
            refused_before_actuation_reconstructed_without_nn_or_physics=True,
            new_independent_scenarios=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import yaml
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_guarded_reference_aim import condition_guarded_reference_aim
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_replay import replay_torque_trial
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial
    from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace
    from rosclaw_soccer.rsi.cftrack_velocity_intercept import condition_velocity_intercept
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.sim.contracts import hash_json

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
        or source["outcome_contract_hash"] != outcome_contract()["contract_hash"]
    ):
        raise ValueError("exact canonical body and original H300 scoring required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from policy.omnicontact.OmniContact import OmniContact

    def initial(velocity):
        data = mujoco.MjData(model)
        data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
        data.qpos[lo.qi] = DEFAULT_JOINT_POS_MJ
        data.qpos[lo.bq : lo.bq + 7] = [1, 0, 0.11, 1, 0, 0, 0]
        data.qvel[lo.bv : lo.bv + 3] = velocity
        data.qvel[lo.bv + 3 : lo.bv + 6] = [-velocity[1] / 0.11, velocity[0] / 0.11, 0.0]
        mujoco.mj_forward(model, data)
        return data

    prototype = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    prototype.task, prototype.reference_source, prototype.active_object_name = (
        "kickball",
        "CFgen",
        "ball",
    )
    prototype.box_dims = prototype.ball_dims = np.full(3, 0.11, np.float32)
    prototype.bbox_scale = prototype.box_dims * 2
    prototype.bbox_offsets_scaled = prototype.bbox_offsets * prototype.bbox_scale.reshape(1, 3)
    data = initial((0.0, 0.0, 0.0))
    view = SimpleNamespace(
        world=SimpleNamespace(data=data, control_steps=0), policy=prototype, layout=lo
    )
    SharedOmniContactPlayer._observe(view)
    prototype.goal_pos_override = np.array([5.0, 0.0, 0.11], np.float32)
    prototype.enter()
    condition_direct_kick_approach(prototype)
    prototype_hash = cftrack_state_hash(prototype)
    cfg = yaml.safe_load(cfg_path.read_text())
    order = cfg["lab2mj"]
    kp = np.array(cfg["kp_lab"], np.float32)[order].astype(np.float64)
    kd = np.array(cfg["kd_lab"], np.float32)[order].astype(np.float64)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    reviewed = []
    calls = steps = 0
    for row, (_, arm, velocity) in zip(source["records"], expected, strict=True):
        data = initial(velocity)
        initial_hash = hash_json(dict(qpos=data.qpos.tolist(), qvel=data.qvel.tolist()))
        if initial_hash != row["initial_state_hash"]:
            raise ValueError("independently reconstructed cold physical state differs")
        policy = copy_cftrack_policy(prototype)
        world = SharedNativeMotorWorld(
            model,
            data,
            players={"a/": NativePDParameters(kp, kd, effort)},
            ball_body_name="soccer_ball",
        )
        player = SharedOmniContactPlayer(
            world,
            "a/",
            policy,
            expected_policy_type=type(prototype),
            goal=np.array([5.0, 0.0, 0.11]),
        )
        failure = None
        try:
            reference = (
                condition_guarded_reference_aim(player, requested_offset_y_m=-0.4)
                if arm == "guarded"
                else dict(reference=condition_velocity_intercept(player), selection=None)
            )
        except ValueError as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
            reference = None
        if row["planning_error"] is not None:
            if (
                failure != row["planning_error"]
                or row["run"]["neural_calls"] != 0
                or row["run"]["native_steps"] != 0
                or world.native_steps != 0
                or policy.counter_step != 0
                or row["outcome"]["safe"] is not None
            ):
                raise ValueError("pre-actuation refusal does not independently reconstruct")
            result = dict(
                case=row["case"],
                arm=arm,
                reviewed=True,
                preactuation_refusal=True,
                physical_steps=0,
                neural_calls=0,
                outcome=row["outcome"],
            )
        else:
            if failure is not None or hash_json(reference) != hash_json(row["reference"]):
                raise ValueError("independent guarded selection/reference differs")
            if not complete_torque_trial(row["run"], controls=300):
                reviewed.append(
                    dict(case=row["case"], arm=arm, reviewed=False, preactuation_refusal=False)
                )
                continue
            path = a.source / row["artifact"]["path"]
            if (
                path.parent != a.source
                or path.is_symlink()
                or digest(path) != row["artifact"]["hash"]
                or pins.get(str(path)) != digest(path)
            ):
                raise ValueError("exact bound measured trace required")
            with np.load(path, allow_pickle=False) as saved:
                raw = {key: saved[key].copy() for key in saved.files}
            plan_data = dict(row["plan"])
            plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
            plan = CFtrackTorqueIntentPlan(**plan_data)
            if plan.body_hash != hash_json(
                lo.profile()
            ) or plan.parent_policy_hash != cftrack_state_hash(policy):
                raise ValueError("independent body or conditioned NN identity differs")
            calls += verify_cftrack_trace(policy, data, lo, raw, controls=300)
            replay = replay_torque_trial(
                model, initial(velocity), lo, raw, plan, kp=kp, kd=kd, effort=effort, controls=300
            )
            steps += replay["native_steps"]
            if (
                replay["saturation_steps"] != row["run"]["saturation_steps"]
                or replay["measured_stop_guard_steps"] != row["run"]["measured_stop_guard_steps"]
            ):
                raise ValueError("independent torque safety counts differ")
            force = np.zeros((300, 1, 6))
            force[:, 0, :3] = raw["ball_forces"][:, 0].reshape(300, 10, 3).max(axis=1)
            q = raw["qpos"].reshape(300, 10, 43)
            outcome = label_first_touch(
                dict(
                    force_n=force,
                    pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None],
                    ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None],
                )
            ).to_dict()
            if outcome != row["outcome"]:
                raise ValueError("independent original H300 outcome differs")
            result = dict(
                case=row["case"],
                arm=arm,
                reviewed=True,
                preactuation_refusal=False,
                neural_calls=300,
                replay=replay,
                outcome=outcome,
                artifact_hash=row["artifact"]["hash"],
            )
        reviewed.append(result)
        write_once(a.output / (row["case"] + "-" + arm + ".json"), result)
    if cftrack_state_hash(prototype) != prototype_hash:
        raise ValueError("independent review modified cold prototype")
    baseline_sources = [
        Path(path)
        for path, value in pins.items()
        if value == baseline["source_hash"] and path.endswith("/complete.json")
    ]
    if len(baseline_sources) != 1 or digest(baseline_sources[0]) != baseline["source_hash"]:
        raise ValueError("unique independently reviewed baseline source required")
    baseline_source = json.loads(baseline_sources[0].read_text())
    recomputed = []
    if all(r["reviewed"] for r in reviewed):
        for case, old_case, _, is_new in CASES:
            candidate = next(r for r in reviewed if r["case"] == case and r["arm"] == "guarded")
            raw_candidate = next(
                r for r in source["records"] if r["case"] == case and r["arm"] == "guarded"
            )
            if is_new:
                base = next(r for r in reviewed if r["case"] == case and r["arm"] == "baseline")
                raw_base = next(
                    r for r in source["records"] if r["case"] == case and r["arm"] == "baseline"
                )
                same_initial = raw_base["initial_state_hash"] == raw_candidate["initial_state_hash"]
                base_outcome = base["outcome"]
            else:
                old = next(
                    r
                    for r in baseline_source["records"]
                    if r["case"] == old_case and r["arm"] == "intercept-plus-fixed-nominal-torque"
                )
                verified = next(
                    r
                    for r in baseline["records"]
                    if r["case"] == old_case and r["arm"] == "intercept-plus-fixed-nominal-torque"
                )
                same_initial = (
                    old["initial_state_hash"] == raw_candidate["initial_state_hash"]
                    and verified["complete"]
                )
                base_outcome = verified["outcome"]
            recomputed.append(
                dict(
                    case=case,
                    new_velocity=is_new,
                    same_initial=same_initial,
                    baseline_outcome=base_outcome,
                    candidate_outcome=candidate["outcome"],
                    rescue=not base_outcome["task_pass"] and candidate["outcome"]["task_pass"],
                    regression=base_outcome["task_pass"] and not candidate["outcome"]["task_pass"],
                )
            )
        if recomputed != source["comparisons"]:
            raise ValueError("independent paired outcome/retention comparison differs")
    check_pins(pins)
    clean(a.expected_head)
    fully_reviewed = len(reviewed) == len(source["records"]) and all(
        r["reviewed"] for r in reviewed
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_GUARDED_AIM_NN_NATIVE_AND_REFUSALS_EXACT"
            if fully_reviewed
            and calls == source["new_neural_calls"]
            and steps == source["new_native_steps"]
            else "CFTRACK_GUARDED_AIM_REVIEW_PARTIAL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=reviewed,
            independently_recomputed_comparisons=recomputed,
            new_neural_calls=calls,
            new_native_steps=steps,
            new_independent_scenarios=0,
            reference_and_selector_math_shared=True,
            neural_optimizer_updates=0,
            full_fresh_exam_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
