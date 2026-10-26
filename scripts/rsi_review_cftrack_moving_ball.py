"""Independent NN and native replay of paired rolling-ball H300 probes."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--velocity-intercept", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.source_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused isolated review and exact moving-ball source required")
    source = json.loads(complete.read_text())
    cases = (
        ("forward", (0.1, 0.0, 0.0)),
        ("backward", (-0.1, 0.0, 0.0)),
        ("left", (0.0, 0.1, 0.0)),
        ("right", (0.0, -0.1, 0.0)),
    )
    arms = (
        ("intercept-reference-only", "intercept-plus-fixed-nominal-torque")
        if a.velocity_intercept
        else ("parent", "frozen-nominal-torque")
    )
    expected_status = (
        "CFTRACK_VELOCITY_INTERCEPT_DEVELOPMENT_RECORDED"
        if a.velocity_intercept
        else "CFTRACK_FROZEN_TORQUE_MOVING_BALL_EXPLORATORY_RECORDED"
    )
    expected = [(name, arm, vel) for name, vel in cases for arm in arms]
    if (
        source["status"] != expected_status
        or [(r["case"], r["arm"], tuple(r["ball_velocity_m_s"])) for r in source["records"]]
        != expected
        or source["new_neural_calls"] > 2400
        or source["new_native_steps"] > 24000
        or source["neural_optimizer_updates"] != 0
        or not source["paired_initial_states_equal"]
        or not source["cold_template_unchanged"]
    ):
        raise ValueError("complete ordered paired moving-ball source and unchanged donor required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(str(x) not in pins for x in (cfg_path, external / "policy/omnicontact/OmniContact.py")):
        raise ValueError("source-bound original NN and PD configuration required")
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
            max_neural_calls=2400,
            max_native_steps=24000,
            producer_factory_not_used=True,
            producer_torque_policy_or_layer_not_used_for_replay=True,
            envelope_math_shared=True,
            old_targets_for_verification_only=True,
            new_independent_scenarios=0,
            velocity_intercept_reference_math_shared=a.velocity_intercept,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import yaml
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
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
        data.qpos[lo.bq : lo.bq + 7] = [1.0, 0.0, 0.11, 1, 0, 0, 0]
        data.qvel[lo.bv : lo.bv + 3] = velocity
        data.qvel[lo.bv + 3 : lo.bv + 6] = [-velocity[1] / 0.11, velocity[0] / 0.11, 0.0]
        mujoco.mj_forward(model, data)
        return data

    def reference(policy, data):
        view = SimpleNamespace(
            world=SimpleNamespace(data=data, control_steps=0), policy=policy, layout=lo
        )
        SharedOmniContactPlayer._observe(view)
        policy.goal_pos_override = np.array([5.0, 0.0, 0.11], np.float32)
        policy.enter()
        condition_direct_kick_approach(policy)

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
    reference(prototype, initial((0.0, 0.0, 0.0)))
    prototype_hash = cftrack_state_hash(prototype)
    cfg = yaml.safe_load(cfg_path.read_text())
    order = cfg["lab2mj"]
    kp = np.array(cfg["kp_lab"], np.float32)[order].astype(np.float64)
    kd = np.array(cfg["kd_lab"], np.float32)[order].astype(np.float64)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    records = []
    calls = steps = 0
    for source_row, (_, _, velocity) in zip(source["records"], expected, strict=True):
        if not complete_torque_trial(source_row["run"], controls=300):
            records.append(dict(case=source_row["case"], arm=source_row["arm"], complete=False))
            continue
        path = a.source / source_row["artifact"]["path"]
        if (
            path.parent != a.source
            or path.is_symlink()
            or digest(path) != source_row["artifact"]["hash"]
            or pins.get(str(path)) != digest(path)
        ):
            raise ValueError("unchanged original NN and native trajectory required")
        with np.load(path, allow_pickle=False) as saved:
            raw = {key: saved[key].copy() for key in saved.files}
        data = initial(velocity)
        policy = copy_cftrack_policy(prototype)
        if a.velocity_intercept:
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
            receipt = condition_velocity_intercept(player)
            if hash_json(receipt) != hash_json(source_row["reference"]):
                raise ValueError("independently regenerated privileged reference differs")
        else:
            reference(policy, data)
        plan_data = dict(source_row["plan"])
        plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
        plan = CFtrackTorqueIntentPlan(**plan_data)
        if (
            plan.body_hash != hash_json(lo.profile())
            or plan.parent_policy_hash != cftrack_state_hash(policy)
            or not np.array_equal(data.qpos, raw["initial_qpos"])
            or not np.array_equal(data.qvel, raw["initial_qvel"])
        ):
            raise ValueError("independent cold moving state, policy or body differs")
        calls += verify_cftrack_trace(policy, data, lo, raw, controls=300)
        replay = replay_torque_trial(
            model, initial(velocity), lo, raw, plan, kp=kp, kd=kd, effort=effort, controls=300
        )
        steps += replay["native_steps"]
        if (
            replay["saturation_steps"] != source_row["run"]["saturation_steps"]
            or replay["measured_stop_guard_steps"] != source_row["run"]["measured_stop_guard_steps"]
        ):
            raise ValueError("independent measured torque safety counts differ")
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
        if outcome != source_row["outcome"]:
            raise ValueError("independent unchanged H300 outcome differs")
        row = dict(
            case=source_row["case"],
            arm=source_row["arm"],
            complete=True,
            original_nn_input_output_and_target_exact=True,
            replay=replay,
            outcome=outcome,
            artifact_hash=source_row["artifact"]["hash"],
        )
        records.append(row)
        write_once(a.output / (row["case"] + "-" + row["arm"] + ".json"), row)
    if cftrack_state_hash(prototype) != prototype_hash:
        raise ValueError("independent review modified cold prototype")
    check_pins(pins)
    clean(a.expected_head)
    reviewed = [r for r in records if r["complete"]]
    pairs = [reviewed[i : i + 2] for i in range(0, len(reviewed), 2)] if len(reviewed) == 8 else []
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "CFTRACK_INTERCEPT_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
                if a.velocity_intercept
                else "CFTRACK_MOVING_BALL_ALL8_NN_AND_MANUAL_NATIVE_EXACT"
            )
            if len(reviewed) == 8 and calls == 2400 and steps == 24000
            else "CFTRACK_PAIRED_MOVING_BALL_PARTIAL_REVIEW_NOT_QUALIFIED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=records,
            parent_passes=sum(p[0]["outcome"]["task_pass"] for p in pairs),
            candidate_passes=sum(p[1]["outcome"]["task_pass"] for p in pairs),
            first_arm=arms[0],
            second_arm=arms[1],
            velocity_intercept_reference_math_shared=a.velocity_intercept,
            new_neural_calls=calls,
            new_native_steps=steps,
            new_independent_scenarios=0,
            neural_optimizer_updates=0,
            fresh_exam_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
