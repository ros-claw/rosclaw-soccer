"""Independently read NN and replay all five frozen torque H300 development arms."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
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
        raise ValueError("unused isolated review and exact H300 source required")
    source = json.loads(complete.read_text())
    cases = (
        ("nominal", (1.0, 0.0)),
        ("x-minus", (0.98, 0.0)),
        ("x-plus", (1.02, 0.0)),
        ("y-minus", (1.0, -0.02)),
        ("y-plus", (1.0, 0.02)),
    )
    if (
        source["status"] != "CFTRACK_TORQUE_FROZEN_H300_DEVELOPMENT_RECORDED"
        or [(r["case"], tuple(r["ball_xy"])) for r in source["records"]] != list(cases)
        or source["new_neural_calls"] > 1500
        or source["new_native_steps"] > 15000
        or source["neural_optimizer_updates"] != 0
        or not source["earlier_candidate_prefixes_exact"]
        or not source["cold_template_unchanged"]
    ):
        raise ValueError("bound five-case development H300 and original prefix required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(str(p) not in pins for p in (cfg_path, external / "policy/omnicontact/OmniContact.py")):
        raise ValueError("source-bound original NN and PD gains required")
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
            maximum_neural_readback_calls=1500,
            maximum_native_replay_steps=15000,
            producer_factory_not_used=True,
            producer_torque_policy_and_layer_not_used_for_replay=True,
            declared_envelope_math_shared=True,
            recorded_targets_are_verification_only=True,
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
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_replay import replay_torque_trial
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial
    from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.sim.contracts import hash_json

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != ("sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d")
        or source["outcome_contract_hash"] != outcome_contract()["contract_hash"]
    ):
        raise ValueError("exact canonical body and original H300 scoring required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from policy.omnicontact.OmniContact import OmniContact

    def initial(xy):
        data = mujoco.MjData(model)
        data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
        data.qpos[lo.qi] = DEFAULT_JOINT_POS_MJ
        data.qpos[lo.bq : lo.bq + 7] = [*xy, 0.11, 1, 0, 0, 0]
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
    reference(prototype, initial((1.0, 0.0)))
    prototype_hash = cftrack_state_hash(prototype)
    cfg = yaml.safe_load(cfg_path.read_text())
    order = cfg["lab2mj"]
    kp = np.array(cfg["kp_lab"], np.float32)[order].astype(np.float64)
    kd = np.array(cfg["kd_lab"], np.float32)[order].astype(np.float64)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    records = []
    calls = steps = 0
    for source_row, (_, xy) in zip(source["records"], cases, strict=True):
        if not complete_torque_trial(source_row["run"], controls=300):
            records.append(dict(case=source_row["case"], complete=False))
            continue
        path = a.source / source_row["artifact"]["path"]
        if (
            path.parent != a.source
            or path.is_symlink()
            or digest(path) != source_row["artifact"]["hash"]
            or pins.get(str(path)) != digest(path)
        ):
            raise ValueError("unchanged measured NN and native trace required")
        with np.load(path, allow_pickle=False) as saved:
            raw = {key: saved[key].copy() for key in saved.files}
        data = initial(xy)
        policy = copy_cftrack_policy(prototype)
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
            raise ValueError("independently regenerated cold state, parent or body differs")
        calls += verify_cftrack_trace(policy, data, lo, raw, controls=300)
        replay = replay_torque_trial(
            model, initial(xy), lo, raw, plan, kp=kp, kd=kd, effort=effort, controls=300
        )
        steps += replay["native_steps"]
        if (
            replay["saturation_steps"] != source_row["run"]["saturation_steps"]
            or replay["measured_stop_guard_steps"] != source_row["run"]["measured_stop_guard_steps"]
        ):
            raise ValueError("independently recomposed torque safety counts differ")
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
            complete=True,
            original_nn_input_output_and_target_exact=True,
            replay=replay,
            outcome=outcome,
            verified_parent_outcome=source_row["verified_parent_outcome"],
            artifact_hash=source_row["artifact"]["hash"],
        )
        records.append(row)
        write_once(a.output / (source_row["case"] + ".json"), row)
    if cftrack_state_hash(prototype) != prototype_hash:
        raise ValueError("independent review modified cold prototype")
    check_pins(pins)
    clean(a.expected_head)
    reviewed = [r for r in records if r["complete"]]
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_TORQUE_H300_ALL5_NN_AND_MANUAL_NATIVE_EXACT"
            if len(reviewed) == 5 and calls == 1500 and steps == 15000
            else "CFTRACK_TORQUE_H300_PARTIAL_REVIEW_NOT_QUALIFIED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=records,
            candidate_passes=sum(r["outcome"]["task_pass"] for r in reviewed),
            parent_passes=sum(r["verified_parent_outcome"]["task_pass"] for r in reviewed),
            new_neural_calls=calls,
            new_native_steps=steps,
            new_independent_scenarios=0,
            neural_optimizer_updates=0,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
