"""Independently rebuild NN histories and replay local torque-response trials."""

import argparse
import json
import sys
from dataclasses import asdict
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
        raise ValueError("unused isolated review and exact completed source required")
    source = json.loads(complete.read_text())
    cases = (
        ("nominal", (1.0, 0.0)),
        ("x-minus", (0.98, 0.0)),
        ("x-plus", (1.02, 0.0)),
        ("y-minus", (1.0, -0.02)),
        ("y-plus", (1.0, 0.02)),
    )
    if (
        source["status"] != "CFTRACK_TORQUE_RESPONSE_DEVELOPMENT_RECORDED"
        or [(r["case"], tuple(r["ball_xy"])) for r in source["records"]] != list(cases)
        or source["new_neural_calls"] > 2400
        or source["new_native_steps"] > 24000
        or source["neural_optimizer_updates"] != 0
        or not source["parent_prefixes_exact"]
        or not source["cold_template_unchanged"]
    ):
        raise ValueError(
            "complete declared development case identities and unchanged donor required"
        )
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(
        str(path) not in pins for path in (cfg_path, external / "policy/omnicontact/OmniContact.py")
    ):
        raise ValueError("bound original NN source and PD configuration required")
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
            max_neural_readback_calls=2400,
            max_native_replay_steps=24000,
            producer_factory_not_used=True,
            producer_torque_policy_or_layer_not_used_for_replay=True,
            declared_envelope_math_shared=True,
            saved_targets_used_for_verification_only=True,
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
    from rosclaw_soccer.rsi.cftrack_motor_authority import summarize_motor_future
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_replay import replay_torque_trial
    from rosclaw_soccer.rsi.cftrack_torque_response import propose_torque_response
    from rosclaw_soccer.rsi.cftrack_torque_trial import complete_torque_trial
    from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.sim.contracts import hash_json

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != (
        "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact canonical compiled body required")
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
    config = yaml.safe_load(cfg_path.read_text())
    order = config["lab2mj"]
    kp = np.array(config["kp_lab"], np.float32)[order].astype(np.float64)
    kd = np.array(config["kd_lab"], np.float32)[order].astype(np.float64)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    records = []
    calls = steps = 0
    for source_case, (_, xy) in zip(source["records"], cases, strict=True):
        for arm in ("parent", "support", "candidate"):
            row = source_case[arm]
            if row is None or not complete_torque_trial(row["run"]):
                records.append(dict(case=source_case["case"], arm=arm, complete=False))
                continue
            path = a.source / row["artifact"]["path"]
            if (
                path.parent != a.source
                or path.is_symlink()
                or digest(path) != row["artifact"]["hash"]
                or pins.get(str(path)) != digest(path)
            ):
                raise ValueError("unaltered native and NN arm evidence required")
            with np.load(path, allow_pickle=False) as saved:
                raw = {key: saved[key].copy() for key in saved.files}
            data = initial(xy)
            policy = copy_cftrack_policy(prototype)
            reference(policy, data)
            plan_data = dict(row["plan"])
            plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
            plan = CFtrackTorqueIntentPlan(**plan_data)
            if (
                plan.body_hash != hash_json(lo.profile())
                or plan.parent_policy_hash != cftrack_state_hash(policy)
                or not np.array_equal(data.qpos, raw["initial_qpos"])
                or not np.array_equal(data.qvel, raw["initial_qvel"])
            ):
                raise ValueError("independently rebuilt cold state/policy/body differs")
            if arm == "candidate":
                proposed, model_receipt = propose_torque_response(
                    source_case["parent"]["metrics"],
                    source_case["support"]["metrics"],
                    body_hash=plan.body_hash,
                    parent_policy_hash=plan.parent_policy_hash,
                )
                if hash_json(asdict(proposed)) != hash_json(row["plan"]) or hash_json(
                    model_receipt
                ) != hash_json(source_case["model"]):
                    raise ValueError("candidate model does not rebuild from own parent and probe")
            count = verify_cftrack_trace(policy, data, lo, raw, controls=160)
            calls += count
            replay = replay_torque_trial(
                model, initial(xy), lo, raw, plan, kp=kp, kd=kd, effort=effort
            )
            steps += replay["native_steps"]
            if (
                replay["saturation_steps"] != row["run"]["saturation_steps"]
                or replay["measured_stop_guard_steps"] != row["run"]["measured_stop_guard_steps"]
            ):
                raise ValueError("independently recomposed actual guard counts differ")
            q = raw["qpos"][:, lo.rq : lo.rq + 7]
            body_failed = bool(
                ((q[:, 2] < 0.65) | (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) < 0.9)).any()
            )
            metrics = summarize_motor_future(
                dict(raw, controls=160, body_failed=body_failed, runtime_failure=None),
                ball_address=lo.bq,
                root_address=lo.rq,
            )
            if metrics != row["metrics"]:
                raise ValueError("independently recalculated local physical metrics differ")
            reviewed = dict(
                case=source_case["case"],
                arm=arm,
                complete=True,
                original_nn_input_output_and_target_exact=True,
                replay=replay,
                metrics=metrics,
                artifact_hash=row["artifact"]["hash"],
            )
            write_once(a.output / (source_case["case"] + "-" + arm + ".json"), reviewed)
            records.append(reviewed)
    if cftrack_state_hash(prototype) != prototype_hash:
        raise ValueError("original cold prototype was modified by independent review")
    check_pins(pins)
    clean(a.expected_head)
    complete_count = sum(r["complete"] for r in records)
    write_once(
        a.output / "complete.json",
        dict(
            status="TORQUE_RESPONSE_ALL15_NN_AND_MANUAL_NATIVE_EXACT"
            if complete_count == 15 and calls == 2400 and steps == 24000
            else "TORQUE_RESPONSE_PARTIAL_REVIEW_NOT_QUALIFIED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=records,
            complete_arms=complete_count,
            new_neural_calls=calls,
            new_native_steps=steps,
            new_independent_scenarios=0,
            neural_optimizer_updates=0,
            full_h300_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
