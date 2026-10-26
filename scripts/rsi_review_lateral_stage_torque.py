"""Independent NN/native replay of bounded lateral-stage torque probes."""

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
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_probe_lateral_stage_torque import ARMS

    clean(a.expected_head)
    source_path = a.source / "complete.json"
    if (
        a.output.exists()
        or source_path.is_symlink()
        or digest(source_path) != a.source_hash
        or any(name in sys.modules for name in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused audit, exact producer, isolated author namespace required")
    source = json.loads(source_path.read_text())
    if (
        source["status"] != "LATERAL_STAGE_TORQUE_AUTHORITY_RECORDED_NOT_TASK"
        or [(r["arm"], r["axis"], r["effort_fraction"]) for r in source["records"]] != list(ARMS)
        or source["new_neural_calls"] > 60
        or source["new_native_steps"] > 600
        or source["neural_optimizer_updates"] != 0
    ):
        raise ValueError("complete bounded four-arm stage-response source required")
    pins = dict(source["input_file_hashes"])
    pins[str(source_path)] = a.source_hash
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(str(x) not in pins for x in (cfg_path, external / "policy/omnicontact/OmniContact.py")):
        raise ValueError("source-bound author policy and PD required")
    source_pins(pins)
    check_pins(pins)
    old_paths = [
        Path(x) for x in pins if x.endswith("/rsi-cftrack-staged-lateral-20261027/complete.json")
    ]
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(old_paths) != 1 or len(scenes) != 1:
        raise ValueError("unique old anchor and canonical scene required")
    old = json.loads(old_paths[0].read_text())
    prior = next(r for r in old["records"] if r["case"] == "new-left")
    if prior["outcome"]["event_status"] != "MEASURED_STAGE_BOUNDARY_NOT_READY":
        raise ValueError("original failed stage decision required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            maximum_neural_calls=60,
            maximum_native_steps=600,
            original_nn_on_rebuilt_measurements=True,
            manual_torque_replay=True,
            reference_boundary_and_c2_math_shared=True,
            no_new_scenarios_or_training=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    import yaml
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_stage_boundary import CFtrackStageBoundary
    from rosclaw_soccer.rsi.cftrack_staged_lateral_reference import (
        condition_staged_lateral_intercept,
    )
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
    from rosclaw_soccer.rsi.cftrack_torque_replay import replay_torque_trial
    from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace
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
    ):
        raise ValueError("exact canonical physical model required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from policy.omnicontact.OmniContact import OmniContact

    def initial():
        data = mujoco.MjData(model)
        data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
        data.qpos[lo.qi] = DEFAULT_JOINT_POS_MJ
        data.qpos[lo.bq : lo.bq + 7] = [1, 0, 0.11, 1, 0, 0, 0]
        data.qvel[lo.bv : lo.bv + 6] = [0, 0.12, 0, -0.12 / 0.11, 0, 0]
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
    data = initial()
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
    guard_data = dict(prior["stage_boundary"])
    for key in ("waypoint_xy", "reference_ball_xyz", "actual_goal_xyz"):
        guard_data[key] = tuple(guard_data[key])
    guard = CFtrackStageBoundary(**guard_data)
    records, calls, steps = [], 0, 0
    for row in source["records"]:
        controls = row["run"]["controls"]
        if (
            type(controls) is not int
            or not 1 <= controls <= 15
            or row["run"]["native_steps"] != controls * 10
            or row["run"]["neural_calls"] != controls
        ):
            raise ValueError("complete bounded native and NN prefix required")
        artifact = a.source / row["artifact"]["path"]
        if (
            artifact.parent != a.source
            or artifact.is_symlink()
            or digest(artifact) != row["artifact"]["hash"]
        ):
            raise ValueError("exact confined probe trace required")
        with np.load(artifact, allow_pickle=False) as saved:
            raw = {k: saved[k].copy() for k in saved.files}
        data = initial()
        if (
            hash_json(dict(qpos=data.qpos.tolist(), qvel=data.qvel.tolist()))
            != row["initial_state_hash"]
        ):
            raise ValueError("probe initial state differs")
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
        if hash_json(condition_staged_lateral_intercept(player)) != hash_json(row["reference"]):
            raise ValueError("probe reference differs")
        plan_data = dict(row["plan"])
        plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
        plan = CFtrackTorqueIntentPlan(**plan_data)
        if plan.parent_policy_hash != cftrack_state_hash(policy) or plan.body_hash != hash_json(
            lo.profile()
        ):
            raise ValueError("actual body and parent policy identity required")
        calls += verify_cftrack_trace(policy, data, lo, raw, controls=controls)
        replay = replay_torque_trial(
            model,
            initial(),
            lo,
            raw,
            plan,
            kp=kp,
            kd=kd,
            effort=effort,
            controls=controls,
            stage_boundary=guard,
        )
        steps += replay["native_steps"]
        if replay["stage_boundary_receipt"] != row["run"]["stage_boundary_receipt"]:
            raise ValueError("measured stage receipt differs")
        receipt = replay["stage_boundary_receipt"]
        baseline = prior["run"]["stage_boundary_receipt"]
        delta = dict(
            root_xy_m=(
                np.array(receipt["measured_root_pose"][:2])
                - np.array(baseline["measured_root_pose"][:2])
            ).tolist(),
            waypoint_error_m=receipt["waypoint_error_m"] - baseline["waypoint_error_m"],
            geometric_lateral_m=receipt["measured_geometry_to_planned_intercept"][
                "lateral_offset_m"
            ]
            - baseline["measured_geometry_to_planned_intercept"]["lateral_offset_m"],
        )
        if delta != row["difference_from_reviewed_parent"]:
            raise ValueError("independent measured parent-relative differences do not match")
        result = dict(
            arm=row["arm"],
            axis=row["axis"],
            effort_fraction=row["effort_fraction"],
            controls=controls,
            replay=replay,
            difference_from_reviewed_parent=delta,
            original_nn_input_output_exact=True,
            task_outcome_not_measured=True,
        )
        records.append(result)
        write_once(a.output / (row["arm"] + ".json"), result)
    if (
        cftrack_state_hash(prototype) != prototype_hash
        or calls != source["new_neural_calls"]
        or steps != source["new_native_steps"]
    ):
        raise ValueError("complete independent replay and unchanged cold prototype required")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="LATERAL_STAGE_TORQUE_NN_NATIVE_AND_BOUNDARY_EXACT",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=records,
            new_neural_calls=calls,
            new_native_steps=steps,
            no_new_scenarios_or_training=True,
            reference_boundary_and_c2_math_shared=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
