"""Independent NN/native replay including actual staged-boundary decisions."""

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
    p.add_argument("--held", action="store_true")
    p.add_argument("--audit-history-codec", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_probe_staged_lateral import CASES

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.source_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused audit and exact staged source required")
    source = json.loads(complete.read_text())
    if (
        source["status"]
        != (
            "CFTRACK_HELD_LATERAL_DEVELOPMENT_RECORDED"
            if a.held
            else "CFTRACK_STAGED_LATERAL_DEVELOPMENT_RECORDED"
        )
        or [r["case"] for r in source["records"]] != [name for name, _ in CASES]
        or (
            not a.held
            and [tuple(r["ball_velocity_m_s"]) for r in source["records"]]
            != [velocity for _, velocity in CASES]
        )
        or source["new_neural_calls"] > 600
        or source["new_native_steps"] > 6000
        or source["neural_optimizer_updates"] != 0
        or not source["cold_template_unchanged"]
    ):
        raise ValueError("complete ordered staged source and unchanged donor required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(
        str(path) not in pins for path in (cfg_path, external / "policy/omnicontact/OmniContact.py")
    ):
        raise ValueError("source-bound original policy and PD required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique canonical scene required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_nn_readbacks=600,
            maximum_native_replay_steps=6000,
            independently_initialized_native_world=True,
            producer_torque_policy_or_layer_not_used=True,
            staged_reference_boundary_and_C2_math_shared=True,
            saved_targets_only_for_verification=True,
            partial_stages_not_complete_h300=True,
            held_reference_controls=20 if a.held else 0,
            history_proprioception_audit=a.audit_history_codec,
            neural_optimizer_updates=0,
            new_independent_scenarios=0,
            hardware_authorized=False,
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
        raise ValueError("exact body and unchanged H300 contract required")
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
    before = cftrack_state_hash(prototype)
    cfg = yaml.safe_load(cfg_path.read_text())
    order = cfg["lab2mj"]
    kp = np.array(cfg["kp_lab"], np.float32)[order].astype(np.float64)
    kd = np.array(cfg["kd_lab"], np.float32)[order].astype(np.float64)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    records = []
    calls = steps = 0
    for row, (_, velocity) in zip(source["records"], CASES, strict=True):
        controls = row["run"]["controls"]
        if (
            type(controls) is not int
            or not 1 <= controls <= 300
            or row["run"]["native_steps"] != controls * 10
            or row["run"]["neural_calls"] != controls
            or row["run"]["stage_boundary_receipt"] is None
        ):
            raise ValueError("complete captured prefix and exact stage receipt required")
        path = a.source / row["artifact"]["path"]
        if (
            path.parent != a.source
            or path.is_symlink()
            or digest(path) != row["artifact"]["hash"]
            or pins.get(str(path)) != digest(path)
        ):
            raise ValueError("bound original staged native trace required")
        with np.load(path, allow_pickle=False) as saved:
            raw = {k: saved[k].copy() for k in saved.files}
        data = initial(velocity)
        if (
            hash_json(dict(qpos=data.qpos.tolist(), qvel=data.qvel.tolist()))
            != row["initial_state_hash"]
        ):
            raise ValueError("independent measured initial state differs")
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
        reference = condition_staged_lateral_intercept(player, hold_controls=20 if a.held else 0)
        if hash_json(reference) != hash_json(row["reference"]):
            raise ValueError("independently regenerated staged reference differs")
        plan_data = dict(row["plan"])
        plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
        plan = CFtrackTorqueIntentPlan(**plan_data)
        if plan.body_hash != hash_json(
            lo.profile()
        ) or plan.parent_policy_hash != cftrack_state_hash(policy):
            raise ValueError("independently rebuilt staged body/policy identity differs")
        guard_data = dict(row["stage_boundary"])
        for key in ("waypoint_xy", "reference_ball_xyz", "actual_goal_xyz"):
            guard_data[key] = tuple(guard_data[key])
        guard = CFtrackStageBoundary(**guard_data)
        history_audit = None
        if a.audit_history_codec:
            from rosclaw_soccer.rsi.cftrack_history_observation import (
                audit_current_ball_observation,
            )

            history_audit = audit_current_ball_observation
        calls += verify_cftrack_trace(
            policy, data, lo, raw, controls=controls, proprioception_audit=history_audit
        )
        replay = replay_torque_trial(
            model,
            initial(velocity),
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
            raise ValueError("actual stage boundary decision differs in independent replay")
        if controls < 300:
            if (
                controls != guard.control
                or replay["stage_boundary_receipt"]["ready"]
                or row["outcome"]["event_status"] != "MEASURED_STAGE_BOUNDARY_NOT_READY"
            ):
                raise ValueError("partial trajectory must end at rejected stage boundary")
            outcome = row["outcome"]
        else:
            if not replay["stage_boundary_receipt"]["ready"]:
                raise ValueError("full trajectory cannot continue after failed stage boundary")
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
                raise ValueError("independent H300 outcome differs")
        result = dict(
            case=row["case"],
            controls=controls,
            original_nn_input_output_and_target_exact=True,
            history_proprioception_audited=a.audit_history_codec,
            replay=replay,
            outcome=outcome,
            verified_previous_outcome=(
                row["previously_verified_outcome"] if a.held else row["verified_previous_outcome"]
            ),
            artifact_hash=row["artifact"]["hash"],
        )
        records.append(result)
        write_once(a.output / (row["case"] + ".json"), result)
    if cftrack_state_hash(prototype) != before:
        raise ValueError("cold NN prototype changed")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "CFTRACK_HELD_LATERAL_NN_NATIVE_AND_BOUNDARY_EXACT"
                if a.held
                else "CFTRACK_STAGED_LATERAL_NN_NATIVE_AND_BOUNDARY_EXACT"
            )
            if calls == source["new_neural_calls"]
            and steps == source["new_native_steps"]
            and len(records) == 2
            else "CFTRACK_STAGED_LATERAL_REVIEW_PARTIAL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            records=records,
            new_neural_calls=calls,
            new_native_steps=steps,
            history_proprioception_audited=a.audit_history_codec,
            new_independent_scenarios=0,
            reference_boundary_and_C2_math_shared=True,
            neural_optimizer_updates=0,
            fresh_not_evaluated=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
