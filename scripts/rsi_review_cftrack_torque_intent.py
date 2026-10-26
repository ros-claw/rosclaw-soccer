"""Independent NN readback and manual native-force replay of torque qualification."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.qualification / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.qualification_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused isolated review and exact torque qualification required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "CFTRACK_TORQUE_INTENT_ZERO_AND_FEEDBACK_QUALIFIED"
        or not source["zero_native_prefix_exact"]
        or len(source["records"]) != 2
        or source["new_neural_calls"] != 320
        or source["new_native_steps"] != 3200
    ):
        raise ValueError("two complete cold native qualification arms required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.qualification_hash
    external = a.external_source.resolve()
    cfg_path = external / "policy/omnicontact/config/OmniContact.yaml"
    if any(
        str(path) not in pins for path in (cfg_path, external / "policy/omnicontact/OmniContact.py")
    ):
        raise ValueError("exact external PD and original NN source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(s) for s in pins if s.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_readback_nn_calls=320,
            maximum_native_replay_steps=3200,
            producer_context_factory_not_used=True,
            producer_torque_policy_and_layer_not_used=True,
            recorded_targets_for_verification_only=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    import mujoco
    import numpy as np
    import yaml
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_torque_intent import CFtrackTorqueIntentPlan
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
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact fixed compiled body required")
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
        mujoco.mj_forward(model, data)
        return data

    def reference(policy, data):
        view = SimpleNamespace(
            world=SimpleNamespace(data=data, control_steps=0), policy=policy, layout=lo
        )
        SharedOmniContactPlayer._observe(view)
        policy.goal_pos_override = np.array([5, 0, 0.11], np.float32)
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
    reference(prototype, initial())
    before = cftrack_state_hash(prototype)
    cfg = yaml.safe_load(cfg_path.read_text())
    order = cfg["lab2mj"]
    kp = np.array(cfg["kp_lab"], np.float32)[order].astype(np.float64)
    kd = np.array(cfg["kd_lab"], np.float32)[order].astype(np.float64)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    raws = []
    nn_records = []
    for arm, row in zip(("zero", "hg8-positive-small"), source["records"], strict=True):
        if row["arm"] != arm or row["controls"] != 160 or row["runtime_failure"] is not None:
            raise ValueError("ordered complete qualification arms required")
        path = a.qualification / row["artifact"]["path"]
        if (
            path.parent != a.qualification
            or path.is_symlink()
            or digest(path) != row["artifact"]["hash"]
            or pins.get(str(path)) != digest(path)
        ):
            raise ValueError("bound native/NN artifact required")
        with np.load(path, allow_pickle=False) as saved:
            raw = {key: saved[key].copy() for key in saved.files}
        data = initial()
        if not np.array_equal(data.qpos, raw["initial_qpos"]) or not np.array_equal(
            data.qvel, raw["initial_qvel"]
        ):
            raise ValueError("independently rebuilt initial state differs")
        policy = copy_cftrack_policy(prototype)
        reference(policy, data)
        plan_data = dict(row["plan"])
        plan_data["fraction_effort_hg"] = tuple(plan_data["fraction_effort_hg"])
        plan = CFtrackTorqueIntentPlan(**plan_data)
        if plan.body_hash != hash_json(
            lo.profile()
        ) or plan.parent_policy_hash != cftrack_state_hash(policy):
            raise ValueError("independent cold body/parent identity differs")
        raw["neural_obs"] = raw["nn_observations"][:, None, :]
        raw["neural_actions"] = raw["nn_actions"][:, None, :]
        count = verify_cftrack_trace(policy, data, lo, raw, controls=160)
        nn_records.append(dict(arm=arm, neural_calls=count, all_inputs_outputs_targets_exact=True))
        raws.append((raw, plan))
    if cftrack_state_hash(prototype) != before:
        raise ValueError("readback changed cold prototype")
    write_once(
        a.output / "nn-readback.json",
        dict(
            status="BOTH_TORQUE_ARMS_NN320_EXACT",
            records=nn_records,
            source_hash=a.qualification_hash,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    feet = [int(model.body("a/" + side + "_ankle_roll_link").id) for side in ("left", "right")]
    parts = {}
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor not in (0, lo.pelvis):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == lo.pelvis:
            parts[body] = feet.index(body) if body in feet else 2
    native_records = []
    for row, (raw, plan) in zip(source["records"], raws, strict=True):
        data = initial()
        offsets = plan.offsets()
        saturated = stopped = 0
        for step in range(1600):
            parent = np.clip(
                kp * (raw["joint_targets"][step // 10] - data.qpos[lo.qi]) - kd * data.qvel[lo.vi],
                -effort,
                effort,
            )
            relative = step / 10 - plan.start_control
            delta = np.zeros(29)
            if 0 <= relative < 48:
                index = int(relative)
                alpha = relative - index
                delta = ((1 - alpha) * offsets[index] + alpha * offsets[index + 1]) * effort
            q = data.qpos[lo.qi]
            limits = model.jnt_range[lo.joints]
            outward = ((q <= limits[:, 0]) & (delta < 0)) | ((q >= limits[:, 1]) & (delta > 0))
            stopped += int(outward.any())
            delta[outward] = 0
            proposed = parent.copy()
            changed = delta != 0
            proposed[changed] += delta[changed]
            actual = np.clip(proposed, -effort, effort)
            saturated += int(not np.array_equal(actual, proposed))
            actual[~changed] = parent[~changed]
            if not np.array_equal(parent, raw["parent_torque"][step, lo.ai]):
                raise ValueError(f"independent PD mismatch {row['arm']}:{step}")
            data.ctrl[lo.ai] = actual
            mujoco.mj_step(model, data)
            forces = np.zeros((1, 3))
            for ci in range(data.ncon):
                c = data.contact[ci]
                b1, b2 = (int(model.geom_bodyid[g]) for g in c.geom)
                other = b2 if b1 == lo.ball else b1 if b2 == lo.ball else -1
                if other in parts:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    forces[0, parts[other]] = max(
                        forces[0, parts[other]], float(np.linalg.norm(wrench[:3]))
                    )
            if (
                any(int(w.number) for w in data.warning)
                or not np.array_equal(data.qpos, raw["qpos"][step])
                or not np.array_equal(data.qvel, raw["qvel"][step])
                or not np.array_equal(data.actuator_force, raw["torque"][step])
                or not np.array_equal(data.actuator_force[lo.ai], actual)
                or not np.array_equal(forces, raw["ball_forces"][step])
            ):
                raise ValueError(
                    f"independent native torque/state/contact mismatch {row['arm']}:{step}"
                )
        if saturated != row["saturation_steps"] or stopped != row["measured_stop_guard_steps"]:
            raise ValueError("independent guard counters differ")
        native_records.append(dict(arm=row["arm"], native_steps=1600, all_native_values_exact=True))
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="TORQUE_INTENT_INDEPENDENT_NN_AND_MANUAL_NATIVE_EXACT",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=a.qualification_hash,
            neural_records=nn_records,
            native_records=native_records,
            new_neural_calls=320,
            new_native_steps=3200,
            new_independent_scenarios=0,
            neural_optimizer_updates=0,
            qualification_not_skill_improvement=True,
            producer_policy_or_layer_used_for_native_replay=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
