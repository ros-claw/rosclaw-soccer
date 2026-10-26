"""Independent H300 bank NN readback; rebuild initialization without producer factory."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-hash", required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.bank / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or digest(complete) != a.bank_hash
        or any(name in sys.modules for name in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused isolated readback and bound complete bank required")
    bank = json.loads(complete.read_text())
    cases = (
        ("nominal", (1.0, 0.0)),
        ("x-minus", (0.98, 0.0)),
        ("x-plus", (1.02, 0.0)),
        ("y-minus", (1.0, -0.02)),
        ("y-plus", (1.0, 0.02)),
    )
    expected = [(name, arm, xy) for name, xy in cases for arm in ["parent", "frozen-motor"]]
    if (
        bank["status"] != "CFTRACK_STATIC_PERTURBATION_BANK_RECORDED_UNVERIFIED"
        or len(bank["records"]) != 10
        or [(r["case"], r["arm"], tuple(r["ball_xy"])) for r in bank["records"]] != expected
        or bank["neural_optimizer_updates"] != 0
    ):
        raise ValueError("complete declared paired five-case H300 bank required")
    pins = dict(bank["input_file_hashes"])
    pins[str(complete)] = a.bank_hash
    external = a.external_source.resolve()
    if str(external / "policy/omnicontact/OmniContact.py") not in pins:
        raise ValueError("qualified external source required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_neural_calls=3000,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            initialization_independently_rebuilt_not_producer_factory=True,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
    from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
    from rosclaw_soccer.rsi.cftrack_trace_readback import verify_cftrack_trace
    from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
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
        raise ValueError("exact compiled body required")
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
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return data

    def reference(policy, data):
        view = SimpleNamespace(
            world=SimpleNamespace(data=data, control_steps=0), policy=policy, layout=lo
        )
        SharedOmniContactPlayer._observe(view)
        policy.goal_pos_override = np.array([5.0, 0.0, 0.11], np.float32)
        policy.enter()
        return condition_direct_kick_approach(policy)

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
    before = cftrack_state_hash(prototype)
    records = []
    frozen_parameters = None
    calls = 0
    failure = None
    try:
        for row in bank["records"]:
            if (
                row["controls"] != 300
                or row["native_steps"] != 3000
                or row["neural_calls"] != 300
                or row["runtime_failure"] is not None
                or row["body_failed"]
            ):
                raise ValueError("partial native trials require separate partial readback protocol")
            path = a.bank / row["artifact"]["path"]
            if (
                path.parent != a.bank
                or path.is_symlink()
                or pins.get(str(path)) != digest(path)
                or row["artifact"]["hash"] != digest(path)
            ):
                raise ValueError("bound ordinary trial raw required")
            with np.load(path, allow_pickle=False) as saved:
                raw = {k: saved[k].copy() for k in saved.files}
            for key, shape in [
                ("torque", (3000, 29)),
                ("ball_forces", (3000, 1, 3)),
                ("initial_qpos", (43,)),
                ("initial_qvel", (41,)),
            ]:
                if raw[key].shape != shape or not np.isfinite(raw[key]).all():
                    raise ValueError("finite complete native measurements required")
            if (raw["ball_forces"] < 0).any() or (
                abs(raw["torque"][:, lo.ai]) > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
            ).any():
                raise ValueError("nonnegative native forces and original torque caps required")
            data = initial(tuple(row["ball_xy"]))
            if not np.array_equal(data.qpos, raw["initial_qpos"]) or not np.array_equal(
                data.qvel, raw["initial_qvel"]
            ):
                raise ValueError("initial body differs from independent declared recipe")
            policy = copy_cftrack_policy(prototype)
            if (
                hash_json(reference(policy, data)) != hash_json(row["reference_conditioning"])
                or cftrack_state_hash(policy) != row["cold_policy_hash"]
            ):
                raise ValueError("independent initial reference/policy state does not rebuild")
            plan = None
            if (row["arm"] == "parent") != (row["motor_plan"] is None):
                raise ValueError("parent and frozen-motor arm identities disagree")
            if row["motor_plan"] is not None:
                value = row["motor_plan"]
                parameters = {k: v for k, v in value.items() if k != "donor_policy_hash"}
                if frozen_parameters is None:
                    frozen_parameters = parameters
                elif parameters != frozen_parameters:
                    raise ValueError("controller parameters changed across perturbations")
                plan = CFtrackMotorResidualPlan(
                    value["origin_counter"],
                    value["start_control"],
                    value["peak_control"],
                    tuple(value["delta_hg"]),
                    value["donor_policy_hash"],
                )
                plan.require_donor(
                    policy_hash=cftrack_state_hash(policy), counter=policy.counter_step
                )
            calls += verify_cftrack_trace(policy, data, lo, raw, controls=300, motor_plan=plan)
            grouped = np.zeros((300, 1, 6))
            grouped[:, 0, :3] = raw["ball_forces"][:, 0].reshape(300, 10, 3).max(axis=1)
            q = raw["qpos"].reshape(300, 10, 43)
            outcome = label_first_touch(
                dict(
                    force_n=grouped,
                    pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None],
                    ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None],
                )
            ).to_dict()
            if hash_json(outcome) != hash_json(row["outcome"]):
                raise ValueError("raw H300 label differs")
            if cftrack_state_hash(prototype) != before:
                raise ValueError("scratch readback mutated cold prototype")
            verified = dict(
                case=row["case"],
                arm=row["arm"],
                artifact_hash=row["artifact"]["hash"],
                motor_plan=row["motor_plan"],
                neural_readbacks=300,
                outcome=outcome,
            )
            write_once(a.output / (row["case"] + "-" + row["arm"] + ".json"), verified)
            records.append(verified)
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_STATIC_BANK_NN_AND_LABELS_VERIFIED"
            if len(records) == 10 and failure is None
            else "CFTRACK_STATIC_BANK_READBACK_NOT_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_bank_hash=a.bank_hash,
            records=records,
            completed_neural_readbacks=calls,
            exact_total_neural_calls=None if failure else calls,
            maximum_neural_calls=3000,
            runtime_failure=failure,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
