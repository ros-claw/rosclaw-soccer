"""Four fixed-control diagnostic replays, not new closed-loop policy trials."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.ball_contact_replay_diagnostic import ball_contact_samples
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    root = Path(cfg["output_root"])
    if (
        root.exists()
        or cfg["new_replay_budget"] != 4
        or cfg["new_substep_budget"] != 12000
        or len(cfg["cases"]) != 4
        or len({r["case_id"] for r in cfg["cases"]}) != 4
        or cfg["mujoco_version"] != mujoco.__version__
        or any(
            cfg.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "neural_training_authorized")
        )
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or mujoco.get_mjcb_control() is not None
    ):
        raise ValueError("new frozen four-case same-version SIM diagnostic required")
    pins = {str(args.config): digest(args.config)}
    for relative in (
        "scripts/rsi_replay_first_touch_contacts.py",
        "src/rosclaw_soccer/physics/ball_contact_replay_diagnostic.py",
        "src/rosclaw_soccer/rsi/first_touch_outcome.py",
        "src/rosclaw_soccer/rsi/first_touch_qualification.py",
    ):
        pins[str(source / relative)] = digest(source / relative)
    receipts = []

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("source seal differs")
        pins[str(path)] = digest(path)
        return value

    for case in cfg["cases"]:
        receipt = sealed(case["receipt_path"], "receipt_hash")
        if (
            pins[case["receipt_path"]] != case["receipt_file_hash"]
            or receipt["receipt_hash"] != case["receipt_hash"]
            or receipt["witnesses"]["world"]["sha256"] != cfg["world_hash"]
        ):
            raise ValueError("exact original receipt and compiled world required")
        for witness in receipt["witnesses"].values():
            if digest(witness["path"]) != witness["sha256"]:
                raise ValueError("whole original witness bytes changed")
            pins[witness["path"]] = witness["sha256"]
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            outcome = label_first_touch(dict(archive))
        review = sealed(receipt["witnesses"]["independent_review"]["path"])
        foundation = sealed(receipt["witnesses"]["foundation_review"]["path"])
        if (
            outcome.to_dict() != receipt["outcome"]
            or require_first_touch_qualification(review, foundation, outcome)
            != receipt["qualification_alignment"]
        ):
            raise ValueError("original primitive and replay qualifications differ")
        receipts.append(receipt)
    if args.preflight:
        print("FOUR_FIXED_REPLAYS_12000_SUBSTEPS_PREFLIGHT_ZERO_MODEL_ALLOCATIONS", flush=True)
        return
    root.mkdir()
    declaration = dict(
        cfg,
        input_file_hashes=pins,
        source_control_clock="original recorded torque every 2ms",
        new_training_episodes=0,
        neural_inference_repeated=False,
        new_closed_loop_policy_trials=0,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    attempted, integrated = 0, 0
    current_case = None
    rows = []
    model = mujoco.MjModel.from_binary_path(receipts[0]["witnesses"]["world"]["path"])
    if (
        (model.nq, model.nv, model.nu) != (43, 41, 29)
        or model.opt.timestep != 0.002
        or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
        or model.nplugin
    ):
        raise ValueError("unchanged 2ms Euler canonical G1 model required")
    qi = [int(model.jnt_qposadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    vi = [int(model.jnt_dofadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
    ai = []
    for joint in joints:
        ids = np.flatnonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == joint)
        )
        if len(ids) != 1:
            raise ValueError("one original actuator per canonical joint required")
        ai.append(int(ids[0]))
    ball = int(model.body("ball").id)
    bg = int(model.geom("ball_geom").id)
    bj = int(model.body_jntadr[ball])
    bq, bv = int(model.jnt_qposadr[bj]), int(model.jnt_dofadr[bj])
    if (
        model.geom_size[bg, 0] != 0.11
        or model.body_mass[ball] != 0.43
        or np.any(model.dof_damping[bv : bv + 6] != 0)
    ):
        raise ValueError("unchanged ball dimensions/mass/damping required")
    body_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_knee_link",
        "right_knee_link",
    )
    bodies = [int(model.body(n).id) for n in body_names]
    pelvis = int(model.body("pelvis").id)

    def equal(a, b, name):
        if not np.allclose(a, b, rtol=0, atol=1e-8):
            raise ValueError("fixed-control physics replay differs: " + name)

    try:
        for case, receipt in zip(cfg["cases"], receipts, strict=True):
            current_case = case["case_id"]
            write_once(
                root / f"{current_case}-started.json",
                dict(
                    case_id=current_case,
                    receipt_hash=receipt["receipt_hash"],
                    replay_attempt_charged=True,
                ),
            )
            with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                trace = dict(archive)
            q, v, tau = trace["canonical_qpos"], trace["canonical_qvel"], trace["torque_nm"]
            if any(
                a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all()
                for a, shape in ((q, (300, 1, 43)), (v, (300, 1, 41)), (tau, (300, 1, 10, 29)))
            ):
                raise ValueError("complete finite original state/control arrays required")
            if np.any(np.abs(tau) > np.asarray(G1_HARD_TORQUE_LIMITS) + 1e-8):
                raise ValueError("recorded control exceeds original hard torque limits")
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
            events = []
            forces = np.zeros((3000, 3))
            robot_forces = np.zeros_like(forces)
            poses = np.zeros((3000, 43))
            velocities = np.zeros((3000, 41))
            velocities_after = np.zeros_like(velocities)
            maximum_force_balance_error = 0.0
            for frame in range(300):
                equal(
                    np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7])),
                    q[frame, 0],
                    "frame qpos",
                )
                equal(
                    np.concatenate((data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])),
                    v[frame, 0],
                    "frame qvel",
                )
                peaks = np.zeros(6)
                for substep in range(10):
                    step = frame * 10 + substep
                    poses[step] = np.concatenate(
                        (data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7])
                    )
                    velocities[step] = np.concatenate(
                        (data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])
                    )
                    before = data.qvel.copy()
                    data.ctrl[ai] = tau[frame, 0, substep]
                    attempted += 1
                    mujoco.mj_step(model, data)
                    integrated += 1
                    velocities_after[step] = np.concatenate(
                        (data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6])
                    )
                    equal(
                        data.actuator_force[ai],
                        trace["actual_actuator_force_nm"][frame, 0, substep],
                        "actual actuator force",
                    )
                    equal(
                        data.xpos[pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, substep],
                        "pelvis height",
                    )
                    contacts = ball_contact_samples(model, data, bg, before)
                    for contact in contacts:
                        force = np.asarray(contact["force_on_ball_world_n"])
                        forces[step] += force
                        body = contact["other_body"]
                        if body != 0:
                            robot_forces[step] += force
                        if body in bodies:
                            channel = bodies.index(body)
                            peaks[channel] = max(peaks[channel], contact["force_norm_n"])
                        if body != 0 and contact["force_norm_n"] > 1.0:
                            events.append(
                                dict(
                                    contact,
                                    substep_index=step,
                                    frame=frame,
                                    ball_center_velocity_before_world_m_s=before[
                                        bv : bv + 3
                                    ].tolist(),
                                    other_body_origin_world_m=data.xpos[body].tolist(),
                                )
                            )
                    error = float(np.max(np.abs(forces[step] - data.qfrc_constraint[bv : bv + 3])))
                    maximum_force_balance_error = max(maximum_force_balance_error, error)
                    if error > 1e-7:
                        raise ValueError(
                            "contact world-force convention differs from ball generalized force"
                        )
                equal(peaks, trace["force_n"][frame, 0], "all six contact force channels")
                equal(
                    data.xpos[ball],
                    trace["ball_position_after_step_m"][frame, 0],
                    "ball after frame",
                )
            arrays_path = root / f"{current_case}-substeps.npz"
            with arrays_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    canonical_qpos_before_substep=poses,
                    canonical_qvel_before_substep=velocities,
                    canonical_qvel_after_substep=velocities_after,
                    all_contact_force_on_ball_world_n=forces,
                    robot_contact_force_on_ball_world_n=robot_forces,
                )
            foot_events = [e for e in events if e["other_body"] in bodies[:2]]
            first = None if not foot_events else foot_events[0]["substep_index"]
            window = None
            if first is not None:
                end = min(first + 10, 3000)
                window = dict(
                    first_substep=first,
                    last_substep_inclusive=end - 1,
                    robot_contact_impulse_world_n_s=(
                        robot_forces[first:end].sum(axis=0) * 0.002
                    ).tolist(),
                    ball_center_velocity_before_world_m_s=velocities[first, 35:38].tolist(),
                    ball_center_velocity_after_world_m_s=velocities_after[end - 1, 35:38].tolist(),
                )
            result = dict(
                case_id=current_case,
                source_receipt_hash=receipt["receipt_hash"],
                original_outcome=receipt["outcome"],
                new_replay_substeps=3000,
                first_foot_contact_event=None if not foot_events else foot_events[0],
                first_foot_20ms_window=window,
                robot_contact_events_over_1n=events,
                unmonitored_robot_contact_events_over_1n=sum(
                    e["other_body"] not in bodies for e in events
                ),
                maximum_world_contact_force_balance_error_n=maximum_force_balance_error,
                arrays=dict(path=str(arrays_path), sha256=digest(arrays_path)),
                new_closed_loop_policy_trial=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
            result["report_hash"] = hash_json(result)
            write_once(root / f"{current_case}-complete.json", result)
            rows.append(result)
            print("CONTACT_REPLAY_COMPLETE", current_case, flush=True)
        if any(digest(path) != h for path, h in pins.items()):
            raise ValueError("original source/input bytes changed")
        complete = dict(
            schema="soccer.rsi.fixed_control_contact_replay_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            attempted_step_calls=attempted,
            integrated_substeps=integrated,
            original_closed_loop_policy_reexecuted=False,
            neural_inference_repeated=False,
            shared_payload_logical_seals_recomputed_here=False,
            contact_kinematic_velocity_clock=(
                "saved before mj_step; Jacobians at that step's collision stage"
            ),
            new_training_episodes=0,
            new_optimizer_updates=0,
            new_independent_courses=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        complete["report_hash"] = hash_json(complete)
        write_once(root / "complete.json", complete)
        print(complete["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "fatal.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                case_id=current_case,
                attempted_step_calls=attempted,
                integrated_substeps=integrated,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
