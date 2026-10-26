"""Preregistered SIM-only paired physics forecast from a real measured boundary.

The foot correction is a privileged geometric teacher proposal, not a learned
policy, real execution, or contact guarantee. Compare both futures at the same
absolute H300 deadline, with the unchanged first-touch contract.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_measured_bridge_reference import condition_measured_bridge_reference
from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.contact_reach_teacher import bounded_contact_reach
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

ROOT = Path("/code/rosclaw/soccer_native_evidence")
SOURCE_HASH = "sha256:75612fbae5c1e4df0741c16b4a5000473aac829f919adbb5367027dce75cb151"
AUDIT_HASH = "sha256:db3b3f1b8d6553ba7df56878bbc292517668c79cf7a454984ea7440048be0c3f"
MODEL_HASH = "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    clean(args.expected_head)
    source_dir = ROOT / "rsi-cftrack-predicted-ball-h300-20261027"
    source_path = source_dir / "complete.json"
    audit_path = ROOT / "rsi-cftrack-predicted-ball-review-20261027/complete.json"
    if (
        args.output.exists()
        or digest(source_path) != SOURCE_HASH
        or digest(audit_path) != AUDIT_HASH
    ):
        raise ValueError("unused output and exact completed source and independent review required")
    source = json.loads(source_path.read_text())
    if source["status"] != "CFTRACK_PREDICTIVE_BALL_H300_PAIRED_NOT_PROMOTED":
        raise ValueError("expected failed predictive source required")
    pins = source["input_file_hashes"]

    def find(suffix):
        matches = [Path(k) for k in pins if k.endswith(suffix)]
        if (
            len(matches) != 1
            or matches[0].is_symlink()
            or digest(matches[0]) != pins[str(matches[0])]
        ):
            raise ValueError(f"exact uniquely pinned input required: {suffix}")
        return matches[0]

    scene = find("/cached-standard/declared-scene.xml")
    actor = find("/velocity/v0/exported/policy.onnx")
    config = find("/velocity/v0/params/deploy.yaml")
    external = find("/policy/omnicontact/OmniContact.py").parents[2]
    kick = find("/policy/omnicontact/model/kick_50k.onnx")
    origin_report = find("/rsi-warm-left-settle-v3-20261027/complete.json")
    history_report = find("/rsi-cftrack-measured-history-20261027/complete.json")
    bridge_report = find("/rsi-terminal-anchor-bridge-20261027/complete.json")

    def load_artifact(report, item):
        path = report.parent / item["path"]
        if path.is_symlink() or path.parent != report.parent or digest(path) != item["hash"]:
            raise ValueError("confined authenticated artifact required")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    old = load_artifact(origin_report, json.loads(origin_report.read_text())["artifact"])
    history = load_artifact(history_report, json.loads(history_report.read_text())["artifact"])
    bridge = load_artifact(
        bridge_report, json.loads(bridge_report.read_text())["records"][0]["artifact"]
    )
    prior = load_artifact(source_path, source["records"][1]["actual_execution"]["artifact"])
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source=SOURCE_HASH,
            independent_audit=AUDIT_HASH,
            source_head=args.expected_head,
            experiment="one damped Jacobian candidate against same-state parent",
            actual_prefix_controls=140,
            candidate_count=1,
            forecast_controls_per_arm=160,
            native_step_budget=4600,
            damping=0.02,
            joint_bound_rad=0.18,
            phase14_witness_cftrack_control=95,
            proposal_start_cftrack_control=83,
            absolute_scoring_horizon=300,
            privileged_future_teacher=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    if compiled_model_hash(model) != MODEL_HASH:
        raise ValueError("compiled body mismatch")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    stream = ContinuousVelocityStream(
        actor,
        config,
        expected_model_hash=pins[str(actor)],
        expected_config_hash=pins[str(config)],
        horizon_controls=301,
    )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = old["initial_qpos"], old["initial_qvel"]
    mujoco.mj_forward(model, data)
    effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    world = SharedNativeMotorWorld(
        model,
        data,
        players={
            "a/": NativePDParameters(
                np.array(stream.params.stiffness), np.array(stream.params.damping), effort
            )
        },
        ball_body_name="soccer_ball",
    )
    collector = CFtrackLiveHistory(world, "a/")
    raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
    for frame in range(57):
        q, v = lo.canonical(data)
        command = np.array([0, 0.25 if 25 <= frame < 57 else 0, 0], np.float32)
        target = stream.propose(q, v, lo.angular_velocity_body(data), command, frame=frame)
        row = world.advance({"a/": target})
        if not np.array_equal(target, old["joint_targets"][frame]) or any(
            not np.array_equal(row[k], old[k][frame * 10 : (frame + 1) * 10]) for k in raw
        ):
            raise ValueError(f"actual velocity replay mismatch {frame}")
        for k in raw:
            raw[k].append(row[k])
        if frame >= 52:
            collector.capture()
    policy = OmniContact(StateAndCmd(29), PolicyOutput(29), onnx_path=str(kick))
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    world.switch_pd_profile(
        "a/",
        NativePDParameters(policy.kps_lab[policy.lab2mj], policy.kds_lab[policy.lab2mj], effort),
    )
    player = SharedOmniContactPlayer(
        world, "a/", policy, expected_policy_type=OmniContact, goal=np.array([5.0, 0.0, 0.11])
    )
    install_owned_cftrack_history(player, collector)
    if not np.array_equal(policy.obs_history_buffer[1:], history["entry57_observations"][:4]):
        raise ValueError("live measured history mismatch")
    receipt = condition_measured_bridge_reference(
        player,
        bridge["candidate_root"],
        bridge["candidate_joint"],
        expected_model_hash=MODEL_HASH,
        ball_prediction_horizon_s=1.74,
    )
    if (
        receipt["conditioned_reference_hash"]
        != source["records"][1]["bridge_reference_receipt"]["conditioned_reference_hash"]
    ):
        raise ValueError("predictive reference mismatch")
    for frame in range(83):
        target, _ = player.propose()
        row = world.advance({"a/": target})
        if not np.array_equal(target, prior["joint_targets"][frame]) or any(
            not np.array_equal(row[k], prior[k][frame * 10 : (frame + 1) * 10]) for k in raw
        ):
            raise ValueError(f"actual CFtrack replay mismatch {frame}")
        for k in raw:
            raw[k].append(row[k])
        player.commit()
        if world.failed:
            raise ValueError("measured donor failed")
    if world.control_steps != 140 or policy.counter_step != 83:
        raise ValueError("exact causal donor clock required")
    forecaster = SharedCFtrackForecaster(
        player,
        expected_policy_type=OmniContact,
        expected_model_hash=MODEL_HASH,
        native_step_budget=3200,
    )
    parent = forecaster.forecast(controls=160)
    if parent["controls"] != 160 or any(
        not np.array_equal(parent[k], prior[k][830:2430]) for k in raw
    ):
        raise ValueError("parent future differs from original real physics")
    scratch = mujoco.MjData(model)
    witness = 12 * 10 + 9
    scratch.qpos[:], scratch.qvel[:] = parent["qpos"][witness], parent["qvel"][witness]
    mujoco.mj_forward(model, scratch)
    ball_geom = int(np.flatnonzero(model.geom_bodyid == lo.ball)[0])
    candidates = []
    for idx in range(1, 8):
        foot = model.geom(f"a/right_foot{idx}_collision").id
        points = np.zeros(6)
        gap = mujoco.mj_geomDistance(model, scratch, ball_geom, foot, 2.0, points)
        candidates.append((gap, foot, points.copy()))
    gap, foot, points = min(candidates, key=lambda x: x[0])
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, scratch, jac, None, points[3:], model.geom_bodyid[foot])
    proposal = bounded_contact_reach(
        jac[:, lo.vi[6:12]], points[:3] - points[3:], damping=0.02, max_joint_delta=0.18
    )
    delta = np.zeros(29)
    delta[6:12] = proposal.delta
    plan = CFtrackMotorResidualPlan(
        origin_counter=83,
        start_control=83,
        peak_control=12,
        delta_hg=tuple(float(x) for x in delta),
        donor_policy_hash=cftrack_state_hash(policy),
    )
    candidate = forecaster.forecast(controls=160, motor_plan=plan)

    def summarize(future, arm):
        arrays = {k: np.concatenate([*raw[k], future[k]]) for k in raw}
        np.savez_compressed(args.output / f"{arm}.npz", **arrays, targets=future["joint_targets"])
        if future["controls"] != 160:
            return dict(
                complete=False,
                runtime_failure=future["runtime_failure"],
                native_steps=future["native_steps"],
                body_failed=future["body_failed"],
            )
        positions = arrays["qpos"].reshape(300, 10, model.nq)
        forces = arrays["ball_forces"].reshape(300, 10, 1, 3).max(axis=1)
        force6 = np.zeros((300, 1, 6), dtype=np.float64)
        force6[:, :, :3] = forces
        outcome = label_first_touch(
            dict(
                force_n=force6,
                pelvis_z_per_substep_m=positions[:, :, lo.rq + 2][:, None, :],
                ball_position_after_step_m=positions[:, -1, lo.bq : lo.bq + 3][:, None, :],
            )
        ).to_dict()
        return dict(
            complete=True,
            runtime_failure=future["runtime_failure"],
            body_failed=future["body_failed"],
            outcome=outcome,
        )

    records = {
        name: summarize(future, name)
        for name, future in [("parent", parent), ("candidate", candidate)]
    }
    write_once(
        args.output / "complete.json",
        dict(
            status="BOUNDED_FOOT_REACH_PRIVATE_PHYSICS_NOT_PROMOTED",
            source=SOURCE_HASH,
            audit=AUDIT_HASH,
            source_head=args.expected_head,
            witness_gap_m=float(gap),
            witness_points=points.tolist(),
            raw_delta=proposal.raw_delta.tolist(),
            delta=proposal.delta.tolist(),
            jacobian=jac[:, lo.vi[6:12]].tolist(),
            predicted_linear_displacement=proposal.predicted_displacement.tolist(),
            records=records,
            counters=forecaster.counters(),
            actual_native_steps=1400,
            simulation_privileged_teacher=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
