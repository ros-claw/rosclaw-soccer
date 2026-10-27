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
from rosclaw_soccer.rsi.cftrack_contact_reference import CFtrackContactPlan
from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.cftrack_measured_bridge_reference import condition_measured_bridge_reference
from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan
from rosclaw_soccer.rsi.contact_reach_teacher import bounded_contact_reach
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.measured_contact_response import (
    fit_measured_contact_response,
    propose_bounded_response,
)
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.sim_contact_bandit import (
    ACTION_NAMES,
    action_delta,
    contact_features,
    select_contact_action,
)
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
    parser.add_argument("--exclude-right-ankle-roll-after-rejection", action="store_true")
    parser.add_argument("--exclude-both-ankles-after-rejection", action="store_true")
    parser.add_argument("--calibrate-hip-knee-response", action="store_true")
    parser.add_argument("--replay-contact-arm", choices=("hg7-plus", "hg9-minus"))
    parser.add_argument("--measured-response-candidate", action="store_true")
    parser.add_argument("--static-contact-calibration", action="store_true")
    parser.add_argument("--replay-static-contact", action="store_true")
    parser.add_argument(
        "--static-holdout", choices=("yminus-vminus", "yminus-vplus", "yplus-vminus", "yplus-vplus")
    )
    parser.add_argument(
        "--static-training-case",
        choices=(
            "yminus-vminus",
            "yminus-vplus",
            "yplus-vminus",
            "yplus-vplus",
            "yminus-near-vminus",
            "yminus-near-vplus",
            "yminus-far-vminus",
            "yminus-far-vplus",
            "yplus-near-vminus",
            "yplus-near-vplus",
            "yplus-far-vminus",
            "yplus-far-vplus",
        ),
    )
    parser.add_argument(
        "--causal-intercept-training-case",
        choices=("yminus-vminus", "yminus-vplus", "yplus-vminus", "yplus-vplus"),
    )
    parser.add_argument(
        "--bandit-eval-case",
        choices=("yminus-vminus", "yminus-vplus", "yplus-vminus", "yplus-vplus"),
    )
    parser.add_argument(
        "--reference-contact-training-case",
        choices=("yminus-near-vminus", "yminus-far-vminus", "yplus-near-vplus", "yplus-far-vplus"),
    )
    args = parser.parse_args()
    if args.reference_contact_training_case and (
        args.bandit_eval_case
        or args.causal_intercept_training_case
        or args.static_training_case
        or args.static_holdout
        or args.replay_static_contact
        or args.static_contact_calibration
        or args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("reference-contact candidate is a separate experiment")
    if args.bandit_eval_case and (
        args.causal_intercept_training_case
        or args.static_training_case
        or args.static_holdout
        or args.replay_static_contact
        or args.static_contact_calibration
        or args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("bandit evaluation is a separate experiment")
    if args.causal_intercept_training_case and (
        args.static_training_case
        or args.static_holdout
        or args.replay_static_contact
        or args.static_contact_calibration
        or args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("causal intercept is a separate experiment")
    if args.static_training_case and (
        args.static_holdout
        or args.replay_static_contact
        or args.static_contact_calibration
        or args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("training case is a separate experiment")
    if args.static_holdout and (
        args.replay_static_contact
        or args.static_contact_calibration
        or args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("holdout is a separate experiment")
    if args.replay_static_contact and (
        args.static_contact_calibration
        or args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("static actual replay is a separate experiment")
    if args.static_contact_calibration and (
        args.measured_response_candidate
        or args.replay_contact_arm
        or args.calibrate_hip_knee_response
        or args.exclude_both_ankles_after_rejection
        or args.exclude_right_ankle_roll_after_rejection
    ):
        raise ValueError("static-contact calibration is a separate experiment")
    static_context = bool(
        args.static_contact_calibration
        or args.replay_static_contact
        or args.static_holdout
        or args.static_training_case
        or args.causal_intercept_training_case
        or args.bandit_eval_case
        or args.reference_contact_training_case
    )
    origin = 0 if args.causal_intercept_training_case else 50 if static_context else 83
    horizon = 243 - origin
    prefix_controls = 57 + origin
    calibration = bool(
        args.static_contact_calibration
        or args.static_training_case
        or args.calibrate_hip_knee_response
    )
    varied = bool(
        args.static_holdout
        or args.static_training_case
        or args.causal_intercept_training_case
        or args.bandit_eval_case
        or args.reference_contact_training_case
    )
    calibration_axes = (
        list(range(4))
        if args.static_contact_calibration or args.static_training_case
        else list(range(6, 10))
    )
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
    rejection = ROOT / "rsi-contact-reach-forecast-20261027/complete.json"
    if varied:
        verified_replay = ROOT / "rsi-static-contact-replay-20261027/complete.json"
        if (
            digest(verified_replay)
            != "sha256:7165f3ab2af1f424a3537831203a727fe6599d32a3978ca032713c64dc42d174"
        ):
            raise ValueError("independently replayed passing source required")
        replay_report = json.loads(verified_replay.read_text())
        if not replay_report["outcome"]["task_pass"] or not replay_report["private_future_exact"]:
            raise ValueError("actual fixed-case pass required")
        case_name = (
            args.static_holdout
            or args.static_training_case
            or args.causal_intercept_training_case
            or args.bandit_eval_case
            or args.reference_contact_training_case
        )
        y_amplitude = (
            0.03
            if args.bandit_eval_case
            else 0.01
            if (args.static_training_case or args.reference_contact_training_case)
            and "-near-" in case_name
            else 0.05
            if (args.static_training_case or args.reference_contact_training_case)
            and "-far-" in case_name
            else 0.04
            if args.static_training_case or args.causal_intercept_training_case
            else 0.02
        )
        v_amplitude = (
            0.015
            if args.bandit_eval_case
            else 0.005
            if (args.static_training_case or args.reference_contact_training_case)
            and ("-near-" in case_name or "-far-" in case_name)
            else 0.01
            if args.static_training_case or args.causal_intercept_training_case
            else 0.02
        )
        y_delta = -y_amplitude if case_name.startswith("yminus") else y_amplitude
        v_delta = -v_amplitude if case_name.endswith("vminus") else v_amplitude
        if args.static_training_case and (
            "-near-" in args.static_training_case or "-far-" in args.static_training_case
        ):
            previous_audit = ROOT / "rsi-contact-bandit-audit-20261027/complete.json"
            if (
                digest(previous_audit)
                != "sha256:d5f34db14f95ca4a2b10dbd079f36c0d809516b713bde7ec76129a9fd614931c"
            ):
                raise ValueError("exact failed-bandit audit required before expanded training")
    if args.reference_contact_training_case:
        grid_audit_path = ROOT / "rsi-contact-grid-audit-20261027/complete.json"
        grid_audit_hash = "sha256:fbf28437058d0978ecb83c5b9733623a12e84470d5194298a2cccc6f1fa83830"
        if digest(grid_audit_path) != grid_audit_hash:
            raise ValueError("exact independently audited grid evidence required")
        grid_audit = json.loads(grid_audit_path.read_text())
        if grid_audit["status"] != "STATIC_GRID_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION":
            raise ValueError("audited grid required")
        grid_parent_hash = grid_audit["records"][args.reference_contact_training_case]["arms"][
            "parent"
        ]["artifact_hash"]
        grid_parent_path = (
            ROOT / f"rsi-contact-grid-{args.reference_contact_training_case}-20261027/parent.npz"
        )
        if grid_parent_path.is_symlink() or digest(grid_parent_path) != grid_parent_hash:
            raise ValueError("exact audited grid parent required")
        with np.load(grid_parent_path, allow_pickle=False) as f:
            grid_parent = {k: f[k].copy() for k in f.files}
    if args.bandit_eval_case:
        training_path = ROOT / "rsi-contact-bandit-training-20261027/complete.json"
        training_hash = "sha256:541d80cd4b8f87753a60b1f0cbff0e56097e1b74dbff199005b49a954d4e921f"
        if digest(training_path) != training_hash:
            raise ValueError("exact preregistered training report required")
        training = json.loads(training_path.read_text())
        if training["status"] != "OFFLINE_CONTEXTUAL_ACTOR_CRITIC_TRAINED_NOT_EVALUATED" or [
            x["seed"] for x in training["records"]
        ] != [17, 43, 71]:
            raise ValueError("all three declared training seeds required")
        bandit_models = []
        for row in training["records"]:
            path = training_path.parent / row["model_path"]
            if (
                path.is_symlink()
                or path.parent != training_path.parent
                or digest(path) != row["model_hash"]
            ):
                raise ValueError("exact confined trained seed required")
            item = json.loads(path.read_text())
            if (
                item["seed"] != row["seed"]
                or item["action_names"] != list(ACTION_NAMES)
                or item["hardware_authorized"] is not False
            ):
                raise ValueError("compatible unpromoted model required")
            bandit_models.append((row["seed"], row["model_hash"], item))
    if args.causal_intercept_training_case:
        training_audit = ROOT / "rsi-approach-training-audit-20261027/complete.json"
        if (
            digest(training_audit)
            != "sha256:d58a37ad318d3edcc50710adb83ff1e88f19e0bf8720b853b06ee364fd78d50a"
        ):
            raise ValueError("exact complete training evidence audit required")
        training_report = json.loads(training_audit.read_text())
        if (
            training_report["status"]
            != "STATIC_TRAINING_HASH_FORCE_GEOMETRY_READBACK_NOT_EXECUTION"
        ):
            raise ValueError("audited training evidence required")
        parent_outcome = training_report["records"][args.causal_intercept_training_case]["arms"][
            "parent"
        ]["outcome"]
    if args.replay_static_contact:
        static_dir = ROOT / "rsi-static-approach-calibration-20261027"
        static_report_path = static_dir / "complete.json"
        static_hash = "sha256:c5a375a56713fd21b304ffb86a3ae8802c1fac1daa7b939e5f739daca527884e"
        if digest(static_report_path) != static_hash:
            raise ValueError("exact static paired calibration required")
        static_report = json.loads(static_report_path.read_text())
        static_record = static_report["records"]["hg1-plus"]
        static_arm_hash = "sha256:e85855ae8f52511f2f760dc7abffd6762c3870cdde5e2f9810eb71a7e1809ffc"
        arm_path = static_dir / "hg1-plus.npz"
        if (
            static_report["status"] != "CFTRACK_STATIC_APPROACH_CONTACT_CALIBRATION_NOT_PROMOTED"
            or static_record["artifact_hash"] != static_arm_hash
            or digest(arm_path) != static_arm_hash
            or static_record["outcome"]["task_pass"] is not True
        ):
            raise ValueError("complete hashed candidate with original task pass required")
        with np.load(arm_path, allow_pickle=False) as f:
            replay_source = {key: f[key].copy() for key in f.files}
    if args.measured_response_candidate:
        if (
            args.replay_contact_arm
            or args.calibrate_hip_knee_response
            or args.exclude_both_ankles_after_rejection
            or args.exclude_right_ankle_roll_after_rejection
        ):
            raise ValueError("response-model candidate is a separate experiment")
        response_report = ROOT / "rsi-hip-knee-response-20261027/complete.json"
        response_hash = "sha256:765c411cfde3d7f5716dc57d30df5cf8a30b44695db7b094aba38047ddbeb29a"
        if digest(response_report) != response_hash:
            raise ValueError("exact measured response source required")
        response_model = fit_measured_contact_response(
            json.loads(response_report.read_text()), source_hash=response_hash
        )
        learned_delta, response_receipt = propose_bounded_response(response_model)
    if args.exclude_right_ankle_roll_after_rejection:
        if (
            digest(rejection)
            != "sha256:e74da7e3dbad9535e3698d66fab43d482c5ce6bd0107dac099197433496d567a"
        ):
            raise ValueError("exact earlier physical-limit rejection required")
        rejected = json.loads(rejection.read_text())["records"]["candidate"]["runtime_failure"]
        if rejected["type"] != "MotorTargetRejected" or rejected["target_rejection"][
            "violating_hg_indices"
        ] != [11]:
            raise ValueError("right ankle roll rejection required")
    if args.exclude_both_ankles_after_rejection:
        if args.exclude_right_ankle_roll_after_rejection:
            raise ValueError("select exactly one rejection-informed experiment")
        constrained = ROOT / "rsi-contact-reach-constrained-20261027/complete.json"
        if (
            digest(constrained)
            != "sha256:6abf4ea2a3f2b1bb7859828f5e90583503d4b2a69990b2cca7a9e95be25e2fee"
        ):
            raise ValueError("exact earlier second physical-limit rejection required")
        rejected = json.loads(constrained.read_text())["records"]["candidate"]["runtime_failure"]
        if rejected["type"] != "MotorTargetRejected" or rejected["target_rejection"][
            "violating_hg_indices"
        ] != [10]:
            raise ValueError("right ankle pitch rejection required")
    if args.calibrate_hip_knee_response:
        if (
            args.exclude_both_ankles_after_rejection
            or args.exclude_right_ankle_roll_after_rejection
        ):
            raise ValueError("calibration is a separate preregistered experiment")
        prior_hip = ROOT / "rsi-contact-reach-hip-knee-20261027/complete.json"
        if (
            digest(prior_hip)
            != "sha256:eea7cbbe6f37e9d210c0ee42b006637068bf3f2cfcf240615eebf7f566d6c518"
        ):
            raise ValueError("exact completed hip-knee failure required")
        failed = json.loads(prior_hip.read_text())["records"]["candidate"]
        if not failed["complete"] or failed["outcome"]["task_pass"]:
            raise ValueError("completed but unsuccessful hip-knee candidate required")
    if args.replay_contact_arm:
        if (
            args.calibrate_hip_knee_response
            or args.exclude_both_ankles_after_rejection
            or args.exclude_right_ankle_roll_after_rejection
        ):
            raise ValueError("actual replay is a separate experiment")
        calibration_dir = ROOT / "rsi-hip-knee-response-20261027"
        if (
            digest(calibration_dir / "complete.json")
            != "sha256:765c411cfde3d7f5716dc57d30df5cf8a30b44695db7b094aba38047ddbeb29a"
        ):
            raise ValueError("exact completed calibration required")
        arm_hashes = {
            "hg7-plus": "sha256:54ea1ceb0d6d28f9096c3272e9c398fd41303c2d738ad7320e9bc7285ba8960f",
            "hg9-minus": "sha256:10c18e9b6a9988626587894ff0ed454437055c32445f84dfeaffc20b6c6bcfb9",
        }
        arm_path = calibration_dir / f"{args.replay_contact_arm}.npz"
        if arm_path.is_symlink() or digest(arm_path) != arm_hashes[args.replay_contact_arm]:
            raise ValueError("exact independently captured private arm artifact required")
        with np.load(arm_path, allow_pickle=False) as f:
            replay_source = {key: f[key].copy() for key in f.files}
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
    source_record = source["records"][0 if static_context else 1]
    prior = load_artifact(source_path, source_record["actual_execution"]["artifact"])
    if static_context:
        outcome = source_record["actual_execution"]["absolute_h300_outcome"]
        if outcome["first_foot_contact_frame"] != 120 or outcome["task_pass"] is not False:
            raise ValueError("exact failed approach-contact baseline required")
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            source=SOURCE_HASH,
            independent_audit=AUDIT_HASH,
            source_head=args.expected_head,
            experiment=(
                "causal bounded left-foot reference candidate"
                if args.reference_contact_training_case
                else "all-seed paired actor-critic holdout"
                if args.bandit_eval_case
                else "causal intercept actual simulation on training state"
                if args.causal_intercept_training_case
                else "paired held-out ball state, fixed candidate"
                if args.static_holdout
                else "actual deterministic replay"
                if args.replay_contact_arm or args.replay_static_contact
                else "paired dynamics calibration"
                if calibration
                else "one damped Jacobian candidate against same-state parent"
            ),
            actual_replay_arm=args.replay_contact_arm,
            actual_static_replay=args.replay_static_contact,
            static_holdout=args.static_holdout,
            static_training_case=args.static_training_case,
            bandit_eval_case=args.bandit_eval_case,
            reference_contact_training_case=args.reference_contact_training_case,
            reference_source_grid_audit=grid_audit_hash
            if args.reference_contact_training_case
            else None,
            reference_candidate_displacement_cap_m=0.08
            if args.reference_contact_training_case
            else None,
            bandit_training_hash=training_hash if args.bandit_eval_case else None,
            causal_intercept_training_case=args.causal_intercept_training_case,
            ball_prediction_horizon_s=1.26 if args.causal_intercept_training_case else None,
            initial_ball_y_offset_m=y_delta if varied else None,
            initial_ball_vy_offset_m_s=v_delta if varied else None,
            measured_response_candidate=args.measured_response_candidate,
            static_approach_contact_calibration=args.static_contact_calibration,
            calibration_artifact_hash=(
                arm_hashes[args.replay_contact_arm] if args.replay_contact_arm else None
            ),
            actual_prefix_controls=prefix_controls,
            candidate_count=0
            if args.replay_contact_arm
            or args.replay_static_contact
            or args.causal_intercept_training_case
            else 3
            if args.bandit_eval_case
            else 8
            if calibration
            else 1,
            excluded_hg_indices=(
                [10, 11]
                if args.exclude_both_ankles_after_rejection
                else [11]
                if args.exclude_right_ankle_roll_after_rejection
                else []
            ),
            forecast_controls_per_arm=horizon,
            native_step_budget=3000
            if args.replay_contact_arm
            or args.replay_static_contact
            or args.causal_intercept_training_case
            else 8790
            if args.bandit_eval_case
            else 4930
            if args.reference_contact_training_case
            else 4930
            if args.static_holdout
            else 18440
            if args.static_contact_calibration or args.static_training_case
            else 15800
            if args.calibrate_hip_knee_response
            else 4600,
            calibration_axes_hg=calibration_axes if calibration else [],
            calibration_delta_rad=0.08 if calibration else None,
            damping=0.02,
            joint_bound_rad=0.18,
            witness_cftrack_control=origin + 12,
            proposal_start_cftrack_control=origin,
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
    if varied:
        data.qpos[lo.bq + 1] += y_delta
        data.qvel[lo.bv + 1] += v_delta
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
        if not varied and (
            not np.array_equal(target, old["joint_targets"][frame])
            or any(not np.array_equal(row[k], old[k][frame * 10 : (frame + 1) * 10]) for k in raw)
        ):
            raise ValueError(f"actual velocity replay mismatch {frame}")
        for k in raw:
            raw[k].append(row[k])
        if frame >= 52:
            collector.capture()
        if world.failed:
            raise ValueError("actual velocity prefix body failed")
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
    if not varied and not np.array_equal(
        policy.obs_history_buffer[1:], history["entry57_observations"][:4]
    ):
        raise ValueError("live measured history mismatch")
    receipt = condition_measured_bridge_reference(
        player,
        bridge["candidate_root"],
        bridge["candidate_joint"],
        expected_model_hash=MODEL_HASH,
        ball_prediction_horizon_s=(
            1.26 if args.causal_intercept_training_case else None if static_context else 1.74
        ),
    )
    if not varied and (
        receipt["conditioned_reference_hash"]
        != source_record["bridge_reference_receipt"]["conditioned_reference_hash"]
    ):
        raise ValueError("predictive reference mismatch")
    for frame in range(origin):
        target, _ = player.propose()
        row = world.advance({"a/": target})
        if not varied and (
            not np.array_equal(target, prior["joint_targets"][frame])
            or any(not np.array_equal(row[k], prior[k][frame * 10 : (frame + 1) * 10]) for k in raw)
        ):
            raise ValueError(f"actual CFtrack replay mismatch {frame}")
        for k in raw:
            raw[k].append(row[k])
        player.commit()
        if world.failed:
            raise ValueError("measured donor failed")
    if world.control_steps != prefix_controls or policy.counter_step != origin:
        raise ValueError("exact causal donor clock required")
    if args.causal_intercept_training_case:
        parent_path = (
            ROOT / f"rsi-approach-train-{args.causal_intercept_training_case}-20261027/parent.npz"
        )
        parent_hash = training_report["records"][args.causal_intercept_training_case]["arms"][
            "parent"
        ]["artifact_hash"]
        if parent_path.is_symlink() or digest(parent_path) != parent_hash:
            raise ValueError("audited exact training parent trace required")
        with np.load(parent_path, allow_pickle=False) as f:
            if any(not np.array_equal(np.concatenate(raw[k]), f[k][:570]) for k in raw):
                raise ValueError(
                    "intercept and training parent do not share the same actual prefix"
                )
        phases = policy.ref_phase.copy()
        original_session = policy.ort_session
        obs, actions, time_steps, actual_targets = [], [], [], []

        class InterceptCapture:
            def run(self, outputs, inputs):
                result = original_session.run(outputs, inputs)
                obs.append(inputs["obs"].copy())
                time_steps.append(inputs["time_step"].copy())
                actions.append(result[0].copy())
                return result

        policy.ort_session = InterceptCapture()
        failure = None
        try:
            for _ in range(243):
                target, _ = player.propose()
                row = world.advance({"a/": target})
                for key in raw:
                    raw[key].append(row[key])
                actual_targets.append(target.copy())
                player.commit()
                if world.failed:
                    raise ValueError("actual intercept body failed")
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error))
        arrays = {key: np.concatenate(raw[key]) for key in raw}
        path = args.output / "actual.npz"
        np.savez_compressed(
            path,
            **arrays,
            targets=np.asarray(actual_targets).reshape(-1, 29),
            neural_obs=np.asarray(obs),
            neural_actions=np.asarray(actions),
            neural_time_steps=np.asarray(time_steps),
        )
        outcome = None
        phase = None
        if failure is None and len(arrays["qpos"]) == 3000 and len(actual_targets) == 243:
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
            first = outcome["first_foot_contact_frame"]
            phase = (
                int(phases[min(first - 57, len(phases) - 1)])
                if first is not None and first >= 57
                else None
            )
        write_once(
            args.output / "complete.json",
            dict(
                status="CAUSAL_INTERCEPT_ACTUAL_TRAINING_NOT_PROMOTED",
                training_case=args.causal_intercept_training_case,
                source=SOURCE_HASH,
                training_audit_hash="sha256:d58a37ad318d3edcc50710adb83ff1e88f19e0bf8720b853b06ee364fd78d50a",
                parent_artifact_hash=parent_hash,
                common_actual_prefix_exact=True,
                reference_receipt=receipt,
                actual_artifact_hash=digest(path),
                actual_native_steps=len(arrays["qpos"]),
                actual_cftrack_calls=len(actions),
                failure=failure,
                first_contact_reference_phase=phase,
                outcome=outcome,
                parent_outcome=parent_outcome,
                privileged_sim_ball_velocity=True,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return
    if args.replay_contact_arm or args.replay_static_contact:
        axis, amount = (
            (1, 0.08)
            if args.replay_static_contact
            else (7, 0.08)
            if args.replay_contact_arm == "hg7-plus"
            else (9, -0.08)
        )
        residual = np.zeros(29)
        residual[axis] = amount
        player.install_motor_plan(
            CFtrackMotorResidualPlan(
                origin_counter=origin,
                start_control=origin,
                peak_control=12,
                delta_hg=tuple(float(x) for x in residual),
                donor_policy_hash=cftrack_state_hash(policy),
            )
        )
        phases = policy.ref_phase.copy()
        actual_targets = []
        original_session = policy.ort_session
        obs, actions = [], []

        class Capture:
            def run(self, outputs, inputs):
                result = original_session.run(outputs, inputs)
                obs.append(inputs["obs"].copy())
                actions.append(result[0].copy())
                return result

        policy.ort_session = Capture()
        for step in range(horizon):
            target, _ = player.propose()
            row = world.advance({"a/": target})
            if not np.array_equal(target, replay_source["targets"][step]) or any(
                not np.array_equal(
                    row[k],
                    replay_source[k][
                        prefix_controls * 10 + step * 10 : prefix_controls * 10 + (step + 1) * 10
                    ],
                )
                for k in raw
            ):
                raise ValueError(
                    f"independent actual replay differs from private forecast at {step}"
                )
            for k in raw:
                raw[k].append(row[k])
            actual_targets.append(target.copy())
            player.commit()
            if world.failed:
                raise ValueError("actual replay body failed")
        arrays = {k: np.concatenate(raw[k]) for k in raw}
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
        first = outcome["first_foot_contact_frame"]
        phase = int(phases[min(first - 57, len(phases) - 1)]) if first is not None else None
        path = args.output / "actual.npz"
        np.savez_compressed(
            path,
            **arrays,
            targets=np.stack(actual_targets),
            neural_obs=np.stack(obs),
            neural_actions=np.stack(actions),
        )
        write_once(
            args.output / "complete.json",
            dict(
                status="ACTUAL_CALIBRATION_CONTACT_REPLAY_NOT_PROMOTED",
                arm="static-hg1-plus" if args.replay_static_contact else args.replay_contact_arm,
                source=SOURCE_HASH,
                calibration_hash=(
                    static_hash
                    if args.replay_static_contact
                    else "sha256:765c411cfde3d7f5716dc57d30df5cf8a30b44695db7b094aba38047ddbeb29a"
                ),
                calibration_artifact_hash=(
                    static_arm_hash
                    if args.replay_static_contact
                    else arm_hashes[args.replay_contact_arm]
                ),
                source_head=args.expected_head,
                actual_artifact_hash=digest(path),
                actual_controls=300,
                native_steps=3000,
                neural_calls=horizon,
                private_future_exact=True,
                first_contact_reference_phase=phase,
                outcome=outcome,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return
    forecaster = SharedCFtrackForecaster(
        player,
        expected_policy_type=OmniContact,
        expected_model_hash=MODEL_HASH,
        native_step_budget=(
            4 * horizon * 10
            if args.bandit_eval_case
            else 9 * horizon * 10
            if calibration
            else 2 * horizon * 10
            if args.static_holdout or args.reference_contact_training_case
            else 3200
        ),
    )
    parent = forecaster.forecast(controls=horizon)
    if args.reference_contact_training_case and any(
        not np.array_equal(np.concatenate([*raw[k], parent[k]]), grid_parent[k]) for k in raw
    ):
        raise ValueError("exact actual prefix and parent physics must match audited grid")
    if parent["controls"] != horizon or (
        not varied and any(not np.array_equal(parent[k], prior[k][origin * 10 : 2430]) for k in raw)
    ):
        raise ValueError("parent future differs from original real physics")
    scratch = mujoco.MjData(model)
    witness = 12 * 10 + 9
    scratch.qpos[:], scratch.qvel[:] = parent["qpos"][witness], parent["qvel"][witness]
    mujoco.mj_forward(model, scratch)
    ball_geom = int(np.flatnonzero(model.geom_bodyid == lo.ball)[0])
    candidates = []
    for idx in range(1, 8):
        foot = model.geom(f"a/{'left' if static_context else 'right'}_foot{idx}_collision").id
        points = np.zeros(6)
        gap = mujoco.mj_geomDistance(model, scratch, ball_geom, foot, 2.0, points)
        candidates.append((gap, foot, points.copy()))
    gap, foot, points = min(candidates, key=lambda x: x[0])
    if args.measured_response_candidate and not np.array_equal(
        points[:3] - points[3:], np.asarray(response_model.origin_relative_xyz_m)
    ):
        raise ValueError("measured response model must match the exact donor witness")
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, scratch, jac, None, points[3:], model.geom_bodyid[foot])
    active = np.arange(
        6,
        10
        if args.exclude_both_ankles_after_rejection or args.measured_response_candidate
        else 11
        if args.exclude_right_ankle_roll_after_rejection
        else 12,
    )
    proposal = bounded_contact_reach(
        jac[:, lo.vi[active]], points[:3] - points[3:], damping=0.02, max_joint_delta=0.18
    )
    delta = np.zeros(29)
    delta[active] = proposal.delta
    if args.measured_response_candidate:
        active = np.arange(6, 10)
        delta[:] = 0.0
        delta[active] = learned_delta
    if args.static_holdout:
        delta[:] = 0.0
        delta[1] = 0.08
    plan = CFtrackMotorResidualPlan(
        origin_counter=origin,
        start_control=origin,
        peak_control=12,
        delta_hg=tuple(float(x) for x in delta),
        donor_policy_hash=cftrack_state_hash(policy),
    )

    def summarize(future, arm):
        arrays = {k: np.concatenate([*raw[k], future[k]]) for k in raw}
        artifact = args.output / f"{arm}.npz"
        np.savez_compressed(artifact, **arrays, targets=future["joint_targets"])
        if future["controls"] != horizon:
            return dict(
                complete=False,
                runtime_failure=future["runtime_failure"],
                native_steps=future["native_steps"],
                body_failed=future["body_failed"],
                artifact_hash=digest(artifact),
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
            artifact_hash=digest(artifact),
        )

    if args.reference_contact_training_case:
        offset = points[:3] - points[3:]
        distance = float(np.linalg.norm(offset))
        if distance > 0.08:
            offset *= 0.08 / distance
        reference_pos = np.asarray(policy.ref_left_ankle_future_pos[origin + 12], dtype=np.float64)
        reference_quat = np.asarray(
            policy.ref_left_ankle_future_quat[origin + 12], dtype=np.float64
        )
        reference_quat /= np.linalg.norm(reference_quat)
        plan_ref = CFtrackContactPlan(
            origin_counter=origin,
            start_frame=origin,
            peak_control=12,
            active_foot="left",
            position_world=tuple(float(x) for x in reference_pos + offset),
            quaternion_world=tuple(float(x) for x in reference_quat),
            donor_policy_hash=cftrack_state_hash(policy),
            bounded_position_path=False,
        )
        records = {"parent": summarize(parent, "parent")}
        try:
            future = forecaster.forecast(controls=horizon, contact_plan=plan_ref)
        except Exception as error:
            records["candidate"] = dict(
                complete=False,
                forecast_exception=dict(type=type(error).__name__, message=str(error)),
            )
        else:
            records["candidate"] = summarize(future, "candidate")
            records["candidate"]["reference_conditioning"] = future["reference_conditioning"]
        write_once(
            args.output / "complete.json",
            dict(
                status="BOUNDED_FOOT_REFERENCE_TRAINING_NOT_PROMOTED",
                case=args.reference_contact_training_case,
                source_head=args.expected_head,
                grid_audit_hash=grid_audit_hash,
                grid_parent_hash=grid_parent_hash,
                witness_gap_m=float(gap),
                witness_points=points.tolist(),
                capped_offset_world_m=offset.tolist(),
                requested_ankle_reference_world_m=list(plan_ref.position_world),
                records=records,
                counters=forecaster.counters(),
                actual_prefix_native_steps=prefix_controls * 10,
                privileged_future_witness=True,
                geometry_not_contact_proof=True,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return

    if args.bandit_eval_case:
        state_features = contact_features(
            float(data.qpos[lo.bq + 1]), float(data.qpos[lo.rq + 1]), float(data.qvel[lo.bv + 1])
        )
        records = {"parent": summarize(parent, "parent")}
        for seed, model_hash, model_item in bandit_models:
            action = select_contact_action(
                state_features,
                np.asarray(model_item["actor_weight"]),
                np.asarray(model_item["actor_bias"]),
            )
            selected_delta = action_delta(action)
            selected_plan = (
                None
                if action == 0
                else CFtrackMotorResidualPlan(
                    origin_counter=origin,
                    start_control=origin,
                    peak_control=12,
                    delta_hg=tuple(float(x) for x in selected_delta),
                    donor_policy_hash=cftrack_state_hash(policy),
                )
            )
            future = forecaster.forecast(controls=horizon, motor_plan=selected_plan)
            summary = summarize(future, f"seed-{seed}")
            summary.update(
                seed=seed,
                model_hash=model_hash,
                selected_action=ACTION_NAMES[action],
                selected_delta_hg=selected_delta.tolist(),
            )
            first = summary.get("outcome", {}).get("first_foot_contact_frame")
            summary["first_contact_reference_phase"] = (
                int(policy.ref_phase[min(first - 57, len(policy.ref_phase) - 1)])
                if first is not None and first >= 57
                else None
            )
            records[f"seed-{seed}"] = summary
            write_once(args.output / f"seed-{seed}.json", summary)
        write_once(
            args.output / "complete.json",
            dict(
                status="OFFLINE_BANDIT_PAIRED_PRIVATE_HOLDOUT_NOT_PROMOTED",
                holdout_id=args.bandit_eval_case,
                training_hash=training_hash,
                source_head=args.expected_head,
                initial_ball_y_offset_m=y_delta,
                initial_ball_vy_offset_m_s=v_delta,
                features=state_features.tolist(),
                records=records,
                counters=forecaster.counters(),
                actual_prefix_native_steps=prefix_controls * 10,
                privileged_sim_ball_velocity=True,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return

    if calibration:
        records = {"parent": summarize(parent, "parent")}
        for axis in calibration_axes:
            for sign in (-1, 1):
                arm = f"hg{axis}-{'minus' if sign < 0 else 'plus'}"
                perturbation = np.zeros(29)
                perturbation[axis] = sign * 0.08
                calibration_plan = CFtrackMotorResidualPlan(
                    origin_counter=origin,
                    start_control=origin,
                    peak_control=12,
                    delta_hg=tuple(float(x) for x in perturbation),
                    donor_policy_hash=cftrack_state_hash(policy),
                )
                future = forecaster.forecast(controls=horizon, motor_plan=calibration_plan)
                summary = summarize(future, arm)
                if future["native_steps"] > witness:
                    scratch.qpos[:], scratch.qvel[:] = (
                        future["qpos"][witness],
                        future["qvel"][witness],
                    )
                    mujoco.mj_forward(model, scratch)
                    current_points = np.zeros(6)
                    current_gap = mujoco.mj_geomDistance(
                        model, scratch, ball_geom, foot, 2.0, current_points
                    )
                    summary["witness_gap_m"] = float(current_gap)
                    summary["witness_relative_xyz_m"] = (
                        current_points[:3] - current_points[3:]
                    ).tolist()
                summary["delta_hg_index"] = axis
                summary["delta_rad"] = sign * 0.08
                records[arm] = summary
                write_once(args.output / f"{arm}.json", summary)
        write_once(
            args.output / "complete.json",
            dict(
                status=(
                    "CFTRACK_STATIC_APPROACH_TRAINING_DATA_NOT_PROMOTED"
                    if args.static_training_case
                    else "CFTRACK_STATIC_APPROACH_CONTACT_CALIBRATION_NOT_PROMOTED"
                    if args.static_contact_calibration
                    else "CFTRACK_HIP_KNEE_PHYSICS_RESPONSE_CALIBRATION_NOT_PROMOTED"
                ),
                source=SOURCE_HASH,
                audit=AUDIT_HASH,
                source_head=args.expected_head,
                witness_cftrack_control=origin + 12,
                witness_parent_gap_m=float(gap),
                witness_parent_relative_xyz_m=(points[:3] - points[3:]).tolist(),
                witness_geom_id=int(foot),
                records=records,
                counters=forecaster.counters(),
                actual_native_steps=prefix_controls * 10,
                training_case=args.static_training_case,
                initial_ball_y_offset_m=y_delta if varied else None,
                initial_ball_vy_offset_m_s=v_delta if varied else None,
                simulation_privileged_teacher=True,
                physical_force_is_contact_authority=True,
                task_promoted=False,
                hardware_authorized=False,
            ),
        )
        return

    candidate = forecaster.forecast(controls=horizon, motor_plan=plan)
    records = {
        name: summarize(future, name)
        for name, future in [("parent", parent), ("candidate", candidate)]
    }
    write_once(
        args.output / "complete.json",
        dict(
            status=(
                "STATIC_CONTACT_PAIRED_HOLDOUT_NOT_PROMOTED"
                if args.static_holdout
                else "BOUNDED_FOOT_REACH_PRIVATE_PHYSICS_NOT_PROMOTED"
            ),
            source=SOURCE_HASH,
            audit=AUDIT_HASH,
            source_head=args.expected_head,
            witness_gap_m=float(gap),
            witness_points=points.tolist(),
            geometric_raw_delta_not_necessarily_applied=proposal.raw_delta.tolist(),
            geometric_bounded_delta_not_necessarily_applied=proposal.delta.tolist(),
            active_hg_indices=active.tolist(),
            jacobian=jac[:, lo.vi[active]].tolist(),
            predicted_linear_displacement=proposal.predicted_displacement.tolist(),
            response_model_receipt=response_receipt if args.measured_response_candidate else None,
            applied_delta_hg=delta.tolist(),
            applied_hg_indices=np.flatnonzero(delta).tolist(),
            records=records,
            counters=forecaster.counters(),
            actual_native_steps=prefix_controls * 10,
            holdout_id=args.static_holdout,
            initial_ball_y_offset_m=y_delta if args.static_holdout else None,
            initial_ball_vy_offset_m_s=v_delta if args.static_holdout else None,
            simulation_privileged_teacher=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
