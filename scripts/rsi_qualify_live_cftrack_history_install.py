"""Install owned histories and optionally examine bounded private futures."""

import argparse
import atexit
import copy
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--history-hash", required=True)
    p.add_argument("--live-review", type=Path, required=True)
    p.add_argument("--live-review-hash", required=True)
    p.add_argument("--private-forecast-proof", type=Path)
    p.add_argument("--private-forecast-proof-hash")
    p.add_argument("--execute-forecast-proof", type=Path)
    p.add_argument("--execute-forecast-proof-hash")
    p.add_argument("--measured-bridge", type=Path)
    p.add_argument("--measured-bridge-hash")
    p.add_argument("--bridge-certificate", type=Path)
    p.add_argument("--bridge-certificate-hash")
    p.add_argument("--static-bridge-proof", type=Path)
    p.add_argument("--static-bridge-proof-hash")
    p.add_argument("--static-bridge-review", type=Path)
    p.add_argument("--static-bridge-review-hash")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete, hist, live = a.source / "complete.json", a.history / "complete.json", a.live_review
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (complete, hist, live))
        or digest(complete) != a.source_hash
        or digest(hist) != a.history_hash
        or digest(live) != a.live_review_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output and exact isolated source lineage required")
    source, history, prior = (json.loads(x.read_text()) for x in (complete, hist, live))
    forecast_enabled = a.private_forecast_proof is not None
    execute_enabled = a.execute_forecast_proof is not None
    bridge_mode = a.measured_bridge is not None
    prediction_mode = a.static_bridge_proof is not None
    if (
        prediction_mode
        != all(
            value is not None
            for value in (
                a.static_bridge_proof_hash,
                a.static_bridge_review,
                a.static_bridge_review_hash,
            )
        )
        or (
            not prediction_mode
            and any(
                value is not None
                for value in (
                    a.static_bridge_proof_hash,
                    a.static_bridge_review,
                    a.static_bridge_review_hash,
                )
            )
        )
        or (prediction_mode and not bridge_mode)
    ):
        raise ValueError("prediction mode requires the exact paired source and independent review")
    if bridge_mode != all(
        value is not None
        for value in (a.measured_bridge_hash, a.bridge_certificate, a.bridge_certificate_hash)
    ) or (
        not bridge_mode
        and any(
            value is not None
            for value in (a.measured_bridge_hash, a.bridge_certificate, a.bridge_certificate_hash)
        )
    ):
        raise ValueError("bridge mode requires both exact source and independent certificate")
    if bridge_mode and not execute_enabled:
        raise ValueError("bridge mode requires exact original parent future evidence")
    if forecast_enabled != (a.private_forecast_proof_hash is not None):
        raise ValueError("private forecast requires both prior installation proof and hash")
    if execute_enabled != (a.execute_forecast_proof_hash is not None) or (
        execute_enabled and forecast_enabled
    ):
        raise ValueError("actual replay requires one exact forecast proof and no forecast mode")
    if forecast_enabled:
        if (
            a.private_forecast_proof.is_symlink()
            or digest(a.private_forecast_proof) != a.private_forecast_proof_hash
        ):
            raise ValueError("exact earlier closed installation report required")
        proof = json.loads(a.private_forecast_proof.read_text())
        if (
            proof["status"] != "LIVE_CFTRACK_HISTORY_INSTALLED_NOT_EXECUTED"
            or proof["source_hash"] != a.source_hash
            or proof["history_hash"] != a.history_hash
            or proof["live_review_hash"] != a.live_review_hash
            or [row["entry_control"] for row in proof["records"]] != [57, 82]
        ):
            raise ValueError("exact matching qualified two-boundary history install required")
    forecast_artifacts = {}
    if execute_enabled:
        if (
            a.execute_forecast_proof.is_symlink()
            or digest(a.execute_forecast_proof) != a.execute_forecast_proof_hash
        ):
            raise ValueError("exact earlier private future report required")
        future_proof = json.loads(a.execute_forecast_proof.read_text())
        if (
            future_proof["status"]
            != "CFTRACK_MEASURED_HISTORY_PRIVATE_FUTURES_RECORDED_NOT_PROMOTED"
            or future_proof["source_hash"] != a.source_hash
            or future_proof["history_hash"] != a.history_hash
            or future_proof["live_review_hash"] != a.live_review_hash
            or [row["entry_control"] for row in future_proof["records"]] != [57, 82]
        ):
            raise ValueError("matching qualified two-boundary private futures required")
        for row in future_proof["records"]:
            for arm in row["private_forecasts"]:
                path = a.execute_forecast_proof.parent / arm["artifact"]["path"]
                key = (row["entry_control"], arm["arm"])
                if (
                    key in forecast_artifacts
                    or path.is_symlink()
                    or path.parent != a.execute_forecast_proof.parent
                    or digest(path) != arm["artifact"]["hash"]
                    or arm["controls"] != 100
                    or arm["native_steps"] != 1000
                    or arm["runtime_failure"] is not None
                ):
                    raise ValueError(
                        "four complete exact confined private future artifacts required"
                    )
                forecast_artifacts[key] = (path, arm["artifact"]["hash"])
        if set(forecast_artifacts) != {
            (57, "measured_history"),
            (57, "zero_history"),
            (82, "measured_history"),
            (82, "zero_history"),
        }:
            raise ValueError("all four declared private futures required")
    bridge_artifact = None
    if bridge_mode:
        if (
            a.measured_bridge.is_symlink()
            or a.bridge_certificate.is_symlink()
            or digest(a.measured_bridge) != a.measured_bridge_hash
            or digest(a.bridge_certificate) != a.bridge_certificate_hash
        ):
            raise ValueError("exact bridge source and independent certificate required")
        bridge_source = json.loads(a.measured_bridge.read_text())
        certificate = json.loads(a.bridge_certificate.read_text())
        if (
            bridge_source["status"] != "MEASURED_TERMINAL_ANCHOR_BRIDGE_STATIC_STUDY_NOT_PHYSICS"
            or bridge_source["source_hash"] != a.source_hash
            or len(bridge_source["records"]) != 1
            or bridge_source["records"][0]["entry"] != 57
            or bridge_source["records"][0]["all_frames_pass"] is not False
            or certificate["status"] != "MEASURED_TERMINAL_ANCHOR_BRIDGE_READBACK_NOT_PHYSICS"
            or certificate["source_hash"] != a.measured_bridge_hash
            or len(certificate["records"]) != 1
            or certificate["records"][0]["sampled_geometry_certificate"] is not True
            or certificate["records"][0]["all_frames_pass"] is not False
        ):
            raise ValueError(
                "failed original gate and independent sampled geometry certificate required"
            )
        item = bridge_source["records"][0]["artifact"]
        bridge_artifact = a.measured_bridge.parent / item["path"]
        if (
            bridge_artifact.is_symlink()
            or bridge_artifact.parent != a.measured_bridge.parent
            or digest(bridge_artifact) != item["hash"]
        ):
            raise ValueError("exact confined sampled bridge artifact required")
    static_bridge_artifact = None
    if prediction_mode:
        if (
            a.static_bridge_proof.is_symlink()
            or a.static_bridge_review.is_symlink()
            or digest(a.static_bridge_proof) != a.static_bridge_proof_hash
            or digest(a.static_bridge_review) != a.static_bridge_review_hash
        ):
            raise ValueError("exact static bridge source and independent review required")
        static_source = json.loads(a.static_bridge_proof.read_text())
        static_review = json.loads(a.static_bridge_review.read_text())
        if (
            static_source["status"] != "CFTRACK_MEASURED_BRIDGE_H300_PAIRED_NOT_PROMOTED"
            or static_source["source_hash"] != a.source_hash
            or static_source["history_hash"] != a.history_hash
            or static_review["status"]
            != "ACTUAL_CFTRACK_BRIDGE_H300_NN_READBACK_EXACT_NOT_PROMOTED"
            or static_review["source_hash"] != a.static_bridge_proof_hash
            or len(static_source["records"]) != 2
            or static_source["records"][1]["actual_execution"]["arm"] != "measured_bridge"
            or static_source["records"][1]["actual_execution"]["absolute_h300_outcome"]["task_pass"]
            is not False
            or static_review["records"][1]["first_contact_reference_phase"] != 12
        ):
            raise ValueError("verified failed approach-phase static baseline required")
        item = static_source["records"][1]["actual_execution"]["artifact"]
        static_bridge_artifact = a.static_bridge_proof.parent / item["path"]
        if (
            static_bridge_artifact.is_symlink()
            or static_bridge_artifact.parent != a.static_bridge_proof.parent
            or digest(static_bridge_artifact) != item["hash"]
        ):
            raise ValueError("exact confined static baseline actual trajectory required")
    if (
        source["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or history["status"] != "MEASURED_CFTRACK_HISTORY_CANDIDATES_NOT_INSTALLED"
        or prior["status"] != "LIVE_MEASURED_CFTRACK_HISTORY_EXACT_NOT_INSTALLED"
        or history["source_hash"] != a.source_hash
        or prior["source_hash"] != a.source_hash
        or prior["history_hash"] != a.history_hash
        or prior["controls"] != 82
    ):
        raise ValueError("exact previously validated actual source and scratch history required")
    trace, histories = (
        a.source / source["artifact"]["path"],
        a.history / history["artifact"]["path"],
    )
    if (
        trace.is_symlink()
        or trace.parent != a.source
        or digest(trace) != source["artifact"]["hash"]
        or histories.is_symlink()
        or histories.parent != a.history
        or digest(histories) != history["artifact"]["hash"]
    ):
        raise ValueError("exact confined source and scratch artifacts required")
    pins = dict(prior["input_file_hashes"])
    pins.update(
        {
            str(complete): a.source_hash,
            str(hist): a.history_hash,
            str(live): a.live_review_hash,
            str(trace): source["artifact"]["hash"],
            str(histories): history["artifact"]["hash"],
        }
    )
    if forecast_enabled:
        pins[str(a.private_forecast_proof)] = a.private_forecast_proof_hash
    if execute_enabled:
        pins[str(a.execute_forecast_proof)] = a.execute_forecast_proof_hash
        pins.update({str(path): digest_value for path, digest_value in forecast_artifacts.values()})
    if bridge_mode:
        pins[str(a.measured_bridge)] = a.measured_bridge_hash
        pins[str(a.bridge_certificate)] = a.bridge_certificate_hash
        pins[str(bridge_artifact)] = bridge_source["records"][0]["artifact"]["hash"]
    if prediction_mode:
        pins[str(a.static_bridge_proof)] = a.static_bridge_proof_hash
        pins[str(a.static_bridge_review)] = a.static_bridge_review_hash
        pins[str(static_bridge_artifact)] = static_source["records"][1]["actual_execution"][
            "artifact"
        ]["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    if not (len(scenes) == len(models) == len(configs) == len(external_files) == 1):
        raise ValueError("one pinned scene, velocity codec and scratch policy required")
    external = external_files[0].parents[2]
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
    from rosclaw_soccer.rsi.cftrack_measured_bridge_reference import (
        condition_measured_bridge_reference,
    )
    from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
    from rosclaw_soccer.rsi.shared_cftrack_forecast import SharedCFtrackForecaster
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact original SIM body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(trace, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in f.files}
    with np.load(histories, allow_pickle=False) as f:
        expected = {k: f[k].copy() for k in f.files}
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            live_review_hash=a.live_review_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            entry_controls=[57, 57] if bridge_mode else [57, 82],
            maximum_velocity_calls=114 if bridge_mode else 278 if execute_enabled else 139,
            maximum_native_steps=7140 if bridge_mode else 6780 if execute_enabled else 1390,
            cftrack_neural_calls_max=600
            if bridge_mode
            else 400
            if (forecast_enabled or execute_enabled)
            else 0,
            private_forecast_native_step_budget=4000 if forecast_enabled else 0,
            private_forecast_horizon_controls=100 if forecast_enabled else 0,
            private_forecast_proof_hash=a.private_forecast_proof_hash,
            private_cold_vs_measured_history=forecast_enabled,
            execute_forecast_proof_hash=a.execute_forecast_proof_hash,
            actual_forecast_replay=execute_enabled,
            measured_bridge_mode=bridge_mode,
            predicted_ball_reference_mode=prediction_mode,
            ball_prediction_horizon_s=1.74 if prediction_mode else None,
            static_bridge_proof_hash=a.static_bridge_proof_hash,
            static_bridge_review_hash=a.static_bridge_review_hash,
            measured_bridge_hash=a.measured_bridge_hash,
            bridge_certificate_hash=a.bridge_certificate_hash,
            absolute_h300_includes_57_approach_controls=bridge_mode,
            no_body_or_ball_reset=True,
            software_pd_switch_explicit=True,
            cftrack_reference_not_qualified=True,
            task_success_not_assumed=True,
            hardware_authorized=False,
        ),
    )
    results, partial_raw = [], {}
    forecast_nn_calls = 0
    forecast_native_steps = 0
    counts = {"controls": 0, "native_steps": 0}

    def retain_partial():
        if (a.output / "complete.json").exists():
            return
        path = a.output / "partial.npz"
        np.savez_compressed(path, **{k: np.concatenate(v) for k, v in partial_raw.items() if v})
        write_once(
            a.output / "failed.json",
            dict(
                status="PARTIAL_NO_AUTOMATIC_RETRY",
                **counts,
                partial_hash=digest(path),
                hardware_authorized=False,
                goal_complete=False,
            ),
        )

    atexit.register(retain_partial)
    actual_cftrack_calls = 0
    specs = (
        [(57, "static_bridge"), (57, "predictive_bridge")]
        if prediction_mode
        else [(57, "parent"), (57, "measured_bridge")]
        if bridge_mode
        else [
            (57, "measured_history"),
            (57, "zero_history"),
            (82, "measured_history"),
            (82, "zero_history"),
        ]
        if execute_enabled
        else [(57, None), (82, None)]
    )
    for entry, actual_arm in specs:
        stream = ContinuousVelocityStream(
            models[0],
            configs[0],
            expected_model_hash=pins[str(models[0])],
            expected_config_hash=pins[str(configs[0])],
            horizon_controls=301,
        )
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = old["initial_qpos"], old["initial_qvel"]
        mujoco.mj_forward(model, data)
        effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
        pd = NativePDParameters(
            np.array(stream.params.stiffness), np.array(stream.params.damping), effort
        )
        world = SharedNativeMotorWorld(
            model, data, players={"a/": pd}, ball_body_name="soccer_ball"
        )
        history_collector = CFtrackLiveHistory(world, "a/")
        partial_raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
        for frame in range(entry):
            q, v = lo.canonical(data)
            command = np.array([0.0, 0.25 if 25 <= frame < 57 else 0.0, 0.0], np.float32)
            target = stream.propose(q, v, lo.angular_velocity_body(data), command, frame=frame)
            result = world.advance({"a/": target})
            for key in partial_raw:
                partial_raw[key].append(result[key])
            counts["controls"] += 1
            counts["native_steps"] += len(result["qpos"])
            if (
                any(
                    not np.array_equal(result[k], old[k][frame * 10 : (frame + 1) * 10])
                    for k in partial_raw
                )
                or not np.array_equal(target, old["joint_targets"][frame])
                or not np.array_equal(stream.last_observation, old["neural_obs"][frame])
                or not np.array_equal(stream.last_action, old["neural_actions"][frame])
            ):
                raise ValueError(
                    f"actual source trajectory mismatch at entry {entry} control {frame}"
                )
            if frame + 1 >= entry - 4:
                history_collector.capture()
            if world.failed:
                raise ValueError("live body failed before declared history installation")
        before = snapshot(data)
        cached_ball = data.xpos[lo.ball].copy(), data.xquat[lo.ball].copy()
        policy = OmniContact(
            StateAndCmd(29),
            PolicyOutput(29),
            onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
        )
        if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
            raise ValueError("canonical scratch FK order required")
        policy.task, policy.reference_source, policy.active_object_name = (
            "kickball",
            "CFgen",
            "ball",
        )
        policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
        policy.bbox_scale = policy.box_dims * 2
        policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
        kp, kd = policy.kps_lab[policy.lab2mj], policy.kds_lab[policy.lab2mj]
        switch = world.switch_pd_profile("a/", NativePDParameters(kp, kd, effort))
        player = SharedOmniContactPlayer(
            world, "a/", policy, expected_policy_type=OmniContact, goal=np.array([5.0, 0.0, 0.11])
        )
        receipt = install_owned_cftrack_history(player, history_collector)
        if (
            not np.array_equal(
                policy.obs_history_buffer[1:], expected[f"entry{entry}_observations"][:4]
            )
            or not np.array_equal(policy.action, expected[f"entry{entry}_observations"][4, 79:108])
            or np.any(policy.obs_history_buffer[0])
            or policy.counter_step != 0
            or player.commits != 0
            or player.pending is not None
        ):
            raise ValueError("installed history must match qualified actual scratch observation")
        bridge_receipt = None
        if bridge_mode and (actual_arm == "measured_bridge" or prediction_mode):
            with np.load(bridge_artifact, allow_pickle=False) as f:
                bridge_roots = f["candidate_root"].copy()
                bridge_joints = f["candidate_joint"].copy()
            bridge_receipt = condition_measured_bridge_reference(
                player,
                bridge_roots,
                bridge_joints,
                expected_model_hash=compiled_model_hash(model),
                ball_prediction_horizon_s=1.74 if actual_arm == "predictive_bridge" else None,
            )
        after = snapshot(data)
        if (
            any(not np.array_equal(before[k], after[k]) for k in before)
            or not np.array_equal(cached_ball[0], data.xpos[lo.ball])
            or not np.array_equal(cached_ball[1], data.xquat[lo.ball])
        ):
            raise ValueError("gain switch or history install modified physical state or ball cache")
        suffix = f"-{actual_arm}" if execute_enabled else ""
        trace_path = a.output / f"entry{entry}{suffix}.npz"
        np.savez_compressed(trace_path, **{k: np.concatenate(v) for k, v in partial_raw.items()})
        private_forecasts = []
        if forecast_enabled:
            original_hash = cftrack_state_hash(policy)
            for arm in ("measured_history", "zero_history"):
                branch_player = player
                if arm == "zero_history":
                    branch_player = copy.copy(player)
                    branch_player.policy = copy_cftrack_policy(policy)
                    branch_player.policy.obs_history_buffer.fill(0)
                    branch_player.policy.action.fill(0)
                    branch_player.measured_history_installed = False
                forecaster = SharedCFtrackForecaster(
                    branch_player,
                    expected_policy_type=OmniContact,
                    expected_model_hash=compiled_model_hash(model),
                    native_step_budget=1000,
                )
                future = forecaster.forecast(controls=100)
                counters = forecaster.counters()
                forecast_nn_calls += counters["future_nn_calls"]
                forecast_native_steps += counters["integrated_native_steps"]
                path = a.output / f"entry{entry}-{arm}.npz"
                np.savez_compressed(
                    path, **{k: v for k, v in future.items() if isinstance(v, np.ndarray)}
                )
                summary = {k: v for k, v in future.items() if not isinstance(v, np.ndarray)}
                summary.update(
                    arm=arm,
                    counters=counters,
                    minimum_root_height_m=(
                        float(np.min(future["qpos"][:, lo.rq + 2]))
                        if future["native_steps"]
                        else None
                    ),
                    ball_displacement_m=(
                        (
                            future["qpos"][-1, lo.bq : lo.bq + 3] - data.qpos[lo.bq : lo.bq + 3]
                        ).tolist()
                        if future["native_steps"]
                        else None
                    ),
                    first_target_change_rad=(
                        float(
                            np.max(
                                np.abs(future["joint_targets"][0] - old["joint_targets"][entry - 1])
                            )
                        )
                        if future["controls"]
                        else None
                    ),
                    first_torque_change_nm=(
                        float(np.max(np.abs(future["torque"][0] - old["torque"][entry * 10 - 1])))
                        if future["native_steps"]
                        else None
                    ),
                    artifact=dict(path=path.name, hash=digest(path)),
                )
                private_forecasts.append(summary)
                write_once(a.output / f"entry{entry}-{arm}.json", summary)
            if cftrack_state_hash(policy) != original_hash:
                raise ValueError("private futures changed the installed source policy")
            actual_after = snapshot(data)
            if any(not np.array_equal(before[k], actual_after[k]) for k in before):
                raise ValueError("private futures changed actual physical state")
        actual_execution = None
        if execute_enabled:
            if actual_arm == "zero_history":
                policy.obs_history_buffer.fill(0)
                policy.action.fill(0)
            forecast_arm = "measured_history" if bridge_mode else actual_arm
            prior_path, _ = forecast_artifacts[(entry, forecast_arm)]
            if prediction_mode and actual_arm == "static_bridge":
                prior_path = static_bridge_artifact
            with np.load(prior_path, allow_pickle=False) as f:
                predicted = {k: f[k].copy() for k in f.files}
            session = policy.ort_session
            saved_obs, saved_actions = [], []

            class Capture:
                def __init__(self, onnx_session, obs, actions):
                    self.session, self.obs, self.actions = onnx_session, obs, actions

                def run(self, outputs, inputs):
                    result = self.session.run(outputs, inputs)
                    self.obs.append(inputs["obs"].copy())
                    self.actions.append(result[0].copy())
                    return result

            policy.ort_session = Capture(session, saved_obs, saved_actions)
            real_rows, real_targets, contacts = [], [], []
            for step in range(300 if bridge_mode else 100):
                target, _ = player.propose()
                row = world.advance({"a/": target})
                for key in partial_raw:
                    partial_raw[key].append(row[key])
                counts["controls"] += 1
                counts["native_steps"] += len(row["qpos"])
                actual_cftrack_calls += 1
                real_rows.append(row)
                real_targets.append(target.copy())
                contacts.append(dict(world.last_ball_contact_bodies["a/"]))
                compare_parent = (prediction_mode and actual_arm == "static_bridge") or (
                    not prediction_mode
                    and (not bridge_mode or (actual_arm == "parent" and step < 100))
                )
                if compare_parent and (
                    not np.array_equal(target, predicted["joint_targets"][step])
                    or any(
                        not np.array_equal(row[k], predicted[k][step * 10 : (step + 1) * 10])
                        for k in partial_raw
                    )
                ):
                    raise ValueError(
                        "actual future differs from private forecast at "
                        f"{entry}/{actual_arm}/{step}"
                    )
                player.commit()
                if world.failed:
                    raise ValueError(f"actual future failed at {entry}/{actual_arm}/{step}")
            future_path = a.output / f"entry{entry}-{actual_arm}-actual.npz"
            np.savez_compressed(
                future_path,
                **{key: np.concatenate([row[key] for row in real_rows]) for key in partial_raw},
                joint_targets=np.stack(real_targets),
                neural_obs=np.stack(saved_obs),
                neural_actions=np.stack(saved_actions),
            )
            actual_execution = dict(
                arm=actual_arm,
                controls=len(real_rows),
                native_steps=sum(len(row["qpos"]) for row in real_rows),
                forecast_arrays_exact=(True if not bridge_mode else actual_arm == "parent"),
                forecast_exact_horizon_controls=(
                    100 if not bridge_mode or actual_arm == "parent" else None
                ),
                static_baseline_actual_exact=(prediction_mode and actual_arm == "static_bridge"),
                ball_contact_bodies_by_control=contacts,
                artifact=dict(path=future_path.name, hash=digest(future_path)),
            )
            write_once(a.output / f"entry{entry}-{actual_arm}-actual.json", actual_execution)
            if bridge_mode:
                from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch

                all_q = np.concatenate(partial_raw["qpos"])
                all_f = np.concatenate(partial_raw["ball_forces"])
                if all_q.shape[0] < 3000 or all_f.shape[0] < 3000:
                    raise ValueError("complete absolute H300 measurements required")
                force = np.zeros((300, 1, 6), dtype=np.float64)
                force[:, 0, :3] = all_f[:3000, 0].reshape(300, 10, 3).max(axis=1)
                q300 = all_q[:3000].reshape(300, 10, model.nq)
                outcome = label_first_touch(
                    dict(
                        force_n=force,
                        pelvis_z_per_substep_m=q300[:, :, lo.rq + 2][:, None],
                        ball_position_after_step_m=q300[:, -1, lo.bq : lo.bq + 3][:, None],
                    )
                ).to_dict()
                actual_execution["absolute_h300_outcome"] = outcome
                actual_execution["absolute_h300_includes_approach_controls"] = 57
                write_once(a.output / f"entry{entry}-{actual_arm}-absolute-h300.json", outcome)
        results.append(
            dict(
                entry_control=entry,
                source_prefix_exact=True,
                installed_rows_exact=True,
                physics_unchanged=True,
                pd_switch_physics_unchanged=switch["physics_unchanged"],
                history_receipt=receipt,
                bridge_reference_receipt=bridge_receipt,
                private_forecasts=private_forecasts,
                actual_execution=actual_execution,
                artifact=dict(path=trace_path.name, hash=digest(trace_path)),
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "CFTRACK_PREDICTIVE_BALL_H300_PAIRED_NOT_PROMOTED"
                if prediction_mode
                else "CFTRACK_MEASURED_BRIDGE_H300_PAIRED_NOT_PROMOTED"
                if bridge_mode
                else "CFTRACK_PRIVATE_FUTURES_REPLAYED_ACTUALLY_NOT_PROMOTED"
                if execute_enabled
                else "CFTRACK_MEASURED_HISTORY_PRIVATE_FUTURES_RECORDED_NOT_PROMOTED"
                if forecast_enabled
                else "LIVE_CFTRACK_HISTORY_INSTALLED_NOT_EXECUTED"
            ),
            source_head=a.expected_head,
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            live_review_hash=a.live_review_hash,
            input_file_hashes=pins,
            records=results,
            **counts,
            velocity_neural_calls=114 if bridge_mode else 278 if execute_enabled else 139,
            cftrack_neural_calls=actual_cftrack_calls if execute_enabled else forecast_nn_calls,
            private_forecast_native_steps=forecast_native_steps,
            task_reference_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    atexit.unregister(retain_partial)


if __name__ == "__main__":
    main()
