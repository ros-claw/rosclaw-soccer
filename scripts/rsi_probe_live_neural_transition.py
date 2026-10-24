"""SIM live/live recovery blend on fixed team-scene starts, not new learning.

Keep both private neural histories causal through ACTUALLY executed targets.
No held action, body reset, ball rewrite, contact tuning or old-exam regrading.
Both targets and PD profiles interpolate over a predeclared20-control window.
"""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "own", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "own-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--contact-release", action="store_true")
    parser.add_argument("--aim-y", type=float, default=0.0)
    parser.add_argument("--single-player-only", action="store_true")
    parser.add_argument("--torque-teacher", action="store_true")
    parser.add_argument("--torque-candidate-root", type=Path)
    parser.add_argument("--torque-candidate-hash")
    parser.add_argument("--torque-intervention-fraction", type=float, default=1.0)
    parser.add_argument("--core-source-root", type=Path)
    parser.add_argument("--core-expected-head")
    parser.add_argument("--capture-torque-proposals", action="store_true")
    parser.add_argument("--torque-exploration-seed", type=int)
    parser.add_argument("--torque-exploration-scale", type=float)
    parser.add_argument("--exploration-plan", type=Path)
    parser.add_argument("--exploration-plan-hash")
    parser.add_argument("--identity-reference", type=Path)
    parser.add_argument("--identity-reference-hash")
    parser.add_argument("--identity-own", type=Path)
    parser.add_argument("--identity-own-hash")
    parser.add_argument("--clock-authority", action="store_true")
    parser.add_argument("--reference-rate", type=float, default=1.0)
    parser.add_argument("--clock-plan", type=Path)
    parser.add_argument("--clock-plan-hash")
    parser.add_argument("--clock-parent-root", type=Path)
    parser.add_argument("--clock-parent-hash")
    parser.add_argument("--clock-parent-own-root", type=Path)
    parser.add_argument("--clock-parent-own-hash")
    a = parser.parse_args()
    if (a.core_source_root is None) != (a.core_expected_head is None):
        parser.error("explicit Core source root and commit must be provided together")
    candidate_mode = a.torque_candidate_root is not None
    explore = a.torque_exploration_seed is not None
    if explore != (a.torque_exploration_scale is not None) or (
        explore
        and (
            not a.capture_torque_proposals
            or not 0 <= a.torque_exploration_seed < 2**32
            or not math.isfinite(a.torque_exploration_scale)
            or not 0 <= a.torque_exploration_scale <= 0.25
        )
    ):
        parser.error("explicit bounded seed/scale exploration requires full proposal capture")
    if (
        explore
        and (
            a.exploration_plan is None
            or a.exploration_plan_hash is None
            or a.torque_intervention_fraction != 1.0
        )
    ) or (not explore and (a.exploration_plan is not None or a.exploration_plan_hash is not None)):
        parser.error("full guarded exploration requires an exact predeclared family plan")
    if a.capture_torque_proposals and (not candidate_mode or a.core_source_root is None):
        parser.error(
            "proposal capture requires a learned candidate and explicit active Core source"
        )
    torque_mode = a.torque_teacher or candidate_mode
    if a.clock_authority:
        if (
            a.torque_teacher
            or explore
            or a.capture_torque_proposals
            or not a.single_player_only
            or not a.contact_release
            or a.aim_y != 0.0
            or a.core_source_root is None
            or not math.isfinite(a.reference_rate)
            or not 1 <= a.reference_rate <= 1.2
            or a.clock_plan is None
            or a.clock_plan_hash is None
        ):
            parser.error(
                "clock authority requires bounded isolated reference-only protocol and plan"
            )
    elif a.reference_rate != 1.0 or a.clock_plan is not None or a.clock_plan_hash is not None:
        parser.error("reference rate and plan require explicit clock authority")
    parent_options = (
        a.clock_parent_root,
        a.clock_parent_hash,
        a.clock_parent_own_root,
        a.clock_parent_own_hash,
    )
    if a.clock_authority and candidate_mode:
        if any(value is None for value in parent_options):
            parser.error("clock candidate requires exact independent fast parent pair")
    elif any(value is not None for value in parent_options):
        parser.error("fast clock parent pair belongs only to clock candidate protocol")
    if (
        candidate_mode != (a.torque_candidate_hash is not None)
        or (candidate_mode and a.torque_teacher)
        or a.torque_intervention_fraction
        not in ((0.0, 0.125, 0.25, 1.0) if a.clock_authority else (0.0, 1.0))
        or (not candidate_mode and a.torque_intervention_fraction != 1.0)
        or (candidate_mode and not a.single_player_only)
    ):
        parser.error("declared single-player candidate and fraction0/1 ablation required")
    if not math.isfinite(a.aim_y) or abs(a.aim_y) > 1.2:
        parser.error("finite qualified neural aim input in [-1.2,1.2] required")
    if (a.single_player_only or a.aim_y != 0.0) and not (
        a.single_player_only and a.contact_release
    ):
        parser.error(
            "new aim reachability protocol requires contact release and single-player-only"
        )
    if torque_mode or a.clock_authority:
        if (
            not a.contact_release
            or a.aim_y != 0.0
            or any(
                value is None
                for value in (
                    a.identity_reference,
                    a.identity_reference_hash,
                    a.identity_own,
                    a.identity_own_hash,
                )
            )
        ):
            parser.error(
                "teacher requires unchanged zero-aim contact release and exact identity source/own"
            )
    elif any(
        value is not None
        for value in (
            a.identity_reference,
            a.identity_reference_hash,
            a.identity_own,
            a.identity_own_hash,
        )
    ):
        parser.error(
            "identity reference options belong only to the declared torque teacher protocol"
        )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or digest(a.source / "complete.json") != a.source_hash
        or digest(a.own / "complete.json") != a.own_hash
    ):
        raise ValueError("unused output and exact closed source/own pair required")
    source = json.loads((a.source / "complete.json").read_text())
    own = json.loads((a.own / "complete.json").read_text())
    if (
        source["status"] != "ROBONALDO_NEURAL_VELOCITY_HANDOFF_DIAGNOSED_NOT_PROMOTED"
        or own["status"] != "ROBONALDO_VELOCITY_RECOVERY_MANUAL_CLOCK_REVIEWED_NOT_PROMOTED"
        or own["source_hash"] != a.source_hash
        or not own["all_exact"]
        or len(source["records"]) != 12
        or len(own["records"]) != 12
        or not all(row["healthy_full_horizon"] for row in source["records"])
    ):
        raise ValueError("complete healthy12, independently replayed predecessor required")
    pins = dict(source["input_file_hashes"])
    for key, value in own["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting predecessor source")
        pins[key] = value
    pins[str(a.source / "complete.json")], pins[str(a.own / "complete.json")] = (
        a.source_hash,
        a.own_hash,
    )
    identity_source = None
    if torque_mode or a.clock_authority:
        for root, expected in (
            (a.identity_reference, a.identity_reference_hash),
            (a.identity_own, a.identity_own_hash),
        ):
            if digest(root / "complete.json") != expected:
                raise ValueError("exact zero-intervention physical identity reference required")
            pins[str(root / "complete.json")] = expected
        identity_source = json.loads((a.identity_reference / "complete.json").read_text())
        identity_own = json.loads((a.identity_own / "complete.json").read_text())
        if (
            identity_source["status"] != "LIVE_NEURAL_TRANSITION_DIAGNOSED_NOT_PROMOTED"
            or len(identity_source["records"]) != 8
            or not identity_own["all_exact"]
            or identity_own["source_hash"] != a.identity_reference_hash
        ):
            raise ValueError("entire independently replayed eight-case identity parent required")
        for record in identity_source["records"]:
            path = a.identity_reference / record["artifact"]["path"]
            if (
                path.parent.resolve() != a.identity_reference.resolve()
                or digest(path) != record["artifact"]["hash"]
            ):
                raise ValueError("confined exact raw identity parent required")
            pins[str(path)] = record["artifact"]["hash"]
    trained = None
    clock_parent = None
    if a.clock_authority:
        if a.clock_plan.is_symlink() or digest(a.clock_plan) != a.clock_plan_hash:
            raise ValueError("exact predeclared reference-clock family plan required")
        plan = json.loads(a.clock_plan.read_text())
        if (
            plan["protocol"] != "robonaldo.reference_clock_authority.v1"
            or a.reference_rate not in plan["reference_rates"]
            or plan["source_commit"] != a.expected_head
            or plan["core_commit"] != a.core_expected_head
            or plan["identity_source_hash"] != a.identity_reference_hash
            or plan["identity_own_hash"] != a.identity_own_hash
            or plan["physical_control_hz"] != 50
            or plan["physical_native_hz"] != 500
        ):
            raise ValueError(
                "exact fixed clock family, physical clock and identity lineage required"
            )
        pins[str(a.clock_plan)] = a.clock_plan_hash
        if candidate_mode:
            if a.torque_intervention_fraction not in plan["intervention_fractions"]:
                raise ValueError("clock intervention must belong to exact declared family")
            for root, expected in (
                (a.clock_parent_root, a.clock_parent_hash),
                (a.clock_parent_own_root, a.clock_parent_own_hash),
            ):
                path = root / "complete.json"
                if path.is_symlink() or digest(path) != expected:
                    raise ValueError("exact completed fast parent source/own required")
                pins[str(path)] = expected
            clock_parent = json.loads((a.clock_parent_root / "complete.json").read_text())
            clock_own = json.loads((a.clock_parent_own_root / "complete.json").read_text())
            for report in (clock_parent, clock_own):
                for key, value in report["input_file_hashes"].items():
                    if key in pins and pins[key] != value:
                        raise ValueError("conflicting fast parent provenance")
                    pins[key] = value
            cp_decl_path = a.clock_parent_root / "declaration.json"
            if clock_own["input_file_hashes"].get(str(cp_decl_path)) != digest(cp_decl_path):
                raise ValueError("fast parent declaration must be bound by independent replay")
            pins[str(cp_decl_path)] = digest(cp_decl_path)
            cp_decl = json.loads(cp_decl_path.read_text())
            if (
                not clock_own["all_exact"]
                or clock_own["source_hash"] != a.clock_parent_hash
                or cp_decl["protocol"] != "robonaldo.reference_clock_authority.v1"
                or cp_decl["starts"] != plan["starts_m"]
                or cp_decl["reference_rate"] != a.reference_rate
                or len(clock_parent["records"]) != len(plan["starts_m"])
                or plan["clock_parent_hash"] != a.clock_parent_hash
                or plan["clock_parent_own_hash"] != a.clock_parent_own_hash
            ):
                raise ValueError(
                    "same declared complete fast parent and independent replay required"
                )
            for record in clock_parent["records"]:
                path = a.clock_parent_root / record["artifact"]["path"]
                if (
                    path.parent.resolve() != a.clock_parent_root.resolve()
                    or path.is_symlink()
                    or digest(path) != record["artifact"]["hash"]
                ):
                    raise ValueError("exact confined fast parent trajectory required")
                pins[str(path)] = record["artifact"]["hash"]
    if candidate_mode:
        training_path = a.torque_candidate_root / "complete.json"
        if digest(training_path) != a.torque_candidate_hash:
            raise ValueError("exact trained Core torque candidate required")
        trained = json.loads(training_path.read_text())
        if (
            trained["status"]
            not in {
                "CORE_TORQUE_ACTOR_BC_TRAINED_NOT_PHYSICALLY_QUALIFIED",
                "CORE_TORQUE_ACTOR_RL_TRAINED_NOT_PHYSICALLY_QUALIFIED",
            }
            or trained["actual_BC_actor_optimizer_updates"] <= 0
            or (
                trained["status"] == "CORE_TORQUE_ACTOR_RL_TRAINED_NOT_PHYSICALLY_QUALIFIED"
                and trained.get("actual_SAC_actor_optimizer_updates", 0) <= 0
            )
        ):
            raise ValueError("actual BC optimizer updates required, not an initialized model")
        for path, expected in trained["input_file_hashes"].items():
            if path in pins and pins[path] != expected:
                raise ValueError("conflicting candidate ancestry")
            pins[path] = expected
        pins[str(training_path)] = a.torque_candidate_hash
        if (
            clock_parent is not None
            and trained["parent_policy_hash"] != cp_decl["reference_controller_hash"]
        ):
            raise ValueError("candidate was not trained for this fast parent controller")
        policy_path = a.torque_candidate_root / trained["policy"]["path"]
        if (
            policy_path.parent.resolve() != a.torque_candidate_root.resolve()
            or policy_path.is_symlink()
            or digest(policy_path) != trained["policy"]["hash"]
        ):
            raise ValueError("confined exact learned Core torque binary required")
        pins[str(policy_path)] = trained["policy"]["hash"]
    if explore:
        if a.exploration_plan.is_symlink() or digest(a.exploration_plan) != a.exploration_plan_hash:
            raise ValueError("exact ordinary predeclared exploration family plan required")
        plan = json.loads(a.exploration_plan.read_text())
        zero = plan["zero_exploration"]
        admissible = (
            a.torque_exploration_seed == zero["seed"]
            and a.torque_exploration_scale == zero["scale"]
        ) or (
            a.torque_exploration_seed in plan["exploration_seeds"]
            and a.torque_exploration_scale == plan["exploration_scale"]
        )
        if (
            not admissible
            or plan["source_commit"] != a.expected_head
            or plan["core_commit"] != a.core_expected_head
            or plan["model_training_hash"] != a.torque_candidate_hash
            or plan["correlation_per_actual_head_call"] != 0.95
            or plan["starts_m"] != [[x, y] for x in (0.97, 1.03) for y in (-0.03, 0.03)]
        ):
            raise ValueError(
                "declared seed/scale, exact source/Core/model and unchanged starts required"
            )
        pins[str(a.exploration_plan)] = a.exploration_plan_hash
    active_core = None
    if a.core_source_root is not None:
        import importlib.util
        import subprocess

        active_core = a.core_source_root.resolve()
        head = subprocess.check_output(
            ["git", "-C", str(active_core), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(active_core), "status", "--porcelain"], text=True
        )
        spec = importlib.util.find_spec("rosclaw")
        expected_package = active_core / "src/rosclaw"
        if (
            head != a.core_expected_head
            or dirty
            or spec is None
            or spec.origin is None
            or Path(spec.origin).resolve().parent != expected_package
        ):
            raise ValueError("clean exact active Core commit must match Python import resolution")
        files = sorted(expected_package.rglob("*.py"))
        if not files or any(path.is_symlink() for path in files):
            raise ValueError("regular active Core Python sources required")
        for path in files:
            pins[str(path)] = digest(path)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.live_neural_transition import (
        ContactReleaseGate,
        blend_live_targets,
        quintic_weight,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
    from rosclaw_soccer.rsi.robonaldo_reference_clock import (
        declared_clock_starts,
        reference_policy_index,
        retimed_reference_velocity,
    )
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.training.directed_plane_crossing import first_directed_plane_crossing

    exploration_config = None
    if explore:
        from rosclaw.control.correlated_exploration import CorrelatedExplorationConfig

        exploration_config = CorrelatedExplorationConfig(
            seed=a.torque_exploration_seed, scale=a.torque_exploration_scale, correlation=0.95
        )

    roots = {Path(key).parents[2] for key in pins if key.endswith("/policy/robonaldo/FreeKick.py")}
    scenes = [Path(key) for key in pins if key.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(key) for key in pins if key.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(key) for key in pins if key.endswith("/velocity/v0/params/deploy.yaml")]
    if len(roots) != 1 or any(len(values) != 1 for values in (scenes, models, configs)):
        raise ValueError("unique qualified skills and exact team scene required")
    root = roots.pop()
    state_type, output_type, policy_type, mapping = _load_robonaldo(root)
    with np.load(root / "policy/robonaldo/model/freekick_motion.npz") as motion:
        pos, quat, joints = (
            motion["body_pos_w"][0, 0].copy(),
            motion["body_quat_w"][0, 0].copy(),
            motion["joint_pos"][0][mapping].copy(),
        )

    def velocity():
        return ContinuousVelocityStream(
            models[0],
            configs[0],
            expected_model_hash=pins[str(models[0])],
            expected_config_hash=pins[str(configs[0])],
            horizon_controls=652,
        )

    class RecordSession:
        def __init__(self, session):
            self.session, self.calls = session, 0
            self.obs, self.action = np.zeros(547, np.float32), np.zeros(29, np.float32)

        def run(self, names, inputs):
            obs = inputs["obs"]
            out = self.session.run(names, inputs)
            if (
                obs.shape != (1, 547)
                or obs.dtype != np.float32
                or len(out) != 1
                or out[0].shape != (1, 29)
                or not np.isfinite(obs).all()
                or not np.isfinite(out[0]).all()
            ):
                raise ValueError("actual finite547/29 neural tensors required")
            self.obs, self.action = obs[0].copy(), out[0][0].copy()
            self.calls += 1
            return out

    def observe(w, state):
        layout = w.layouts["a/"]
        q, v = layout.canonical(w.data)
        torso = w.model.body("a/torso_link").id
        state.q, state.dq, state.tau_est = (
            q[7:36].copy(),
            v[6:35].copy(),
            w.data.ctrl[layout.ai].copy(),
        )
        state.root_lin_vel_b, state.root_ang_vel_b = v[:3].copy(), v[3:6].copy()
        state.torso_pos_w, state.torso_quat_w = (
            w.data.xpos[torso].copy(),
            w.data.xquat[torso].copy(),
        )
        state.pelvis_pos_w, state.pelvis_quat_w = q[:3].copy(), q[3:7].copy()
        state.ball_pos_w, state.ball_vel_w, state.ball_valid = (
            w.data.xpos[layout.ball].copy(),
            v[35:38].copy(),
            True,
        )

    starts = (
        declared_clock_starts(plan)
        if a.clock_authority
        else [(x, y) for x in (0.97, 1.03) for y in (-0.03, 0.03)]
    )
    prepared = []
    arms = (
        (("single", ("a/",)),)
        if a.single_player_only
        else (("single", ("a/",)), ("heterogeneous", ("a/", "b/")))
    )
    for arm, prefixes in arms:
        model = mujoco.MjModel.from_xml_string(
            shared_native_scene(scenes[0].read_text(), prefixes=prefixes)
        )
        for index, (bx, by) in enumerate(starts):
            data = mujoco.MjData(model)
            stream, other = velocity(), velocity() if len(prefixes) == 2 else None
            state, output = state_type(29), output_type(29)
            with contextlib.redirect_stdout(io.StringIO()):
                kick = policy_type(state, output)
            if a.clock_authority:
                kick.motion_joint_vel = retimed_reference_velocity(
                    kick.motion_joint_vel, rate=a.reference_rate
                )
            recorder = RecordSession(kick.ort_session)
            kick.ort_session = recorder
            kick.target_pos_w = np.array([5.0, a.aim_y, 0.11], np.float32)
            cap = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
            oldpd = NativePDParameters(np.array(kick.kps), np.array(kick.kds), cap).owned_copy()
            newpd = NativePDParameters(
                np.array(stream.params.stiffness), np.array(stream.params.damping), cap
            ).owned_copy()
            params = {"a/": oldpd}
            for number, prefix in enumerate(prefixes):
                layout = NativeVelocityLayout(
                    model,
                    robot_prefix=prefix,
                    ball_body_name="soccer_ball",
                    player_count=len(prefixes),
                )
                data.qpos[layout.rq : layout.rq + 3] = pos + np.array([0.0, number * 8.0, 0.0])
                data.qpos[layout.rq + 3 : layout.rq + 7] = quat
                data.qpos[layout.qi] = (
                    joints if prefix == "a/" else np.array(stream.params.defaults)
                )
                if prefix == "b/":
                    params[prefix] = newpd
            data.qpos[layout.bq : layout.bq + 3], data.qpos[layout.bq + 3 : layout.bq + 7] = (
                (bx, by, 0.11),
                (1.0, 0.0, 0.0, 0.0),
            )
            mujoco.mj_forward(model, data)
            world = SharedNativeMotorWorld(
                model, data, players=params, ball_body_name="soccer_ball"
            )
            observe(world, state)
            with contextlib.redirect_stdout(io.StringIO()):
                kick.enter()
            prepared.append(
                (arm, index, world, state, output, kick, recorder, stream, other, oldpd, newpd)
            )
    torque_layers = {}
    torque_nn_counts = {}
    if torque_mode:
        from rosclaw.simforge.g1_neural_torque import (
            G1NeuralTorquePolicy,
            G1TeacherTorqueCollector,
            load_g1_neural_torque_artifact,
        )

        from rosclaw_soccer.rsi.shared_native_torque_layer import SharedNativeTorqueLayer

        for entry in prepared:
            world = entry[2]
            torque_layers[id(world)], torque_nn_counts[id(world)] = {}, {}
            for prefix in world.layouts:
                policy = G1TeacherTorqueCollector()
                if candidate_mode:
                    assert trained is not None
                    artifact = load_g1_neural_torque_artifact(
                        policy_path,
                        expected_hash=trained["policy"]["hash"],
                        expected_body_hash=trained["body_hash"],
                    )
                    policy = G1NeuralTorquePolicy(
                        artifact,
                        expected_body_hash=trained["body_hash"],
                        expected_parent_policy_hash=trained["parent_policy_hash"],
                        **({"exploration": exploration_config} if explore else {}),
                    )
                    counts = dict(gru_calls=0, head_calls=0)
                    advance_hidden, infer = policy._advance_hidden, policy._infer_normalized

                    def measured_gru(values, original=advance_hidden, counts=counts):
                        counts["gru_calls"] += 1
                        return original(values)

                    def measured_head(values, original=infer, counts=counts):
                        counts["head_calls"] += 1
                        return original(values)

                    policy._advance_hidden, policy._infer_normalized = measured_gru, measured_head
                    torque_nn_counts[id(world)][prefix] = counts
                torque_layers[id(world)][prefix] = SharedNativeTorqueLayer(
                    world,
                    prefix,
                    policy,
                    target_y_m=0.0,
                    target_z_m=0.11,
                    horizon_controls=652,
                    expected_parent_policy_hash=trained["parent_policy_hash"]
                    if trained is not None
                    else None,
                    intervention_fraction=a.torque_intervention_fraction,
                    capture_actor_proposals=a.capture_torque_proposals,
                )
    a.output.mkdir(parents=True)
    reference_controller_hash = None
    if a.clock_authority:
        from rosclaw_soccer.sim.contracts import hash_json

        reference_controller_hash = hash_json(
            dict(
                parent_identity_source=a.identity_reference_hash,
                algorithm="integer-rational-reference-index-and-owned-joint-velocity-scale.v1",
                rate=a.reference_rate,
                physical_control_hz=50,
                physical_native_hz=500,
                warmup_unchanged=True,
                original_control_history=True,
            )
        )
    write_once(
        a.output / "declaration.json",
        dict(
            protocol=(
                (
                    "robonaldo.reference_clock_learned_torque.v1"
                    if candidate_mode
                    else "robonaldo.reference_clock_authority.v1"
                )
                if a.clock_authority
                else "robonaldo.native_torque_exploration.v1"
                if explore
                else (
                    "robonaldo.native_guarded_proposal_trace.v1"
                    if a.capture_torque_proposals
                    else (
                        "robonaldo.native_learned_torque_trial.v1"
                        if candidate_mode
                        else (
                            "robonaldo.native_torque_teacher.v1"
                            if a.torque_teacher
                            else (
                                "robonaldo.neural_aim_reachability.v1"
                                if a.single_player_only
                                else (
                                    "robonaldo.contact_release_blend.v1"
                                    if a.contact_release
                                    else "robonaldo.live_neural_blend.v1"
                                )
                            )
                        )
                    )
                )
            ),
            source_head=a.expected_head,
            source_hash=a.source_hash,
            own_hash=a.own_hash,
            input_file_hashes=pins,
            starts=starts,
            worlds=len(starts) * len(arms),
            native_step_ceiling=6520 * len(starts) * len(arms),
            neural_aim_input_y_m=a.aim_y,
            reference_clock_authority=a.clock_authority,
            reference_rate=a.reference_rate,
            reference_controller_hash=reference_controller_hash,
            clock_parent_hash=a.clock_parent_hash,
            clock_parent_own_hash=a.clock_parent_own_hash,
            clock_plan_hash=a.clock_plan_hash,
            physical_control_hz=50,
            physical_native_hz=500,
            reference_velocity_scaled_in_owned_memory=a.clock_authority,
            parent_identity_reference_hash=a.identity_reference_hash,
            parent_controller_changed=a.clock_authority and a.reference_rate != 1.0,
            evaluation_target_world_m=[5.0, 0.0, 0.11],
            aim_input_is_not_evaluation_target=True,
            horizon_controls=652,
            start_control=None if a.contact_release else 285,
            duration_controls=20,
            outgoing_active_until_control_exclusive=None if a.contact_release else 305,
            retired_outgoing_zero_padding_not_training_data=True,
            contact_release=a.contact_release,
            release_condition=dict(clear_controls=3, distance_m_gt=0.35, separation_mps_gt=0.5),
            live_outgoing_proposals=True,
            live_incoming_proposals=True,
            actual_history_both_ABIs=True,
            same_effort_caps=True,
            torque_teacher=a.torque_teacher,
            capture_actor_proposals=a.capture_torque_proposals,
            exploration_config=__import__("dataclasses").asdict(exploration_config)
            if explore
            else None,
            exploration_plan_hash=a.exploration_plan_hash,
            torque_candidate_hash=a.torque_candidate_hash,
            torque_intervention_fraction=a.torque_intervention_fraction if torque_mode else None,
            teacher_motion_recovery_not_precision_qualified=a.torque_teacher,
            physics_retuned=False,
            optimizer_updates=0,
            hardware_authorized=False,
            publicity_rights_not_established=True,
            promotion_authorized=False,
            new_protocol_not_old_M0=True,
        ),
    )
    results, steps = [], 0
    try:
        for arm, index, w, state, output, kick, recorder, stream, other, oldpd, newpd in prepared:
            raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
            layers = torque_layers.get(id(w))
            if layers:
                raw["parent_torque"], raw["applied_torque_change"] = [], []
            controls = {
                key: []
                for key in (
                    "targets",
                    "blend_weight",
                    "kp",
                    "kd",
                    "kick_observation",
                    "kick_raw_action",
                    "kick_inferred_mask",
                    "velocity_observation",
                    "velocity_proposed_action",
                    "kick_proposed_target",
                    "velocity_proposed_target",
                    "release_distance_m",
                    "release_separation_mps",
                    "release_foot_contact",
                    "release_clear_controls",
                )
            }
            prefixes = tuple(w.layouts)
            if a.clock_authority:
                controls["reference_policy_index"], controls["reference_motion_frame"] = [], []
            foot = None
            gate = ContactReleaseGate()
            transition_start = None if a.contact_release else 285
            previous_weight = 0.0
            for frame in range(652):
                distance, speed, foot_contact = 0.0, 0.0, False
                if a.contact_release:
                    layout = w.layouts["a/"]
                    foot_body = w.model.body("a/right_ankle_roll_link").id
                    separation = w.data.xpos[layout.ball] - w.data.xpos[foot_body]
                    distance = float(np.linalg.norm(separation))
                    jac = np.zeros((3, w.model.nv))
                    mujoco.mj_jacBody(w.model, w.data, jac, None, foot_body)
                    relative_velocity = w.data.qvel[layout.bv : layout.bv + 3] - jac @ w.data.qvel
                    speed = float(np.dot(separation, relative_velocity) / max(1e-12, distance))
                    foot_contact = "a/right_ankle_roll_link" in w.last_ball_contact_bodies["a/"]
                    transition_start = gate.observe(
                        frame,
                        foot_contact=foot_contact,
                        distance_m=distance,
                        separation_mps=speed,
                    )
                weight = (
                    0.0
                    if transition_start is None
                    else quintic_weight(frame, start=transition_start, duration=20)
                )
                layout = w.layouts["a/"]
                q, v = layout.canonical(w.data)
                incoming = stream.propose(
                    q,
                    v,
                    layout.angular_velocity_body(w.data),
                    np.zeros(3, np.float32),
                    frame=stream.next_frame,
                )
                outgoing = np.zeros(29, np.float64)
                old_calls = recorder.calls
                ref_index, motion_frame = -1, -1
                if weight < 1.0:
                    if a.clock_authority:
                        ref_index = reference_policy_index(
                            frame, warmup=kick.WARMUP_STEPS, rate=a.reference_rate
                        )
                        kick.time_step = ref_index
                        if ref_index >= kick.WARMUP_STEPS:
                            motion_frame = kick._motion_frame_index(ref_index - kick.WARMUP_STEPS)
                    observe(w, state)
                    with contextlib.redirect_stdout(io.StringIO()):
                        kick.run()
                    outgoing = np.asarray(output.actions, np.float64).copy()
                    if not np.array_equal(output.kps, oldpd.stiffness) or not np.array_equal(
                        output.kds, oldpd.damping
                    ):
                        raise ValueError("original outgoing gains changed")
                target, pd = blend_live_targets(outgoing, incoming, oldpd, newpd, weight=weight)
                if weight > 0.0 and weight != previous_weight:
                    w.switch_pd_profile("a/", pd)
                previous_weight = weight
                proposals = {"a/": target}
                if other is not None:
                    other_layout = w.layouts["b/"]
                    oq, ov = other_layout.canonical(w.data)
                    proposals["b/"] = other.propose(
                        oq,
                        ov,
                        other_layout.angular_velocity_body(w.data),
                        np.zeros(3, np.float32),
                        frame=other.next_frame,
                    )
                inferred = recorder.calls > old_calls
                values = (
                    np.stack([proposals[prefix] for prefix in prefixes]),
                    weight,
                    pd.stiffness.copy(),
                    pd.damping.copy(),
                    recorder.obs.copy() if inferred else np.zeros(547, np.float32),
                    recorder.action.copy() if inferred else np.zeros(29, np.float32),
                    inferred,
                    stream.last_observation.copy(),
                    stream.last_action.copy(),
                    outgoing.copy(),
                    incoming.copy(),
                    distance,
                    speed,
                    foot_contact,
                    gate.clear_controls,
                )
                if a.clock_authority:
                    values += (ref_index, motion_frame)
                for key, value in zip(controls, values, strict=True):
                    controls[key].append(value)
                result = w.advance(proposals, torque_layers=layers)
                steps += 10
                if weight < 1.0:
                    stream.last_action = policy_action_from_hg_target(
                        target,
                        np.array(stream.params.action_offset),
                        np.array(stream.params.action_scale),
                        np.arange(29),
                    )
                if 0.0 < weight < 1.0:
                    kick.last_action_il = policy_action_from_hg_target(
                        target, np.array(kick.default_q_mj), np.array(kick.action_scale_mj), mapping
                    )
                if foot is None and any(
                    "ankle_roll_link" in body for body in w.last_ball_contact_bodies["a/"]
                ):
                    foot = frame
                for key in raw:
                    raw[key].append(result[key])
                if w.failed:
                    break
            arrays = {key: np.concatenate(value) for key, value in raw.items()}
            arrays.update({key: np.asarray(value) for key, value in controls.items()})
            primary_layout = w.layouts["a/"]
            ball_positions = arrays["qpos"][:, primary_layout.bq : primary_layout.bq + 3]
            crossing = first_directed_plane_crossing(
                ball_positions[:, :2][:, None, :],
                origin=np.array([5.0, 0.0]),
                forward=np.array([1.0, 0.0]),
            )
            plane = None
            if bool(crossing.crossed[0]):
                ci, fraction = int(crossing.segment_index[0]), float(crossing.fraction[0])
                point = ball_positions[ci] + fraction * (
                    ball_positions[ci + 1] - ball_positions[ci]
                )
                plane = dict(
                    y_m=float(point[1]),
                    z_m=float(point[2]),
                    error_to_fixed_target_m=float(np.linalg.norm(point[1:] - [0.0, 0.11])),
                )
            artifact = a.output / f"{arm}-{index}.npz"
            np.savez_compressed(artifact, **arrays)
            teacher_artifacts = {}
            full_identity = None
            if a.clock_authority and a.reference_rate == 1.0 and index < 4:
                compare_source = clock_parent if clock_parent is not None else identity_source
                compare_root = (
                    a.clock_parent_root if clock_parent is not None else a.identity_reference
                )
                identity = next(
                    row
                    for row in compare_source["records"]
                    if row["arm"] == arm and row["index"] == index
                )
                with np.load(compare_root / identity["artifact"]["path"]) as ref:
                    full_identity = {
                        key: bool(np.array_equal(arrays[key], ref[key])) for key in ref.files
                    }
                if not all(full_identity.values()):
                    raise ValueError("rate1 reference clock changed original parent trajectory")
            if layers:
                assert identity_source is not None
                compare_source = clock_parent if clock_parent is not None else identity_source
                compare_root = (
                    a.clock_parent_root if clock_parent is not None else a.identity_reference
                )
                identity = next(
                    row
                    for row in compare_source["records"]
                    if row["arm"] == arm and row["index"] == index
                )
                with np.load(compare_root / identity["artifact"]["path"]) as ref:
                    full_identity = {
                        key: bool(np.array_equal(arrays[key], ref[key])) for key in ref.files
                    }
                if (a.torque_teacher or a.torque_intervention_fraction == 0.0) and (
                    not all(full_identity.values()) or np.any(arrays["applied_torque_change"])
                ):
                    raise ValueError("pass-through teacher changed parent physics or controls")
                for prefix, layer in layers.items():
                    episode = layer.policy.episode()
                    artifact_kind = "teacher" if a.torque_teacher else "learned"
                    path = a.output / f"{artifact_kind}-{arm}-{index}-{prefix.rstrip('/')}.npz"
                    np.savez_compressed(
                        path,
                        observations=episode.observations,
                        actions=episode.actions,
                        parent_actions=episode.parent_actions,
                    )
                    teacher_artifacts[prefix] = dict(
                        path=path.name,
                        hash=digest(path),
                        body_hash=layer.body_hash,
                        observations=len(episode.observations),
                        commands=layer.commands,
                        applied=layer.applied,
                        policy_receipt=layer.policy.build_receipt().to_dict()
                        if candidate_mode
                        else None,
                        actual_NN_call_counts=torque_nn_counts[id(w)].get(prefix),
                    )
                    if layer.decision_trace is not None:
                        trace = layer.decision_trace
                        if trace.count != layer.applied:
                            raise ValueError(
                                "one completed proposal decision per actual native step required"
                            )
                        trace_path = (
                            a.output / f"guarded-proposals-{arm}-{index}-{prefix.rstrip('/')}.npz"
                        )
                        values = trace.arrays()
                        np.savez_compressed(trace_path, **values)
                        teacher_artifacts[prefix]["guarded_proposal_trace"] = dict(
                            path=trace_path.name,
                            hash=digest(trace_path),
                            context_hash=trace.context_hash,
                            steps=trace.count,
                            actual_actor_proposals=int(values["actor_proposal_valid"].sum()),
                            missing_proposals_are_not_zero_action_labels=True,
                            action_semantics="ACTOR_PROPOSAL_THROUGH_GUARD",
                        )
                        if explore:
                            snapshot = layer.policy.exploration_snapshot
                            expected_draws = (
                                int(values["actor_proposal_valid"].sum())
                                if a.torque_exploration_scale > 0
                                else 0
                            )
                            if snapshot["draw_steps"] != expected_draws:
                                raise ValueError(
                                    "exploration draws must match actual inferred proposals"
                                )
                            rng_path = (
                                a.output
                                / f"exploration-rng-{arm}-{index}-{prefix.rstrip('/')}.json"
                            )
                            write_once(rng_path, snapshot)
                            teacher_artifacts[prefix]["exploration_rng"] = dict(
                                path=rng_path.name,
                                hash=digest(rng_path),
                                behavior_hash=layer.policy.behavior_hash,
                                config_hash=layer.policy.exploration_config_hash,
                            )
            prefix_exact = None
            if arm == "single" and a.aim_y == 0.0 and not a.clock_authority:
                baseline = next(
                    row
                    for row in source["records"]
                    if row["arm"] == "actual-shadow" and row["index"] == index
                )
                with np.load(a.source / baseline["artifact"]["path"]) as ref:
                    prefix_end = (
                        min(285, transition_start if transition_start is not None else 285) * 10
                    )
                    prefix_exact = {
                        key: bool(np.array_equal(arrays[key][:prefix_end], ref[key][:prefix_end]))
                        for key in ("qpos", "qvel", "torque")
                    }
            start = transition_start if transition_start is not None else 285
            window = arrays["targets"][max(0, start - 1) : start + 22, 0]
            jump = float(np.max(np.abs(np.diff(window, axis=0)))) if len(window) > 1 else None
            row = dict(
                arm=arm,
                index=index,
                torque_teacher_artifacts=teacher_artifacts if a.torque_teacher else {},
                torque_execution_artifacts=teacher_artifacts,
                torque_intervention_fraction=a.torque_intervention_fraction
                if torque_mode
                else None,
                actual_torque_change_rms_nm=float(
                    np.sqrt(np.mean(np.square(arrays["applied_torque_change"])))
                )
                if layers
                else None,
                actual_torque_change_peak_nm=float(np.max(np.abs(arrays["applied_torque_change"])))
                if layers
                else None,
                zero_intervention_full_identity=full_identity,
                reference_rate=a.reference_rate,
                neural_aim_input_y_m=a.aim_y,
                actual_goal_plane=plane,
                illegal_body_ball_contact=bool(np.any(arrays["ball_forces"][:, 0, 2] > 1.0)),
                controls=w.control_steps,
                native_steps=w.native_steps,
                healthy_full_horizon=w.control_steps == 652 and not w.failed,
                body_failed=result["body_failed"],
                actual_foot_contact_control=foot,
                transition_start=transition_start,
                contact_release_gate=a.contact_release,
                prehandoff_prefix_exact=prefix_exact,
                max_target_step_in_transition_window_rad=jump,
                actual_kick_NN_calls=recorder.calls,
                actual_velocity_NN_calls=stream.neural_calls,
                other_NN_calls=other.neural_calls if other is not None else 0,
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            results.append(row)
            write_once(a.output / f"record-{arm}-{index}.json", row)
            print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="LIVE_NEURAL_TRANSITION_DIAGNOSED_NOT_PROMOTED",
                records=results,
                native_steps=steps,
                input_file_hashes=pins,
                source_head=a.expected_head,
                active_core_source_root=str(active_core) if active_core is not None else None,
                active_core_commit=a.core_expected_head,
                optimizer_updates=0,
                independent_manual_replay=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed=len(results),
                known_completed_control_native_steps=steps,
                incomplete_control_steps_unknown=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
