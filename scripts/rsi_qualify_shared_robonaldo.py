"""SIM-only RoboNaldo prior in the actual canonical team scene, no learning.

Four fixed ball starts, first one player then two separated neural players.
The latter is clock/ownership qualification, NOT passing or a football match.
All source cases, including early physical failures, remain in the report.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("asset-root", "teacher-root", "team-scene", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("teacher-hash", "team-scene-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only output required")
    teacher_path = a.teacher_root / "complete.json"
    if digest(teacher_path) != a.teacher_hash or digest(a.team_scene) != a.team_scene_hash:
        raise ValueError("exact teacher/scene commitments required")
    teacher = json.loads(teacher_path.read_text())
    if (
        len(teacher["records"]) != 4
        or not teacher["all_same_integrator_repeats_match"]
        or teacher["optimizer_updates"] != 0
    ):
        raise ValueError("closed four-case prior diagnostic required, including failures")
    pins = {str(path): digest(path) for path in a.asset_root.rglob("*") if path.is_file()}
    pins.update({str(teacher_path): a.teacher_hash, str(a.team_scene): a.team_scene_hash})
    for path, expected in teacher["input_file_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError("teacher's qualified source changed")
        pins[path] = expected
    source_pins(pins)
    check_pins(pins)
    import contextlib
    import io

    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo, qualify_g1_assets
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer

    class RecordedSession:
        def __init__(self, session):
            self.session = session
            self.calls = 0
            self.observation = np.zeros(547, np.float32)
            self.action = np.zeros(29, np.float32)

        def get_inputs(self):
            return self.session.get_inputs()

        def get_outputs(self):
            return self.session.get_outputs()

        def run(self, names, inputs):
            obs = inputs["obs"]
            if obs.shape != (1, 547) or obs.dtype != np.float32 or not np.isfinite(obs).all():
                raise ValueError("actual finite547 neural input required")
            out = self.session.run(names, inputs)
            if len(out) != 1 or out[0].shape != (1, 29) or not np.isfinite(out[0]).all():
                raise ValueError("actual finite29 neural output required")
            self.observation, self.action = obs[0].copy(), out[0][0].copy()
            self.calls += 1
            return out

    qualification = qualify_g1_assets(a.asset_root)
    qualification.require_eligible()
    state_type, output_type, policy_type, mapping = _load_robonaldo(a.asset_root)
    xml = a.team_scene.read_text()
    configs = [("single", ("a/",)), ("two-separated", ("a/", "b/"))]
    models = {
        name: mujoco.MjModel.from_xml_string(shared_native_scene(xml, prefixes=prefixes))
        for name, prefixes in configs
    }
    motion_path = a.asset_root / "policy/robonaldo/model/freekick_motion.npz"
    with np.load(motion_path) as motion:
        root_pos = motion["body_pos_w"][0, 0].copy()
        root_quat = motion["body_quat_w"][0, 0].copy()
        joints = motion["joint_pos"][0][mapping].copy()
    starts = [(x, y) for x in (0.97, 1.03) for y in (-0.03, 0.03)]
    # Compile/layout/policy qualification for BOTH declarations before stepping.
    policies = {}
    for name, prefixes in configs:
        m = models[name]
        policies[name] = {}
        for prefix in prefixes:
            NativeVelocityLayout(
                m, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=len(prefixes)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                policies[name][prefix] = policy_type(state_type(29), output_type(29))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="robonaldo.actual_team_scene.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            starts=starts,
            controls_per_world_ceiling=652,
            native_step_ceiling=52160,
            worlds=8,
            robot_b_offset_m=[0.0, 8.0, 0.0],
            compiled_scene_hashes={
                name: "sha256:"
                + hashlib.sha256(shared_native_scene(xml, prefixes=prefixes).encode()).hexdigest()
                for name, prefixes in configs
            },
            initial_ball_height_m=0.11,
            optimizer_updates=0,
            teacher_success_not_assumed=True,
            publicity_rights_not_established=True,
            hardware_authorized=False,
            promotion_authorized=False,
            test_is_not_team_game=True,
        ),
    )
    records = []
    count = 0
    try:
        for name, prefixes in configs:
            m = models[name]
            for index, (bx, by) in enumerate(starts):
                d = mujoco.MjData(m)
                params, local = {}, {}
                for number, prefix in enumerate(prefixes):
                    layout = NativeVelocityLayout(
                        m,
                        robot_prefix=prefix,
                        ball_body_name="soccer_ball",
                        player_count=len(prefixes),
                    )
                    d.qpos[layout.rq : layout.rq + 3] = root_pos + np.array(
                        [0.0, number * 8.0, 0.0]
                    )
                    d.qpos[layout.rq + 3 : layout.rq + 7] = root_quat
                    d.qpos[layout.qi] = joints
                    if index == 0:
                        policy = policies[name][prefix]
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            policy = policy_type(state_type(29), output_type(29))
                    local[prefix] = policy
                    policy.ort_session = RecordedSession(policy.ort_session)
                    params[prefix] = NativePDParameters(
                        np.array(policy.kps),
                        np.array(policy.kds),
                        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
                    )
                d.qpos[layout.bq : layout.bq + 3] = (bx, by, 0.11)
                d.qpos[layout.bq + 3 : layout.bq + 7] = (1.0, 0.0, 0.0, 0.0)
                mujoco.mj_forward(m, d)
                w = SharedNativeMotorWorld(m, d, players=params, ball_body_name="soccer_ball")
                players = {
                    prefix: SharedRoboNaldoPlayer(
                        w, prefix, local[prefix], goal=np.array([5.0, number * 8.0, 0.11])
                    )
                    for number, prefix in enumerate(prefixes)
                }
                raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
                targets = []
                neural_obs, neural_actions, neural_masks = [], [], []
                first_contact = {prefix: None for prefix in prefixes}
                end = None
                for frame in range(652):
                    previous_calls = {
                        prefix: local[prefix].ort_session.calls for prefix in prefixes
                    }
                    proposed = {prefix: players[prefix].propose() for prefix in prefixes}
                    proposals = {prefix: proposed[prefix][0] for prefix in prefixes}
                    obs, actions, masks = [], [], []
                    for prefix in prefixes:
                        session = local[prefix].ort_session
                        used = not proposed[prefix][1]["warmup_without_NN"]
                        if session.calls - previous_calls[prefix] != int(used):
                            raise ValueError(
                                "one actual NN inference per non-warmup control required"
                            )
                        obs.append(
                            session.observation.copy() if used else np.zeros(547, np.float32)
                        )
                        actions.append(session.action.copy() if used else np.zeros(29, np.float32))
                        masks.append(used)
                    result = w.advance(proposals)
                    count += 10
                    for prefix in prefixes:
                        players[prefix].commit()
                        if first_contact[prefix] is None and w.last_ball_contact_bodies[prefix]:
                            first_contact[prefix] = dict(
                                control=frame, bodies=w.last_ball_contact_bodies[prefix]
                            )
                    for key in raw:
                        raw[key].append(result[key])
                    targets.append(np.stack([proposals[prefix] for prefix in prefixes]))
                    neural_obs.append(np.stack(obs))
                    neural_actions.append(np.stack(actions))
                    neural_masks.append(masks)
                    end = result
                    if w.failed:
                        break
                arrays = {key: np.concatenate(values) for key, values in raw.items()}
                arrays["joint_targets"] = np.stack(targets)
                arrays["NN_observations"] = np.stack(neural_obs)
                arrays["NN_raw_actions"] = np.stack(neural_actions)
                arrays["NN_inferred_mask"] = np.array(neural_masks, dtype=bool)
                artifact = a.output / f"{name}-{index}.npz"
                np.savez_compressed(artifact, **arrays)
                record = dict(
                    group=name,
                    index=index,
                    ball_initial_xy=[bx, by],
                    native_steps=w.native_steps,
                    controls=w.control_steps,
                    body_failed=end["body_failed"],
                    reached_full_horizon=w.control_steps == 652,
                    first_physical_ball_contacts=first_contact,
                    final_ball_position=d.qpos[layout.bq : layout.bq + 3].tolist(),
                    final_root_positions={
                        prefix: d.qpos[body.rq : body.rq + 3].tolist()
                        for prefix, body in w.layouts.items()
                    },
                    proposals={prefix: player.proposals for prefix, player in players.items()},
                    commits={prefix: player.commits for prefix, player in players.items()},
                    actual_neural_inferences={
                        prefix: local[prefix].ort_session.calls for prefix in prefixes
                    },
                    artifact=dict(path=artifact.name, hash=digest(artifact)),
                )
                records.append(record)
                write_once(a.output / f"record-{name}-{index}.json", record)
                print(json.dumps(record), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ROBONALDO_SHARED_TEAM_SCENE_DIAGNOSED_NOT_PROMOTED",
                records=records,
                native_steps=count,
                input_file_hashes=pins,
                source_head=a.expected_head,
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
                completed=len(records),
                known_completed_control_steps=count,
                incomplete_control_native_steps_unknown=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
