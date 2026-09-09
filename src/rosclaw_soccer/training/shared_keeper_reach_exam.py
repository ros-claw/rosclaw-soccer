"""Eight-body integration exam, NOT an autonomous football match.

A declared test launcher initializes one shot after settling. Subsequently
only robot actuators act on bodies/ball. Baseline and candidate share physics.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.keeper_frame import KeeperFrame
from rosclaw_soccer.providers.g1.mujoco_primitives import load_robonaldo
from rosclaw_soccer.providers.g1.shared_keeper_reach import (
    SharedKeeperReach,
    SharedKeeperReachConfig,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    _fill_locomotion_state,
    _make_player_controller,
    _normalized_locomotion_command,
    _pelvis_yaw,
    _project_joint_safe_torque,
    _rotate_z,
    _run_locomotion,
)
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.world.goalkeeper_glove_material import GoalkeeperGloveMaterial
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model


def _source_hash() -> str:
    root = Path(__file__).parents[1]
    return str(
        hash_json(
            {
                str(p.relative_to(root)): hash_bytes(p.read_bytes())
                for p in sorted(root.rglob("*.py"))
            }
        )
    )


_LOADED_SOURCE_HASH = _source_hash()
_MODEL_CACHE: dict[str, tuple[Any, str, Path, str]] = {}


def run_case(
    root: Path,
    output: Path,
    *,
    team: str,
    lateral: float,
    height: float,
    enabled: bool,
    config: SharedKeeperReachConfig,
    lateral_tracking: bool = False,
    glove_material: GoalkeeperGloveMaterial | None = None,
) -> dict[str, Any]:
    import mujoco

    if _source_hash() != _LOADED_SOURCE_HASH:
        raise RuntimeError("source changed since this exam worker imported; restart worker")
    if team not in {"red", "blue"} or not np.isfinite((lateral, height)).all():
        raise ValueError("finite bilateral keeper exam required")
    if not -0.65 <= lateral <= 0.65 or not 0.65 <= height <= 1.70:
        raise ValueError("shot outside declared standing keeper integration course")
    output.mkdir(parents=True, exist_ok=False)
    fixture = build_four_vs_four_fixture(root)
    cache_key = str(
        hash_json(
            (
                str(root.resolve()),
                fixture.fixture_hash,
                None if glove_material is None else asdict(glove_material),
            )
        )
    )
    if cache_key in _MODEL_CACHE:
        model, model_hash, cached_path, compressed_hash = _MODEL_CACHE[cache_key]
        if hash_bytes(cached_path.read_bytes()) != compressed_hash:
            raise RuntimeError("shared model artifact changed between episodes")
        os.link(cached_path, output / "world.mjb.gz")
    else:
        model = build_g1_multi_player_stadium_model(
            root, players=fixture.players, spec=fixture.goal, left_goal_plane_x_m=-1.5
        )
        model.opt.timestep = 0.002
        if glove_material is not None:
            for player in fixture.players:
                if player.goalkeeper_gloves:
                    glove_material.apply(model, prefix=player.body_prefix)
        model_file = output / "world.mjb"
        mujoco.mj_saveModel(model, str(model_file), None)
        raw_model = model_file.read_bytes()
        model_hash = str(hash_bytes(raw_model))
        compressed = gzip.compress(raw_model, compresslevel=1, mtime=0)
        if hash_bytes(gzip.decompress(compressed)) != model_hash:
            raise RuntimeError("compressed simulation model verification failed")
        cached_path = output / "world.mjb.gz"
        cached_path.write_bytes(compressed)
        _MODEL_CACHE[cache_key] = (model, model_hash, cached_path, str(hash_bytes(compressed)))
        model_file.unlink()  # Exact generated target, recoverable from verified gzip.
        del raw_model, compressed
    data = mujoco.MjData(model)
    state_type, output_type, _, _ = load_robonaldo(root)
    loco = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
    controllers = [
        _make_player_controller(
            model=model,
            data=data,
            spec=player,
            cell=next(c for c in fixture.cells if c.agent_id == player.agent_id),
            pelvis_height=0.793,
            state_type=state_type,
            output_type=output_type,
            loco_type=loco,
        )
        for player in fixture.players
    ]
    keeper = next(c for c in controllers if c.cell.agent_id == f"{team}.goalkeeper")
    frame = KeeperFrame(keeper.spec.origin_m[:2], keeper.spec.yaw_rad)
    reach = SharedKeeperReach(
        asset_root=root,
        goal=fixture.goal,
        frame=frame,
        prefix=keeper.spec.body_prefix,
        config=config,
    )
    keeper.keeper_reach = reach if enabled else None
    ball = model.body("ball").id
    ball_geom = model.geom("ball_geom").id
    bq = int(model.jnt_qposadr[model.joint("ball_free").id])
    bv = int(model.jnt_dofadr[model.joint("ball_free").id])
    data.qpos[bq : bq + 3] = (3, -6, 0.12)
    mujoco.mj_forward(model, data)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS) * 0.85
    poses, velocities, controls, times, contacts, activations = [], [], [], [], [], []
    ball_contacts = []
    muscle_observations = []
    minimum_pelvis = math.inf
    peak_tilt = 0.0
    goal_crossed = False
    hand_contact = False
    first_contact_velocity = None
    first_hand_time = None
    first_robot_contact_glove = None
    first_robot_contact_time = None
    first_robot_geoms: set[int] = set()
    gloves = keeper.left_glove_geoms | keeper.right_glove_geoms
    hands = gloves | {
        model.geom(keeper.spec.body_prefix + side + "_hand_collision").id
        for side in ("left", "right")
    }
    outward_speed = 0.0
    closest_incoming_glove_surface_m = 2.0
    finite_state = True
    maximum_joint_limit_excess_rad = 0.0
    maximum_actuator_limit_excess_nm = 0.0
    robot_geoms = set().union(*(c.robot_geoms for c in controllers))
    for tick in range(250):
        if tick == 50:
            flight = 3.0 / 6.5
            local = np.array((3.0, lateral, fixture.goal.ball_radius_m))
            canonical = np.array((4.52 - local[0], -local[1], local[2]))
            data.qpos[bq : bq + 3] = np.array((*frame.origin_xy, 0.0)) + frame.world_vector(
                canonical - (4.52, 0, 0)
            )
            data.qpos[bq + 3 : bq + 7] = (1, 0, 0, 0)
            data.qvel[bv : bv + 6] = 0
            data.qvel[bv : bv + 3] = frame.world_vector(
                np.array(
                    (6.5, 0, (height - fixture.goal.ball_radius_m + 4.905 * flight**2) / flight)
                )
            )
            mujoco.mj_forward(model, data)
        for c in controllers:
            _fill_locomotion_state(c, data, ball, bv)
            command = np.zeros(3)
            if c is keeper and enabled and lateral_tracking and reach.active:
                pelvis = frame.point(data.qpos[c.qpos_base : c.qpos_base + 3])
                speed = float(np.clip(2.0 * (-reach.last_intercept[1] - pelvis[1]), -0.5, 0.5))
                world = frame.world_vector(np.array((0.0, speed, 0.0)))
                command = _rotate_z(
                    world, -_pelvis_yaw(data.qpos[c.qpos_base + 3 : c.qpos_base + 7])
                )
            c.state.vel_cmd = _normalized_locomotion_command(c.policy, command)
            _run_locomotion(c, mirror=bool(command[1] < -1e-6), correct_mirrored_yaw=True)
        if enabled:
            keeper.output.actions = reach.step(model, data, keeper.output.actions)
            keeper.output.kps, keeper.output.kds = reach.impedance(
                keeper.output.kps, keeper.output.kds
            )
        activations.append((int(reach.active), reach.peak_residual_rad))
        if reach.last_muscle_observation is not None:
            muscle_observations.append(reach.last_muscle_observation.copy())
        for _ in range(10):
            before_ball_velocity = data.qvel[bv : bv + 3].copy()
            for c in controllers:
                q, dq = data.qpos[c.joint_qpos], data.qvel[c.joint_qvel]
                torque = c.output.kps * (c.output.actions - q) - c.output.kds * dq
                if c.keeper_reach is not None:
                    torque += c.keeper_reach.torque_nm
                torque = _project_joint_safe_torque(
                    joint_position=q,
                    joint_velocity=dq,
                    commanded_torque=torque,
                    joint_ranges=model.jnt_range[c.joint_ids],
                    limited=model.jnt_limited[c.joint_ids].astype(bool),
                )
                data.ctrl[c.actuators] = np.clip(torque, -limits, limits)
            mujoco.mj_step(model, data)
            finite_state = finite_state and all(
                np.isfinite(v).all() for v in (data.qpos, data.qvel, data.ctrl)
            )
            if not finite_state:
                break
            q = data.qpos[keeper.joint_qpos]
            ranges = model.jnt_range[keeper.joint_ids]
            limited = model.jnt_limited[keeper.joint_ids].astype(bool)
            maximum_joint_limit_excess_rad = max(
                maximum_joint_limit_excess_rad,
                float(
                    np.max(
                        np.maximum(ranges[limited, 0] - q[limited], q[limited] - ranges[limited, 1])
                    )
                ),
            )
            maximum_actuator_limit_excess_nm = max(
                maximum_actuator_limit_excess_nm,
                float(np.max(abs(data.ctrl[keeper.actuators]) - np.asarray(G1_HARD_TORQUE_LIMITS))),
            )
            for i in range(data.ncon):
                contact = data.contact[i]
                if ball_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == ball_geom else contact.geom1
                ball_contacts.append((float(data.time), int(other), float(contact.dist)))
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                if float(force[0]) <= 1e-6:
                    continue
                if tick >= 50 and other in robot_geoms:
                    if first_robot_contact_time is None:
                        first_robot_contact_time = float(data.time)
                    if data.time == first_robot_contact_time:
                        first_robot_geoms.add(int(other))
                        first_robot_contact_glove = first_robot_geoms <= hands and bool(
                            first_robot_geoms & gloves
                        )
                if other in keeper.left_glove_geoms | keeper.right_glove_geoms:
                    if enabled and np.linalg.norm(force[:3]) > 1e-6:
                        reach.notify_glove_contact(float(data.time))
                    contacts.append((float(data.time), int(other), float(contact.dist), *force))
                    if not hand_contact:
                        first_contact_velocity = before_ball_velocity.tolist()
                        first_hand_time = float(data.time)
                    hand_contact = True
            if first_hand_time is not None and 0.03 <= data.time - first_hand_time <= 0.20:
                outward_speed = max(
                    outward_speed, -float((frame.rotation @ data.qvel[bv : bv + 3])[0])
                )
            if tick >= 50:
                p = frame.point(data.qpos[bq : bq + 3])
                if 3.9 <= p[0] <= 4.75 and first_robot_contact_time is None:
                    closest_incoming_glove_surface_m = min(
                        closest_incoming_glove_surface_m,
                        *(
                            float(mujoco.mj_geomDistance(model, data, ball_geom, g, 2.0, None))
                            for g in gloves
                        ),
                    )
                if (
                    p[0] > 5.07
                    and abs(p[1]) < fixture.goal.width_m / 2
                    and p[2] < fixture.goal.height_m
                ):
                    goal_crossed = True
                minimum_pelvis = min(minimum_pelvis, float(data.qpos[keeper.qpos_base + 2]))
                peak_tilt = max(
                    peak_tilt,
                    math.acos(
                        float(np.clip(data.xmat[keeper.pelvis_body].reshape(3, 3)[2, 2], -1, 1))
                    ),
                )
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        controls.append(data.ctrl.copy())
        times.append(data.time)
        if not finite_state:
            break
    np.savez_compressed(
        output / "trajectory.npz",
        qpos=poses,
        qvel=velocities,
        ctrl=controls,
        time=times,
        glove_contacts=contacts,
        activation=activations,
        ball_contacts=ball_contacts,
        muscle_observations=np.asarray(muscle_observations).reshape(-1, 65),
    )
    physical_safe = bool(
        finite_state
        and maximum_joint_limit_excess_rad <= 1e-5
        and maximum_actuator_limit_excess_nm <= 1e-5
        and minimum_pelvis > 0.55
        and peak_tilt < 0.8
    )
    result = dict(
        schema_version="s229.shared_keeper_reach_exam.v2",
        activation_ceiling="SIM_ONLY",
        autonomous_match=False,
        test_launcher=True,
        candidate_promoted=False,
        team=team,
        lateral=lateral,
        height=height,
        enabled=enabled,
        config=asdict(config),
        hand_contact=hand_contact,
        goal_crossed=goal_crossed,
        minimum_pelvis_m=minimum_pelvis,
        peak_tilt_rad=peak_tilt,
        finite_state=finite_state,
        maximum_joint_limit_excess_rad=maximum_joint_limit_excess_rad,
        maximum_actuator_limit_excess_nm=maximum_actuator_limit_excess_nm,
        physical_safe=physical_safe,
        first_contact_velocity=first_contact_velocity,
        completed_hand_save=hand_contact
        and physical_safe
        and first_robot_contact_glove is True
        and not goal_crossed
        and minimum_pelvis > 0.55
        and peak_tilt < 0.8,
        lateral_tracking=lateral_tracking,
        glove_material=None if glove_material is None else asdict(glove_material),
        stable_save=hand_contact
        and physical_safe
        and first_robot_contact_glove is True
        and outward_speed > 1.0
        and not goal_crossed
        and minimum_pelvis > 0.55
        and peak_tilt < 0.8,
        first_robot_contact_glove=first_robot_contact_glove,
        first_robot_contact_geoms=sorted(first_robot_geoms),
        outward_speed_mps=outward_speed,
        closest_incoming_glove_surface_m=closest_incoming_glove_surface_m,
        active_frames=sum(a[0] for a in activations),
        peak_residual_rad=reach.peak_residual_rad,
        fixture_hash=fixture.fixture_hash,
        model_hash=None if reach.gmt_contract is None else reach.gmt_contract.checkpoint_hash,
        imitation_hash=None if reach.gmt is None else reach.gmt.skill.skill_hash,
        muscle_policy_hash=None if reach.muscle is None else reach.muscle.policy_hash,
        keeper_policy_hash=reach.policy_hash,
        simulation_model_hash=model_hash,
        mujoco_version=mujoco.__version__,
        source_integrity_verified=_source_hash() == _LOADED_SOURCE_HASH,
        trajectory_hash=hash_bytes((output / "trajectory.npz").read_bytes()),
        implementation_hash=_LOADED_SOURCE_HASH,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--team", choices=("red", "blue"), default="red")
    parser.add_argument("--height", type=float, default=1.3)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--gmt-model", type=str)
    parser.add_argument("--gmt-skill", type=str)
    args = parser.parse_args()
    print(
        json.dumps(
            run_case(
                args.asset_root,
                args.output,
                team=args.team,
                lateral=args.lateral,
                height=args.height,
                enabled=args.enabled,
                config=SharedKeeperReachConfig(
                    gmt_model_path=args.gmt_model, gmt_skill_path=args.gmt_skill
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
