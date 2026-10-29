"""SIM-only eight-player airborne classroom baseline with exact fresh replay."""

import argparse
import importlib
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean, source_pins


def exact_arrays(a, b):
    if set(a) != set(b):
        return False
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not np.array_equal(x, y, equal_nan=x.dtype.kind in "fc"):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("asset-root", "policy", "protocol", "core-source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("policy-hash", "protocol-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    p.add_argument("--standard-ball", action="store_true")
    p.add_argument("--focal-nearest", action="store_true")
    p.add_argument("--focal-nearest-validation", action="store_true")
    p.add_argument("--focal-exploration-seed-base", type=int)
    p.add_argument("--capture-residual-target-authority", action="store_true")
    a = p.parse_args()
    if a.focal_nearest and not a.standard_ball:
        raise ValueError("focal-nearest study requires the declared standard ball")
    if a.focal_nearest_validation and (
        not a.focal_nearest or a.focal_exploration_seed_base is not None
    ):
        raise ValueError("disjoint validation requires deterministic focal-nearest mode")
    if a.focal_exploration_seed_base is not None and (
        not a.focal_nearest or not 0 <= a.focal_exploration_seed_base <= 2**32 - 8
    ):
        raise ValueError("seeded exploration requires focal courses and eight valid seeds")
    if a.capture_residual_target_authority and a.focal_exploration_seed_base is None:
        raise ValueError("authority capture requires the declared focal exploration pilot")
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external evidence path required")
    if (
        a.policy.is_symlink()
        or a.protocol.is_symlink()
        or digest(a.policy) != a.policy_hash
        or digest(a.protocol) != a.protocol_hash
    ):
        raise ValueError("exact policy and protocol required")
    assets = a.asset_root.resolve()
    core = a.core_source.resolve() / "rosclaw"
    if not assets.is_dir() or not (core / "__init__.py").is_file():
        raise ValueError("explicit simulator assets and core source required")
    asset_files = [path for path in assets.rglob("*") if path.is_file()]
    if not asset_files or any(path.is_symlink() for path in asset_files):
        raise ValueError("ordinary simulator asset files required")
    pins = {str(path): digest(path) for path in asset_files}
    pins.update({str(path): digest(path) for path in core.rglob("*.py")})
    pins.update({str(a.policy): a.policy_hash, str(a.protocol): a.protocol_hash})
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import rosclaw

    from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
    from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
    from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
    from rosclaw_soccer.providers.g1.mujoco_primitives import load_robonaldo
    from rosclaw_soccer.sim.physical_checkpoint import PhysicalCheckpoint, compiled_model_hash
    from rosclaw_soccer.skills.team.independent_team_world import _make_player_controller
    from rosclaw_soccer.training.airborne_receiving_courses import (
        airborne_receiving_ball_launch,
        airborne_receiving_course_bank,
        disjoint_focal_airborne_validation_bank,
        focal_nearest_airborne_course_bank,
    )
    from rosclaw_soccer.training.continuous_competitive_match_growth import _write_trajectory
    from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
    from rosclaw_soccer.training.near_ball_residual_ppo import _validate_on_policy
    from rosclaw_soccer.training.receiving_classroom import (
        coached_receiving_cells,
        r1_contact_tap_receiving_configuration,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
    from rosclaw_soccer.training.role_receiving_courses import ROSTER
    from rosclaw_soccer.training.team_capture_curriculum import team_capture_trials
    from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

    if Path(rosclaw.__file__).resolve() != core / "__init__.py":
        raise ValueError("imported core differs from pinned source")
    courses = (
        disjoint_focal_airborne_validation_bank()
        if a.focal_nearest_validation
        else focal_nearest_airborne_course_bank()
        if a.focal_nearest
        else airborne_receiving_course_bank()
    )
    bank_hash = preflight_receiving_courses(courses)
    fixture = collection_fixture(assets, keeper_preview=True)
    if a.standard_ball:
        fixture = replace(
            fixture, goal=replace(fixture.goal, ball_radius_m=0.110, ball_mass_kg=0.430)
        )
    world, _ = r1_contact_tap_receiving_configuration()
    qualification = qualify_g1_assets(assets)
    qualification.require_eligible()
    model = build_g1_multi_player_stadium_model(
        assets,
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=world.left_goal_plane_x_m if world.bilateral_goals else None,
    )
    model.opt.timestep = 0.002
    if world.glove_material is not None:
        for player in fixture.players:
            if player.goalkeeper_gloves:
                world.glove_material.apply(model, prefix=player.body_prefix)
    ball_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
    dimensions = inspect_native_ball_dimensions(model, geom_id=ball_geom)
    if a.standard_ball and (
        not math.isclose(dimensions.radius_m, 0.110, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(dimensions.body_mass_kg, 0.430, rel_tol=0.0, abs_tol=1e-12)
        or not dimensions.size_and_mass_in_ifab_range
    ):
        raise ValueError("compiled standard-sized sphere mismatch")
    ball_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    if ball_joint < 0:
        raise ValueError("compiled free ball joint required")
    ball_q, ball_v = int(model.jnt_qposadr[ball_joint]), int(model.jnt_dofadr[ball_joint])
    state_type, output_type, _, _ = load_robonaldo(qualification.asset_root)
    loco_type = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
    with np.load(qualification.asset_root / "policy/robonaldo/model/freekick_motion.npz") as motion:
        pelvis_height = float(np.asarray(motion["body_pos_w"])[0, 0, 2])
    template = mujoco.MjData(model)
    cells = coached_receiving_cells(fixture.cells, focal_agent_id=courses[0].agent_id)
    by_cell = {cell.agent_id: cell for cell in cells}
    by_player = {player.agent_id: player for player in fixture.players}
    for agent in sorted(fixture.roster.agents, key=lambda item: item.agent_id):
        _make_player_controller(
            model=model,
            data=template,
            spec=by_player[agent.agent_id],
            cell=by_cell[agent.agent_id],
            pelvis_height=pelvis_height,
            state_type=state_type,
            output_type=output_type,
            loco_type=loco_type,
        )
    # Whole-bank physical startup preflight before running any episode.
    starts = {}
    focal_nearest_margins = {}
    for course in courses:
        data = mujoco.MjData(model)
        data.qpos[:] = template.qpos
        position, velocity = airborne_receiving_ball_launch(
            course, origin=by_player[course.agent_id].origin_m, radius_m=dimensions.radius_m
        )
        if a.focal_nearest:
            distances = sorted(
                (
                    math.hypot(player.origin_m[0] - position[0], player.origin_m[1] - position[1]),
                    player.agent_id,
                )
                for player in fixture.players
            )
            margin = distances[1][0] - distances[0][0]
            if distances[0][1] != course.agent_id or margin < 0.30:
                raise ValueError("focal-nearest launch margin rejected for whole bank")
            focal_nearest_margins[course.seed] = margin
        data.qpos[ball_q : ball_q + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
        data.qvel[ball_v : ball_v + 3] = velocity
        mujoco.mj_forward(model, data)
        ball_contacts = [c for c in data.contact[: data.ncon] if ball_geom in c.geom]
        if ball_contacts:
            raise ValueError(f"initial ball contact in course {course.seed}; reject whole bank")
        starts[course.seed] = PhysicalCheckpoint.capture(model, data).state_hash
    model_hash = compiled_model_hash(model)
    indices = list(range(a.shard, len(courses), 4))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            protocol_hash=a.protocol_hash,
            policy_hash=a.policy_hash,
            shard=a.shard,
            course_bank_hash=bank_hash,
            courses=[asdict(c) for c in courses],
            initial_integration_hashes=starts,
            ball_dimensions=asdict(dimensions),
            compiled_model_hash=model_hash,
            fixture_hash=fixture.fixture_hash,
            focal_nearest_margins_m=focal_nearest_margins,
            standard_ball=a.standard_ball,
            focal_nearest=a.focal_nearest,
            focal_nearest_validation=a.focal_nearest_validation,
            focal_exploration_seed_base=a.focal_exploration_seed_base,
            capture_residual_target_authority=a.capture_residual_target_authority,
            input_file_hashes=pins,
            coach_assigned_receive=True,
            classroom_profile="R1_CONTACT_TAP",
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    records = []
    try:
        for i in indices:
            course = courses[i]
            original = None
            for repeat in range(2):
                result, trace = simulate_r0_receiving_course(
                    asset_root=assets,
                    reference_policy_path=a.policy,
                    course=course,
                    scenario_id=f"s199.airborne-receive-{course.seed}",
                    configuration_profile="R1_CONTACT_TAP",
                    airborne_standard_ball=a.standard_ball,
                    research_focal_exploration_seed=(
                        None
                        if a.focal_exploration_seed_base is None
                        else a.focal_exploration_seed_base + i
                    ),
                    capture_residual_target_authority=a.capture_residual_target_authority,
                )
                if str(trace["initial_integration_hash"][0]) != starts[course.seed]:
                    raise ValueError("actual initial state differs from whole-bank preflight")
                if str(trace["initial_physics_hash"][0]) != model_hash:
                    raise ValueError("actual compiled model differs from physical preflight")
                if original is not None and not exact_arrays(trace, original):
                    raise ValueError("full native reexecution differs")
                if original is None:
                    original = {k: np.asarray(v).copy() for k, v in trace.items()}
                frames = len(trace["time"])
                aligned = {k: v for k, v in trace.items() if np.ndim(v) and len(v) == frames}
                exploration_diagnostics = None
                if a.focal_exploration_seed_base is not None:
                    _validate_on_policy(NearBallResidualPolicy.load(a.policy), aligned)
                    mask = aligned["residual_exploration_mask"]
                    if (
                        mask.shape != (frames, len(ROSTER))
                        or not np.all(mask[:, ROSTER.index(course.agent_id)])
                        or np.any(np.delete(mask, ROSTER.index(course.agent_id), axis=1))
                    ):
                        raise ValueError("exploration was not scoped to the focal player")
                    focal_index = ROSTER.index(course.agent_id)
                    exploration_diagnostics = {
                        "seed": a.focal_exploration_seed_base + i,
                        "focal_active_frames": int(
                            np.count_nonzero(aligned["residual_active"][:, focal_index])
                        ),
                        "focal_nonzero_applied_frames": int(
                            np.count_nonzero(
                                np.any(aligned["residual_applied"][:, focal_index] != 0, axis=1)
                            )
                        ),
                    }
                    if a.capture_residual_target_authority:
                        injected = aligned["residual_pd_target_injected_substeps"]
                        overridden = aligned["residual_pd_target_override_substeps"]
                        if (
                            injected.shape != mask.shape
                            or overridden.shape != mask.shape
                            or injected.dtype.kind not in "iu"
                            or overridden.dtype.kind not in "iu"
                            or np.any(injected < 0)
                            or np.any(overridden < 0)
                            or np.any(injected + overridden > 10)
                        ):
                            raise ValueError("invalid residual PD target authority telemetry")
                        exploration_diagnostics.update(
                            focal_pd_target_injected_substeps=int(injected[:, focal_index].sum()),
                            focal_pd_target_override_substeps=int(overridden[:, focal_index].sum()),
                        )
                _, trials = team_capture_trials(aligned, agent_ids=ROSTER)
                artifact = _write_trajectory(a.output / f"course-{i:02d}-{repeat}.npz", trace)
                records.append(
                    dict(
                        index=i,
                        seed=course.seed,
                        repeat=repeat,
                        result=result.to_dict(),
                        receiving_trials=trials,
                        exploration_diagnostics=exploration_diagnostics,
                        artifact=artifact,
                    )
                )
        clean(a.expected_head)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="AIRBORNE_RECEIVING_BASELINE_SIM_ONLY",
                source_head=a.expected_head,
                protocol_hash=a.protocol_hash,
                policy_hash=a.policy_hash,
                course_bank_hash=bank_hash,
                shard=a.shard,
                records=records,
                coach_assigned_receive=True,
                classroom_profile="R1_CONTACT_TAP",
                standard_ball=a.standard_ball,
                focal_nearest=a.focal_nearest,
                focal_nearest_validation=a.focal_nearest_validation,
                focal_exploration_seed_base=a.focal_exploration_seed_base,
                capture_residual_target_authority=a.capture_residual_target_authority,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error_type=type(error).__name__, error=str(error), completed_records=records),
        )
        raise


if __name__ == "__main__":
    main()
