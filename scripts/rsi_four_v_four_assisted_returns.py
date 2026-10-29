"""Preregistered 4v4 dead-ball training-assist probe; never a match exam."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest
from scripts.rsi_native_context_recovery_common import check_pins
from scripts.rsi_native_fresh_common import clean, source_pins


def ball_fixture(fixture, *, standard_ball: bool):
    """Opt in to jointly changed ball size and mass; leave legacy default intact."""
    if type(standard_ball) is not bool:
        raise ValueError("explicit boolean standard-ball selector required")
    if not standard_ball:
        return fixture
    return replace(fixture, goal=replace(fixture.goal, ball_radius_m=0.110, ball_mass_kg=0.430))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--policy-hash", required=True)
    p.add_argument("--returned-policy", type=Path)
    p.add_argument("--returned-policy-hash")
    p.add_argument("--core-source", type=Path)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exploration-seed", type=int, choices=(24001, 24002))
    p.add_argument("--only-explore-after-return", action="store_true")
    p.add_argument("--standard-ball", action="store_true")
    a = p.parse_args()
    if bool(a.returned_policy) != bool(a.returned_policy_hash):
        raise ValueError("returned candidate and exact digest required together")
    if a.returned_policy and a.exploration_seed is not None:
        raise ValueError("gated candidate comparison must be deterministic")
    if a.only_explore_after_return and a.exploration_seed is None:
        raise ValueError("returned-live exploration requires an explicit seed")
    if a.standard_ball and a.core_source is None:
        raise ValueError("standard-ball experiment requires pinned ROSClaw core source")
    clean(a.expected_head)
    if a.output.resolve().is_relative_to(Path.cwd()) or a.output.resolve().is_relative_to(
        a.asset_root.resolve()
    ):
        raise ValueError("external output required")
    if (
        file_digest(a.protocol) != a.protocol_hash
        or file_digest(a.policy) != a.policy_hash
        or (
            a.returned_policy is not None
            and file_digest(a.returned_policy) != a.returned_policy_hash
        )
    ):
        raise ValueError("exact protocol and private policy required")
    core_pins = {}
    if a.core_source is not None:
        core = a.core_source.resolve() / "rosclaw"
        if not (core / "__init__.py").is_file():
            raise ValueError("explicit ROSClaw core source required")
        core_pins = {str(path): file_digest(path) for path in core.rglob("*.py")}
    experiment_pins = {}
    if a.standard_ball:
        files = [path for path in a.asset_root.rglob("*") if path.is_file()]
        if not files or any(path.is_symlink() for path in files):
            raise ValueError("standard-ball smoke requires ordinary pinned asset files")
        experiment_pins = {str(path): file_digest(path) for path in files}
        experiment_pins.update(core_pins)
        experiment_pins.update({str(a.protocol): a.protocol_hash, str(a.policy): a.policy_hash})
        if a.returned_policy is not None:
            experiment_pins[str(a.returned_policy)] = a.returned_policy_hash
        source_pins(experiment_pins)
        check_pins(experiment_pins)
    a.output.mkdir(parents=True, exist_ok=False)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            protocol_hash=a.protocol_hash,
            policy_hash=a.policy_hash,
            returned_policy_hash=a.returned_policy_hash,
            core_source_hashes=core_pins,
            mode="ASSISTED_TRAINING_ONLY",
            standard_ball=a.standard_ball,
            standard_ball_input_pins=experiment_pins,
            exploration_seed=a.exploration_seed,
            only_explore_after_return=a.only_explore_after_return,
            exploration_agent_ids=(
                ["blue.playmaker", "red.finisher"] if a.exploration_seed is not None else []
            ),
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        import numpy as np

        if a.core_source is not None:
            import rosclaw

            if Path(rosclaw.__file__).resolve() != a.core_source.resolve() / "rosclaw/__init__.py":
                raise ValueError("imported core does not match pinned source")

        from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
        from rosclaw_soccer.skills.team.independent_team_world import (
            IndependentTeamWorldScenario,
            simulate_independent_team_world,
        )
        from rosclaw_soccer.training.contact_control_profile import ContactControlProfile
        from rosclaw_soccer.training.continuous_competitive_match_growth import (
            _write_trajectory,
            default_continuous_match_config,
            default_continuous_match_options,
        )
        from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
        from rosclaw_soccer.training.near_ball_residual_ppo import _validate_on_policy
        from rosclaw_soccer.training.return_turnaround import return_turnaround_metrics
        from rosclaw_soccer.training.returned_ball_learning import returned_ball_segments
        from rosclaw_soccer.world.training_ball_return import TrainingBallReturnConfig

        fixture = ball_fixture(
            build_four_vs_four_fixture(a.asset_root, basic_ball_play=True),
            standard_ball=a.standard_ball,
        )
        fixture = replace(
            fixture,
            cells=tuple(
                replace(c, tactical_profile=replace(c.tactical_profile, anticipatory_contact=True))
                for c in fixture.cells
            ),
        )
        scenario = IndependentTeamWorldScenario(
            scenario_id="s199.continuous.symmetric-4v4.baseline",
            ball_initial_position_m=(3.0, 0.0, fixture.goal.ball_radius_m),
            ball_initial_velocity_mps=(0.0, 0.0, 0.0),
            seed=1928,
        )
        world = replace(
            default_continuous_match_config(),
            simulation_duration_sec=25.0,
            bilateral_goals=True,
            stationary_ball_acquisition=True,
            predictive_separation=True,
            all_role_clearance=True,
            strike_residual_enabled=True,
            training_ball_return=TrainingBallReturnConfig.rapid_practice(),
        )
        from rosclaw_soccer.growth.locomotion_contact_teacher import (
            G1LocomotionContactTeacherConfig,
        )

        world, teacher = ContactControlProfile(strike_residual_enabled=True).apply(
            world, G1LocomotionContactTeacherConfig()
        )
        dimensions = None
        model_hash = None
        if a.standard_ball:
            import mujoco

            from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
            from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
            from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

            model = build_g1_multi_player_stadium_model(
                a.asset_root,
                players=fixture.players,
                spec=fixture.goal,
                left_goal_plane_x_m=world.left_goal_plane_x_m,
            )
            model.opt.timestep = 0.002
            if world.glove_material is not None:
                for player in fixture.players:
                    if player.goalkeeper_gloves:
                        world.glove_material.apply(model, prefix=player.body_prefix)
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
            dimensions = inspect_native_ball_dimensions(model, geom_id=geom)
            if dimensions.radius_m != 0.110 or dimensions.body_mass_kg != 0.430:
                raise ValueError("compiled standard-sized sphere mismatch")
            model_hash = compiled_model_hash(model)
        option = default_continuous_match_options(world)
        policy = NearBallResidualPolicy.load(a.policy)
        returned_policy = (
            NearBallResidualPolicy.load(a.returned_policy) if a.returned_policy else None
        )
        results = []
        artifacts = []
        for label in ("primary", "replay"):
            result, trace = simulate_independent_team_world(
                asset_root=a.asset_root,
                roster=fixture.roster,
                cells=fixture.cells,
                players=fixture.players,
                scenario=scenario,
                goal=fixture.goal,
                config=world,
                contact_teacher_config=teacher,
                option_bridge_config=option,
                near_ball_policy=policy,
                near_ball_returned_policy=returned_policy,
                near_ball_explore=a.exploration_seed is not None,
                near_ball_seed=a.exploration_seed or 0,
                near_ball_exploration_agent_ids=(
                    ("blue.playmaker", "red.finisher") if a.exploration_seed is not None else None
                ),
                near_ball_explore_after_return=a.only_explore_after_return,
                capture_initial_physics=a.standard_ball,
            )
            if a.standard_ball and trace["initial_physics_hash"][0] != model_hash:
                raise ValueError("actual physical model differs from compiled preflight")
            sampled_by_role = None
            if a.exploration_seed is not None:
                _validate_on_policy(policy, trace)
                live_parts = returned_ball_segments(
                    {k: v for k, v in trace.items() if np.ndim(v) and len(v) == len(trace["time"])}
                )
                sampled_by_role = {
                    agent: int(
                        sum(
                            np.sum(
                                part["residual_active"][:, i]
                                & part["residual_exploration_mask"][:, i]
                            )
                            for part in live_parts
                        )
                    )
                    for i, agent in enumerate(policy.agent_ids)
                }
            returns = return_turnaround_metrics(trace["time"], trace["training_return_event_code"])
            reentries = np.flatnonzero(trace["training_return_event_code"] == 3)
            first_reentry = int(reentries[0]) if len(reentries) else len(trace["time"])
            later_contacts = (
                np.flatnonzero(trace["ball_contact_force_n"][first_reentry:] > 1) + first_reentry
            )
            later_nonfoot = (
                np.flatnonzero(trace["ball_nonfoot_contact_force_n"][first_reentry:] > 1)
                + first_reentry
            )
            artifacts.append(_write_trajectory(a.output / f"{label}.npz", trace))
            results.append(
                dict(
                    world_result=result.to_dict(),
                    returns=returns,
                    first_reentry_frame=None
                    if first_reentry == len(trace["time"])
                    else first_reentry,
                    later_recorded_foot_contact_frames=len(later_contacts),
                    later_recorded_nonfoot_contact_frames=len(later_nonfoot),
                    returned_live_active_exploration_frames=sampled_by_role,
                )
            )
        if (
            file_digest(a.protocol) != a.protocol_hash
            or file_digest(a.policy) != a.policy_hash
            or (
                a.returned_policy is not None
                and file_digest(a.returned_policy) != a.returned_policy_hash
            )
        ):
            raise ValueError("input changed during simulation")
        if any(file_digest(Path(path)) != h for path, h in core_pins.items()):
            raise ValueError("pinned core changed during simulation")
        if a.standard_ball:
            check_pins(experiment_pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="ASSISTED_TRAINING_ONLY_NOT_MATCH",
                source_head=a.expected_head,
                protocol_hash=a.protocol_hash,
                policy_hash=a.policy_hash,
                returned_policy_hash=a.returned_policy_hash,
                exploration_seed=a.exploration_seed,
                only_explore_after_return=a.only_explore_after_return,
                fixture_hash=fixture.fixture_hash,
                world_config=asdict(world),
                artifacts=artifacts,
                results=results,
                exact_replay=bool(
                    results[0] == results[1]
                    and artifacts[0]["trajectory_digest"] == artifacts[1]["trajectory_digest"]
                ),
                external_ball_state_write=True,
                external_returns_are_player_passes=False,
                standard_ball=a.standard_ball,
                ball_dimensions=None if dimensions is None else asdict(dimensions),
                compiled_model_hash=model_hash,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error_type=type(error).__name__, error=str(error), hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
