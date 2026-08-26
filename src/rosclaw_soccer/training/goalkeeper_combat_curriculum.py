"""Outcome-supervised combat curriculum for the goalkeeper teacher gate.

The curriculum runs paired counterfactual rollouts: identical shots are faced
with several bounded teacher gates.  The safest best-performing gate becomes a
supervision label for the actor.  This converts simulator success and failure
into a reusable policy before PPO resumes online improvement.
"""

from __future__ import annotations

import math
from typing import Any


def causal_lateral_teacher_action(
    *,
    torch: Any,
    observation: Any,
    gate_ceiling: float,
    maximum_lateral_command_mps: float,
    allow_recovery: bool,
    allow_full_predictive_cue: bool = False,
    release_schedule_sec: tuple[float, float, float, float] | None = None,
    episode_duration_sec: float | None = None,
) -> Any:
    """Return a causal time-to-intercept command with velocity-aware braking.

    Observation features are part of the public actor contract: ball x and
    velocity are features 0 and 3, intercept-relative y is feature 7, root
    lateral velocity is feature 13, and the final phase triplet declares an
    active shot.  No sampled future target or simulator-only state is used.
    """

    if (
        not math.isfinite(gate_ceiling)
        or not 0.0 <= gate_ceiling <= 1.0
        or not math.isfinite(maximum_lateral_command_mps)
        or not 0.05 <= maximum_lateral_command_mps <= 0.40
        or observation.ndim != 2
        or observation.shape[1] < 14
    ):
        raise ValueError("goalkeeper causal lateral teacher contract is invalid")
    if (release_schedule_sec is None) != (episode_duration_sec is None):
        raise ValueError("goalkeeper causal release schedule is incomplete")
    if release_schedule_sec is not None:
        first_release, _, second_release, second_end = release_schedule_sec
        if (
            not all(math.isfinite(value) for value in release_schedule_sec)
            or tuple(sorted(release_schedule_sec)) != release_schedule_sec
            or episode_duration_sec is None
            or not math.isfinite(episode_duration_sec)
            or second_end >= episode_duration_sec
        ):
            raise ValueError("goalkeeper causal release schedule is invalid")
    target_error_y = observation[:, 7] / 0.5
    root_lateral_velocity = observation[:, 13] / 0.5
    ball_relative_x = observation[:, 0] / 0.4
    ball_velocity_x = observation[:, 3] / 0.2
    shot_active = (observation[:, -3] > 0.5) | (observation[:, -2] > 0.5)
    predictive_cue = torch.zeros_like(shot_active)
    if allow_full_predictive_cue and observation.shape[1] == 77:
        predictive_cue = (observation[:, -4] > 0.5) & (observation[:, -5] > 0.5)
    time_to_line = torch.clamp(
        -ball_relative_x / torch.clamp(ball_velocity_x, min=0.10),
        min=0.10,
        max=1.20,
    )
    # Recovery has more time than a shot, while the derivative term causes a
    # sign reversal before centre crossing instead of a bang-bang oscillation.
    ready_horizon = torch.full_like(time_to_line, 0.55)
    if release_schedule_sec is not None and episode_duration_sec is not None:
        elapsed_sec = observation[:, -1] * episode_duration_sec
        next_release = torch.where(
            elapsed_sec < first_release,
            torch.full_like(elapsed_sec, first_release),
            torch.full_like(elapsed_sec, second_release),
        )
        ready_horizon = torch.clamp(next_release - elapsed_sec + 0.08, 0.12, 1.50)
    horizon = torch.where(shot_active, time_to_line + 0.08, ready_horizon)
    desired_world_velocity = target_error_y / horizon
    desired_world_velocity -= 0.55 * root_lateral_velocity
    local_command = -desired_world_velocity / maximum_lateral_command_mps
    recovery_ceiling = min(gate_ceiling, 0.25)
    ceiling = torch.where(
        shot_active | predictive_cue,
        torch.full_like(local_command, gate_ceiling),
        torch.full_like(local_command, recovery_ceiling),
    )
    enabled = torch.ones_like(shot_active) if allow_recovery else shot_active
    return torch.where(
        enabled,
        torch.clamp(local_command, min=-ceiling, max=ceiling),
        torch.zeros_like(local_command),
    )


def pretrain_combat_teacher_gate(
    *,
    torch: Any,
    model: Any,
    environment: Any,
    device: Any,
    batches: int,
    epochs: int,
    seed: int,
    task_space_reach_blend: float = 0.0,
    task_space_reach_atlas_enabled: bool = False,
    second_shot_reach_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Learn a causal gate from paired physical outcomes, never future state."""

    if (
        not isinstance(task_space_reach_atlas_enabled, bool)
        or (task_space_reach_atlas_enabled and task_space_reach_blend <= 0.0)
        or not 1 <= batches <= 64
        or not 1 <= epochs <= 100
        or not math.isfinite(task_space_reach_blend)
        or not 0.0 <= task_space_reach_blend <= 0.85
        or not math.isfinite(second_shot_reach_multiplier)
        or not 1.0 <= second_shot_reach_multiplier <= 2.0
        or task_space_reach_blend * second_shot_reach_multiplier > 0.85
    ):
        raise ValueError("goalkeeper combat curriculum dimensions are invalid")
    reach_model = None
    reach_report = None
    if task_space_reach_blend > 0.0:
        from rosclaw_soccer.training.goalkeeper_reach import (
            GoalkeeperReachConfig,
            build_g1_task_space_reach_atlas,
            build_g1_task_space_reach_model,
            reach_model_payload,
        )

        reach_config = GoalkeeperReachConfig(
            damping=0.12,
            reach_gain=0.95,
            maximum_position_error_m=0.75,
            support_arm_scale=0.60,
            central_support_scale=0.95,
            residual_scale=environment.config.residual_scale,
            arm_authority_scale=environment.config.agility.arm_authority_scale,
        )
        reach_model = (
            build_g1_task_space_reach_atlas(environment._asset_root, config=reach_config)
            if task_space_reach_atlas_enabled
            else build_g1_task_space_reach_model(environment._asset_root, config=reach_config)
        )
        reach_report = reach_model_payload(reach_model)
    mobility_option_enabled = bool(getattr(environment, "mobility_option_enabled", False))
    lateral_ceiling = (
        environment.mobility_option_config.lateral_command_limit
        if mobility_option_enabled
        else 0.35
    )
    gates = (0.0, 0.33, 0.66, 1.0) if mobility_option_enabled else (0.0, 0.15, 0.25, 0.35)
    observation_batches: list[Any] = []
    target_batches: list[Any] = []
    first_label_counts = [0 for _ in gates]
    second_label_counts = [0 for _ in gates]
    gate_outcomes = [
        {
            "episodes": 0,
            "first_save": 0,
            "first_hand_save": 0,
            "recovery": 0,
            "second_attempt_save": 0,
            "second_attempt_hand_save": 0,
            "second_save": 0,
            "second_hand_save": 0,
            "failed": 0,
            "first_min_hand_distance_sum_m": 0.0,
            "second_min_hand_distance_sum_m": 0.0,
            "first_max_root_angular_speed_sum_rad_s": 0.0,
            "second_max_root_angular_speed_sum_rad_s": 0.0,
            "second_release_lateral_error_sum_m": 0.0,
        }
        for _ in gates
    ]
    for batch_index in range(batches):
        paired_seed = seed + 65_537 * batch_index
        first_scores: list[Any] = []
        second_scores: list[Any] = []
        frames_by_gate: list[Any] = []
        for gate_index, gate in enumerate(gates):
            observation = environment.reset(seed=paired_seed)
            rollout_frames: list[Any] = []
            phase_reward = torch.zeros((2, environment.count), device=device)
            phase_min_hand_distance = torch.full((2, environment.count), 10.0, device=device)
            phase_max_root_angular_speed = torch.zeros((2, environment.count), device=device)
            action = torch.zeros((environment.count, environment.action_size), device=device)
            first_release_step = int(
                round(environment.config.first_shot_release_sec / environment.config.control_dt_sec)
            )
            first_end_step = int(
                round(environment.config.first_shot_end_sec / environment.config.control_dt_sec)
            )
            second_release_step = int(
                round(
                    environment.config.second_shot_release_sec / environment.config.control_dt_sec
                )
            )
            for step_index in range(environment.config.episode_steps):
                if step_index % 2 == 0:
                    rollout_frames.append(observation.detach().cpu())
                action.zero_()
                action[:, 0] = causal_lateral_teacher_action(
                    torch=torch,
                    observation=observation,
                    gate_ceiling=lateral_ceiling if mobility_option_enabled else gate,
                    maximum_lateral_command_mps=(environment.config.maximum_lateral_command_mps),
                    allow_recovery=(
                        (not mobility_option_enabled and step_index >= first_release_step)
                        or (
                            environment.config.shot_intent_cue_enabled
                            and (
                                step_index < first_release_step
                                or first_end_step <= step_index < second_release_step
                            )
                        )
                    ),
                    allow_full_predictive_cue=(
                        mobility_option_enabled
                        and environment.config.shot_intent_cue_enabled
                        and environment.mobility_option_config.anticipatory_arm_reach_enabled
                    ),
                    release_schedule_sec=(
                        environment.config.first_shot_release_sec,
                        environment.config.first_shot_end_sec,
                        environment.config.second_shot_release_sec,
                        environment.config.second_shot_end_sec,
                    ),
                    episode_duration_sec=environment.config.episode_duration_sec,
                )
                if mobility_option_enabled:
                    action[:, 1] = gate
                observation, reward, _, info = environment.step(action)
                event_shot_index = info["event_shot_index"]
                left_hand = environment.geom_xpos[:, environment._left_hand_geom]
                right_hand = environment.geom_xpos[:, environment._right_hand_geom]
                target = info["target_m"]
                hand_distance = torch.minimum(
                    torch.linalg.vector_norm(left_hand - target, dim=1),
                    torch.linalg.vector_norm(right_hand - target, dim=1),
                )
                root_angular_speed = torch.linalg.vector_norm(environment.qvel[:, 3:6], dim=1)
                for phase_index, shot_index in enumerate((1, 2)):
                    phase_mask = event_shot_index == shot_index
                    phase_reward[phase_index] += torch.where(
                        phase_mask, reward, torch.zeros_like(reward)
                    )
                    phase_min_hand_distance[phase_index] = torch.where(
                        phase_mask,
                        torch.minimum(phase_min_hand_distance[phase_index], hand_distance),
                        phase_min_hand_distance[phase_index],
                    )
                    phase_max_root_angular_speed[phase_index] = torch.where(
                        phase_mask,
                        torch.maximum(
                            phase_max_root_angular_speed[phase_index],
                            root_angular_speed,
                        ),
                        phase_max_root_angular_speed[phase_index],
                    )
            failed = environment.task.phase == 7
            reach_quality = torch.clamp(1.0 - phase_min_hand_distance / 1.25, 0.0, 1.0)
            safety_penalty = 1000.0 * failed.to(torch.float32)
            first_score = phase_reward[0]
            first_score += 100.0 * environment.task.first_save.to(torch.float32)
            first_score += 160.0 * environment.task.first_hand_save.to(torch.float32)
            first_score += 25.0 * environment.task.recovered_after_first.to(torch.float32)
            first_score += 35.0 * reach_quality[0]
            first_score -= 14.0 * phase_max_root_angular_speed[0]
            first_score -= safety_penalty + 0.50 * gate
            second_score = phase_reward[1]
            second_score += 120.0 * environment.task.second_attempt_save.to(torch.float32)
            second_score += 180.0 * environment.task.second_attempt_hand_save.to(torch.float32)
            second_score += 180.0 * environment.task.second_save.to(torch.float32)
            second_score += 260.0 * environment.task.second_hand_save.to(torch.float32)
            second_score += 45.0 * reach_quality[1]
            second_score -= 18.0 * phase_max_root_angular_speed[1]
            second_score -= 35.0 * environment._second_release_lateral_error
            second_score -= safety_penalty + 0.50 * gate
            first_scores.append(first_score.detach().cpu())
            second_scores.append(second_score.detach().cpu())
            frames_by_gate.append(torch.stack(rollout_frames))
            outcome = gate_outcomes[gate_index]
            outcome["episodes"] += environment.count
            outcome["first_save"] += int(environment.task.first_save.sum())
            outcome["first_hand_save"] += int(environment.task.first_hand_save.sum())
            outcome["recovery"] += int(environment.task.recovered_after_first.sum())
            outcome["second_attempt_save"] += int(environment.task.second_attempt_save.sum())
            outcome["second_attempt_hand_save"] += int(
                environment.task.second_attempt_hand_save.sum()
            )
            outcome["second_save"] += int(environment.task.second_save.sum())
            outcome["second_hand_save"] += int(environment.task.second_hand_save.sum())
            outcome["failed"] += int(failed.sum())
            outcome["first_min_hand_distance_sum_m"] += float(phase_min_hand_distance[0].sum())
            outcome["second_min_hand_distance_sum_m"] += float(phase_min_hand_distance[1].sum())
            outcome["first_max_root_angular_speed_sum_rad_s"] += float(
                phase_max_root_angular_speed[0].sum()
            )
            outcome["second_max_root_angular_speed_sum_rad_s"] += float(
                phase_max_root_angular_speed[1].sum()
            )
            outcome["second_release_lateral_error_sum_m"] += float(
                environment._second_release_lateral_error.sum()
            )
        best_first_gate_index = torch.stack(first_scores).argmax(dim=0)
        best_second_gate_index = torch.stack(second_scores).argmax(dim=0)
        second_enabled = environment._second_enabled.detach().cpu()
        for gate_index, gate in enumerate(gates):
            for phase, best_index, phase_label_counts in (
                (1, best_first_gate_index, first_label_counts),
                (2, best_second_gate_index, second_label_counts),
            ):
                selected = best_index == gate_index
                if phase == 2:
                    selected &= second_enabled
                count = int(selected.sum())
                phase_label_counts[gate_index] += count
                if count == 0:
                    continue
                training_frames = frames_by_gate[gate_index][:, selected, :].flatten(0, 1)
                # The final four features are three causal phase indicators
                # plus elapsed time.  Each shot earns its own outcome label.
                phase_frames = training_frames[training_frames[:, -4 + phase] > 0.5]
                targets = torch.zeros(
                    (phase_frames.shape[0], environment.action_size),
                    dtype=torch.float32,
                )
                targets[:, 0] = causal_lateral_teacher_action(
                    torch=torch,
                    observation=phase_frames,
                    gate_ceiling=lateral_ceiling if mobility_option_enabled else gate,
                    maximum_lateral_command_mps=(environment.config.maximum_lateral_command_mps),
                    allow_recovery=False,
                    release_schedule_sec=(
                        environment.config.first_shot_release_sec,
                        environment.config.first_shot_end_sec,
                        environment.config.second_shot_release_sec,
                        environment.config.second_shot_end_sec,
                    ),
                    episode_duration_sec=environment.config.episode_duration_sec,
                )
                if mobility_option_enabled:
                    targets[:, 1] = gate
                if reach_model is not None:
                    from rosclaw_soccer.training.goalkeeper_reach import (
                        task_space_reach_teacher_action,
                    )

                    reach_target = task_space_reach_teacher_action(phase_frames, model=reach_model)
                    reach_blend = task_space_reach_blend
                    if phase == 2:
                        reach_blend *= second_shot_reach_multiplier
                    targets[:, 4:] = reach_blend * reach_target[:, 4:]
                observation_batches.append(phase_frames)
                target_batches.append(targets)
                if (
                    phase == 1
                    and mobility_option_enabled
                    and environment.config.shot_intent_cue_enabled
                    and environment.mobility_option_config.anticipatory_arm_reach_enabled
                ):
                    anticipation_frames = frames_by_gate[gate_index][:, selected, :].flatten(0, 1)
                    anticipation_frames = anticipation_frames[
                        (anticipation_frames[:, -4] > 0.5)
                        & (
                            anticipation_frames[:, -1]
                            < environment.config.first_shot_release_sec
                            / environment.config.episode_duration_sec
                        )
                    ]
                    anticipation_targets = torch.zeros(
                        (anticipation_frames.shape[0], environment.action_size),
                        dtype=torch.float32,
                    )
                    anticipation_targets[:, 0] = causal_lateral_teacher_action(
                        torch=torch,
                        observation=anticipation_frames,
                        gate_ceiling=lateral_ceiling,
                        maximum_lateral_command_mps=(
                            environment.config.maximum_lateral_command_mps
                        ),
                        allow_recovery=True,
                        allow_full_predictive_cue=True,
                        release_schedule_sec=(
                            environment.config.first_shot_release_sec,
                            environment.config.first_shot_end_sec,
                            environment.config.second_shot_release_sec,
                            environment.config.second_shot_end_sec,
                        ),
                        episode_duration_sec=environment.config.episode_duration_sec,
                    )
                    if environment.mobility_option_config.predictive_teacher_warmstart_enabled:
                        anticipation_targets[:, 1] = gate * (anticipation_frames[:, -5] > 0.5).to(
                            torch.float32
                        )
                    if reach_model is not None:
                        from rosclaw_soccer.training.goalkeeper_reach import (
                            task_space_reach_teacher_action,
                        )

                        anticipation_reach = task_space_reach_teacher_action(
                            anticipation_frames,
                            model=reach_model,
                            allow_intent_cue=True,
                        )
                        anticipation_targets[:, 4:] = (
                            task_space_reach_blend * anticipation_reach[:, 4:]
                        )
                    observation_batches.append(anticipation_frames)
                    target_batches.append(anticipation_targets)
            selected_for_recovery = best_second_gate_index == gate_index
            selected_for_recovery &= second_enabled
            recovery_supervision = (
                not mobility_option_enabled or environment.config.shot_intent_cue_enabled
            )
            if recovery_supervision and bool(torch.any(selected_for_recovery)):
                recovery_frames = frames_by_gate[gate_index][:, selected_for_recovery, :].flatten(
                    0, 1
                )
                elapsed = recovery_frames[:, -1]
                recovery_frames = recovery_frames[
                    (recovery_frames[:, -4] > 0.5)
                    & (
                        elapsed
                        >= (
                            environment.config.first_shot_end_sec
                            if environment.config.shot_intent_cue_enabled
                            else environment.config.first_shot_release_sec
                        )
                        / environment.config.episode_duration_sec
                    )
                    & (
                        elapsed
                        < environment.config.second_shot_release_sec
                        / environment.config.episode_duration_sec
                    )
                ]
                recovery_targets = torch.zeros(
                    (recovery_frames.shape[0], environment.action_size),
                    dtype=torch.float32,
                )
                recovery_targets[:, 0] = causal_lateral_teacher_action(
                    torch=torch,
                    observation=recovery_frames,
                    gate_ceiling=lateral_ceiling if mobility_option_enabled else gate,
                    maximum_lateral_command_mps=(environment.config.maximum_lateral_command_mps),
                    allow_recovery=True,
                    allow_full_predictive_cue=(
                        environment.config.shot_intent_cue_enabled
                        and environment.mobility_option_config.anticipatory_arm_reach_enabled
                    ),
                    release_schedule_sec=(
                        environment.config.first_shot_release_sec,
                        environment.config.first_shot_end_sec,
                        environment.config.second_shot_release_sec,
                        environment.config.second_shot_end_sec,
                    ),
                    episode_duration_sec=environment.config.episode_duration_sec,
                )
                if (
                    mobility_option_enabled
                    and environment.mobility_option_config.predictive_teacher_warmstart_enabled
                ):
                    recovery_targets[:, 1] = gate * (recovery_frames[:, -5] > 0.5).to(torch.float32)
                if (
                    reach_model is not None
                    and environment.config.shot_intent_cue_enabled
                    and environment.mobility_option_config.anticipatory_arm_reach_enabled
                ):
                    from rosclaw_soccer.training.goalkeeper_reach import (
                        task_space_reach_teacher_action,
                    )

                    recovery_reach = task_space_reach_teacher_action(
                        recovery_frames,
                        model=reach_model,
                        allow_intent_cue=True,
                    )
                    recovery_targets[:, 4:] = (
                        task_space_reach_blend
                        * second_shot_reach_multiplier
                        * recovery_reach[:, 4:]
                    )
                observation_batches.append(recovery_frames)
                target_batches.append(recovery_targets)
        inactive_frames = frames_by_gate[0].flatten(0, 1)
        elapsed = inactive_frames[:, -1]
        recovery_window = (
            elapsed
            >= environment.config.first_shot_release_sec / environment.config.episode_duration_sec
        ) & (
            elapsed
            < environment.config.second_shot_release_sec / environment.config.episode_duration_sec
        )
        if environment.config.shot_intent_cue_enabled:
            recovery_window |= (
                elapsed
                < environment.config.first_shot_release_sec
                / environment.config.episode_duration_sec
            )
        inactive_frames = inactive_frames[(inactive_frames[:, -4] > 0.5) & ~recovery_window][::5]
        observation_batches.append(inactive_frames)
        target_batches.append(
            torch.zeros(
                (inactive_frames.shape[0], environment.action_size),
                dtype=torch.float32,
            )
        )
    if not observation_batches:
        raise RuntimeError("goalkeeper combat curriculum produced no labeled observations")
    observations = torch.cat(observation_batches).to(device)
    targets = torch.cat(target_batches).to(device)
    if observations.shape[1] != environment.observation_size:
        raise RuntimeError("goalkeeper combat curriculum observation contract changed")

    with torch.no_grad():
        parent_mean, _, _ = model(observations)
        parent_actions = torch.tanh(parent_mean).detach()

    def imitation_loss(mean: Any, target: Any, parent_target: Any) -> Any:
        bounded = torch.tanh(mean)
        gate_width = 2 if mobility_option_enabled else 1
        gate_loss = torch.mean(torch.square(bounded[:, :gate_width] - target[:, :gate_width]))
        residual_loss = torch.mean(torch.square(bounded[:, gate_width:] - target[:, gate_width:]))
        retention_loss = torch.mean(torch.square(bounded - parent_target))
        return 8.0 * gate_loss + 2.0 * residual_loss + 0.50 * retention_loss

    with torch.no_grad():
        initial_mean, _, _ = model(observations)
        initial_loss = float(imitation_loss(initial_mean, targets, parent_actions))
    optimizer = torch.optim.Adam(
        tuple(model.trunk.parameters()) + tuple(model.actor.parameters()),
        lr=1.0e-3,
        eps=1.0e-5,
    )
    minibatch_size = min(4096, observations.shape[0])
    final_loss = math.inf
    for _ in range(epochs):
        permutation = torch.randperm(observations.shape[0], device=device)
        for start in range(0, observations.shape[0], minibatch_size):
            indices = permutation[start : start + minibatch_size]
            mean, _, _ = model(observations[indices])
            loss = imitation_loss(mean, targets[indices], parent_actions[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            final_loss = float(loss.detach())
    with torch.no_grad():
        final_mean, _, _ = model(observations)
        final_loss = float(imitation_loss(final_mean, targets, parent_actions))
        final_parent_retention_loss = float(
            torch.mean(torch.square(torch.tanh(final_mean) - parent_actions))
        )
        reported_gate_channel = 1 if mobility_option_enabled else 0
        active = torch.abs(targets[:, reported_gate_channel]) > 0.0
        predicted_active_gate = (
            float(torch.abs(torch.tanh(final_mean[active, reported_gate_channel])).mean())
            if bool(torch.any(active))
            else 0.0
        )
    if not math.isfinite(final_loss) or final_loss >= initial_loss:
        raise RuntimeError("goalkeeper combat curriculum failed to improve its gate loss")
    outcome_rates = []
    for gate, outcome_counts in zip(gates, gate_outcomes, strict=True):
        episodes = int(outcome_counts["episodes"])
        outcome_rates.append(
            {
                "gate": gate,
                "episodes": episodes,
                "first_save_rate": int(outcome_counts["first_save"]) / episodes,
                "first_hand_save_rate": int(outcome_counts["first_hand_save"]) / episodes,
                "recovery_rate": int(outcome_counts["recovery"]) / episodes,
                "second_attempt_save_rate": int(outcome_counts["second_attempt_save"]) / episodes,
                "second_attempt_hand_save_rate": int(outcome_counts["second_attempt_hand_save"])
                / episodes,
                "second_save_rate": int(outcome_counts["second_save"]) / episodes,
                "second_hand_save_rate": int(outcome_counts["second_hand_save"]) / episodes,
                "failed_rate": int(outcome_counts["failed"]) / episodes,
                "mean_first_min_hand_distance_m": float(
                    outcome_counts["first_min_hand_distance_sum_m"]
                )
                / episodes,
                "mean_second_min_hand_distance_m": float(
                    outcome_counts["second_min_hand_distance_sum_m"]
                )
                / episodes,
                "mean_first_max_root_angular_speed_rad_s": float(
                    outcome_counts["first_max_root_angular_speed_sum_rad_s"]
                )
                / episodes,
                "mean_second_max_root_angular_speed_rad_s": float(
                    outcome_counts["second_max_root_angular_speed_sum_rad_s"]
                )
                / episodes,
                "mean_second_release_lateral_error_m": float(
                    outcome_counts["second_release_lateral_error_sum_m"]
                )
                / episodes,
            }
        )
    return {
        "schema_version": "rosclaw_soccer.goalkeeper_combat_curriculum.v6",
        "method": ("PHASE_CONDITIONED_COUNTERFACTUAL_OUTCOME_AND_RECOVERY_SUPERVISION"),
        "paired_shot_batches": batches,
        "episodes": batches * environment.count * len(gates),
        "physics_world_steps": (
            batches
            * environment.count
            * len(gates)
            * environment.config.episode_steps
            * environment.config.physics_substeps
        ),
        "causal_observation_samples": int(observations.shape[0]),
        "candidate_gates": list(gates),
        "selected_gate_counts": {
            "first_shot": {
                str(gate): count for gate, count in zip(gates, first_label_counts, strict=True)
            },
            "second_shot": {
                str(gate): count for gate, count in zip(gates, second_label_counts, strict=True)
            },
        },
        "gate_outcomes": outcome_rates,
        "initial_imitation_loss": initial_loss,
        "final_imitation_loss": final_loss,
        "final_parent_retention_loss": final_parent_retention_loss,
        "predicted_active_gate_magnitude_mean": predicted_active_gate,
        "task_space_reach_blend": task_space_reach_blend,
        "task_space_reach_atlas_enabled": task_space_reach_atlas_enabled,
        "second_shot_reach_multiplier": second_shot_reach_multiplier,
        "effective_second_shot_reach_blend": (
            task_space_reach_blend * second_shot_reach_multiplier
        ),
        "task_space_reach_model": reach_report,
        "improved": final_loss < initial_loss,
        "failure_replay": True,
        "counterfactual_pairing": True,
        "causal_time_to_intercept_braking": True,
        "learned_between_shot_recovery": (
            not mobility_option_enabled or environment.config.shot_intent_cue_enabled
        ),
        "causal_shot_intent_cue": environment.config.shot_intent_cue_enabled,
        "mobility_option_enabled": mobility_option_enabled,
        "separate_lateral_and_teacher_gate": mobility_option_enabled,
        "residual_plasticity_scale": (
            environment.mobility_option_config.residual_plasticity_scale
            if mobility_option_enabled
            else 1.0
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }


__all__ = ["causal_lateral_teacher_action", "pretrain_combat_teacher_gate"]
