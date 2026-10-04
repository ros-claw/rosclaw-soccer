"""Extract current-parent learning data only AFTER full CPU dynamics replay.

No simulator, optimizer, reward change, trajectory selection or activation.
All 270 causal motor frames, including later knee contacts/failures, are kept.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.proposal_sampling_motor import EXPLORATION_PROFILES
from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor, make_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_fit_protected_phase_step_motor import cpu_features

if TYPE_CHECKING:
    pass


def ordered_cpu_arrays(
    observation: Any,
    phase: Any,
    draws: list[tuple[Any, float]],
    *,
    outcome: dict[str, Any],
    std: float,
    group: int,
    exploration_profile: str = "nominal",
) -> dict[str, Any]:
    x, p = np.asarray(observation), np.asarray(phase)
    if type(exploration_profile) is not str or exploration_profile not in EXPLORATION_PROFILES:
        raise ValueError("explicit bounded CPU exploration profile required")
    if (
        type(group) is not int
        or not 0 <= group < 52 * 16
        or type(std) is not float
        or std != EXPLORATION_PROFILES[exploration_profile]
        or x.shape != (270, 134)
        or p.shape != (270,)
        or p.dtype.kind not in "iu"
        or not set(p.tolist()) <= {0, 1, 2}
        or not np.isfinite(x).all()
        or len(draws) != 270
    ):
        raise ValueError("all ordered finite 270 CPU motor observations required")
    actions = np.stack([v[0] for v in draws])
    logp = np.asarray([v[1] for v in draws])
    if (
        actions.shape != (270, 12)
        or logp.shape != (270,)
        or not all(np.isfinite(v).all() for v in (actions, logp))
    ):
        raise ValueError("all complete reconstructed latent draws/likelihoods required")
    return dict(
        observation=x.copy(),
        phase_index=p.copy(),
        latent_action=actions,
        old_log_probability=logp,
        terminal_return=np.full(270, terminal_return(outcome)),
        std_raw=np.full(270, std),
        trajectory_index=np.full(270, group, dtype=np.int64),
    )


def audit_cpu_learning_rollout(
    folder: Path,
    runner: Path,
    *,
    mean_model: dict[str, Any],
    expected_view_hash: str,
    expected_sampling_seed: int,
    course: tuple[int, int],
    group: int,
    sampling_decoder_factory: Any = None,
    expected_exploration_profile: str = "nominal",
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _sealed(folder / "report.json")
    view = raw.get("executed_motor_policy", {}).get("step_motor_proof", {}).get("model", {})
    proposal_sampling = mean_model.get("schema") == "soccer.rsi.proposal_memory_motor.v1"
    if (
        type(expected_exploration_profile) is not str
        or expected_exploration_profile not in EXPLORATION_PROFILES
        or (not proposal_sampling and expected_exploration_profile != "nominal")
    ):
        raise ValueError("explicit source-appropriate CPU exploration profile required")
    expected_std = EXPLORATION_PROFILES[expected_exploration_profile]
    expected_schema = (
        "soccer.rsi.proposal_memory_sampling.v1"
        if proposal_sampling
        else "soccer.rsi.smooth_memory_sampling.v1"
    )
    if proposal_sampling and sampling_decoder_factory is not None:
        from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
            OwnedProposalSamplingEpisodeFactory,
        )
        from rosclaw_soccer.rsi.proposal_sampling_episode_factory import (
            ProposalSamplingEpisodeFactory,
        )

        if type(sampling_decoder_factory) not in (
            ProposalSamplingEpisodeFactory,
            OwnedProposalSamplingEpisodeFactory,
        ):
            raise ValueError("proposal sampling requires independent original reference replay")
    if (
        (raw.get("seed"), raw.get("lane")) != course
        or raw.get("execution_profile") != "taskspace_plus_motor"
        or "observation_contract" in raw
        or raw.get("source_hash") != hash_bytes(runner.read_bytes())
        or raw.get("step_model_hash") != expected_view_hash
        or view.get("model_hash") != expected_view_hash
        or view.get("model_hash") != hash_json({k: v for k, v in view.items() if k != "model_hash"})
        or view.get("schema") != expected_schema
        or view.get("mean_model") != mean_model
        or type(view.get("seed")) is not int
        or view["seed"] != expected_sampling_seed
        or view.get("std_raw") != expected_std
        or (
            proposal_sampling
            and view.get("exploration_profile", "nominal") != expected_exploration_profile
        )
        or view.get("rho") != 0.9
        or view.get("training_only") is not True
        or any(raw.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("exact declared CPU current-parent physical sampling view required")
    review = (
        audit_cpu_transfer(folder, runner)
        if sampling_decoder_factory is None
        else audit_cpu_transfer(folder, runner, sampling_decoder_factory=sampling_decoder_factory)
    )
    if (
        any(
            review.get(k) is not True
            for k in (
                "actual_mujoco_dynamics_replayed",
                "actual_pd_torque_reconstructed",
                "neural_target_reconstructed",
            )
        )
        or review.get("physical_substeps") != 3000
    ):
        raise ValueError("complete CPU dynamics, PD and neural replay required")
    write_once(folder / "review.json", review)
    x, phase, reread = cpu_features(
        folder,
        dict(
            report_hash=raw["report_hash"],
            review_hash=review["report_hash"],
        ),
    )
    if reread != raw:
        raise ValueError("CPU physical evidence changed during extraction")
    decoder: CompiledSmoothMemoryMotor
    if proposal_sampling:
        from rosclaw_soccer.rsi.proposal_sampling_motor import CompiledProposalSamplingMotor

        decoder = (
            CompiledProposalSamplingMotor(raw["executed_motor_policy"])
            if sampling_decoder_factory is None
            else sampling_decoder_factory.bind(raw["executed_motor_policy"])
        )
    else:
        decoder = (
            CompiledSmoothMemoryMotor(make_preview(view))
            if sampling_decoder_factory is None
            else sampling_decoder_factory.bind(raw["executed_motor_policy"])
        )
    draws = [
        decoder.latent_sample(v, frame, int(p))
        for frame, (v, p) in enumerate(zip(x, phase, strict=True), start=30)
    ]
    outcome = {
        **review,
        "reward": first_touch_reward(
            {**review, "max_lateral_excursion_m": review["maximum_lateral_excursion_m"]}
        ),
    }
    arrays = ordered_cpu_arrays(
        x,
        phase,
        draws,
        outcome=outcome,
        std=expected_std,
        group=group,
        exploration_profile=expected_exploration_profile,
    )
    record = dict(
        group=group,
        seed=course[0],
        lane=course[1],
        sampling_seed=expected_sampling_seed,
        exploration_profile=expected_exploration_profile,
        behavior_std_raw=expected_std,
        backend="MuJoCo",
        folder=str(folder),
        report_hash=raw["report_hash"],
        review_hash=review["report_hash"],
        view_hash=expected_view_hash,
        outcome=outcome,
    )
    return record, arrays
