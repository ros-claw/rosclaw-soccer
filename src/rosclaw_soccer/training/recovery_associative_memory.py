"""Proprioceptive associative muscle memory for recovery control.

The bank stores only successful state/action episodes.  At deployment it is
given a post-skill body-state identity and then uses current proprioception plus
an internal monotonic step counter to retrieve motor targets.  It never reads a
motion-reference phase, teacher identity, or future reference state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.training.recovery_snapshot import RecoverySnapshot
from rosclaw_soccer.training.recovery_student import RecoveryDistillationCorpus


@dataclass(frozen=True)
class RecoveryAssociativeMemoryConfig:
    """Bounded retrieval settings selected before a sealed physics exam."""

    nearest_neighbors: int = 3
    softmax_temperature: float = 0.20
    initial_distance_weight: float = 0.20
    retrieval_interval_steps: int = 10
    dynamic_retrieval: bool = True
    phase_search_back_steps: int = 0
    phase_search_forward_steps: int = 0
    phase_deviation_penalty: float = 0.05
    restrict_to_base_snapshot: bool = True
    exact_match_fixed_replay: bool = True
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_associative_memory_config.v1"

    def __post_init__(self) -> None:
        if (
            not 1 <= self.nearest_neighbors <= 16
            or not math.isfinite(self.softmax_temperature)
            or not 0.01 <= self.softmax_temperature <= 2.0
            or not math.isfinite(self.initial_distance_weight)
            or not 0.0 <= self.initial_distance_weight <= 2.0
            or not 1 <= self.retrieval_interval_steps <= 100
            or not 0 <= self.phase_search_back_steps <= 50
            or not 0 <= self.phase_search_forward_steps <= 100
            or (self.phase_search_back_steps > 0) != (self.phase_search_forward_steps > 0)
            or not math.isfinite(self.phase_deviation_penalty)
            or not 0.0 <= self.phase_deviation_penalty <= 2.0
            or (
                self.phase_search_forward_steps > 0
                and (self.nearest_neighbors != 1 or self.dynamic_retrieval)
            )
            or not self.restrict_to_base_snapshot
            or not self.exact_match_fixed_replay
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery associative-memory config is invalid")


@dataclass(frozen=True)
class RecoveryMemoryEpisode:
    episode_index: int
    base_snapshot_hash: str
    initial_snapshot_hash: str
    initial_signature: NDArray[np.float32]
    proprioception: NDArray[np.float32]
    absolute_motor_targets_rad: NDArray[np.float32]

    def __post_init__(self) -> None:
        if (
            self.episode_index < 0
            or not self.base_snapshot_hash.startswith("sha256:")
            or not self.initial_snapshot_hash.startswith("sha256:")
            or self.initial_signature.shape != (58,)
            or self.proprioception.ndim != 2
            or self.proprioception.shape[1] != 93
            or self.absolute_motor_targets_rad.shape
            != (self.proprioception.shape[0], 29)
            or self.proprioception.shape[0] == 0
            or not np.all(np.isfinite(self.initial_signature))
            or not np.all(np.isfinite(self.proprioception))
            or not np.all(np.isfinite(self.absolute_motor_targets_rad))
        ):
            raise ValueError("recovery memory episode is invalid")


def recovery_initial_signature(
    snapshot: RecoverySnapshot,
    corpus: RecoveryDistillationCorpus,
) -> NDArray[np.float32]:
    """Normalize deployment-visible initial joints for memory routing."""

    signature: NDArray[np.float32] = np.concatenate(
        (
            (
                np.asarray(snapshot.qpos[7:], dtype=np.float32)
                - corpus.default_joint_position_rad
            )
            / 0.030,
            (np.asarray(snapshot.qvel[6:], dtype=np.float32) * 0.05) / 0.004,
        )
    ).astype(np.float32)
    return signature


def recovery_feedback_signature(
    proprioception: NDArray[np.floating[Any]],
) -> NDArray[np.float32]:
    """Normalize current proprioception for phase-local associative lookup."""

    frame = np.asarray(proprioception, dtype=np.float32)
    if frame.shape != (93,) or not np.all(np.isfinite(frame)):
        raise ValueError("recovery associative query must contain 93 finite values")
    signature: NDArray[np.float32] = np.concatenate(
        (
            frame[0:3] / 0.05,
            frame[3:6] / 0.025,
            frame[6:35] / 0.030,
            frame[35:64] / 0.004,
        )
    ).astype(np.float32)
    return signature


def build_recovery_memory_episodes(
    corpus: RecoveryDistillationCorpus,
    *,
    excluded_initial_hashes: frozenset[str] = frozenset(),
) -> tuple[RecoveryMemoryEpisode, ...]:
    """Materialize successful privileged episodes without pickle or references."""

    episodes: list[RecoveryMemoryEpisode] = []
    for row in corpus.rows:
        initial_hash = str(row["initial_snapshot_hash"])
        if (
            initial_hash in excluded_initial_hashes
            or str(row.get("rollout_controller", "PRIVILEGED_TEACHER"))
            != "PRIVILEGED_TEACHER"
        ):
            continue
        start = int(row["start_row"])
        count = int(row["row_count"])
        proprio = np.asarray(corpus.proprio[start : start + count], dtype=np.float32)
        episodes.append(
            RecoveryMemoryEpisode(
                episode_index=int(row["episode_index"]),
                base_snapshot_hash=str(row["base_snapshot_hash"]),
                initial_snapshot_hash=initial_hash,
                initial_signature=np.concatenate(
                    (
                        proprio[0, 6:35] / 0.030,
                        proprio[0, 35:64] / 0.004,
                    )
                ).astype(np.float32),
                proprioception=proprio,
                absolute_motor_targets_rad=np.asarray(
                    corpus.absolute_motor_targets_rad[start : start + count],
                    dtype=np.float32,
                ),
            )
        )
    if not episodes:
        raise ValueError("recovery associative memory has no successful episodes")
    hashes = [episode.initial_snapshot_hash for episode in episodes]
    if len(hashes) != len(set(hashes)):
        raise ValueError("recovery associative memory contains duplicate initial states")
    return tuple(episodes)


class RecoveryAssociativeMemory:
    """Stateful, current-proprioception feedback over successful episodes."""

    def __init__(
        self,
        episodes: tuple[RecoveryMemoryEpisode, ...],
        *,
        config: RecoveryAssociativeMemoryConfig | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("recovery associative memory cannot be empty")
        self.episodes = episodes
        self.config = config or RecoveryAssociativeMemoryConfig()
        self._candidates: tuple[RecoveryMemoryEpisode, ...] = ()
        self._initial_distances = np.empty(0, dtype=np.float32)
        self._fixed: RecoveryMemoryEpisode | None = None
        self._last_indexes: tuple[int, ...] = ()
        self._last_weights = np.empty(0, dtype=np.float64)
        self._last_lookup_step = -1
        self.selection_switch_count = 0
        self.phase_adjustment_count = 0
        self.phase_hold_count = 0
        self.maximum_query_distance = 0.0
        self._phase_index = 0

    def reset(
        self,
        snapshot: RecoverySnapshot,
        *,
        base_snapshot_hash: str,
        corpus: RecoveryDistillationCorpus,
    ) -> None:
        candidates = tuple(
            episode
            for episode in self.episodes
            if episode.base_snapshot_hash == base_snapshot_hash
        )
        if not candidates:
            raise ValueError("recovery associative memory has no matching base skill")
        exact = tuple(
            episode
            for episode in candidates
            if episode.initial_snapshot_hash == snapshot.snapshot_hash
        )
        query = recovery_initial_signature(snapshot, corpus)
        self._candidates = candidates
        self._initial_distances = np.asarray(
            [
                float(np.sqrt(np.mean(np.square(query - episode.initial_signature))))
                for episode in candidates
            ],
            dtype=np.float32,
        )
        self._fixed = exact[0] if exact and self.config.exact_match_fixed_replay else None
        self._last_indexes = ()
        self._last_weights = np.empty(0, dtype=np.float64)
        self._last_lookup_step = -1
        self.selection_switch_count = 0
        self.phase_adjustment_count = 0
        self.phase_hold_count = 0
        self.maximum_query_distance = float(np.min(self._initial_distances))
        self._phase_index = 0

    def target(
        self,
        step: int,
        proprioception: NDArray[np.floating[Any]],
    ) -> NDArray[np.float32]:
        if step < 0 or not self._candidates:
            raise ValueError("recovery associative memory must be reset before lookup")
        if self._fixed is not None:
            index = min(step, self._fixed.absolute_motor_targets_rad.shape[0] - 1)
            return np.asarray(self._fixed.absolute_motor_targets_rad[index], dtype=np.float32)
        refresh = bool(
            not self._last_indexes
            or (
                self.config.dynamic_retrieval
                and step - self._last_lookup_step >= self.config.retrieval_interval_steps
            )
        )
        if refresh:
            query = recovery_feedback_signature(proprioception)
            distances = []
            for initial_distance, episode in zip(
                self._initial_distances, self._candidates, strict=True
            ):
                index = min(step, episode.proprioception.shape[0] - 1)
                feedback = recovery_feedback_signature(episode.proprioception[index])
                distance = float(np.sqrt(np.mean(np.square(query - feedback))))
                distances.append(
                    distance
                    + self.config.initial_distance_weight * float(initial_distance)
                )
            distance_array = np.asarray(distances, dtype=np.float64)
            count = min(self.config.nearest_neighbors, len(self._candidates))
            indexes = np.argsort(distance_array, kind="stable")[:count]
            selected = tuple(int(index) for index in indexes)
            if self._last_indexes and selected != self._last_indexes:
                self.selection_switch_count += 1
            self._last_indexes = selected
            self._last_lookup_step = step
            self.maximum_query_distance = max(
                self.maximum_query_distance,
                float(distance_array[indexes[0]]),
            )
            logits = -distance_array[indexes] / self.config.softmax_temperature
            logits -= np.max(logits)
            self._last_weights = np.exp(logits)
            self._last_weights /= np.sum(self._last_weights)
        indexes = np.asarray(self._last_indexes, dtype=np.int64)
        weights = self._last_weights
        target_steps = [
            min(step, self._candidates[int(index)].absolute_motor_targets_rad.shape[0] - 1)
            for index in indexes
        ]
        if self.config.phase_search_forward_steps > 0:
            episode = self._candidates[int(indexes[0])]
            query = recovery_feedback_signature(proprioception)
            center = self._phase_index
            lower = max(0, center - self.config.phase_search_back_steps)
            upper = min(
                episode.proprioception.shape[0] - 1,
                center + self.config.phase_search_forward_steps,
            )
            candidates = np.arange(lower, upper + 1, dtype=np.int64)
            phase_distances = np.asarray(
                [
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    query
                                    - recovery_feedback_signature(
                                        episode.proprioception[int(candidate)]
                                    )
                                )
                            )
                        )
                    )
                    + self.config.phase_deviation_penalty
                    * abs(int(candidate) - center)
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            selected_phase = int(candidates[int(np.argmin(phase_distances))])
            target_steps[0] = selected_phase
            next_phase = max(center, selected_phase + 1)
            self.phase_adjustment_count += int(selected_phase != step)
            self.phase_hold_count += int(next_phase == center)
            self._phase_index = next_phase
        targets = np.stack(
            [
                self._candidates[int(index)].absolute_motor_targets_rad[
                    target_step
                ]
                for index, target_step in zip(indexes, target_steps, strict=True)
            ]
        )
        result: NDArray[np.float32] = np.asarray(
            np.sum(targets * weights[:, None], axis=0, dtype=np.float64),
            dtype=np.float32,
        )
        return result


__all__ = [
    "RecoveryAssociativeMemory",
    "RecoveryAssociativeMemoryConfig",
    "RecoveryMemoryEpisode",
    "build_recovery_memory_episodes",
    "recovery_feedback_signature",
    "recovery_initial_signature",
]
