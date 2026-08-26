from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.training.recovery_associative_memory import (
    RecoveryAssociativeMemory,
    RecoveryAssociativeMemoryConfig,
    RecoveryMemoryEpisode,
    recovery_feedback_signature,
)

_BASE_A = "sha256:" + "a" * 64
_BASE_B = "sha256:" + "b" * 64
_INITIAL_A = "sha256:" + "c" * 64
_INITIAL_B = "sha256:" + "d" * 64
_QUERY = "sha256:" + "e" * 64


def _episode(
    index: int,
    *,
    base_hash: str,
    initial_hash: str,
    joint_offset: float,
    target: float,
) -> RecoveryMemoryEpisode:
    proprio = np.zeros((3, 93), dtype=np.float32)
    proprio[:, 6] = joint_offset
    signature = np.zeros(58, dtype=np.float32)
    signature[0] = joint_offset / 0.030
    return RecoveryMemoryEpisode(
        episode_index=index,
        base_snapshot_hash=base_hash,
        initial_snapshot_hash=initial_hash,
        initial_signature=signature,
        proprioception=proprio,
        absolute_motor_targets_rad=np.full((3, 29), target, dtype=np.float32),
    )


def _snapshot(snapshot_hash: str, joint_offset: float) -> SimpleNamespace:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[3] = 1.0
    qpos[7] = joint_offset
    return SimpleNamespace(
        snapshot_hash=snapshot_hash,
        qpos=qpos,
        qvel=np.zeros(35, dtype=np.float32),
    )


def test_feedback_signature_rejects_invalid_and_ignores_last_target() -> None:
    frame = np.zeros(93, dtype=np.float32)
    changed = frame.copy()
    changed[64:] = 100.0
    assert recovery_feedback_signature(frame) == pytest.approx(
        recovery_feedback_signature(changed)
    )
    with pytest.raises(ValueError, match="93 finite"):
        recovery_feedback_signature(np.zeros(92))


def test_associative_memory_preserves_exact_episode_and_routes_by_base() -> None:
    exact = _episode(
        0,
        base_hash=_BASE_A,
        initial_hash=_INITIAL_A,
        joint_offset=0.0,
        target=0.10,
    )
    nearby = _episode(
        1,
        base_hash=_BASE_A,
        initial_hash=_INITIAL_B,
        joint_offset=0.03,
        target=0.20,
    )
    wrong_base = _episode(
        2,
        base_hash=_BASE_B,
        initial_hash=_QUERY,
        joint_offset=0.03,
        target=0.90,
    )
    corpus = SimpleNamespace(default_joint_position_rad=np.zeros(29, dtype=np.float32))
    memory = RecoveryAssociativeMemory(
        (exact, nearby, wrong_base),
        config=RecoveryAssociativeMemoryConfig(nearest_neighbors=1),
    )

    memory.reset(
        _snapshot(_INITIAL_A, 0.0),
        base_snapshot_hash=_BASE_A,
        corpus=corpus,
    )
    assert memory.target(2, np.full(93, 50.0)) == pytest.approx(np.full(29, 0.10))

    query = _snapshot(_QUERY, 0.03)
    memory.reset(query, base_snapshot_hash=_BASE_A, corpus=corpus)
    frame = np.zeros(93, dtype=np.float32)
    frame[6] = 0.03
    assert memory.target(0, frame) == pytest.approx(np.full(29, 0.20))
    assert memory.selection_switch_count == 0


def test_associative_memory_fails_closed_without_matching_base() -> None:
    episode = _episode(
        0,
        base_hash=_BASE_A,
        initial_hash=_INITIAL_A,
        joint_offset=0.0,
        target=0.10,
    )
    memory = RecoveryAssociativeMemory((episode,))
    corpus = SimpleNamespace(default_joint_position_rad=np.zeros(29, dtype=np.float32))
    with pytest.raises(ValueError, match="matching base"):
        memory.reset(
            _snapshot(_QUERY, 0.0),
            base_snapshot_hash=_BASE_B,
            corpus=corpus,
        )


def test_associative_memory_retrieval_interval_prevents_per_step_chatter() -> None:
    first = _episode(
        0,
        base_hash=_BASE_A,
        initial_hash=_INITIAL_A,
        joint_offset=0.0,
        target=0.10,
    )
    second = _episode(
        1,
        base_hash=_BASE_A,
        initial_hash=_INITIAL_B,
        joint_offset=0.03,
        target=0.20,
    )
    corpus = SimpleNamespace(default_joint_position_rad=np.zeros(29, dtype=np.float32))
    memory = RecoveryAssociativeMemory(
        (first, second),
        config=RecoveryAssociativeMemoryConfig(
            nearest_neighbors=1,
            retrieval_interval_steps=10,
        ),
    )
    memory.reset(_snapshot(_QUERY, 0.0), base_snapshot_hash=_BASE_A, corpus=corpus)
    first_frame = np.zeros(93, dtype=np.float32)
    second_frame = first_frame.copy()
    second_frame[6] = 0.03
    assert memory.target(0, first_frame) == pytest.approx(np.full(29, 0.10))
    assert memory.target(1, second_frame) == pytest.approx(np.full(29, 0.10))
    assert memory.target(10, second_frame) == pytest.approx(np.full(29, 0.20))
    assert memory.selection_switch_count == 1


def test_associative_memory_estimates_internal_phase_from_proprioception() -> None:
    proprio = np.zeros((3, 93), dtype=np.float32)
    proprio[:, 6] = np.asarray((0.0, 0.03, 0.06), dtype=np.float32)
    targets = np.stack(
        [np.full(29, value, dtype=np.float32) for value in (0.10, 0.20, 0.30)]
    )
    episode = RecoveryMemoryEpisode(
        episode_index=0,
        base_snapshot_hash=_BASE_A,
        initial_snapshot_hash=_INITIAL_A,
        initial_signature=np.zeros(58, dtype=np.float32),
        proprioception=proprio,
        absolute_motor_targets_rad=targets,
    )
    corpus = SimpleNamespace(default_joint_position_rad=np.zeros(29, dtype=np.float32))
    memory = RecoveryAssociativeMemory(
        (episode,),
        config=RecoveryAssociativeMemoryConfig(
            nearest_neighbors=1,
            dynamic_retrieval=False,
            phase_search_back_steps=1,
            phase_search_forward_steps=2,
        ),
    )
    memory.reset(_snapshot(_QUERY, 0.0), base_snapshot_hash=_BASE_A, corpus=corpus)
    frame = np.zeros(93, dtype=np.float32)
    frame[6] = 0.03
    assert memory.target(0, frame) == pytest.approx(np.full(29, 0.20))
    assert memory.phase_adjustment_count == 1


def test_associative_memory_is_simulation_only() -> None:
    with pytest.raises(ValueError, match="invalid"):
        RecoveryAssociativeMemoryConfig(hardware_authorized=True)
