from __future__ import annotations

from collections import Counter
from dataclasses import replace

from test_s49_recovery_snapshot import _snapshot

from rosclaw_soccer.training.recovery_curriculum import RecoveryReplaySampler


def test_recovery_replay_is_deterministic_and_prioritizes_failures() -> None:
    source = tuple(_snapshot(environment_index=index) for index in range(6))
    source += (
        _snapshot(environment_index=6, stage="FAILURE_TERMINAL"),
    )
    source = source[:-1] + (
        replace(source[-1], failed=True, posture_cluster="PRONE"),
    )
    sampler = RecoveryReplaySampler(snapshots=source)
    first = sampler.sample_indices(batch_size=4_000, seed=49_101)
    second = sampler.sample_indices(batch_size=4_000, seed=49_101)
    assert (first == second).all()
    counts = Counter(int(index) for index in first)
    assert counts[6] > sum(counts[index] for index in range(6)) / 6
    assert sampler.summary()["activation_ceiling"] == "SIM_ONLY"
