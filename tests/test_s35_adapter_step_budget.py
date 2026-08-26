from __future__ import annotations

from rosclaw_soccer.evidence.opentrack_adapter_train import (
    AdapterStepBudget,
    calculate_adapter_step_budget,
)


def _budget(maximum: int) -> AdapterStepBudget:
    return calculate_adapter_step_budget(
        sealed_maximum_world_steps=maximum,
        batch_size=512,
        unroll_length=20,
        num_minibatches=32,
        action_repeat=1,
        num_evals=0,
        num_resets_per_eval=0,
    )


def test_step_budget_exposes_s34_parallel_batch_overshoot() -> None:
    budget = _budget(20_000_000)

    assert budget.step_quantum == 327_680
    assert budget.expected_world_steps == 20_316_160
    assert not budget.aligned


def test_step_budget_accepts_exact_s35_contract_ceiling() -> None:
    budget = _budget(19_988_480)

    assert budget.expected_world_steps == 19_988_480
    assert budget.aligned
