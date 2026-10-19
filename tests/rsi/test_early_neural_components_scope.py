"""Six-body bounded component study, not full causal policy qualification."""

import pytest

from scripts.rsi_audit_early_neural_components import require_scope


def scope():
    return dict(
        profile="six-early-neural-components-v1",
        mode="SIM_ONLY",
        source_cases=6,
        planner_call_budget=90,
        executed_encoder_decoder_pairs=1800,
        new_physical_steps=0,
        new_optimizer_updates=0,
        additional_original_onnx_comparison_calls=True,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
        whole_controller_qualification_claimed=False,
    )


def test_exact_fixed_no_learning_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_domain(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)
