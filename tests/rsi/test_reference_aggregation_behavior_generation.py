"""Behaviour-source dispatch only; original provenance loader is mocked here."""

import pytest

from scripts import rsi_load_reference_ablation_exam as loader
from scripts.rsi_prepare_reference_state_aggregation import require_behavior_fit


@pytest.mark.parametrize("generation", ["original", "ablation"])
@pytest.mark.parametrize("tamper", [None, "fit", "stage", "model"])
def test_new_own_body_inputs_cannot_be_relabelled_as_original_student(
    monkeypatch, generation, tamper
):
    model = dict(model_hash="mock-current-model")
    calls = []

    def loaded(producer, **kw):
        calls.append((producer, kw))
        return model, dict(report_hash="current-stage"), dict(report_hash="current-fit")

    monkeypatch.setattr(loader, "load_ablation", loaded)
    declaration = dict(
        fit_report_hash="current-fit" if generation == "ablation" else "original-fit",
        fit_stage_hash="current-stage" if generation == "ablation" else "original-stage",
        models=dict(student=model),
    )
    if tamper == "fit":
        declaration["fit_report_hash"] = "borrowed-fit"
    elif tamper == "stage":
        declaration["fit_stage_hash"] = "borrowed-stage"
    elif tamper == "model":
        declaration["models"]["student"] = dict(model_hash="borrowed-model")
    profile = (
        f"proposed-original16-{'ablation-' if generation == 'ablation' else ''}own13-aggregation-v1"
    )

    def run():
        require_behavior_fit(
            profile,
            {},
            declaration,
            original_stage=dict(report_hash="original-stage"),
            original_fit=dict(report_hash="original-fit", numeric_model=model),
            sealed=lambda _: None,
            pins={},
        )

    if tamper:
        with pytest.raises(ValueError):
            run()
    else:
        run()
    assert len(calls) == (1 if generation == "ablation" else 0)


def test_unknown_generation_cannot_silently_use_a_previous_fit():
    with pytest.raises(ValueError, match="explicit original"):
        require_behavior_fit(
            "future", {}, {}, original_stage={}, original_fit={}, sealed=lambda _: None, pins={}
        )
