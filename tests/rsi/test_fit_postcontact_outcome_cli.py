"""Fake closed physical bank and optimizer orchestration, not real fitting."""

import json
import sys

import numpy as np
import pytest

from scripts import rsi_fit_postcontact_outcome_models as mod
from tests.rsi.test_postcontact_outcome_dataset import synthetic_readback


@pytest.mark.parametrize("fail_seed", [None, 2027])
def test_three_seeds_two_group_folds_and_local_only_final(monkeypatch, tmp_path, fail_seed):
    def seal(path, value):
        value = dict(value)
        value["report_hash"] = mod.hash_json(value)
        path.write_text(json.dumps(value))
        return value

    rb = seal(tmp_path / "rb.json", dict(synthetic_readback(), complete_hash="bank"))
    root = tmp_path / "physics"
    root.mkdir()
    pd = seal(
        root / "declaration.json",
        dict(
            raw_readback_hash=rb["report_hash"], producer_complete_hash="bank", input_file_hashes={}
        ),
    )
    rows = []
    for i in range(252):
        ci, ai = divmod(i, 14)
        rows.append(
            seal(
                root / f"context-{ci}-action-{ai}-complete.json",
                dict(
                    context_index=ci,
                    action_index=ai,
                    physical_execution_reused=ci == 1,
                    physical_review=dict(
                        actual_physical_steps=3000, maximum_absolute_errors=dict(q=0.0)
                    ),
                ),
            )
        )
    seal(
        root / "complete.json",
        dict(
            schema="soccer.rsi.recorded_postcontact_full_bank_complete.v1",
            declaration_hash=pd["report_hash"],
            rows=rows,
            comparison_bodies_verified=252,
            new_physical_bodies=238,
            new_physical_steps=714000,
            reused_physical_bodies=14,
            reused_physical_steps=42000,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    output = tmp_path / "fit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--root",
            str(root),
            "--unit",
            "physics.service",
            "--readback",
            str(tmp_path / "rb.json"),
            "--readback-file-hash",
            mod.digest(tmp_path / "rb.json"),
            "--source-head",
            "fake",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        mod.subprocess, "check_output", lambda argv, **kw: "fake" if "rev-parse" in argv else ""
    )
    units = []
    monkeypatch.setattr(mod, "require_terminal", units.append)
    calls = []

    def fit(x, y, mask, *, seed, dataset_hash, training_context_indices):
        assert all(i < 12 for i in training_context_indices)
        assert x.shape == (len(training_context_indices), 121)
        assert y.shape == mask.shape == (len(x), 13, 2)
        if seed == fail_seed:
            raise RuntimeError("synthetic fitting failure")
        calls.append((seed, training_context_indices))
        return dict(
            report_hash=mod.hash_json([seed, training_context_indices]), optimizer_updates=500
        )

    monkeypatch.setattr(mod, "fit", fit)
    monkeypatch.setattr(mod, "predict", lambda model, x: np.full((len(x), 13, 2), 0.5))
    if fail_seed is not None:
        with pytest.raises(RuntimeError, match="synthetic fitting failure"):
            mod.main()
        failure = json.loads((output / "failure.json").read_text())
        assert (
            failure["completed_models"] == 3
            and failure["completed_model_optimizer_updates"] == 1500
        )
        assert (
            failure["current_fit_optimizer_updates_unreported"] and not failure["automatic_retry"]
        )
        assert not (output / "complete.json").exists()
    else:
        mod.main()
        result = json.loads((output / "complete.json").read_text())
        assert result["models_fitted"] == 9 and result["optimizer_updates"] == 4500
        assert result["physical_steps"] == 0 and not result["fitted_model_is_motor_actor"]
        assert result["observed_local_training_states"] == 8
        assert result["original_clusters_with_training_observations"] == 2
        for seed in mod.SEEDS:
            assert (seed, list(range(4))) in calls
            assert (seed, list(range(4, 8))) in calls
            assert (seed, list(range(8))) in calls
        declaration = json.loads((output / "declaration.json").read_text())
        assert declaration["retention_contexts"] == list(range(12, 18))
        assert len(declaration["all18_original_candidate_rows"]) == 18
        assert declaration["optimizer_update_budget"] == 4500
    assert units == ["physics.service"]
