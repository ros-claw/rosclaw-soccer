"""Synthetic arithmetic contracts, not physical or data-admission evidence."""

import copy
import json

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.actual_winner_batch import prepare_actual_winner_batch
from rosclaw_soccer.rsi.actual_winner_supervision import select_actual_winner_paths
from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES
from rosclaw_soccer.rsi.reference_student_execution import make_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts import rsi_fit_actual_winner_batch as fitter
from scripts.rsi_load_actual_winner_exam import require_actual_fit


def bank():
    groups = (12, 13, 28, 29, 76, 77, 92, 93)

    def rows(role, passed):
        return [
            dict(
                group=g,
                role=role,
                receipt_hash=f"{role}-{g}",
                outcome=dict(task_pass=g in passed, safe=True),
            )
            for g in groups
        ]

    return dict(
        parent_model_hash="sha256:" + "a" * 64,
        zeros=rows("zero", (12, 13, 92, 93)),
        teachers=rows("teacher", (12, 13, 76, 93)),
        zero_nonfoot={g: False for g in groups},
        teacher_nonfoot={g: False for g in groups},
    )


def inputs():
    rng = np.random.default_rng(714)
    zx, tx = [rng.normal(size=(8, 300, 135)) for _ in range(2)]
    for x in (zx, tx):
        x[:, :, -1] = (300 - np.arange(300)) / 300
    zy = np.zeros((8, 300, 12))
    ty = np.full_like(zy, 0.12)
    ty[:, :30] = 0
    return dict(
        zero_x=zx,
        zero_y=zy,
        teacher_x=tx,
        teacher_y=ty,
        selection=select_actual_winner_paths(**bank()),
    )


def test_zero_anchors_receive_positive_weight_and_own_inputs_remain_exact():
    kw = inputs()
    batch, report = prepare_actual_winner_batch(**kw)
    assert (report["train_paths"], report["validation_paths"]) == (3, 2)
    assert report["train_positive_weight_frames"] == 810
    assert batch["original_observations"][::2].tobytes() == kw["zero_x"].tobytes()
    assert batch["original_observations"][1::2].tobytes() == kw["teacher_x"].tobytes()
    assert np.all(batch["weights"][[0, 2], 30:] == 1)
    assert np.all(batch["targets"][[0, 2]] == 0)
    assert np.all(batch["weights"][9, 30:] == 5)
    assert np.all(batch["weights"][[1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]] == 0)
    assert not report["neural_training_authorized"]


def test_validation_and_failed_inputs_do_not_change_normalization():
    kw = inputs()
    before, _ = prepare_actual_winner_batch(**kw)
    kw["zero_x"][6:, :, :-1] += 100
    kw["teacher_x"][:4, :, :-1] -= 100
    after, _ = prepare_actual_winner_batch(**kw)
    for k in ("mean", "scale", "weights", "train"):
        np.testing.assert_array_equal(before[k], after[k])


@pytest.mark.parametrize("tamper", ["nonzero_nav", "partial", "nan", "seal", "role", "query"])
def test_rejects_incomplete_or_counterfactual_data(tamper):
    kw = inputs()
    if tamper == "nonzero_nav":
        kw["zero_y"][0, 31, 0] = 0.001
    elif tamper == "partial":
        kw["teacher_x"] = kw["teacher_x"][:-1]
    elif tamper == "nan":
        kw["zero_x"][0, 0, 0] = np.nan
    elif tamper == "seal":
        kw["selection"]["rows"][0]["partition"] = "validation"
    else:
        kw["selection"]["rows"][0]["role" if tamper == "role" else "counterfactual_labels"] = (
            "teacher" if tamper == "role" else True
        )
        s = kw["selection"]
        s["report_hash"] = hash_json({k: v for k, v in s.items() if k != "report_hash"})
    with pytest.raises(ValueError):
        prepare_actual_winner_batch(**kw)


@pytest.fixture
def fit_inputs(tmp_path, monkeypatch):
    kw = inputs()
    batch, numeric = prepare_actual_winner_batch(**kw)
    path = tmp_path / "batch.npz"
    np.savez(
        path,
        actual_observations=batch["original_observations"],
        reference_targets=batch["targets"],
        partitions=batch["partitions"],
        source_seeds=batch["source_seeds"],
        eligible=batch["eligible"],
    )
    native, contact = tmp_path / "native.json", tmp_path / "contact.json"
    parent, world = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    native.write_text(
        json.dumps(dict(behavior_model_hash=parent, mesh_task_contact_world_hash=world))
    )
    contact.write_text(json.dumps(dict(producer_config=str(native))))
    prepared = dict(
        schema="soccer.rsi.actual_winner_batch_prepared.v1",
        configuration=dict(
            profile="proposed-actual-winner-zero-anchor-batch-v1",
            mode="SIM_ONLY",
            source_root=str(tmp_path),
            source_head="fake-test-head",
            zero_cohort=dict(contact_configuration=str(contact)),
        ),
        source_authentication_performed=True,
        original_task_labels_unchanged=True,
        input_file_hashes={
            str(native): fitter.digest(native),
            str(contact): fitter.digest(contact),
        },
        selection=kw["selection"],
        numeric_batch=numeric,
        numeric_arrays=dict(path=str(path), sha256=fitter.digest(path)),
        optimizer_updates=0,
        new_physical_steps=0,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    prepared["report_hash"] = hash_json(prepared)
    cfg = dict(
        prepared_unit="test.service",
        prepared_complete=str(tmp_path / "complete.json"),
        prepared_complete_hash=prepared["report_hash"],
        parent_model_hash=parent,
        world_hash=world,
    )
    monkeypatch.setattr(fitter, "require_terminal", lambda _: None)
    monkeypatch.setattr(
        fitter.subprocess,
        "check_output",
        lambda cmd, **_: "" if "status" in cmd else "fake-test-head",
    )
    return cfg, prepared, batch, native


def test_fit_loader_rebuilds_selection_split_and_numeric_batch(fit_inputs):
    cfg, prepared, expected, _ = fit_inputs
    before = copy.deepcopy(prepared)
    _, batch = fitter.load_batch(cfg, sealed=lambda _: prepared, pins={})
    assert prepared == before
    for key in ("mean", "scale", "weights", "train", "targets", "original_observations"):
        np.testing.assert_array_equal(batch[key], expected[key])


@pytest.mark.parametrize("tamper", ["seal", "unsafe", "world", "bytes", "weight", "updates"])
def test_fit_loader_rejects_provenance_or_selection_tampering(fit_inputs, tamper):
    cfg, prepared, _, native = fit_inputs
    if tamper == "seal":
        cfg["prepared_complete_hash"] = "other"
    elif tamper == "unsafe":
        prepared["selection"]["rows"][0]["original_outcome"]["safe"] = False
    elif tamper == "world":
        cfg["world_hash"] = "other"
    elif tamper == "bytes":
        native.write_text("changed")
    elif tamper == "weight":
        prepared["numeric_batch"]["train_paths"] = 9
    else:
        prepared["optimizer_updates"] = True
    with pytest.raises(ValueError):
        fitter.load_batch(cfg, sealed=lambda _: prepared, pins={})


@pytest.fixture
def candidate_lineage(fit_inputs):
    cfg, prepared, batch, _ = fit_inputs
    configuration = dict(
        cfg,
        profile="one-actual-winner-zero-anchor-fit-v1",
        fit_seed=2026101902,
        fit_steps=128,
        maximum_numeric_candidates=1,
    )
    initial = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
    ignored = [
        i
        for i, n in enumerate(FEATURE_NAMES)
        if n.startswith("previous_applied_reference_increment:")
    ]
    w = np.asarray(initial["weight_ih"])
    w[:, ignored] = 0
    initial["weight_ih"] = w.tolist()
    model = make_model(
        initial,
        mean=batch["mean"],
        scale=batch["scale"],
        world_hash=cfg["world_hash"],
        parent_model_hash=cfg["parent_model_hash"],
    )
    # Synthetic lineage only: no optimization or physical success in this test.
    fit = dict(
        report_hash="synthetic-fit",
        numeric_model=model,
        learning_receipt=dict(
            parameters=model["parameters"],
            fitted_parameters_hash=hash_json(model["parameters"]),
            completed_optimizer_steps=128,
            config=dict(
                steps=128,
                batch_episodes=4,
                learning_rate=0.0004,
                residual_cap=0.2,
                seed=2026101902,
                compute_device="cpu",
            ),
        ),
        normalization_hash=hash_json(
            dict(mean=batch["mean"].tolist(), scale=batch["scale"].tolist())
        ),
        whole_observations_hash=hash_bytes(batch["original_observations"].tobytes()),
        whole_reference_labels_hash=hash_bytes(batch["targets"].tobytes()),
        initial_parameters_hash=hash_json(initial),
        past_reference_input_ablation=True,
        ignored_input_weights_remain_exactly_zero=True,
        previous_reference_inputs_are_teacher_forced=False,
        ignored_input_feature_names=[FEATURE_NAMES[i] for i in ignored],
        input_episode_count=16,
        optimized_episode_count=3,
        train_source_seeds=[20261077, 20261093],
        validation_source_seeds=[20261096],
        validation_excluded_from_gradient_and_normalization=True,
    )
    stage = dict(
        schema="soccer.rsi.actual_winner_fit_complete.v1",
        status="NUMERIC_CANDIDATE_REQUIRES_OWN_BODY_EXAM",
        configuration=copy.deepcopy(configuration),
        numeric_fit_hash=fit["report_hash"],
        prepared_complete_hash=prepared["report_hash"],
        selection_hash=prepared["selection"]["report_hash"],
        optimizer_updates=128,
        own_body_student_evaluated=False,
        validation_checkpoint_selection=False,
        dataset_admission_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return stage, fit, dict(configuration=configuration, prepared=prepared, batch=batch)


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "steps",
        "world",
        "labels",
        "norm",
        "seed",
        "initial",
        "selection",
        "validation",
        "model",
        "flags",
    ],
)
def test_candidate_cannot_enter_exam_with_changed_training_lineage(candidate_lineage, tamper):
    stage, fit, kw = candidate_lineage
    if tamper == "steps":
        stage["optimizer_updates"] = 129
    elif tamper == "world":
        kw["configuration"]["world_hash"] = "other"
    elif tamper == "labels":
        fit["whole_reference_labels_hash"] = "other"
    elif tamper == "norm":
        kw["batch"]["mean"][0] += 0.01
    elif tamper == "seed":
        fit["learning_receipt"]["config"]["seed"] += 1
    elif tamper == "initial":
        fit["initial_parameters_hash"] = "other"
    elif tamper == "selection":
        stage["selection_hash"] = "other"
    elif tamper == "validation":
        fit["train_source_seeds"].append(20261096)
    elif tamper == "model":
        fit["learning_receipt"]["fitted_parameters_hash"] = "other"
    elif tamper == "flags":
        stage["promotion_authorized"] = True
    if tamper is None:
        assert require_actual_fit(stage, fit, **kw) == fit["numeric_model"]
    else:
        with pytest.raises(ValueError):
            require_actual_fit(stage, fit, **kw)
