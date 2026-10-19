"""Synthetic raw reader tests, not physical or neural qualification."""

import copy
import json
import sys

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_parity import SHAPES
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_full_task_zero as mod
from scripts.rsi_read_full_task_zero import review_case
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_full_task_branch_qualification import scope


def fixture():
    main = {}
    for key, shape in SHAPES.items():
        dtype = (
            np.bool_
            if key == "recurrent_sampling_performed"
            else np.float32
            if key.startswith("foundation_neural_")
            else np.float64
        )
        main[key] = np.zeros((300, 1, *shape), dtype=dtype)
    main.update({PREFIX + k: np.zeros((300, 1, w)) for k, w in FIELDS.items()})
    main["force_n"][185, 0, 0] = 2
    main["pelvis_z_per_substep_m"][:] = 0.79
    outcome = label_first_touch(main).to_dict()
    predictions = []
    data = {"main": main, "gold": copy.deepcopy(main)}
    for bi in range(2):
        data[str(bi)] = {k: a[186:].copy() for k, a in main.items()}
        predictions.append(
            dict(
                index=0,
                branch_index=bi,
                start_frame=186,
                persisted_before_main_advances_frame=186,
                source_future_used_for_prediction=False,
                predicted_outcome=outcome,
                arrays=str(bi),
                review=dict(
                    schema="soccer.rsi.full_task_branch_review.v1",
                    action_name="baseline",
                    start_frame=186,
                    end_exclusive=300,
                    borrowed_prefix_frames=186,
                    new_native_attempted_steps=1140,
                    new_native_integrated_steps=1140,
                    recomputed_control_frames=114,
                    recomputed_foundation_encoder_decoder_pairs=114,
                    new_planner_onnx_calls=5,
                    donor_integration_hash="same-donor",
                    future_observations_from_own_simulation=True,
                    recorded_future_targets_or_states_consumed=False,
                    prefix_steps_counted_as_new=False,
                    new_optimizer_updates=0,
                    teacher_qualified=False,
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
            )
        )
    row = dict(
        index=0,
        course=[2.5, 0.1, 0.3],
        source_case_hash="gold-case",
        arrays="main",
        outcome=outcome,
        predictions=predictions,
        main_review=dict(native_attempted_steps=3000, native_integrated_steps=3000),
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    job = dict(
        index=0,
        course=row["course"],
        source_case_hash="gold-case",
        gold_arrays="gold",
        comparison_start_after_measured_foot_bin=186,
    )
    return row, job, data


def test_every_tensor_and_label_recomputed():
    row, job, data = fixture()
    report = review_case(row, job, data.__getitem__)
    assert report["new_main_steps"] == 3000
    assert report["new_prediction_steps"] == 2280
    assert report["predicted_planner_calls"] == 10


@pytest.mark.parametrize(
    "change",
    [
        "future_tensor",
        "missing_tensor",
        "dtype",
        "label",
        "gold",
        "count",
        "refresh",
        "donor",
        "promotion",
        "recorded_future",
        "prefix",
        "neural_count",
    ],
)
def test_forged_claims_and_one_bit_future_errors_rejected(change):
    row, job, data = fixture()
    r = row["predictions"][1]["review"]
    if change == "future_tensor":
        data["1"]["joint_target_rad"][0, 0, 0] = np.nextafter(0.0, 1.0)
    elif change == "missing_tensor":
        del data["1"]["canonical_qpos"]
    elif change == "dtype":
        data["1"]["joint_target_rad"] = data["1"]["joint_target_rad"].astype(np.float32)
    elif change == "label":
        row["predictions"][1]["predicted_outcome"] = dict(row["outcome"], task_pass=True)
    elif change == "gold":
        data["gold"]["joint_target_rad"][0, 0, 0] = 1
    elif change == "count":
        r["new_native_integrated_steps"] = 3000
    elif change == "refresh":
        r["new_planner_onnx_calls"] = 4
    elif change == "donor":
        r["donor_integration_hash"] = "other"
    elif change == "promotion":
        row["promotion_authorized"] = True
    elif change == "recorded_future":
        r["recorded_future_targets_or_states_consumed"] = True
    elif change == "prefix":
        r["prefix_steps_counted_as_new"] = True
    elif change == "neural_count":
        r["recomputed_foundation_encoder_decoder_pairs"] = 300
    with pytest.raises(ValueError):
        review_case(row, job, data.__getitem__)


@pytest.mark.parametrize("promote", [False, True])
def test_whole18_sealed_readback_or_reject_promotion(tmp_path, monkeypatch, promote):
    root = tmp_path / "producer"
    root.mkdir()

    def sealed(name, value):
        value = dict(value)
        value["report_hash"] = hash_json(value)
        (root / name).write_text(json.dumps(value))
        return value

    def arrays(name, values):
        path = root / name
        np.savez_compressed(path, **values)
        return dict(path=str(path), sha256=digest(path))

    rows, jobs = [], []
    feet = [
        185,
        183,
        184,
        183,
        151,
        148,
        151,
        148,
        235,
        209,
        234,
        209,
        184,
        184,
        148,
        148,
        228,
        228,
    ]
    for i, foot in enumerate(feet):
        row, job, data = fixture()
        main = data["main"]
        main["force_n"][:] = 0
        main["force_n"][foot, 0, 0] = 2
        start = foot + 1
        outcome = label_first_touch(main).to_dict()
        row.update(index=i, outcome=outcome, arrays=arrays(f"main-{i}.npz", main))
        job.update(
            index=i,
            gold_arrays=arrays(f"gold-{i}.npz", main),
            comparison_start_after_measured_foot_bin=start,
        )
        for bi, pred in enumerate(row["predictions"]):
            pred.update(
                index=i,
                start_frame=start,
                predicted_outcome=outcome,
                persisted_before_main_advances_frame=start,
                arrays=arrays(f"future-{i}-{bi}.npz", {k: a[start:] for k, a in main.items()}),
            )
            pred["review"].update(
                start_frame=start,
                borrowed_prefix_frames=start,
                new_native_attempted_steps=(300 - start) * 10,
                new_native_integrated_steps=(300 - start) * 10,
                recomputed_control_frames=300 - start,
                recomputed_foundation_encoder_decoder_pairs=300 - start,
                new_planner_onnx_calls=sum(f % 20 == 0 for f in range(start, 300)),
            )
            row["predictions"][bi] = sealed(f"case-{i}-branch-{bi}-prediction.json", pred)
        rows.append(sealed(f"case-{i}-complete.json", row))
        jobs.append(job)
    declaration = sealed(
        "declaration.json",
        dict(
            schema="soccer.rsi.full_task_zero18_declaration.v1",
            config=dict(scope(), output_root=str(root)),
            jobs=jobs,
            input_file_hashes={},
        ),
    )
    sealed(
        "complete.json",
        dict(
            schema="soccer.rsi.full_task_zero18_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            new_main_native_steps=54000,
            new_prediction_native_steps=40840,
            combined_new_native_steps=94840,
            main_reconstruction_runs=18,
            prediction_runs=36,
            executed_control_foundation_pairs=9484,
            initialization_foundation_parity_pairs=4,
            new_optimizer_updates=0,
            baseline_whole_task_prediction_qualified_on_this_consumed_scope=True,
            nonzero_action_prediction_qualified=False,
            teacher_qualified=promote,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    output = tmp_path / "review.json"
    monkeypatch.setattr(mod, "require_terminal", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "--root",
            str(root),
            "--unit",
            "fake",
            "--declaration-hash",
            declaration["report_hash"],
            "--output",
            str(output),
        ],
    )
    if promote:
        with pytest.raises(ValueError):
            mod.main()
        assert not output.exists()
    else:
        mod.main()
        result = json.loads(output.read_text())
        assert result["new_physical_steps"] == 0
        assert len(result["rows"]) == 18
        assert result["producer_steps_reviewed"]["combined_new_native_steps"] == 94840
        assert result["temporal_execution_order_independently_attested"] is False
