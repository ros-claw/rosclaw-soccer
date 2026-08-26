from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_corrective_scale import (
    validate_recovery_corrective_frozen_exam_evidence,
    write_recovery_corrective_frozen_exam_evidence,
)
from rosclaw_soccer.training.recovery_corrective_student import (
    CorrectiveTemporalLeaseConfig,
    RecoveryCorrectiveStudentConfig,
    attach_corrective_temporal_lease,
    attach_corrective_veto_aware_temporal_trigger,
    calibrate_corrective_channel_veto,
    calibrate_corrective_confidence_gate,
    corrective_stability_retention,
    derive_corrective_channel_gain,
    derive_corrective_effect_budget_gain,
    fit_corrective_channel_veto,
    fit_corrective_confidence_gate,
    fit_corrective_historical_veto_gate,
    initial_corrective_temporal_gate_state,
    mine_corrective_temporal_hard_negatives,
    mix_corrective_cross_domain_normal_replay,
    mix_corrective_normal_dagger_replay,
    mix_corrective_training_normal_sources,
    predict_corrective_channel_veto_numpy,
    predict_corrective_confidence_numpy,
    predict_corrective_raw_numpy,
    predict_corrective_student_numpy,
    step_corrective_temporal_gate_numpy,
    stratified_source_split,
    validate_recovery_corrective_repeat_evidence,
    validate_recovery_corrective_student_evidence,
    write_recovery_corrective_repeat_evidence,
    write_recovery_corrective_student_evidence,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _lineage() -> dict[str, str]:
    names = (
        "teacher_report_hash",
        "teacher_report_file_hash",
        "teacher_corpus_hash",
        "failure_state_manifest_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    return {name: _digest(format(index + 1, "x")) for index, name in enumerate(names)}


def _inputs(
    config: RecoveryCorrectiveStudentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source_count = 12
    observation_dim = 10
    random = np.random.default_rng(56)
    control_steps = np.repeat(np.asarray((200, 300, 400), dtype=np.int32), 4)
    train, holdout = stratified_source_split(
        control_steps, holdout_per_window=1, random_seed=config.random_seed
    )
    corpus = {
        "failure_observation": random.normal(
            size=(source_count, config.trace_steps, observation_dim)
        ).astype(np.float32),
        "failure_parent_action": np.zeros((source_count, config.trace_steps, 29), dtype=np.float32),
        "failure_target_increment": np.full(
            (source_count, config.trace_steps, 29), 0.01, dtype=np.float32
        ),
        "failure_state_index": np.arange(source_count, dtype=np.int32),
        "failure_control_step": control_steps,
        "normal_observation": random.normal(
            size=(source_count, config.normal_sample_count_per_route, observation_dim)
        ).astype(np.float32),
        "normal_parent_action": np.zeros(
            (source_count, config.normal_sample_count_per_route, 29), dtype=np.float32
        ),
        "train_source_mask": train,
        "holdout_source_mask": holdout,
    }
    hidden_0, hidden_1 = config.hidden_sizes
    model = {
        "observation_mean": np.zeros((observation_dim,), dtype=np.float32),
        "observation_scale": np.ones((observation_dim,), dtype=np.float32),
        "hidden_0_weight": np.zeros((observation_dim, hidden_0), dtype=np.float32),
        "hidden_0_bias": np.zeros((hidden_0,), dtype=np.float32),
        "hidden_1_weight": np.zeros((hidden_0, hidden_1), dtype=np.float32),
        "hidden_1_bias": np.zeros((hidden_1,), dtype=np.float32),
        "output_weight": np.zeros((hidden_1, 29), dtype=np.float32),
        "output_bias": np.zeros((29,), dtype=np.float32),
    }
    return corpus, model


def test_corrective_student_config_and_split_are_fail_closed() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    steps = np.repeat(np.asarray((200, 300), dtype=np.int32), 4)
    train, holdout = stratified_source_split(steps, holdout_per_window=1, random_seed=56)
    assert np.array_equal(train, ~holdout)
    assert np.sum(holdout) == 2
    assert {int(value) for value in steps[holdout]} == {200, 300}
    assert config.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="invalid"):
        RecoveryCorrectiveStudentConfig(required_gpu_count=2)
    with pytest.raises(ValueError, match="disjoint"):
        stratified_source_split(
            np.repeat(np.asarray((200, 300), dtype=np.int32), 4),
            holdout_per_window=4,
            random_seed=56,
        )


def test_zero_output_head_is_exactly_quiet() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    prediction = predict_corrective_student_numpy(
        model, corpus["failure_observation"], maximum_increment=config.maximum_action_increment
    )
    assert np.array_equal(prediction, np.zeros_like(prediction))

    model["output_bias"] = np.ones((29,), dtype=np.float32)
    full = predict_corrective_student_numpy(
        model, corpus["failure_observation"], maximum_increment=config.maximum_action_increment
    )
    model["output_gain"] = np.asarray((0.1,), dtype=np.float32)
    bounded = predict_corrective_student_numpy(
        model, corpus["failure_observation"], maximum_increment=config.maximum_action_increment
    )
    assert np.allclose(bounded, 0.1 * full, rtol=1.0e-6, atol=1.0e-7)


def test_channel_gain_keeps_only_data_supported_beneficial_actuators() -> None:
    jacobian = np.zeros((4, 4, 29), dtype=np.float32)
    prediction = np.ones((4, 29), dtype=np.float32)
    failure_trace = np.ones((4, 3, 29), dtype=np.float32)
    normal_trace = np.ones((4, 3, 29), dtype=np.float32)
    jacobian[:, 0, 2] = -1.0
    jacobian[:, 0, 7] = -0.5
    jacobian[:, 0, 9] = 1.0
    normal_trace[:, :, 2] = 0.1
    normal_trace[:, :, 7] = 0.5
    gain, report = derive_corrective_channel_gain(
        action_effect_jacobian=jacobian,
        failure_prediction=prediction,
        failure_trace_prediction=failure_trace,
        normal_trace_prediction=normal_trace,
        channel_count=2,
        active_gain=0.16,
    )
    assert np.flatnonzero(gain).tolist() == [2, 7]
    assert gain[9] == 0.0
    assert report["selected_joint_indices"] == [2, 7]
    assert (
        report["failure_normal_discrimination_ratio"][2]
        > report["failure_normal_discrimination_ratio"][7]
    )
    conservative_gain, conservative_report = derive_corrective_channel_gain(
        action_effect_jacobian=jacobian,
        failure_prediction=prediction,
        failure_trace_prediction=failure_trace,
        normal_trace_prediction=normal_trace,
        channel_count=8,
        active_gain=0.16,
        allow_fewer_beneficial=True,
    )
    assert np.flatnonzero(conservative_gain).tolist() == [2, 7]
    assert conservative_report["requested_maximum_channel_count"] == 8
    assert conservative_report["selected_channel_count"] == 2


def test_effect_budget_reduces_harmful_channels_without_breaking_mirror_pairs() -> None:
    jacobian = np.zeros((4, 4, 29), dtype=np.float32)
    failure = np.ones((4, 29), dtype=np.float32)
    historical = np.full((8, 3, 29), 0.1, dtype=np.float32)
    jacobian[:, 0, 2] = -1.0
    jacobian[:, 0, 3] = 0.7
    jacobian[:, 0, 9] = -0.2
    jacobian[:, 0, 12] = 0.5
    gain, report = derive_corrective_effect_budget_gain(
        action_effect_jacobian=jacobian,
        failure_prediction=failure,
        historical_normal_prediction=historical,
        base_gain=np.asarray((0.8,), dtype=np.float32),
        mirrored_channel_pairs=((3, 9),),
        minimum_gain_fraction=0.5,
    )
    assert gain.shape == (29,)
    assert gain[2] == pytest.approx(0.8)
    assert gain[3] == pytest.approx(0.4)
    assert gain[9] == pytest.approx(0.4)
    assert gain[12] == pytest.approx(0.4)
    assert np.all(gain <= 0.8)
    assert report["attenuated_joint_indices"] == [3, 9, 12]
    assert report["authority_monotonicity"].startswith("DERIVED_GAIN_CAN_ONLY")

    risk_weighted_historical = historical.copy()
    risk_weighted_historical[..., 3] = 0.9
    risk_weighted_historical[..., 9] = 0.8
    risk_weighted_historical[..., 12] = 0.05
    adaptive_gain, adaptive_report = derive_corrective_effect_budget_gain(
        action_effect_jacobian=jacobian,
        failure_prediction=failure,
        historical_normal_prediction=risk_weighted_historical,
        base_gain=np.asarray((0.8,), dtype=np.float32),
        mirrored_channel_pairs=((3, 9),),
        minimum_gain_fraction=0.5,
        maximum_gain_fraction=0.7,
    )
    assert adaptive_gain[3] == pytest.approx(0.4)
    assert adaptive_gain[9] == pytest.approx(0.4)
    assert adaptive_gain[12] == pytest.approx(0.56)
    assert adaptive_report["maximum_gain_fraction"] == pytest.approx(0.7)
    assert "RISK_WEIGHTED" in adaptive_report["algorithm"]

    with pytest.raises(ValueError, match="mirror pairs"):
        derive_corrective_effect_budget_gain(
            action_effect_jacobian=jacobian,
            failure_prediction=failure,
            historical_normal_prediction=historical,
            base_gain=0.8,
            mirrored_channel_pairs=((3, 9), (9, 10)),
        )


def test_normal_dagger_replay_is_balanced_and_source_aligned() -> None:
    parent_observation = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    candidate_observation = parent_observation + 1_000.0
    parent_action = np.zeros((2, 4, 29), dtype=np.float32)
    candidate_action = np.full((2, 4, 29), 0.25, dtype=np.float32)
    observation, action, candidate_mask = mix_corrective_normal_dagger_replay(
        parent_observation=parent_observation,
        parent_action=parent_action,
        candidate_observation=candidate_observation,
        candidate_parent_action=candidate_action,
    )
    assert candidate_mask.tolist() == [False, True, False, True]
    assert observation[:, 0] == pytest.approx(parent_observation[:, 0])
    assert observation[:, 1] == pytest.approx(candidate_observation[:, 1])
    assert observation[:, 2] == pytest.approx(parent_observation[:, 2])
    assert observation[:, 3] == pytest.approx(candidate_observation[:, 3])
    assert action[:, 0::2] == pytest.approx(0.0)
    assert action[:, 1::2] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="invalid"):
        mix_corrective_normal_dagger_replay(
            parent_observation=parent_observation[:, :3],
            parent_action=parent_action[:, :3],
            candidate_observation=candidate_observation[:, :3],
            candidate_parent_action=candidate_action[:, :3],
        )


def test_training_normal_source_mix_never_touches_current_holdout() -> None:
    current_observation = np.arange(8 * 4 * 3, dtype=np.float32).reshape((8, 4, 3))
    current_action = np.zeros((8, 4, 29), dtype=np.float32)
    train = np.asarray((True, False, True, True, False, True, True, True), dtype=np.bool_)
    frozen_observation = np.full((3, 4, 3), 999.0, dtype=np.float32)
    frozen_action = np.full((3, 4, 29), 0.25, dtype=np.float32)
    observation, action, frozen_mask = mix_corrective_training_normal_sources(
        current_observation=current_observation,
        current_parent_action=current_action,
        current_train_source_mask=train,
        frozen_training_observation=frozen_observation,
        frozen_training_parent_action=frozen_action,
    )
    assert np.sum(frozen_mask) == 3
    assert not np.any(frozen_mask & ~train)
    assert np.array_equal(observation[~train], current_observation[~train])
    assert np.array_equal(action[~train], current_action[~train])
    assert np.all(observation[frozen_mask] == 999.0)
    assert np.all(action[frozen_mask] == 0.25)


def test_temporal_hard_negative_mining_is_source_local_and_consecutive() -> None:
    observation = np.arange(2 * 8 * 3, dtype=np.float32).reshape((2, 8, 3))
    action = np.zeros((2, 8, 29), dtype=np.float32)
    confidence = np.asarray(
        (
            (0.1, 0.9, 0.8, 0.2, 0.7, 0.95, 0.6, 0.1),
            (0.95, 0.9, 0.1, 0.8, 0.7, 0.1, 0.6, 0.5),
        ),
        dtype=np.float32,
    )
    selected_observation, selected_action, selected_index, report = (
        mine_corrective_temporal_hard_negatives(
            observation=observation,
            parent_action=action,
            confidence=confidence,
            sample_count_per_source=4,
            consecutive_window_steps=2,
        )
    )
    assert selected_index.tolist() == [[1, 2, 4, 5], [0, 1, 3, 4]]
    assert selected_observation == pytest.approx(
        observation[np.arange(2, dtype=np.int32)[:, None], selected_index]
    )
    assert selected_action.shape == (2, 4, 29)
    assert report["algorithm"] == "SOURCE_LOCAL_NON_OVERLAPPING_TOP_MIN_CONFIDENCE_WINDOWS"
    excluded = np.zeros((2, 8), dtype=np.bool_)
    excluded[np.arange(2, dtype=np.int32)[:, None], selected_index] = True
    _, _, refreshed_index, _ = mine_corrective_temporal_hard_negatives(
        observation=observation,
        parent_action=action,
        confidence=confidence,
        sample_count_per_source=2,
        consecutive_window_steps=2,
        excluded_index_mask=excluded,
    )
    assert not np.any(excluded[np.arange(2, dtype=np.int32)[:, None], refreshed_index])
    with pytest.raises(ValueError, match="invalid"):
        mine_corrective_temporal_hard_negatives(
            observation=observation,
            parent_action=action,
            confidence=confidence,
            sample_count_per_source=3,
            consecutive_window_steps=2,
        )


def test_cross_domain_normal_replay_is_balanced_unique_and_fail_closed() -> None:
    current_observation = np.arange(8 * 4 * 3, dtype=np.float32).reshape((8, 4, 3))
    current_action = np.zeros((8, 4, 29), dtype=np.float32)
    frozen_observation = -np.arange(4 * 4 * 3, dtype=np.float32).reshape((4, 4, 3)) - 1.0
    frozen_action = np.full((4, 4, 29), 0.25, dtype=np.float32)

    observation, action, frozen_mask = mix_corrective_cross_domain_normal_replay(
        current_observation=current_observation,
        current_parent_action=current_action,
        frozen_observation=frozen_observation,
        frozen_parent_action=frozen_action,
    )

    assert np.array_equal(frozen_mask, np.array([True, False] * 4))
    assert np.array_equal(observation[frozen_mask], frozen_observation)
    assert np.array_equal(action[frozen_mask], frozen_action)
    assert np.array_equal(observation[~frozen_mask], current_observation[1::2])
    with pytest.raises(ValueError, match="cross-domain"):
        mix_corrective_cross_domain_normal_replay(
            current_observation=current_observation,
            current_parent_action=current_action,
            frozen_observation=frozen_observation[:3],
            frozen_parent_action=frozen_action[:3],
        )


def test_stability_is_an_independent_fail_closed_gate() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    parent = np.zeros((4, 4), dtype=np.float32)
    parent[:, 3] = 0.5
    candidate = parent.copy()
    candidate[:, 3] = 0.5101
    passed, tolerance = corrective_stability_retention(
        parent_effect=parent,
        candidate_effect=candidate,
        config=config,
        allow_configured_tolerance=True,
    )
    assert tolerance == pytest.approx(0.01)
    assert passed is False
    strict_passed, strict_tolerance = corrective_stability_retention(
        parent_effect=parent,
        candidate_effect=parent,
        config=config,
        allow_configured_tolerance=False,
    )
    assert strict_tolerance == 0.0
    assert strict_passed is True


def test_confidence_gate_learns_silence_and_fails_closed_out_of_distribution() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    _, model = _inputs(config)
    for index in range(10):
        model["hidden_0_weight"][index, index] = 1.0
        model["hidden_1_weight"][index, index] = 1.0
    failure = np.ones((8, 4, 10), dtype=np.float32)
    normal = -np.ones((8, 4, 10), dtype=np.float32)
    gate, report = fit_corrective_confidence_gate(
        model=model,
        failure_observation=failure,
        normal_observation=normal,
        training_steps=300,
    )
    model.update(gate)
    failure_confidence = predict_corrective_confidence_numpy(model, failure)
    normal_confidence = predict_corrective_confidence_numpy(model, normal)
    assert float(np.mean(failure_confidence)) > 0.95
    assert float(np.mean(normal_confidence)) < 0.05
    assert report["failure_mean_confidence"] > report["normal_mean_confidence"]
    assert report["sample_weight_semantics"] == "WITHIN_CLASS_UNIT_MEAN_BALANCED"
    assert report["failure_sample_weight_minimum"] == pytest.approx(1.0)
    assert report["failure_sample_weight_maximum"] == pytest.approx(1.0)

    ood = np.full((1, 10), 100.0, dtype=np.float32)
    assert predict_corrective_confidence_numpy(model, ood) == pytest.approx(0.0)

    constant_gate, constant_report = fit_corrective_confidence_gate(
        model=model,
        failure_observation=failure,
        normal_observation=normal,
        failure_sample_weight=np.full(failure.shape[:-1], 20.0, dtype=np.float32),
        normal_sample_weight=np.full(normal.shape[:-1], 3.0, dtype=np.float32),
        training_steps=300,
    )
    for name, value in gate.items():
        assert constant_gate[name] == pytest.approx(value)
    assert constant_report["failure_sample_weight_minimum"] == pytest.approx(1.0)
    assert constant_report["normal_sample_weight_maximum"] == pytest.approx(1.0)

    prefix_weight = np.ones(failure.shape[:-1], dtype=np.float32)
    prefix_weight[:, :2] = 20.0
    _, weighted_report = fit_corrective_confidence_gate(
        model=model,
        failure_observation=failure,
        normal_observation=normal,
        failure_sample_weight=prefix_weight,
        training_steps=300,
    )
    assert weighted_report["failure_sample_weight_minimum"] < 1.0
    assert weighted_report["failure_sample_weight_maximum"] > 1.0
    with pytest.raises(ValueError, match="confidence-gate corpus"):
        fit_corrective_confidence_gate(
            model=model,
            failure_observation=failure,
            normal_observation=normal,
            failure_sample_weight=np.ones((8, 3), dtype=np.float32),
            training_steps=300,
        )


def test_confidence_gate_calibration_adds_a_smooth_conservative_deadband() -> None:
    model = {
        "gate_weight": np.asarray((1.0,), dtype=np.float32),
        "gate_bias": np.asarray((0.0,), dtype=np.float32),
        "gate_ood_center": np.asarray((0.0,), dtype=np.float32),
        "gate_ood_scale": np.asarray((1.0,), dtype=np.float32),
        "gate_ood_radius": np.asarray((2.0,), dtype=np.float32),
    }
    calibrated = calibrate_corrective_confidence_gate(model, threshold=0.8, logit_temperature=8.0)
    threshold_logit = np.log(0.8 / 0.2)
    assert calibrated["gate_weight"] == pytest.approx(8.0)
    assert calibrated["gate_bias"] == pytest.approx(-8.0 * threshold_logit)
    assert calibrated["gate_ood_radius"] == pytest.approx(model["gate_ood_radius"])
    with pytest.raises(ValueError, match="calibration"):
        calibrate_corrective_confidence_gate(model, threshold=0.49, logit_temperature=8.0)


def test_historical_veto_can_only_reduce_primary_authority() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    _, model = _inputs(config)
    model["hidden_0_weight"][:] = 0.1
    model["hidden_1_weight"][:] = 0.1
    failure = np.ones((8, 4, 10), dtype=np.float32)
    normal = -np.ones((8, 4, 10), dtype=np.float32)
    primary, _ = fit_corrective_confidence_gate(
        model=model,
        failure_observation=failure,
        normal_observation=normal,
        training_steps=300,
    )
    model.update(primary)
    primary_confidence = predict_corrective_confidence_numpy(model, failure)
    veto, report = fit_corrective_historical_veto_gate(
        model=model,
        failure_observation=failure,
        frozen_normal_observation=normal,
        training_steps=300,
        minimum_authority=0.5,
    )
    model.update(veto)
    vetoed_confidence = predict_corrective_confidence_numpy(model, failure)
    assert np.all(vetoed_confidence <= primary_confidence + 1.0e-7)
    assert veto["veto_gate_minimum_authority"] == pytest.approx(0.5)
    assert report["authority_monotonicity"].startswith("VETO_CAN_ONLY")


def test_channel_veto_is_mirrored_state_conditioned_and_source_monotone() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    _, model = _inputs(config)
    for index in range(10):
        model["hidden_0_weight"][index, index] = 1.0
        model["hidden_1_weight"][index, index] = 1.0
    random = np.random.default_rng(61)
    model["output_weight"] = random.normal(0.0, 0.2, size=(64, 29)).astype(np.float32)
    model["output_gain"] = np.full((29,), 0.8, dtype=np.float32)
    failure = random.normal(1.0, 0.1, size=(16, 4, 10)).astype(np.float32)
    normal = random.normal(-1.0, 0.1, size=(16, 4, 10)).astype(np.float32)
    veto, report = fit_corrective_channel_veto(
        model=model,
        failure_observation=failure,
        normal_observation=normal,
        mirrored_channel_pairs=((3, 9), (5, 11)),
        training_steps=300,
        batch_size=64,
        random_seed=61,
    )
    vetoed_model = {**model, **veto}
    failure_authority = predict_corrective_channel_veto_numpy(vetoed_model, failure)
    normal_authority = predict_corrective_channel_veto_numpy(vetoed_model, normal)
    assert np.mean(failure_authority) > np.mean(normal_authority) + 0.5
    assert np.array_equal(failure_authority[..., 3], failure_authority[..., 9])
    assert np.array_equal(normal_authority[..., 5], normal_authority[..., 11])
    raw_without_veto = predict_corrective_raw_numpy(
        model, normal, maximum_increment=config.maximum_action_increment
    )
    raw_with_veto = predict_corrective_raw_numpy(
        vetoed_model, normal, maximum_increment=config.maximum_action_increment
    )
    assert np.all(np.abs(raw_with_veto) <= np.abs(raw_without_veto) + 1.0e-7)
    assert report["authority_monotonicity"].startswith("CHANNEL_VETO_CAN_ONLY")

    calibrated_model = {
        **vetoed_model,
        **calibrate_corrective_channel_veto(vetoed_model, logit_temperature=2.0),
    }
    calibrated_failure = predict_corrective_channel_veto_numpy(calibrated_model, failure)
    calibrated_normal = predict_corrective_channel_veto_numpy(calibrated_model, normal)
    assert np.mean(calibrated_failure) > np.mean(failure_authority)
    assert np.mean(calibrated_normal) < np.mean(normal_authority)
    assert np.array_equal(calibrated_failure[..., 3], calibrated_failure[..., 9])

    fitted_hot_veto, fitted_hot_report = fit_corrective_channel_veto(
        model=model,
        failure_observation=failure,
        normal_observation=normal,
        mirrored_channel_pairs=((3, 9), (5, 11)),
        training_steps=100,
        batch_size=64,
        logit_temperature=2.0,
        random_seed=61,
    )
    assert fitted_hot_report["calibration"].startswith("IN_PROCESS_LOGIT")
    assert np.array_equal(
        fitted_hot_veto["channel_veto_weight"],
        np.asarray(
            fitted_hot_veto["channel_veto_uncalibrated_weight"] * 2.0,
            dtype=np.float32,
        ),
    )
    recalibrated_hot_veto = calibrate_corrective_channel_veto(
        fitted_hot_veto, logit_temperature=3.0
    )
    assert np.array_equal(
        recalibrated_hot_veto["channel_veto_weight"],
        np.asarray(
            fitted_hot_veto["channel_veto_uncalibrated_weight"] * 3.0,
            dtype=np.float32,
        ),
    )
    assert not np.array_equal(
        recalibrated_hot_veto["channel_veto_weight"],
        np.asarray(fitted_hot_veto["channel_veto_weight"] * 3.0, dtype=np.float32),
    )
    recall_calibrated_veto = calibrate_corrective_channel_veto(
        fitted_hot_veto,
        logit_temperature=2.0,
        failure_recall_logit_margin=1.5,
    )
    assert np.array_equal(
        recall_calibrated_veto["channel_veto_weight"],
        fitted_hot_veto["channel_veto_weight"],
    )
    assert np.array_equal(
        recall_calibrated_veto["channel_veto_bias"],
        np.asarray(fitted_hot_veto["channel_veto_bias"] + 1.5, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="mirror pairs"):
        fit_corrective_channel_veto(
            model=model,
            failure_observation=failure,
            normal_observation=normal,
            mirrored_channel_pairs=((3, 9), (9, 10)),
            training_steps=100,
        )
    with pytest.raises(ValueError, match="calibration"):
        calibrate_corrective_channel_veto(vetoed_model, logit_temperature=0.9)
    with pytest.raises(ValueError, match="calibration"):
        calibrate_corrective_channel_veto(
            vetoed_model,
            logit_temperature=2.0,
            failure_recall_logit_margin=-0.1,
        )


def test_temporal_gate_requires_evidence_and_bounds_intervention_lease() -> None:
    gate_model = {
        "gate_weight": np.asarray((1.0,), dtype=np.float32),
        "gate_bias": np.asarray((0.0,), dtype=np.float32),
        "gate_ood_center": np.asarray((0.0,), dtype=np.float32),
        "gate_ood_scale": np.asarray((1.0,), dtype=np.float32),
        "gate_ood_radius": np.asarray((2.0,), dtype=np.float32),
    }
    temporal = attach_corrective_temporal_lease(
        gate_model,
        CorrectiveTemporalLeaseConfig(
            required_open_steps=3,
            maximum_lease_steps=2,
            cooldown_steps=4,
            maximum_slew=0.5,
        ),
    )
    model = {**gate_model, **temporal}
    state = initial_corrective_temporal_gate_state(model, (1,))
    outputs = []
    for confidence in (0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9):
        output, state = step_corrective_temporal_gate_numpy(
            model, np.asarray(((confidence,),), dtype=np.float32), state
        )
        outputs.append(float(output[0, 0]))
    assert outputs[:4] == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert outputs[4:6] == pytest.approx((0.5, 0.9))
    assert outputs[6:] == pytest.approx((0.4, 0.0))
    assert state[0, 3] > 0.0

    with pytest.raises(ValueError, match="temporal lease"):
        CorrectiveTemporalLeaseConfig(exit_threshold=0.6, open_threshold=0.5)


def test_veto_aware_temporal_trigger_blocks_false_lease_without_reducing_amplitude() -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    _, model = _inputs(config)
    model["output_bias"] = np.full((29,), 0.5, dtype=np.float32)
    model.update(
        {
            "gate_weight": np.zeros((64,), dtype=np.float32),
            "gate_bias": np.asarray((10.0,), dtype=np.float32),
            "gate_ood_center": np.zeros((64,), dtype=np.float32),
            "gate_ood_scale": np.ones((64,), dtype=np.float32),
            "gate_ood_radius": np.asarray((1_000.0,), dtype=np.float32),
            "channel_veto_weight": np.zeros((64, 29), dtype=np.float32),
            "channel_veto_bias": np.zeros((29,), dtype=np.float32),
            "channel_veto_ood_center": np.zeros((64,), dtype=np.float32),
            "channel_veto_ood_scale": np.ones((64,), dtype=np.float32),
            "channel_veto_ood_radius": np.asarray((1_000.0,), dtype=np.float32),
        }
    )
    model.update(
        attach_corrective_temporal_lease(
            model,
            CorrectiveTemporalLeaseConfig(
                open_threshold=0.95,
                exit_threshold=0.02,
                required_open_steps=2,
                maximum_lease_steps=4,
                cooldown_steps=20,
                maximum_slew=1.0,
            ),
        )
    )
    observation = np.zeros((1, 4, 10), dtype=np.float32)
    legacy = predict_corrective_student_numpy(
        model, observation, maximum_increment=config.maximum_action_increment
    )
    aware_model = {**model, **attach_corrective_veto_aware_temporal_trigger(model)}
    aware = predict_corrective_student_numpy(
        aware_model, observation, maximum_increment=config.maximum_action_increment
    )
    assert np.any(legacy != 0.0)
    assert np.array_equal(aware, np.zeros_like(aware))

    high_authority = dict(model)
    high_authority["channel_veto_bias"] = np.full((29,), 20.0, dtype=np.float32)
    legacy_high = predict_corrective_student_numpy(
        high_authority, observation, maximum_increment=config.maximum_action_increment
    )
    aware_high = predict_corrective_student_numpy(
        {
            **high_authority,
            **attach_corrective_veto_aware_temporal_trigger(high_authority),
        },
        observation,
        maximum_increment=config.maximum_action_increment,
    )
    assert np.array_equal(aware_high, legacy_high)

    ambiguous = dict(model)
    ambiguous["channel_veto_bias"] = np.full((29,), 4.0, dtype=np.float32)
    eligibility_only = predict_corrective_student_numpy(
        {
            **ambiguous,
            **attach_corrective_veto_aware_temporal_trigger(ambiguous),
        },
        observation,
        maximum_increment=config.maximum_action_increment,
    )
    consensus_scaled = predict_corrective_student_numpy(
        {
            **ambiguous,
            **attach_corrective_veto_aware_temporal_trigger(
                ambiguous, scale_amplitude_by_consensus=True
            ),
        },
        observation,
        maximum_increment=config.maximum_action_increment,
    )
    assert np.any(consensus_scaled != 0.0)
    assert np.linalg.norm(consensus_scaled) < np.linalg.norm(eligibility_only)


def test_corrective_student_evidence_binds_model_corpus_and_authority(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    output = tmp_path / "student"
    report = write_recovery_corrective_student_evidence(
        output_dir=output,
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"initial_output_was_exact_zero": True, "steps": 10},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )
    assert report["balanced_failure_normal_training"] is True
    assert report["promotion_authority"] == "NONE"
    assert report["student_development_retained"] is True

    path = output / "student-report.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["promotion_eligible"] = True
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(path)

    inconsistent = dict(report)
    inconsistent["student_development_retained"] = False
    inconsistent.pop("report_hash")
    inconsistent["report_hash"] = hash_json(inconsistent)
    path.write_text(json.dumps(inconsistent), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(path)


def test_corrective_student_evidence_binds_effect_budget_to_model(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    source_gain = np.full((29,), 0.8, dtype=np.float32)
    derived_gain = source_gain.copy()
    derived_gain[[5, 11]] = 0.56
    model["output_gain"] = derived_gain
    effect_budget = {
        "source_gain": source_gain.tolist(),
        "derived_gain": derived_gain.tolist(),
        "authority_monotonicity": "DERIVED_GAIN_CAN_ONLY_RETAIN_OR_REDUCE_SOURCE_GAIN",
        "selection_split": "CURRENT_FAILURE_TRAIN_AND_FROZEN_NORMAL_TRAIN_ONLY",
        "frozen_holdout_consumed_for_selection": False,
        "failure_training_source_count": 9,
        "historical_normal_training_source_count": 72,
        "failure_prediction_content_hash": _digest("a"),
        "historical_normal_prediction_content_hash": _digest("b"),
    }
    output = tmp_path / "student-budget"
    write_recovery_corrective_student_evidence(
        output_dir=output,
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"effect_channel_budget": effect_budget},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )

    path = output / "student-report.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["training"]["effect_channel_budget"]["derived_gain"][5] = 0.55
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(path)


def test_corrective_student_evidence_binds_channel_veto_provenance(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    model.update(
        {
            "output_gain": np.full((29,), 0.8, dtype=np.float32),
            "channel_veto_weight": np.zeros((64, 29), dtype=np.float32),
            "channel_veto_bias": np.zeros((29,), dtype=np.float32),
            "channel_veto_ood_center": np.zeros((64,), dtype=np.float32),
            "channel_veto_ood_scale": np.ones((64,), dtype=np.float32),
            "channel_veto_ood_radius": np.asarray((1_000.0,), dtype=np.float32),
        }
    )
    hash_fields = (
        "frozen_closed_loop_observation_content_hash",
        "frozen_closed_loop_parent_action_content_hash",
        "frozen_selected_index_content_hash",
        "frozen_replay_source_mask_content_hash",
        "source_student_report_hash",
        "source_student_report_file_hash",
        "current_normal_student_report_hash",
        "current_normal_student_report_file_hash",
        "current_normal_corpus_hash",
        "frozen_normal_student_report_hash",
        "frozen_normal_student_report_file_hash",
        "frozen_normal_corpus_hash",
        "frozen_failure_state_manifest_hash",
    )
    channel_veto = {
        "algorithm": "RAW_ACTIVITY_WEIGHTED_MIRRORED_LATENT_VECTOR_VETO",
        "combination": "STATIC_OUTPUT_GAIN_TIMES_STATE_CONDITIONED_CHANNEL_VETO",
        "authority_monotonicity": "CHANNEL_VETO_CAN_ONLY_RETAIN_OR_REDUCE_STATIC_GAIN",
        "selection_split": (
            "CURRENT_FAILURE_TRAIN_CURRENT_NORMAL_TRAIN_AND_FROZEN_CLOSED_LOOP_NORMAL_TRAIN_ONLY"
        ),
        "current_holdout_consumed_for_selection": False,
        "frozen_holdout_consumed_for_selection": False,
        "current_training_source_count": 9,
        "frozen_training_source_count": 4,
        "failure_sample_count": 36,
        "normal_sample_count": 36,
        "frozen_closed_loop_rollout_steps": 20,
        "static_output_gain": [0.8] * 29,
        "failure_mean_authority": [0.5] * 29,
        "normal_mean_authority": [0.5] * 29,
        "failure_ood_fraction": 0.0,
        "normal_ood_fraction": 0.0,
        "ood_radius": 1_000.0,
        "mirrored_channel_pairs": [[3, 9], [5, 11]],
        "hard_negative_mining": {},
        **{name: _digest("a") for name in hash_fields},
    }
    output = tmp_path / "student-channel-veto"
    write_recovery_corrective_student_evidence(
        output_dir=output,
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"channel_veto": channel_veto},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )

    margin_model = dict(model)
    margin_model.update(
        {
            "channel_veto_uncalibrated_weight": np.zeros((64, 29), dtype=np.float32),
            "channel_veto_uncalibrated_bias": np.zeros((29,), dtype=np.float32),
            "channel_veto_bias": np.full((29,), 1.5, dtype=np.float32),
        }
    )
    margin_failure_authority = predict_corrective_channel_veto_numpy(
        margin_model, corpus["failure_observation"][~corpus["holdout_source_mask"]]
    )
    margin_normal_authority = predict_corrective_channel_veto_numpy(
        margin_model, corpus["normal_observation"][~corpus["holdout_source_mask"]]
    )
    margin_channel_veto = {
        **channel_veto,
        "calibration": "IN_PROCESS_LOGIT_TEMPERATURE_WITH_FAILURE_RECALL_MARGIN",
        "calibration_logit_temperature": 2.0,
        "calibration_failure_recall_logit_margin": 1.5,
        "failure_mean_authority": np.mean(
            margin_failure_authority,
            axis=tuple(range(margin_failure_authority.ndim - 1)),
        ).tolist(),
        "normal_mean_authority": np.mean(
            margin_normal_authority,
            axis=tuple(range(margin_normal_authority.ndim - 1)),
        ).tolist(),
    }
    write_recovery_corrective_student_evidence(
        output_dir=tmp_path / "student-channel-veto-recall-margin",
        config=config,
        corpus=corpus,
        model=margin_model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"channel_veto": margin_channel_veto},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )

    aware_model = dict(model)
    aware_model.update(
        {
            "gate_weight": np.zeros((64,), dtype=np.float32),
            "gate_bias": np.zeros((1,), dtype=np.float32),
            "gate_ood_center": np.zeros((64,), dtype=np.float32),
            "gate_ood_scale": np.ones((64,), dtype=np.float32),
            "gate_ood_radius": np.asarray((1_000.0,), dtype=np.float32),
        }
    )
    aware_model.update(
        attach_corrective_temporal_lease(
            aware_model,
            CorrectiveTemporalLeaseConfig(
                required_open_steps=2,
                maximum_lease_steps=4,
                cooldown_steps=20,
            ),
        )
    )
    aware_model.update(attach_corrective_veto_aware_temporal_trigger(aware_model))
    aware_channel_veto = {
        **channel_veto,
        "temporal_trigger": "PRIMARY_CONFIDENCE_TIMES_MEAN_CHANNEL_AUTHORITY",
        "temporal_trigger_amplitude_semantics": (
            "PRIMARY_CONFIDENCE_UNCHANGED_AFTER_TRIGGER_QUALIFIES"
        ),
        "temporal_trigger_source_student_report_hash": _digest("b"),
        "temporal_trigger_source_student_report_file_hash": _digest("c"),
    }
    aware_output = tmp_path / "student-channel-veto-aware-trigger"
    write_recovery_corrective_student_evidence(
        output_dir=aware_output,
        config=config,
        corpus=corpus,
        model=aware_model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"channel_veto": aware_channel_veto},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )
    aware_path = aware_output / "student-report.json"
    aware_tampered = json.loads(aware_path.read_text(encoding="utf-8"))
    aware_tampered["training"]["channel_veto"].pop("temporal_trigger")
    aware_tampered.pop("report_hash")
    aware_tampered["report_hash"] = hash_json(aware_tampered)
    aware_path.write_text(json.dumps(aware_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(aware_path)

    repair_channel_veto = {
        **channel_veto,
        "source_repair_mode": True,
        "source_failure_gate_passed": True,
        "current_normal_on_policy_child_bound": True,
        "hard_negative_mining": {
            "score_semantics": "SOURCE_RAW_INCREMENT_RMS_DIVIDED_BY_MAXIMUM_INCREMENT"
        },
    }
    repair_output = tmp_path / "student-channel-veto-repair"
    write_recovery_corrective_student_evidence(
        output_dir=repair_output,
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"channel_veto": repair_channel_veto},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )
    repair_path = repair_output / "student-report.json"
    repair_tampered = json.loads(repair_path.read_text(encoding="utf-8"))
    repair_tampered["training"]["channel_veto"]["current_normal_on_policy_child_bound"] = False
    repair_tampered.pop("report_hash")
    repair_tampered["report_hash"] = hash_json(repair_tampered)
    repair_path.write_text(json.dumps(repair_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(repair_path)

    path = output / "student-report.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["training"]["channel_veto"]["static_output_gain"][3] = 0.7
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(path)


def test_corrective_student_evidence_recomputes_closed_loop_dagger_mining(
    tmp_path: Path,
) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    model.update(
        {
            "output_gain": np.full((29,), 0.8, dtype=np.float32),
            "gate_weight": np.zeros((64,), dtype=np.float32),
            "gate_bias": np.zeros((1,), dtype=np.float32),
            "gate_ood_center": np.zeros((64,), dtype=np.float32),
            "gate_ood_scale": np.ones((64,), dtype=np.float32),
            "gate_ood_radius": np.asarray((1_000.0,), dtype=np.float32),
            "channel_veto_weight": np.zeros((64, 29), dtype=np.float32),
            "channel_veto_bias": np.zeros((29,), dtype=np.float32),
            "channel_veto_ood_center": np.zeros((64,), dtype=np.float32),
            "channel_veto_ood_scale": np.ones((64,), dtype=np.float32),
            "channel_veto_ood_radius": np.asarray((1_000.0,), dtype=np.float32),
        }
    )
    model.update(
        attach_corrective_temporal_lease(
            model,
            CorrectiveTemporalLeaseConfig(
                required_open_steps=2,
                maximum_lease_steps=4,
                cooldown_steps=20,
            ),
        )
    )
    train_count = int(np.sum(corpus["train_source_mask"]))
    frozen_count = 4
    current_score = np.broadcast_to(
        np.linspace(0.0, 1.0, config.normal_rollout_steps, dtype=np.float32),
        (train_count, config.normal_rollout_steps),
    ).copy()
    frozen_score = np.flip(current_score[:frozen_count], axis=1).copy()
    current_applied = np.broadcast_to(
        current_score[..., None] * config.maximum_action_increment,
        current_score.shape + (29,),
    ).copy()
    frozen_applied = np.broadcast_to(
        frozen_score[..., None] * config.maximum_action_increment,
        frozen_score.shape + (29,),
    ).copy()
    current_score = np.clip(
        np.sqrt(np.mean(np.square(current_applied), axis=-1)) / config.maximum_action_increment,
        0.0,
        1.0,
    ).astype(np.float32)
    frozen_score = np.clip(
        np.sqrt(np.mean(np.square(frozen_applied), axis=-1)) / config.maximum_action_increment,
        0.0,
        1.0,
    ).astype(np.float32)

    def mine(score: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        _, _, index, report = mine_corrective_temporal_hard_negatives(
            observation=np.zeros(score.shape + (10,), dtype=np.float32),
            parent_action=np.zeros(score.shape + (29,), dtype=np.float32),
            confidence=score,
            sample_count_per_source=config.normal_sample_count_per_route,
            consecutive_window_steps=2,
        )
        report.update(
            {
                "score_semantics": ("ACTUAL_APPLIED_INCREMENT_RMS_DIVIDED_BY_MAXIMUM_INCREMENT"),
                "full_trace_applied_increment_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.broadcast_to(
                                    score[..., None] * config.maximum_action_increment,
                                    score.shape + (29,),
                                )
                            )
                        )
                    )
                ),
            }
        )
        return index, report

    current_index, current_mining = mine(current_score)
    frozen_index, frozen_mining = mine(frozen_score)
    frozen_replay = np.zeros((12,), dtype=np.bool_)
    frozen_replay[np.flatnonzero(corpus["train_source_mask"])[:frozen_count]] = True
    corpus.update(
        {
            "dagger_current_applied_increment": current_applied,
            "dagger_current_selected_index": current_index,
            "dagger_frozen_applied_increment": frozen_applied,
            "dagger_frozen_selected_index": frozen_index,
            "dagger_frozen_replay_source_mask": frozen_replay,
        }
    )
    common_hash_fields = (
        "frozen_closed_loop_observation_content_hash",
        "frozen_closed_loop_parent_action_content_hash",
        "source_student_report_hash",
        "source_student_report_file_hash",
        "current_normal_student_report_hash",
        "current_normal_student_report_file_hash",
        "current_normal_corpus_hash",
        "dagger_candidate_student_report_hash",
        "dagger_candidate_student_report_file_hash",
        "frozen_normal_student_report_hash",
        "frozen_normal_student_report_file_hash",
        "frozen_normal_corpus_hash",
        "frozen_failure_state_manifest_hash",
        "current_closed_loop_observation_content_hash",
        "current_closed_loop_parent_action_content_hash",
    )
    channel_veto = {
        "algorithm": "RAW_ACTIVITY_WEIGHTED_MIRRORED_LATENT_VECTOR_VETO",
        "combination": "STATIC_OUTPUT_GAIN_TIMES_STATE_CONDITIONED_CHANNEL_VETO",
        "authority_monotonicity": "CHANNEL_VETO_CAN_ONLY_RETAIN_OR_REDUCE_STATIC_GAIN",
        "selection_split": "CURRENT_AND_FROZEN_CANDIDATE_CLOSED_LOOP_NORMAL_TRAIN_ONLY",
        "current_holdout_consumed_for_selection": False,
        "frozen_holdout_consumed_for_selection": False,
        "current_training_source_count": train_count,
        "frozen_training_source_count": frozen_count,
        "failure_sample_count": train_count * config.trace_steps,
        "normal_sample_count": train_count * config.normal_sample_count_per_route,
        "current_closed_loop_rollout_steps": config.normal_rollout_steps,
        "frozen_closed_loop_rollout_steps": config.normal_rollout_steps,
        "static_output_gain": [0.8] * 29,
        "failure_mean_authority": [0.5] * 29,
        "normal_mean_authority": [0.5] * 29,
        "failure_ood_fraction": 0.0,
        "normal_ood_fraction": 0.0,
        "ood_radius": 1_000.0,
        "mirrored_channel_pairs": [],
        "hard_negative_mining": {
            "current": current_mining,
            "frozen": frozen_mining,
        },
        "current_closed_loop_applied_increment_content_hash": hash_bytes(
            np.ascontiguousarray(current_applied).tobytes()
        ),
        "current_selected_index_content_hash": hash_bytes(
            np.ascontiguousarray(current_index).tobytes()
        ),
        "frozen_closed_loop_applied_increment_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_applied).tobytes()
        ),
        "frozen_selected_index_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_index).tobytes()
        ),
        "frozen_replay_source_mask_content_hash": hash_bytes(
            np.ascontiguousarray(frozen_replay).tobytes()
        ),
        **{name: _digest("a") for name in common_hash_fields},
    }
    output = tmp_path / "student-channel-veto-dagger"
    write_recovery_corrective_student_evidence(
        output_dir=output,
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={"channel_veto": channel_veto},
        failure_exam={"passed": True},
        normal_exam={"passed": True},
    )
    path = output / "student-report.json"
    validate_recovery_corrective_student_evidence(path)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["training"]["channel_veto"]["hard_negative_mining"]["current"][
        "selected_window_score_mean"
    ] += 0.1
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(path)


def test_corrective_student_evidence_recomputes_per_source_exam(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    failure_exam = {
        "passed": True,
        "state_count": 2,
        "mean_parent_cost": 1.0,
        "mean_candidate_cost": 0.98,
        "mean_cost_improvement_fraction": 0.02,
        "median_cost_improvement_fraction": 0.02,
        "minimum_cost_improvement_fraction": 0.02,
        "mean_action_increment_rms": 0.01,
        "mean_parent_effect_metrics": [0.1, 0.1, 0.1, 0.2],
        "mean_candidate_effect_metrics": [0.09, 0.09, 0.09, 0.19],
        "directional_retention_passed": True,
        "stability_retention_passed": True,
        "stability_retention_tolerance": 0.0,
        "normal_cost_regression_fraction": None,
        "finite_fraction": 1.0,
        "paired_execution_semantics": (
            "LOCKSTEP_SINGLE_GRAPH_EXACT_ZERO_CAUSAL_COUPLING_SHARED_RESET_AND_ACTION_RNG"
        ),
        "exact_zero_intervention_causal_identity_enforced": True,
        "source_diagnostics": {
            "parent_cost": [1.0, 1.0],
            "candidate_cost": [0.98, 0.98],
            "parent_effect_metrics": [[0.1, 0.1, 0.1, 0.2]] * 2,
            "candidate_effect_metrics": [[0.09, 0.09, 0.09, 0.19]] * 2,
            "action_increment_rms": [0.01, 0.01],
            "finite": [True, True],
        },
    }
    normal_exam = {
        "passed": True,
        "state_count": 2,
        "mean_parent_cost": 1.0,
        "mean_candidate_cost": 1.0,
        "mean_cost_improvement_fraction": 0.0,
        "median_cost_improvement_fraction": 0.0,
        "minimum_cost_improvement_fraction": 0.0,
        "mean_action_increment_rms": 0.0,
        "mean_parent_effect_metrics": [0.1, 0.1, 0.1, 0.2],
        "mean_candidate_effect_metrics": [0.1, 0.1, 0.1, 0.2],
        "directional_retention_passed": True,
        "stability_retention_passed": True,
        "stability_retention_tolerance": 0.004,
        "normal_cost_regression_fraction": 0.0,
        "finite_fraction": 1.0,
        "paired_execution_semantics": (
            "LOCKSTEP_SINGLE_GRAPH_EXACT_ZERO_CAUSAL_COUPLING_SHARED_RESET_AND_ACTION_RNG"
        ),
        "exact_zero_intervention_causal_identity_enforced": True,
        "source_diagnostics": {
            "parent_cost": [1.0, 1.0],
            "candidate_cost": [1.0, 1.0],
            "parent_effect_metrics": [[0.1, 0.1, 0.1, 0.2]] * 2,
            "candidate_effect_metrics": [[0.1, 0.1, 0.1, 0.2]] * 2,
            "action_increment_rms": [0.0, 0.0],
            "finite": [True, True],
        },
    }
    output = tmp_path / "student-source-diagnostics"
    write_recovery_corrective_student_evidence(
        output_dir=output,
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={},
        failure_exam=failure_exam,
        normal_exam=normal_exam,
    )

    path = output / "student-report.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["normal_route_paired_physics_exam"]["source_diagnostics"]["candidate_cost"][0] = 2.0
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_student_evidence(path)


def test_frozen_exam_v2_binds_valid_sources_and_recomputes_routes(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    source_reports = []
    source_paths = []
    for name in ("candidate", "frozen"):
        output = tmp_path / name
        report = write_recovery_corrective_student_evidence(
            output_dir=output,
            config=config,
            corpus=corpus,
            model=model,
            lineage=_lineage(),
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            training={},
            failure_exam={"passed": True},
            normal_exam={"passed": True},
        )
        source_reports.append(report)
        source_paths.append(output / "student-report.json")
    state_count = int(source_reports[1]["holdout_source_count"])
    failure_exam = {
        "passed": True,
        "state_count": state_count,
        "route_kind": "UNSEEN_EXACT_FAILURE_STATES",
        "mean_parent_cost": 1.0,
        "mean_candidate_cost": 0.98,
        "mean_cost_improvement_fraction": 0.02,
        "median_cost_improvement_fraction": 0.02,
        "minimum_cost_improvement_fraction": 0.02,
        "mean_action_increment_rms": 0.01,
        "mean_parent_effect_metrics": [0.1, 0.1, 0.1, 0.2],
        "mean_candidate_effect_metrics": [0.09, 0.09, 0.09, 0.19],
        "directional_retention_passed": True,
        "stability_retention_passed": True,
        "stability_retention_tolerance": 0.0,
        "normal_cost_regression_fraction": None,
        "finite_fraction": 1.0,
        "source_diagnostics": {
            "parent_cost": [1.0] * state_count,
            "candidate_cost": [0.98] * state_count,
            "parent_effect_metrics": [[0.1, 0.1, 0.1, 0.2]] * state_count,
            "candidate_effect_metrics": [[0.09, 0.09, 0.09, 0.19]] * state_count,
            "action_increment_rms": [0.01] * state_count,
            "finite": [True] * state_count,
        },
    }
    normal_exam = {
        **failure_exam,
        "route_kind": "NORMAL_PARENT_ROUTE",
        "mean_candidate_cost": 1.0,
        "mean_cost_improvement_fraction": 0.0,
        "median_cost_improvement_fraction": 0.0,
        "minimum_cost_improvement_fraction": 0.0,
        "mean_action_increment_rms": 0.0,
        "mean_candidate_effect_metrics": [0.1, 0.1, 0.1, 0.2],
        "stability_retention_tolerance": 0.004,
        "normal_cost_regression_fraction": 0.0,
        "source_diagnostics": {
            "parent_cost": [1.0] * state_count,
            "candidate_cost": [1.0] * state_count,
            "parent_effect_metrics": [[0.1, 0.1, 0.1, 0.2]] * state_count,
            "candidate_effect_metrics": [[0.1, 0.1, 0.1, 0.2]] * state_count,
            "action_increment_rms": [0.0] * state_count,
            "finite": [True] * state_count,
        },
    }
    output = tmp_path / "frozen-v2.json"
    report = write_recovery_corrective_frozen_exam_evidence(
        candidate_report=source_reports[0],
        candidate_report_path=source_paths[0],
        frozen_report=source_reports[1],
        frozen_report_path=source_paths[1],
        failure_exam=failure_exam,
        normal_exam=normal_exam,
        output_path=output,
    )
    assert report["schema_version"].endswith("v2")
    assert report["frozen_benchmark_passed"] is True

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["candidate_student_report_path"] = tampered["frozen_benchmark_student_report_path"]
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_frozen_exam_evidence(output)


def test_corrective_repeat_gate_uses_content_identity_and_worst_run(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    report_paths = []
    for index, normal_passed in enumerate((True, False), start=1):
        output = tmp_path / f"repeat-{index}"
        write_recovery_corrective_student_evidence(
            output_dir=output,
            config=config,
            corpus=corpus,
            model=model,
            lineage=_lineage(),
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            training={},
            failure_exam={
                "passed": True,
                "mean_cost_improvement_fraction": 0.02,
                "directional_retention_passed": True,
                "stability_retention_passed": True,
                "finite_fraction": 1.0,
            },
            normal_exam={
                "passed": normal_passed,
                "normal_cost_regression_fraction": 0.0 if normal_passed else 0.02,
                "directional_retention_passed": normal_passed,
                "stability_retention_passed": True,
                "finite_fraction": 1.0,
            },
        )
        report_paths.append(output / "student-report.json")
    gate_path = tmp_path / "repeat-gate.json"
    gate = write_recovery_corrective_repeat_evidence(
        report_paths=tuple(report_paths), output_path=gate_path
    )
    assert gate["repeat_count"] == 2
    assert gate["worst_normal_cost_regression_fraction"] == pytest.approx(0.02)
    assert gate["all_repeats_retained"] is False
    assert gate["repeat_gate_passed"] is False

    tampered = json.loads(gate_path.read_text(encoding="utf-8"))
    tampered["repeat_gate_passed"] = True
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    gate_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_repeat_evidence(gate_path)


def test_corrective_student_rejects_non_disjoint_masks(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    corpus["holdout_source_mask"] = np.array(corpus["train_source_mask"], copy=True)
    with pytest.raises(ValueError, match="contract"):
        write_recovery_corrective_student_evidence(
            output_dir=tmp_path / "bad",
            config=config,
            corpus=corpus,
            model=model,
            lineage=_lineage(),
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            training={},
            failure_exam={"passed": False},
            normal_exam={"passed": False},
        )


def test_corrective_student_stability_gate_cannot_be_hidden_by_exam_pass(tmp_path: Path) -> None:
    config = RecoveryCorrectiveStudentConfig(
        trace_steps=4,
        normal_rollout_steps=20,
        normal_sample_count_per_route=4,
        holdout_states_per_window=1,
        training_steps=10,
    )
    corpus, model = _inputs(config)
    report = write_recovery_corrective_student_evidence(
        output_dir=tmp_path / "unstable",
        config=config,
        corpus=corpus,
        model=model,
        lineage=_lineage(),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        training={},
        failure_exam={"passed": True, "stability_retention_passed": True},
        normal_exam={"passed": True, "stability_retention_passed": False},
    )
    assert report["student_development_retained"] is False
