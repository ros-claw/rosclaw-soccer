from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from rosclaw.continual.reproducibility import (
    NumericalEnvironmentCheck,
    NumericalRuntimeContract,
)

from rosclaw_soccer.evidence import goalkeeper_v2
from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoverageTrial,
    aggregate_coverage_time,
)
from rosclaw_soccer.skills.goalkeeper_v2.policy import (
    GoalkeeperActorArtifact,
    GoalkeeperDenseLayer,
    save_goalkeeper_actor_artifact,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _trials(policy_hash: str) -> tuple[GoalkeeperCoverageTrial, ...]:
    return tuple(
        GoalkeeperCoverageTrial(
            scenario_hash=_hash(f"scenario:{deadline}"),
            frozen_shooter_policy_hash=_hash("shooter"),
            numerical_contract_hash=_hash("cpu"),
            seed=0,
            target_region="center",
            target_y_m=0.0,
            target_z_m=1.0,
            deadline_sec=deadline,
            observed_flight_start_sec=0.02,
            first_action_sec=0.10,
            ball_contact=False,
            true_save=False,
            intercept_error_m=0.1,
            recovery_time_sec=None,
            second_save_success=False,
            idle_ratio=0.0,
            human_motion_score=None,
            safety_cost=0.0,
            actor_observation_contract_hash=_hash("actor"),
            evaluated_actor_policy_hash=policy_hash,
        )
        for deadline in (1.0, 0.8, 0.6, 0.5, 0.4)
    )


def test_candidate_evidence_binds_training_actor_and_rejects_incomplete_gate(
    tmp_path, monkeypatch
) -> None:
    actor = GoalkeeperActorArtifact(
        policy_id="keeper.test",
        generation=1,
        parent_policy_hash=_hash("parent"),
        body_hash=_hash("body"),
        actor_observation_contract_hash=_hash("actor"),
        motion_library_hash=_hash("motion"),
        training_run_hash=_hash("run"),
        layers=(
            GoalkeeperDenseLayer(
                weights=((0.0, 0.0),),
                bias=(0.0, 0.0),
                activation="tanh",
            ),
            GoalkeeperDenseLayer(
                weights=((0.0,) * 30, (0.0,) * 30),
                bias=(0.0,) * 30,
                activation="tanh",
            ),
        ),
        maximum_lateral_speed_mps=0.5,
        maximum_joint_residual_rad=(0.1,) * 29,
    )
    actor_path = tmp_path / "actor.json"
    save_goalkeeper_actor_artifact(actor, actor_path, source_checkout=tmp_path / "source")
    report_path = tmp_path / "training.json"
    report_path.write_text(
        json.dumps(
            {
                "candidate_policy_hash": actor.policy_hash,
                "parent_policy_hash": actor.parent_policy_hash,
                "report_hash": _hash("training"),
            }
        ),
        encoding="utf-8",
    )
    parent_trials = _trials(actor.parent_policy_hash)
    candidate_trials = tuple(
        replace(trial, evaluated_actor_policy_hash=actor.policy_hash) for trial in parent_trials
    )
    calls = iter((parent_trials, parent_trials, candidate_trials, candidate_trials))

    def fake_run(**_kwargs):
        trials = next(calls)
        return aggregate_coverage_time(trials, strict_replay=False, sealed_holdout=False), trials

    monkeypatch.setattr(goalkeeper_v2, "run_parent_coverage_time_baseline", fake_run)
    contract = NumericalRuntimeContract.single_threaded_cpu(random_seed=0)
    monkeypatch.setattr(
        NumericalRuntimeContract,
        "verify_environment",
        lambda self: NumericalEnvironmentCheck(
            expected=self.required_environment,
            observed=self.required_environment,
            mismatches=(),
        ),
    )

    evidence = goalkeeper_v2.run_goalkeeper_v2_candidate_evidence(
        asset_root=tmp_path / "assets",
        actor_artifact_path=actor_path,
        training_report_path=report_path,
        output_dir=tmp_path / "evidence",
        source_checkout=tmp_path / "source",
        numerical_contract=contract,
    )

    assert evidence.promotion_decision.verdict == "REJECTED"
    assert evidence.strict_parent_replay
    assert evidence.strict_candidate_replay
    assert evidence.implementation_hash == goalkeeper_v2.goalkeeper_v2_implementation_hash()
    assert (tmp_path / "evidence" / "goalkeeper-v2-candidate-evidence.json").is_file()
