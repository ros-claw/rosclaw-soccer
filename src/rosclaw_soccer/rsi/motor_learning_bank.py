"""Offline actor/critic bootstrap bank from independently reviewed physical data.

Observation inputs end at frame 30, before ANY candidate's residual starts.
Future outcomes are learning labels only. Course identities are audit metadata,
not model inputs. This bank grants no runtime, test-set or promotion authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_contract import load_policy
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed, review_curriculum
from rosclaw_soccer.rsi.precontact_proprio_policy import FEATURE_NAMES, proprio_vector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES


def causal_context(body: Any, motor: Any, first_contact: int | None) -> tuple[float, ...]:
    if first_contact is not None and (type(first_contact) is not int or first_contact <= 30):
        raise ValueError("contact occurred before context decision")
    delta = motor["applied_joint_delta_rad"]
    if delta.shape != (300, 1, 12) or not np.isfinite(delta).all() or np.any(delta[:31] != 0):
        raise ValueError("candidate affected its own supposedly frozen context")
    return proprio_vector(
        body["root_pose_xyzw_m"][30, 0],
        body["root_velocity_world"][30, 0],
        body["ball_position_before_step_m"][20, 0],
        body["ball_position_before_step_m"][30, 0],
        body["ball_linear_velocity_before_step_m_s"][30, 0],
        body["foot_geometry_position_before_step_m"][20, 0],
        body["foot_geometry_position_before_step_m"][30, 0],
    )


def build_bank(training_root: Path, validation_root: Path) -> dict[str, Any]:
    # Recompute evidence instead of trusting a supplied review or cached labels.
    review = review_curriculum(training_root, validation_root)
    if not review["training_complete"] or review["complete_candidate_count"] != 32:
        raise ValueError("finished, independently reviewed curriculum required for learning bank")
    contexts: dict[tuple[int, int], tuple[float, ...]] = {}
    samples = []
    parameters: dict[str, list[float]] = {}
    for candidate in review["candidates"]:
        generation, number = candidate["generation"], candidate["candidate"]
        policy, knots = load_policy(Path(candidate["policy"]))
        parameters[policy["policy_hash"]] = [*knots.ravel().tolist(), policy["phase_gap_end_m"]]
        for index, (seed, lane) in enumerate(COURSES):
            folder = training_root / f"seed{seed}-lane{lane}-g{generation}-c{number}-actor"
            report = _sealed(folder / "report.json")
            with (
                np.load(folder / "body_trace.npz", allow_pickle=False) as body,
                np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
            ):
                context = causal_context(
                    body, motor, report["environments"][0]["first_contact_frame"]
                )
            key = (seed, lane)
            if key in contexts and contexts[key] != context:
                raise ValueError("context differs between candidate branches before motor onset")
            contexts[key] = context
            outcome = candidate["rows"][index]
            samples.append(
                {
                    "observation": list(context),
                    "motor_parameters": parameters[policy["policy_hash"]],
                    "learning_labels": {
                        key: outcome[key]
                        for key in (
                            "high_quality",
                            "clean_foot_only",
                            "minimum_pelvis_z_m",
                            "maximum_lateral_excursion_m",
                            "forward_60_m",
                            "lateral_60_m",
                            "reward",
                        )
                    },
                    "audit_metadata": {
                        "course": list(key),
                        "policy_hash": policy["policy_hash"],
                        "source_report_hash": candidate["raw_report_hashes"][index],
                        "body_trace_hash": report["body_trace_hash"],
                    },
                }
            )
    teachers = []
    for donor in review["offline_learning_frontier"]["demonstration_donors"]:
        key = tuple(donor["course"])
        teachers.append(
            {
                "observation": list(contexts[key]),
                "teacher_motor_parameters": parameters[donor["donor_policy_hash"]],
                "audit_metadata": donor,
            }
        )
    unique_pairs = {
        (tuple(sample["observation"]), tuple(sample["motor_parameters"])) for sample in samples
    }
    result = {
        "schema": "soccer.rsi.causal_motor_actor_critic_bootstrap_bank.v1",
        "activation_ceiling": "SIM_ONLY",
        "partition": "TRAIN_CONSUMED",
        "observation_names": list(FEATURE_NAMES),
        "decision_frame": 30,
        "observation_dimension": 13,
        "motor_parameter_dimension": 37,
        "sample_count": len(samples),
        "unique_condition_action_count": len(unique_pairs),
        "duplicate_condition_action_count": len(samples) - len(unique_pairs),
        "distinct_consumed_course_count": len(contexts),
        "successful_teacher_course_count": len(teachers),
        "uncovered_courses": review["offline_learning_frontier"]["uncovered_courses"],
        "critic_samples": samples,
        "actor_teacher_samples": teachers,
        "independent_review_hash": review["report_hash"],
        "bank_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "feature_source_hash": hash_bytes(
            Path(__file__).with_name("precontact_proprio_policy.py").read_bytes()
        ),
        "selection_uses_future_outcome_labels": True,
        "runtime_selection_authorized": False,
        "promotion_authorized": False,
        "fresh_holdout_open_authorized": False,
        "split_unit": "whole physical course, never random frames or candidate rows",
        "limitations": [
            "world +X pilot only; not heading-invariant or multi-role qualification",
            "12 contexts, not 384 independent situations",
            "critic fitting and demonstration cloning are not online RL",
            "even complete offline teacher coverage does not prove a runnable actor",
        ],
    }
    result["report_hash"] = hash_json(result)
    return result
