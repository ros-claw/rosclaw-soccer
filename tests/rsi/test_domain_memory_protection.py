"""Synthetic contract tests only; fake summaries are not physical evidence."""

import copy
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth.domain_anchor_bank import build_domain_anchor_bank

from rosclaw_soccer.rsi import domain_memory_protection as protection
from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import (
    CompiledProposalMemoryMotor,
    initial_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_phase_balanced_proposal_motor import imbalanced_complete_batch
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def seal(value):
    value.pop("report_hash", None)
    value["report_hash"] = hash_json(value)
    return value


def synthetic_protection(initial, states):
    legacy = initial["baseline"]
    manifest, memory = legacy["consolidation_manifest"], legacy["memory"]
    rows = []
    for i in range(52):
        report = "sha256:" + f"{i:064x}"
        outcome = seal(
            dict(
                high_quality=i == 0,
                clean_foot_only=True,
                safety_passed=True,
                physical_substeps=3000,
                reviewed_report_hash=report,
                actual_mujoco_dynamics_replayed=True,
                actual_pd_torque_reconstructed=True,
                neural_target_reconstructed=True,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        )
        rows.append(
            dict(
                index=i,
                seed=42 + i,
                lane=0,
                report_hash=report,
                review_hash=outcome["report_hash"],
                outcome=outcome,
            )
        )
    commitment = seal(
        dict(
            schema="soccer.rsi.cpu_retained_parent_full_coverage_commitment.v1",
            model_hash=initial["parent_model_hash"],
            partition="TRAIN_CONSUMED_CPU_DOMAIN",
            courses=[[r["seed"], r["lane"]] for r in rows],
            physics_change=False,
            fresh_exam_authorized=False,
            learning_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )
    summary = seal(
        dict(
            schema="soccer.rsi.cpu_retained_parent_full_coverage.v1",
            rows=rows,
            commitment_hash=commitment["report_hash"],
            independent_contexts=52,
            all_physical_substeps_replayed=156000,
            high_quality=1,
            fresh_exam_authorized=False,
            learning_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    )
    domains = [
        dict(
            domain_id="inherited-gpu-memory",
            source_evidence_hash=manifest["report_hash"],
            context_ids=["intact-legacy-memory"],
            context_rows=[len(memory["observations"])],
            observations=memory["observations"],
        ),
        dict(
            domain_id="cpu-retained-parent",
            source_evidence_hash=summary["report_hash"],
            context_ids=["seed42-lane0"],
            context_rows=[270],
            observations=states,
        ),
    ]
    encoded = build_domain_anchor_bank(
        domains, parent_hash=initial["parent_model_hash"], encoder_hash=memory["encoder_hash"]
    )
    record = {k: rows[0][k] for k in ("index", "seed", "lane", "report_hash", "review_hash")}
    return seal(
        dict(
            schema=protection.SCHEMA,
            source_hash=hash_bytes(Path(protection.__file__).read_bytes()),
            selection="ALL_SEALED_CPU_SUCCESSES_PLUS_INTACT_INHERITED_GPU_MEMORY",
            bank=encoded,
            cpu_summary=summary,
            cpu_commitment=commitment,
            cpu_records=[dict(**record, frames=270, state_hash=hash_json(states))],
            **dict.fromkeys(protection.FLAGS, False),
        )
    )


def test_multidomain_guard_keeps_cross_domain_context_names_and_exact_base_output(current):  # noqa: F811
    actor, _ = current
    plain = CompiledProposalMemoryMotor(make_preview(initial_model(actor, maximum_mean_kl=0.05)))
    x = np.full(134, 0.73)
    states = np.tile(np.append(plain.features(x)[:134], 1), (270, 1)).tolist()
    bundle = synthetic_protection(actor, states)
    proposal = initial_model(actor, maximum_mean_kl=0.05, protected_domain_bank=bundle)
    decoder = CompiledProposalMemoryMotor(make_preview(proposal))
    assert decoder._guard.gate(states[0]) == 0
    assert np.array_equal(decoder.raw_mean(x, 1), plain.raw_mean(x, 1))
    assert (
        decoder._guard.bank()["row_count"] == len(actor["baseline"]["memory"]["observations"]) + 270
    )
    bundle["bank"]["domains"][1]["observations"][0][0] = 99
    assert decoder._guard.gate(states[0]) == 0
    validate_model(proposal)


def test_learning_receipt_and_execution_use_the_same_all_domain_guard(current, smooth_parent):  # noqa: F811
    actor, _ = current
    data = imbalanced_complete_batch(smooth_parent)
    plain = CompiledProposalMemoryMotor(make_preview(initial_model(actor, maximum_mean_kl=0.05)))
    x, phase = data["observation"][0], int(data["phase_index"][0])
    states = np.tile(np.append(plain.features(x)[:134], phase), (270, 1)).tolist()
    bundle = synthetic_protection(actor, states)
    parent = initial_model(actor, maximum_mean_kl=0.05, protected_domain_bank=bundle)
    learned = fit_update(parent, data, batch_hash="sha256:" + "a" * 64)
    decoder = CompiledProposalMemoryMotor(make_preview(learned))
    assert np.array_equal(decoder.raw_mean(x, phase), plain.raw_mean(x, phase))
    receipt = learned["learning_receipt"]
    assert receipt["protected_memory_hash"] == bundle["bank"]["bank_hash"]
    assert receipt["protected_memory_rows"] == bundle["bank"]["row_count"]
    assert receipt["protected_anchor_contexts"] == 2
    forged = copy.deepcopy(learned)
    forged["protected_domain_bank"] = None
    forged.pop("model_hash")
    forged["model_hash"] = hash_json(forged)
    with pytest.raises(ValueError, match="regression receipt"):
        validate_model(forged)


@pytest.mark.parametrize("fault", ["subset", "parent", "authority", "state"])
def test_incomplete_or_unbound_protection_rejected(current, fault):  # noqa: F811
    actor, _ = current
    states = np.zeros((270, 135)).tolist()
    bundle = synthetic_protection(actor, states)
    if fault == "subset":
        bundle["cpu_records"] = []
    elif fault == "parent":
        bundle["cpu_commitment"]["model_hash"] = "sha256:" + "f" * 64
        seal(bundle["cpu_commitment"])
    elif fault == "authority":
        bundle["hardware_authorized"] = True
    else:
        bundle["cpu_records"][0]["state_hash"] = "sha256:" + "f" * 64
    seal(bundle)
    with pytest.raises(ValueError):
        initial_model(actor, maximum_mean_kl=0.05, protected_domain_bank=bundle)
