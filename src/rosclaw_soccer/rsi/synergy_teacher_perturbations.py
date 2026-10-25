"""Outcome-independent local robustness courses for a qualified SIM teacher.

Conditional proposal intervals are declared BEFORE sampling. No generated
course is clipped, replaced, or dropped. Not sealed Fresh or broad robustness.
"""

from dataclasses import asdict
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.training.receiving_course_preflight import (
    FirstTouchCourse,
    preflight_receiving_courses,
)


def declare_synergy_perturbations(
    originals: tuple[FirstTouchCourse, ...], *, seed: int
) -> tuple[tuple[FirstTouchCourse, ...], dict[str, Any]]:
    if type(originals) is not tuple or len(originals) != 40:
        raise ValueError("complete40 public source contexts required")
    source_hash = preflight_receiving_courses(originals)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("explicit uint32 local robustness seed required")
    buckets = [validate_public_course(c.course) for c in originals]
    if [buckets.count(i) for i in range(4)] != [10] * 4:
        raise ValueError("balanced ten-per-class source declaration required")
    rng = np.random.default_rng(seed)
    records, courses = [], []
    for i, source in enumerate(originals):
        x, y, vx = source.course
        xmin, xmax = (2.2, 2.4) if x < 2.4 else (2.6, 2.8)
        vmin, vmax = (0.25, 0.7) if vx > 0 else (-0.7, -0.25)
        # Same conditional-domain convention as first_touch_mechanism_expansion.
        ranges = (
            (max(xmin, x - 0.005), min(xmax, x + 0.005)),
            (max(-0.16, y - 0.01), min(0.16, y + 0.01)),
            (max(vmin, vx - 0.025), min(vmax, vx + 0.025)),
        )
        values = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
        row = (values[0], values[1], values[2])
        if validate_public_course(row) != buckets[i]:
            raise ValueError("reject entire robustness declaration; no clipping or replacement")
        course = FirstTouchCourse(source.agent_id, 2026102800 + i, row)
        courses.append(course)
        records.append(
            dict(
                source_context_index=i,
                source=asdict(source),
                proposal_ranges=ranges,
                perturbed=asdict(course),
            )
        )
    declared = tuple(courses)
    bank_hash = preflight_receiving_courses(declared)
    if set(c.course for c in declared) & set(c.course for c in originals):
        raise ValueError("perturbations must be disjoint from all nominal source states")
    return declared, dict(
        schema="soccer.synergy_teacher_local_perturbations.v1",
        seed=seed,
        source_bank_hash=source_hash,
        course_bank_hash=bank_hash,
        records=records,
        maximum_delta=(0.005, 0.01, 0.025),
        conditional_domain_before_draw=True,
        all_courses_preflighted=True,
        source_context_clusters=40,
        distinct_perturbed_states=40,
        private_fresh_accessed=False,
        no_outcome_based_sampling=True,
        hardware_authorized=False,
    )


def declare_feedback_validation(
    originals: tuple[FirstTouchCourse, ...],
) -> tuple[tuple[FirstTouchCourse, ...], dict[str, Any]]:
    """One preregistered local bank, disjoint from both consumed development banks.

    There is intentionally no seed argument: this bank must not become an
    outcome-driven seed search. Construction alone never authorizes physics,
    student training, promotion, or a claim of sealed Fresh generalization.
    """
    old, old_report = declare_synergy_perturbations(originals, seed=2026102519)
    new, declaration = declare_synergy_perturbations(originals, seed=2026102529)
    consumed = {c.course for c in (*originals, *old)}
    if consumed & {c.course for c in new}:
        raise ValueError("reject whole local validation bank overlapping consumed development")
    # Different identity namespace too; identities never enter policy observations.
    declared = tuple(
        FirstTouchCourse(c.agent_id, 2026102900 + i, c.course) for i, c in enumerate(new)
    )
    for i, record in enumerate(declaration["records"]):
        record["perturbed"] = asdict(declared[i])
    declaration.update(
        schema="soccer.feedback_teacher_disjoint_local_validation.v1",
        course_bank_hash=preflight_receiving_courses(declared),
        consumed_perturbation_seed=2026102519,
        consumed_perturbation_bank_hash=old_report["course_bank_hash"],
        disjoint_consumed_states=len(consumed),
        disjoint_from_both_development_banks=True,
        validation_seed_not_configurable=True,
        teacher_development_audits_still_required=True,
        student_training_authorized=False,
        promotion_authorized=False,
    )
    return declared, declaration
