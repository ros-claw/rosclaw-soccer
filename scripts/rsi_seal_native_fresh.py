"""Seal a disjoint reference-motion Fresh bank after weights and eligibility freeze."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-hash", "frontier-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.native_fresh_curriculum import (
        FRESH_PROTOCOL,
        FRESH_SEED,
        native_fresh_courses,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import (
        admit_frontier,
        clean,
        open_fresh_actors,
        source_pins,
    )

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused private Fresh namespace required")
    pins, baseline, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    courses = native_fresh_courses(contexts)
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(mode=0o700)
    write_once(a.output / "private-courses.json", courses)
    declaration = dict(
        protocol=FRESH_PROTOCOL,
        sampling_seed=FRESH_SEED,
        source_head=a.expected_head,
        input_file_hashes=pins,
        frontier_hash=a.frontier_hash,
        frontier_head=a.frontier_head,
        baseline_training_hash=baseline,
        candidates=candidates,
        courses_hash=digest(a.output / "private-courses.json"),
        courses=56,
        base40=40,
        run16=16,
        all_candidate_weights_frozen_before_sampling=True,
        worker_count=4,
        eligibility_rule="ALL_SEEDS_WITH_ZERO_CUMULATIVE_RETENTION_LOSS_NOT_BEST_SCORE",
        course_sampling_adapts_to_results=False,
        FRESH_fit_rows=0,
        screen_rule=(
            "base_net>=0 and run_net>=4 and every_condition_net>=0 "
            "and zero_new_falls_or_forbidden_contacts"
        ),
        legacy52_equivalence=False,
        goal_plane_precision_proven=False,
        team_game_proven=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration)
    try:
        from dataclasses import asdict

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from scripts.rsi_stroke_search_common import make_episode

        bench, _ = open_fresh_actors(pins, declaration, courses)
        for c in courses:
            make_episode(bench, c["ctx"])
        if bench.native_steps != 0:
            raise ValueError("all56 full preflights must precede first physical step")
        dimensions = inspect_native_ball_dimensions(
            bench.template.model, geom_id=int(bench.template.model.geom("soccer_ball_geom").id)
        )
        if not dimensions.size_and_mass_in_ifab_range:
            raise ValueError("frozen source ball dimensions changed")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                declaration,
                status="NATIVE_FRESH56_SEALED_AFTER_ALL_WEIGHTS_FROZEN",
                declaration_hash=digest(a.output / "declaration.json"),
                physical_steps=0,
                fit_rows=0,
                all56_preflighted_before_worker_allocation=True,
                compiled_ball=asdict(dimensions),
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                physical_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
