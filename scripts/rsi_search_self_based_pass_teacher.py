"""Bounded two-body TRAIN CEM on whole learned105025; never a student update.

Four independent processes may cover the predeclared states20/23/26/29.
Reuse4 independently replayed calibration worlds per shard. Execute68 NEW
worlds:2 goals x(2 generations x16 proposals +2 whole selected repetitions).
All40 original initial states preflight before local search; no Fresh access.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("calibration-root", "calibration-own-root", "review-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("calibration-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(20, 23, 26, 29), required=True)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_self_teacher_search_common import (
        admit,
        calibrations,
        compare_common,
        open_setup,
        rank,
        verify_search,
    )

    if a.output.exists():
        raise ValueError("unused bounded new teacher-search namespace required")
    s, d, pf, pins, bank = admit(
        a.calibration_root, a.calibration_own_root, a.review_root, a.calibration_head
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.whole_learned_self_teacher_CEM68.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            calibration_head=a.calibration_head,
            calibration_root=str(a.calibration_root),
            calibration_own_root=str(a.calibration_own_root),
            review_root=str(a.review_root),
            calibration_hash=digest(a.calibration_root / "complete.json"),
            calibration_own_hash=digest(a.calibration_own_root / "complete.json"),
            review_hash=digest(a.review_root / "complete.json"),
            state_id=a.state_id,
            preflight_only=a.preflight_only,
            rng_seed_base=146023 + a.state_id * 101,
            generations_per_goal=2,
            proposals_per_generation=16,
            selected_repeats_per_goal=2,
            new_worlds=68,
            inherited_calibration_worlds=4,
            maximum_native_steps=272000,
            maximum_actor_calls=54400,
            nominal_training_hash=d["student_training_hash"],
            nominal_kind=d["student_kind"],
            original_total_authority_rad=0.2,
            formal_controls=300,
            required_A_stand=100,
            all_source_failures_retained=True,
            optimizer_updates=0,
            Fresh_cases_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
            rank="healthy+fullNNrecovery+noBexcluded,cleanfoot,actualcontrolledpass,1000*strictcomposite-error-.03*max(speed-3,0)^2",
        ),
    )
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    generations = []
    selections = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites
        from scripts.rsi_pass_transfer_trial import collect_trial

        setup, actor = open_setup(d, pf, pins, bank, a.calibration_root / "scene.xml")
        (a.output / "scene.xml").write_text(setup.xml)
        if a.preflight_only:
            check_pins(pins)
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="WHOLE_LEARNED_SELF_TEACHER_SEARCH_PREFLIGHT_NOT_PHYSICS",
                    source_head=a.expected_head,
                    input_file_hashes=pins,
                    physical_steps=0,
                    optimizer_updates=0,
                    all40_compiled_states_match=True,
                    hardware_authorized=False,
                    promotion_authorized=False,
                    goal_complete=False,
                ),
            )
            print(json.dumps(dict(stage="CEM_ALL40_PREFLIGHT", physical_steps=0)), flush=True)
            return

        def run(course, z, tag, task):
            if len(records) >= 68:
                raise ValueError("declared68-world search budget exhausted")
            c = course | dict(latent=np.asarray(z).tolist())
            row = collect_trial(
                setup,
                actor,
                c,
                output=a.output,
                index=len(records),
                counts=counts,
                emit_course_rows=False,
            )
            row.update(tag=tag, search_task=task)
            records.append(row)  # preserve charged trajectory even if identity check fails
            zero = calibrations(s, a.state_id, task)[0]
            row["prefix_identity_fields"] = compare_common(
                a.output, row, a.calibration_root, zero, prefix=80
            )
            write_once(a.output / f"trial{row['index']}.json", row)
            return dict(tag=tag, row=row, root=a.output, latent=np.asarray(z).copy())

        for task in (0, 1):
            old = calibrations(s, a.state_id, task)
            values = [
                dict(
                    tag=f"cal-t{task}-p{r['student_prefix_controls']}",
                    row=r,
                    root=a.calibration_root,
                    latent=np.array(r["course"]["latent"], np.float64),
                )
                for r in old
            ]
            incumbent = max(enumerate(values), key=lambda kv: (rank(kv[1]["row"]), -kv[0]))[1]
            mean = incumbent["latent"].copy()
            std = np.full((2, 6), 0.4)
            rng = np.random.default_rng(146023 + a.state_id * 101 + task * 7)
            course = old[1]["course"]
            for g in range(2):
                proposals, draws = sample_generation(rng, mean, std)
                values = [incumbent] + [
                    run(course, z, f"t{task}-g{g}-c{k}", task) for k, z in enumerate(proposals)
                ]
                ranks = [rank(v["row"]) for v in values]
                updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
                best = max(range(17), key=lambda k: (ranks[k], -k))
                incumbent = values[best]
                generations.append(
                    dict(
                        task=task,
                        generation=g,
                        mean_before=mean.tolist(),
                        std_before=std.tolist(),
                        draws=draws,
                        tags=[v["tag"] for v in values],
                        ranks=[list(v) for v in ranks],
                        elite=elite,
                        mean_after=updated.tolist(),
                        std_after=spread.tolist(),
                        selected=incumbent["tag"],
                    )
                )
                mean, std = updated, spread
                print(
                    json.dumps(
                        dict(
                            stage="CEM_GENERATION",
                            task=task,
                            generation=g,
                            selected_rank=list(ranks[best]),
                        )
                    ),
                    flush=True,
                )
            for repeat in range(2):
                actual = run(course, incumbent["latent"], f"t{task}-repeat{repeat}", task)
                if actual["row"]["outcome"] != incumbent["row"]["outcome"]:
                    raise ValueError("selected teacher outcome did not repeat")
                compare_common(a.output, actual["row"], incumbent["root"], incumbent["row"])
            selections.append(
                dict(
                    task=task,
                    selected=incumbent["tag"],
                    latent=incumbent["latent"].tolist(),
                    rank=list(rank(incumbent["row"])),
                    outcome=incumbent["row"]["outcome"],
                )
            )
        if len(records) != 68 or counts["native_steps"] > 272000 or counts["actor_calls"] > 54400:
            raise ValueError("whole bounded search required")
        verify_search(
            dict(
                state_id=a.state_id, records=records, generations=generations, selections=selections
            ),
            s,
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="WHOLE_LEARNED_SELF_TEACHER_SEARCH_COLLECTED_PENDING_ALL68_OWN",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                scene_hash=digest(a.output / "scene.xml"),
                state_id=a.state_id,
                counts=counts,
                records=records,
                generations=generations,
                selections=selections,
                optimizer_updates=0,
                Fresh_cases_read=0,
                teacher_search_is_neural_training=False,
                correction_bank_ready=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as e:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(e).__name__,
                error=str(e),
                counts=counts,
                records=records,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
