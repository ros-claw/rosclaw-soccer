"""Single-process, bounded shared-stroke teacher CEM with actual neural recovery.

72 maximum native worlds. No new PPO/student qualification; every failure is
retained. No spawn workers or mutable shared per-player recurrent context.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("exam-root", "audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("audit-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_stroke_search_common import admit, collect, open_motor

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("exact clean frozen stroke search source required")

    clean()
    if a.output.exists():
        raise ValueError("unused search output required")
    pins, training_hash, contexts, centers, exam = admit(a.exam_root, a.audit_root, a.audit_hash)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    from rosclaw_soccer.rsi.stroke_search import (
        CANDIDATES,
        CONTEXTS,
        CONTROL,
        GENERATIONS,
        SEED,
        WORLDS,
    )

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.TRAIN.shared_joint_contact_stroke_CEM72.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            training_hash=training_hash,
            source_exam_hash=digest(a.exam_root / "complete.json"),
            source_audit_hash=a.audit_hash,
            contexts={str(k): v for k, v in contexts.items()},
            centers={str(k): v for k, v in centers.items()},
            search_contexts=list(CONTEXTS),
            unsearched_control=CONTROL,
            generations=GENERATIONS,
            candidates_per_generation=CANDIDATES,
            seed=SEED,
            maximum_worlds=WORLDS,
            maximum_native_steps=WORLDS * 4000,
            maximum_neural_decisions=WORLDS * 400,
            execution="single_native_process_private_history_per_world",
            initial_spread=0.4,
            minimum_spread=0.1,
            elite=4,
            maximum_rejection_draws_per_generation=4096,
            selected_repetitions=2,
            teacher_assisted=True,
            PPO_actor_rows=0,
            student_success_claim=False,
            automatic_retry=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    records, generations = [], []
    bench = None
    try:
        import numpy as np

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.stroke_search import pair_rank, sample_generation, update_elites

        bench, actor = open_motor(pins, training_hash, contexts)
        baselines = {}
        for i in contexts:
            old = next(r for r in exam["records"] if r["context"] == i and r["repeat"] == 0)
            baselines[i] = {}
            for name in ("kick", "stand"):
                if name not in old["artifacts"]:
                    continue
                art = old["artifacts"][name]
                path = a.exam_root / art["path"]
                if digest(path) != art["hash"]:
                    raise ValueError("exact original baseline required")
                with np.load(path, allow_pickle=False) as z:
                    baselines[i][name] = {k: z[k].copy() for k in z.files}

        def world(i, z, tag):
            if len(records) >= WORLDS:
                raise ValueError("bounded physical search exhausted")
            ctx = dict(contexts[i], context_id=i)
            row, blobs = collect(
                bench, actor, ctx, centers[i], z, a.output / tag, baseline=baselines[i]["kick"]
            )
            if not z.any():
                if set(blobs) != set(baselines[i]):
                    raise ValueError("zero stroke must preserve source whole termination")
                for name, raw in blobs.items():
                    source = baselines[i][name]
                    for key in set(raw) & set(source):
                        if not np.array_equal(raw[key], source[key]):
                            raise ValueError(
                                "zero stroke is not bit-exact source physics/history: " + key
                            )
            write_once(a.output / tag / "complete.json", row)
            records.append(row)
            print(
                json.dumps(
                    dict(
                        world=len(records),
                        tag=tag,
                        quality=row["healthy_quality"],
                        forbidden=row["result"]["forbidden"],
                        native=bench.native_steps,
                    )
                ),
                flush=True,
            )
            if (
                bench.native_steps > WORLDS * 4000
                or sum(r["actor_decisions"] for r in records) > WORLDS * 400
            ):
                raise ValueError("actual physics/neural budget exceeded")
            return row, blobs

        def pair(z, tag):
            values = [world(i, z, f"{tag}-context{i}") for i in CONTEXTS]
            return dict(
                tag=tag,
                latent=z.copy(),
                rows=[v[0] for v in values],
                blobs=[v[1] for v in values],
                rank=pair_rank([v[0] for v in values]),
            )

        incumbent = pair(np.zeros((2, 6)), "baseline")
        rng = np.random.default_rng(SEED)
        mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
        for g in range(GENERATIONS):
            samples, draws = sample_generation(rng, mean, std)
            candidates = [incumbent] + [
                pair(z, f"g{g}-candidate{k}") for k, z in enumerate(samples)
            ]
            ranks = [c["rank"] for c in candidates]
            updated, spread, elite = update_elites([c["latent"] for c in candidates], ranks)
            best = max(range(len(candidates)), key=lambda k: (ranks[k], -k))
            report = dict(
                generation=g,
                mean_before=mean.tolist(),
                std_before=std.tolist(),
                draws=draws,
                tags=[c["tag"] for c in candidates],
                ranks=[list(r) for r in ranks],
                elite=elite,
                mean_after=updated.tolist(),
                std_after=spread.tolist(),
                selected=candidates[best]["tag"],
            )
            write_once(a.output / f"generation{g}.json", report)
            generations.append(report)
            incumbent = candidates[best]
            mean, std = updated, spread
        for slot, i in enumerate(CONTEXTS):
            for repeat in range(2):
                row, blobs = world(i, incumbent["latent"], f"selected-context{i}-repeat{repeat}")
                expected = incumbent["blobs"][slot]
                if set(blobs) != set(expected):
                    raise ValueError("selected whole termination differs")
                for name in blobs:
                    identical(blobs[name], expected[name])
        control = []
        for repeat in range(2):
            control.append(
                world(CONTROL, incumbent["latent"], f"control-context{CONTROL}-repeat{repeat}")
            )
        if set(control[0][1]) != set(control[1][1]):
            raise ValueError("control repetition differs")
        for name in control[0][1]:
            identical(control[0][1][name], control[1][1][name])
        check_pins(pins)
        clean()
        if len(records) != WORLDS:
            raise ValueError("complete declared physical search required")
        write_once(
            a.output / "complete.json",
            dict(
                status="SHARED_STROKE72_PHYSICAL_SEARCH_CLOSED_PENDING_INDEPENDENT_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                training_hash=training_hash,
                records=records,
                generations=generations,
                selected_tag=incumbent["tag"],
                selected_latent=incumbent["latent"].tolist(),
                selected_rank=list(incumbent["rank"]),
                selected_teacher_successes=sum(r["healthy_quality"] for r in incumbent["rows"]),
                unsearched_control_quality=control[0][0]["healthy_quality"],
                actual_native_steps=bench.native_steps,
                actual_neural_decisions=sum(r["actor_decisions"] for r in records),
                whole_selected_repetitions_exact=True,
                PPO_actor_rows=0,
                student_success_claim=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed_worlds=len(records),
                actual_native_steps=bench.native_steps if bench else 0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
