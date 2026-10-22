"""One of four independent OS processes collecting or replaying a sealed Fresh bank.

No multiprocessing spawn, raw weight fitting, adaptive courses, or retries.
The audit mode owns its recurrent history and recomputes every PD/native step.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("seal-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--producer-root", type=Path)
    p.add_argument("--mode", choices=("collect", "audit"), required=True)
    p.add_argument("--shard", type=int, choices=(0, 1, 2, 3), required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, open_fresh_actors, read_seal, source_pins

    clean(a.expected_head)
    if a.output.exists() or (a.mode == "audit") != (a.producer_root is not None):
        raise ValueError("unused exact typed native Fresh worker output required")
    seal, pins, courses = read_seal(a.seal_root, a.expected_head)
    own = [c for c in courses if c["index"] % 4 == a.shard]
    arms = ["baseline"] + [str(s["seed"]) for s in seal["candidates"]]
    if len(own) != 14 or len(courses) != 56:
        raise ValueError("four complete disjoint fourteen-course shards required")
    production = None
    if a.mode == "audit":
        pp = a.producer_root / "complete.json"
        if pp.is_symlink() or (a.producer_root / "failed.json").exists():
            raise ValueError("complete unmodified producer shard required")
        production = json.loads(pp.read_text())
        if (
            production["status"] != "NATIVE_FRESH_SHARD_COLLECTED_PENDING_OWN_REPLAY"
            or production["source_head"] != a.expected_head
            or production["shard"] != a.shard
            or production["seal_hash"] != digest(a.seal_root / "complete.json")
            or len(production["records"]) != 14 * len(arms)
        ):
            raise ValueError("complete same-shard source worlds required")
        pins[str(pp)] = digest(pp)
        for row in production["records"]:
            for art in row["artifacts"].values():
                path = a.producer_root / art["path"]
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(a.producer_root.resolve())
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("exact confined private Fresh raw required")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(mode=0o700, parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            mode=a.mode,
            shard=a.shard,
            source_head=a.expected_head,
            input_file_hashes=pins,
            seal_hash=digest(a.seal_root / "complete.json"),
            course_indices=[c["index"] for c in own],
            arms=arms,
            maximum_native_steps=14 * len(arms) * 4000,
            maximum_actor_decisions=14 * len(arms) * 400,
            teacher_enabled=False,
            FRESH_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, kinematic_intent=0)
    records = []
    bench = None
    try:
        import numpy as np

        from rosclaw_soccer.rsi.motor_reference_rollout import collect_parent_reference
        from rosclaw_soccer.rsi.paid_run_teacher_audit import replay_teacher_world
        from scripts.rsi_stroke_search_common import make_episode, score

        bench, actors = open_fresh_actors(pins, seal, courses)
        # The seal preflight preceded worker allocation; each worker also checks
        # the entire declaration before integrating its own first course.
        for c in courses:
            make_episode(bench, c["ctx"])
        if bench.native_steps != 0:
            raise ValueError("no physics before full sealed-bank preflight")
        lookup = (
            {(r["course_index"], r["arm"]): r for r in production["records"]} if production else {}
        )
        if production and set(lookup) != {(c["index"], arm) for c in own for arm in arms}:
            raise ValueError("no omitted, duplicated or reassigned Fresh source worlds")
        for course in own:
            for arm in arms:
                ep = make_episode(bench, course["ctx"])
                before = dict(counts)
                foot = "left" if "left" in course["ctx"]["case"]["source"] else "right"
                if a.mode == "collect":
                    kick, stand, _, calls = collect_parent_reference(
                        ep, actors[arm], foot=foot, hidden_width=256
                    )
                    blobs = dict(kick=kick)
                    if stand is not None:
                        blobs["stand"] = stand
                    counts["neural"] += calls
                    counts["native"] += sum(len(z["qpos"]) * 10 for z in blobs.values())
                    folder = a.output / f"course{course['index']}-{arm}"
                    folder.mkdir()
                    artifacts = {}
                    for name, z in blobs.items():
                        path = folder / (name + ".npz")
                        np.savez_compressed(path, **z)
                        artifacts[name] = dict(
                            path=str(path.relative_to(a.output)), hash=digest(path)
                        )
                else:
                    claimed = lookup[course["index"], arm]
                    blobs = {}
                    for name, art in claimed["artifacts"].items():
                        with np.load(a.producer_root / art["path"], allow_pickle=False) as z:
                            blobs[name] = {k: z[k].copy() for k in z.files}
                    replay_teacher_world(
                        ep,
                        actors[arm],
                        blobs["kick"],
                        blobs.get("stand"),
                        latent=np.zeros((2, 6), np.float64),
                        center=105,
                        foot=foot,
                        counts=counts,
                        hidden_width=256,
                    )
                    artifacts = claimed["artifacts"]
                result, stand_summary, good = score(
                    blobs, ep.goal, intended_foot=0 if foot == "left" else 1
                )
                row = dict(
                    course_index=course["index"],
                    arm=arm,
                    cohort=course["cohort"],
                    condition=course["condition"],
                    healthy_quality=good,
                    body_failed=result["body_failed"],
                    forbidden=result["forbidden"],
                    result=result,
                    stand=stand_summary,
                    artifacts=artifacts,
                    native_steps=counts["native"] - before["native"],
                    actor_decisions=counts["neural"] - before["neural"],
                    unassisted=True,
                )
                if a.mode == "audit" and row != claimed:
                    raise ValueError("own native physics, private NN history or result differs")
                records.append(row)
                write_once(a.output / f"world{course['index']}-{arm}.json", row)
                print(
                    json.dumps(
                        dict(
                            shard=a.shard,
                            mode=a.mode,
                            course_index=course["index"],
                            arm=arm,
                            quality=good,
                            actual_native=counts["native"],
                        )
                    ),
                    flush=True,
                )
                if (
                    counts["native"] > 14 * len(arms) * 4000
                    or counts["neural"] > 14 * len(arms) * 400
                ):
                    raise ValueError("bounded actual matched Fresh world budget exceeded")
        if a.mode == "collect" and bench.native_steps != counts["native"]:
            raise ValueError("actual producer physical accounting differs")
        if a.mode == "audit" and (
            counts["native"] != production["counts"]["native"]
            or counts["neural"] != production["counts"]["neural"]
        ):
            raise ValueError("all source native/neural world work must be replayed")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_FRESH_SHARD_COLLECTED_PENDING_OWN_REPLAY"
                if a.mode == "collect"
                else "NATIVE_FRESH_SHARD_ALL_OWN_NATIVE_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                shard=a.shard,
                records=records,
                counts=counts,
                seal_hash=digest(a.seal_root / "complete.json"),
                producer_hash=digest(a.producer_root / "complete.json") if production else None,
                independent_whole_world_replay=a.mode == "audit",
                shared_observation_codec_and_scoring=True,
                optimizer_updates=0,
                FRESH_fit_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                completed_worlds=len(records),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
