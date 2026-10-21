"""Extend an independently qualified learned contact policy to 2..4 skills.

The first two complete physical histories MUST reproduce their qualified
parent arrays. New successor states are genuine continuing physics, never
reset poses. This is practice with a declared ball return, not a team game.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exam", "primitive", "neural"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--skills", type=int, choices=(2, 3, 4), default=3)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean frozen chain source required")
    paths = [root / "complete.json" for root in (a.exam_root, a.primitive_root, a.neural_root)]
    hashes = [a.exam_hash, a.primitive_hash, a.neural_hash]
    if any(digest(f) != h for f, h in zip(paths, hashes, strict=True)):
        raise ValueError("all exact completed parent proofs required")
    exam, primitive, neural = [json.loads(f.read_text()) for f in paths]
    if (
        exam["status"] != "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED"
        or primitive["status"] != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
        or neural["status"] != "PAID_GATED_CONTACT_ALL_NEURAL_PROPOSALS_EXACT"
        or primitive["input_file_hashes"].get(str(paths[0])) != a.exam_hash
        or neural["input_file_hashes"].get(str(paths[0])) != a.exam_hash
        or neural["input_file_hashes"].get(str(paths[1])) != a.primitive_hash
        or neural["own_neural_rows"] != exam["actual_neural_rows"]
        or len(exam["candidates"]) != 1
        or exam["contact_residual_trained"] is not True
    ):
        raise ValueError("complete independently audited physical AND own-neural parent required")
    candidate = exam["candidates"][0]
    if (
        candidate["full_continuous_skill_gate"] is not True
        or candidate["continuous"]["quality_regressions"]
        or candidate["continuous"]["stability_regressions"]
        or candidate["continuous"]["stable"] != 16
    ):
        raise ValueError("only protected actually learned two-skill parents may advance")
    pins = dict(neural["input_file_hashes"])
    pins.update({str(f): h for f, h in zip(paths, hashes, strict=True)})
    dp = a.exam_root / "declaration.json"
    if pins.get(str(dp)) != digest(dp):
        raise ValueError("exact parent course declaration required")
    parent_d = json.loads(dp.read_text())
    cases = parent_d["old_cases"]
    if len(cases) != 16:
        raise ValueError("all16 frozen practice conditions required")
    root = Path(parent_d["candidates"][candidate["candidate"]])
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=cases,
            skills=a.skills,
            independent_full_world_repeats=2,
            first_two_arrays_must_match_parent=True,
            fixed_world_goal=True,
            body_resets_between_skills=0,
            first_actor="qualified_R2motor",
            successor_actor="qualified_R2motor_plus_learned_contact_gate_and_residual",
            checkpoint_hash=candidate["checkpoint_hash"],
            training_receipt_hash=candidate["training_receipt_hash"],
            all16_cases_retained=True,
            private_recurrent_h0_each_skill=True,
            actual_previous_action_carried=True,
            practice_ball_return_is_not_teammate_pass=True,
            team_game=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_practice_chain import run_practice_chain
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene_hash = open_paid_bench(pins, cases)
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=root / "contact-residual-epoch256.pt",
            checkpoint_hash=candidate["checkpoint_hash"],
            training_receipt=root / "complete.json",
            training_receipt_hash=candidate["training_receipt_hash"],
            scene_hash=scene_hash,
        )

        def same(x, y):
            if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
                raise ValueError("whole native/neural physical history differs")

        def read(row, parent=False):
            source = a.exam_root if parent else a.output
            path = source / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(source.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact trajectory required")
            if parent:
                pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        result = {}
        for name, actors, parent in (
            ("baseline", [actor.base] * a.skills, exam["baseline_continuous"]),
            (
                "learned",
                [actor.base] + [actor] * (a.skills - 1),
                candidate["continuous"]["results"],
            ),
        ):
            worlds = []
            for i, case in enumerate(cases):
                replicas = []
                for repeat in range(2):
                    folder = a.output / name / f"case{i}-repeat{repeat}"
                    folder.mkdir(parents=True)
                    artifacts = []
                    prefix_references = {
                        f"skill{k}": read(parent[i][key], True)
                        for k, key in enumerate(("first", "second"))
                    }

                    def store(
                        key, raw, folder=folder, artifacts=artifacts, references=prefix_references
                    ):
                        if key in references:
                            # Fail before allocating any later skill, not after
                            # spending new physics on an unqualified prefix.
                            same(raw, references[key])
                        path = folder / (key + ".npz")
                        if path.exists():
                            raise ValueError("trajectory overwrite forbidden")
                        np.savez_compressed(path, **raw)
                        artifact = dict(
                            raw_path=str(path.relative_to(a.output)), raw_hash=digest(path)
                        )
                        artifacts.append(artifact)
                        return artifact

                    world = run_practice_chain(bench, case, actors, store=store)
                    if len(world["skills"]) < 2:
                        raise ValueError("qualified two-skill parent did not reproduce")
                    for k, key in enumerate(("first", "second")):
                        same(read(world["skills"][k]["result"]), read(parent[i][key], True))
                    world["case"] = i
                    world["artifacts"] = artifacts
                    world["quality_by_skill"] = [
                        bool(contact_rank([s["result"], s["result"]])[1] > 0)
                        for s in world["skills"]
                    ]
                    write_once(folder / "complete.json", world)
                    replicas.append(world)
                if replicas[0]["status"] != replicas[1]["status"] or len(
                    replicas[0]["artifacts"]
                ) != len(replicas[1]["artifacts"]):
                    raise ValueError("full-chain branch repeats differ")
                for x, y in zip(replicas[0]["artifacts"], replicas[1]["artifacts"], strict=True):
                    same(read(x), read(y))
                worlds.append(dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"]))
                print(
                    json.dumps(
                        dict(
                            arm=name,
                            case=i,
                            status=worlds[-1]["status"],
                            quality_by_skill=worlds[-1]["quality_by_skill"],
                        )
                    ),
                    flush=True,
                )
            result[name] = dict(
                worlds=worlds,
                stable=sum(w["world_stable"] for w in worlds),
                quality_by_skill=[
                    sum(len(w["quality_by_skill"]) > k and w["quality_by_skill"][k] for w in worlds)
                    for k in range(a.skills)
                ],
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTINUING_CHAIN_CPU_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=result,
                skills=a.skills,
                all_first_two_arrays_exact_parent=True,
                all_independent_chain_repeats_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                goal_complete=False,
                team_game=False,
                fresh_exam=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_native_steps=0 if bench is None else bench.native_steps,
                actual_neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
