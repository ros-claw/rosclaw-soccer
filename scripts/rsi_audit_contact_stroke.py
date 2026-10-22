"""Own CEM resampling and complete own-neural/native replay of all72 stroke worlds."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_stroke_search_common import (
        admit,
        make_episode,
        open_motor,
        score,
        stroke_action,
    )

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean pinned independent stroke audit source required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, dp))
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("closed ordinary source and unused independent output required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != "SHARED_STROKE72_PHYSICAL_SEARCH_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or d["protocol"] != "dual256.TRAIN.shared_joint_contact_stroke_CEM72.v1"
        or any(
            v["source_head"] != a.producer_head
            or v["hardware_authorized"] is not False
            or v["promotion_authorized"] is not False
            for v in (r, d)
        )
        or r["PPO_actor_rows"] != 0
        or r["student_success_claim"] is not False
        or d["maximum_worlds"] != 72
        or d["maximum_native_steps"] != 288000
        or d["maximum_neural_decisions"] != 28800
        or d["generations"] != 2
        or d["candidates_per_generation"] != 16
        or d["seed"] != 55023
        or d["search_contexts"] != [4, 20]
        or d["unsearched_control"] != 21
        or d["elite"] != 4
        or d["initial_spread"] != 0.4
        or d["minimum_spread"] != 0.1
        or d["maximum_rejection_draws_per_generation"] != 4096
        or d["selected_repetitions"] != 2
        or not r["whole_selected_repetitions_exact"]
    ):
        raise ValueError("complete finite teacher-only search declaration required")
    pins = dict(r["input_file_hashes"])

    def receipt(h):
        ps = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(ps) != 1:
            raise ValueError("unique exact source receipt required")
        return ps[0]

    ep, ap = receipt(d["source_exam_hash"]), receipt(d["source_audit_hash"])
    _, training_hash, contexts, centers, original_exam = admit(
        ep.parent, ap.parent, d["source_audit_hash"]
    )
    if (
        r["training_hash"] != training_hash
        or d["training_hash"] != training_hash
        or d["contexts"] != {str(i): v for i, v in contexts.items()}
        or d["centers"] != {str(i): v for i, v in centers.items()}
    ):
        raise ValueError("exact original examined fixtures and measured planning centers required")
    from rosclaw_soccer.rsi.stroke_search import verify_search_selection

    verify_search_selection(r)
    for path in (rp, dp):
        pins[str(path)] = digest(path)
    for rec in r["records"]:
        if rec["center"] != centers[rec["context"]]:
            raise ValueError("source-planned phase changed")
        for art in rec["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(a.root.resolve())
                or digest(path) != art["hash"]
            ):
                raise ValueError("confined exact complete raw trajectory required")
            pins[str(path)] = art["hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(path): digest(path) for path in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=digest(rp),
            producer_head=a.producer_head,
            worlds=72,
            input_file_hashes=pins,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, kinematic_intent=0)
    results = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world
        from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts

        bench, actor = open_motor(pins, training_hash, contexts)

        def blobs(rec):
            data = {}
            for name, art in rec["artifacts"].items():
                with np.load(a.root / art["path"], allow_pickle=False) as z:
                    data[name] = {k: z[k].copy() for k in z.files}
            return data

        for rec in r["records"]:
            i = rec["context"]
            raw = blobs(rec)
            kick, stand = raw["kick"], raw.get("stand")
            latent = np.asarray(rec["latent"], np.float64)
            if not latent.any():
                original = next(
                    x for x in original_exam["records"] if x["context"] == i and x["repeat"] == 0
                )
                if set(original["artifacts"]) != set(raw):
                    raise ValueError("zero stroke whole termination changed")
                for name, art in original["artifacts"].items():
                    path = ep.parent / art["path"]
                    if digest(path) != art["hash"]:
                        raise ValueError("exact original zero baseline required")
                    with np.load(path, allow_pickle=False) as z:
                        for k in set(z.files) & set(raw[name]):
                            equal(raw[name][k], z[k], "zero baseline " + k)
            episode = make_episode(bench, contexts[i])
            events = []
            before = dict(counts)

            def adapter(actual, mean, evidence, *, latent=latent, i=i, kick=kick):
                ref = evidence["reference_nominal"][0, 0].numpy().copy()
                action, delta, clipped = stroke_action(actual, mean, ref, centers[i], latent)
                equal(ref, kick["NN_reference_nominal"][actual.frame], "own frozen reference")
                equal(delta, kick["STROKE_delta_HG"][actual.frame], "own stroke proposal")
                equal(
                    clipped,
                    kick["STROKE_authority_clipped"][actual.frame],
                    "own total authority processing",
                )
                return action, 0

            replay_teacher_world(
                episode,
                actor,
                kick,
                stand,
                latent=np.zeros((2, 6)),
                center=centers[i],
                foot="left",
                counts=counts,
                hidden_width=256,
                action_adapter=adapter,
                contact_observer=events.append,
            )
            result, ss, good = score(raw, episode.goal)
            if (
                result != rec["result"]
                or ss != rec["stand"]
                or good != rec["healthy_quality"]
                or counts["native"] - before["native"] != rec["native_steps"]
                or counts["neural"] - before["neural"] != rec["actor_decisions"]
            ):
                raise ValueError("independent complete physical response or work differs")
            if rec["tag"].startswith("selected-"):
                source = next(
                    x for x in r["records"] if x["tag"] == f"{r['selected_tag']}-context{i}"
                )
                expected = blobs(source)
                if set(raw) != set(expected):
                    raise ValueError("selected repetition termination differs")
                for name in raw:
                    for k in raw[name]:
                        equal(raw[name][k], expected[name][k], "whole selected repeat " + k)
            if rec["tag"] == "control-context21-repeat1":
                source = next(x for x in r["records"] if x["tag"] == "control-context21-repeat0")
                expected = blobs(source)
                if set(raw) != set(expected):
                    raise ValueError("control repetition termination differs")
                for name in raw:
                    for k in raw[name]:
                        equal(raw[name][k], expected[name][k], "whole control repeat " + k)
            path = a.output / (rec["tag"] + "-contacts.json")
            write_once(path, dict(events=events, summary=summarize_contacts(events, episode.goal)))
            results.append(
                dict(
                    tag=rec["tag"],
                    context=i,
                    quality=good,
                    contact_path=path.name,
                    contact_hash=digest(path),
                )
            )
            print(
                json.dumps(
                    dict(
                        world=len(results),
                        tag=rec["tag"],
                        quality=good,
                        own_native=counts["native"],
                    )
                ),
                flush=True,
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_decisions"]
            or counts["native"] > 288000
            or counts["neural"] > 28800
            or counts["kinematic_intent"] != 0
        ):
            raise ValueError("exact bounded own work accounting required")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="SHARED_STROKE72_ALL_OWN_NATIVE_AND_CEM_SELECTION_AUDIT_PASS",
                source_head=a.expected_head,
                source_hash=digest(rp),
                input_file_hashes=pins,
                worlds=72,
                training_hash=training_hash,
                counts=counts,
                results=results,
                selected_teacher_successes=r["selected_teacher_successes"],
                unsearched_control_quality=r["unsearched_control_quality"],
                shared_proposal_and_scoring_implementation=True,
                independent_whole_world_replay=True,
                actor_updates=0,
                critic_updates=0,
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
                counts=counts,
                completed_worlds=len(results),
                automatic_retry=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
