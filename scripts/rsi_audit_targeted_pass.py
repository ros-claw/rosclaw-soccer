"""Whole72 own-history native replay and CEM resampling of pass teacher pilot.

Shares admitted observation codec and stroke rule on OWN states. Does NOT
execute saved actions or borrow h/c, producer integrator, or stand driver.
One-body proof only; two-body transfer is a separate required experiment.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "producer-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp, dp, pp = [a.root / n for n in ("complete.json", "declaration.json", "preflight.json")]
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp, pp))
        or (a.root / "failed.json").exists()
        or digest(rp) != a.source_hash
    ):
        raise ValueError("exact completed source and unused native audit root required")
    s, d, pf = [json.loads(x.read_text()) for x in (rp, dp, pp)]
    if (
        s["status"] != "TARGETED_PASS_CEM72_COLLECTED_PENDING_OWN_REPLAY_AND_TEAM_TRANSFER"
        or d["protocol"] != "soccer.targeted_pass_joint_teacher_CEM72.v1"
        or s["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or s["input_file_hashes"] != d["input_file_hashes"]
        or len(s["records"]) != 72
        or d["maximum_worlds"] != 72
        or d["maximum_native_steps"] != 288000
        or d["maximum_actor_calls"] != 28800
        or d["generations"] != 2
        or d["candidates_per_generation"] != 16
        or d["selected_repetitions"] != 2
        or d["context"] != s["context"]
        or d["seed"] != 66023 + s["context"] * 101
        or d["goals"] != [dict(distance_m=3.75, lateral_m=y) for y in (0.0, 0.2)]
        or pf["native_steps"] != 0
        or any(
            r[k] is not False
            for r in (s, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete bounded72-world unpromoted pass teacher study required")
    pins = dict(s["input_file_hashes"])
    for path in (rp, dp, pp):
        pins[str(path)] = digest(path)
    for r in s["records"]:
        for art in r["artifacts"].values():
            path = a.root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (a.root / r["tag"]).resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("exact confined whole raw teacher streams required")
            pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            source_preflight_reconstructed_not_blindly_trusted=True,
            producer_v1_did_not_hash_preflight_in_complete=True,
            source_actions_executed=False,
            borrowed_memory=False,
            shared_codecs_and_stroke_on_own_states=True,
            producer_integrator_used=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, kinematic_intent=0)
    records = []
    try:
        import numpy as np

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.targeted_pass_precision import targeted_pass_precision
        from rosclaw_soccer.rsi.targeted_pass_search_audit import verify_targeted_search
        from scripts.rsi_stroke_search_common import open_motor
        from scripts.rsi_stroke_world_audit import replay_stroke_world

        base_hash = d["baseline_training_hash"]
        audits = [
            json.loads(Path(path).read_text())
            for path in pins
            if path.endswith("/native-audit/complete.json")
        ]
        audits = [r for r in audits if r.get("training_hash") == base_hash]
        if len(audits) != 1:
            raise ValueError("one pinned original baseline context proof required")
        source_path, native = receipt(pins, audits[0]["source_hash"])
        context_path = source_path.parent / "declaration.json"
        if pins[str(context_path)] != digest(context_path) or native["declaration_hash"] != digest(
            context_path
        ):
            raise ValueError("exact original native contexts required")
        original = json.loads(context_path.read_text())["contexts"][s["context"]]
        bench, base = open_motor(pins, base_hash, {s["context"]: original})
        c = d["candidate"]
        _, training = receipt(pins, c["training_hash"])
        if c["seed"] != 65023 or c["kind"] != "contact-consolidated-v2":
            raise ValueError("same retained motor identity required")
        actor, _ = load_dual_candidate_for_sim_audit(
            base.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(c["training_root"]),
            expected_receipt_hash=c["training_hash"],
            checkpoint_kind=c["kind"],
        )
        ep = bench.make(original["case"])
        q, v = ep.layout.canonical(ep.env.data)
        forward = ep.goal[:2] - q[36:38]
        forward /= np.linalg.norm(forward)
        lateral = np.array([-forward[1], forward[0]])
        expected = []
        for goal in d["goals"]:
            point = q[36:38] + goal["distance_m"] * forward + goal["lateral_m"] * lateral
            target = [float(point[0]), float(point[1]), 0.11]
            case = {
                k: val
                for k, val in original["case"].items()
                if k not in ("ball_local", "velocity_local")
            }
            case.update(
                ball_world_initial=q[36:39].tolist(),
                velocity_world_initial=v[35:37].tolist(),
                goal_world=target,
            )
            expected.append(
                dict(context=dict(original, case=case, context_id=s["context"]), goal=target)
            )
        foot = "left" if "left" in original["case"]["source"] else "right"
        if pf["tasks"] != expected or pf["stroke_foot"] != foot:
            raise ValueError("source unbound preflight differs from own pinned-data reconstruction")
        verify_targeted_search(
            s,
            context=s["context"],
            center=pf["center"],
            targets=[r["goal"] for r in expected],
            foot=foot,
        )
        for claim in s["records"]:
            raw = {}
            for key, art in claim["artifacts"].items():
                with np.load(a.root / art["path"], allow_pickle=False) as z:
                    raw[key] = {k: z[k].copy() for k in z.files}
            result, stand, good, _, goal = replay_stroke_world(
                bench,
                actor,
                expected[claim["task"]]["context"],
                pf["center"],
                np.array(claim["latent"]),
                raw,
                counts,
                stroke_foot=foot,
            )
            if (
                result != claim["result"]
                or stand != claim["stand"]
                or good != claim["healthy_quality"]
            ):
                raise ValueError("own original first-touch diagnostics differ")
            # All arrays below have just been matched to own NN/manual native
            # primitives, so recomputing this task score does not borrow truth.
            qn = np.concatenate([r["qpos"] for r in raw.values()]).reshape(-1, 43)
            vn = np.concatenate([r["qvel"] for r in raw.values()]).reshape(-1, 41)
            fn = np.concatenate([r["forces"] for r in raw.values()]).reshape(-1, 3)
            measure = targeted_pass_precision(
                qn,
                vn,
                fn,
                goal,
                body_healthy=not result["body_failed"],
                recovery_observed=bool(stand and stand["healthy"] and stand["stability_observed"]),
            )
            if measure != claim["passing_result"]:
                raise ValueError("own pass precision outcome differs")
            row = dict(tag=claim["tag"], task=claim["task"], passing_result=measure)
            records.append(row)
            write_once(a.output / (claim["tag"] + ".json"), row)
            print(
                json.dumps(dict(tag=claim["tag"], counts=counts, passing_result=measure)),
                flush=True,
            )
        if (
            counts["native"] != s["counts"]["native_steps"]
            or counts["neural"] != s["counts"]["actor_calls"]
            or counts["kinematic_intent"] != 0
        ):
            raise ValueError("own complete physical/NN counts differ")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="TARGETED_PASS_ALL72_OWN_NATIVE_AND_CEM_REPLAY_MATCH",
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                input_file_hashes=pins,
                context=s["context"],
                counts=counts,
                records=records,
                teammate_receiving_verified=False,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="TARGETED_PASS_AUDIT_FAILED_NO_RETRY",
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
