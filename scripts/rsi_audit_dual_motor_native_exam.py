"""Independent full-world own-dual256 replay of every learned-motor exam world."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("source-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen clean independent dual native audit required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != a.source_hash
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("exact closed complete native exam required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    if (
        r["status"] != "DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or d["protocol"] != "dual256.unassisted_TRAIN20_and_consumed_run2.v1"
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or len(r["records"]) != 44
        or len(d["contexts"]) != 22
        or d["repeats"] != 2
        or d["hidden_width"] != 256
        or d["online_teacher_enabled"] is not False
        or d["Fresh_cases"] != 0
        or not r["whole_repeats_exact"]
        or r["training_hash"] != d["training_hash"]
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "legacy52_claim")
        )
    ):
        raise ValueError("fixed complete unassisted dual44 TRAIN-only exam required")
    pins = dict(r["input_file_hashes"])
    for x in (rp, dp):
        pins[str(x)] = digest(x)

    def bound_receipt(h):
        paths = [
            Path(x) for x, value in pins.items() if value == h and x.endswith("/complete.json")
        ]
        if len(paths) != 1:
            raise ValueError("unique exact source-bound receipt required")
        return paths[0], json.loads(paths[0].read_text())

    training_path, training = bound_receipt(d["training_hash"])
    parent_path, parent_record = bound_receipt(training["parent_lineage_hash"])
    data_path, data = bound_receipt(training["data_hash"])
    baseline_path, baseline = bound_receipt(d["baseline_hash"])
    from rosclaw_soccer.rsi.contact_consolidation_data import context_witness_hash

    tdpath = (
        bound_receipt(context_witness_hash(data, d.get("checkpoint_kind", "supervised")))[0].parent
        / "declaration.json"
    )
    bdpath = baseline_path.parent / "declaration.json"
    if pins.get(str(tdpath)) != digest(tdpath) or pins.get(str(bdpath)) != digest(bdpath):
        raise ValueError("original task declarations must be source-bound")
    td, bd = [json.loads(x.read_text()) for x in (tdpath, bdpath)]
    if (
        [x["case"] for x in d["contexts"][:20]] != bd["cases"]
        or [x["backoff_m"] for x in d["contexts"][:20]] != [0.0] * 20
        or [x["case"] for x in d["contexts"][20:]] != [td["source_case"]] * 2
        or [x["backoff_m"] for x in d["contexts"][20:]] != [x["backoff_m"] for x in td["courses"]]
    ):
        raise ValueError("full original old-skill and consumed run fixture identities required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(native=0, neural=0, kinematic_intent=0)
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw
        from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world
        from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [x["case"] for x in d["contexts"]])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=parent_path,
            training_receipt_hash=training["parent_lineage_hash"],
            scene_hash=scene,
        )
        actor, _ = load_dual_candidate_for_sim_audit(
            parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=training_path.parent,
            expected_receipt_hash=d["training_hash"],
            checkpoint_kind=d.get("checkpoint_kind", "supervised"),
        )
        seen = set()
        previous = {}
        verified = []
        contacts = []
        for rec in r["records"]:
            index, repeat = rec["context"], rec["repeat"]
            if (
                type(index) is not int
                or not 0 <= index < 22
                or repeat not in (0, 1)
                or (index, repeat) in seen
                or rec["unassisted"] is not True
            ):
                raise ValueError("every distinct unassisted actual world must audit")
            seen.add((index, repeat))
            ctx = d["contexts"][index]
            if rec["kind"] != ctx["kind"]:
                raise ValueError("original task group identity required")
            blobs = {}
            for name, art in rec["artifacts"].items():
                x = a.root / art["path"]
                if (
                    x.is_symlink()
                    or not x.resolve().is_relative_to(a.root.resolve())
                    or digest(x) != art["hash"]
                ):
                    raise ValueError("exact confined native/own-neural source required")
                pins[str(x)] = art["hash"]
                with np.load(x, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            if index in previous:
                if set(previous[index]) != set(blobs):
                    raise ValueError("whole repeat artifact set differs")
                for name, raw in blobs.items():
                    if set(raw) != set(previous[index][name]):
                        raise ValueError("whole repeat raw fields differ")
                    for k, v in raw.items():
                        equal(v, previous[index][name][k], "whole repeat " + k)
            else:
                previous[index] = blobs
            ep = bench.make(ctx["case"])
            if ctx["backoff_m"]:
                q, _ = ep.layout.canonical(ep.env.data)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx["backoff_m"] * np.array(
                    [np.cos(yaw), np.sin(yaw)]
                )
                mujoco.mj_forward(ep.env.model, ep.env.data)
            before = dict(counts)
            events = []
            kick, stand = blobs["kick"], blobs.get("stand")
            replay_teacher_world(
                ep,
                actor,
                kick,
                stand,
                latent=np.zeros((2, 6), np.float64),
                center=105,
                foot="left" if "left" in ctx["case"]["source"] else "right",
                counts=counts,
                hidden_width=256,
                contact_observer=events.append,
            )
            if (
                counts["native"] - before["native"] != rec["native_steps"]
                or counts["neural"] - before["neural"] != rec["actor_decisions"]
            ):
                raise ValueError("all actual per-world work must be accounted")
            raw = {
                k: np.concatenate([kick[k], stand[k]]) if stand is not None else kick[k]
                for k in ("qpos", "qvel", "forces", "body_failed")
            }
            f = raw["forces"].reshape(-1, 3)
            post = native_contact_window(
                force=f[:, :2],
                canonical_qpos=raw["qpos"].reshape(-1, 43),
                canonical_qvel=raw["qvel"].reshape(-1, 41),
                goal=ep.goal,
                intended_foot=0 if "left" in ctx["case"]["source"] else 1,
            )
            result = dict(
                body_failed=bool(raw["body_failed"].any()),
                forbidden=bool((f[:, 2] > 1).any()),
                nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                postcontact=post,
            )
            ss = stand_from_raw(stand) if stand is not None else None
            good = bool(
                contact_quality(result) and ss and ss["healthy"] and ss["stability_observed"]
            )
            if result != rec["result"] or ss != rec["stand"] or good != rec["healthy_quality"]:
                raise ValueError("independent physical success/body/contact outcome differs")
            summary = summarize_contacts(events, ep.goal)
            path = a.output / f"context{index}-repeat{repeat}-contacts.json"
            write_once(path, dict(events=events, summary=summary))
            contacts.append(
                dict(
                    context=index, repeat=repeat, path=path.name, hash=digest(path), summary=summary
                )
            )
            verified.append(
                dict(context=index, repeat=repeat, quality=good, body_failed=result["body_failed"])
            )
            print(
                json.dumps(
                    dict(context=index, repeat=repeat, quality=good, own_native=counts["native"])
                ),
                flush=True,
            )
        protected = [x["case"] for x in baseline["records"] if x["healthy_quality"]]
        achieved = [
            x["context"]
            for x in verified
            if x["context"] < 20 and x["repeat"] == 0 and x["quality"]
        ]
        novel = sum(x["quality"] for x in verified if x["context"] >= 20 and x["repeat"] == 0)
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_actor_decisions"]
            or 2 * counts["neural"] != r["recurrent_branch_queries"]
            or protected != r["old_protected_cases"]
            or achieved != r["old_success_cases"]
            or all(x in achieved for x in protected) != r["old_protected_retained"]
            or novel != r["consumed_run_successes"]
        ):
            raise ValueError("independent full exam counts/retention grade differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=a.source_hash,
                training_hash=d["training_hash"],
                checkpoint_kind=d.get("checkpoint_kind", "supervised"),
                checkpoint_hash=training["checkpoint_hash"],
                counts=counts,
                recurrent_branch_queries=2 * counts["neural"],
                worlds=44,
                results=verified,
                contact_event_artifacts=contacts,
                independent_whole_world_replay=True,
                shared_observation_codec=True,
                identity_intent_only=True,
                old_protected_cases=protected,
                old_success_cases=achieved,
                old_protected_retained=all(x in achieved for x in protected),
                consumed_run_successes=novel,
                Fresh_success_claim=False,
                legacy52_claim=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error), counts=counts, automatic_retry=False, hardware_authorized=False
            ),
        )
        raise


if __name__ == "__main__":
    main()
