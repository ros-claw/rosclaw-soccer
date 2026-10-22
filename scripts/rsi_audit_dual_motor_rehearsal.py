"""Recompute all actual parent rehearsal trajectories before learning admission."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--source-hash")
    p.add_argument("--producer-head")
    p.add_argument("--declaration-hash")
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.queued_audit_commitment import bind_complete
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen rehearsal auditor required")

    clean()
    source_hash = bind_complete(
        a.root,
        complete_hash=a.source_hash,
        producer_head=a.producer_head,
        declaration_hash=a.declaration_hash,
    )
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if a.output.exists():
        raise ValueError("unused independent audit output required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    if (
        r["status"] != "ACTUAL_PARENT_TRAIN20_REHEARSAL_COLLECTED_PENDING_NATIVE_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or d["protocol"] != "dual_motor.actual_parent_TRAIN20_first_reference_rehearsal.v1"
        or len(r["records"]) != 20
        or [x["case"] for x in r["records"]] != list(range(20))
        or len(d["cases"]) != 20
        or [x["case"] for x in d["cases"]] != list(range(20))
        or d["DEV_rows"] != 0
        or d["Fresh_rows"] != 0
        or d["hidden_state_labels_for_learner"] is not False
        or d["first_skill_only"] is not True
        or r["learner_hidden_state_labels"] is not False
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete declared original TRAIN20 actual parent rehearsal required")
    pins = dict(r["input_file_hashes"])
    for x in (rp, dp):
        pins[str(x)] = digest(x)
    paths = [
        Path(x)
        for x, h in pins.items()
        if h == d["parent_lineage_hash"] and x.endswith("/complete.json")
    ]
    if len(paths) != 1:
        raise ValueError("unique source-bound canonical parent required")
    training = json.loads(paths[0].read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(native=0, neural=0, kinematic_intent=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw
        from rosclaw_soccer.rsi.paid_run_teacher_audit import replay_teacher_world

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, d["cases"])
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=d["parent_lineage_hash"],
            scene_hash=scene,
        )
        verified = []
        for case, rec in zip(d["cases"], r["records"], strict=True):
            if rec["source"] != case["source"]:
                raise ValueError("actual source motion identity required")
            blobs = {}
            for name, art in rec["artifacts"].items():
                x = a.root / art["path"]
                if (
                    x.is_symlink()
                    or not x.resolve().is_relative_to(a.root.resolve())
                    or digest(x) != art["hash"]
                ):
                    raise ValueError("exact confined raw rehearsal artifact required")
                pins[str(x)] = art["hash"]
                with np.load(x, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            ep = bench.make(case)
            before = dict(counts)
            kick, stand = blobs["kick"], blobs.get("stand")
            replay_teacher_world(
                ep,
                actor,
                kick,
                stand,
                latent=np.zeros((2, 6), np.float64),
                center=105,
                foot="left" if "left" in case["source"] else "right",
                counts=counts,
            )
            if (
                counts["native"] - before["native"] != rec["native_steps"]
                or counts["neural"] - before["neural"] != rec["neural_calls"]
            ):
                raise ValueError("actual per-world work count changed")
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
                intended_foot=0 if "left" in case["source"] else 1,
            )
            result = dict(
                body_failed=bool(raw["body_failed"].any()),
                forbidden=bool((f[:, 2] > 1).any()),
                nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                postcontact=post,
            )
            ss = stand_from_raw(stand) if stand is not None else None
            if (
                result != rec["result"]
                or ss != rec["stand"]
                or bool(
                    contact_quality(result) and ss and ss["healthy"] and ss["stability_observed"]
                )
                != rec["healthy_quality"]
            ):
                raise ValueError("independent rehearsal outcome differs")
            verified.append(dict(case=case["case"], healthy_quality=rec["healthy_quality"]))
            print(json.dumps(dict(case=case["case"], own_native=counts["native"])), flush=True)
        if counts != dict(
            native=r["actual_native_steps"],
            neural=r["actual_neural_calls"],
            kinematic_intent=r["identity_intent_calls"],
        ):
            raise ValueError("all actual rehearsal work must be accounted")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="ACTUAL_PARENT_TRAIN20_REHEARSAL_WHOLE_NATIVE_OWN_NN_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=source_hash,
                counts=counts,
                worlds=20,
                results=verified,
                independent_whole_world_replay=True,
                shared_observation_codec=True,
                zero_intent_identity_only=True,
                learner_hidden_state_labels=False,
                first_skill_only=True,
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
