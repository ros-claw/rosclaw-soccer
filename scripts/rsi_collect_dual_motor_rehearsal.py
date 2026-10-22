"""Collect actual canonical-parent TRAIN20 first-skill rehearsal, not fake anchors."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("qualification-root", "cases-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("qualification-hash", "expected-head"):
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
            raise ValueError("frozen clean rehearsal source required")

    clean()
    qp = a.qualification_root / "complete.json"
    cp = a.cases_root / "declaration.json"
    if (
        a.output.exists()
        or qp.is_symlink()
        or cp.is_symlink()
        or digest(qp) != a.qualification_hash
    ):
        raise ValueError("exact initial-motor equivalence proof and unused output required")
    q = json.loads(qp.read_text())
    pins = dict(q["input_file_hashes"])
    if (
        q["status"] != "DUAL256_INITIAL_NATIVE_REFERENCE_STANDING_EQUIVALENCE_PASS_NOT_TRAINED"
        or q["actual_native_steps"] != 15560
        or q["optimizer_updates"] != 0
        or q["hardware_authorized"] is not False
        or q["promotion_authorized"] is not False
        or pins.get(str(cp)) != digest(cp)
    ):
        raise ValueError("qualified initial dual architecture and original TRAIN20 required")
    cases = json.loads(cp.read_text())["cases"]
    if len(cases) != 20 or [x["case"] for x in cases] != list(range(20)):
        raise ValueError("all original TRAIN first-skill cases required")
    qd = a.qualification_root / "declaration.json"
    if digest(qd) != q["declaration_hash"]:
        raise ValueError("exact qualification declaration required")
    lineage = json.loads(qd.read_text())["parent_lineage_hash"]
    paths = [Path(x) for x, h in pins.items() if h == lineage and x.endswith("/complete.json")]
    if len(paths) != 1:
        raise ValueError("unique exact canonical parent required")
    training = json.loads(paths[0].read_text())
    for path in (qp, cp, qd):
        pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual_motor.actual_parent_TRAIN20_first_reference_rehearsal.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            parent_lineage_hash=lineage,
            cases=cases,
            maximum_native_steps=80000,
            maximum_neural_calls=8000,
            DEV_rows=0,
            Fresh_rows=0,
            hidden_state_labels_for_learner=False,
            first_skill_only=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    calls = 0
    identities = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.motor_reference_rollout import collect_parent_reference
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=lineage,
            scene_hash=scene,
        )
        records = []
        for case in cases:
            ep = bench.make(case)
            kick, stand, summary, nn = collect_parent_reference(
                ep, actor, foot="left" if "left" in case["source"] else "right"
            )
            calls += nn
            identities += len(kick["qpos"])
            folder = a.output / f"case{case['case']}"
            folder.mkdir()
            artifacts = {}
            for name, raw in (("kick", kick), ("stand", stand)):
                if raw is not None:
                    path = folder / (name + ".npz")
                    np.savez_compressed(path, **raw)
                    artifacts[name] = dict(path=str(path.relative_to(a.output)), hash=digest(path))
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
            row = dict(
                case=case["case"],
                source=case["source"],
                result=result,
                stand=summary,
                healthy_quality=bool(
                    contact_quality(result)
                    and summary
                    and summary["healthy"]
                    and summary["stability_observed"]
                ),
                native_steps=len(raw["qpos"]) * 10,
                neural_calls=nn,
                artifacts=artifacts,
            )
            write_once(folder / "complete.json", row)
            records.append(row)
            print(
                json.dumps(
                    dict(
                        case=case["case"],
                        native_steps=bench.native_steps,
                        parent_quality=row["healthy_quality"],
                    )
                ),
                flush=True,
            )
            if calls > 8000 or bench.native_steps > 80000:
                raise ValueError("bounded original rehearsal budget exceeded")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="ACTUAL_PARENT_TRAIN20_REHEARSAL_COLLECTED_PENDING_NATIVE_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                actual_native_steps=bench.native_steps,
                actual_neural_calls=calls,
                identity_intent_calls=identities,
                learner_hidden_state_labels=False,
                optimizer_updates=0,
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
                actual_native_steps=bench.native_steps if bench else 0,
                actual_neural_calls=calls,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
