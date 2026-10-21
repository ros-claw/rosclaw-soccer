"""Hash-bound three-arm Fresh comparison after all own-NN/native audits close."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "declaration-root",
        "output",
        *[s + k for s in ("r2", "parent", "progressive") for k in ("-root", "-audit-root")],
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("declaration-hash", "expected-head", "producer-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("immutable clean Fresh reviewer required")

    clean()
    if a.output.exists():
        raise ValueError("unused Fresh review output required")
    roots = {
        arm: (getattr(a, arm + "_root"), getattr(a, arm + "_audit_root"))
        for arm in ("r2", "parent", "progressive")
    }
    deadline = time.monotonic() + 7200
    while not all((root / "complete.json").exists() for pair in roots.values() for root in pair):
        clean()
        if any((root / "failed.json").exists() for pair in roots.values() for root in pair):
            raise ValueError("upstream contract failure, no implicit retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("all three closed audits required")
        time.sleep(10)
    clean()
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_declaration
    from rosclaw_soccer.rsi.paid_fresh_protocol import paired_fresh_review

    preflight, d = load_fresh_declaration(a.declaration_root, a.declaration_hash)
    pins = dict(preflight["input_file_hashes"])
    arms = {}
    reports = {}
    for arm, (root, audit_root) in roots.items():
        rp, ap = root / "complete.json", audit_root / "complete.json"
        r, review = [json.loads(f.read_text()) for f in (rp, ap)]
        if (
            any((x / "failed.json").exists() for x in (root, audit_root))
            or r["source_head"] != a.producer_head
            or r["fresh_hash"] != a.declaration_hash
            or r["arm"] != arm
            or review["arm"] != arm
            or review["source_hash"] != digest(rp)
            or review["fresh_hash"] != a.declaration_hash
            or review["status"] != "PAID_FRESH32_OWN_NEURAL_AND_ALL_PHASES_NATIVE_REPLAY_PASS"
            or review["summary"] != r["summary"]
            or review["counts"]["worlds"] != 64
            or review["own_native_steps"] != r["actual_native_steps"]
            or review["own_trajectory_neural_rows"] != r["actual_neural_rows"]
            or review["first_kick_replayed"] is not True
            or any(
                obj[k] is not False
                for obj in (r, review)
                for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("complete same-protocol own-neural/native paired evidence required")
        for path, h in {
            **review["input_file_hashes"],
            str(rp): digest(rp),
            str(ap): digest(ap),
        }.items():
            if path in pins and pins[path] != h:
                raise ValueError("three-arm evidence lineage conflict")
            pins[path] = h
        arms[arm] = r["worlds"]
        reports[arm] = r
    import numpy as np

    from rosclaw_soccer.rsi.paid_continuity_audit import identical

    for i in range(32):
        reference = None
        for arm, (root, _) in roots.items():
            world = reports[arm]["worlds"][i]
            links = {Path(row["raw_path"]).stem: row for row in world["artifacts"]}
            keys = {"world-initial", "kick0", "kick-terminal0"} | (
                {"stand0"} if "stand0" in links else set()
            )
            arrays = {}
            for key in keys:
                row = links[key]
                f = root / row["raw_path"]
                if (
                    f.is_symlink()
                    or not f.resolve().is_relative_to(root.resolve())
                    or digest(f) != row["raw_hash"]
                ):
                    raise ValueError("prefix binding changed")
                with np.load(f, allow_pickle=False) as z:
                    arrays[key] = {k: z[k].copy() for k in z.files}
            if reference is None:
                reference = arrays
            else:
                if set(reference) != set(arrays):
                    raise ValueError("first R2 prefix phase coverage differs")
                for key in reference:
                    identical(reference[key], arrays[key])
    outcome = paired_fresh_review(arms, d["cases"], d["evaluation"])
    for f in (a.declaration_root / "complete.json", a.declaration_root / "declaration.json"):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_FRESH32_PAIRED_REVIEW_CLOSED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            fresh_hash=a.declaration_hash,
            first_R2_prefix_exact_across_all_arms=True,
            independent_audits_and_native_replays_all_closed=True,
            **outcome,
            own_native_steps=0,
            own_neural_rows=0,
            optimizer_updates=0,
            Fresh_episodes_fit=0,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
