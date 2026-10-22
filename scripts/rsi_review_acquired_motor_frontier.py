"""Review every seed against original AND later acquired neural motor skills.

Consumes complete independent physical proofs; performs no physics, fitting,
candidate activation or new promotion. Uses Core TaskRetention per context.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank-root", "core-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--exam-roots", nargs=3, type=Path, required=True)
    for key in ("exam-source-head", "core-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def head(root):
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or head(Path.cwd()) != a.expected_head
            or head(a.core_root) != a.core_head
        ):
            raise ValueError("exact pinned clean scientific frontier review source required")

    clean()
    bp = a.bank_root / "complete.json"
    if a.output.exists() or bp.is_symlink() or (a.bank_root / "failed.json").exists():
        raise ValueError("unused review output and complete actual correction bank required")
    bank = json.loads(bp.read_text())
    if (
        bank["status"] != "STUDENT_PREFIX_CORRECTION_BANK110_READY_NOT_TRAINED"
        or len(bank["records"]) != 110
    ):
        raise ValueError("exact all110 correction histories required")
    pins = dict(bank["input_file_hashes"])
    pins[str(bp)] = digest(bp)

    def receipt(h):
        paths = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != h:
            raise ValueError("unique exact complete evidence lineage required")
        return paths[0], json.loads(paths[0].read_text())

    _, base = receipt(bank["base_data_hash"])
    w6_training = base["warm_start_hash"]
    candidates = []
    for seed, root in zip((65023, 65024, 65025), a.exam_roots, strict=True):
        cp, ap = root / "complete.json", root / "native-audit/complete.json"
        if any(p.is_symlink() for p in (cp, ap)) or any(
            (r / "failed.json").exists() for r in (root, ap.parent)
        ):
            raise ValueError("all three completely independently audited candidates required")
        chain, own = [json.loads(p.read_text()) for p in (cp, ap)]
        if (
            chain["status"] != "CONTACT_CORRECTION_EXAM_CHAIN_CLOSED_NOT_GOAL_COMPLETE"
            or chain["source_head"] != a.exam_source_head
            or own["source_head"] != a.exam_source_head
            or own["checkpoint_kind"] != "contact-consolidated-v2"
        ):
            raise ValueError("explicit completed v2 candidate exam sources required")
        for path, h in own["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting candidate/historical proof pin")
            pins[path] = h
        for path in (cp, ap):
            pins[str(path)] = digest(path)
        _, training = receipt(own["training_hash"])
        if (
            training["config"]["seed"] != seed
            or training["data_hash"] != digest(bp)
            or chain["training_hash"] != own["training_hash"]
            or training["warm_start_hash"] != bank["warm_student_hashes"][str(seed)]
        ):
            raise ValueError("same-data independently continued seed identities required")
        candidates.append((seed, ap, own, training))

    def native_witness(path):
        own = json.loads(path.read_text())
        if (
            own["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
            or own["worlds"] != 44
            or own["independent_whole_world_replay"] is not True
            or own["identity_intent_only"] is not True
            or any(
                own[k] is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "Fresh_success_claim",
                    "legacy52_claim",
                )
            )
        ):
            raise ValueError("unassisted neural full44 proof required, not a teacher or fit result")
        np, native = receipt(own["source_hash"])
        dp = np.parent / "declaration.json"
        if (
            dp.is_symlink()
            or digest(dp) != native["declaration_hash"]
            or pins.get(str(dp)) != digest(dp)
        ):
            raise ValueError("same-task physical context declaration required")
        declaration = json.loads(dp.read_text())
        if (
            declaration["protocol"] != "dual256.unassisted_TRAIN20_and_consumed_run2.v1"
            or len(declaration["contexts"]) != 22
            or own["counts"]["native"] != native["actual_native_steps"]
            or own["counts"]["neural"] != native["actual_actor_decisions"]
            or len(native["records"]) != 44
            or any(
                r["unassisted"] is not True
                or r["result"]["postcontact"]["profile"]
                != "native_reference_motion.contact10N.post1p2s.v1"
                for r in native["records"]
            )
        ):
            raise ValueError("same complete native22 profile and actual world accounting required")
        return own, declaration

    audit_paths = [Path(p) for p in pins if p.endswith("/native-audit/complete.json")]

    def ancestor(h):
        matches = []
        for path in audit_paths:
            if path.is_symlink() or digest(path) != pins[str(path)]:
                raise ValueError("historical audit changed")
            row = json.loads(path.read_text())
            if (
                row.get("status") == "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
                and row["training_hash"] == h
            ):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError("exact unique whole44 historical neural witness required")
        return matches[0], native_witness(matches[0])

    old_path, (old, scope) = ancestor(w6_training)
    history = []
    for seed, path, own, training in candidates:
        warm_path, (warm, warm_scope) = ancestor(training["warm_start_hash"])
        current, current_scope = native_witness(path)
        if (
            warm_scope["contexts"] != scope["contexts"]
            or current_scope["contexts"] != scope["contexts"]
            or own["old_protected_cases"] != old["old_protected_cases"]
        ):
            raise ValueError(
                "cannot compare different physical contexts or change original protection"
            )
        history.append(
            (
                seed,
                path,
                current,
                [(pins[str(old_path)], old["results"]), (pins[str(warm_path)], warm["results"])],
            )
        )
    scenes = {h for p, h in pins.items() if p.endswith("/cached-standard/declared-scene.xml")}
    if len(scenes) != 1:
        raise ValueError("one immutable physical body/ball scene required")
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_root / "src"):
        pins.update({str(p): digest(p) for p in root.rglob("*.py")})
    check_pins(pins)
    import rosclaw.continual.stability as stability

    from rosclaw_soccer.rsi.acquired_motor_retention import acquired_motor_retention
    from rosclaw_soccer.sim.contracts import hash_json

    if (
        Path(stability.__file__).resolve()
        != (a.core_root / "src/rosclaw/continual/stability.py").resolve()
    ):
        raise ValueError("Core retention contract must resolve from the pinned source")
    ids = [
        hash_json(
            dict(
                scene_hash=next(iter(scenes)),
                context=c,
                profile="native_reference_motion.contact10N.post1p2s.v1",
                recovery_controls=100,
            )
        )
        for c in scope["contexts"]
    ]
    reports = []
    for seed, path, own, witnesses in history:
        report = acquired_motor_retention(
            context_hashes=ids,
            original_protected=old["old_protected_cases"],
            historical=witnesses,
            current=own["results"],
        )
        if report["original_protection_retained"] != own["old_protected_retained"]:
            raise ValueError("source original protection grade differs")
        reports.append(
            dict(
                seed=seed,
                own_audit_hash=pins[str(path)],
                checkpoint_hash=own["checkpoint_hash"],
                **report,
            )
        )
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="ALL_THREE_ACQUIRED_MOTOR_FRONTIERS_REVIEWED_NOT_PROMOTED",
            source_head=a.expected_head,
            core_head=a.core_head,
            input_file_hashes=pins,
            reports=reports,
            retained_seed_ids=[r["seed"] for r in reports if r["acquired_frontier_retained"]],
            physical_steps=0,
            optimizer_updates=0,
            scientific_success_not_inferred_from_counts=True,
            legacy52_verified=False,
            Fresh_verified=False,
            team_game_verified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
