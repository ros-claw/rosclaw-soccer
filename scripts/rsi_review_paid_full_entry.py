"""Fixed three-seed full-entry decision after new TRAIN and original TRAIN/DEV audits."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("family-root", "reference-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("family-prefix", "stamp", "reference-hash", "producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    p.add_argument("--compact-support", action="store_true")
    a = p.parse_args()
    if (
        re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", a.family_prefix) is None
        or re.fullmatch(r"\d{8}", a.stamp) is None
    ):
        p.error("confined family name and eight-digit experiment stamp required")

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("immutable clean full-entry reviewer required")

    clean()
    if a.output.exists():
        raise ValueError("unused final review output required")
    paths = {}
    for i in range(3):
        stage = "verified" if a.compact_support else "trained"
        paths[i] = {"trained": a.family_root / f"{a.family_prefix}-s{i}-{stage}-{a.stamp}"}
        for bank in ("train20", "train16", "dev16"):
            for mode in ("exam", "audit"):
                paths[i][bank + "-" + mode] = (
                    a.family_root / f"{a.family_prefix}-s{i}-{bank}-{mode}-{a.stamp}"
                )
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_head=a.producer_head,
            reference_hash=a.reference_hash,
            roots={str(i): {k: str(v) for k, v in r.items()} for i, r in paths.items()},
            primary_seed=0,
            all_three_seeds_required=True,
            first_enabled_required=True,
            novel_first_cases_required=[4, 5],
            protect_best_existing_orchestration=True,
            compact_support=a.compact_support,
            Fresh_rows_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    deadline = time.monotonic() + 7200
    roots = [r for values in paths.values() for r in values.values()]
    while not all((r / "complete.json").exists() for r in roots):
        clean()
        if any((r / "failed.json").exists() for r in roots):
            raise ValueError("upstream contract failed; diagnose without implicit retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("all first-enabled audit receipts required")
        time.sleep(10)
    from rosclaw_soccer.rsi.paid_router_review import compare_router_frontier

    pins = {}

    def read(root, status):
        rp = root / "complete.json"
        r = json.loads(rp.read_text())
        if (
            (root / "failed.json").exists()
            or rp.is_symlink()
            or r["status"] != status
            or any(r[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
        ):
            raise ValueError("closed SIM-only evidence required")
        for path, h in r["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting physical evidence lineage")
            pins[path] = h
        pins[str(rp)] = digest(rp)
        return r

    qp = a.reference_root / "complete.json"
    if digest(qp) != a.reference_hash:
        raise ValueError("exact current-router frontier required")
    q = read(a.reference_root, "ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH")
    qd = json.loads((a.reference_root / "declaration.json").read_text())
    if q["all_three_seeds_pass"] is not True or q["primary_seed"] != 0:
        raise ValueError("fixed previously qualified primary parent required")
    refs = {}
    reference_roots = {
        "train20": Path(qd["new_exams"][0]),
        "train16": Path(qd["pipelines"][0]) / "train-exam",
        "dev16": Path(qd["pipelines"][0]) / "dev-exam",
    }
    for bank, root in reference_roots.items():
        rp = root / "complete.json"
        if q["input_file_hashes"].get(str(rp)) != digest(rp):
            raise ValueError("best existing orchestration reference changed")
        refs[bank] = json.loads(rp.read_text())["worlds"]
        if [w["case"] for w in refs[bank]] != list(range(20 if bank == "train20" else 16)):
            raise ValueError("complete original case order required")
    seeds = []
    for seed, values in paths.items():
        trained = read(
            values["trained"],
            "PAID_COMPACT_SUPPORT_BATCH1_AUDITED_NOT_PHYSICALLY_QUALIFIED"
            if a.compact_support
            else "PAID_FULL_ENTRY_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED",
        )
        th = digest(values["trained"] / "complete.json")
        if (
            trained["config"]["seed_index"] != seed
            or trained["source_head"] != a.producer_head
            or (
                trained["config"].get("protocol") != "full-entry-contact.compact-support.v3"
                if a.compact_support
                else trained["config"].get("trajectory_support") is not True
            )
        ):
            raise ValueError("fixed predeclared trajectory-support seed identity required")
        comparisons = {}
        novel = False
        for bank in ("train20", "train16", "dev16"):
            root, audit = values[bank + "-exam"], values[bank + "-audit"]
            original = bank != "train20"
            r = read(
                root,
                "PAID_FULL_ENTRY_ORIGINAL16_WORLDS_CLOSED_PENDING_AUDIT"
                if original
                else "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT",
            )
            ar = read(
                audit,
                "PAID_FULL_ENTRY_ORIGINAL16_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
                if original
                else "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS",
            )
            dp = root / "declaration.json"
            d = json.loads(dp.read_text())
            n = 16 if original else 20
            if (
                r["declaration_hash"] != digest(dp)
                or ar["source_hash"] != digest(root / "complete.json")
                or any(v["source_head"] != a.producer_head for v in (r, d, ar))
                or any(v["student_hash"] != th for v in (r, d, ar))
                or d.get("first_enabled") is not True
                or d["schedule"] != ["student"] * 3
                or r["bank"] != ("dev" if bank == "dev16" else "train")
                or [w["case"] for w in r["worlds"]] != list(range(n))
                or ar["counts"]["worlds"] != 2 * n
                or ar["own_native_steps"] != r["actual_native_steps"]
                or ar["own_trajectory_neural_rows"] != r["actual_neural_rows"]
                or ar["first_kick_replayed"] is not True
                or ar["summary"] != r["summary"]
            ):
                raise ValueError(
                    "same fixed student must pass every own-NN/all-phase first-enabled bank"
                )
            comparisons[bank] = compare_router_frontier(refs[bank], r["worlds"])
            if bank == "train20":
                novel = all(
                    r["worlds"][i]["stable"] and r["worlds"][i]["quality_by_kick"][0]
                    for i in (4, 5)
                )
        passed = novel and all(v["retention_pass"] for v in comparisons.values())
        seeds.append(
            dict(
                seed=seed,
                comparisons=comparisons,
                novel_first4_and5_pass=novel,
                full_entry_frontier_pass=passed,
            )
        )
    check_pins(pins)
    clean()
    passed = all(s["full_entry_frontier_pass"] for s in seeds)
    write_once(
        a.output / "complete.json",
        dict(
            status="FULL_ENTRY_FRONTIER_PASS_PENDING_NEW_FRESH"
            if passed
            else "FULL_ENTRY_FRONTIER_REJECTED_KEEP_ROUTER",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seeds=seeds,
            all_three_seeds_pass=passed,
            primary_seed=0,
            Fresh_rows_read=0,
            next_action="DECLARE_NEW_FRESH_V3_FOR_FIXED_SEED0"
            if passed
            else "DIAGNOSE_OWN_POLICY_SUPPORT_AND_CONTACT_NO_BLIND_EPOCHS",
            own_native_steps=0,
            own_neural_rows=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
