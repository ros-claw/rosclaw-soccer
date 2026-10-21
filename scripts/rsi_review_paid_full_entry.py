"""Fixed three-seed full-entry decision after new TRAIN and original TRAIN/DEV audits."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def validate_interventional_pair(trained, control):
    """A same-data/no-SGD control isolates weights from applicability geometry."""
    for key in ("data_hash", "warm_hash", "body_scene_hash", "parent_hash"):
        if not trained.get(key) or trained[key] != control.get(key):
            raise ValueError("same intervention data, warm ancestor and body required")
    generation = trained["config"].get("generation", 1)
    if type(generation) is not int or generation not in (1, 2):
        raise ValueError("bounded typed interventional generation required")
    learner_protocol = (
        "interventional_dagger.replanned_contact.v2"
        if generation == 2
        else "interventional_dagger.warm_contact.v1"
    )
    control_protocol = (
        "interventional_dagger.replanned_geometry_control.v2"
        if generation == 2
        else "interventional_dagger.geometry_only_control.v1"
    )
    if (
        trained["config"].get("protocol") != learner_protocol
        or trained["config"].get("new_steps") != 2048
        or control["config"].get("protocol") != control_protocol
        or control["config"].get("generation", 1) != generation
        or control["config"].get("control") is not True
        or control["config"].get("seed_index") != 0
        or control["config"].get("new_steps") != 0
        or trained["counts"].get("optimizer_updates") != 2048
        or control["counts"].get("optimizer_updates") != 0
    ):
        raise ValueError("fixed actual SGD learner and zero-update causal control required")


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("family-root", "reference-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("family-prefix", "stamp", "reference-hash", "producer-head", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--compact-support", action="store_true")
    mode.add_argument("--interventional", action="store_true")
    p.add_argument("--generation", type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    if a.generation == 2 and not a.interventional:
        p.error("second generation requires explicit interventional mode")
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
    control_paths = {}
    if a.interventional:
        control_paths["trained"] = a.family_root / f"{a.family_prefix}-geometry-control-{a.stamp}"
        for mode in ("exam", "audit"):
            control_paths[mode] = (
                a.family_root / f"{a.family_prefix}-geometry-control-train20-{mode}-{a.stamp}"
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
            interventional=a.interventional,
            generation=a.generation,
            geometry_control_roots={k: str(v) for k, v in control_paths.items()},
            Fresh_rows_read=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    deadline = time.monotonic() + 7200
    roots = [r for values in paths.values() for r in values.values()]
    roots += list(control_paths.values())
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
    control = control_worlds = None
    if a.interventional:
        control = read(
            control_paths["trained"],
            "PAID_REPLANNED_INTERVENTIONAL_GEOMETRY_CONTROL_NOT_PHYSICALLY_QUALIFIED"
            if a.generation == 2
            else "PAID_INTERVENTIONAL_GEOMETRY_ONLY_CONTROL_NOT_PHYSICALLY_QUALIFIED",
        )
        cr = read(control_paths["exam"], "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT")
        ca = read(
            control_paths["audit"],
            "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS",
        )
        cdp = control_paths["exam"] / "declaration.json"
        cd = json.loads(cdp.read_text())
        ch = digest(control_paths["trained"] / "complete.json")
        if (
            any(v["source_head"] != a.producer_head for v in (control, cr, ca, cd))
            or any(v["student_hash"] != ch for v in (cr, ca, cd))
            or cr["declaration_hash"] != digest(cdp)
            or ca["source_hash"] != digest(control_paths["exam"] / "complete.json")
            or cd.get("first_enabled") is not True
            or cd["schedule"] != ["student"] * 3
            or cr["bank"] != "train"
            or [w["case"] for w in cr["worlds"]] != list(range(20))
            or ca["counts"]["worlds"] != 40
            or ca["own_native_steps"] != cr["actual_native_steps"]
            or ca["own_trajectory_neural_rows"] != cr["actual_neural_rows"]
            or ca["first_kick_replayed"] is not True
            or ca["summary"] != cr["summary"]
        ):
            raise ValueError("same-source independently audited geometry control required")
        control_worlds = cr["worlds"]
    seeds = []
    for seed, values in paths.items():
        trained = read(
            values["trained"],
            "PAID_REPLANNED_INTERVENTIONAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
            if a.generation == 2
            else "PAID_INTERVENTIONAL_DAGGER_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
            if a.interventional
            else "PAID_COMPACT_SUPPORT_BATCH1_AUDITED_NOT_PHYSICALLY_QUALIFIED"
            if a.compact_support
            else "PAID_FULL_ENTRY_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED",
        )
        th = digest(values["trained"] / "complete.json")
        if (
            trained["config"]["seed_index"] != seed
            or trained["source_head"] != a.producer_head
            or (
                trained["config"].get("protocol") != "interventional_dagger.replanned_contact.v2"
                if a.generation == 2
                else trained["config"].get("protocol") != "interventional_dagger.warm_contact.v1"
                if a.interventional
                else trained["config"].get("protocol") != "full-entry-contact.compact-support.v3"
                if a.compact_support
                else trained["config"].get("trajectory_support") is not True
            )
        ):
            raise ValueError("fixed predeclared trajectory-support seed identity required")
        if a.interventional:
            validate_interventional_pair(trained, control)
        comparisons = {}
        parameter_contrast = None
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
                if a.interventional:
                    parameter_contrast = compare_router_frontier(control_worlds, r["worlds"])
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
                same_geometry_parameter_contrast=parameter_contrast,
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
            interventional=a.interventional,
            neural_parameter_learning_is_not_physical_gain=True,
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
