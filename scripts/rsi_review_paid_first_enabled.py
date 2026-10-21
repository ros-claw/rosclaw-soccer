"""Audit-bound first-control handover diagnosis; no fitting, scoring changes or promotion."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "qualification-root",
        "parent-root",
        "parent-audit-root",
        "router-root",
        "router-audit-root",
        "output",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("qualification-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen clean reviewer required")

    clean()
    if a.output.exists():
        raise ValueError("unused diagnostic output required")
    rp = a.qualification_root / "complete.json"
    if digest(rp) != a.qualification_hash:
        raise ValueError("exact closed qualification required")
    qualified = json.loads(rp.read_text())
    if (
        qualified["status"] != "ROUTER_FRONTIER_PASS_PENDING_NEW_FRESH"
        or qualified["all_three_seeds_pass"] is not True
        or qualified["primary_seed"] != 0
    ):
        raise ValueError("fixed qualified seed0 required")
    qd = json.loads((a.qualification_root / "declaration.json").read_text())
    roots = [a.parent_root, a.parent_audit_root, a.router_root, a.router_audit_root]
    deadline = time.monotonic() + 7200
    while not all((r / "complete.json").exists() for r in roots):
        clean()
        if any((r / "failed.json").exists() for r in roots):
            raise ValueError("upstream failed, no retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("both first-enabled independent audits required")
        time.sleep(10)
    import numpy as np

    from rosclaw_soccer.rsi.paid_contact_diagnosis import summarize_contact_failures
    from rosclaw_soccer.rsi.paid_continuity_audit import identical
    from rosclaw_soccer.rsi.paid_router_review import compare_router_frontier

    pins = dict(qualified["input_file_hashes"])
    pins[str(rp)] = a.qualification_hash
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})

    def read_raw(root, link):
        f = root / link["raw_path"]
        if (
            f.is_symlink()
            or not f.resolve().is_relative_to(root.resolve())
            or digest(f) != link["raw_hash"]
        ):
            raise ValueError("exact confined trajectory required")
        pins[str(f)] = link["raw_hash"]
        with np.load(f, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    baseline = []
    for i, root_name in enumerate(qd["reference_shards"]):
        root = Path(root_name)
        f = root / "complete.json"
        if qualified["input_file_hashes"].get(str(f)) != digest(f):
            raise ValueError("qualified parent baseline changed")
        r = json.loads(f.read_text())
        if r["shard"] != i:
            raise ValueError("complete ordered reference shards required")
        baseline.extend((root / "grown", g["arms"]["grown"]) for g in r["groups"])
    router_old_root = Path(qd["new_exams"][0])
    f = router_old_root / "complete.json"
    if qualified["input_file_hashes"].get(str(f)) != digest(f):
        raise ValueError("qualified old router exam changed")
    router_old = json.loads(f.read_text())["worlds"]
    if len(baseline) != 20 or [w["case"] for _, w in baseline] != list(range(20)):
        raise ValueError("all20 baseline cases required")
    arms = {}
    for arm, root, audit in (
        ("parent", a.parent_root, a.parent_audit_root),
        ("router", a.router_root, a.router_audit_root),
    ):
        rp, dp, ap = root / "complete.json", root / "declaration.json", audit / "complete.json"
        r, d, ar = [json.loads(f.read_text()) for f in (rp, dp, ap)]
        if (
            any((v / "failed.json").exists() for v in (root, audit))
            or r["declaration_hash"] != digest(dp)
            or ar["source_hash"] != digest(rp)
            or r["status"] != "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT"
            or ar["status"] != "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
            or d.get("first_enabled") is not True
            or d["schedule"] != ["student"] * 3
            or r["source_head"] != d["source_head"]
            or r["summary"] != ar["summary"]
            or ar["counts"]["worlds"] != 40
            or ar["own_native_steps"] != r["actual_native_steps"]
            or ar["own_trajectory_neural_rows"] != r["actual_neural_rows"]
            or [w["case"] for w in r["worlds"]] != list(range(20))
            or ar["student_hash"] != r["student_hash"]
            or any(
                v[k] is not False
                for v in (r, d, ar)
                for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("full same-student first-enabled own-NN/physics audit required")
        expected = (
            Path(qd["reference_pipeline"]) if arm == "parent" else Path(qd["pipelines"][0])
        ) / "trained"
        if (
            Path(d["student_root"]) != expected
            or digest(expected / "complete.json") != r["student_hash"]
        ):
            raise ValueError("fixed current parent and primary router only required")
        for f in (rp, dp, ap):
            pins[str(f)] = digest(f)
        for path, h in ar["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting first-enabled evidence lineage")
            pins[path] = h
        old = baseline if arm == "parent" else [(router_old_root, w) for w in router_old]
        changes = []
        for i, ((old_root, ow), w) in enumerate(zip(old, r["worlds"], strict=True)):
            olinks = {Path(v["raw_path"]).stem: v for v in ow["artifacts"]}
            nlinks = {Path(v["raw_path"]).stem: v for v in w["artifacts"]}
            identical(
                read_raw(old_root, olinks["world-initial"]), read_raw(root, nlinks["world-initial"])
            )
            oz, nz = read_raw(old_root, olinks["kick0"]), read_raw(root, nlinks["kick0"])
            if len(oz["NN_action"]) != len(nz["NN_action"]):
                raise ValueError("first-kick horizon changed")
            indices = np.flatnonzero(np.any(oz["NN_action"] != nz["NN_action"], axis=1))
            changed_q = np.flatnonzero(
                np.any(oz["native_qpos_before"] != nz["native_qpos_before"], axis=1)
            )
            changes.append(
                dict(
                    case=i,
                    changed_action_rows=len(indices),
                    first_action_difference_control=int(indices[0]) if len(indices) else None,
                    first_measured_body_difference_control=int(changed_q[0])
                    if len(changed_q)
                    else None,
                    maximum_same_index_PD_target_difference_rad=float(
                        np.abs(oz["target"] - nz["target"]).max()
                    ),
                    later_differences_include_changed_physical_state=True,
                )
            )
        arms[arm] = dict(
            first_enabled_summary=r["summary"],
            paired_vs_same_motor_successor_only=compare_router_frontier(
                [w for _, w in old], r["worlds"]
            ),
            first_kick=summarize_contact_failures([w["kicks"][0]["result"] for w in r["worlds"]]),
            actual_first_action_changes=changes,
        )
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_FIRST_ENABLED_OWN_PHYSICS_BOUND_CONTACT_AND_SUCCESSOR_DIAGNOSIS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            arms=arms,
            Fresh_rows_read=0,
            optimizer_updates=0,
            own_native_steps=0,
            own_neural_rows=0,
            interpretation="cached own-audited physical diagnosis, not new inference or replay",
            score_contract_changed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
