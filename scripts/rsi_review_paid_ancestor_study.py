"""Close all four TRAIN counterfactual shards; select only physical same-entry witnesses."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path, nargs=4, required=True)
    p.add_argument("--audits", type=Path, nargs=4, required=True)
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--plan-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen reviewer required")

    clean()
    if a.output.exists() or len(set(a.shards)) != 4 or len(set(a.audits)) != 4:
        raise ValueError("unused output and four distinct shards required")
    deadline = time.monotonic() + 7200
    while not all((root / "complete.json").exists() for root in [*a.shards, *a.audits]):
        clean()
        if any((root / "failed.json").exists() for root in [*a.shards, *a.audits]):
            raise ValueError("upstream failure,no retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("all audited shards required")
        time.sleep(10)
    clean()
    from rosclaw_soccer.rsi.paid_ancestor_study import ancestor_comparison, load_ancestor_plan

    preflight, plan = load_ancestor_plan(a.plan_root, a.plan_hash)
    pins = dict(preflight["input_file_hashes"])
    groups = []
    sources = {}
    native = neural = worlds = 0
    for shard, (root, auditroot) in enumerate(zip(a.shards, a.audits, strict=True)):
        rp, ap = root / "complete.json", auditroot / "complete.json"
        r, v = [json.loads(f.read_text()) for f in (rp, ap)]
        if (
            any((p / "failed.json").exists() for p in (root, auditroot))
            or r["source_head"] != a.producer_head
            or r["plan_hash"] != a.plan_hash
            or v["plan_hash"] != a.plan_hash
            or r["status"] != "PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_SHARD_PENDING_AUDIT"
            or v["status"] != "PAID_TRAIN_ANCESTOR_COUNTERFACTUAL_OWN_NN_AND_PHYSICS_AUDIT_PASS"
            or r["shard"] != shard
            or v["shard"] != shard
            or v["source_hash"] != digest(rp)
            or v["own_native_steps"] != r["actual_native_steps"]
            or v["own_trajectory_neural_rows"] != r["actual_neural_rows"]
            or v["counts"]["worlds"] != 30
            or v["shared_actual_prefixes_exact"] is not True
            or v["first_kick_replayed"] is not True
            or v["comparison"] != r["comparison"]
            or [g["case"] for g in r["groups"]] != list(range(5 * shard, 5 * (shard + 1)))
            or [g["case"] for g in v["groups"]] != list(range(5 * shard, 5 * (shard + 1)))
            or any(o["bank"] != "train" or o["Fresh_fit_rows"] != 0 for o in (r, v))
            or any(
                o[k] is not False
                for o in (r, v)
                for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("complete exact audited counterfactual shard required")
        for path, h in {**v["input_file_hashes"], str(rp): digest(rp), str(ap): digest(ap)}.items():
            if path in pins and pins[path] != h:
                raise ValueError("shard lineage conflict")
            pins[path] = h
        groups.extend(v["groups"])
        sources.update({g["case"]: (root, g) for g in r["groups"]})
        native += v["own_native_steps"]
        neural += v["own_trajectory_neural_rows"]
        worlds += v["counts"]["worlds"]
    comparison = ancestor_comparison(groups)
    selected = []
    for i in comparison["same_entry_ancestor_rescues"]:
        root, g = sources[i]
        w = g["arms"]["ancestor_third"]
        selected.append(
            dict(
                case=i,
                case_declaration=plan["cases"][i],
                teacher_root=str(root / "ancestor_third"),
                grown_root=str(root / "grown"),
                kick=w["kicks"][2]["result"],
                stand=w["stands"][2]["result"],
                actual_third_entry=w["boundaries"][1]["boundary"],
                source_shard=str(root / "complete.json"),
                source_shard_hash=digest(root / "complete.json"),
            )
        )
    for f in (a.plan_root / "complete.json", a.plan_root / "declaration.json"):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_TRAIN20_ANCESTOR_COUNTERFACTUAL_ALL_AUDITS_CLOSED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_hash=a.plan_hash,
            groups=groups,
            comparison=comparison,
            selected_physical_teachers=selected,
            source_own_native_steps=native,
            source_own_trajectory_neural_rows=neural,
            source_worlds=worlds,
            own_native_steps=0,
            own_neural_rows=0,
            optimizer_updates=0,
            bank="train",
            Fresh_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
