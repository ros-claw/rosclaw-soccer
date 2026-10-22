"""Frozen, bounded native Fresh study admission and model/data helpers."""

import json
import subprocess
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest


def clean(head):
    if (
        subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != head
    ):
        raise ValueError("exact clean frozen native Fresh source required")


def source_pins(pins):
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})


def receipt(pins, h):
    paths = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
    if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != h:
        raise ValueError("unique exact immutable native Fresh lineage required")
    return paths[0], json.loads(paths[0].read_text())


def admit_frontier(root, expected_head, expected_hash):
    rp = root / "complete.json"
    if rp.is_symlink() or (root / "failed.json").exists() or digest(rp) != expected_hash:
        raise ValueError("exact completed accumulated-skill frontier report required")
    r = json.loads(rp.read_text())
    if (
        r["status"] != "ALL_THREE_ACQUIRED_MOTOR_FRONTIERS_REVIEWED_NOT_PROMOTED"
        or r["source_head"] != expected_head
        or [x["seed"] for x in r["reports"]] != [65023, 65024, 65025]
        or any(
            r[k] is not False
            for k in (
                "promotion_authorized",
                "hardware_authorized",
                "legacy52_verified",
                "Fresh_verified",
                "team_game_verified",
            )
        )
    ):
        raise ValueError("all three reviewed, unpromoted native motor candidates required")
    eligible = [x["seed"] for x in r["reports"] if x["acquired_frontier_retained"]]
    if not eligible or eligible != r["retained_seed_ids"]:
        raise ValueError("only candidates without any cumulative historical loss may open Fresh")
    pins = dict(r["input_file_hashes"])
    pins[str(rp)] = expected_hash
    candidates = []
    training_data = None
    for report in r["reports"]:
        required = set(report["required_contexts"])
        achieved = set(report["achieved_contexts"])
        if (
            report["lost_contexts"] != sorted(required - achieved)
            or report["acquired_frontier_retained"] != (not report["lost_contexts"])
            or any(t["critical"] is not True for t in report["task_retention"])
        ):
            raise ValueError(
                "exact zero-loss eligibility, not highest-score seed selection, required"
            )
        if report["seed"] not in eligible:
            continue
        _, own = receipt(pins, report["own_audit_hash"])
        if (
            own["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
            or own["worlds"] != 44
            or own["checkpoint_kind"] != "contact-consolidated-v2"
            or own["checkpoint_hash"] != report["checkpoint_hash"]
        ):
            raise ValueError("independent whole44 actual candidate identity required")
        tp, training = receipt(pins, own["training_hash"])
        if training["config"]["seed"] != report["seed"]:
            raise ValueError("exact frozen independently trained seed required")
        if training_data is not None and training_data != training["data_hash"]:
            raise ValueError("same fixed teacher/correction training bank required")
        training_data = training["data_hash"]
        candidates.append(
            dict(
                seed=report["seed"],
                training_root=str(tp.parent),
                training_hash=own["training_hash"],
                checkpoint_hash=own["checkpoint_hash"],
                kind="contact-consolidated-v2",
            )
        )
    _, bank = receipt(pins, training_data)
    _, base = receipt(pins, bank["base_data_hash"])
    baseline_hash = base["warm_start_hash"]
    baseline_audits = []
    for p in pins:
        if p.endswith("/native-audit/complete.json"):
            row = json.loads(Path(p).read_text())
            if (
                row.get("status") == "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
                and row["training_hash"] == baseline_hash
            ):
                baseline_audits.append(row)
    if len(baseline_audits) != 1:
        raise ValueError("unique actual W6 comparison policy required")
    np, native = receipt(pins, baseline_audits[0]["source_hash"])
    dp = np.parent / "declaration.json"
    if digest(dp) != native["declaration_hash"] or pins.get(str(dp)) != digest(dp):
        raise ValueError("unchanged consumed source curriculum required")
    contexts = json.loads(dp.read_text())["contexts"]
    return pins, baseline_hash, candidates, contexts


def read_seal(root, expected_head):
    rp = root / "complete.json"
    if rp.is_symlink() or (root / "failed.json").exists():
        raise ValueError("successful immutable pre-physics seal required")
    seal = json.loads(rp.read_text())
    if (
        seal["status"] != "NATIVE_FRESH56_SEALED_AFTER_ALL_WEIGHTS_FROZEN"
        or seal["source_head"] != expected_head
        or seal["physical_steps"] != 0
        or seal["fit_rows"] != 0
    ):
        raise ValueError("new unseen initial conditions sealed before any Fresh physics required")
    dp = root / "declaration.json"
    if dp.is_symlink() or digest(dp) != seal["declaration_hash"]:
        raise ValueError("exact pre-physics study declaration required")
    declared = json.loads(dp.read_text())
    if any(seal.get(k) != v for k, v in declared.items()):
        raise ValueError("completed seal differs from declaration")
    cp = root / "private-courses.json"
    if cp.is_symlink() or digest(cp) != seal["courses_hash"]:
        raise ValueError("immutable private course declaration required")
    courses = json.loads(cp.read_text())
    pins = dict(seal["input_file_hashes"])
    frontier_path, _ = receipt(pins, seal["frontier_hash"])
    _, baseline, candidates, contexts = admit_frontier(
        frontier_path.parent, seal["frontier_head"], seal["frontier_hash"]
    )
    from rosclaw_soccer.rsi.native_fresh_curriculum import native_fresh_courses

    if (
        courses != native_fresh_courses(contexts)
        or candidates != seal["candidates"]
        or baseline != seal["baseline_training_hash"]
    ):
        raise ValueError("sealed courses, eligibility or model identities changed")
    pins[str(rp)], pins[str(cp)] = digest(rp), seal["courses_hash"]
    pins[str(dp)] = seal["declaration_hash"]
    return seal, pins, courses


def open_fresh_actors(pins, seal, courses):
    from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
    from scripts.rsi_stroke_search_common import open_motor

    bench, baseline = open_motor(
        pins, seal["baseline_training_hash"], {c["index"]: c["ctx"] for c in courses}
    )
    actors = {"baseline": baseline}
    for candidate in seal["candidates"]:
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        actors[str(candidate["seed"])] = actor
    return bench, actors
