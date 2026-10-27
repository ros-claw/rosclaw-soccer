"""Compare all replayed same-condition relay feedback arms and old baselines."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.relay_balance_comparison import compare_balance_outcomes
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def report(
    root: Path,
    audit_path: Path,
    expected_audit_hash: str,
    expected_status: str,
    *,
    worlds: int = 32,
) -> tuple[dict, dict, str]:
    path = root / "complete.json"
    if (
        path.is_symlink()
        or audit_path.is_symlink()
        or (root / "failed.json").exists()
        or (audit_path.parent / "failed.json").exists()
        or digest(audit_path) != expected_audit_hash
    ):
        raise ValueError("exact pinned source and independent replay required")
    source_hash = digest(path)
    source, audit = (json.loads(p.read_text()) for p in (path, audit_path))
    if (
        source.get("status") != expected_status
        or audit.get("status") != "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH"
        or audit.get("producer_hash") != source_hash
        or audit.get("worlds") != worlds
        or source.get("counts") != audit.get("counts")
        or len(source.get("records", [])) != worlds
        or len(audit.get("records", [])) != worlds
        or any(
            v.get(k) is not False
            for v in (source, audit)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("complete unpromoted own-native relay replay required")
    for index, (row, own) in enumerate(zip(source["records"], audit["records"], strict=True)):
        if row["index"] != index or any(
            row[k] != own[k] for k in ("index", "course", "termination", "result")
        ):
            raise ValueError("source and own replay outcomes must match")
    return source, audit, source_hash


def bind(root: Path, row: dict, name: str) -> Path | None:
    item = row["artifacts"][name]
    if item is None:
        return None
    path = root / item["path"]
    if (
        path.is_symlink()
        or path.resolve().parent != (root / f"trial{row['index']}").resolve()
        or digest(path) != item["hash"]
    ):
        raise ValueError("exact confined original trajectory required")
    return path


def verify_same_handoff(root: Path, baseline: dict, candidate: dict) -> None:
    """Bind all arms to one real pre-intervention trajectory and private state."""
    b, c = baseline["handoff"], candidate["handoff"]
    if (b is None) != (c is None):
        raise ValueError("teacher cannot alter pre-handoff readiness")
    if baseline["artifacts"]["initial"]["hash"] != candidate["artifacts"]["initial"]["hash"]:
        raise ValueError("paired initial physical state differs")
    if b is None:
        for name in baseline["artifacts"]:
            old, new = baseline["artifacts"][name], candidate["artifacts"][name]
            if (old is None) != (new is None) or old is not None and old["hash"] != new["hash"]:
                raise ValueError("teacher changed a world without a handoff")
        return
    if (
        b["control"] != c["control"]
        or b["boundary"]["hash"] != c["boundary"]["hash"]
        or b["return_goal"] != c["return_goal"]
        or b["kick_origin"] != c["kick_origin"]
    ):
        raise ValueError("paired physical and private handoff state differs")
    count = b["control"]
    for name in ("physics", "passer_nn", "receiver_velocity", "receiver_shadow"):
        left, right = bind(root, baseline, name), bind(root, candidate, name)
        if left is None or right is None:
            raise ValueError("complete pre-handoff trace required")
        with np.load(left, allow_pickle=False) as x, np.load(right, allow_pickle=False) as y:
            if set(x.files) != set(y.files) or any(
                x[key].shape[0] < count
                or y[key].shape[0] < count
                or not np.array_equal(x[key][:count], y[key][:count])
                for key in x.files
            ):
                raise ValueError("physical, target or private neural prefix differs")


def measured(root: Path, row: dict) -> dict:
    physics = bind(root, row, "physics")
    if physics is None:
        raise ValueError("native physics trajectory required")
    with np.load(physics, allow_pickle=False) as f:
        failure = f["body_failed"]
        if failure.dtype != np.bool_ or failure.shape != (row["controls"], 2):
            raise ValueError("actual paired body failure trace required")
        last = failure[-1]
    nn = bind(root, row, "receiver_return_nn")
    max_delta = 0.0
    changed = 0
    if nn is not None:
        with np.load(nn, allow_pickle=False) as f:
            if "BAL_actual_HG" in f:
                delta = f["BAL_actual_HG"]
                if (
                    delta.ndim != 2
                    or delta.shape[1] != 29
                    or not np.isfinite(delta).all()
                    or np.max(np.abs(delta)) > 0.080002
                ):
                    raise ValueError("recorded physical teacher correction escaped authority")
                max_delta = float(np.max(np.abs(delta)))
                changed = int(np.any(delta != 0, axis=1).sum())
    return dict(
        body_failure=row["termination"] == "BODY_FAILURE",
        A_failed=bool(last[0]),
        B_failed=bool(last[1]),
        handoff=row["result"]["handoff_executed"],
        outgoing_foot_strike=row["result"]["outgoing_foot_strike"],
        return_0p1m_and_healthy=row["result"]["return_0p1m_and_healthy"],
        max_teacher_delta_rad=max_delta,
        changed_teacher_controls=changed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        nargs=6,
        action="append",
        required=True,
        metavar=(
            "CURRENT",
            "CURRENT_AUDIT",
            "PRIOR",
            "PRIOR_AUDIT",
            "AUDIT_HASH",
            "PRIOR_AUDIT_HASH",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        len(args.group) != 3
        or args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("three complete paired studies and unused external output required")
    comparisons = {0: [], 1: [], 2: []}
    detail, provenance, seen = [], [], set()
    for current_name, audit_name, prior_name, prior_audit_name, ah, ph in args.group:
        root, old_root = Path(current_name), Path(prior_name)
        current, _, current_hash = report(
            root,
            Path(audit_name),
            ah,
            "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
        )
        old, _, old_hash = report(
            old_root,
            Path(prior_audit_name),
            ph,
            "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
        )
        dec = json.loads((root / "declaration.json").read_text())
        old_dec = json.loads((old_root / "declaration.json").read_text())
        block = dec.get("development_context_block")
        if (
            type(block) is not int
            or block not in (0, 1, 2)
            or block in seen
            or old_dec.get("development_context_block") != block
            or digest(root / "declaration.json") != current["declaration_hash"]
            or digest(old_root / "declaration.json") != old["declaration_hash"]
        ):
            raise ValueError("exact matched predeclared development blocks required")
        seen.add(block)
        old_baselines = {
            (r["course"]["context"], r["course"]["lateral_m"]): r
            for r in old["records"]
            if r["course"]["arm"]["start"] == 80
            and r["course"]["arm"]["memory"] == "own_shadow_carry"
        }
        new = {
            (
                r["course"]["context"],
                r["course"]["lateral_m"],
                r["course"]["arm"]["balance_teacher"],
            ): r
            for r in current["records"]
        }
        if len(old_baselines) != 8 or len(new) != 32:
            raise ValueError("complete baseline and all 3 alternatives required")
        for (context, side), old_row in old_baselines.items():
            base = new[(context, side, -1)]
            if (
                base["termination"] != old_row["termination"]
                or base["result"] != old_row["result"]
                or bool(base["handoff"]) != bool(old_row["handoff"])
            ):
                raise ValueError("old baseline outcome changed")
            for name in base["artifacts"]:
                a, b = base["artifacts"][name], old_row["artifacts"][name]
                bind(root, base, name)
                bind(old_root, old_row, name)
                if (a is None) != (b is None) or a is not None and a["hash"] != b["hash"]:
                    raise ValueError("old baseline raw stream not byte-identical")
            if (
                base["handoff"] is not None
                and base["handoff"]["boundary"]["hash"] != old_row["handoff"]["boundary"]["hash"]
            ):
                raise ValueError("old baseline handoff changed")
            parent = measured(root, base)
            for teacher in (0, 1, 2):
                candidate_row = new[(context, side, teacher)]
                verify_same_handoff(root, base, candidate_row)
                candidate = measured(root, candidate_row)
                comparisons[teacher].append((parent, candidate))
                detail.append(
                    dict(
                        block=block,
                        context=context,
                        lateral_m=side,
                        teacher=teacher,
                        parent=parent,
                        candidate=candidate,
                    )
                )
        provenance.append(
            dict(
                block=block,
                source_hash=current_hash,
                source_audit_hash=ah,
                prior_hash=old_hash,
                prior_audit_hash=ph,
            )
        )
    if seen != {0, 1, 2} or any(len(v) != 24 for v in comparisons.values()):
        raise ValueError("all 24 matched initial conditions and every teacher required")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="RELAY_BALANCE_PAIRED_DEVELOPMENT_NOT_PROMOTED",
            source_head=args.expected_head,
            provenance=provenance,
            summaries={str(k): compare_balance_outcomes(v) for k, v in comparisons.items()},
            records=detail,
            original_baselines_raw_byte_identical=True,
            paired_physical_and_private_handoff_prefix_exact=True,
            teachers_are_not_learned_actor=True,
            optimization_updates=0,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
