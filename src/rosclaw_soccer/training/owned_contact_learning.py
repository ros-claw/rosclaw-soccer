"""Offline bounded policy search on bilateral physical pass outcomes, not motion."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import run_probe, validate_probe


def reward(report: dict[str, Any]) -> tuple[int, int, int]:
    """Safety first, then contact-confirmed pass, then controlled ball progress."""
    if not report["exact_replay"] or not report["results"][0]["safe"]:
        return (-1, 0, 0)
    events = report["assessment"]["events"]
    return (
        1,
        sum(r["physical_receive_confirmed"] is True for r in report.get("causal_pass_feedback", []))
        if report["assessment"].get("gates", {}).get("foot_only_ball_control") is True
        else 0,
        sum(e["skill"] in {"dribble", "ball_win"} for e in events),
    )


def _run(job: tuple[str, str, bool, OwnedBallContactPolicy | None, float]) -> str:
    asset, output, blue, policy, offset = job
    run_probe(
        asset_root=Path(asset),
        output=Path(output),
        active=True,
        duration=20.0,
        four_vs_four=True,
        blue_kickoff=blue,
        contact_policy=policy,
        kickoff_offset_m=offset,
    )
    return str(Path(output) / "probe.json")


def train(*, asset_root: Path, output: Path, workers: int = 2) -> dict[str, Any]:
    if output.exists() or workers not in (1, 2):
        raise ValueError("use a fresh evidence directory and one or two workers")
    policies = (
        None,
        OwnedBallContactPolicy(0.24, 0.12, 1.5),
        OwnedBallContactPolicy(0.20, 0.12, 2.0),
        OwnedBallContactPolicy(0.30, -0.12, 1.5),
        OwnedBallContactPolicy(0.20, -0.12, 2.0),
    )
    output.mkdir(parents=True)
    request = {
        "method": "bounded_offline_parameter_policy_search",
        "causal_pass_required": True,
        "policies": [None if p is None else asdict(p) for p in policies],
        "training_offsets_m": [0.0],
        "heldout_offsets_m": [0.12, -0.12],
        "candidate_count": len(policies),
        "workers": workers,
        "neural_network_training": False,
        "activation_ceiling": "SIM_ONLY",
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    jobs = [
        (str(asset_root), str(output / f"train-{i}-{'blue' if blue else 'red'}"), blue, p, 0.0)
        for i, p in enumerate(policies)
        for blue in (False, True)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(_run, jobs))
    reports = [validate_probe(Path(p)) for p in paths]
    # The worse side is decisive: do not learn a red-only advantage.
    scores = [min(reward(r) for r in reports[2 * i : 2 * i + 2]) for i in range(len(policies))]
    selected = max(range(1, len(policies)), key=lambda i: (scores[i], -i))
    policy = policies[selected]
    assert policy is not None
    selection = {
        "training_scores": scores,
        "selected_index": selected,
        "selected_policy": asdict(policy),
        "policy_hash": policy.policy_hash,
        "selected_before_heldout": True,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    jobs = [
        (str(asset_root), str(output / f"exam-{name}-{'blue' if blue else 'red'}"), blue, p, offset)
        for name, p in (("baseline", None), ("candidate", policy))
        for blue, offset in ((False, 0.12), (True, -0.12))
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        exam_paths = list(pool.map(_run, jobs))
    exam = [validate_probe(Path(p)) for p in exam_paths]
    candidate_passes = all(reward(r)[0] == 1 and reward(r)[1] > 0 for r in exam[2:])
    gain = sum(reward(r)[1] for r in exam[2:]) > sum(reward(r)[1] for r in exam[:2])
    retention = scores[selected][0] == 1 and scores[selected] >= scores[0]
    result = {
        "schema_version": "rosclaw_soccer.owned_contact_learning.v1",
        "request": request,
        "selection": selection,
        "training_sources": {p: hash_bytes(Path(p).read_bytes()) for p in paths},
        "exam_sources": {p: hash_bytes(Path(p).read_bytes()) for p in exam_paths},
        "exam_scores": [reward(r) for r in exam],
        "heldout_bilateral_pass_gain": candidate_passes and gain,
        "training_retention_passed": retention,
        "status": "PASS_BOUNDED_PASS_EXAM"
        if candidate_passes and gain and retention
        else "REJECTED_PASS_EXAM",
        "promotion_eligible": False,
        "hardware_command_sent": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "learning.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit(path: Path) -> dict[str, Any]:
    """Recompute outcomes and rejection from all hash-bound physical sources."""
    result = json.loads(path.read_text())
    expected = result.pop("report_hash")
    if hash_json(result) != expected:
        raise ValueError("learning report changed")
    groups = []
    diagnostics = {}
    for key in ("training_sources", "exam_sources"):
        reports = []
        for name, digest in result[key].items():
            source = Path(name).resolve()
            if (
                not source.is_relative_to(path.parent.resolve())
                or hash_bytes(source.read_bytes()) != digest
            ):
                raise ValueError("learning evidence source changed or escaped its root")
            report = validate_probe(source)
            reports.append(report)
            with np.load(source.parent / "primary.npz", allow_pickle=False) as archive:
                trace = {k: archive[k] for k in archive.files}
            ids = tuple(sorted(c["self_model"]["agent_id"] for c in report["cells"]))
            diagnostics[name] = diagnose_passes(trace, ids)
        groups.append(reports)
    training, exam = groups
    if len(training) != 10 or len(exam) != 4:
        raise ValueError("bounded search evidence set is incomplete")

    # Preserve the original scoring contract when auditing historical runs,
    # while deriving the stronger causal gate below from raw traces.
    def source_reward(r: dict[str, Any]) -> tuple[int, int, int]:
        score = reward(r)
        if not result["request"].get("causal_pass_required", False) and score[0] == 1:
            return (1, sum(e["skill"] == "pass" for e in r["assessment"]["events"]), score[2])
        return score

    scores = [min(source_reward(r) for r in training[2 * i : 2 * i + 2]) for i in range(5)]
    selected = max(range(1, 5), key=lambda i: (scores[i], -i))
    selection = result["selection"]
    if (
        selection["training_scores"] != [list(v) for v in scores]
        or selection["selected_index"] != selected
    ):
        raise ValueError("selection differs from physical training outcomes")
    policy = OwnedBallContactPolicy(**selection["selected_policy"])
    if (
        policy.policy_hash != selection["policy_hash"]
        or selection["selected_policy"] != result["request"]["policies"][selected]
    ):
        raise ValueError("candidate policy binding changed")
    exam_scores = [source_reward(r) for r in exam]
    gain = all(v[0] == 1 and v[1] > 0 for v in exam_scores[2:]) and sum(
        v[1] for v in exam_scores[2:]
    ) > sum(v[1] for v in exam_scores[:2])
    retention = scores[selected][0] == 1 and scores[selected] >= scores[0]
    if (
        result["exam_scores"] != [list(v) for v in exam_scores]
        or result["heldout_bilateral_pass_gain"] is not gain
        or result["promotion_eligible"] is not False
        or result["hardware_command_sent"] is not False
        or result["activation_ceiling"] != "SIM_ONLY"
    ):
        raise ValueError("learning outcome or authority claims differ from evidence")
    derived_status = "PASS_BOUNDED_PASS_EXAM" if retention and gain else "REJECTED_PASS_EXAM"
    if result["status"] != derived_status:
        raise ValueError("training failures cannot be bypassed by a heldout success")
    exam_names = list(result["exam_sources"])
    causal_counts = [
        sum(row["physical_receive_confirmed"] for row in diagnostics[name]) for name in exam_names
    ]
    causal_gain = (
        all(n > 0 for n in causal_counts[2:])
        and all(r["results"][0]["safe"] for r in exam[2:])
        and all(
            r["assessment"].get("gates", {}).get("foot_only_ball_control") is True for r in exam[2:]
        )
        and sum(causal_counts[2:]) > sum(causal_counts[:2])
    )
    strict_status = "PASS_BOUNDED_PASS_EXAM" if retention and causal_gain else "REJECTED_PASS_EXAM"
    output = {
        "source_hash": hash_bytes(path.read_bytes()),
        "source_report_hash": expected,
        "status": strict_status,
        "original_contract_status": derived_status,
        "training_retention_passed": retention,
        "heldout_bilateral_pass_gain": gain,
        "causal_heldout_pass_gain": causal_gain,
        "causal_pass_counts": {
            name: sum(row["physical_receive_confirmed"] for row in rows)
            for name, rows in diagnostics.items()
        },
        "historical_postcontact_intent_credit": not result["request"].get(
            "causal_pass_required", False
        ),
        "pass_diagnostics": diagnostics,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
    }
    output["audit_hash"] = hash_json(output)
    destination = path.with_suffix(".strict-audit.json")
    if destination.exists() and json.loads(destination.read_text()) != output:
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.audit is not None:
        print(json.dumps(audit(args.audit)))
    else:
        if args.asset_root is None or args.output is None:
            parser.error("training requires --asset-root and --output")
        print(
            json.dumps(train(asset_root=args.asset_root, output=args.output, workers=args.workers))
        )


if __name__ == "__main__":
    main()
