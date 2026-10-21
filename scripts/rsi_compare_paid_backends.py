"""Fixed paired native ONNX/Torch backend gate; no best-case selection."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-hash", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen exact comparison code required")
    roots = [a.baseline, a.candidate]
    pins, receipts, declarations = {}, [], []
    for root, expected, status in zip(
        roots,
        [a.baseline_hash, a.candidate_hash],
        [
            "PAID_RESEARCH_NATIVE_TEACHER_PROBE_NOT_PROMOTED",
            "PAID_RESEARCH_NATIVE_TORCH_PROBE_NOT_PROMOTED",
        ],
        strict=True,
    ):
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("exact backend receipt required")
        r = json.loads(path.read_text())
        if (
            r["status"] != status
            or r["strict_replays_exact"] is not True
            or r["whole_repeats_exact"] is not True
        ):
            raise ValueError("both actual native backend experiments must be fully closed")
        pins.update(r["input_file_hashes"])
        pins[str(path)] = expected
        declaration = root / "declaration.json"
        pins[str(declaration)] = digest(declaration)
        for relative, h in r["raw_hashes"].items():
            pins[str(root / relative)] = h
        declarations.append(json.loads(declaration.read_text()))
        receipts.append(r)
    check_pins(pins)
    for key in (
        "profiles",
        "cases",
        "repetitions",
        "maximum_reference_controls",
        "maximum_held_target_tail_controls",
    ):
        if declarations[0][key] != declarations[1][key]:
            raise ValueError("paired fixed case/physics/reference protocol differs")
    maps = [{(x["profile"], x["case"], x["repeat"]): x for x in r["results"]} for r in receipts]
    if set(maps[0]) != set(maps[1]) or any(len(m) != 96 for m in maps):
        raise ValueError("all full16/profile/repeat pairs required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            expected_head=a.expected_head,
            allowed_ball_window_difference_m=0.05,
            allowed_contact_time_difference_substeps=5,
            allowed_minimum_height_and_upright_difference=0.01,
            require_exact_boolean_outcomes=True,
            all96_pairs=True,
            native_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            commercial_publicity_authorized=False,
        ),
    )
    results = []
    for key in sorted(maps[0]):
        old, new = [m[key] for m in maps]
        for name in ("source", "ball_mass_kg", "ball_world_initial", "goal_world"):
            if old[name] != new[name]:
                raise ValueError("declared paired task identity differs")
        if digest(roots[0] / key[0] / "declared-scene.xml") != digest(
            roots[1] / key[0] / "declared-scene.xml"
        ):
            raise ValueError("backend compiled scene recipes must be identical")
        op, np_ = old["postcontact"], new["postcontact"]
        reasons, errors = [], {}
        for name in ("body_failed", "forbidden", "controls", "neural_rows"):
            if old[name] != new[name]:
                reasons.append(name)
        for name in (
            "correct_foot_first",
            "post_contact_quality_observed",
            "post_contact_status",
            "other_foot_contact_in_window",
        ):
            if op[name] != np_[name]:
                reasons.append(name)
        for name, tolerance in (
            ("first_foot_contact_substep", 5),
            ("goal_progress_1p2s_m", 0.05),
            ("goal_lateral_1p2s_m", 0.05),
            ("minimum_post_contact_pelvis_height_m", 0.01),
            ("minimum_post_contact_upright", 0.01),
            ("maximum_pelvis_retreat_after_contact_m", 0.01),
        ):
            if op[name] is None or np_[name] is None:
                if op[name] != np_[name]:
                    reasons.append(name)
            else:
                error = abs(op[name] - np_[name])
                errors[name] = error
                if not np.isfinite(error) or error > tolerance:
                    reasons.append(name)
        results.append(
            dict(
                profile=key[0],
                case=key[1],
                repeat=key[2],
                rejected_reasons=reasons,
                absolute_errors=errors,
            )
        )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="PAID_BACKEND_NATIVE_PARITY_PASSED"
            if all(not x["rejected_reasons"] for x in results)
            else "PAID_BACKEND_NATIVE_PARITY_REJECTED",
            input_file_hashes=pins,
            results=results,
            native_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            football_skill_improved=False,
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
