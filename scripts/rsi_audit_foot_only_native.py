"""Independently audit all predeclared native foot-only trials from raw arrays."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen auditor required")
    import numpy as np

    from rosclaw_soccer.rsi.native_contact_window import native_contact_window

    pins = {
        str(f): digest(f)
        for root in (Path.cwd() / "src", Path.cwd() / "scripts")
        for f in root.rglob("*.py")
    }
    results = []
    baselines = {}
    steps = calls = 0
    for candidate in ("pilot", "warmup"):
        for world in (0, 1, 8, 16, 24):
            stem = f"foot-only-native-{candidate}-world{world}-20261021"
            wait_for_terminal_collection("rosclaw-" + stem + ".service")
            root = a.evidence_root / ("rsi-" + stem)
            receipt = root / "complete.json"
            report = json.loads(receipt.read_text())
            if (
                report.get("status") != "FOOT_ONLY_NATIVE_CONTACT_NOT_GOAL_OR_TEAM_QUALIFIED"
                or report.get("source_head") != "5938cba23586e1b7a256877e539c6fdb8ba9b36a"
                or report.get("all_full_independent_repeats_exact") is not True
                or len(report.get("results", [])) != 8
            ):
                raise ValueError("exact complete new-intent native campaign required")
            check_pins(report["input_file_hashes"])
            pins.update(report["input_file_hashes"])
            pins[str(receipt)] = digest(receipt)
            declaration = root / "declaration.json"
            pins[str(declaration)] = digest(declaration)
            if report.get("native_steps") != 18360 or report.get("neural_calls") != 1836:
                raise ValueError("full actual native and neural budgets required")
            steps += report["native_steps"]
            calls += report["neural_calls"]
            labels = {}
            for source in (7, 10):
                for arm in ("initial", "final"):
                    repeats = []
                    for repeat in (0, 1):
                        row = next(
                            r
                            for r in report["results"]
                            if (r["source"], r["arm"], r["repeat"]) == (source, arm, repeat)
                        )
                        path = root / f"source{source}-{arm}-repeat{repeat}/native.npz"
                        if digest(path) != row["raw_hash"]:
                            raise ValueError("raw native array hash differs")
                        pins[str(path)] = row["raw_hash"]
                        with np.load(path, allow_pickle=False) as raw:
                            values = {k: raw[k].copy() for k in raw.files}
                        if any(not np.isfinite(v).all() for v in values.values()):
                            raise ValueError("nonfinite native evidence")
                        if not np.array_equal(
                            values["canonical_qpos_before"][1:],
                            values["canonical_qpos_after"][:-1, -1],
                        ):
                            raise ValueError("native position sequence discontinuity")
                        if not np.array_equal(
                            values["canonical_qvel_before"][1:],
                            values["canonical_qvel_after"][:-1, -1],
                        ):
                            raise ValueError("native velocity sequence discontinuity")
                        repeats.append(values)
                    if repeats[0].keys() != repeats[1].keys() or any(
                        not np.array_equal(v, repeats[1][k]) for k, v in repeats[0].items()
                    ):
                        raise ValueError("whole independent NN/physics repeats differ")
                    raw = repeats[0]
                    if arm == "initial":
                        key = (world, source)
                        if candidate == "pilot":
                            baselines[key] = raw
                        elif any(not np.array_equal(v, baselines[key][k]) for k, v in raw.items()):
                            raise ValueError(
                                "identical CPU baseline diverged across candidate campaigns"
                            )
                    force = raw["foot_ball_contact_force_n"].reshape(-1, 2)
                    nf = raw["nonfoot_ball_contact_force_n"].reshape(-1)
                    cq = raw["canonical_qpos_after"].reshape(-1, 43)
                    cv = raw["canonical_qvel_after"].reshape(-1, 41)
                    with np.load(
                        a.evidence_root
                        / "rsi-ball-aware-course200-dev-20261021"
                        / f"{0 if source == 7 else 1}-initial-raw.npz"
                    ) as plan:
                        goal = plan["goal_target_world"][0, world].copy()
                    post = native_contact_window(
                        force=force,
                        canonical_qpos=cq,
                        canonical_qvel=cv,
                        goal=goal,
                        intended_foot=0 if source == 7 else 1,
                    )
                    forbidden = bool((nf > 1.0).any())
                    clean = bool(post["correct_foot_first"] and not forbidden)
                    stable = bool(
                        cq[:, 2].min() >= 0.65
                        and (1 - 2 * (cq[:, 4] ** 2 + cq[:, 5] ** 2)).min() >= 0.9
                    )
                    labels[arm] = dict(
                        clean=clean,
                        forbidden=forbidden,
                        correct_foot_first=post["correct_foot_first"],
                        stable=stable,
                    )
                    results.append(
                        dict(
                            candidate=candidate,
                            world=world,
                            source=source,
                            arm=arm,
                            **labels[arm],
                            maximum_nonfoot_force_n=float(nf.max()),
                            post_contact_window=post,
                        )
                    )
            # A failed candidate is preserved, never a missing result.
    check_pins(pins)
    summary = {}
    for c in ("pilot", "warmup"):
        summary[c] = {}
        for arm in ("initial", "final"):
            rows = [r for r in results if r["candidate"] == c and r["arm"] == arm]
            summary[c][arm] = {
                k: sum(bool(r[k]) for r in rows)
                for k in ("clean", "forbidden", "correct_foot_first", "stable")
            }
    rejected = all(summary[c]["final"]["clean"] <= summary[c]["initial"]["clean"] for c in summary)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_FOOT_ONLY_CANDIDATES_REJECTED_NOT_GOAL_COMPLETE"
            if rejected
            else "NATIVE_FOOT_ONLY_AUDITED_NOT_GOAL_COMPLETE",
            input_file_hashes=pins,
            summary=summary,
            results=results,
            audited_native_steps=steps,
            audited_neural_calls=calls,
            new_physics_steps=0,
            new_neural_calls=0,
            all_twenty_baseline_raw_repeats_bitexact=True,
            source_head=a.expected_head,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
