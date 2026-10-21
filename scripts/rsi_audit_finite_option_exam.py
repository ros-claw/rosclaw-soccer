"""Independently reconstruct finite-option frame-zero exam outcomes, no stepping."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    from rosclaw_soccer.rsi.contact_replication_decision import ball_outcome
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exam-root", type=Path, required=True)
    p.add_argument("--exam-hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    receipt = a.exam_root / "complete.json"
    if a.output.exists() or digest(receipt) != a.exam_hash:
        raise ValueError("unused output and exact completed exam required")
    r = json.loads(receipt.read_text())
    if (
        r["checkpoint_intent"]
        not in ("bounded-recovery-option", "bounded-recovery-continuation", "bounded-approach40")
        or r["optimizer_updates"] != 0
        or any(
            r[k] is not True
            for k in ("all_strict_replays_exact", "all_representative_repeats_exact")
        )
    ):
        raise ValueError("actual complete native finite-option proof required")
    pins = {**r["input_file_hashes"], str(receipt): a.exam_hash}
    pins[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    approach = r["checkpoint_intent"] == "bounded-approach40"
    initialization_rows = {}
    initialization_root = None
    if approach:
        paths = [
            Path(p)
            for p, h in pins.items()
            if p.endswith("/complete.json") and h == r["initialization132_exam_hash"]
        ]
        if len(paths) != 1 or r.get("initial132_full_trajectory_exact") is not True:
            raise ValueError("exact source132 exam and declared new baseline proof required")
        path = paths[0]
        if digest(path) != r["initialization132_exam_hash"]:
            raise ValueError("initialization exam changed")
        parent = json.loads(path.read_text())
        if (
            parent.get("checkpoint_intent") != "bounded-recovery-continuation"
            or parent.get("task_set") != r["task_set"]
        ):
            raise ValueError("same task parent132 reference required")
        initialization_rows = {
            (x["world"], x["repeat"]): x for x in parent["results"] if x["arm"] == "final"
        }
        initialization_root = path.parent

    def pinned(suffix):
        found = [Path(p) for p in pins if p.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("unambiguous actual input required: " + suffix)
        return found[0]

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    dev = r["task_set"] == "dev64"
    if dev:
        sources = [
            load(pinned("/" + n + "/starts.npz"))
            for n in ("soccer-standard-007_left", "soccer-standard-010_right")
        ]
        starts = {k: np.concatenate([x[k] for x in sources]) for k in sources[0]}
        repeats = [0, 1, 2, 3, 32, 33, 34, 35]
    else:
        starts = load(pinned("/train-starts.npz"))
        repeats = list(range(8))
    expected = {
        (w, arm, rep)
        for w in range(64)
        for arm in ("initial", "final")
        for rep in range(2 if w in repeats else 1)
    }
    ids = [(x["world"], x["arm"], x["repeat"]) for x in r["results"]]
    if len(ids) != 144 or set(ids) != expected:
        raise ValueError("full64 paired arms plus declared repeats required")
    check_pins(pins)
    summary = {arm: {k: 0 for k in r["summary"][arm]} for arm in ("initial", "final")}
    first = {}
    controls = 0
    for row in r["results"]:
        w, arm, rep = row["world"], row["arm"], row["repeat"]
        path = a.exam_root / f"world{w}-{arm}-repeat{rep}/native.npz"
        if digest(path) != row["raw_hash"]:
            raise ValueError("raw trajectory hash changed")
        pins[str(path)] = row["raw_hash"]
        z = load(path)
        length = len(z["target_rad"])
        if row["controls"] != length:
            raise ValueError("control count differs")
        controls += length
        q = z["canonical_qpos_after"].reshape(-1, 43)
        v = z["canonical_qvel_after"].reshape(-1, 41)
        f = z["foot_force"].reshape(-1, 2)
        post = native_contact_window(
            force=f,
            canonical_qpos=q,
            canonical_qvel=v,
            goal=starts["goal_world"][w],
            intended_foot=int(starts["expected_foot"][w]),
        )
        actual = dict(
            postcontact=post,
            forbidden=bool((z["nonfoot_force"] > 1).any()),
            stable=bool(
                q[:, 2].min() >= 0.65 and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
            ),
            correct_foot_first=post["correct_foot_first"],
        )
        actual["clean"] = bool(actual["correct_foot_first"] and not actual["forbidden"])
        actual["ball_outcome"] = ball_outcome(actual)
        actual["correct_foot_ball_outcome"] = bool(actual["ball_outcome"] and actual["clean"])
        if any(row[k] != val for k, val in actual.items()):
            raise ValueError("raw-derived outcomes differ")
        hits = np.flatnonzero((f > 10).any(axis=1))
        contact = int(hits[0] // 10) if len(hits) else None
        if contact != row["first_measured_contact_control"]:
            raise ValueError("contact clock is not measured")
        name = row["source"].removesuffix(".npz")
        horizon = len(
            load(pinned("/" + name + "/plan.npz" if dev else "/" + name + "-plan.npz"))["joint"]
        )
        expected_head = min(horizon, contact + 15) if contact is not None else horizon
        head = min(length, expected_head)
        if (
            row["reference_controls"] != head
            or row["recovery_controls"] != length - head
            or length - head > 60
        ):
            raise ValueError("handoff used another clock/budget")
        if not np.array_equal(z["recovery_phase"], (np.arange(length) >= head).astype(int)):
            raise ValueError("controller phase is not a single measured handoff")
        if z["observation"][head:, 98:].any() or z["observation"][head:, 6:9].any():
            raise ValueError("recovery used reference/ball/future commands")
        bad = (
            (z["canonical_qpos_after"][:, :, 2] < 0.65)
            | (
                1
                - 2
                * (
                    z["canonical_qpos_after"][:, :, 4] ** 2
                    + z["canonical_qpos_after"][:, :, 5] ** 2
                )
                < 0.9
            )
        ).any(axis=1)
        if bad.any() and not np.array_equal(np.flatnonzero(bad), [length - 1]):
            raise ValueError("body kept advancing after guard failure")
        if approach and arm == "initial":
            reference = initialization_rows[w, rep]
            path = initialization_root / f"world{w}-final-repeat{rep}/native.npz"
            if digest(path) != reference["raw_hash"]:
                raise ValueError("parent132 source trajectory changed")
            pins[str(path)] = reference["raw_hash"]
            baseline = load(path)
            if set(baseline) != set(z) or any(not np.array_equal(baseline[k], z[k]) for k in z):
                raise ValueError("new initial132 full execution differs from actual parent132")
        elif not dev and arm == "initial" and rep == 0:
            baseline = load(
                pinned(f"/rsi-body-bound-contact-episodes-20261021/world{w}-repeat0.npz")
            )
            if any(not np.array_equal(baseline[k][:head], z[k][:head]) for k in baseline):
                raise ValueError("original approach/reference prefix changed")
        if w in repeats:
            key = (w, arm)
            if rep == 0:
                first[key] = z
            else:
                previous = first.pop(key)
                if set(z) != set(previous) or any(not np.array_equal(z[k], previous[k]) for k in z):
                    raise ValueError("whole independent neural repeat differs")
        if rep == 0:
            for k in summary[arm]:
                summary[arm][k] += bool(actual[k])
    if (
        first
        or summary != r["summary"]
        or controls != r["policy_rows"]
        or controls * 10 != r["live_native_steps"]
        or r["live_native_steps"] != r["replay_native_steps"]
    ):
        raise ValueError("aggregate or actual work counters differ")
    check_pins(pins)
    write_once(
        a.output,
        dict(
            schema="soccer.rsi.finite_option_exam_independent_audit.v1",
            input_file_hashes=pins,
            summary=summary,
            raw_trajectories=144,
            phase_clock_and_stop_exact=True,
            recovery_inputs_zero79_zero_command=True,
            whole_repeats_exact=True,
            original_baseline_prefix_exact=True if not dev and not approach else None,
            initial132_full_trajectory_exact=True if approach else None,
            new_physics_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    print(
        json.dumps(dict(summary=summary, raw_trajectories=144, new_physics=0, new_NN=0)), flush=True
    )


if __name__ == "__main__":
    main()
