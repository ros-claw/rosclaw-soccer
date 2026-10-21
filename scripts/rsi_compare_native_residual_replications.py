"""Read-only full raw-array audit and all-three-seed decision. No simulator."""

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.contact_replication_decision import ball_outcome, decide
from rosclaw_soccer.rsi.native_contact_window import native_contact_window


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused audit output required")
    summaries, pins = [], {}
    trajectories = 0
    for seed in (7602, 7605, 7606):
        for task in ("train64", "dev64"):
            name = (
                "rsi-native-residual-full-train64-exam-20261021"
                if seed == 7602 and task == "train64"
                else "rsi-native-residual-dev64-exam-20261021"
                if seed == 7602
                else f"rsi-native-residual-seed{seed}-{task}-exam-20261021"
            )
            root = args.evidence_root / name
            report_path = root / "complete.json"
            report = json.loads(report_path.read_text())
            if (
                report.get("task_set", "train64") != task
                or report["checkpoint_intent"] != "bounded-residual"
                or report["all_strict_replays_exact"] is not True
                or report["all_representative_repeats_exact"] is not True
                or report["optimizer_updates"] != 0
                or report["goal_complete"] is not False
            ):
                raise ValueError("complete residual physical exam with exact replays required")
            pins.update(report["input_file_hashes"])
            pins[str(report_path)] = digest(report_path)
            starts_paths = [
                Path(p) for p in report["input_file_hashes"] if p.endswith("/train-starts.npz")
            ]
            if task == "train64":
                if len(starts_paths) != 1:
                    raise ValueError("one exact TRAIN start bank required")
                with np.load(starts_paths[0], allow_pickle=False) as z:
                    goals, feet = z["goal_world"].copy(), z["expected_foot"].copy()
            else:
                from rosclaw_soccer.rsi.mjlab_motion_curriculum import DEVELOPMENT_MOTION_NAMES

                devroot = args.evidence_root / "rsi-native-body-bound-dev-export-20261021"
                goals_list, feet_list = [], []
                for motion in DEVELOPMENT_MOTION_NAMES:
                    path = devroot / motion.removesuffix(".npz") / "starts.npz"
                    if str(path) not in report["input_file_hashes"]:
                        raise ValueError("exact DEV starts must be pinned by exam")
                    with np.load(path, allow_pickle=False) as z:
                        goals_list.append(z["goal_world"].copy())
                        feet_list.append(z["expected_foot"].copy())
                goals, feet = np.concatenate(goals_list), np.concatenate(feet_list)
            repeats = [0, 1, 2, 3, 32, 33, 34, 35] if task == "dev64" else list(range(8))
            expected = {
                (w, arm, rep)
                for w in range(64)
                for arm in ("initial", "final")
                for rep in range(2 if w in repeats else 1)
            }
            rows = report["results"]
            identities = [(r["world"], r["arm"], r["repeat"]) for r in rows]
            if len(rows) != 144 or set(identities) != expected:
                raise ValueError("complete64 paired arms and representative repeats required")
            summary = {
                arm: dict(clean=0, correct_foot_first=0, forbidden=0, stable=0, ball_outcome=0)
                for arm in ("initial", "final")
            }
            for row in rows:
                w, arm, rep = row["world"], row["arm"], row["repeat"]
                path = root / f"world{w}-{arm}-repeat{rep}/native.npz"
                if digest(path) != row["raw_hash"]:
                    raise ValueError("trajectory hash changed")
                pins[str(path)] = row["raw_hash"]
                with np.load(path, allow_pickle=False) as z:
                    raw = {k: z[k].copy() for k in z.files}
                cq = raw["canonical_qpos_after"].reshape(-1, 43)
                post = native_contact_window(
                    force=raw["foot_force"].reshape(-1, 2),
                    canonical_qpos=cq,
                    canonical_qvel=raw["canonical_qvel_after"].reshape(-1, 41),
                    goal=goals[w],
                    intended_foot=int(feet[w]),
                )
                actual = dict(
                    postcontact=post,
                    correct_foot_first=post["correct_foot_first"],
                    forbidden=bool((raw["nonfoot_force"] > 1).any()),
                    stable=bool(
                        cq[:, 2].min() >= 0.65
                        and (1 - 2 * (cq[:, 4] ** 2 + cq[:, 5] ** 2)).min() >= 0.9
                    ),
                )
                actual["clean"] = bool(actual["correct_foot_first"] and not actual["forbidden"])
                if any(actual[k] != row[k] for k in actual):
                    raise ValueError("independent raw metric recomputation differs")
                if rep:
                    with np.load(
                        root / f"world{w}-{arm}-repeat0/native.npz", allow_pickle=False
                    ) as first:
                        if set(first.files) != set(raw) or any(
                            not np.array_equal(first[k], raw[k]) for k in raw
                        ):
                            raise ValueError("whole independent neural repeat differs")
                else:
                    for key in report["summary"][arm]:
                        summary[arm][key] += bool(actual[key])
                    summary[arm]["ball_outcome"] += ball_outcome(actual)
                trajectories += 1
            for arm in summary:
                if any(summary[arm][k] != v for k, v in report["summary"][arm].items()):
                    raise ValueError("reported aggregate differs")
            summaries.append(dict(seed=seed, task_set=task, **summary))
    check_pins(pins)
    result = decide(summaries)
    result.update(
        schema="soccer.rsi.native_residual_three_seed_decision.v1",
        diagnostic_ball_outcome=(
            "observed1.2s,stable,no nonfoot>1N,goalprogress>=1m,"
            "lateral<=.3progress; NOT formal52"
        ),
        raw_trajectories_audited=trajectories,
        input_file_hashes=pins,
        new_physics_steps=0,
        new_neural_rows=0,
        optimizer_updates=0,
    )
    write_once(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_file_hashes"}), flush=True)


if __name__ == "__main__":
    main()
