"""Independent32-context own-NN audit AND every-phase native physics integration."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "declaration-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("declaration-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen clean Fresh auditor required")

    clean()
    if a.output.exists():
        raise ValueError("unused audit output required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        clean()
        if (a.root / "failed.json").exists():
            raise ValueError("upstream failed; no physics retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed Fresh arm required")
        time.sleep(10)
    clean()
    from rosclaw_soccer.rsi.paid_fresh_evidence import load_fresh_actor, load_fresh_declaration
    from rosclaw_soccer.rsi.paid_fresh_protocol import fresh_summary

    preflight, fd = load_fresh_declaration(a.declaration_root, a.declaration_hash)
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(path.read_text()) for path in (rp, dp)]
    if (
        (a.root / "failed.json").exists()
        or r["status"] != "PAID_FRESH32_LIFECYCLE_ALL_WORLDS_CLOSED_PENDING_INDEPENDENT_AUDIT"
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or r["declaration_hash"] != digest(dp)
        or r["fresh_hash"] != a.declaration_hash
        or d["fresh_hash"] != a.declaration_hash
        or d["cases"] != fd["cases"]
        or r["arm"] != d["arm"]
        or r["arm"] not in fd["arms"]
        or d["repeats"] != 2
        or d["skills"] != 3
        or d["record_initial_snapshot_fields"] != 10
        or r["all_whole_world_repeats_exact"] is not True
        or len(r["worlds"]) != 32
        or [w["case"] for w in r["worlds"]] != list(range(32))
        or any(
            obj[k] is not False
            for obj in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or r["Fresh_episodes_fit"] != 0
        or d["Fresh_episodes_fit"] != 0
    ):
        raise ValueError("exact all-context Fresh world protocol required")
    pins = dict(r["input_file_hashes"])
    for f in (
        rp,
        dp,
        a.declaration_root / "complete.json",
        a.declaration_root / "declaration.json",
    ):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=digest(rp),
            fresh_hash=a.declaration_hash,
            arm=r["arm"],
            own_neural_recompute=True,
            own_native_integration=True,
            first_kick_replayed=True,
            each_phase_starts_from_full_snapshot=True,
            independent_whole_world_replay=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    counts = dict(worlds=0, native=0, neural=0, own_neural=0)
    replay = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical, snapshot_fields
        from rosclaw_soccer.rsi.paid_native_phase_replay import PaidNativePhaseReplay
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_standing_lifecycle

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, fd["cases"])
        observer = StandingObservationAuditor(auditor)
        policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
        actor, variant = load_fresh_actor(policy, fd, r["arm"])
        if variant != r["variant"]:
            raise ValueError("wrong arm variant")
        replay = PaidNativePhaseReplay(auditor)
        visited = set()

        def read(row):
            f = a.root / row["raw_path"]
            if (
                f.is_symlink()
                or not f.resolve().is_relative_to(a.root.resolve())
                or digest(f) != row["raw_hash"]
            ):
                raise ValueError("exact confined Fresh raw required")
            visited.add(row["raw_path"])
            pins[str(f)] = row["raw_hash"]
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        results, phases = [], []
        for i, (world, case) in enumerate(zip(r["worlds"], fd["cases"], strict=True)):
            replicas, raws = [], []
            for repeat, artifacts in enumerate((world["artifacts"], world["repeat_artifacts"])):
                folder = a.root / f"case{i}-repeat{repeat}"
                wp = folder / "complete.json"
                actual = json.loads(wp.read_text())
                pins[str(wp)] = digest(wp)
                if actual["artifacts"] != artifacts or (
                    repeat == 0
                    and actual != {k: v for k, v in world.items() if k != "repeat_artifacts"}
                ):
                    raise ValueError("whole-world sealed record differs")
                links = {Path(row["raw_path"]).stem: row for row in artifacts}
                if len(links) != len(artifacts) or any(
                    (a.root / row["raw_path"]).parent != folder for row in artifacts
                ):
                    raise ValueError("all raw must belong to its own unique world")
                rows = [actual["initial"]] + [
                    row[k] for row in actual["kicks"] for k in ("result", "terminal")
                ]
                rows += [row["result"] for row in actual["stands"]]
                rows += [row for b in actual["boundaries"] for row in b.values() if row is not None]
                for row in rows:
                    link = links[Path(row["raw_path"]).stem]
                    if any(link[k] != row[k] for k in ("raw_path", "raw_hash")):
                        raise ValueError("phase hash binding differs")
                raw = {key: read(row) for key, row in links.items()}
                initial = raw["world-initial"]
                q, v = auditor.canonical(initial["qpos"], initial["qvel"])
                if (
                    not np.array_equal(q[36:39], case["ball_world_initial"])
                    or not np.array_equal(v[35:37], case["velocity_world_initial"])
                    or not np.array_equal(
                        initial["mocap_pos"][auditor.goal_mocap_id], case["goal_world"]
                    )
                ):
                    raise ValueError("Fresh initial ball/velocity/goal differs")
                outcome = audit_standing_lifecycle(
                    auditor,
                    observer,
                    actor,
                    actual,
                    raw,
                    motions[case["source"]],
                    raw["kick0"],
                    variant=variant,
                    skills=3,
                    count_inference=inference,
                )
                for key in ("native", "neural"):
                    counts[key] += outcome[key]
                counts["worlds"] += 1
                for key, z in raw.items():
                    if key.startswith("kick") and key[4:].isdigit():
                        k = int(key[4:])
                        entry = (
                            initial
                            if k == 0
                            else snapshot_fields(raw[f"between{k - 1}-{k}-boundary"], "after_")
                        )
                        terminal = raw[f"kick-terminal{k}"]
                    elif (key.startswith("stand") and key[5:].isdigit()) or key.endswith("-settle"):
                        entry, terminal = (
                            snapshot_fields(z, "initial_"),
                            snapshot_fields(z, "terminal_"),
                        )
                    else:
                        continue
                    certificate = replay.replay(z, entry=entry, terminal=terminal)
                    phases.append(
                        dict(
                            case=i,
                            repeat=repeat,
                            phase=key,
                            raw_hash=links[key]["raw_hash"],
                            **certificate,
                        )
                    )
                replicas.append(outcome)
                raws.append(raw)
            if replicas[0] != replicas[1] or set(raws[0]) != set(raws[1]):
                raise ValueError("whole-world repeated outcomes differ")
            for key in raws[0]:
                identical(raws[0][key], raws[1][key])
            results.append(
                dict(
                    case=i,
                    quality_by_kick=replicas[0]["quality_by_kick"],
                    stable=replicas[0]["stable"],
                    status=replicas[0]["status"],
                )
            )
            print(json.dumps(dict(case=i, status=replicas[0]["status"])), flush=True)
        summary = fresh_summary(results, fd["cases"])
        if (
            summary != r["summary"]
            or counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or replay.native_steps != counts["native"]
            or counts["worlds"] != 64
            or visited != {str(p.relative_to(a.root)) for p in a.root.rglob("*.npz")}
        ):
            raise ValueError("full32-case own-NN/native replay or raw accounting differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_FRESH32_OWN_NEURAL_AND_ALL_PHASES_NATIVE_REPLAY_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=digest(rp),
                fresh_hash=a.declaration_hash,
                arm=r["arm"],
                summary=summary,
                results=results,
                counts=counts,
                own_native_steps=replay.native_steps,
                own_trajectory_neural_rows=counts["own_neural"],
                own_kinematic_observations=observer.evaluations,
                phases=phases,
                first_kick_replayed=True,
                independent_whole_world_replay=False,
                Fresh_episodes_fit=0,
                optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                own_native_steps=replay.native_steps if replay else 0,
                hardware_authorized=False,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
