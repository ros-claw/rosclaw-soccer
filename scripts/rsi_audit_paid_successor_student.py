"""Independent all-world body observations, student NN, PD and retention audit."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen clean auditor required")
    deadline = time.monotonic() + 7200
    while not (a.root / "complete.json").exists():
        if (a.root / "failed.json").exists():
            raise ValueError("student physics failed; diagnose without automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("closed student worlds required; no physics retry")
        time.sleep(10)
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    r, d = [json.loads(path.read_text()) for path in (rp, dp)]
    if (
        (a.root / "failed.json").exists()
        or r["status"] != "PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["source_head"] != a.producer_head
        or d["source_head"] != a.producer_head
        or len(d["cases"]) != 16
        or d["repeats"] != 2
        or d["skills"] != 3
        or d["standing_controls"] != 100
        or d["body_resets"] != 0
        or r["bank"] != d["bank"]
        or r["all_first_and_full_repeat_arrays_exact"] is not True
        or any(
            obj[key] is not False
            for obj in (r, d)
            for key in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("exact all16 continuous neural-student lifecycle protocol required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    root, audit = Path(d["source_root"]), Path(d["source_audit_root"])
    standing_gate(root, audit, variant=1, bank=d["bank"])
    source = json.loads((root / "complete.json").read_text())
    student_root = Path(d["student_root"])
    training = json.loads((student_root / "complete.json").read_text())
    if digest(student_root / "complete.json") != d["student_hash"]:
        raise ValueError("exact admitted training receipt required")
    pins = dict(r["input_file_hashes"])
    pins.update({str(rp): digest(rp), str(dp): digest(dp)})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            own_native_steps=0,
            own_student_and_body_observation_recompute=True,
            hardware_authorized=False,
        ),
    )
    counts = dict(native=0, neural=0, own_neural=0, worlds=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_neural_stand_audit import StandingObservationAuditor
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_auditor
        from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_standing_lifecycle

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        auditor, motions = open_paid_auditor(pins, d["cases"])
        observer = StandingObservationAuditor(auditor)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        scene = next(
            h for path, h in pins.items() if path.endswith("/cached-standard/declared-scene.xml")
        )
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=student_root / "complete.json",
            training_receipt_hash=d["student_hash"],
            scene_hash=scene,
        )
        visited = set()

        def read(row, parent=a.root):
            path = parent / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(parent.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined neural/physical raw required")
            pins[str(path)] = row["raw_hash"]
            if parent == a.root:
                visited.add(row["raw_path"])
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def inference():
            counts["own_neural"] += 1

        results = []
        if len(r["worlds"]) != 16 or [w["case"] for w in r["worlds"]] != list(range(16)):
            raise ValueError("all16 conditions must be retained")
        for i, (world, case) in enumerate(zip(r["worlds"], d["cases"], strict=True)):
            first = read(source["worlds"][i]["kicks"][0]["result"], root)
            repetitions, raw_repetitions = [], []
            for repeat, artifacts in enumerate((world["artifacts"], world["repeat_artifacts"])):
                folder = a.root / f"case{i}-repeat{repeat}"
                wp = folder / "complete.json"
                actual = json.loads(wp.read_text())
                pins[str(wp)] = digest(wp)
                if actual["artifacts"] != artifacts or (
                    repeat == 0
                    and actual != {k: v for k, v in world.items() if k != "repeat_artifacts"}
                ):
                    raise ValueError("sealed whole-world record differs")
                links = {Path(row["raw_path"]).stem: row for row in artifacts}
                if len(links) != len(artifacts) or any(
                    (a.root / row["raw_path"]).parent != folder for row in artifacts
                ):
                    raise ValueError("unique phase artifacts in own complete world required")
                rows = [x[key] for x in actual["kicks"] for key in ("result", "terminal")]
                rows += [x["result"] for x in actual["stands"]]
                rows += [row for b in actual["boundaries"] for row in b.values() if row is not None]
                for row in rows:
                    link = links[Path(row["raw_path"]).stem]
                    if any(link[key] != row[key] for key in ("raw_path", "raw_hash")):
                        raise ValueError("phase row lost its hash binding")
                raw = {key: read(row) for key, row in links.items()}
                result = audit_standing_lifecycle(
                    auditor,
                    observer,
                    actor,
                    actual,
                    raw,
                    motions[case["source"]],
                    first,
                    variant=2,
                    skills=3,
                    count_inference=inference,
                )
                for key in ("native", "neural"):
                    counts[key] += result[key]
                counts["worlds"] += 1
                repetitions.append(result)
                raw_repetitions.append(raw)
            if repetitions[0] != repetitions[1] or set(raw_repetitions[0]) != set(
                raw_repetitions[1]
            ):
                raise ValueError("complete world repeats differ")
            for key in raw_repetitions[0]:
                identical(raw_repetitions[0][key], raw_repetitions[1][key])
            results.append(repetitions[0])
            print(json.dumps(dict(case=i, status=results[-1]["status"])), flush=True)
        summary = dict(
            stable=sum(w["stable"] for w in results),
            quality_by_kick=[
                sum(len(w["quality_by_kick"]) > k and w["quality_by_kick"][k] for w in results)
                for k in range(3)
            ],
        )
        if (
            summary != r["summary"]
            or counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_rows"]
            or counts["own_neural"] != counts["neural"]
            or visited != {str(p.relative_to(a.root)) for p in a.root.rglob("*.npz")}
        ):
            raise ValueError("complete student physics/neural accounting or raw coverage differs")
        retention = []
        for k in range(3):
            old = [
                len(w["quality_by_kick"]) > k and w["quality_by_kick"][k] for w in source["worlds"]
            ]
            new = [len(w["quality_by_kick"]) > k and w["quality_by_kick"][k] for w in results]
            retention.append(
                dict(
                    kick=k,
                    rescue=[i for i in range(16) if new[i] and not old[i]],
                    regression=[i for i in range(16) if old[i] and not new[i]],
                )
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_STUDENT_ALL_NATIVE_PRIMITIVES_AND_OWN_NEURAL_AUDIT_PASS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                summary=summary,
                retention=retention,
                counts=counts,
                own_native_steps=0,
                own_kinematic_observations=observer.evaluations,
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
                own_native_steps=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
