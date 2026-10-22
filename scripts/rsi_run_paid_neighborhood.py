"""One fixed32-context learned/control/teacher perturbation arm; no search or SGD."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-root", type=Path, required=True)
    p.add_argument("--plan-hash", required=True)
    p.add_argument(
        "--arm",
        choices=("frozen_parent", "learned_memory", "fixed_successful_teacher"),
        required=True,
    )
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen perturbation runner required")

    clean()
    if a.output.exists():
        raise ValueError("unused perturbation arm required, no automatic retry")
    from rosclaw_soccer.rsi.paid_neighborhood_sources import load_neighborhood_plan

    _, plan, pins = load_neighborhood_plan(a.plan_root, a.plan_hash)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    clean()
    a.output.mkdir()
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_own_state_teacher import OwnStateTeacherHandoff
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, plan["cases"])
        if scene != plan["body_scene_hash"] or bench.native_steps or bench.neural_rows:
            raise ValueError("all declared cases must preflight before rollout")
        spec = plan["specifications"]["models"][a.arm]
        rp = Path(spec["root"]) / "complete.json"
        r = json.loads(rp.read_text())
        actor, _ = load_paid_student(
            policy,
            checkpoint=Path(r["checkpoint_path"]),
            checkpoint_hash=r["checkpoint_hash"],
            training_receipt=rp,
            training_receipt_hash=spec["hash"],
            scene_hash=scene,
        )
        common = dict(
            source_head=a.expected_head,
            arm=a.arm,
            plan_root=str(a.plan_root),
            plan_hash=a.plan_hash,
            config=plan["config"],
            model_hash=spec["hash"],
            body_scene_hash=scene,
            input_file_hashes=pins,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(
            a.output / "declaration.json",
            dict(**common, base_stand=True, teacher_reoptimized=False),
        )
        records = []
        for case in plan["cases"]:
            first_blobs = None
            for repeat in range(2):
                folder = a.output / f"case{case['case']}-repeat{repeat}"
                folder.mkdir()
                blobs = {}

                def store(key, z, folder=folder, blobs=blobs):
                    path = folder / (key + ".npz")
                    if path.exists():
                        raise ValueError("raw overwrite forbidden")
                    np.savez_compressed(path, **z)
                    blobs[key] = z
                    return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

                ep = bench.make(case)
                initial = snapshot(ep.env.data)
                with np.load(
                    a.plan_root / f"case{case['case']}-initial.npz", allow_pickle=False
                ) as z:
                    identical(initial, {k: z[k].copy() for k in z.files})
                entry = store("initial", initial)
                teacher = None
                if a.arm == "fixed_successful_teacher":
                    target = plan["specifications"]["teachers"][str(case["source_case"])]
                    intent = FootTaskSpaceIntent(
                        ep.env.model,
                        foot="left" if "left" in case["source"] else "right",
                        center=target["center"],
                        latent=np.array(target["latent"], np.float64),
                    )
                    teacher = OwnStateTeacherHandoff(
                        ep, actor.base, intent=intent, switch_control=target["switch_control"]
                    )
                raw, result = bench.collect_current(
                    ep, actor, causal_transform=teacher, stop_at_reference=True
                )
                row = dict(
                    case=case["case"],
                    source_case=case["source_case"],
                    repeat=repeat,
                    initial=entry,
                    kick=dict(result, **store("kick", raw)),
                    kick_terminal=store("kick-terminal", snapshot(ep.env.data)),
                    stand=None,
                )
                if not result["body_failed"]:

                    def inference():
                        with bench.lock:
                            bench.neural_rows += 1

                    task = NeuralStandingTask(
                        ep,
                        actor.base,
                        state=tuple(
                            torch.from_numpy(raw[k][-1].copy())
                            for k in ("NN_h_after", "NN_c_after")
                        ),
                        memory_mode="carry",
                        count_inference=inference,
                    )
                    sr, ss = task.collect(100)
                    row["stand"] = dict(ss, **store("stand", sr))
                row["quality"] = bool(contact_rank([result, result])[1])
                row["healthy_success"] = bool(
                    row["quality"]
                    and row["stand"]
                    and row["stand"]["healthy"]
                    and row["stand"]["stability_observed"]
                )
                if first_blobs is not None:
                    if set(first_blobs) != set(blobs):
                        raise ValueError("whole repeat phase mismatch")
                    for k in blobs:
                        identical(first_blobs[k], blobs[k])
                first_blobs = blobs
                records.append(row)
                if bench.native_steps > plan["config"]["maximum_native_steps"] // 3:
                    raise ValueError("finite per-arm native budget exceeded")
            print(
                json.dumps(
                    dict(
                        case=case["case"],
                        arm=a.arm,
                        quality=row["quality"],
                        healthy_success=row["healthy_success"],
                    )
                ),
                flush=True,
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTACT_NEIGHBORHOOD32_ARM_CLOSED_PENDING_AUDIT",
                **common,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                episodes=len(records),
                repeats_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                Fresh_rows_read=0,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                actual_native_steps=bench.native_steps if bench else 0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
