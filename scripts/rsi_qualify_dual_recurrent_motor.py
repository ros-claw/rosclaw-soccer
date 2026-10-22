"""Untrained dual-memory motor: exact full-run initial equivalence in native SIM.

New dual256 context is exercised through actual reference and active standing.
No learned-gain, old-bank retention, online-RL or promotion claim.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("audit-hash", "lineage-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("frozen clean dual-motor qualification source required")

    clean()
    rp, dp, ap = (
        a.root / "complete.json",
        a.root / "declaration.json",
        a.audit_root / "complete.json",
    )
    if a.output.exists() or any(x.is_symlink() for x in (rp, dp, ap)) or digest(ap) != a.audit_hash:
        raise ValueError("exact native source qualification and unused output required")
    r, d, ar = [json.loads(x.read_text()) for x in (rp, dp, ap)]
    if (
        ar["status"] != "INHERITED_BASE_RUN_WHOLE_NATIVE_OWN_NN_AUDIT_PASS"
        or ar["source_hash"] != digest(rp)
        or r["declaration_hash"] != digest(dp)
        or ar["own_native_steps"] != 33920
        or not ar["independent_whole_world_replay"]
        or len(r["records"]) != 8
        or any(ar[k] is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("exact full inherited-motor whole-world witness required")
    pins = dict(ar["input_file_hashes"])
    for x in (rp, dp, ap):
        pins[str(x)] = digest(x)
    paths = [
        Path(x) for x, h in pins.items() if h == a.lineage_hash and x.endswith("/complete.json")
    ]
    if len(paths) != 1:
        raise ValueError("canonical parent must be exact source-bound lineage")
    training = json.loads(paths[0].read_text())
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.initial_full_reference_stand_equivalence.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_hash=digest(rp),
            audit_hash=a.audit_hash,
            parent_lineage_hash=a.lineage_hash,
            contexts=2,
            repeats=2,
            hidden_width=256,
            maximum_native_steps=15560,
            maximum_actor_decisions=1556,
            expected_recurrent_branch_calls=3112,
            optimizer_updates=0,
            learned_gain_claim=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    decisions = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_teacher_audit import equal
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [d["case"]])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(training["checkpoint_path"]),
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=paths[0],
            training_receipt_hash=a.lineage_hash,
            scene_hash=scene,
        )
        model = DualRecurrentMotor(parent).eval().requires_grad_(False)
        reports = []
        sources = [x for x in r["records"] if x["arm"] == "reference_from_start"]
        if len(sources) != 4:
            raise ValueError("complete paired reference-from-start subset required")
        for source in sources:
            ci, repeat = source["course"], source["repeat"]
            folder = a.output / f"course{ci}-repeat{repeat}"
            folder.mkdir()
            blobs = {}
            for name, art in source["artifacts"].items():
                x = a.root / art["raw_path"]
                if (
                    x.is_symlink()
                    or not x.resolve().is_relative_to(a.root.resolve())
                    or digest(x) != art["raw_hash"]
                ):
                    raise ValueError("source artifact commitments changed")
                pins[str(x)] = art["raw_hash"]
                with np.load(x, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            ep = bench.make(d["case"])
            data = ep.env.data
            q, _ = ep.layout.canonical(data)
            w, x, y, z = q[3:7]
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            data.qpos[ep.layout.rq : ep.layout.rq + 2] -= d["courses"][ci]["backoff_m"] * np.array(
                [np.cos(yaw), np.sin(yaw)]
            )
            mujoco.mj_forward(ep.env.model, data)
            for k, v in snapshot(data).items():
                equal(v, blobs["initial"][k], "initial body " + k)
            state = model.initial_state()
            rows = []
            while not ep.done and ep.frame < ep.clip.length:
                frame = ep.frame
                obs = ep.observation()
                equal(obs, blobs["R2_applied"]["observation"][frame], "own actual input")
                for value, k in zip(state, ("h_before", "c_before"), strict=True):
                    for begin in (0, 128):
                        equal(
                            value.numpy()[..., begin : begin + 128],
                            blobs["R2_applied"][k][frame],
                            "initial private memory " + k,
                        )
                before = [t.numpy().copy() for t in state]
                with torch.no_grad():
                    mean, state = model(torch.from_numpy(obs).reshape(1, 1, 160), state)
                decisions += 1
                action = mean[0, 0].numpy().copy()
                equal(action, blobs["R2_applied"]["applied_action"][frame], "initial motor action")
                row = ep.step(action)
                row.update(
                    NN_obs=obs,
                    NN_action=action,
                    NN_h_before=before[0],
                    NN_c_before=before[1],
                    NN_h_after=state[0].numpy().copy(),
                    NN_c_after=state[1].numpy().copy(),
                )
                rows.append(row)
                if ep.frame == ep.clip.length and not ep.done:
                    ep.admit_reference_recovery()
            kick = {k: np.array([v[k] for v in rows]) for k in rows[0]}

            def count():
                nonlocal decisions
                decisions += 1

            stand, summary = NeuralStandingTask(
                ep, model, state=state, memory_mode="carry", count_inference=count, hidden_width=256
            ).collect(100)
            for k in ("qpos", "qvel", "forces", "torque", "body_failed"):
                equal(
                    np.concatenate((kick[k], stand[k])),
                    blobs["physical"][k],
                    "complete physical " + k,
                )
            equal(stand["NN_action"], blobs["stand"]["NN_action"], "standing actual actions")
            equal(stand["NN_obs"], blobs["stand"]["NN_obs"], "standing actual observations")
            if summary != source["stand"]:
                raise ValueError("initial recovery summary changed")
            artifacts = {}
            for name, raw in (("kick", kick), ("stand", stand)):
                x = folder / (name + ".npz")
                np.savez_compressed(x, **raw)
                artifacts[name] = dict(path=str(x.relative_to(a.output)), hash=digest(x))
            reports.append(
                dict(
                    course=ci,
                    repeat=repeat,
                    full_native_equality=True,
                    source_healthy_success=source["healthy_success"],
                    artifacts=artifacts,
                )
            )
            print(
                json.dumps(
                    dict(
                        course=ci,
                        repeat=repeat,
                        initial_equivalence=True,
                        native=bench.native_steps,
                    )
                ),
                flush=True,
            )
        if bench.native_steps != 15560 or decisions != 1556:
            raise ValueError("all four real full-run contexts must execute")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL256_INITIAL_NATIVE_REFERENCE_STANDING_EQUIVALENCE_PASS_NOT_TRAINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                source_hash=digest(rp),
                actual_native_steps=bench.native_steps,
                actual_actor_decisions=decisions,
                recurrent_branch_queries=2 * decisions,
                results=reports,
                optimizer_updates=0,
                learned_gain_claim=False,
                old_bank_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                actual_native_steps=bench.native_steps if bench else 0,
                actual_actor_decisions=decisions,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
