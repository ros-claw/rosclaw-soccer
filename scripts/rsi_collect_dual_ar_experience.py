"""Bounded pure-learner AR(1) exploration calibration, NOT yet PPO training data.

Rejected proposals are never clipped/executed/credited. An episode with such
a proposal is explicitly ineligible for actor learning. Independent native
replay and sampling-law audit must precede admission of any collected data.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generation-index", type=int, choices=range(100), default=0)
    for n in ("exam-root", "audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("audit-hash", "expected-head"):
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
            raise ValueError("clean frozen on-policy calibration source required")

    clean()
    rp, dp, ap = (
        a.exam_root / "complete.json",
        a.exam_root / "declaration.json",
        a.audit_root / "complete.json",
    )
    if a.output.exists() or any(x.is_symlink() for x in (rp, dp, ap)) or digest(ap) != a.audit_hash:
        raise ValueError("exact closed unassisted motor exam/audit required")
    r, d, audit = [json.loads(x.read_text()) for x in (rp, dp, ap)]
    if (
        audit["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or audit["source_hash"] != digest(rp)
        or audit["worlds"] != 44
        or audit["counts"]["native"] != r["actual_native_steps"]
        or r["declaration_hash"] != digest(dp)
        or audit["hardware_authorized"] is not False
        or audit["promotion_authorized"] is not False
    ):
        raise ValueError("complete actual own-history source required")
    pins = dict(audit["input_file_hashes"])
    for x in (rp, dp, ap):
        pins[str(x)] = digest(x)

    def receipt(h):
        paths = [Path(x) for x, v in pins.items() if v == h and x.endswith("/complete.json")]
        if len(paths) != 1:
            raise ValueError("unique exact neural checkpoint lineage required")
        return paths[0], json.loads(paths[0].read_text())

    tp, training = receipt(d["training_hash"])
    pp, parent_record = receipt(training["parent_lineage_hash"])
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    seeds = [710220 + 1000 * a.generation_index + i for i in range(4)]
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.actual_AR1_calibration.TRAIN2x4.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_exam_hash=digest(rp),
            source_audit_hash=a.audit_hash,
            training_hash=d["training_hash"],
            contexts=[d["contexts"][i] for i in (20, 21)],
            seeds=seeds,
            generation_index=a.generation_index,
            checkpoint_kind=d.get("checkpoint_kind", "supervised"),
            rho=0.9,
            sigma_policy_units=0.005,
            noisy_reference_only=True,
            standing_deterministic_neural_carry=True,
            maximum_worlds=8,
            maximum_native_steps=32000,
            maximum_neural_decisions=3200,
            online_teacher=False,
            action_clipping=False,
            actor_updates=0,
            critic_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    calls, executed = 0, 0
    records = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.correlated_motor_exploration import sample_action
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        contexts = [d["contexts"][i] for i in (20, 21)]
        bench, policy, scene = open_paid_bench(pins, [x["case"] for x in contexts])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=pp,
            training_receipt_hash=training["parent_lineage_hash"],
            scene_hash=scene,
        )
        actor, _ = load_dual_candidate_for_sim_audit(
            parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=tp.parent,
            expected_receipt_hash=d["training_hash"],
            checkpoint_kind=d.get("checkpoint_kind", "supervised"),
        )
        scale = actor.parent.get_buffer("action_scale").numpy().copy()
        log_std = torch.full((29,), math.log(0.005), dtype=torch.float32)
        for context, ctx in zip((20, 21), contexts, strict=True):
            for seed in seeds:
                folder = a.output / f"context{context}-seed{seed}"
                folder.mkdir()
                ep = bench.make(ctx["case"])
                q, _ = ep.layout.canonical(ep.env.data)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx["backoff_m"] * np.array(
                    [np.cos(yaw), np.sin(yaw)]
                )
                mujoco.mj_forward(ep.env.model, ep.env.data)
                initial = snapshot(ep.env.data)
                state = actor.initial_state()
                rng = torch.Generator(device="cpu").manual_seed(seed)
                previous_action = previous_mean = None
                rows, rejection = [], None
                while not ep.done and ep.frame < ep.clip.length:
                    obs = ep.observation()
                    before = [t.numpy().copy() for t in state]
                    with torch.no_grad():
                        mean, after, e = actor.forward_with_evidence(
                            torch.from_numpy(obs).reshape(1, 1, 160), state
                        )
                        proposal, logp = sample_action(
                            mean[0, 0],
                            log_std,
                            rho=0.9,
                            previous_action=previous_action,
                            previous_mean=previous_mean,
                            generator=rng,
                        )
                    calls += 1
                    action = proposal.numpy().copy()
                    nominal = mean[0, 0].numpy().copy()
                    ref = e["reference_nominal"][0, 0].numpy().copy()
                    if np.abs((action - ref) * scale).max() > 0.20002:
                        rejectpath = folder / "unexecuted-proposal.npz"
                        np.savez_compressed(
                            rejectpath,
                            observation=obs,
                            proposal=action,
                            mean=nominal,
                            reference_nominal=ref,
                            logp=logp.numpy(),
                            h_before=before[0],
                            c_before=before[1],
                        )
                        rejection = dict(
                            path=rejectpath.name,
                            hash=digest(rejectpath),
                            executed=False,
                            frame=ep.frame,
                        )
                        break
                    row = ep.step(action)
                    state = after
                    executed += 1
                    # Commit stochastic history ONLY after actual physical control.
                    previous_action, previous_mean = proposal.clone(), mean[0, 0].clone()
                    row.update(
                        NN_obs=obs,
                        NN_nominal_action=nominal,
                        NN_action=action,
                        NN_h_before=before[0],
                        NN_c_before=before[1],
                        NN_h_after=state[0].numpy().copy(),
                        NN_c_after=state[1].numpy().copy(),
                        AR_logp=logp.numpy().copy(),
                        AR_reference_nominal=ref,
                    )
                    rows.append(row)
                    if ep.frame == ep.clip.length and not ep.done:
                        ep.admit_reference_recovery()
                kick = {k: np.array([row[k] for row in rows]) for k in rows[0]} if rows else None
                if kick is not None:
                    kick.update({"initial_" + k: v for k, v in initial.items()})
                    kick.update({"terminal_" + k: v for k, v in snapshot(ep.env.data).items()})
                stand = summary = None
                if rejection is None and ep.termination_reason == "RECOVERY_ADMITTED":

                    def count():
                        nonlocal calls
                        calls += 1

                    stand, summary = NeuralStandingTask(
                        ep,
                        actor,
                        state=state,
                        memory_mode="carry",
                        count_inference=count,
                        hidden_width=256,
                    ).collect(100)
                artifacts = {}
                for name, raw in (("kick", kick), ("stand", stand)):
                    if raw is not None:
                        path = folder / (name + ".npz")
                        np.savez_compressed(path, **raw)
                        artifacts[name] = dict(
                            path=str(path.relative_to(a.output)), hash=digest(path)
                        )
                result = None
                good = False
                if kick is not None:
                    raw = {
                        k: np.concatenate([kick[k], stand[k]]) if stand is not None else kick[k]
                        for k in ("qpos", "qvel", "forces", "body_failed")
                    }
                    f = raw["forces"].reshape(-1, 3)
                    post = native_contact_window(
                        force=f[:, :2],
                        canonical_qpos=raw["qpos"].reshape(-1, 43),
                        canonical_qvel=raw["qvel"].reshape(-1, 41),
                        goal=ep.goal,
                        intended_foot=0,
                    )
                    result = dict(
                        body_failed=bool(raw["body_failed"].any()),
                        forbidden=bool((f[:, 2] > 1).any()),
                        nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
                        postcontact=post,
                    )
                    good = bool(
                        rejection is None
                        and contact_quality(result)
                        and summary
                        and summary["healthy"]
                        and summary["stability_observed"]
                    )
                row = dict(
                    context=context,
                    seed=seed,
                    result=result,
                    stand=summary,
                    healthy_quality=good,
                    rejection=rejection,
                    stochastic_executed=len(rows),
                    assisted=False,
                    actor_training_admitted=False,
                    artifacts=artifacts,
                )
                write_once(folder / "complete.json", row)
                records.append(row)
                print(
                    json.dumps(
                        dict(
                            context=context,
                            seed=seed,
                            quality=good,
                            rejected=rejection is not None,
                            native=bench.native_steps,
                            neural=calls,
                        )
                    ),
                    flush=True,
                )
                if bench.native_steps > 32000 or calls > 3200:
                    raise ValueError("bounded on-policy calibration budget exceeded")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_AR1_ACTUAL_CALIBRATION_CLOSED_PENDING_NATIVE_AND_LAW_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                training_hash=d["training_hash"],
                records=records,
                actual_native_steps=bench.native_steps,
                actual_neural_decisions=calls,
                stochastic_executed=executed,
                actor_updates=0,
                critic_updates=0,
                actor_training_admitted=False,
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
                actual_neural_decisions=calls,
                stochastic_executed=executed,
                worlds=len(records),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
