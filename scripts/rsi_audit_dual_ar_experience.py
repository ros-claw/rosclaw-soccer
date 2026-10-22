"""Independently replay actual AR motor sampling AND every native control.

No actor updates/admission. Rejected proposals get an own-history query, but
are never executed or counted as stochastic physical actions.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("source-hash", "expected-head"):
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
            raise ValueError("frozen clean independent AR audit source required")

    clean()
    rp, dp = a.root / "complete.json", a.root / "declaration.json"
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (rp, dp))
        or digest(rp) != a.source_hash
        or (a.root / "failed.json").exists()
    ):
        raise ValueError("exact complete AR experience source required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    from rosclaw_soccer.rsi.motor_exploration_curriculum import declared_motor_curriculum

    curriculum = declared_motor_curriculum(d)
    if (
        r["status"] != "DUAL_AR1_ACTUAL_CALIBRATION_CLOSED_PENDING_NATIVE_AND_LAW_AUDIT"
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or d["rho"] != 0.9
        or d["sigma_policy_units"] != 0.005
        or len(r["records"]) != curriculum.worlds
        or d["online_teacher"] is not False
        or d["action_clipping"] is not False
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or r["actor_updates"] != 0
        or r["critic_updates"] != 0
        or r["actor_training_admitted"] is not False
    ):
        raise ValueError("fixed untrained pure-learner eight-world AR calibration required")
    pins = dict(r["input_file_hashes"])
    for x in (rp, dp):
        pins[str(x)] = digest(x)

    def receipt(h):
        paths = [Path(x) for x, v in pins.items() if v == h and x.endswith("/complete.json")]
        if len(paths) != 1:
            raise ValueError("unique exact motor lineage required")
        return paths[0], json.loads(paths[0].read_text())

    tp, training = receipt(d["training_hash"])
    pp, parent_record = receipt(training["parent_lineage_hash"])
    original_path, original = receipt(d["source_exam_hash"])
    original_decl = original_path.parent / "declaration.json"
    if digest(original_decl) != original["declaration_hash"]:
        raise ValueError("exact original examined TRAIN task commitment required")
    original_contexts = json.loads(original_decl.read_text())["contexts"]
    if d["contexts"] != [original_contexts[i] for i in curriculum.context_ids]:
        raise ValueError("actual curriculum fixtures cannot drift from examined task identities")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(native=0, neural=0, kinematic_intent=0)
    executed = 0
    verified = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp, sample_action
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw
        from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world
        from rosclaw_soccer.rsi.run_contact_geometry import summarize_contacts

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, [x["case"] for x in d["contexts"]])
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
        expected = [(c, s) for c in curriculum.context_ids for s in d["seeds"]]
        if [(x["context"], x["seed"]) for x in r["records"]] != expected:
            raise ValueError("complete fixed case/seed Cartesian product required")
        for rec in r["records"]:
            context, seed = rec["context"], rec["seed"]
            ctx = d["contexts"][curriculum.context_ids.index(context)]
            if rec["assisted"] is not False or rec["actor_training_admitted"] is not False:
                raise ValueError("no admitted or assisted source allowed")
            blobs = {}
            for name, art in rec["artifacts"].items():
                path = a.root / art["path"]
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(a.root.resolve())
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("exact confined actual physical primitive required")
                pins[str(path)] = art["hash"]
                with np.load(path, allow_pickle=False) as z:
                    blobs[name] = {k: z[k].copy() for k in z.files}
            ep = bench.make(ctx["case"])
            q, _ = ep.layout.canonical(ep.env.data)
            w, x, y, z = q[3:7]
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx["backoff_m"] * np.array(
                [np.cos(yaw), np.sin(yaw)]
            )
            mujoco.mj_forward(ep.env.model, ep.env.data)
            rng = torch.Generator(device="cpu").manual_seed(seed)
            history = []
            kick, stand = blobs.get("kick"), blobs.get("stand")

            def adapter(actual, nominal, evidence, *, history=history, rng=rng, kick=kick):
                ref = evidence["reference_nominal"][0, 0].numpy().copy()
                previous_action = previous_mean = None
                if history:
                    last_frame, previous_action, previous_mean = history[-1]
                    if actual.frame != last_frame + 1:
                        raise ValueError("sampling history advanced without physical control")
                proposal, logp = sample_action(
                    torch.from_numpy(nominal),
                    log_std,
                    rho=0.9,
                    previous_action=previous_action,
                    previous_mean=previous_mean,
                    generator=rng,
                )
                action = proposal.numpy().copy()
                equal(ref, kick["AR_reference_nominal"][actual.frame], "own AR reference")
                equal(logp.numpy(), kick["AR_logp"][actual.frame], "own sampled likelihood")
                if np.abs((action - ref) * scale).max() > 0.20002:
                    raise ValueError("executed source proposal exceeds fixed authority")
                history.append((actual.frame, proposal.clone(), torch.from_numpy(nominal.copy())))
                return action, 0

            events = []
            state = actor.initial_state()
            if kick is not None:
                state = replay_teacher_world(
                    ep,
                    actor,
                    kick,
                    stand,
                    latent=np.zeros((2, 6)),
                    center=105,
                    foot="left",
                    counts=counts,
                    hidden_width=256,
                    action_adapter=adapter,
                    contact_observer=events.append,
                    allow_incomplete_healthy_prefix=rec["rejection"] is not None,
                )
                executed += len(kick["qpos"])
                with torch.no_grad():
                    probabilities = conditional_logp(
                        torch.from_numpy(kick["NN_nominal_action"]),
                        torch.from_numpy(kick["NN_action"]),
                        log_std,
                        rho=0.9,
                    )
                np.testing.assert_allclose(
                    probabilities.numpy(), kick["AR_logp"], atol=1e-4, rtol=1e-6
                )
            if rec["rejection"] is not None:
                if stand is not None or (kick is not None and kick["body_failed"].any()):
                    raise ValueError(
                        "rejection cannot masquerade as physical failure or completed suffix"
                    )
                rej = rec["rejection"]
                path = a.root / f"context{context}-seed{seed}" / rej["path"]
                if (
                    rej["executed"] is not False
                    or path.is_symlink()
                    or not path.resolve().is_relative_to(a.root.resolve())
                    or digest(path) != rej["hash"]
                ):
                    raise ValueError("exact unexecuted rejection evidence required")
                pins[str(path)] = rej["hash"]
                with np.load(path, allow_pickle=False) as z:
                    rejected = {k: z[k].copy() for k in z.files}
                obs = ep.observation()
                equal(obs, rejected["observation"], "own rejected observation")
                for v, k in zip(state, ("h_before", "c_before"), strict=True):
                    equal(v.numpy(), rejected[k], "own rejection history")
                with torch.no_grad():
                    mean, _, e = actor.forward_with_evidence(
                        torch.from_numpy(obs).reshape(1, 1, 160), state
                    )
                    proposal, logp = sample_action(
                        mean[0, 0],
                        log_std,
                        rho=0.9,
                        previous_action=history[-1][1] if history else None,
                        previous_mean=history[-1][2] if history else None,
                        generator=rng,
                    )
                counts["neural"] += 1
                ref = e["reference_nominal"][0, 0].numpy().copy()
                for actual, key in (
                    (mean[0, 0].numpy(), "mean"),
                    (proposal.numpy(), "proposal"),
                    (ref, "reference_nominal"),
                    (logp.numpy(), "logp"),
                ):
                    equal(actual, rejected[key], "own unexecuted " + key)
                if (
                    np.abs((proposal.numpy() - ref) * scale).max() <= 0.20002
                    or rej["frame"] != ep.frame
                ):
                    raise ValueError("rejection must be real and at exact unexecuted boundary")
            ss = stand_from_raw(stand) if stand is not None else None
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
                    rec["rejection"] is None
                    and contact_quality(result)
                    and ss
                    and ss["healthy"]
                    and ss["stability_observed"]
                )
            if (
                result != rec["result"]
                or ss != rec["stand"]
                or good != rec["healthy_quality"]
                or rec["stochastic_executed"] != len(history)
            ):
                raise ValueError("physical outcome and actually executed sample count differs")
            path = a.output / f"context{context}-seed{seed}-contacts.json"
            write_once(path, dict(events=events, summary=summarize_contacts(events, ep.goal)))
            verified.append(
                dict(
                    context=context,
                    seed=seed,
                    quality=good,
                    rejected=rec["rejection"] is not None,
                    contact_path=path.name,
                    contact_hash=digest(path),
                )
            )
            print(
                json.dumps(dict(context=context, seed=seed, quality=good, native=counts["native"])),
                flush=True,
            )
        if (
            counts["native"] != r["actual_native_steps"]
            or counts["neural"] != r["actual_neural_decisions"]
            or executed != r["stochastic_executed"]
            or counts["kinematic_intent"] != 0
        ):
            raise ValueError("all actual physics/neural/sampling work must account exactly")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=curriculum.audit_status,
                source_head=a.expected_head,
                input_file_hashes=pins,
                source_hash=a.source_hash,
                training_hash=d["training_hash"],
                counts=counts,
                stochastic_executed=executed,
                results=verified,
                shared_sampling_and_observation_implementation=True,
                actor_training_admitted=False,
                actor_updates=0,
                critic_updates=0,
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
                counts=counts,
                worlds=len(verified),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
