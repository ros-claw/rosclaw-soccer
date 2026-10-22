"""SIM-only stroke search data utilities; physics imports occur after admission."""

import json
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest


def admit(exam_root, audit_root, audit_hash):
    ep, ap, dp = (
        exam_root / "complete.json",
        audit_root / "complete.json",
        exam_root / "declaration.json",
    )
    if any(p.is_symlink() for p in (ep, ap, dp)) or digest(ap) != audit_hash:
        raise ValueError("exact source exam and independent own-native witness required")
    e, a, d = [json.loads(p.read_text()) for p in (ep, ap, dp)]
    if (
        a["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
        or a["source_hash"] != digest(ep)
        or a["worlds"] != 44
        or e["declaration_hash"] != digest(dp)
        or d["checkpoint_kind"] != "ar1-actor-critic"
        or any(
            v["hardware_authorized"] is not False or v["promotion_authorized"] is not False
            for v in (e, a, d)
        )
    ):
        raise ValueError("fully audited SIM-only actual recurrent learner required")
    pins = dict(a["input_file_hashes"])
    for p in (ep, ap, dp):
        pins[str(p)] = digest(p)
    contexts = {i: d["contexts"][i] for i in (4, 20, 21)}
    centers = {}
    for i, ctx in contexts.items():
        if ctx["case"]["bank"] != "train" or ctx["case"]["not_fresh"] is not True:
            raise ValueError("only explicitly consumed TRAIN fixtures may be searched")
        r = next(r for r in e["records"] if r["context"] == i and r["repeat"] == 0)
        substep = r["result"]["postcontact"]["first_foot_contact_substep"]
        if (
            type(substep) is not int
            or not 98 <= substep // 10 <= 180
            or "left" not in ctx["case"]["source"]
        ):
            raise ValueError(
                "observed left-foot contact and preserved prefix planning domain required"
            )
        centers[i] = substep // 10
    return pins, d["training_hash"], contexts, centers, e


def open_motor(pins, training_hash, contexts):
    import torch

    from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
    from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
    from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def receipt(h):
        ps = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(ps) != 1:
            raise ValueError("unique exact model lineage required")
        return ps[0], json.loads(ps[0].read_text())

    tp, tr = receipt(training_hash)
    pp, parent = receipt(tr["parent_lineage_hash"])
    bench, policy, scene = open_paid_bench(pins, [c["case"] for c in contexts.values()])
    p, _ = load_paid_student(
        policy,
        checkpoint=Path(parent["checkpoint_path"]),
        checkpoint_hash=parent["checkpoint_hash"],
        training_receipt=pp,
        training_receipt_hash=tr["parent_lineage_hash"],
        scene_hash=scene,
    )
    actor, _ = load_dual_candidate_for_sim_audit(
        p,
        parent_lineage_hash=tr["parent_lineage_hash"],
        root=tp.parent,
        expected_receipt_hash=training_hash,
        checkpoint_kind="ar1-actor-critic",
    )
    return bench, actor


def make_episode(bench, ctx):
    import mujoco
    import numpy as np

    ep = bench.make(ctx["case"])
    q, _ = ep.layout.canonical(ep.env.data)
    w, x, y, z = q[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= ctx["backoff_m"] * np.array(
        [np.cos(yaw), np.sin(yaw)]
    )
    mujoco.mj_forward(ep.env.model, ep.env.data)
    return ep


def stroke_action(ep, mean, reference, center, latent):
    import numpy as np

    from rosclaw_soccer.rsi.contact_stroke_intent import contact_stroke_delta
    from rosclaw_soccer.rsi.own_state_motor_assistance import blend_actual_motor

    delta = contact_stroke_delta(foot="left", frame=ep.frame, center=center, latent=latent)
    if not delta.any():
        return mean.copy(), delta, False
    oracle = mean.copy()
    oracle[ep.order] = (oracle[ep.order] + delta / ep.scale).astype(np.float32)
    action, info = blend_actual_motor(
        mean, oracle, reference, ep.scale[np.argsort(ep.order)], share=1.0
    )
    changed = np.zeros(29, dtype=bool)
    changed[ep.order] = delta != 0
    action[~changed] = mean[~changed]
    return action, delta, info["oracle_authority_clipped"]


def score(blobs, goal, *, intended_foot=0):
    import numpy as np

    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality, stand_from_raw

    raw = {
        k: np.concatenate([z[k] for z in blobs.values()])
        for k in ("qpos", "qvel", "forces", "body_failed")
    }
    f = raw["forces"].reshape(-1, 3)
    p = native_contact_window(
        force=f[:, :2],
        canonical_qpos=raw["qpos"].reshape(-1, 43),
        canonical_qvel=raw["qvel"].reshape(-1, 41),
        goal=goal,
        intended_foot=intended_foot,
    )
    r = dict(
        body_failed=bool(raw["body_failed"].any()),
        forbidden=bool((f[:, 2] > 1).any()),
        nonfoot_impulse_Ns=float(f[:, 2].sum() * 0.002),
        postcontact=p,
    )
    ss = stand_from_raw(blobs["stand"]) if "stand" in blobs else None
    return r, ss, bool(contact_quality(r) and ss and ss["healthy"] and ss["stability_observed"])


def collect(
    bench,
    actor,
    ctx,
    center,
    latent,
    folder,
    *,
    baseline=None,
    shadow_student=None,
    student_prefix_controls=None,
    student_source=None,
):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.paid_neural_stand import NeuralStandingTask
    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

    folder.mkdir()
    ep = make_episode(bench, ctx)
    initial = snapshot(ep.env.data)
    private = actor.initial_state()
    pair = None
    if shadow_student is not None:
        from rosclaw_soccer.rsi.same_state_motor_pair import SameStateMotorPair

        if (
            student_prefix_controls not in (40, 80)
            or student_source is None
            or baseline is not None
        ):
            raise ValueError("explicit source-bound 40/80-control student prefix required")
        pair = SameStateMotorPair(shadow_student, actor)
        pair_state = pair.initial_state()
        private = pair_state[1]
    elif student_prefix_controls is not None or student_source is not None:
        raise ValueError("separate actual student required")
    rows, neural = [], 0
    while not ep.done and ep.frame < ep.clip.length:
        obs = ep.observation()
        before = [t.numpy().copy() for t in private]
        if pair is not None:
            student_before = [t.numpy().copy() for t in pair_state[0]]
            student_mean, m, pair_state, both = pair.observe(
                torch.from_numpy(obs).reshape(1, 1, 160), pair_state
            )
            private, evidence = pair_state[1], both["teacher"]
            neural += 2
        else:
            with torch.no_grad():
                m, private, evidence = actor.forward_with_evidence(
                    torch.from_numpy(obs).reshape(1, 1, 160), private
                )
            neural += 1
        mean, ref = m[0, 0].numpy().copy(), evidence["reference_nominal"][0, 0].numpy().copy()
        frame = ep.frame
        if baseline is not None and frame < 80:
            for actual, key in (
                (obs, "NN_obs"),
                (mean, "NN_action"),
                (before[0], "NN_h_before"),
                (before[1], "NN_c_before"),
            ):
                if not np.array_equal(actual, baseline[key][frame]):
                    raise ValueError("actual learner prefix diverged")
        action, delta, clipped = stroke_action(ep, mean, ref, center, latent)
        if pair is not None and frame < student_prefix_controls:
            action = student_mean[0, 0].numpy().copy()
            for value, key in (
                (obs, "NN_obs"),
                (action, "NN_action"),
                (student_before[0], "NN_h_before"),
                (student_before[1], "NN_c_before"),
            ):
                if not np.array_equal(value, student_source[key][frame]):
                    raise ValueError("actual private student prefix differs from source exam")
        row = ep.step(action)
        if (
            pair is not None
            and frame < student_prefix_controls
            and any(
                not np.array_equal(row[k], student_source[k][frame])
                for k in ("qpos", "qvel", "forces", "torque")
            )
        ):
            raise ValueError("actual student-owned physical prefix differs")
        if (
            baseline is not None
            and frame < 80
            and any(
                not np.array_equal(row[k], baseline[k][frame])
                for k in ("qpos", "qvel", "forces", "torque")
            )
        ):
            raise ValueError("actual physical prefix differs from certified source")
        row.update(
            NN_obs=obs,
            NN_nominal_action=mean,
            NN_action=action,
            NN_h_before=before[0],
            NN_c_before=before[1],
            NN_h_after=private[0].numpy().copy(),
            NN_c_after=private[1].numpy().copy(),
            NN_reference_nominal=ref,
            STROKE_delta_HG=delta,
            STROKE_authority_clipped=clipped,
        )
        if pair is not None:
            row.update(
                PAIR_student_mean=student_mean[0, 0].numpy().copy(),
                PAIR_student_h_before=student_before[0],
                PAIR_student_c_before=student_before[1],
                PAIR_student_h_after=pair_state[0][0].numpy().copy(),
                PAIR_student_c_after=pair_state[0][1].numpy().copy(),
                PAIR_student_executed=frame < student_prefix_controls,
            )
        rows.append(row)
        if ep.frame == ep.clip.length and not ep.done:
            ep.admit_reference_recovery()
    kick = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    kick.update({"initial_" + k: v for k, v in initial.items()})
    kick.update({"terminal_" + k: v for k, v in snapshot(ep.env.data).items()})
    blobs = dict(kick=kick)
    if ep.termination_reason == "RECOVERY_ADMITTED":

        def count():
            nonlocal neural
            neural += 1

        stand, _ = NeuralStandingTask(
            ep, actor, state=private, memory_mode="carry", count_inference=count, hidden_width=256
        ).collect(100)
        blobs["stand"] = stand
    result, ss, good = score(
        blobs, ep.goal, intended_foot=0 if "left" in ctx["case"]["source"] else 1
    )
    artifacts = {}
    for name, z in blobs.items():
        p = folder / (name + ".npz")
        np.savez_compressed(p, **z)
        artifacts[name] = dict(path=str(p.relative_to(folder.parent)), hash=digest(p))
    row = dict(
        context=ctx["context_id"],
        tag=folder.name,
        latent=latent.tolist(),
        center=center,
        result=result,
        stand=ss,
        healthy_quality=good,
        artifacts=artifacts,
        native_steps=sum(len(z["qpos"]) * 10 for z in blobs.values()),
        actor_decisions=neural,
        assisted=True,
        PPO_actor_rows=0,
    )
    if pair is not None:
        row.update(
            student_prefix_controls=student_prefix_controls,
            shadow_student_queries=len(kick["NN_obs"]),
        )
    return row, blobs
