"""Independent own-neural/native recurrence for one saved stroke world."""


def replay_stroke_world(
    bench,
    actor,
    ctx,
    center,
    latent,
    raw,
    counts,
    *,
    shadow_student=None,
    prefix_controls=None,
    student_source=None,
    stroke_foot="left",
):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world
    from scripts.rsi_stroke_search_common import make_episode, score, stroke_action

    ep = make_episode(bench, ctx)
    kick, stand = raw["kick"], raw.get("stand")
    events = []
    shadow_state = shadow_student.initial_state() if shadow_student is not None else None
    if shadow_student is not None and (prefix_controls not in (40, 80) or student_source is None):
        raise ValueError("explicit separately owned student prefix audit required")

    def adapter(actual, mean, evidence):
        nonlocal shadow_state
        ref = evidence["reference_nominal"][0, 0].numpy().copy()
        action, delta, clipped = stroke_action(actual, mean, ref, center, latent, foot=stroke_foot)
        if shadow_student is not None:
            observation = evidence["actual_observation"]
            equal(observation, kick["NN_obs"][actual.frame], "one actual causal observation")
            for t, k in zip(
                shadow_state, ("PAIR_student_h_before", "PAIR_student_c_before"), strict=True
            ):
                equal(t.numpy(), kick[k][actual.frame], "own shadow student before")
            with torch.no_grad():
                sm, shadow_state = shadow_student(
                    torch.from_numpy(observation).reshape(1, 1, 160), shadow_state
                )
            counts["neural"] += 1
            counts["shadow_neural"] += 1
            proposal = sm[0, 0].numpy().copy()
            equal(
                proposal, kick["PAIR_student_mean"][actual.frame], "own actual-stream student mean"
            )
            for t, k in zip(
                shadow_state, ("PAIR_student_h_after", "PAIR_student_c_after"), strict=True
            ):
                equal(t.numpy(), kick[k][actual.frame], "own shadow student after")
            prefix = actual.frame < prefix_controls
            equal(prefix, kick["PAIR_student_executed"][actual.frame], "own actor ownership")
            if prefix:
                equal(
                    observation,
                    student_source["NN_obs"][actual.frame],
                    "source student observation",
                )
                equal(proposal, student_source["NN_action"][actual.frame], "source student action")
                action = proposal
        equal(ref, kick["NN_reference_nominal"][actual.frame], "own frozen nominal")
        equal(delta, kick["STROKE_delta_HG"][actual.frame], "own coherent stroke")
        equal(clipped, kick["STROKE_authority_clipped"][actual.frame], "own authority processing")
        return action, 0

    replay_teacher_world(
        ep,
        actor,
        kick,
        stand,
        latent=np.zeros((2, 6)),
        center=center,
        foot="left" if "left" in ctx["case"]["source"] else "right",
        counts=counts,
        hidden_width=256,
        action_adapter=adapter,
        contact_observer=events.append,
    )
    result, ss, good = score(
        raw, ep.goal, intended_foot=0 if "left" in ctx["case"]["source"] else 1
    )
    return result, ss, good, events, ep.goal.copy()


def verify_matched_prefix(aa, bb):
    import numpy as np

    aa, bb = aa["kick"], bb["kick"]
    n = min(80, len(aa["qpos"]), len(bb["qpos"]))
    if n < 80 and len(aa["qpos"]) != len(bb["qpos"]):
        raise ValueError("prefix termination differs")
    for k in (
        "NN_obs",
        "NN_h_before",
        "NN_c_before",
        "NN_action",
        "qpos",
        "qvel",
        "forces",
        "torque",
    ):
        if not np.array_equal(aa[k][:n], bb[k][:n]):
            raise ValueError("paired counterfactual prefix differs")
