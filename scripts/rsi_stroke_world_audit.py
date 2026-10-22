"""Independent own-neural/native recurrence for one saved stroke world."""


def replay_stroke_world(bench, actor, ctx, center, latent, raw, counts):
    import numpy as np

    from rosclaw_soccer.rsi.paid_run_teacher_audit import equal, replay_teacher_world
    from scripts.rsi_stroke_search_common import make_episode, score, stroke_action

    ep = make_episode(bench, ctx)
    kick, stand = raw["kick"], raw.get("stand")
    events = []

    def adapter(actual, mean, evidence):
        ref = evidence["reference_nominal"][0, 0].numpy().copy()
        action, delta, clipped = stroke_action(actual, mean, ref, center, latent)
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
        foot="left",
        counts=counts,
        hidden_width=256,
        action_adapter=adapter,
        contact_observer=events.append,
    )
    result, ss, good = score(raw, ep.goal)
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
