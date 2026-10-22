import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.stroke_search import (
    CONTEXTS,
    SEED,
    pair_rank,
    sample_generation,
    update_elites,
    verify_search_selection,
)


def fixture():
    records = []

    def row(tag, i, z):
        clean = bool(z[0, 0] > 0)
        r = dict(
            tag=tag,
            context=i,
            latent=z.tolist(),
            assisted=True,
            PPO_actor_rows=0,
            healthy_quality=clean,
            result=dict(
                body_failed=False,
                forbidden=not clean,
                nonfoot_impulse_Ns=0.0 if clean else 0.1,
                postcontact=dict(
                    correct_foot_first=True,
                    post_contact_quality_observed=True,
                    goal_progress_1p2s_m=2.0 + float(0.2 * np.tanh(z.sum())),
                    goal_lateral_1p2s_m=0.2,
                    maximum_pelvis_retreat_after_contact_m=0.0,
                ),
            ),
        )
        records.append(r)
        return r

    def pair(tag, z):
        rr = [row(f"{tag}-context{i}", i, z) for i in CONTEXTS]
        return dict(tag=tag, latent=z, rows=rr, rank=pair_rank(rr))

    inc = pair("baseline", np.zeros((2, 6)))
    rng = np.random.default_rng(SEED)
    mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
    generations = []
    for g in range(2):
        zs, draws = sample_generation(rng, mean, std)
        vals = [inc] + [pair(f"g{g}-candidate{k}", z) for k, z in enumerate(zs)]
        ranks = [v["rank"] for v in vals]
        mu, sigma, elite = update_elites([v["latent"] for v in vals], ranks)
        best = max(range(17), key=lambda k: (ranks[k], -k))
        generations.append(
            dict(
                generation=g,
                mean_before=mean.tolist(),
                std_before=std.tolist(),
                draws=draws,
                tags=[v["tag"] for v in vals],
                ranks=[list(v) for v in ranks],
                elite=elite,
                mean_after=mu.tolist(),
                std_after=sigma.tolist(),
                selected=vals[best]["tag"],
            )
        )
        inc = vals[best]
        mean, std = mu, sigma
    for i in CONTEXTS:
        for k in range(2):
            row(f"selected-context{i}-repeat{k}", i, inc["latent"])
    control = [row(f"control-context21-repeat{k}", 21, inc["latent"]) for k in range(2)]
    return dict(
        records=records,
        generations=generations,
        selected_tag=inc["tag"],
        selected_latent=inc["latent"].tolist(),
        selected_rank=list(inc["rank"]),
        selected_teacher_successes=sum(r["healthy_quality"] for r in inc["rows"]),
        unsearched_control_quality=control[0]["healthy_quality"],
    )


def test_all72_proposal_order_and_adaptive_cem_are_recomputed():
    r = fixture()
    assert len(r["records"]) == 72 and r["selected_tag"] != "baseline"
    verify_search_selection(r)


@pytest.mark.parametrize(
    "fault", ["missing", "latent", "draw", "elite", "context", "winner", "credit"]
)
def test_cannot_hide_bad_trial_or_change_selection(fault):
    r = copy.deepcopy(fixture())
    if fault == "missing":
        r["records"].pop()
    elif fault == "latent":
        r["records"][3]["latent"][0][0] += 0.001
    elif fault == "draw":
        r["generations"][0]["draws"][0]["latent"][0][0] += 0.001
    elif fault == "elite":
        r["generations"][0]["elite"] = [0, 1, 2, 3]
    elif fault == "context":
        r["records"][3]["context"] = 21
    elif fault == "winner":
        r["selected_tag"] = "baseline"
    else:
        r["records"][3]["PPO_actor_rows"] = 1
    with pytest.raises(ValueError):
        verify_search_selection(r)
