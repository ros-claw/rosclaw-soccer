import copy

import pytest
from test_refinement_growth_review import rows

from rosclaw_soccer.rsi.joint_delivery_teacher import (
    joint_delivery_rank,
    pilot_config,
    verify_pilot_search,
)


def row(error, speed, *, received=True):
    o = copy.deepcopy(rows()[0]["outcome"])
    o["B_legal_functional_receive_before_deadline"] = received
    o["incoming"]["B_first_ball_contact_substep"] = 20 if received else None
    o["incoming"]["incoming_target_error_3d_before_B_contact_m"] = error
    o["incoming_arrival_speed_mps"] = speed
    o["combined_precision_and_functional_receive"] = bool(
        error is not None and speed is not None and error <= 0.1 and speed <= 3 and received
    )
    return dict(outcome=o)


def test_joint_objective_cannot_be_hidden_by_controlled_receive_alone():
    precise_fast, slow_miss = row(0.0266, 3.59), row(0.4, 2.7)
    assert joint_delivery_rank(row(0.08, 2.9)) > joint_delivery_rank(precise_fast)
    assert joint_delivery_rank(row(0.09, 2.9, received=False)) > joint_delivery_rank(precise_fast)
    assert joint_delivery_rank(row(0.08, 2.9)) > joint_delivery_rank(row(0.08, 2.9, received=False))
    assert joint_delivery_rank(row(0.0266, 3.1)) > joint_delivery_rank(precise_fast)
    assert joint_delivery_rank(row(0.2, 2.7)) > joint_delivery_rank(slow_miss)


def test_safety_precedes_precision_and_unobserved_is_not_fabricated_success():
    a = row(0.01, 2)
    bad = copy.deepcopy(a)
    bad["outcome"]["passer_clean_foot"] = False
    assert joint_delivery_rank(a) > joint_delivery_rank(bad)
    assert joint_delivery_rank(row(None, None))[2:] == (0, -1e6)


@pytest.mark.parametrize("v", [float("nan"), float("inf"), -1, True])
def test_bad_observed_metrics_rejected(v):
    with pytest.raises(ValueError):
        joint_delivery_rank(row(v, 2))


def test_exact_bounded_two_pilot_budget_and_no_fresh_or_optimizer_claim():
    for state, seed in ((25, 115025), (26, 115023)):
        c = pilot_config(state)
        assert c["source_expert_seed"] == seed and c["maximum_worlds"] == 38
        assert c["maximum_native_steps"] == 152000
        assert c["Fresh_cases_read"] == c["new_optimizer_updates"] == 0
    with pytest.raises(ValueError):
        pilot_config(24)


def search_fixture():
    """Synthetic outcome stream to test accounting; NOT actual G1 physics."""
    import numpy as np

    from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites

    baseline = row(0.0266, 3.59)
    baseline["course"] = dict(
        state_id=26, task=0, assisted=False, ball_planar_offset_world=[0.0, 0.0]
    )
    course = baseline["course"]
    variants = [course | dict(ball_planar_offset_world=[offset, 0.0]) for offset in (-0.005, 0.005)]
    cfg = pilot_config(26)
    records, generations = [], []

    def take(tag, course, latent, o):
        r = dict(
            index=len(records),
            tag=tag,
            course=copy.deepcopy(course) | dict(assisted=True, latent=latent.tolist()),
            outcome=copy.deepcopy(o),
            neural_actor_optimizer_updates=0,
            prefix_identity_fields=1 if len(records) < 34 else None,
        )
        records.append(r)
        return dict(tag=tag, latent=latent, row=r)

    incumbent = dict(tag="inherited", latent=np.zeros((2, 6)), row=baseline)
    mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
    rng = np.random.default_rng(cfg["rng_seed"])
    for g in range(2):
        samples, draws = sample_generation(rng, mean, std)
        values = [incumbent] + [
            take(
                f"g{g}-c{k}",
                course,
                z,
                row(0.02 + abs(float(z[0, 0])) * 0.1, 3.2 + float(z[0, 1]) * 0.1)["outcome"],
            )
            for k, z in enumerate(samples)
        ]
        ranks = [joint_delivery_rank(v["row"]) for v in values]
        updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
        incumbent = values[max(range(17), key=lambda k: (ranks[k], -k))]
        generations.append(
            dict(
                generation=g,
                mean_before=mean.tolist(),
                std_before=std.tolist(),
                draws=draws,
                tags=[v["tag"] for v in values],
                ranks=[list(r) for r in ranks],
                elite=elite,
                mean_after=updated.tolist(),
                std_after=spread.tolist(),
                selected=incumbent["tag"],
            )
        )
        mean, std = updated, spread
    for r in range(2):
        take(f"selected-repeat{r}", course, incumbent["latent"], incumbent["row"]["outcome"])
    for index, c in enumerate(variants):
        for r in range(2):
            take(
                f"validation{index}-repeat{r}", c, incumbent["latent"], incumbent["row"]["outcome"]
            )
    return (
        dict(
            config=cfg,
            records=records,
            generations=generations,
            selected=dict(
                tag=incumbent["tag"],
                latent=incumbent["latent"].tolist(),
                rank=list(joint_delivery_rank(incumbent["row"])),
                outcome=incumbent["row"]["outcome"],
            ),
        ),
        baseline,
        variants,
    )


def test_full_bounded_search_accounting_is_reproducible():
    report, baseline, variants = search_fixture()
    verify_pilot_search(report, baseline, variants)


@pytest.mark.parametrize(
    "fault", ["draw", "rank", "winner", "drop", "repeat", "validation", "updates"]
)
def test_changed_search_or_omitted_failures_cannot_be_qualified(fault):
    report, baseline, variants = search_fixture()
    if fault == "draw":
        report["generations"][0]["draws"][0]["latent"][0][0] += 0.01
    elif fault == "rank":
        report["generations"][0]["ranks"][0][-1] += 0.01
    elif fault == "winner":
        report["selected"]["tag"] = "other"
    elif fault == "drop":
        report["records"].pop()
    elif fault == "repeat":
        report["records"][33]["outcome"]["incoming"][
            "incoming_target_error_3d_before_B_contact_m"
        ] += 0.01
    elif fault == "validation":
        report["records"][34]["course"]["ball_planar_offset_world"][0] += 0.01
    else:
        report["records"][0]["neural_actor_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        verify_pilot_search(report, baseline, variants)
