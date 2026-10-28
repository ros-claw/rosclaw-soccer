"""One-dimensional development-only physical-proxy trust update.

This uses a matched symmetric finite-difference experiment. It neither trains a
neural player nor establishes clean possession, transfer, or promotion.
"""

import numpy as np


def _proxy(measurement: dict) -> float:
    if type(measurement) is not dict or type(measurement.get("first_contact")) is not dict:
        raise ValueError("measured first contact and terminal state required")
    contact = measurement["first_contact"]
    groups = contact.get("first_contact_groups")
    if type(groups) is not list or any(
        x not in ("left_foot", "right_foot", "nonfoot") for x in groups
    ):
        raise ValueError("explicit native first-contact groups required")
    speed = measurement.get("final_relative_speed_mps")
    distance = measurement.get("final_root_ball_distance_m")
    if any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in (speed, distance)):
        raise ValueError("finite measured relative speed and distance required")
    contact_penalty = 3.0 if "nonfoot" in groups else 0.0 if groups else 2.0
    return float(contact_penalty + 0.25 * speed + distance)


def development_trust_update(positive: dict, negative: dict) -> dict:
    """Fit a quadratic to matched plus/zero/minus outcomes; no policy promotion."""
    if type(positive) is not dict or type(negative) is not dict:
        raise ValueError("two distinct matched readbacks required")

    def index(report: dict) -> dict:
        rows = report.get("rows")
        if type(rows) is not list or len(rows) != 8:
            raise ValueError("all eight development cases required")
        result = {r["index"]: r for r in rows}
        if len(result) != 8:
            raise ValueError("unique paired cases required")
        return result

    plus, minus = index(positive), index(negative)
    if set(plus) != set(minus):
        raise ValueError("same development conditions required")
    zero_cost, plus_cost, minus_cost = [], [], []
    for i in sorted(plus):
        a, b = plus[i], minus[i]
        if (
            a.get("baseline") != b.get("baseline")
            or (a.get("context"), a.get("side")) != (b.get("context"), b.get("side"))
            or any(
                r.get("common_controls") != 36
                or r.get("same_exposure") is not True
                or r.get("baseline_body_failure") is not False
                or r.get("candidate_body_failure") is not False
                or r.get("clean_reception_certified") is not False
                for r in (a, b)
            )
        ):
            raise ValueError("identical complete healthy paired 36-control histories required")
        zero_cost.append(_proxy(a["baseline"]))
        plus_cost.append(_proxy(a["candidate"]))
        minus_cost.append(_proxy(b["candidate"]))
    f0, fp, fm = (float(np.mean(v)) for v in (zero_cost, plus_cost, minus_cost))
    gradient = (fp - fm) / 0.2
    curvature = (fp + fm - 2 * f0) / 0.01
    if curvature > 1e-9:
        proposal = float(np.clip(-gradient / curvature, -0.1, 0.1))
        method = "bounded_quadratic"
    else:
        choices = ((f0, 0.0), (fp, 0.1), (fm, -0.1))
        proposal = min(choices)[1]
        method = "nonconvex_choose_observed"
    leg = [0.0, 0.0, 0.0, proposal, -0.5 * proposal, 0.0] * 2
    return dict(
        status="DEVELOPMENT_PROXY_TRUST_UPDATE_NOT_TRAINED_POLICY",
        baseline_proxy=f0,
        positive_proxy=fp,
        negative_proxy=fm,
        gradient_per_rad=gradient,
        curvature_per_rad2=curvature,
        knee_proposal_rad=proposal,
        leg_proposal_rad=leg,
        method=method,
        optimizer_updates=1,
        neural_weight_updates=0,
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
