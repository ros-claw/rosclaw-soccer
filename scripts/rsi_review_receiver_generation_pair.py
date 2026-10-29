"""Replay-bound parent/child soccer and actual teammate contact observations."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_receiver_match_prepared import physical_outcome
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def observed_body_margins(root, row, horizon, a_layout, b_layout):
    """Measured threshold margins, not predictions or physical safety proof."""
    path = bind(root, row, "physics")
    with np.load(path, allow_pickle=False) as z:
        positions = z["qpos"][:horizon]
    if positions.ndim != 3 or len(positions) != horizon or not np.isfinite(positions).all():
        raise ValueError("complete finite common-horizon physics required")
    result = {}
    for name, layout in (("a", a_layout), ("b", b_layout)):
        root_pose = positions[:, :, layout.rq : layout.rq + 7].reshape(-1, 7)
        height = root_pose[:, 2]
        upright = 1 - 2 * (root_pose[:, 4] ** 2 + root_pose[:, 5] ** 2)
        bad = np.flatnonzero((height < 0.65) | (upright < 0.9))
        result[name] = dict(
            minimum_root_height_m=float(height.min()),
            minimum_height_margin_m=float(height.min() - 0.65),
            minimum_upright_cosine=float(upright.min()),
            minimum_upright_margin=float(upright.min() - 0.9),
            first_failed_substep=int(bad[0]) if len(bad) else None,
        )
    return result


def team_followup(root, row, horizon, model, a_layout):
    """Observe A after B's real directed strike, within the specified exposure."""
    handoff = row["handoff"]
    if handoff is None or handoff["control"] >= horizon:
        return None
    path = bind(root, row, "physics")
    with np.load(path, allow_pickle=False) as z:
        q = z["qpos"][:horizon].reshape(-1, model.nq)
        v = z["qvel"][:horizon].reshape(-1, model.nv)
        force = z["ball_forces"][:horizon].reshape(-1, 2, 3)
    direction = np.asarray(handoff["return_goal"][:2]) - np.asarray(handoff["kick_origin"][:2])
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("finite real return direction required")
    first = handoff["control"] * 10
    candidates = np.flatnonzero(
        (force[first:, 1, :2].max(axis=1) > 10)
        & (v[first:, a_layout.bv : a_layout.bv + 2] @ (direction / norm) >= 1)
    )
    if not len(candidates):
        return dict(directed_b_strike=False, teammate_possession_proven=False)
    strike = first + int(candidates[0])
    a_foot = np.flatnonzero((force[strike:, 0, :2] > 10).any(axis=1))
    a_other = np.flatnonzero(force[strike:, 0, 2] > 1)
    ball = q[strike:, a_layout.bq : a_layout.bq + 2]
    teammate = q[strike:, a_layout.rq : a_layout.rq + 2]
    return dict(
        directed_b_strike=True,
        b_strike_substep=strike,
        a_first_foot_substep=strike + int(a_foot[0]) if len(a_foot) else None,
        a_first_nonfoot_substep=strike + int(a_other[0]) if len(a_other) else None,
        min_ball_a_root_planar_distance_m=float(np.linalg.norm(ball - teammate, axis=1).min()),
        teammate_possession_proven=False,
        nonfoot_is_not_legality=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--prior-audit", type=Path, required=True)
    p.add_argument("--prior-audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source, _, sh = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_CHILD_VELOCITY_GENERATION_PAIR16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    prior, _, ph = report(
        a.prior,
        a.prior_audit,
        a.prior_audit_hash,
        "NEURAL_RELAY_CHILD_VELOCITY_CANDIDATE_FIXED16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    if scene.is_symlink() or digest(a.prior / "scene.xml") != scene_hash:
        raise ValueError("identical qualified scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    la = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    lb = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows, candidate_calls = [], 0
    for i in range(0, 16, 2):
        parent, child, expected = (
            source["records"][i],
            source["records"][i + 1],
            prior["records"][i + 1],
        )
        if (
            parent["course"]["context"] != child["course"]["context"]
            or parent["course"]["lateral_m"] != child["course"]["lateral_m"]
            or parent["course"]["arm"]["receiver_child_candidate"] is not False
            or child["course"]["arm"]["receiver_child_candidate"] is not True
            or parent["artifacts"]["initial"]["hash"] != child["artifacts"]["initial"]["hash"]
            or parent["artifacts"]["physics"]["hash"] != expected["artifacts"]["physics"]["hash"]
        ):
            raise ValueError("same initial conditions and replayed parent physics required")
        with (
            np.load(bind(a.source, parent, "receiver_velocity"), allow_pickle=False) as x,
            np.load(bind(a.source, child, "receiver_velocity"), allow_pickle=False) as y,
        ):
            if x["candidate_actor_selected"].any() or not y["candidate_actor_selected"].all():
                raise ValueError("actual paired actor selection required")
            candidate_calls += len(y["candidate_actor_selected"])
            n = min(len(x["action"]), len(y["action"]))
            delta = np.max(np.abs(x["action"][:n] - y["action"][:n]), axis=1)
            found = np.flatnonzero(delta != 0)
            first = int(found[0]) if len(found) else None
            if first is not None and any(
                not np.array_equal(x[k][first], y[k][first])
                for k in ("obs", "context14", "reference_action")
            ):
                raise ValueError("first different action must use identical inputs")
            amount = float(delta[first]) if first is not None else 0.0
        common = min(parent["controls"], child["controls"])
        rows.append(
            dict(
                context=parent["course"]["context"],
                side=parent["course"]["lateral_m"],
                first_different_action=first,
                same_input_difference_policy_units=amount,
                parent_controls=parent["controls"],
                child_controls=child["controls"],
                common_controls=common,
                parent_termination=parent["termination"],
                child_termination=child["termination"],
                parent_full=physical_outcome(a.source, parent, parent["controls"], model, lb),
                child_full=physical_outcome(a.source, child, child["controls"], model, lb),
                parent_common=physical_outcome(a.source, parent, common, model, lb),
                child_common=physical_outcome(a.source, child, common, model, lb),
                parent_body_margins_common=observed_body_margins(a.source, parent, common, la, lb),
                child_body_margins_common=observed_body_margins(a.source, child, common, la, lb),
                parent_team_common=team_followup(a.source, parent, common, model, la),
                child_team_common=team_followup(a.source, child, common, model, la),
                parent_result=parent["result"],
                child_result=child["result"],
            )
        )
    if candidate_calls == 0 or candidate_calls != source["counts"]["receiver_candidate_calls"]:
        raise ValueError("all actual candidate actor calls must be accounted for")
    if digest(scene) != scene_hash:
        raise ValueError("scene changed during review")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="REPLAYED_GENERATION_PAIR_DEVELOPMENT_NOT_POSSESSION",
            source_hash=sh,
            audit_hash=a.audit_hash,
            prior_hash=ph,
            prior_audit_hash=a.prior_audit_hash,
            scene_hash=scene_hash,
            code_head=a.expected_head,
            rows=rows,
            candidate_calls=candidate_calls,
            physical_retention_proven=False,
            team_pass_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
