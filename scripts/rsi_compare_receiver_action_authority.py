"""Compare executed targets, joint response and ball motion in paired SIM worlds."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiver_motion_effect import paired_receiver_motion_effect
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, measured, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        nargs=3,
        action="append",
        required=True,
        metavar=("CANDIDATE", "PARENT", "COMPARISON"),
    )
    for name in ("candidate-head", "parent-head", "first-parent-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if len(a.group) != 3 or a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError(
            "three matched audited development blocks and unused external output required"
        )
    rows, blocks, frozen = [], set(), []
    for candidate_name, parent_name, comparison_name in a.group:
        candidate, parent, comparison = map(Path, (candidate_name, parent_name, comparison_name))
        if comparison.is_symlink():
            raise ValueError("owned comparison required")
        comparison_hash = digest(comparison)
        evidence = json.loads(comparison.read_text())
        expected = {(r["context"], r["side"]): r for r in evidence["records"]}
        if (
            evidence.get("status") != "RECEIVER_CONTINUATION_MATCHED_DEVELOPMENT_COMPARISON"
            or len(expected) != 8
            or any(
                evidence.get(k) is not False
                for k in ("promotion_authorized", "hardware_authorized", "goal_complete")
            )
        ):
            raise ValueError("complete replayed matched comparison required")
        datasets = []
        for root, status, head in (
            (
                candidate,
                "NEURAL_RELAY_RECEIVER_CONTINUATION_PAIR16_PENDING_OWN_REPLAY",
                a.candidate_head,
            ),
            (parent, "NEURAL_RELAY_RECEIVER_DENSE_PAIR16_PENDING_OWN_REPLAY", None),
        ):
            source, audit = root / "source", root / "audit/complete.json"
            ah = digest(audit)
            data, _, sh = report(source, audit, ah, status, worlds=16)
            dec = json.loads((source / "declaration.json").read_text())
            block = dec["development_context_block"]
            expected_head = (
                head if head is not None else a.first_parent_head if block == 0 else a.parent_head
            )
            if data["source_head"] != expected_head or block not in (0, 1, 2):
                raise ValueError("exact completed development producer required")
            entries = {
                (r["course"]["context"], r["course"]["lateral_m"]): r
                for r in data["records"]
                if r["course"]["arm"]["receiver_learned"]
            }
            if len(entries) != 8:
                raise ValueError("eight actual candidate histories required")
            datasets.append((source, entries, block))
            frozen.extend(((source / "complete.json", sh), (audit, ah)))
        (cr, ci, cb), (pr, pi, pb) = datasets
        if cb != pb or cb in blocks or set(ci) != set(pi) or set(ci) != set(expected):
            raise ValueError("unique matched complete context blocks required")
        blocks.add(cb)
        cm, pm = (
            mujoco.MjModel.from_xml_path(str(cr / "scene.xml")),
            mujoco.MjModel.from_xml_path(str(pr / "scene.xml")),
        )
        cl = NativeVelocityLayout(
            cm, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        pl = NativeVelocityLayout(
            pm, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        for condition in sorted(ci):
            ct, pt, original = ci[condition], pi[condition], expected[condition]
            if (
                measured(cr, ct) != original["continuation"]["original"]
                or measured(pr, pt) != original["parent"]["original"]
            ):
                raise ValueError("full physical outcome must match prior comparison")
            cv, pv, cp, pp = (
                bind(cr, ct, "receiver_velocity"),
                bind(pr, pt, "receiver_velocity"),
                bind(cr, ct, "physics"),
                bind(pr, pt, "physics"),
            )
            if any(path is None for path in (cv, pv, cp, pp)):
                raise ValueError("actual own actor and physics histories required")
            with (
                np.load(cv, allow_pickle=False) as ca,
                np.load(pv, allow_pickle=False) as pa,
                np.load(cp, allow_pickle=False) as c,
                np.load(pp, allow_pickle=False) as old,
            ):
                action_effect = paired_receiver_motion_effect(dict(pa), dict(ca))
                n = min(len(ca["control"]), len(pa["control"]), len(c["qpos"]), len(old["qpos"]))
                if n < 1:
                    raise ValueError("nonempty synchronized actual trajectories required")
                target_delta = c["target_b"][:n] - old["target_b"][:n]
                joint_delta = c["qpos"][:n, -1, cl.qi] - old["qpos"][:n, -1, pl.qi]
                root_delta = (
                    c["qpos"][:n, -1, cl.rq : cl.rq + 3] - old["qpos"][:n, -1, pl.rq : pl.rq + 3]
                )
                ball_delta = (
                    c["qpos"][:n, -1, cl.bq : cl.bq + 3] - old["qpos"][:n, -1, pl.bq : pl.bq + 3]
                )
                torque_delta = c["torque"][:n, :, cl.ai] - old["torque"][:n, :, pl.ai]
                if any(
                    not np.isfinite(v).all()
                    for v in (target_delta, joint_delta, root_delta, ball_delta, torque_delta)
                ):
                    raise ValueError("finite actual motor and physical response required")
                rows.append(
                    dict(
                        block=cb,
                        context=condition[0],
                        side=condition[1],
                        controls=n,
                        max_target_difference_rad=float(np.abs(target_delta).max()),
                        rms_target_difference_rad=float(np.sqrt(np.mean(target_delta**2))),
                        max_joint_response_difference_rad=float(np.abs(joint_delta).max()),
                        max_torque_difference_nm=float(np.abs(torque_delta).max()),
                        max_root_difference_m=float(np.linalg.norm(root_delta, axis=1).max()),
                        max_ball_difference_m=float(np.linalg.norm(ball_delta, axis=1).max()),
                        actual_own_actor_effect=action_effect,
                        parent=original["parent"],
                        candidate=original["continuation"],
                    )
                )
        frozen.append((comparison, comparison_hash))
    if blocks != {0, 1, 2} or len(rows) != 24 or any(digest(path) != h for path, h in frozen):
        raise ValueError("all unchanged matched physical sources required")
    metrics = (
        "max_target_difference_rad",
        "rms_target_difference_rad",
        "max_joint_response_difference_rad",
        "max_torque_difference_nm",
        "max_root_difference_m",
        "max_ball_difference_m",
    )
    summary = {
        key: dict(
            median=float(np.median([r[key] for r in rows])),
            maximum=float(np.max([r[key] for r in rows])),
        )
        for key in metrics
    }
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_CONTINUATION_ACTION_AUTHORITY_DIAGNOSED",
            source_head=a.expected_head,
            conditions=24,
            summary=summary,
            records=rows,
            same_start_but_divergent_closed_loop_states=True,
            fixed_state_causal_effect_proven=False,
            success_or_generalization_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
