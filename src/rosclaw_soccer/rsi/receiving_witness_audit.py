"""Independent audit of passive native contact and readiness evidence."""

import numpy as np

from rosclaw_soccer.rsi.receiving_handoff_opportunity import body_contact_kind


def audit_receiving_witness(physics, witness, *, receiver_q, receiver_v, ball_q, ball_v):
    q, v, forces, failed = (physics[k] for k in ("qpos", "qvel", "ball_forces", "body_failed"))
    contacts, rows = witness["contacts"], witness["readiness"]
    n = len(q)
    if (
        q.shape != (n, 10, 79)
        or v.shape != (n, 10, 76)
        or forces.shape != (n, 10, 2, 3)
        or failed.shape != (n, 2)
        or len(contacts) != n
        or len(rows) > n
        or any(not np.isfinite(x).all() for x in (q, v, forces))
        or (forces < 0).any()
    ):
        raise ValueError("complete finite native shared controls required")
    for i, contact in enumerate(contacts):
        if set(contact) != {"a/", "b/"}:
            raise ValueError("both player contact witnesses required")
        for j, prefix in enumerate(("a/", "b/")):
            peak = np.zeros(3)
            for name, force in contact[prefix].items():
                if not name.startswith(prefix) or not np.isfinite(force) or force <= 1:
                    raise ValueError("actual named >1N witnesses required")
                part = (
                    0
                    if name == prefix + "left_ankle_roll_link"
                    else 1
                    if name == prefix + "right_ankle_roll_link"
                    else 2
                )
                peak[part] = max(peak[part], force)
            measured = forces[i, :, j, :].max(axis=0)
            if not np.array_equal(peak, np.where(measured > 1, measured, 0)):
                raise ValueError("named witnesses disagree with native force groups")
    seen = excluded = False
    consecutive = best = 0
    first_ready = None
    failures = dict(distance=0, relative_speed=0, ball_height=0, front=0, root_height=0, upright=0)
    for i, row in enumerate(rows):
        if row["control"] != i + 1 or failed[i].any():
            raise ValueError("contiguous healthy pre-handoff gate controls required")
        for name in contacts[i]["b/"]:
            if body_contact_kind(name, prefix="b/") == "excluded_or_unknown":
                excluded = True
            else:
                seen = True
        root = q[i, -1, receiver_q : receiver_q + 7]
        ball = q[i, -1, ball_q : ball_q + 3]
        delta = ball - root[:3]
        qw, x, y, z = root[3:7]
        metrics = dict(
            distance_xy_m=float(np.linalg.norm(delta[:2])),
            relative_speed_xy_mps=float(
                np.linalg.norm(
                    v[i, -1, ball_v : ball_v + 2] - v[i, -1, receiver_v : receiver_v + 2]
                )
            ),
            ball_height_m=float(ball[2]),
            forward_m=float(
                np.array([1 - 2 * (y * y + z * z), 2 * (x * y + qw * z), 2 * (x * z - qw * y)])
                @ delta
            ),
            root_height_m=float(root[2]),
            upright=float(1 - 2 * (x * x + y * y)),
        )
        if any(not np.isclose(row[k], value, rtol=0, atol=1e-12) for k, value in metrics.items()):
            raise ValueError("readiness metrics disagree with recorded physical state")
        predicates = dict(
            distance=metrics["distance_xy_m"] <= 0.65,
            relative_speed=metrics["relative_speed_xy_mps"] <= 0.6,
            ball_height=metrics["ball_height_m"] <= 0.3,
            front=metrics["forward_m"] >= 0.05,
            root_height=metrics["root_height_m"] >= 0.65,
            upright=metrics["upright"] >= 0.9,
        )
        if seen:
            for key, value in predicates.items():
                failures[key] += not value
        good = seen and not excluded and all(predicates.values())
        consecutive = consecutive + 1 if good else 0
        best = max(best, consecutive)
        ready = good and consecutive >= 15
        if ready and first_ready is None:
            first_ready = i + 1
        if (
            row["contact_seen"],
            row["excluded_contact"],
            row["good"],
            row["consecutive"],
            row["ready"],
            row["first_ready_control"],
        ) != (seen, excluded, good, consecutive, ready, first_ready):
            raise ValueError("recorded readiness latches do not reconstruct")
    first_contact = next((i + 1 for i, c in enumerate(contacts) if c["b/"]), None)
    first_excluded = next(
        (
            i + 1
            for i, c in enumerate(contacts)
            if any(
                body_contact_kind(name, prefix="b/") == "excluded_or_unknown" for name in c["b/"]
            )
        ),
        None,
    )
    return dict(
        first_receiver_contact_control=first_contact,
        first_excluded_contact_control=first_excluded,
        first_ready_control=first_ready,
        best_consecutive_controls=best,
        post_contact_failed_controls=failures,
        gate_controls=len(rows),
    )
