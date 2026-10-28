"""Screen motion-reference events; never infer physical reception success."""

import numpy as np


def candidate_events(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    body_position: np.ndarray,
    contact_info: np.ndarray,
    fps: int,
) -> list[dict]:
    n = len(ball_position)
    if (
        type(fps) is not int
        or not 30 <= fps <= 240
        or n < fps
        or ball_position.shape != (n, 3)
        or ball_velocity.shape != (n, 3)
        or body_position.shape != (n, 39, 3)
        or contact_info.shape != (n, 4, 1)
        or not np.isfinite(ball_position).all()
        or not np.isfinite(ball_velocity).all()
        or not np.isfinite(body_position).all()
        or not np.isin(contact_info, (0, 1)).all()
    ):
        raise ValueError("finite aligned source arrays and binary reference contacts required")
    window = round(0.3 * fps)
    labels = contact_info[:, :2, 0].astype(bool)
    rows = []
    for foot, body in ((0, 25), (1, 26)):
        starts = np.flatnonzero(labels[:, foot] & ~np.r_[False, labels[:-1, foot]])
        for frame in starts:
            if frame < window or frame + window >= n:
                continue
            point = body_position[frame, body]
            gap = float(np.linalg.norm(ball_position[frame] - point))
            pre_velocity = np.median(ball_velocity[frame - window : frame], axis=0)
            post_velocity = np.median(ball_velocity[frame + 1 : frame + window + 1], axis=0)
            before, after = (
                float(np.linalg.norm(pre_velocity)),
                float(np.linalg.norm(post_velocity)),
            )
            toward = point - ball_position[frame - window]
            norm = float(np.linalg.norm(toward))
            approach = float(pre_velocity @ toward / norm) if norm > 1e-9 else 0.0
            # Immutable research screen, not a calibrated physical contact grade.
            if (
                gap <= 0.4
                and ball_position[frame, 2] <= 0.3
                and approach >= 0.3
                and before - after >= 0.2
            ):
                rows.append(
                    dict(
                        frame=int(frame),
                        foot="left" if foot == 0 else "right",
                        reference_ball_ankle_distance_m=gap,
                        reference_ball_height_m=float(ball_position[frame, 2]),
                        incoming_projection_mps=approach,
                        pre_median_speed_mps=before,
                        post_median_speed_mps=after,
                        reference_only_not_actual_contact=True,
                    )
                )
    return sorted(rows, key=lambda r: (r["frame"], r["foot"]))
