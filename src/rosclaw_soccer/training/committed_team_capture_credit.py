"""Conservative, delayed team capture credit for physical soccer research.

This is sampled simulation shaping, not a certified pass, policy promotion,
hardware authority, or a replacement for dense-contact and holdout exams.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.training.receiving_rollout import explain_receiving_window
from rosclaw_soccer.training.team_episode_cut import sampled_team_episode_cut


def committed_team_capture_credit(
    trace: dict[str, Any], ids: tuple[str, ...]
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Credit both players only after a committed, sampled, controlled handoff.

    A pre-integration same-team source/receiver/target snapshot must accompany
    the source foot contact. The first subsequent non-source contact must be
    the receiver's foot, the ball must travel at least 0.5 m and arrive within
    0.75 m of the committed target, and a complete 100-frame receiver capture
    window must pass before team safety or in-play termination. Credit is
    delivered at the *end* of that window, not backdated to the sender. A
    duplicated commitment is counted once, and an incomplete horizon is
    censored rather than a failure or reward. The caller must authenticate all
    traces, align actions and perform independent dense physical validation.
    """
    if (
        not isinstance(trace, dict)
        or type(ids) is not tuple
        or len(ids) != 8
        or any(not isinstance(x, str) for x in ids)
        or tuple(sorted(set(ids))) != ids
        or any(x.partition(".")[0] not in ("red", "blue") or not x.partition(".")[2] for x in ids)
    ):
        raise ValueError("eight unique sorted team-qualified players required")
    required = (
        "time",
        "pass_entry_time_sec",
        "pass_entry_source_code",
        "pass_entry_receiver_code",
        "pass_entry_created_sec",
        "pass_entry_target_valid",
        "pass_entry_target_xy_m",
        "pass_entry_ball_xy_m",
        "pass_entry_ball_velocity_xy_mps",
        "team_native_pre_control_qpos",
        "ball_pose",
        "ball_control_entry_velocity",
        "ball_contact_agent_code",
        "ball_contact_effector_code",
        "ball_contact_force_n",
        "ball_nonfoot_contact_force_n",
        "pass_source_agent_code",
        "pass_target_agent_code",
    )
    try:
        v = {key: np.asarray(trace[key]) for key in required}
    except KeyError as exc:
        raise ValueError("complete pre-control pass and contact evidence required") from exc
    time = v["time"]
    n = len(time) if time.ndim == 1 else 0
    scalar = required[:6] + required[12:]
    if (
        n < 2
        or any(v[k].shape != (n,) for k in scalar)
        or any(v[k].shape != (n, 2) for k in required[6:9])
        or v["team_native_pre_control_qpos"].shape != (n, 8, 43)
        or v["ball_pose"].shape != (n, 7)
        or v["ball_control_entry_velocity"].shape != (n, 6)
        or v["pass_entry_target_valid"].dtype != np.bool_
        or any(
            v[k].dtype.kind not in "iu"
            for k in (
                "pass_entry_source_code",
                "pass_entry_receiver_code",
                "ball_contact_agent_code",
                "ball_contact_effector_code",
                "pass_source_agent_code",
                "pass_target_agent_code",
            )
        )
        or any(v[k].dtype.kind not in "biuf" or not np.isfinite(v[k]).all() for k in required)
        or not np.allclose(np.diff(time), 0.02, atol=1e-7, rtol=0)
        or not np.allclose(time - v["pass_entry_time_sec"], 0.02, atol=1e-7, rtol=0)
        or np.any(v["ball_contact_force_n"] < 0)
        or np.any(v["ball_nonfoot_contact_force_n"] < 0)
    ):
        raise ValueError("aligned finite 50 Hz pre-control team evidence required")
    for key in (
        "pass_entry_source_code",
        "pass_entry_receiver_code",
        "ball_contact_agent_code",
        "pass_source_agent_code",
        "pass_target_agent_code",
    ):
        if not np.isin(v[key], range(9)).all():
            raise ValueError("unknown player code")
    valid = v["pass_entry_target_valid"]
    if np.any(
        valid
        & (
            (v["pass_entry_source_code"] == 0)
            | (v["pass_entry_receiver_code"] == 0)
            | (v["pass_entry_source_code"] == v["pass_entry_receiver_code"])
            | (v["pass_entry_created_sec"] < 0)
            | (v["pass_entry_created_sec"] > v["pass_entry_time_sec"] + 1e-9)
        )
    ):
        raise ValueError("invalid committed pass identity or clock")
    commitments: dict[tuple[int, int, float], NDArray[Any]] = {}
    for frame_index in np.flatnonzero(valid):
        frame = int(frame_index)
        source = int(v["pass_entry_source_code"][frame]) - 1
        receiver = int(v["pass_entry_receiver_code"][frame]) - 1
        if (
            ids[source].split(".")[0] != ids[receiver].split(".")[0]
            or not np.array_equal(
                v["pass_entry_ball_xy_m"][frame],
                v["team_native_pre_control_qpos"][frame, source, 36:38],
            )
            or not np.array_equal(
                v["pass_entry_ball_velocity_xy_mps"][frame],
                v["ball_control_entry_velocity"][frame, :2],
            )
        ):
            raise ValueError("committed pass does not bind same-team pre-control ball state")
        commitment_key = (source, receiver, float(v["pass_entry_created_sec"][frame]))
        target = v["pass_entry_target_xy_m"][frame]
        if commitment_key in commitments and not np.array_equal(
            commitments[commitment_key], target
        ):
            raise ValueError("committed pass target changed within one handoff")
        commitments[commitment_key] = target.copy()
    terminal = sampled_team_episode_cut(trace, agent_ids=ids).first_terminal_frame
    end = n if terminal is None else terminal
    credit = np.zeros((n, 8), dtype=np.float64)
    events: list[dict[str, Any]] = []
    seen: set[tuple[int, int, float]] = set()
    for frame in range(end):
        source = int(v["pass_entry_source_code"][frame])
        receiver = int(v["pass_entry_receiver_code"][frame])
        if not (
            valid[frame]
            and v["ball_contact_agent_code"][frame] == source
            and v["ball_contact_effector_code"][frame] in (1, 2)
            and v["ball_contact_force_n"][frame] > 1
            and v["ball_nonfoot_contact_force_n"][frame] <= 1
        ):
            continue
        if (
            v["pass_source_agent_code"][frame] != source
            or v["pass_target_agent_code"][frame] != receiver
        ):
            raise ValueError("scored sender contact disagrees with committed source or receiver")
        commitment_key = (source, receiver, float(v["pass_entry_created_sec"][frame]))
        if commitment_key in seen:
            continue
        seen.add(commitment_key)
        limit = min(frame + 101, end)
        outcome, receive_frame, credit_frame = "NO_RECEIVER", None, None
        for step in range(frame + 1, limit):
            if v["ball_nonfoot_contact_force_n"][step] > 1:
                outcome = "NONFOOT_INTERFERENCE"
                break
            who = int(v["ball_contact_agent_code"][step])
            if v["ball_contact_force_n"][step] <= 1:
                continue
            if who == source and v["ball_contact_effector_code"][step] in (1, 2):
                continue
            if who != receiver or v["ball_contact_effector_code"][step] not in (1, 2):
                outcome = "OTHER_CONTACT"
                break
            receive_frame = step
            distance = np.linalg.norm(v["ball_pose"][step, :2] - v["pass_entry_ball_xy_m"][frame])
            target_error = np.linalg.norm(
                v["ball_pose"][step, :2] - v["pass_entry_target_xy_m"][frame]
            )
            if distance < 0.5 or target_error > 0.75:
                outcome = "GEOMETRY_REJECTED"
                break
            start = max(1, step - 1)
            if start + 100 > end:
                outcome = "CENSORED_CAPTURE"
                break
            scored = explain_receiving_window(
                trace, agent_ids=ids, agent_id=ids[receiver - 1], start=start, frames=100
            )
            outcome = "CONTROLLED_SAMPLED" if scored["controlled_reception"] else "UNCONTROLLED"
            if scored["controlled_reception"]:
                observed = scored.get("post_contact_observation_sec")
                if (
                    not isinstance(observed, (int, float))
                    or isinstance(observed, bool)
                    or not math.isfinite(observed)
                    or abs((time[start + 99] - observed) - time[step]) > 1e-7
                ):
                    raise ValueError("capture score is not bound to the observed receiver contact")
                credit_frame = start + 99
                credit[credit_frame, source - 1] += 0.1
                credit[credit_frame, receiver - 1] += 0.1
            break
        if outcome == "NO_RECEIVER" and limit < frame + 101:
            outcome = "CENSORED_FLIGHT"
        events.append(
            {
                "source": ids[source - 1],
                "receiver": ids[receiver - 1],
                "source_frame": frame,
                "receiver_frame": receive_frame,
                "credit_frame": credit_frame,
                "outcome": outcome,
                "sampled_only": True,
                "promotion_authorized": False,
            }
        )
    return credit, events
