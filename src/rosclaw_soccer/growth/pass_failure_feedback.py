"""Contact-derived diagnostics for attempted passes; never promotion evidence."""

from __future__ import annotations

from typing import Any

import numpy as np


def diagnose_passes(
    trace: dict[str, Any], agent_ids: tuple[str, ...], *, launch_relative: bool = False
) -> list[dict[str, Any]]:
    if type(launch_relative) is not bool:
        raise ValueError("pass feedback contract must be explicit")
    time = np.asarray(trace["time"], dtype=float)
    if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("pass feedback requires finite monotonic physical time")
    if tuple(sorted(set(agent_ids))) != agent_ids:
        raise ValueError("pass feedback agent codes must be unique and sorted")

    def codes(name: str) -> np.ndarray:
        raw = np.asarray(trace[name], dtype=float)
        if raw.shape != time.shape or not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)):
            raise ValueError("agent codes must be finite integers matching physical time")
        return raw.astype(int)

    source = codes("pass_source_agent_code")
    target = codes("pass_target_agent_code")
    contact = codes("ball_contact_agent_code")
    foot = np.isin(trace["ball_contact_effector_code"], [1, 2])
    force = np.asarray(trace["ball_contact_force_n"], dtype=float)
    ball = np.asarray(trace["ball_pose"], dtype=float)[:, :3]
    velocity = np.asarray(trace["ball_velocity"], dtype=float)[:, :3]
    for values in (source, target, contact, force, ball, velocity):
        if len(values) != len(time) or not np.all(np.isfinite(values)):
            raise ValueError("pass feedback arrays must match physical time")
    if any(np.any((codes < 0) | (codes > len(agent_ids))) for codes in (source, target, contact)):
        raise ValueError("pass agent code is outside the roster")
    rows = []
    previous = (0, 0)
    for start, pair in enumerate(zip(source, target, strict=True)):
        is_new = pair != previous
        previous = pair
        if not is_new or 0 in pair:
            continue
        sender, receiver = (agent_ids[int(code) - 1] for code in pair)
        if sender == receiver or sender.split(".")[0] != receiver.split(".")[0]:
            raise ValueError("pass commitment must name a distinct teammate")
        end = int(np.searchsorted(time, time[start] + 3.0, side="right"))
        commitment_end = start
        while (
            commitment_end + 1 < len(time)
            and (source[commitment_end + 1], target[commitment_end + 1]) == pair
        ):
            commitment_end += 1
        contact_end = min(
            end, int(np.searchsorted(time, time[commitment_end] + 0.30, side="right"))
        )
        attempts = np.flatnonzero(
            (contact[start:contact_end] == pair[0])
            & foot[start:contact_end]
            & (force[start:contact_end] > 0)
        )
        row: dict[str, Any] = {
            "sender": sender,
            "receiver": receiver,
            "commitment_time_sec": float(time[start]),
            "diagnostic_only": True,
            "physical_receive_confirmed": False,
        }
        if not len(attempts):
            row["failure"] = "NO_POST_COMMITMENT_FOOT_CONTACT"
            rows.append(row)
            continue
        first = start + int(attempts[0])
        if launch_relative:
            end = int(np.searchsorted(time, time[first] + 3.0, side="right"))
        opponent_codes = [
            i + 1 for i, name in enumerate(agent_ids) if name.split(".")[0] != sender.split(".")[0]
        ]
        intervening = np.flatnonzero(np.isin(contact[first + 1 : end], opponent_codes))
        nonfoot_interruption = False
        if launch_relative:
            nonfoot = codes("ball_nonfoot_contact_agent_code")
            nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"], dtype=float)
            if nonfoot_force.shape != time.shape or not np.all(np.isfinite(nonfoot_force)):
                raise ValueError("pass feedback requires finite nonfoot contact forces")
            if nonfoot[first] > 0 and nonfoot_force[first] > 1e-6:
                row.update(foot_contact_time_sec=float(time[first]), failure="NONFOOT_INTERRUPTION")
                rows.append(row)
                continue
            nonfoot_events = np.flatnonzero(
                (nonfoot[first + 1 : end] > 0) & (nonfoot_force[first + 1 : end] > 1e-6)
            )
            if len(nonfoot_events) and (
                not len(intervening) or nonfoot_events[0] <= intervening[0]
            ):
                intervening = nonfoot_events
                nonfoot_interruption = True
        if len(intervening):
            end = first + 1 + int(intervening[0])
        key = receiver.replace(".", "_")
        feet = [
            np.asarray(trace[key + suffix], dtype=float)
            for suffix in ("_left_foot_position", "_right_foot_position")
        ]
        if any(f.shape != (len(time), 3) or not np.all(np.isfinite(f)) for f in feet):
            raise ValueError("pass feedback receiver feet are invalid")
        distance = np.minimum(
            *(np.linalg.norm(f[first:end] - ball[first:end], axis=1) for f in feet)
        )
        receiver_xy = (feet[0][first, :2] + feet[1][first, :2]) / 2.0
        direction = receiver_xy - ball[first, :2]
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        peak_toward = float(np.max(velocity[first:end, :2] @ direction))
        received = bool(
            np.any(
                (contact[first + 1 : end] == pair[1])
                & foot[first + 1 : end]
                & (force[first + 1 : end] > 0)
            )
        )
        row.update(
            {
                "foot_contact_time_sec": float(time[first]),
                "receiver_minimum_foot_distance_m": float(distance.min()),
                "peak_velocity_toward_receiver_mps": peak_toward,
                "physical_receive_confirmed": received,
                "failure": None
                if received
                else "NONFOOT_INTERRUPTION"
                if nonfoot_interruption
                else "OPPONENT_TOUCH"
                if len(intervening)
                else "WRONG_DIRECTION"
                if peak_toward <= 0
                else "MISSED_RECEIVER",
            }
        )
        rows.append(row)
    return rows
