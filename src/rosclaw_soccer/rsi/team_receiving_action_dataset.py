"""Import measured team-reception action maps without inventing raw evidence.

Rows originate in an eight-body SIM experiment. They contain physical trace
hashes but not raw trajectories, so validation of report integrity and pairing
must not be represented as independent replay or control authorization.
"""

import re
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

SOURCE_HASH = "sha256:bce0bdf60cdbaf12694465ce9e6840fe43b077987e6cd5cf01ed92a06ce1fcea"


def import_team_action_map(report: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(report, dict)
        or report.get("report_hash") != SOURCE_HASH
        or hash_json({k: v for k, v in report.items() if k != "report_hash"}) != SOURCE_HASH
        or report.get("status") != "ACTION_MAP_INSUFFICIENT_POSITIVE_COURSES"
        or report.get("promotion_authorized") is not False
        or report.get("video_authorized") is not False
        or report.get("activation_ceiling") != "SIM_ONLY"
    ):
        raise ValueError("sealed, rejected eight-body action-map evidence required")
    parent, actions = report.get("parent"), report.get("actions")
    if (
        not isinstance(parent, list)
        or len(parent) != 12
        or not isinstance(actions, list)
        or len(actions) != 15
    ):
        raise ValueError("complete original twelve-course fifteen-action map required")
    features, courses, traces, eligible = [], [], [], []
    rewards = np.zeros((12, 16), np.float32)
    safe = np.zeros((12, 16), np.bool_)
    action_offsets = [np.zeros(12, np.float64)]
    ids = [None]

    def label(row: dict[str, Any]) -> tuple[float, bool]:
        if (
            not isinstance(row, dict)
            or type(row.get("safe")) is not bool
            or type(row.get("controlled_reception")) is not bool
            or not isinstance(row.get("fault_agents"), list)
            or not isinstance(row.get("own_nonfoot_frames"), list)
            or not isinstance(row.get("physical_trace_hash"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", row["physical_trace_hash"]) is None
        ):
            raise ValueError("complete explicit physical safety and contact evidence required")
        physically_safe = row["safe"] and not row["fault_agents"]
        clean = (
            physically_safe
            and row.get("first_foot_frame") is not None
            and not row["own_nonfoot_frames"]
        )
        return float(clean and row["controlled_reception"]), physically_safe

    for i, row in enumerate(parent):
        obs = np.asarray(row.get("activation_features"), dtype=np.float64)
        if (
            obs.shape != (10,)
            or not np.isfinite(obs).all()
            or not isinstance(row.get("course"), dict)
            or row.get("activation_decided") is not True
        ):
            raise ValueError("complete measured pre-decision observation required")
        features.append(obs)
        courses.append(row["course"])
        if (
            type(row.get("protected_episode")) is not bool
            or type(row.get("activation_enabled")) is not bool
        ):
            raise ValueError("explicit protection and activation fields required")
        eligible.append(row["activation_enabled"] and not row["protected_episode"])
        value, is_safe = label(row)
        rewards[i, 0], safe[i, 0] = value, is_safe
        traces.append([row["physical_trace_hash"]])
    if len({row["seed"] for row in courses}) != 12:
        raise ValueError("distinct source course identities required")
    for j, action in enumerate(actions, start=1):
        offset = np.asarray(action.get("offset"), dtype=np.float64)
        rows, action_id = action.get("rows"), action.get("action_id")
        if (
            offset.shape != (12,)
            or not np.isfinite(offset).all()
            or np.abs(offset).max() > 1
            or not isinstance(rows, list)
            or len(rows) != 12
            or not isinstance(action_id, list)
            or len(action_id) != 2
        ):
            raise ValueError("complete finite original action and row identity required")
        action_offsets.append(offset)
        ids.append(action_id)
        for i, row in enumerate(rows):
            if row.get("course") != courses[i] or not np.array_equal(
                np.asarray(row.get("activation_features")), features[i]
            ):
                raise ValueError("paired action must share exact pre-decision measured state")
            value, is_safe = label(row)
            rewards[i, j], safe[i, j] = value, is_safe
            traces[i].append(row["physical_trace_hash"])
    if ids[1:] != report.get("action_ids"):
        raise ValueError("action identity or order drift")
    if len({tuple(item) for item in ids[1:]}) != 15:
        raise ValueError("unique action identities required")
    return dict(
        features=np.stack(features),
        actions=np.stack(action_offsets),
        rewards=rewards,
        safe=safe,
        eligible=np.asarray(eligible, dtype=np.bool_),
        courses=courses,
        action_ids=ids,
        trace_hashes=traces,
        source_hash=SOURCE_HASH,
        raw_trajectories_available=False,
        independent_replay_verified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
