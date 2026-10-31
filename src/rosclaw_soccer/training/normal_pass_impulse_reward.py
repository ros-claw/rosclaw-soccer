"""Measured foot-normal pass shaping; simulation-only, not pass certification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.role_self_model import TacticalIntent

LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def pass_normal_adjustment(trace: dict[str, Any], ids: tuple[str, ...]) -> np.ndarray[Any, Any]:
    """Replace PASS speed credit with attributed foot-normal impulse credit.

    The contact force is integrated over 500 Hz but the role, target, and
    commitment are 50 Hz samples. Ground and tangential impulses are excluded.
    Contact ambiguity is conservatively censored; a result is not evidence of
    a completed pass, target arrival, or authorization for physical motion.
    """
    n = len(trace["time"])
    ball = np.asarray(trace["ball_pose"], dtype=float)
    velocity = np.asarray(trace["ball_velocity"], dtype=float)
    normal = np.asarray(trace["contact_normal_impulse_ns"], dtype=float)
    substeps = np.asarray(trace["contact_normal_substeps"])
    ambiguous = np.asarray(trace["contact_normal_ambiguous"])
    safe = np.asarray(trace["contact_normal_world_safe"])
    contact = np.asarray(trace["ball_contact_agent_code"])
    force = np.asarray(trace["ball_contact_force_n"], dtype=float)
    foot = np.isin(trace["ball_contact_effector_code"], (1, 2))
    source = np.asarray(trace["pass_source_agent_code"])
    target_agent = np.asarray(trace["pass_target_agent_code"])
    if (
        len(ids) != 8
        or tuple(sorted(set(ids))) != ids
        or ball.shape != (n, 7)
        or velocity.shape != (n, 6)
        or normal.shape != (n, 8, 3)
        or substeps.shape != (n, 8)
        or substeps.dtype.kind not in "iu"
        or np.any((substeps < 0) | (substeps > 10))
        or ambiguous.shape != (n, 8)
        or ambiguous.dtype != np.bool_
        or safe.shape != (n,)
        or safe.dtype != np.bool_
        or any(v.shape != (n,) for v in (contact, force, foot, source, target_agent))
        or any(v.dtype.kind not in "iu" for v in (source, target_agent))
        or not np.isin(source, range(9)).all()
        or not np.isin(target_agent, range(9)).all()
        or any(not np.isfinite(v).all() for v in (ball, velocity, normal, force))
        or np.any(force < 0)
        or np.any((substeps == 0) & np.any(normal != 0, axis=2))
    ):
        raise ValueError("complete finite 500Hz contact evidence required")
    result = np.zeros((n, 8), dtype=np.float64)
    pass_code = list(TacticalIntent).index(TacticalIntent.PASS)
    for i, agent in enumerate(ids):
        key = agent.replace(".", "_")
        intent = np.asarray(trace[key + "_intent_code"])
        target = np.asarray(trace[key + "_target_position"], dtype=float)
        if (
            intent.shape != (n,)
            or intent.dtype.kind not in "iu"
            or not np.isin(intent, range(len(TacticalIntent))).all()
            or target.shape != (n, 3)
            or not np.isfinite(target).all()
        ):
            raise ValueError("valid sampled pass intent and target required")
        ray = target[:, :2] - ball[:, :2]
        length = np.linalg.norm(ray, axis=1)
        direction = ray / np.maximum(length[:, None], 1e-6)
        passing = intent == pass_code
        touching = (contact == i + 1) & foot & (force > 1)
        toward = np.sum(velocity[:, :2] * direction, axis=1)
        result[:, i] -= passing * touching * 0.04 * np.clip(toward, -2, 2)
        teammates = [
            j + 1
            for j, other in enumerate(ids)
            if j != i and other.split(".")[0] == agent.split(".")[0]
        ]
        eligible = (
            passing
            & (substeps[:, i] > 0)
            & ~ambiguous[:, i]
            & safe
            & (source == i + 1)
            & np.isin(target_agent, teammates)
            & (length > 1e-6)
        )
        # Ball mass for this particular task is 0.43 kg; the bounded value is
        # used only as a comparable velocity-scale reward, not as physics proof.
        dv = normal[:, i, :2] / 0.43
        forward = np.sum(dv * direction, axis=1)
        lateral = dv[:, 0] * -direction[:, 1] + dv[:, 1] * direction[:, 0]
        result[:, i] += eligible * 0.04 * (np.clip(forward, -2, 2) - np.clip(np.abs(lateral), 0, 2))
    if not np.isfinite(result).all():
        raise ValueError("invalid normal-contact reward")
    return result
