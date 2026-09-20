"""Check suppression telemetry without mistaking eligibility for execution.

Callers must independently authenticate trajectory hashes and producer code.
This verifies recorded consistency, not the provenance or physical truth of
arbitrary caller-supplied arrays. It never authorizes policy promotion.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.training.contact_teacher_ablation import ContactTeacherSuppression


@dataclass(frozen=True)
class TeacherSuppressionWitness:
    contract_hash: str
    suppressed_frames: int
    eligible_but_unexecuted_frames: int


def inspect_teacher_suppression(
    trace: Mapping[str, Any],
    *,
    contract: ContactTeacherSuppression,
    agent_ids: tuple[str, ...],
) -> TeacherSuppressionWitness:
    contract.__post_init__()
    if (
        type(agent_ids) is not tuple
        or not agent_ids
        or any(type(agent) is not str for agent in agent_ids)
        or tuple(sorted(set(agent_ids))) != agent_ids
        or contract.agent_id not in agent_ids
    ):
        raise ValueError("explicit sorted physical roster required")
    time = np.asarray(trace["time"])
    if (
        time.ndim != 1
        or not contract.start_frame < len(time) <= 50_000
        or time.dtype.kind not in "fiu"
        or not np.isfinite(time).all()
        or np.any(time < 0)
        or not np.allclose(np.diff(time), 0.02, atol=1e-7, rtol=0)
    ):
        raise ValueError("measured contiguous 50 Hz suppression window required")
    n = len(time)

    def array(name: str, shape: tuple[int, ...], kinds: str) -> np.ndarray:
        values = np.asarray(trace[name])
        if values.shape != shape or values.dtype.kind not in kinds:
            raise ValueError(f"invalid suppression telemetry: {name}")
        if values.dtype.kind != "U" and not np.isfinite(values).all():
            raise ValueError(f"nonfinite suppression telemetry: {name}")
        return values

    hashes = array("contact_teacher_suppression_contract", (n,), "U")
    agents = array("contact_teacher_suppressed_agent", (n,), "U")
    expected = np.asarray([contract.suppressed_agent(i) or "" for i in range(n)])
    if not np.all(hashes == contract.contract_hash) or not np.array_equal(agents, expected):
        raise ValueError("suppression contract or boundary differs")
    codes = array("contact_teacher_agent_code", (n,), "iu")
    if not np.isin(codes, np.arange(len(agent_ids) + 1)).all():
        raise ValueError("invalid teacher candidate identity")
    active = array("contact_teacher_active", (n,), "b")
    valid = array("contact_teacher_last_substep_valid", (n,), "b")
    peak = array("contact_teacher_peak_torque_nm", (n,), "fiu")
    if np.any(peak < 0):
        raise ValueError("teacher peak torque cannot be negative")
    no_candidate = codes == 0
    if active[no_candidate].any() or valid[no_candidate].any() or np.any(peak[no_candidate] != 0):
        raise ValueError("executed teacher telemetry has no candidate identity")
    focal = (codes == agent_ids.index(contract.agent_id) + 1) & (
        np.arange(n) >= contract.start_frame
    )
    if active[focal].any() or valid[focal].any() or np.any(peak[focal] != 0):
        raise ValueError("suppressed teacher was recorded executing")
    for name in (
        "contact_teacher_residual_torque_nm",
        "contact_teacher_tracking_adjustment_nm",
    ):
        values = array(name, (n, 29), "fiu")
        if np.any(values[no_candidate] != 0):
            raise ValueError("teacher task torque has no candidate identity")
        if np.any(values[focal] != 0):
            raise ValueError("suppressed teacher has nonzero recorded task torque")
    return TeacherSuppressionWitness(
        contract.contract_hash, n - contract.start_frame, int(focal.sum())
    )
