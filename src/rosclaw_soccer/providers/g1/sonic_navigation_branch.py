"""Owned plain SIM navigation state, including future planner refresh state.

Unlike the short tracker branch, this copies the planner counter, reference,
command and observation history. It does not advance a planner, neural model,
clock or physics, and is NOT a complete robot controller/body fork. Session
resources are shared read-only; every mutable per-episode field is copied.
No existing controller selects this research helper by default.
"""

import math
from collections import deque
from typing import Any

import numpy as np
import onnxruntime as ort

from rosclaw_soccer.providers.g1.sonic_navigation import (
    G1SonicNavigation,
    SonicNavigationConfig,
    _StreamingBackend,
)
from rosclaw_soccer.providers.g1.sonic_runup import (
    _VARIANTS,
    G1SonicQualification,
    G1SonicRunupConfig,
)

NAVIGATION_FIELDS = frozenset(
    {
        "agent_id",
        "config",
        "backend",
        "contract_hash",
        "_next_frame",
        "_faulted",
        "_retired",
        "_origin_frame",
        "_ready_from_handoff",
        "_ready_from_observation",
        "_boundary_observation_hash",
        "_batch_pending",
    }
)
BACKEND_FIELDS = frozenset(
    {
        "navigation",
        "command",
        "facing",
        "planner_calls",
        "last_planned_frame",
        "last_planned_command",
        "events",
        "latent_records",
        "config",
        "_variant",
        "qualification",
        "_planner",
        "_encoder",
        "_decoder",
        "_history",
        "reference",
        "reference_digest",
        "action",
        "target",
        "_history_handoff_binding",
        "_kp",
        "_kd",
        "_action_scale",
    }
)
MAX_OWNED_BYTES = 8 * 1024 * 1024


def fork_plain_navigation(navigation: G1SonicNavigation, *, next_frame: int) -> G1SonicNavigation:
    """Fork one completed, nonfaulted, observation-started H300 option.

    This interface accepts neither future observations, external reference
    arrays, policy targets, nor callables. The caller must separately own and
    qualify body, Foundation tracker, IK and residual state before forecasting.
    """
    expected = SonicNavigationConfig(
        maximum_frames=300,
        planner_seed=30300,
        model_variant="low_latency",
        experimental_maximum_speed_mps=1.5,
        inference_threads=1,
        record_default_planner_events=True,
    )
    if (
        type(navigation) is not G1SonicNavigation
        or set(vars(navigation)) != NAVIGATION_FIELDS
        or type(next_frame) is not int
        or not 1 <= next_frame < 300
        or type(navigation._next_frame) is not int
        or type(navigation._origin_frame) is not int
        or navigation._next_frame != next_frame
        or navigation._origin_frame != 0
        or navigation._faulted is not False
        or navigation._retired is not False
        or navigation._batch_pending is not False
        or navigation._ready_from_handoff is not False
        or navigation._ready_from_observation is not True
        or type(navigation._boundary_observation_hash) is not str
        or type(navigation.config) is not SonicNavigationConfig
        or navigation.config != expected
        or type(navigation.backend) is not _StreamingBackend
    ):
        raise ValueError("exact completed plain H300 navigation boundary required")
    b = navigation.backend
    if (
        set(vars(b)) != BACKEND_FIELDS
        or b.navigation is not navigation.config
        or type(b.config) is not G1SonicRunupConfig
        or b.config != G1SonicRunupConfig(model_variant="low_latency", execution_duration_sec=4.5)
        or b._variant is not _VARIANTS["low_latency"]
        or type(b.qualification) is not G1SonicQualification
        or b.qualification.model_variant != "low_latency"
        or type(b.planner_calls) is not int
        or b.planner_calls != 1 + (next_frame - 1) // 20
        or type(b.last_planned_frame) is not int
        or b.last_planned_frame != ((next_frame - 1) // 20) * 20
        or type(b.events) is not list
        or len(b.events) != b.planner_calls
        or type(b.latent_records) is not list
        or b.latent_records != []
        or b._history_handoff_binding is not None
        or type(b._history) is not deque
        or b._history.maxlen != 10
        or len(b._history) != 10
        or not isinstance(b.reference, np.ndarray)
        or b.reference.shape != (410, 36)
        or b.reference.dtype != np.float64
    ):
        raise ValueError("whole reviewed owned planner/reference/history state required")
    b.qualification.require_eligible()
    for field, shape, dtype in (
        ("action", (29,), np.float32),
        ("target", (29,), np.float64),
        ("_kp", (29,), np.float64),
        ("_kd", (29,), np.float64),
        ("_action_scale", (29,), np.float64),
    ):
        value = getattr(b, field)
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != dtype:
            raise ValueError("complete canonical navigation numeric fields required")
    if any(
        type(entry) is not tuple
        or len(entry) != 5
        or any(
            type(a) is not np.ndarray or a.shape != (width,) or a.dtype != np.float64
            for a, width in zip(entry, (3, 29, 29, 29, 3), strict=True)
        )
        for entry in b._history
    ):
        raise ValueError("complete ten-frame measured navigation history required")
    for field in ("_planner", "_encoder", "_decoder"):
        session = getattr(b, field)
        if type(session) is not ort.InferenceSession or session.get_providers() != [
            "CPUExecutionProvider"
        ]:
            raise ValueError("same original read-only CPU ONNX session resources required")
    size = nodes = 0

    def clone(value: Any, depth: int = 0) -> Any:
        nonlocal size, nodes
        nodes += 1
        if depth > 32 or nodes > 20000:
            raise ValueError("bounded acyclic navigation state required")
        if type(value) is np.ndarray:
            size += value.nbytes
            if (
                size > MAX_OWNED_BYTES
                or value.dtype not in (np.float32, np.float64)
                or not np.isfinite(value).all()
            ):
                raise ValueError("bounded finite owned navigation tensors required")
            return value.copy()
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("finite owned navigation metadata required")
            return value
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is list:
            return [clone(v, depth + 1) for v in value]
        if type(value) is tuple:
            return tuple(clone(v, depth + 1) for v in value)
        if type(value) is dict and all(type(k) is str for k in value):
            return {k: clone(v, depth + 1) for k, v in value.items()}
        if type(value) is deque and value.maxlen == 10:
            return deque((clone(v, depth + 1) for v in value), maxlen=10)
        raise ValueError("unreviewed mutable navigation state cannot be shared")

    backend = object.__new__(_StreamingBackend)
    shared = {
        "navigation",
        "config",
        "_variant",
        "qualification",
        "_planner",
        "_encoder",
        "_decoder",
    }
    for key, value in vars(b).items():
        setattr(backend, key, value if key in shared else clone(value))
    branch = object.__new__(G1SonicNavigation)
    for key, value in vars(navigation).items():
        setattr(
            branch, key, backend if key == "backend" else value if key == "config" else clone(value)
        )
    return branch
