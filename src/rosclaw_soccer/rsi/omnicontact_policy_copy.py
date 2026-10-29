"""Explicit private CFtrack software/FK copy; not the generic NumPy policy copier.

Only the stateless CPU ONNX session and immutable, fingerprinted FK model are
shared. The execution world is not accepted by or reachable through this API.
External source qualification remains the caller's prerequisite.
"""

import hashlib
import threading
from enum import Enum
from typing import Any

import numpy as np


def _clone_or_hash(policy: Any, *, clone: bool, reference_source: str) -> tuple[Any, str]:
    import mujoco
    import onnxruntime as ort

    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    if (
        type(policy.ort_session) is not ort.InferenceSession
        or policy.ort_session.get_providers() != ["CPUExecutionProvider"]
        or policy.task != "kickball"
        or reference_source not in ("CFgen", "NPZmotion")
        or policy.reference_source != reference_source
        or any(
            getattr(policy, k) is not None
            for k in ("async_stage_thread", "async_stage_pending", "async_stage_error")
        )
        or type(policy.async_stage_lock) is not type(threading.Lock())
        or policy.async_stage_lock.locked()
        or policy.kinematics.__class__.__module__ != "common.mujoco_kinematics"
        or type(policy.kinematics.model) is not mujoco.MjModel
        or type(policy.kinematics.data) is not mujoco.MjData
        or (policy.kinematics.model.nq, policy.kinematics.model.nv) != (36, 35)
    ):
        raise ValueError("qualified synchronous private CFtrack CPU/FK resources required")
    allowed = (
        type(policy),
        type(policy.state_cmd),
        type(policy.policy_output),
        type(policy.kinematics),
    )
    h = hashlib.sha256()
    active: set[int] = set()

    def visit(value: Any) -> Any:
        kind = type(value)
        h.update((kind.__module__ + "." + kind.__qualname__).encode())
        if value is None or kind in (str, bool, int, float) or isinstance(value, Enum):
            if kind is float and not np.isfinite(value):
                raise ValueError("nonfinite CFtrack state")
            h.update(repr(value).encode())
            return value
        if isinstance(value, np.generic):
            visit(value.item())
            return value.copy() if clone else value
        if kind is np.ndarray:
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("finite numeric CFtrack arrays required")
            h.update(str(value.dtype).encode())
            h.update(repr(value.shape).encode())
            h.update(value.tobytes())
            return value.copy() if clone else value
        if kind is mujoco.MjModel:
            if value is not policy.kinematics.model:
                raise ValueError("only source-qualified scratch FK model may be shared")
            h.update(compiled_model_hash(value).encode())
            return value
        if kind is mujoco.MjData:
            if value is not policy.kinematics.data:
                raise ValueError("only source-qualified scratch FK data may be copied")
            visit(
                {
                    k: getattr(value, k)
                    for k in (
                        "time",
                        "qpos",
                        "qvel",
                        "act",
                        "ctrl",
                        "qacc_warmstart",
                        "qfrc_applied",
                        "xfrc_applied",
                        "mocap_pos",
                        "mocap_quat",
                        "userdata",
                        "xpos",
                        "xquat",
                        "xmat",
                    )
                }
            )
            if not clone:
                return value
            copied = mujoco.MjData(policy.kinematics.model)
            mujoco.mj_copyData(copied, policy.kinematics.model, value)
            return copied
        if kind is type(threading.Lock()):
            if value is not policy.async_stage_lock or value.locked():
                raise ValueError("only inactive private stage lock may be copied")
            h.update(b"unlocked")
            return threading.Lock() if clone else value
        if id(value) in active:
            raise ValueError("cyclic CFtrack software graph rejected")
        active.add(id(value))
        try:
            if kind is dict:
                if any(type(k) is not str for k in value):
                    raise ValueError("string-key CFtrack state dictionary required")
                h.update(str(len(value)).encode())
                result_dict = {}
                for key in sorted(value):
                    visit(key)
                    result_dict[key] = visit(value[key])
                return result_dict
            if kind in (list, tuple):
                h.update(str(len(value)).encode())
                result = [visit(v) for v in value]
                return tuple(result) if kind is tuple else result
            if kind in allowed or kind.__module__.startswith("policy.omnicontact.CFgen_"):
                # Do not invoke external __copy__/__deepcopy__/pickle hooks.
                data = visit(vars(value))
                if not clone:
                    return value
                result = object.__new__(kind)
                vars(result).update(data)
                return result
            raise ValueError("unsupported resource in CFtrack graph: " + repr(kind))
        finally:
            active.remove(id(value))

    state = visit({k: v for k, v in vars(policy).items() if k != "ort_session"})
    if not clone:
        return policy, "sha256:" + h.hexdigest()
    copied = object.__new__(type(policy))
    vars(copied).update(state)
    copied.ort_session = policy.ort_session
    return copied, "sha256:" + h.hexdigest()


def cftrack_state_hash(policy: Any, *, reference_source: str = "CFgen") -> str:
    return _clone_or_hash(policy, clone=False, reference_source=reference_source)[1]


def copy_cftrack_policy(policy: Any, *, reference_source: str = "CFgen") -> Any:
    copied, before = _clone_or_hash(policy, clone=True, reference_source=reference_source)
    if (
        cftrack_state_hash(copied, reference_source=reference_source) != before
        or cftrack_state_hash(policy, reference_source=reference_source) != before
    ):
        raise ValueError("private CFtrack resource copy changed state")
    return copied
