"""Pure measured-geometry projection; no simulator or motor authority.

Callers must establish that the supplied world-frame unit hinge axes and
anchors are ancestors of the measured point. Array shapes cannot prove that
kinematic relationship or qualify a policy for physical execution.
"""

from typing import Any


def hinge_point_jacobian(*, axis: Any, anchor: Any, point: Any) -> Any:
    """Return d(point)/d(hinge angle) as independent [world, xyz, joint] values.

    Explicit, detached float32/64 world-frame geometry only. No prismatic or
    free-joint inference, body-index lookup, generalized force, or state writes.
    """
    import torch

    if (
        not isinstance(axis, torch.Tensor)
        or axis.ndim != 3
        or not 1 <= axis.shape[0] <= 4096
        or not 1 <= axis.shape[1] <= 64
        or axis.shape[2] != 3
        or axis.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("bounded batched unit hinge axes required")
    for value, shape in ((axis, axis.shape), (anchor, axis.shape), (point, (len(axis), 3))):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != axis.dtype
            or value.device != axis.device
            or value.layout != torch.strided
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1e6).any())
        ):
            raise ValueError("finite detached matching world-frame geometry required")
    if bool((torch.linalg.vector_norm(axis, dim=-1) - 1).abs().gt(1e-4).any()):
        raise ValueError("unit world-frame hinge axes required")
    result = torch.linalg.cross(axis, point[:, None] - anchor, dim=-1)
    return result.transpose(1, 2).contiguous()
