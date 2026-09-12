"""Explicit planar vector basis changes; no robot-specific feature layout.

Callers identify world-vector pairs. Local angular velocity, gravity, joints,
and latent actions must not be implicitly rotated. This helper grants no
execution authority and does not make a learned policy rotation invariant.
Torch remains optional until invocation; gradients are preserved.
"""

from typing import Any


def rotate_planar_feature_pairs(
    features: Any, angles: Any, *, pairs: tuple[tuple[int, int], ...]
) -> Any:
    """Rotate selected XY columns by one finite angle per batch row.

    Inputs are never modified. Angles and features share dtype/device; inputs
    must be float32/64. Column pairs are explicit, disjoint and in bounds.
    Bounds and semantic frame ownership remain the caller's responsibility.
    """
    import torch

    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 65536
        or not 1 <= features.shape[1] <= 4096
        or features.dtype not in (torch.float32, torch.float64)
        or not isinstance(angles, torch.Tensor)
        or angles.shape != (features.shape[0],)
        or angles.dtype != features.dtype
        or angles.device != features.device
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(angles).all())
        or bool((angles.abs() > torch.pi).any())
    ):
        raise ValueError("aligned finite feature matrix and wrapped planar angles required")
    if type(pairs) is not tuple or not pairs:
        raise ValueError("explicit nonempty tuple of disjoint XY column pairs required")
    used: set[int] = set()
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("each planar vector needs two column indices")
        for index in pair:
            if type(index) is not int or not 0 <= index < features.shape[1] or index in used:
                raise ValueError("planar columns must be distinct integer indices in bounds")
            used.add(index)
    output = features.clone()
    cosine, sine = torch.cos(angles), torch.sin(angles)
    for x, y in pairs:
        output[:, x] = cosine * features[:, x] - sine * features[:, y]
        output[:, y] = sine * features[:, x] + cosine * features[:, y]
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("planar feature rotation overflowed")
    return output
