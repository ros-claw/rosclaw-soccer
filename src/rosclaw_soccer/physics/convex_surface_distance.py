"""Closest point on a caller-supplied closed convex triangulated surface.

Pure geometry, independent of a simulator's collision or force solver. Caller
owns hull construction, units, transforms and input provenance. No rollout or
contact-force inference is performed here.
"""

import numpy as np


class ConvexSurfaceDistance:
    def __init__(self, triangles: np.ndarray, planes: np.ndarray) -> None:
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or not 4 <= len(triangles) <= 50000
            or planes.ndim != 2
            or planes.shape[1] != 4
            or len(planes) < 4
            or any(
                v.dtype != np.float64 or not np.isfinite(v).all() or np.max(np.abs(v)) > 100
                for v in (triangles, planes)
            )
            or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError("finite closed convex surface and unit outward planes required")
        self._tri = triangles.copy()
        self._planes = planes.copy()
        self._a = self._tri[:, 0]
        self._u = self._tri[:, 1] - self._a
        self._v = self._tri[:, 2] - self._a
        self._normal = np.cross(self._u, self._v)
        self._n2 = np.sum(self._normal**2, axis=1)
        if np.any(self._n2 <= 1e-28):
            raise ValueError("nondegenerate surface triangles required")
        self._d00 = np.sum(self._u**2, axis=1)
        self._d11 = np.sum(self._v**2, axis=1)
        self._d01 = np.sum(self._u * self._v, axis=1)
        self._edge_start = self._tri.reshape(-1, 3)
        self._edge_direction = (self._tri[:, [1, 2, 0]] - self._tri).reshape(-1, 3)
        self._edge_n2 = np.sum(self._edge_direction**2, axis=1)

    def nearest(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        if (
            point.shape != (3,)
            or point.dtype != np.float64
            or not np.isfinite(point).all()
            or np.max(np.abs(point)) > 100
        ):
            raise ValueError("finite local query point required")
        ap = point - self._a
        d20, d21 = np.sum(ap * self._u, axis=1), np.sum(ap * self._v, axis=1)
        u = (self._d11 * d20 - self._d01 * d21) / self._n2
        v = (self._d00 * d21 - self._d01 * d20) / self._n2
        projected = self._a + u[:, None] * self._u + v[:, None] * self._v
        projection_valid = (u >= 0) & (v >= 0) & (u + v <= 1)
        t = np.clip(
            np.sum((point - self._edge_start) * self._edge_direction, axis=1) / self._edge_n2, 0, 1
        )
        edge_points = self._edge_start + t[:, None] * self._edge_direction
        points = np.concatenate((projected, edge_points))
        distance2 = np.sum((points - point) ** 2, axis=1)
        distance2[: len(projected)][~projection_valid] = np.inf
        index = int(np.argmin(distance2))
        distance = float(np.sqrt(distance2[index]))
        if np.all(self._planes[:, :3] @ point + self._planes[:, 3] <= 0):
            distance = -distance
        return distance, points[index].copy()
