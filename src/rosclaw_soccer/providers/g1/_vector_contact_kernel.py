"""Lazy-loaded Warp sensor kernel. Writes only private observation buffers."""

import warp as wp


@wp.kernel
def accumulate_ball_contacts(
    nacon: wp.array[int],
    world: wp.array[int],
    geom: wp.array[wp.vec2i],
    force: wp.array[wp.spatial_vector],
    labels: wp.array[int],
    ball_geom: int,
    worlds: int,
    geometry_count: int,
    peak: wp.array2d[float],
    samples: wp.array2d[int],
    invalid: wp.array[int],
):
    i = wp.tid()
    if nacon[0] < 0 or nacon[0] > geom.shape[0]:
        wp.atomic_max(invalid, 0, 1)
        return
    if i >= nacon[0]:
        return
    a, b = geom[i][0], geom[i][1]
    if a < 0 or b < 0 or a >= geometry_count or b >= geometry_count:
        wp.atomic_max(invalid, 0, 1)
        return
    other = int(-1)  # noqa: UP018 - Warp needs a dynamic variable, not a constant.
    if a == ball_geom:
        other = b
    elif b == ball_geom:
        other = a
    if other < 0:
        return
    kind, w = labels[other], world[i]
    if kind < 0 or kind > 3 or w < 0 or w >= worlds:
        wp.atomic_max(invalid, 0, 1)
        return
    if kind == 0:
        return
    normal = force[i][0]
    if not wp.isfinite(normal):
        wp.atomic_max(invalid, 0, 1)
        return
    if normal > 0.0:
        wp.atomic_max(peak, w, kind - 1, normal)
        wp.atomic_add(samples, w, kind - 1, 1)
