"""Bounded near-duplicate diagnostics for aligned numeric motion references."""

from typing import Any

import numpy as np


def motion_overlap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_frame_shift: int = 30,
    absolute_tolerance: float = 1e-6,
    min_overlap_frames: int = 100,
    required_matching_fraction: float = 0.95,
) -> dict[str, Any]:
    """Find the best bounded temporal shift without treating a pair as independent.

    Positive shift means the first frame of the first clip aligns to that index
    of the second clip. This detects duplicate data, not physical skill quality.
    """
    if (
        type(first) is not np.ndarray
        or type(second) is not np.ndarray
        or first.ndim != 2
        or second.ndim != 2
        or first.shape[1] != second.shape[1]
        or not 1 <= first.shape[1] <= 256
        or not 2 <= len(first) <= 60000
        or not 2 <= len(second) <= 60000
        or first.dtype.kind != "f"
        or second.dtype.kind != "f"
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or type(max_frame_shift) is not int
        or not 0 <= max_frame_shift <= 300
        or type(min_overlap_frames) is not int
        or min_overlap_frames < 2
        or min_overlap_frames > min(len(first), len(second))
        or type(absolute_tolerance) not in (float, int)
        or not np.isfinite(absolute_tolerance)
        or not 0 <= absolute_tolerance <= 0.01
        or type(required_matching_fraction) not in (float, int)
        or not np.isfinite(required_matching_fraction)
        or not 0 < required_matching_fraction <= 1
    ):
        raise ValueError("bounded finite motion arrays and explicit matching criteria required")
    best: dict[str, Any] | None = None
    best_shared: dict[str, Any] | None = None
    for shift in range(-max_frame_shift, max_frame_shift + 1):
        i, j = max(0, -shift), max(0, shift)
        n = min(len(first) - i, len(second) - j)
        if n < min_overlap_frames:
            continue
        error = np.abs(first[i : i + n] - second[j : j + n])
        matching = int((error.max(axis=1) <= absolute_tolerance).sum())
        fraction = matching / n
        row = dict(
            shift_frames=shift,
            first_start=i,
            second_start=j,
            overlap_frames=n,
            matching_frames=matching,
            matching_frame_fraction=fraction,
            maximum_absolute_error=float(error.max()),
            mean_absolute_error=float(error.mean()),
        )
        if best is None or (matching, fraction, n, -abs(shift)) > (
            best["matching_frames"],
            best["matching_frame_fraction"],
            best["overlap_frames"],
            -abs(best["shift_frames"]),
        ):
            best = row
        if fraction >= required_matching_fraction and (
            best_shared is None
            or (matching, fraction, n, -abs(shift))
            > (
                best_shared["matching_frames"],
                best_shared["matching_frame_fraction"],
                best_shared["overlap_frames"],
                -abs(best_shared["shift_frames"]),
            )
        ):
            best_shared = row
    if best is None:
        raise ValueError("no temporal overlap meets declared minimum")
    if best_shared is not None:
        best = best_shared
    return dict(
        **best,
        absolute_tolerance=absolute_tolerance,
        required_matching_fraction=required_matching_fraction,
        overlap_coverage_of_shorter=best["overlap_frames"] / min(len(first), len(second)),
        shared_segment=best["matching_frame_fraction"] >= required_matching_fraction,
        near_duplicate=(
            best["matching_frame_fraction"] >= required_matching_fraction
            and best["overlap_frames"] / min(len(first), len(second)) >= 0.9
        ),
        source_independence_proven=False,
        physical_execution_qualified=False,
    )


def require_no_cross_split_overlap(
    clips: dict[str, np.ndarray], splits: dict[str, str]
) -> dict[str, Any]:
    """Reject a detected shared segment across splits; not proof of independence."""
    if (
        type(clips) is not dict
        or type(splits) is not dict
        or not 2 <= len(clips) <= 100
        or set(clips) != set(splits)
        or any(type(name) is not str or not name for name in clips)
        or any(type(split) is not str or not split for split in splits.values())
    ):
        raise ValueError("bounded explicitly named clips and split membership required")
    names = sorted(clips)
    examined = 0
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            if splits[name] == splits[other]:
                continue
            result = motion_overlap(clips[name], clips[other])
            examined += 1
            if result["shared_segment"]:
                raise ValueError(
                    f"motion overlap across splits: {name!r} and {other!r}; "
                    f"shift={result['shift_frames']}, overlap={result['overlap_frames']}"
                )
    if not examined:
        raise ValueError("at least one cross-split comparison required")
    return dict(
        cross_split_pairs_examined=examined,
        detected_shared_segments=0,
        source_independence_proven=False,
    )
