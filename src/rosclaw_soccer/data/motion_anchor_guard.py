"""Bounded candidate indexing for cross-split motion-segment leakage.

Exact rounded anchor equality is an incomplete candidate filter; a lack of
hits never certifies independent sources. Budget exhaustion is an error.
"""

import hashlib
from typing import Any

import numpy as np


class MotionAnchorGuard:
    def __init__(
        self,
        train: dict[str, np.ndarray],
        *,
        window: int = 8,
        stride: int = 32,
        decimals: int = 5,
        translation_columns: tuple[int, ...] = (),
        max_index_entries: int = 100000,
        max_anchor_hits: int = 100000,
        max_candidate_alignments: int = 10000,
    ) -> None:
        if (
            type(train) is not dict
            or not 1 <= len(train) <= 100
            or any(type(k) is not str or not k for k in train)
            or type(window) is not int
            or not 2 <= window <= 32
            or type(stride) is not int
            or not 1 <= stride <= 1000
            or type(decimals) is not int
            or not 0 <= decimals <= 8
            or type(translation_columns) is not tuple
            or len(set(translation_columns)) != len(translation_columns)
            or any(type(i) is not int or not 0 <= i < 256 for i in translation_columns)
            or any(
                type(v) is not int or not 1 <= v <= 1000000
                for v in (max_index_entries, max_anchor_hits, max_candidate_alignments)
            )
        ):
            raise ValueError("explicit bounded train clips and anchor budgets required")
        self.window, self.stride, self.decimals = window, stride, decimals
        self.translation_columns = translation_columns
        self.max_anchor_hits, self.max_candidate_alignments = (
            max_anchor_hits,
            max_candidate_alignments,
        )
        self.train: dict[str, np.ndarray] = {}
        self.index: dict[bytes, list[tuple[str, int]]] = {}
        self.index_entries = 0
        for name, values in sorted(train.items()):
            self._validate(values)
            copied = values.astype(np.float64, copy=True)
            copied.flags.writeable = False
            self.train[name] = copied
            for start in range(0, len(copied) - window + 1, stride):
                self.index_entries += 1
                if self.index_entries > max_index_entries:
                    raise ValueError("motion anchor index budget exhausted")
                key = self._anchor(copied[start : start + window])
                self.index.setdefault(key, []).append((name, start))

    def _validate(self, values: np.ndarray) -> None:
        if (
            type(values) is not np.ndarray
            or values.ndim != 2
            or not self.window <= len(values) <= 60000
            or not 1 <= values.shape[1] <= 256
            or any(i >= values.shape[1] for i in self.translation_columns)
            or values.dtype.kind != "f"
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
        ):
            raise ValueError("finite bounded floating-point motion reference required")

    def _anchor(self, values: np.ndarray) -> bytes:
        normalized = values.astype(np.float64, copy=True)
        if self.translation_columns:
            columns = list(self.translation_columns)
            normalized[:, columns] -= normalized[0, columns].copy()
        rounded = np.round(normalized, decimals=self.decimals)
        rounded[rounded == 0] = 0.0  # Normalize negative zero before hashing.
        return hashlib.blake2b(rounded.astype("<f8", copy=False).tobytes(), digest_size=16).digest()

    def inspect(self, name: str, values: np.ndarray) -> dict[str, Any]:
        """Find and fully compare anchored candidate alignments in one held-out clip."""
        if type(name) is not str or not name or name in self.train:
            raise ValueError("distinct named held-out clip required")
        self._validate(values)
        values = values.astype(np.float64, copy=False)
        candidates: set[tuple[str, int]] = set()
        hits = 0
        for start in range(len(values) - self.window + 1):
            key = self._anchor(values[start : start + self.window])
            for train_name, train_start in self.index.get(key, ()):
                hits += 1
                if hits > self.max_anchor_hits:
                    raise ValueError("motion anchor hit budget exhausted")
                if self.train[train_name].shape[1] != values.shape[1]:
                    raise ValueError("cross-split motion column mismatch")
                candidates.add((train_name, start - train_start))
                if len(candidates) > self.max_candidate_alignments:
                    raise ValueError("motion anchor candidate budget exhausted")
        for train_name, shift in sorted(candidates):
            original = self.train[train_name]
            i, j = max(0, -shift), max(0, shift)
            n = min(len(original) - i, len(values) - j)
            minimum = min(100, len(original), len(values))
            if n < minimum:
                continue
            difference = original[i : i + n] - values[j : j + n]
            if self.translation_columns:
                columns = list(self.translation_columns)
                difference[:, columns] -= difference[0, columns].copy()
            error = np.abs(difference)
            fraction = float((error.max(axis=1) <= 1e-6).mean())
            if fraction >= 0.95:
                raise ValueError(
                    f"motion overlap across splits: {train_name!r} and {name!r}; "
                    f"shift={shift}, overlap={n}"
                )
        return dict(
            held_out_clip=name,
            index_entries=self.index_entries,
            anchor_hits=hits,
            candidate_alignments=len(candidates),
            detected_shared_segments=0,
            detection_method="rounded_exact_anchors_then_full_alignment",
            translation_invariant_columns=self.translation_columns,
            possible_unanchored_near_duplicates_not_excluded=True,
            source_independence_proven=False,
        )
