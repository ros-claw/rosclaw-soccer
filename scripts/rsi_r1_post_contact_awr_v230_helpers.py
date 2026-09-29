"""Strict sealed trajectory reader for offline successful-contact learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes


def load_success_trajectories(
    directory: Path, report: dict[str, Any], holdout_seeds: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_states = []
    train_actions = []
    holdout_states = []
    holdout_actions = []
    manifest = []
    for history in report["history"]:
        update = history["update"]
        for index, row in enumerate(history["sample_trajectories"]):
            path = directory / f"sample-u{update}-e{index}.npz"
            if hash_bytes(path.read_bytes()) != row["trajectory_hash"]:
                raise ValueError("sealed physical online trajectory hash required")
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != {"frames", "features", "logits"}:
                    raise ValueError("only finite measured post-contact trajectory fields allowed")
                frames = np.asarray(arrays["frames"], dtype=np.int64)
                features = np.asarray(arrays["features"], dtype=np.float64)
                logits = np.asarray(arrays["logits"], dtype=np.float64)
            if (
                frames.ndim != 1
                or features.shape != (len(frames), 48)
                or logits.shape != (len(frames), 12)
                or not np.isfinite(features).all()
                or not np.isfinite(logits).all()
                or np.any(np.diff(frames) <= 0)
            ):
                raise ValueError("complete aligned finite post-contact trajectories required")
            successful = bool(
                row["safe"] and row["controlled_reception"] and not row["own_nonfoot_frames"]
            )
            seed = int(row["course"]["seed"])
            part = "holdout" if seed in holdout_seeds else "train"
            manifest.append(
                {
                    "seed": seed,
                    "update": update,
                    "episode": index,
                    "partition": part,
                    "successful": successful,
                    "frames": len(frames),
                    "trajectory_hash": row["trajectory_hash"],
                }
            )
            if not successful or not len(frames):
                continue
            if part == "holdout":
                holdout_states.extend(features)
                holdout_actions.extend(logits)
            else:
                train_states.extend(features)
                train_actions.extend(logits)
    return (
        np.asarray(train_states, dtype=np.float32).reshape(-1, 48),
        np.asarray(train_actions, dtype=np.float32).reshape(-1, 12),
        np.asarray(holdout_states, dtype=np.float32).reshape(-1, 48),
        np.asarray(holdout_actions, dtype=np.float32).reshape(-1, 12),
        manifest,
    )
