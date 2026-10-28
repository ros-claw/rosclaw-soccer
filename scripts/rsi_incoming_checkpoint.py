"""Immutable episode-boundary checkpoints for SIM incoming PPO experiments."""

import hashlib
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def save_checkpoint(path, *, policy, optimizer, commitment, episode, updates, stats, histories):
    """Publish once only after a complete episode and optimizer update."""
    if path.exists() or path.is_symlink():
        raise ValueError("checkpoint destination must be unused")
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state = {
        "schema": "incoming-ppo-episode-checkpoint-v1",
        "commitment": commitment,
        "next_episode": episode + 1,
        "updates": updates,
        "stats": stats,
        "histories": [x.tolist() for x in histories],
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": [algorithm, keys.tolist(), position, has_gauss, cached_gaussian],
        "python_rng": random.getstate(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".checkpoint-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest_file(path)


def restore_checkpoint(path, expected_hash, *, policy, optimizer, commitment):
    """Fail closed on file/protocol mismatch; restore RNG after model initialization."""
    if path.is_symlink() or not path.is_file() or digest_file(path) != expected_hash:
        raise ValueError("exact regular checkpoint hash required")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("schema") != "incoming-ppo-episode-checkpoint-v1"
        or state.get("commitment") != commitment
        or type(state.get("next_episode")) is not int
        or state["next_episode"] < 1
        or len(state["stats"]) != state["next_episode"]
        or len(state["histories"]) != state["next_episode"]
    ):
        raise ValueError("checkpoint experiment commitment or episode count mismatch")
    for record in state["stats"]:
        if digest_file(Path(record["raw_path"])) != record["raw_hash"]:
            raise ValueError("checkpoint trajectory hash mismatch")
    if len(state["cuda_rng"]) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA device count mismatch")
    policy.load_state_dict(state["policy"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    algorithm, keys, position, has_gauss, cached_gaussian = state["numpy_rng"]
    np.random.set_state(
        (algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )
    random.setstate(state["python_rng"])
    return state
