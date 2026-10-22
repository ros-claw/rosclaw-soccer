"""Materialize actual successful current-state recovery labels and retain motor memory."""

import argparse
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff80", "audit80", "handoff96", "audit96"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean frozen growing-memory curation source required")
    from rosclaw_soccer.rsi.paid_growth_memory_data import (
        DATA_STATUS,
        SCHEMA,
        SOURCES,
        reconstruct,
        witnesses,
    )

    identity = {k: str(getattr(a, k)) for n in SOURCES for k in (n + "_root", n + "_hash")}
    _, _, _, pins = witnesses(identity)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        common = dict(
            **identity,
            schema=SCHEMA,
            source_head=a.expected_head,
            bank="train",
            input_file_hashes=pins,
            Fresh_fit_rows=0,
            original_DEV16_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", dict(**common))
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        _, _, _, z, manifest = reconstruct(identity, policy)
        features = a.output / "features.npz"
        np.savez_compressed(features, **z)
        memory = a.output / "growth-behavior-memory.json"
        write_once(memory, manifest)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=DATA_STATUS,
                **common,
                declaration_hash=digest(a.output / "declaration.json"),
                feature_path=str(features),
                feature_hash=digest(features),
                behavior_memory_hash=digest(memory),
                manifest=manifest,
                rows=len(z["features"]),
                active_rows=int(z["active"].sum()),
                protected_rows=int((~z["active"]).sum()),
                consumed_neighborhood_development_fit_rows=sum(
                    t["rows"] for t in manifest["added_trajectories"]
                ),
                **{
                    k: manifest[k]
                    for k in (
                        "parent_root",
                        "parent_hash",
                        "parent_checkpoint_hash",
                        "body_scene_hash",
                    )
                },
                optimizer_updates=0,
                own_native_steps=0,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                optimizer_updates=0,
                own_native_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
