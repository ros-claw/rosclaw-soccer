"""Reconstruct a fixed neural/prototype wrapper; zero new SGD or native motion."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen gate experiment required")

    clean()
    if a.output.exists():
        raise ValueError("unused prototype-wrapper evidence required")
    rp = a.audit_root / "complete.json"
    if rp.is_symlink() or digest(rp) != a.audit_hash:
        raise ValueError("exact independently checked neural head required")
    audited = json.loads(rp.read_text())
    pins = dict(audited["input_file_hashes"])
    pins[str(rp)] = a.audit_hash
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import torch

        from rosclaw_soccer.rsi.paid_negative_core import (
            STATUS,
            negative_core_config,
            reconstruct_negative_core,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        _, _, _, fit, data, cert = reconstruct_negative_core(a.audit_root, a.audit_hash, policy)
        common = dict(
            source_head=a.expected_head,
            config=negative_core_config(),
            audit_root=str(a.audit_root),
            audit_hash=a.audit_hash,
            **{k: fit[k] for k in ("body_scene_hash", "parent_hash", "data_hash")},
            feature_hash=data["feature_hash"],
        )
        write_once(
            a.output / "declaration.json",
            dict(
                **common,
                input_file_hashes=pins,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        state = dict(
            schema="paid.negative_core_neural.references.v1",
            config=common["config"],
            audit_hash=a.audit_hash,
            data_hash=fit["data_hash"],
            feature_hash=data["feature_hash"],
            parent_hash=fit["parent_hash"],
            body_scene_hash=fit["body_scene_hash"],
            certificate=cert,
            declaration_hash=digest(a.output / "declaration.json"),
        )
        cp = a.output / "negative-core-references.pt"
        torch.save(state, cp)
        if torch.load(cp, map_location="cpu", weights_only=True) != state:
            raise ValueError("safe reference metadata reload differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=STATUS,
                **common,
                input_file_hashes=pins,
                certificate=cert,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                declaration_hash=state["declaration_hash"],
                optimizer_updates=0,
                inherited_neural_optimizer_updates=6144,
                own_native_steps=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), automatic_retry=False, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
