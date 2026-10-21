"""Rebuild support only; reuse exact learned head, zero new optimization or physics."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp = a.source_root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.source_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused support experiment, exact source and frozen code required")
    source = json.loads(rp.read_text())
    pins = dict(source["input_file_hashes"])
    for f in (rp, a.source_root / "declaration.json", Path(source["checkpoint_path"])):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import torch

        from rosclaw_soccer.rsi.paid_active_compact_student import (
            ACTIVE_STATUS,
            active_compact_config,
            apply_active_support,
            validate_active_payload,
        )
        from rosclaw_soccer.rsi.paid_compact_student import audit_compact_batch1, load_compact_fit

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        _, head, source, z = load_compact_fit(a.source_root, a.source_hash, policy)
        before = {k: v.clone() for k, v in head.net.state_dict().items()}
        identity = dict(
            config=active_compact_config(source["config"]["seed_index"]),
            source_head=a.expected_head,
            source_fit_root=str(a.source_root),
            source_fit_hash=a.source_hash,
            data_hash=source["data_hash"],
            body_scene_hash=source["body_scene_hash"],
        )
        write_once(
            a.output / "declaration.json",
            dict(
                **identity,
                input_file_hashes=pins,
                source_head_weights_exact=True,
                optimizer_updates=0,
                Fresh_fit_rows=0,
                DEV_fit_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        apply_active_support(head, z)
        if any(not torch.equal(v, before[k]) for k, v in head.net.state_dict().items()):
            raise ValueError("support ablation changed neural weights")
        cert = audit_compact_batch1(head, source, z, active_only=True)
        s = dict(
            schema="paid.active_compact_support.v1",
            **identity,
            support={k: v.clone() for k, v in head.support.state_dict().items()},
            hardware_authorized=False,
            promotion_authorized=False,
        )
        cp = a.output / "active-support.pt"
        r = dict(
            status=ACTIVE_STATUS,
            **identity,
            input_file_hashes=pins,
            batch1_certificate=cert,
            optimizer_updates=0,
            inherited_optimizer_updates=1024,
            head_weights_exact_source=True,
            own_native_steps=0,
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        validate_active_payload(s, r, source)
        torch.save(s, cp)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                **r,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                declaration_hash=digest(a.output / "declaration.json"),
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
