"""Fit a deterministic contextual muscle-memory reference, with zero-coefficient control."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--zero-control", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused cardinal fitting output and clean frozen source required")
    rp = a.data_root / "complete.json"
    if rp.is_symlink() or digest(rp) != a.data_hash:
        raise ValueError("exact closed TRAIN data required")
    data = json.loads(rp.read_text())
    pins = dict(data["input_file_hashes"])
    for path in (rp, a.data_root / "declaration.json"):
        pins[str(path)] = digest(path)
    pins[data["feature_path"]] = data["feature_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import torch

        from rosclaw_soccer.rsi.paid_cardinal_memory import (
            CARDINAL_CONTROL_STATUS,
            CARDINAL_STATUS,
            cardinal_certificate,
            cardinal_config,
            fit_cardinal_head,
        )
        from rosclaw_soccer.rsi.paid_replan_data import load_replanned_data

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, data, z = load_replanned_data(a.data_root, a.data_hash, policy)
        before = {k: v.clone() for k, v in parent.residual.state_dict().items()}
        identity = dict(
            config=cardinal_config(control=a.zero_control),
            source_head=a.expected_head,
            data_hash=a.data_hash,
            **{k: data[k] for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")},
        )
        write_once(
            a.output / "declaration.json",
            dict(
                **identity,
                data_root=str(a.data_root),
                input_file_hashes=pins,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        head, counts = fit_cardinal_head(z, control=a.zero_control)
        cert = cardinal_certificate(head, z, control=a.zero_control)
        if any(not torch.equal(v, before[k]) for k, v in parent.residual.state_dict().items()):
            raise ValueError("frozen motor ancestor changed during memory fitting")
        state = dict(
            schema="paid.cardinal_motor_memory.v1",
            **identity,
            declaration_hash=digest(a.output / "declaration.json"),
            memory=head.state_dict(),
            counts=counts,
            learning_certificate=cert,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        cp = a.output / "cardinal-memory.pt"
        torch.save(state, cp)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=CARDINAL_CONTROL_STATUS if a.zero_control else CARDINAL_STATUS,
                **identity,
                declaration_hash=state["declaration_hash"],
                input_file_hashes=pins,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                counts=counts,
                learning_certificate=cert,
                optimizer_updates=0,
                own_native_steps=0,
                parent_parameters_exact_unchanged=True,
                Fresh_fit_rows=0,
                DEV_fit_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
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
