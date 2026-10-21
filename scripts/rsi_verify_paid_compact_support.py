"""Independent safe-load, TRAIN geometry and CPU batch1 readback before SIM exams."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-root", type=Path, required=True)
    p.add_argument("--fit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp = a.fit_root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.fit_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused verification, exact fit and frozen source required")
    r = json.loads(rp.read_text())
    pins = dict(r["input_file_hashes"])
    for f in (rp, a.fit_root / "declaration.json", Path(r["checkpoint_path"])):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    declaration = dict(
        source_head=a.expected_head,
        fit_root=str(a.fit_root),
        fit_hash=a.fit_hash,
        input_file_hashes=pins,
        Fresh_rows_read=0,
        own_native_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration)
    try:
        import torch

        from rosclaw_soccer.rsi.paid_compact_student import (
            AUDITED_STATUS,
            audit_compact_batch1,
            load_compact_fit,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        _, head, fit, z = load_compact_fit(a.fit_root, a.fit_hash, policy)
        certificate = audit_compact_batch1(head, fit, z)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                **declaration,
                status=AUDITED_STATUS,
                declaration_hash=digest(a.output / "declaration.json"),
                **{
                    k: fit[k]
                    for k in (
                        "checkpoint_path",
                        "checkpoint_hash",
                        "config",
                        "body_scene_hash",
                        "data_hash",
                    )
                },
                batch1_certificate=certificate,
                CPU_direct_distance_pairs=int(z["teacher"].sum()) * int((~z["teacher"]).sum()),
                own_CPU_cached_parent_verification_rows=len(z["features"]),
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
