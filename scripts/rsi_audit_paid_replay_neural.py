"""Wait for one fixed neural seed and independently verify its complete fit."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-root", type=Path, required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-for-completion", action="store_true")
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
            raise ValueError("clean frozen independent verifier required")

    clean()
    if a.output.exists():
        raise ValueError("unused independent audit evidence required")
    rp = a.fit_root / "complete.json"
    deadline = time.monotonic() + 14400
    while not rp.exists():
        clean()
        if (a.fit_root / "failed.json").exists():
            raise ValueError("producer failed; never automatic retry")
        if not a.wait_for_completion or time.monotonic() > deadline:
            raise TimeoutError("bounded fixed-producer completion wait")
        time.sleep(10)
    r = json.loads(rp.read_text())
    if r["source_head"] != a.producer_head:
        raise ValueError("fixed predeclared producer source required")
    pins = dict(r["input_file_hashes"])
    for p in (rp, a.fit_root / "declaration.json", Path(r["checkpoint_path"])):
        pins[str(p)] = digest(p)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import torch

        from rosclaw_soccer.rsi.paid_replay_neural_student import AUDITED_STATUS, load_replay_fit

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        _, _, own = load_replay_fit(a.fit_root, digest(rp), policy)
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=AUDITED_STATUS,
                source_head=a.expected_head,
                producer_head=a.producer_head,
                input_file_hashes=pins,
                fit_root=str(a.fit_root),
                fit_hash=digest(rp),
                config=own["config"],
                certificate=own["certificate"],
                counts=own["counts"],
                checkpoint_path=own["checkpoint_path"],
                checkpoint_hash=own["checkpoint_hash"],
                body_scene_hash=own["body_scene_hash"],
                parent_hash=own["parent_hash"],
                own_native_steps=0,
                optimizer_updates=0,
                CPU_fit_reconstruction=True,
                runtime_admitted=False,
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
