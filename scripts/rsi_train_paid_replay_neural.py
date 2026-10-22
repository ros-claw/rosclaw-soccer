"""Train one preregistered soft-replay neural motor pilot; no runtime admission."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--seed-index", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--tail-replay", action="store_true")
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
            raise ValueError("clean frozen neural fit source required")

    clean()
    if a.output.exists():
        raise ValueError("unused neural pilot output; no automatic retries")
    rp = a.data_root / "complete.json"
    if rp.is_symlink() or digest(rp) != a.data_hash:
        raise ValueError("exact reconstructed growth TRAIN required")
    r = json.loads(rp.read_text())
    pins = dict(r["input_file_hashes"])
    for path in (rp, a.data_root / "declaration.json"):
        pins[str(path)] = digest(path)
    pins[r["feature_path"]] = r["feature_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    actual = 0
    try:
        import torch

        from rosclaw_soccer.rsi.paid_growth_memory_data import load_growth_data
        from rosclaw_soccer.rsi.paid_replay_neural import (
            FIT_STATUS,
            fit_replay_head,
            replay_certificate,
            replay_config,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if a.device != "cpu":
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("one explicitly assigned free CUDA device required")
            free, _ = torch.cuda.mem_get_info()
            if free < 4 * 1024**3:
                raise ValueError("insufficient free GPU capacity; do not stop other workloads")
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, data, z = load_growth_data(a.data_root, a.data_hash, policy)
        before = {k: v.clone() for k, v in parent.state_dict().items()}
        identity = dict(
            config=replay_config(a.seed_index, tail_replay=a.tail_replay),
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
                device=a.device,
                physical_rollout_budget=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )

        def progress(row):
            nonlocal actual
            actual = row["completed_updates"]
            write_once(a.output / f"update-{actual}.json", row)
            print(json.dumps(row), flush=True)

        head, initial, counts, optimizer = fit_replay_head(
            z, a.seed_index, a.device, progress, tail_replay=a.tail_replay
        )
        cert = replay_certificate(head, z)
        if any(not torch.equal(v, before[k]) for k, v in parent.state_dict().items()):
            raise ValueError("frozen motor parent changed")
        cp = a.output / "soft-replay-neural.pt"
        torch.save(
            dict(
                schema="paid.soft_replay_neural.offline.v1",
                **identity,
                model=head.state_dict(),
                initial_model=initial,
                optimizer=optimizer,
                counts=counts,
                certificate=cert,
                declaration_hash=digest(a.output / "declaration.json"),
            ),
            cp,
        )
        # Safe owned artifact reload, never an arbitrary pickle payload.
        loaded = torch.load(cp, map_location="cpu", weights_only=True)
        if any(not torch.equal(v, loaded["model"][k]) for k, v in head.state_dict().items()):
            raise ValueError("weights-only reload differs")
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status=FIT_STATUS,
                **identity,
                input_file_hashes=pins,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                declaration_hash=digest(a.output / "declaration.json"),
                counts=counts,
                certificate=cert,
                device=a.device,
                optimizer_updates=counts["completed_updates"],
                own_native_steps=0,
                weights_only_checkpoint_reload=True,
                parent_parameters_exact_unchanged=True,
                runtime_admitted=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed_update_checkpoint=actual,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
