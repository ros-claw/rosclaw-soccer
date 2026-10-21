"""Admit a trainable research actor against every recorded teacher NN row.

Two tests, all episodes: single-step recorded-state parity and free recurrent
history on recorded observations. Neither is a new closed-loop physics test.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    a = p.parse_args()
    receipt = a.root / "complete.json"
    if (
        a.output.exists()
        or digest(receipt) != a.hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen source/completed teacher required")
    r = json.loads(receipt.read_text())
    if (
        r["status"] != "PAID_RESEARCH_NATIVE_TEACHER_PROBE_NOT_PROMOTED"
        or not r["whole_repeats_exact"]
    ):
        raise ValueError("complete repeated research teacher required")
    pins = {**r["input_file_hashes"], str(receipt): a.hash}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    for relative, expected in r["raw_hashes"].items():
        pins[str(a.root / relative)] = expected
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            expected_head=a.expected_head,
            absolute_tolerance=0.0002,
            maximum_new_neural_rows=r["neural_rows"] * 2,
            native_steps=0,
            optimizer_updates=0,
            commercial_publicity_authorized=False,
            single_step_and_free_history=True,
            physics_equivalence_proven=False,
            inference_device=a.device,
        ),
    )
    import torch

    from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(a.device)
    compute = dict(device=a.device, physics_backend="none", distributed_training=False)
    if device.type == "cuda":
        import os

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.startswith("GPU-") or "," in visible or torch.cuda.device_count() != 1:
            raise ValueError("one explicitly UUID-isolated GPU required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("predeclared deterministic CUDA workspace required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("visible CUDA device does not match the declared physical UUID")
        compute.update(
            visible_uuid=visible,
            actual_uuid=str(props.uuid),
            name=props.name,
            total_memory_bytes=props.total_memory,
            tf32=False,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    write_once(a.output / "compute.json", compute)
    paths = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
    if len(paths) != 1:
        raise ValueError("one pinned full teacher required")
    model = PaidTrainableActor(paths[0], expected_hash=PUBLISHED_HASH).to(device).eval()
    counts, results = 0, []
    try:
        with torch.no_grad():
            for row in r["results"]:
                path = a.root / f"{row['profile']}/case{row['case']}-repeat{row['repeat']}.npz"
                with np.load(path, allow_pickle=False) as z:
                    obs = torch.from_numpy(z["NN_observation"].copy()).to(device)
                    actions = torch.from_numpy(z["NN_action"].copy())
                    h0, c0 = [
                        torch.from_numpy(z["NN_" + n + "_before"].copy())
                        .squeeze(2)
                        .transpose(0, 1)
                        .contiguous()
                        .to(device)
                        for n in ("h", "c")
                    ]
                    hr, cr = [
                        torch.from_numpy(z["NN_" + n + "_after"].copy())
                        .squeeze(2)
                        .transpose(0, 1)
                        .contiguous()
                        for n in ("h", "c")
                    ]
                maximum = dict(
                    recorded_action=0.0,
                    recorded_h=0.0,
                    recorded_c=0.0,
                    recurrent_action=0.0,
                    recurrent_h=0.0,
                    recurrent_c=0.0,
                )

                def compare(key, x, reference, path=path, maximum=maximum):
                    reference = reference.to(device)
                    error = float((x - reference).abs().max())
                    if not torch.isfinite(x).all() or error > 0.0002:
                        raise ValueError(f"{key} parity failed at {path}: {error}")
                    maximum[key] = max(maximum[key], error)

                for start in range(0, len(obs), 64):
                    end = min(start + 64, len(obs))
                    ac, (h, c) = model(
                        obs[start:end].unsqueeze(0), (h0[:, start:end], c0[:, start:end])
                    )
                    counts += end - start
                    compare("recorded_action", ac[0], actions[start:end])
                    compare("recorded_h", h, hr[:, start:end])
                    compare("recorded_c", c, cr[:, start:end])
                state = (
                    torch.zeros(2, 1, 128, device=device),
                    torch.zeros(2, 1, 128, device=device),
                )
                for frame in range(len(obs)):
                    ac, state = model(obs[frame].reshape(1, 1, 160), state)
                    counts += 1
                    compare("recurrent_action", ac[0, 0], actions[frame])
                    compare("recurrent_h", state[0], hr[:, frame : frame + 1])
                    compare("recurrent_c", state[1], cr[:, frame : frame + 1])
                results.append(
                    dict(
                        profile=row["profile"],
                        case=row["case"],
                        repeat=row["repeat"],
                        maximum_absolute_errors=maximum,
                    )
                )
                print(json.dumps(results[-1]), flush=True)
        if counts != r["neural_rows"] * 2:
            raise ValueError("all recorded neural rows must be counted in both modes")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "PAID_TRAINABLE_RECORDED_OBSERVATION_PARITY_NOT_PHYSICS_QUALIFICATION"
                    if device.type == "cpu"
                    else "PAID_TRAINABLE_CUDA_RECORDED_OBSERVATION_PARITY_NOT_PHYSICS_QUALIFICATION"
                ),
                compute=compute,
                input_file_hashes=pins,
                results=results,
                new_neural_rows=counts,
                native_steps=0,
                optimizer_updates=0,
                physics_equivalence_proven=False,
                commercial_publicity_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
                hardware_authorized=False,
                method_sources=[
                    "https://onnx.ai/onnx/operators/onnx__LSTM.html",
                    "https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html",
                ],
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), new_neural_rows=counts, native_steps=0, optimizer_updates=0),
        )
        raise


if __name__ == "__main__":
    main()
