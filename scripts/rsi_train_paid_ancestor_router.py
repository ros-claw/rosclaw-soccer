"""Bounded offline competence gate fit, followed by exact CPU retention checks."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-hash", required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp = a.data_root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.data_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, exact TRAIN data and clean pinned source required")
    data = json.loads(rp.read_text())
    pins = dict(data["input_file_hashes"])
    for path in (rp, a.data_root / "declaration.json", Path(data["feature_path"])):
        pins[str(path)] = digest(path)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(
        optimizer_updates=0, positive_rows=0, negative_rows=0, gate_training_forward_rows=0
    )
    try:
        import copy

        import numpy as np
        import torch
        import torch.nn.functional as F

        from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter
        from rosclaw_soccer.rsi.paid_ancestor_router_student import (
            ROUTER_STATUS,
            calibrate_router_gate,
            load_router_data,
            router_certificate,
            router_config,
            validate_router_state,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        parent, data, arrays = load_router_data(a.data_root, a.data_hash, policy)
        config = router_config(a.seed)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not visible.startswith("GPU-")
            or "," in visible
            or torch.cuda.device_count() != 1
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one isolated UUID GPU and deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("actual CUDA device differs from declaration")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = np.random.default_rng(config["seed"])
        model = AncestralResidualRouter(copy.deepcopy(parent.residual))
        initial = {k: v.clone() for k, v in model.gate.state_dict().items()}
        parent_before = {k: v.clone() for k, v in model.parent.state_dict().items()}
        model = model.to("cuda:0").train()
        features = torch.from_numpy(arrays["features"]).cuda()
        negative = features[torch.from_numpy(arrays["protected_nonzero"]).cuda()]
        core = np.flatnonzero(arrays["positive_core"])
        if len(negative) < 256 or not len(core):
            raise ValueError("full hard-negative and positive-core supports required")
        declaration = dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            data_root=str(a.data_root),
            data_hash=a.data_hash,
            feature_hash=data["feature_hash"],
            parent_hash=data["parent_hash"],
            parent_checkpoint_hash=data["parent_checkpoint_hash"],
            body_scene_hash=data["body_scene_hash"],
            Fresh_fit_rows=0,
            parent_parameters_in_optimizer=0,
            base_parameters_in_optimizer=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", declaration)
        write_once(
            a.output / "compute.json",
            dict(
                UUID=visible,
                name=props.name,
                torch_version=str(torch.__version__),
                CUDA_version=torch.version.cuda,
                TF32=False,
                physics_steps=0,
            ),
        )
        optimizer = torch.optim.Adam(model.gate.parameters(), lr=config["lr"])
        for step in range(config["steps"]):
            with torch.no_grad():
                hard = model.gate(negative)[:, 0].topk(256).indices
            pi = torch.from_numpy(rng.choice(core, 512, replace=True)).cuda()
            ni = torch.from_numpy(rng.integers(0, len(negative), size=256)).cuda()
            pl = model.gate(features[pi])[:, 0]
            nl = model.gate(negative[torch.cat((hard, ni))])[:, 0]
            loss = F.softplus(0.2 - pl + nl.max()).mean() + 0.1 * (
                F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0)
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(norm)
                or any(
                    v.grad is None or not torch.isfinite(v.grad).all()
                    for v in model.gate.parameters()
                )
                or any(v.grad is not None for v in model.parent.parameters())
            ):
                raise ValueError("nonfinite/missing gate gradient or frozen-parent gradient")
            optimizer.step()
            counts["optimizer_updates"] += 1
            counts["positive_rows"] += len(pl)
            counts["negative_rows"] += len(nl)
            counts["gate_training_forward_rows"] += len(negative) + len(pl) + len(nl)
            if (step + 1) % 64 == 0:
                event = dict(step=step + 1, loss=float(loss.detach()), counts=counts.copy())
                write_once(a.output / f"step{step + 1}.json", event)
                print(json.dumps(event), flush=True)
        calibration = calibrate_router_gate(model.gate, negative)
        candidate = copy.deepcopy(model).cpu().eval()
        if any(
            not torch.equal(v, parent_before[k]) for k, v in candidate.parent.state_dict().items()
        ):
            raise ValueError("frozen muscle memory changed")
        certificate = router_certificate(candidate, arrays)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        identity = {
            k: declaration[k]
            for k in (
                "config",
                "source_head",
                "data_hash",
                "feature_hash",
                "parent_hash",
                "parent_checkpoint_hash",
                "body_scene_hash",
            )
        }
        state = cpu(
            dict(
                schema="paid.ancestral_router_student.v1",
                **identity,
                declaration_hash=digest(a.output / "declaration.json"),
                gate=candidate.gate.state_dict(),
                initial_gate=initial,
                optimizer=optimizer.state_dict(),
                calibration=calibration,
                retention_certificate=certificate,
                counts=counts,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
        receipt = dict(
            status=ROUTER_STATUS,
            **identity,
            declaration_hash=state["declaration_hash"],
            input_file_hashes=pins,
            calibration=calibration,
            retention_certificate=certificate,
            counts=counts,
            parent_parameters_in_optimizer=0,
            base_parameters_in_optimizer=0,
            parent_parameters_exact_unchanged=True,
            Fresh_fit_rows=0,
            own_native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        validate_router_state(state, receipt, initial)
        path = a.output / "ancestor-router.pt"
        torch.save(state, path)
        check_pins(pins)
        receipt.update(checkpoint_path=str(path), checkpoint_hash=digest(path))
        write_once(a.output / "complete.json", receipt)
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                hardware_authorized=False,
                own_native_steps=0,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
