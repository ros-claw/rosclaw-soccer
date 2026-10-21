"""Finite staged CUDA full-entry imitation over immutable routed muscle memory."""

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
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    a = p.parse_args()
    rp = a.data_root / "complete.json"
    if (
        a.output.exists()
        or digest(rp) != a.data_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output, exact TRAIN data and clean frozen learner required")
    data = json.loads(rp.read_text())
    pins = dict(data["input_file_hashes"])
    for f in (rp, a.data_root / "declaration.json", Path(data["feature_path"])):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(
        optimizer_updates=0,
        gate_optimizer_updates=0,
        head_optimizer_updates=0,
        gate_training_forward_rows=0,
        positive_forward_rows=0,
        retention_forward_rows=0,
    )
    try:
        import copy

        import numpy as np
        import torch
        import torch.nn.functional as F

        from rosclaw_soccer.rsi.paid_ancestor_router_student import calibrate_router_gate
        from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
        from rosclaw_soccer.rsi.paid_full_entry_contact import FullEntryContactResidual
        from rosclaw_soccer.rsi.paid_full_entry_student import (
            FULL_ENTRY_STATUS,
            full_entry_certificate,
            full_entry_config,
            load_full_entry_data,
            validate_full_entry_state,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, data, arrays = load_full_entry_data(a.data_root, a.data_hash, policy)
        config = full_entry_config(a.seed)
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not uuid.startswith("GPU-")
            or "," in uuid
            or torch.cuda.device_count() != 1
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one isolated UUID GPU and deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != uuid:
            raise ValueError("actual CUDA UUID differs")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = np.random.default_rng(config["seed"])
        plastic = DecoupledContactResidual()
        initial = {k: v.clone() for k, v in plastic.state_dict().items()}
        parent_before = {k: v.clone() for k, v in parent.residual.state_dict().items()}
        model = (
            FullEntryContactResidual(copy.deepcopy(parent.residual), plastic).to("cuda:0").train()
        )
        features, targets, prior = [
            torch.from_numpy(arrays[k]).cuda() for k in ("features", "target", "parent_delta")
        ]
        # Prior labels were independently recomputed CPU batch1 by the loader.
        # Training uses those exact frozen runtime outputs, not a GPU batched
        # counterfactual whose hard ancestral gate could round differently.
        negative = features[torch.from_numpy(~arrays["active"]).cuda()]
        core = np.flatnonzero(np.abs(arrays["target"]).max(axis=1) > 0.005)
        positives, protected = np.flatnonzero(arrays["teacher"]), np.flatnonzero(~arrays["teacher"])
        declaration = dict(
            config=config,
            source_head=a.expected_head,
            input_file_hashes=pins,
            data_root=str(a.data_root),
            data_hash=a.data_hash,
            feature_hash=data["feature_hash"],
            parent_hash=data["parent_hash"],
            parent_checkpoint_hash=data["parent_checkpoint_hash"],
            body_scene_hash=data["body_scene_hash"],
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            frozen_prior_source="independently_recomputed_CPU_batch1_cache",
            parent_parameters_in_optimizer=0,
            base_parameters_in_optimizer=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", declaration)
        write_once(
            a.output / "compute.json",
            dict(
                UUID=uuid,
                name=props.name,
                torch_version=str(torch.__version__),
                CUDA_version=torch.version.cuda,
                TF32=False,
                physics_steps=0,
            ),
        )
        plastic.net.requires_grad_(False)
        gate_optimizer = torch.optim.Adam(plastic.gate.parameters(), lr=config["lr"])
        for step in range(512):
            with torch.no_grad():
                hard = plastic.gate(negative)[:, 0].topk(256).indices
            pi = torch.from_numpy(rng.choice(core, 512, replace=True)).cuda()
            ni = torch.from_numpy(rng.integers(0, len(negative), size=256)).cuda()
            pl, nl = (
                plastic.gate(features[pi])[:, 0],
                plastic.gate(negative[torch.cat((hard, ni))])[:, 0],
            )
            loss = F.softplus(0.2 - pl + nl.max()).mean() + 0.1 * (
                F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
            )
            gate_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(plastic.gate.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("nonfinite gate learning")
            gate_optimizer.step()
            counts["gate_optimizer_updates"] += 1
            counts["optimizer_updates"] += 1
            counts["gate_training_forward_rows"] += len(negative) + 1024
            if (step + 1) % 128 == 0:
                event = dict(gate_step=step + 1, loss=float(loss.detach()), counts=counts.copy())
                write_once(a.output / f"gate{step + 1}.json", event)
                print(json.dumps(event), flush=True)
        calibration = calibrate_router_gate(plastic.gate, negative)
        plastic.gate.requires_grad_(False).zero_grad(set_to_none=True)
        plastic.net.requires_grad_(True)
        optimizer = torch.optim.Adam(plastic.net.parameters(), lr=config["lr"])
        active_gpu = torch.from_numpy(arrays["active"]).cuda()

        def prediction(indices):
            delta, logits = plastic.forward_with_gate(features[indices])
            return torch.where(
                logits >= 0, torch.clamp(prior[indices] + 2 * delta, -0.2, 0.2), prior[indices]
            )

        print(json.dumps(dict(stage="gate_calibrated", calibration=calibration)), flush=True)
        for step in range(1024):
            pi = torch.from_numpy(rng.choice(positives, 512, replace=True)).cuda()
            ni = torch.from_numpy(rng.choice(protected, 512, replace=True)).cuda()
            pred, retained = prediction(pi), prediction(ni)
            weights = torch.where(active_gpu[pi], 5.0, 1.0)
            loss = (weights[:, None] * (pred - prior[pi] - targets[pi]).square()).mean()
            loss = loss + 5 * (retained - prior[ni]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(plastic.net.parameters(), 1.0)
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(norm)
                or any(
                    v.grad is None or not torch.isfinite(v.grad).all()
                    for v in plastic.net.parameters()
                )
                or any(v.grad is not None for v in model.parent.parameters())
            ):
                raise ValueError("nonfinite/missing head gradient or changed parent gradient")
            optimizer.step()
            counts["head_optimizer_updates"] += 1
            counts["optimizer_updates"] += 1
            counts["positive_forward_rows"] += 512
            counts["retention_forward_rows"] += 512
            if (step + 1) % 128 == 0:
                event = dict(head_step=step + 1, loss=float(loss.detach()), counts=counts.copy())
                write_once(a.output / f"head{step + 1}.json", event)
                print(json.dumps(event), flush=True)
        candidate = copy.deepcopy(model).cpu().eval()
        if any(
            not torch.equal(v, parent_before[k]) for k, v in candidate.parent.state_dict().items()
        ):
            raise ValueError("frozen routed muscle memory changed")
        certificate = full_entry_certificate(candidate, arrays)

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
                schema="paid.full_entry_student.v1",
                **identity,
                declaration_hash=digest(a.output / "declaration.json"),
                plastic=candidate.plastic.state_dict(),
                initial_plastic=initial,
                optimizer=optimizer.state_dict(),
                gate_optimizer=gate_optimizer.state_dict(),
                calibration=calibration,
                retention_certificate=certificate,
                counts=counts,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
        receipt = dict(
            status=FULL_ENTRY_STATUS,
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
            DEV_fit_rows=0,
            own_native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        validate_full_entry_state(state, receipt, initial)
        cp = a.output / "full-entry-student.pt"
        torch.save(state, cp)
        check_pins(pins)
        receipt.update(checkpoint_path=str(cp), checkpoint_hash=digest(cp))
        write_once(a.output / "complete.json", receipt)
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                own_native_steps=0,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
