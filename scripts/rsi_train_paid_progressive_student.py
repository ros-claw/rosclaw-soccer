"""Offline CUDA progressive fit with exact acquired-parent fallback; SIM only."""

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
        raise ValueError("unused output, exact data and clean frozen trainer required")
    data = json.loads(rp.read_text())
    pins = dict(data["input_file_hashes"])
    for path in (rp, a.data_root / "declaration.json", Path(data["feature_path"])):
        pins[str(path)] = digest(path)
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

        from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
        from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual
        from rosclaw_soccer.rsi.paid_progressive_student import (
            TRAINED_STATUS,
            load_progressive_data,
            progressive_config,
            validate_progressive_state,
        )
        from rosclaw_soccer.rsi.paid_successor_calibration import calibrate_successor_gate

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(path) for path in pins if path.endswith("/ckp/policy_30000.onnx"))
        parent_actor, data, arrays = load_progressive_data(a.data_root, a.data_hash, policy)
        config = progressive_config(a.seed)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not visible.startswith("GPU-")
            or "," in visible
            or torch.cuda.device_count() != 1
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one isolated UUID GPU with deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("actual training GPU differs from declaration")
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
        parent_before = {k: v.clone() for k, v in parent_actor.residual.state_dict().items()}
        model = (
            ProgressiveContactResidual(copy.deepcopy(parent_actor.residual), plastic)
            .to("cuda:0")
            .train()
        )
        x, y = (arrays[k] for k in ("features", "target"))
        teacher, active = (arrays[k] for k in ("teacher", "active"))
        features, targets = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        negative = features[torch.from_numpy(~active).cuda()]
        with torch.no_grad():
            prior = model.parent(features)
            if not torch.equal(model(features), prior):
                raise ValueError("new plastic head must initially preserve all parent actions")
        declaration = dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            data_root=str(a.data_root),
            data_hash=a.data_hash,
            features_hash=data["feature_hash"],
            parent_hash=data["parent_hash"],
            parent_checkpoint_hash=data["parent_checkpoint_hash"],
            body_scene_hash=data["body_scene_hash"],
            feature_rows=data["rows"],
            negative_support_rows=data["negative_support_rows"],
            parent_parameters_in_optimizer=0,
            base_parameters_in_optimizer=0,
            DEV_episodes_fit=0,
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
        core_ids = np.flatnonzero(teacher & (np.abs(y).max(axis=1) > 0.005))
        if not len(core_ids):
            raise ValueError("nontrivial new teacher core required")
        plastic.net.requires_grad_(False)
        gate_optimizer = torch.optim.Adam(plastic.gate.parameters(), lr=config["lr"])
        for _ in range(512):
            with torch.no_grad():
                hard = plastic.gate(negative)[:, 0].topk(min(256, len(negative))).indices
                counts["gate_training_forward_rows"] += len(negative)
            pos = torch.from_numpy(rng.choice(core_ids, 512, replace=True)).cuda()
            random_neg = torch.from_numpy(rng.integers(0, len(negative), size=256)).cuda()
            px, nx = features[pos], negative[torch.cat((hard, random_neg))]
            pl, nl = plastic.gate(px)[:, 0], plastic.gate(nx)[:, 0]
            ranking = F.softplus(0.2 - pl + nl.max()).mean()
            classification = F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
            classification += F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
            loss = ranking + 0.1 * classification
            gate_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(plastic.gate.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("nonfinite learned-parent gate training")
            gate_optimizer.step()
            counts["gate_optimizer_updates"] += 1
            counts["optimizer_updates"] += 1
            counts["gate_training_forward_rows"] += len(px) + len(nx)
        calibration = calibrate_successor_gate(plastic.gate, negative)
        plastic.gate.requires_grad_(False).zero_grad(set_to_none=True)
        plastic.net.requires_grad_(True)
        optimizer = torch.optim.Adam(plastic.net.parameters(), lr=config["lr"])
        print(json.dumps(dict(stage="gate_calibrated", calibration=calibration)), flush=True)
        positive_ids, protect_ids = np.flatnonzero(teacher), np.flatnonzero(~teacher)
        for step in range(1024):
            pi = torch.from_numpy(rng.choice(positive_ids, 512, replace=True)).cuda()
            ni = torch.from_numpy(rng.choice(protect_ids, 512, replace=True)).cuda()
            pred, retained = model(features[pi]), model(features[ni])
            weights = torch.where(torch.from_numpy(active).cuda()[pi], 5.0, 1.0)
            imitation = (weights[:, None] * (pred - prior[pi] - targets[pi]).square()).mean()
            loss = imitation + 5.0 * (retained - prior[ni]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(plastic.net.parameters(), 1.0)
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(norm)
                or any(
                    p.grad is None or not torch.isfinite(p.grad).all()
                    for p in plastic.net.parameters()
                )
            ):
                raise ValueError("nonfinite or missing progressive action-head gradients")
            optimizer.step()
            counts["head_optimizer_updates"] += 1
            counts["optimizer_updates"] += 1
            counts["positive_forward_rows"] += 512
            counts["retention_forward_rows"] += 512
            if (step + 1) % 128 == 0:
                event = dict(head_step=step + 1, loss=float(loss.detach()), counts=counts.copy())
                write_once(a.output / f"step{step + 1}.json", event)
                print(json.dumps(dict(head_step=step + 1, loss=float(loss.detach()))), flush=True)
        candidate = copy.deepcopy(model).cpu().eval()
        if any(
            not torch.equal(v, parent_before[k]) for k, v in candidate.parent.state_dict().items()
        ):
            raise ValueError("frozen learned parent changed")
        with torch.no_grad():
            fx, mask = torch.from_numpy(x), torch.from_numpy(~active)
            parent_pred = candidate.parent(fx)
            pred, logits = candidate(fx), candidate.plastic.gate(fx)
            maximum = float(logits[mask].max())
            if maximum > -0.1 or not torch.equal(pred[mask], parent_pred[mask]):
                raise ValueError("CPU acquired-parent exact retention certificate failed")
            certificate = dict(
                maximum_negative_logit_CPU=maximum,
                all_negative_predictions_exact_parent=True,
                all_negative_corrections_exact_zero=True,
                negative_support_rows=int(mask.sum()),
                active_rows=int(active.sum()),
                active_gate_open_rows=int((logits[~mask] >= 0).sum()),
                active_correction_MSE=float(
                    (pred[~mask] - parent_pred[~mask] - torch.from_numpy(y[active])).square().mean()
                ),
            )

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        identity = {
            k: data[k]
            for k in (
                "parent_hash",
                "parent_checkpoint_hash",
                "body_scene_hash",
                "base_wrapper_checkpoint_hash",
                "base_wrapper_training_receipt_hash",
            )
        }
        state = cpu(
            dict(
                schema="soccer.progressive_contact_student.v1",
                config=config,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                data_hash=a.data_hash,
                features_hash=data["feature_hash"],
                **identity,
                plastic=candidate.plastic.state_dict(),
                initial_plastic=initial,
                optimizer=optimizer.state_dict(),
                gate_optimizer=gate_optimizer.state_dict(),
                gate_calibration=calibration,
                retention_certificate=certificate,
                counts=counts,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
        receipt = {
            k: state[k]
            for k in (
                "config",
                "source_head",
                "declaration_hash",
                "data_hash",
                "features_hash",
                "parent_hash",
                "parent_checkpoint_hash",
                "body_scene_hash",
                "base_wrapper_checkpoint_hash",
                "base_wrapper_training_receipt_hash",
                "gate_calibration",
                "retention_certificate",
                "counts",
            )
        }
        receipt.update(
            status=TRAINED_STATUS,
            input_file_hashes=pins,
            parent_parameters_in_optimizer=0,
            base_parameters_in_optimizer=0,
            parent_parameters_exact_unchanged=True,
            DEV_episodes_fit=0,
            own_native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        validate_progressive_state(state, receipt, initial)
        path = a.output / "progressive-student.pt"
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
