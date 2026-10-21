"""Three-seed frozen-base neural student, guarded gate retention, offline CUDA."""

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
    p.add_argument("--staged", action="store_true")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rp, dp = a.data_root / "complete.json", a.data_root / "declaration.json"
    if (
        a.output.exists()
        or digest(rp) != a.data_hash
        or (a.data_root / "failed.json").exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen TRAIN curation required")
    data = json.loads(rp.read_text())
    if (
        data["status"] != "PAID_SUCCESSOR_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"
        or data["declaration_hash"] != digest(dp)
        or data["bank"] != "train"
        or data["DEV_episodes_fit"] != 0
        or data["protected_successes_by_kick"] != [16, 14, 6]
        or len(data["trajectories"]) != 40
        or data["CPU_feature_recompute_rows"] != data["rows"]
        or data["optimizer_updates"] != 0
        or data["own_native_steps"] != 0
        or data["hardware_authorized"] is not False
    ):
        raise ValueError(
            "complete causal TRAIN-only features and protected success evidence required"
        )
    path = Path(data["feature_path"])
    if path.is_symlink() or path.parent != a.data_root or digest(path) != data["feature_hash"]:
        raise ValueError("exact confined numeric feature archive required")
    pins = dict(data["input_file_hashes"])
    pins.update({str(rp): a.data_hash, str(dp): digest(dp), str(path): data["feature_hash"]})
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(
        optimizer_updates=0,
        positive_forward_rows=0,
        retention_forward_rows=0,
        projection_calls=0,
        projection_forward_rows=0,
        projection_gate_evaluations=0,
    )
    projection_histogram = {}
    try:
        import copy

        import numpy as np
        import torch
        import torch.nn.functional as F

        from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual
        from rosclaw_soccer.rsi.paid_gate_retention import project_contact_gate
        from rosclaw_soccer.rsi.paid_successor_features import (
            staged_successor_config,
            successor_student_config,
        )

        config = staged_successor_config(a.seed) if a.staged else successor_student_config(a.seed)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not visible.startswith("GPU-")
            or "," in visible
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.cuda.device_count() != 1
        ):
            raise ValueError("one isolated UUID GPU and deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("actual device differs from declared isolated UUID")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = np.random.default_rng(config["seed"])
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        x, y, teacher, active = (arrays[k] for k in ("features", "target", "teacher", "active"))
        n = data["rows"]
        if (
            x.shape != (n, 288)
            or y.shape != (n, 29)
            or x.dtype != np.float32
            or y.dtype != np.float32
            or teacher.shape != (n,)
            or active.shape != (n,)
            or teacher.dtype != np.bool_
            or active.dtype != np.bool_
            or not all(np.isfinite(v).all() for v in arrays.values())
            or abs(y).max() > 0.20001
            or y[~teacher].any()
            or not np.array_equal(active, np.abs(y).max(axis=1) > 1e-6)
            or int(active.sum()) != data["active_rows"]
            or int(teacher.sum()) != data["teacher_rows"]
        ):
            raise ValueError(
                "finite true-zero protected labels and original-only288 features required"
            )
        model = ContactResidual().to("cuda:0")
        features, targets = torch.from_numpy(x).to("cuda:0"), torch.from_numpy(y).to("cuda:0")
        negative = features[torch.from_numpy(~active).to("cuda:0")]
        positive_ids, negative_ids = np.flatnonzero(teacher), np.flatnonzero(~teacher)
        if len(negative) != data["negative_support_rows"]:
            raise ValueError("complete negative functional support required")
        with torch.no_grad():
            if model.eval()(features).count_nonzero():
                raise ValueError("initial head must leave all recorded base proposals exact")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        declaration = dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            data_root=str(a.data_root),
            data_hash=a.data_hash,
            features_hash=data["feature_hash"],
            base_wrapper_checkpoint_hash=data["base_wrapper_checkpoint_hash"],
            base_wrapper_training_receipt_hash=data["base_wrapper_training_receipt_hash"],
            body_scene_hash=data["body_scene_hash"],
            feature_rows=n,
            negative_support_rows=len(negative),
            teacher_rows=data["teacher_rows"],
            DEV_episodes_fit=0,
            runtime_case_lookup=False,
            base_parameters_in_optimizer=0,
            native_steps=0,
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

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        initial = {k: cpu(v) for k, v in model.state_dict().items()}
        gate_optimizer = None
        calibration = None
        if a.staged:
            from rosclaw_soccer.rsi.paid_successor_calibration import calibrate_successor_gate

            counts.update(
                gate_optimizer_updates=0, head_optimizer_updates=0, gate_training_forward_rows=0
            )
            core_ids = np.flatnonzero(teacher & (np.abs(y).max(axis=1) > 0.005))
            if not len(core_ids):
                raise ValueError("actual material teacher correction support required")
            model.net.requires_grad_(False)
            gate_optimizer = torch.optim.Adam(model.gate.parameters(), lr=0.0003)
            for _step in range(512):
                with torch.no_grad():
                    hard = model.gate(negative)[:, 0].topk(min(256, len(negative))).indices
                    counts["gate_training_forward_rows"] += len(negative)
                pos = rng.choice(core_ids, 512, replace=True)
                random_neg = torch.from_numpy(rng.integers(0, len(negative), size=256)).to("cuda:0")
                px = features[torch.from_numpy(pos).to("cuda:0")]
                nx = negative[torch.cat((hard, random_neg))]
                pl, nl = model.gate(px)[:, 0], model.gate(nx)[:, 0]
                ranking = F.softplus(0.2 - pl + nl.max()).mean()
                classification = F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                classification += F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
                loss = ranking + 0.1 * classification
                gate_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise ValueError("nonfinite gate ranking training")
                gate_optimizer.step()
                counts["optimizer_updates"] += 1
                counts["gate_optimizer_updates"] += 1
                counts["gate_training_forward_rows"] += len(px) + len(nx)
            calibration = calibrate_successor_gate(model.gate, negative)
            model.gate.requires_grad_(False)
            model.gate.zero_grad(set_to_none=True)
            model.net.requires_grad_(True)
            optimizer = torch.optim.Adam(model.net.parameters(), lr=0.0003)
            print(json.dumps(dict(stage="gate_calibrated", calibration=calibration)), flush=True)
        history = []
        for epoch in range(256):
            last_loss = None
            for _ in range(4):
                pos = rng.choice(positive_ids, 512, replace=len(positive_ids) < 512)
                neg = rng.choice(negative_ids, 512, replace=len(negative_ids) < 512)
                pindex, nindex = (
                    torch.from_numpy(pos).to("cuda:0"),
                    torch.from_numpy(neg).to("cuda:0"),
                )
                pp, pl = model.forward_with_gate(features[pindex])
                npred, nl = model.forward_with_gate(features[nindex])
                if a.staged:
                    pp = torch.where(pl >= 0, pp, torch.zeros_like(pp))
                    npred = torch.where(nl >= 0, npred, torch.zeros_like(npred))
                mask = torch.from_numpy(active[pos]).to("cuda:0")
                weights = torch.where(mask, 5.0, 1.0)
                imitation = (weights[:, None] * (pp - targets[pindex]).square()).mean()
                retention = npred.square().mean()
                gate_loss = F.binary_cross_entropy_with_logits(
                    pl[:, 0], mask.float(), pos_weight=torch.tensor(5.0, device="cuda:0")
                )
                gate_loss += F.binary_cross_entropy_with_logits(
                    nl[:, 0], torch.zeros_like(nl[:, 0])
                )
                hard_negative = torch.relu(nl + 0.5).square().mean()
                loss = imitation + 5.0 * retention + 0.001 * gate_loss + 0.005 * hard_negative
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite student loss; no repair")
                previous = {k: p.detach().clone() for k, p in model.gate.named_parameters()}
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if not torch.isfinite(norm) or any(
                    p.grad is None or not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.requires_grad
                ):
                    raise ValueError("nonfinite or missing plastic-head gradient")
                optimizer.step()
                projected = (
                    None if a.staged else project_contact_gate(model.gate, previous, negative)
                )
                counts["optimizer_updates"] += 1
                counts["positive_forward_rows"] += 512
                counts["retention_forward_rows"] += 512
                if a.staged:
                    counts["head_optimizer_updates"] += 1
                else:
                    counts["projection_calls"] += 1
                    counts["projection_forward_rows"] += projected["gate_forward_rows"]
                    counts["projection_gate_evaluations"] += projected["gate_evaluations"]
                    fraction = str(projected["accepted_fraction"])
                    projection_histogram[fraction] = projection_histogram.get(fraction, 0) + 1
                last_loss = float(loss.detach())
            if (epoch + 1) % 32 == 0:
                event = dict(
                    epoch=epoch + 1,
                    loss=last_loss,
                    counts=counts.copy(),
                    accepted_fractions=projection_histogram.copy(),
                )
                write_once(a.output / f"epoch{epoch + 1}.json", event)
                history.append(event)
                print(json.dumps(dict(epoch=epoch + 1, loss=last_loss)), flush=True)
        candidate = copy.deepcopy(model).cpu().eval()
        with torch.no_grad():
            fx = torch.from_numpy(x)
            logits = candidate.gate(fx)
            pred = candidate(fx)
            maximum = float(logits[torch.from_numpy(~active)].max())
            if maximum > -0.1 + 1e-5 or pred[torch.from_numpy(~active)].count_nonzero():
                raise ValueError("CPU functional retention certificate failed")
            metric = dict(
                active_rows=data["active_rows"],
                active_gate_open_rows=int((logits[:, 0][torch.from_numpy(active)] >= 0).sum()),
                active_joint_MSE=float(
                    (pred[torch.from_numpy(active)] - torch.from_numpy(y[active])).square().mean()
                ),
                maximum_negative_logit_CPU=maximum,
                all_negative_predictions_exact_zero=True,
            )
        state = cpu(
            dict(
                schema="soccer.successor_contact_student.v2"
                if a.staged
                else "soccer.successor_contact_student.v1",
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                residual=candidate.state_dict(),
                initial_residual=initial,
                optimizer=optimizer.state_dict(),
                **(
                    dict(gate_optimizer=gate_optimizer.state_dict(), gate_calibration=calibration)
                    if a.staged
                    else {}
                ),
                counts=counts.copy(),
                projection_histogram=projection_histogram,
                retention_certificate=metric,
                data_hash=a.data_hash,
                features_hash=data["feature_hash"],
                base_wrapper_checkpoint_hash=data["base_wrapper_checkpoint_hash"],
                base_wrapper_training_receipt_hash=data["base_wrapper_training_receipt_hash"],
                body_scene_hash=data["body_scene_hash"],
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
        checkpoint = a.output / "successor-student-epoch256.pt"
        torch.save(state, checkpoint)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_NEURAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                checkpoint_path=str(checkpoint),
                checkpoint_hash=digest(checkpoint),
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                data_hash=a.data_hash,
                features_hash=data["feature_hash"],
                base_wrapper_checkpoint_hash=data["base_wrapper_checkpoint_hash"],
                base_wrapper_training_receipt_hash=data["base_wrapper_training_receipt_hash"],
                body_scene_hash=data["body_scene_hash"],
                counts=counts,
                projection_histogram=projection_histogram,
                retention_certificate=metric,
                plastic_parameters_changed=any(
                    not torch.equal(initial[k], v) for k, v in candidate.state_dict().items()
                ),
                base_parameters_in_optimizer=0,
                **(dict(gate_calibration=calibration) if a.staged else {}),
                DEV_episodes_fit=0,
                own_native_steps=0,
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
                counts=counts,
                hardware_authorized=False,
                own_native_steps=0,
            ),
        )
        raise


if __name__ == "__main__":
    main()
