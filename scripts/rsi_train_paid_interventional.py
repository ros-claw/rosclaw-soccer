"""Three finite warm neural updates plus an explicit zero-update geometry control."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "warm"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--geometry-control", action="store_true")
    p.add_argument("--generation", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.geometry_control and a.seed != 0:
        p.error("one fixed seed0 geometry control required")
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused learner output and clean frozen source required")
    pins = {}
    for name in ("data", "warm"):
        root = getattr(a, name + "_root")
        rp = root / "complete.json"
        if rp.is_symlink() or digest(rp) != getattr(a, name + "_hash"):
            raise ValueError("exact closed data and primary warm source required")
        r = json.loads(rp.read_text())
        for path, h in r["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting learner source lineage")
            pins[path] = h
        for f in (rp, root / "declaration.json"):
            pins[str(f)] = digest(f)
        if name == "data":
            f = Path(r["feature_path"])
            pins[str(f)] = r["feature_hash"]
        else:
            f = Path(r["checkpoint_path"])
            pins[str(f)] = r["checkpoint_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(
        optimizer_updates=0,
        inherited_optimizer_updates={1: 1024, 2: 3072, 3: 5120}[a.generation],
        canonical_forward_rows=0,
        intervention_forward_rows=0,
        retention_forward_rows=0,
        nonzero_gradient_steps=0,
    )
    try:
        import copy

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_compact_support import CompactSupportedHead
        from rosclaw_soccer.rsi.paid_interventional_data import load_interventional_data
        from rosclaw_soccer.rsi.paid_interventional_student import (
            data_support,
            learner_config,
            learner_status,
            learning_certificate,
            load_precision_warm,
            load_replanned_warm,
            load_warm_state,
            precision_imitation_loss,
            validate_learner_state,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        if a.generation in (2, 3):
            from rosclaw_soccer.rsi.paid_replan_data import load_replanned_data

            parent, data, z = load_replanned_data(a.data_root, a.data_hash, policy)
            if a.generation == 3:
                warm, wr = load_precision_warm(
                    a.warm_root, a.warm_hash, data, policy, data_hash=a.data_hash
                )
            else:
                warm, wr = load_replanned_warm(a.warm_root, a.warm_hash, data, policy)
        else:
            parent, data, z = load_interventional_data(a.data_root, a.data_hash, policy)
            warm, wr = load_warm_state(a.warm_root, a.warm_hash, data)
        config = learner_config(a.seed, control=a.geometry_control, generation=a.generation)
        initial = {
            k.removeprefix("net."): v.clone()
            for k, v in warm["model"].items()
            if k.startswith("net.")
        }
        parent_before = {k: v.clone() for k, v in parent.residual.state_dict().items()}
        device = "cpu"
        compute = dict(
            device="cpu",
            geometry_only_control=a.geometry_control,
            torch_version=str(torch.__version__),
        )
        if not a.geometry_control:
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
                raise ValueError("CUDA UUID mismatch")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed_all(config["seed"])
            device = "cuda:0"
            compute.update(device=device, UUID=uuid, name=props.name, TF32=False)
        torch.manual_seed(config["seed"])
        head = CompactSupportedHead(data_support(z))
        head.net.load_state_dict(initial, strict=True)
        head = head.to(device).train(not a.geometry_control)
        optimizer = torch.optim.Adam(head.net.parameters(), lr=config["lr"])
        optimizer.load_state_dict(copy.deepcopy(warm["optimizer"]))
        identity = dict(
            config=config,
            source_head=a.expected_head,
            data_hash=a.data_hash,
            warm_hash=a.warm_hash,
            **{k: data[k] for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")},
        )
        write_once(
            a.output / "declaration.json",
            dict(
                **identity,
                data_root=str(a.data_root),
                warm_root=str(a.warm_root),
                input_file_hashes=pins,
                parent_parameters_in_optimizer=0,
                base_parameters_in_optimizer=0,
                Fresh_fit_rows=0,
                DEV_fit_rows=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        write_once(a.output / "compute.json", compute)
        if not a.geometry_control:
            x, y, prior = [
                torch.from_numpy(z[k]).to(device) for k in ("features", "target", "parent_delta")
            ]
            rows = np.arange(len(x))
            canonical = np.flatnonzero(z["teacher"] & (rows < data["original_rows"]))
            intervention = np.flatnonzero(z["teacher"] & (rows >= data["original_rows"]))
            protected = np.flatnonzero(~z["teacher"])
            if not len(intervention) or not z["active"][intervention].any():
                raise ValueError("actual new own-state teaching signal required")
            active = torch.from_numpy(z["active"]).to(device)
            one = torch.ones(512, 1, device=device)
            zero = torch.zeros_like(one)
            rng = np.random.default_rng(config["seed"])
            for step in range(config["new_steps"]):
                pi = torch.from_numpy(
                    np.r_[
                        rng.choice(canonical, 256, replace=True),
                        rng.choice(intervention, 256, replace=True),
                    ]
                ).to(device)
                ni = torch.from_numpy(rng.choice(protected, 512, replace=True)).to(device)
                kept = head.combine(x[ni], prior[ni], zero)
                if a.generation == 3:
                    teacher_loss = precision_imitation_loss(
                        head.correction(x[pi]), y[pi], active[pi]
                    )
                else:
                    pred = head.combine(x[pi], prior[pi], one)
                    weights = torch.where(active[pi], 5.0, 1.0)
                    teacher_loss = (weights[:, None] * (pred - prior[pi] - y[pi]).square()).mean()
                loss = teacher_loss + 5 * (kept - prior[ni]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.net.parameters(), 1.0)
                if (
                    not torch.isfinite(loss)
                    or not torch.isfinite(norm)
                    or any(
                        v.grad is None or not torch.isfinite(v.grad).all()
                        for v in head.net.parameters()
                    )
                    or any(v.grad is not None for v in parent.parameters())
                ):
                    raise ValueError("finite isolated actual neural gradient required")
                optimizer.step()
                counts["optimizer_updates"] += 1
                counts["canonical_forward_rows"] += 256
                counts["intervention_forward_rows"] += 256
                counts["retention_forward_rows"] += 512
                counts["nonzero_gradient_steps"] += int(norm > 0)
                if (step + 1) % 256 == 0:
                    event = dict(step=step + 1, loss=float(loss.detach()), counts=counts.copy())
                    write_once(a.output / f"step{step + 1}.json", event)
                    print(json.dumps(event), flush=True)
        head = head.cpu().eval()
        if any(
            not torch.equal(v, parent_before[k]) for k, v in parent.residual.state_dict().items()
        ):
            raise ValueError("frozen old motor memory changed")
        cert = learning_certificate(
            head, z, data["original_rows"], initial, control=a.geometry_control
        )

        def cpu(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().clone()
            if isinstance(v, dict):
                return {k: cpu(t) for k, t in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(cpu(t) for t in v)
            return v

        state = cpu(
            dict(
                schema=f"paid.interventional_dagger_student.v{a.generation}",
                **identity,
                declaration_hash=digest(a.output / "declaration.json"),
                head=head.state_dict(),
                initial_net=initial,
                optimizer=optimizer.state_dict(),
                counts=counts,
                learning_certificate=cert,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
        receipt = dict(
            status=learner_status(a.generation, a.geometry_control),
            **identity,
            declaration_hash=state["declaration_hash"],
            input_file_hashes=pins,
            counts=counts,
            learning_certificate=cert,
            parent_parameters_in_optimizer=0,
            base_parameters_in_optimizer=0,
            parent_parameters_exact_unchanged=True,
            own_native_steps=0,
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        validate_learner_state(state, receipt, warm)
        cp = a.output / "interventional-student.pt"
        torch.save(state, cp)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(**receipt, checkpoint_path=str(cp), checkpoint_hash=digest(cp)),
        )
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
