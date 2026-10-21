"""Finite TRAIN-only compact-support motor fit; no runtime admission or physics claim."""

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
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
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
        raise ValueError("unused output, exact TRAIN data and clean frozen learner required")
    d = json.loads(rp.read_text())
    pins = dict(d["input_file_hashes"])
    for f in (rp, a.data_root / "declaration.json", Path(d["feature_path"])):
        pins[str(f)] = digest(f)
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    counts = dict(optimizer_updates=0, positive_forward_rows=0, retention_forward_rows=0)
    try:
        import time

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_compact_support import (
            CompactSupportedHead,
            CompactTeacherSupport,
            nearest_protected_radii,
        )
        from rosclaw_soccer.rsi.paid_full_entry_student import (
            head_rng_after_v1_gate,
            load_full_entry_data,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        parent, data, z = load_full_entry_data(a.data_root, a.data_hash, policy)
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not uuid.startswith("GPU-")
            or "," in uuid
            or torch.cuda.device_count() != 1
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one UUID GPU and deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != uuid:
            raise ValueError("CUDA UUID mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        seed = 2026102800 + a.seed
        config = dict(
            protocol="full-entry-contact.compact-support.v3",
            seed_index=a.seed,
            seed=seed,
            head_steps=1024,
            gate_steps=0,
            lr=0.0003,
            batch_rows=512,
            active_weight=5.0,
            retention_weight=5.0,
            radius_fraction=0.45,
            radius_distance="direct_float64_euclidean_own288",
            head_minibatches="exact_v1_PCG_stream_after512_gate_iterations",
            final_R2_bound_rad=0.2,
            correction_bound_rad=0.4,
            parent_frozen=True,
            base_frozen=True,
            runtime_case_lookup=False,
            Fresh_fit_rows=0,
            DEV_fit_rows=0,
        )
        identity = dict(
            config=config,
            source_head=a.expected_head,
            data_hash=a.data_hash,
            **{
                k: data[k]
                for k in (
                    "feature_hash",
                    "parent_hash",
                    "parent_checkpoint_hash",
                    "body_scene_hash",
                )
            },
        )
        write_once(
            a.output / "declaration.json",
            dict(
                **identity,
                data_root=str(a.data_root),
                input_file_hashes=pins,
                physical_qualification_required=True,
                automatic_promotion=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        write_once(
            a.output / "compute.json",
            dict(UUID=uuid, name=props.name, torch_version=str(torch.__version__), TF32=False),
        )
        x, y, prior = [
            torch.from_numpy(z[k]).cuda() for k in ("features", "target", "parent_delta")
        ]
        pos, neg = np.flatnonzero(z["teacher"]), np.flatnonzero(~z["teacher"])
        radii = nearest_protected_radii(x[pos], x[neg])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CompactSupportedHead(CompactTeacherSupport(x[pos], radii)).cuda().train()
        # Same MLP shape and seeded six tensors as the rejected full-entry head.
        initial = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
        core = np.flatnonzero(np.abs(z["target"]).max(axis=1) > 0.005)
        rng = head_rng_after_v1_gate(seed, core, int((~z["active"]).sum()))
        optimizer = torch.optim.Adam(model.net.parameters(), lr=0.0003)
        teacher_gains = torch.ones((512, 1), device="cuda")
        protected_gains = torch.zeros_like(teacher_gains)
        active = torch.from_numpy(z["active"]).cuda()
        gradient_steps = 0
        for step in range(1024):
            pi = torch.from_numpy(rng.choice(pos, 512, replace=True)).cuda()
            ni = torch.from_numpy(rng.choice(neg, 512, replace=True)).cuda()
            # These exact anchor gains are proved from direct distances below,
            # not arbitrary labels substituted for runtime support decisions.
            pred = model.combine(x[pi], prior[pi], teacher_gains)
            retained = model.combine(x[ni], prior[ni], protected_gains)
            weight = torch.where(active[pi], 5.0, 1.0)
            loss = (weight[:, None] * (pred - prior[pi] - y[pi]).square()).mean()
            loss = loss + 5 * (retained - prior[ni]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(norm)
                or any(
                    v.grad is None or not torch.isfinite(v.grad).all()
                    for v in model.net.parameters()
                )
            ):
                raise ValueError("finite actual neural gradient required")
            gradient_steps += int(norm > 0)
            optimizer.step()
            counts["optimizer_updates"] += 1
            counts["positive_forward_rows"] += 512
            counts["retention_forward_rows"] += 512
            if (step + 1) % 128 == 0:
                event = dict(
                    step=step + 1, loss=float(loss.detach()), gradient_steps=gradient_steps
                )
                write_once(a.output / f"head{step + 1}.json", event)
                print(json.dumps(event), flush=True)
        model = model.cpu().eval()
        changed = max(
            float((v - initial[k]).abs().max()) for k, v in model.net.state_dict().items()
        )
        if gradient_steps != 1024 or changed <= 0:
            raise ValueError("no-op optimization is a rejected fit, never a trained candidate")
        features = torch.from_numpy(z["features"])
        protected_exact = teacher_open = 0
        squared_error = baseline_error = 0.0
        max_protected_gain = 0.0
        total_support_seconds = 0.0
        with torch.no_grad():
            for start in range(0, len(features), 16):
                stop = min(start + 16, len(features))
                batch = features[start:stop]
                t0 = time.monotonic()
                gain = model.support(batch)
                total_support_seconds += time.monotonic() - t0
                p0 = torch.from_numpy(z["parent_delta"][start:stop])
                pred = model.combine(batch, p0, gain)
                mask = torch.from_numpy(z["teacher"][start:stop])
                if mask.any():
                    if not torch.equal(gain[mask], torch.ones_like(gain[mask])):
                        raise ValueError("teacher centre support must be exactly one on CPU")
                    teacher_open += int(mask.sum())
                if (~mask).any():
                    max_protected_gain = max(max_protected_gain, float(gain[~mask].max()))
                    if gain[~mask].count_nonzero() or not torch.equal(pred[~mask], p0[~mask]):
                        raise ValueError("protected CPU predictions must remain bitwise parent")
                    protected_exact += int((~mask).sum())
                am = torch.from_numpy(z["active"][start:stop])
                target = torch.from_numpy(z["target"][start:stop])
                squared_error += float((pred[am] - p0[am] - target[am]).square().sum())
                baseline_error += float(target[am].square().sum())
        certificate = dict(
            protected_rows_exact_parent=protected_exact,
            teacher_centres_gain_exact_one=teacher_open,
            maximum_protected_gain_CPU=max_protected_gain,
            active_correction_MSE=squared_error / (int(z["active"].sum()) * 29),
            no_op_active_correction_MSE=baseline_error / (int(z["active"].sum()) * 29),
            maximum_head_parameter_change=changed,
            nonzero_gradient_steps=gradient_steps,
            CPU_support_query_rows=len(features),
            CPU_support_batch_shape=16,
            CPU_support_seconds=total_support_seconds,
            CPU_neural_head_forward_rows=len(features),
            runtime_batch1_not_yet_certified=True,
            physical_qualification=False,
        )

        # Numerical fitting, parameter effect, and closed-loop physics are
        # separate gates. No loader or public runtime admits this artifact yet.
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(cpu(v) for v in value)
            return value

        cp = a.output / "compact-support-student.pt"
        state = dict(
            schema="paid.compact_support.fit_only.v1",
            **identity,
            declaration_hash=digest(a.output / "declaration.json"),
            model=cpu(model.state_dict()),
            initial_net=initial,
            optimizer=cpu(optimizer.state_dict()),
            counts=counts,
            retention_certificate=certificate,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        torch.save(state, cp)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_COMPACT_SUPPORT_OFFLINE_FIT_ONLY_NOT_ADMITTED",
                **identity,
                declaration_hash=state["declaration_hash"],
                input_file_hashes=pins,
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                counts=counts,
                retention_certificate=certificate,
                parent_parameters_in_optimizer=0,
                base_parameters_in_optimizer=0,
                Fresh_fit_rows=0,
                DEV_fit_rows=0,
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
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
