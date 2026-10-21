"""Learn gated29-joint corrections; frozen motor and true zero-label retention.

Offline CUDA gradients only. All base features are causally recomputed on CPU
and checked against raw trajectories, never copied from privileged teachers.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("curation", "curation-audit", "zero", "zero-audit"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--variant", type=int, choices=range(4), required=True)
    p.add_argument("--epochs", type=int, choices=(1, 256), default=256)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean frozen residual training source required")
    pins, reports = {}, {}
    for key in ("curation", "curation_audit", "zero", "zero_audit"):
        f = getattr(a, key + "_root") / "complete.json"
        h = getattr(a, key + "_hash")
        if digest(f) != h:
            raise ValueError("exact completed prerequisite required: " + key)
        reports[key] = json.loads(f.read_text())
        for path, ph in reports[key]["input_file_hashes"].items():
            if path in pins and pins[path] != ph:
                raise ValueError("conflicting prerequisite lineage")
            pins[path] = ph
        pins[str(f)] = h
    if (
        reports["curation"]["status"] != "PAID_SUCCESSOR_SPECIALIST_TRAINED_NOT_PHYSICALLY_EXAMINED"
        or reports["curation_audit"]["status"] != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
        or reports["curation_audit"]["input_file_hashes"].get(
            str(a.curation_root / "complete.json")
        )
        != a.curation_hash
        or reports["zero_audit"]["status"] != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
        or reports["zero_audit"]["input_file_hashes"].get(str(a.zero_root / "complete.json"))
        != a.zero_hash
        or reports["zero"]["contact_residual_zero"] is not True
        or len(reports["zero"]["candidates"]) != 1
        or reports["zero"]["candidates"][0]["continuous"]["second_quality"] != 8
        or reports["zero"]["candidates"][0]["continuous"]["stable"] != 16
    ):
        raise ValueError("audited curation and exact unchanged physical zero-motor proof required")
    dp = a.curation_root / "declaration.json"
    if pins.get(str(dp)) != digest(dp):
        raise ValueError("actual sealed same-prefix curriculum required")
    curated = json.loads(dp.read_text())
    if [x["case"] for x in curated["worlds"]] != list(range(16)):
        raise ValueError("complete unfiltered sixteen-case training bank required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    config = dict(
        protocol="frozen-contact-residual-v1",
        variant=a.variant,
        seed=20261104 + (a.variant if a.variant < 3 else 0),
        epochs=a.epochs,
        lr=0.0003,
        retention_weight=5.0 if a.variant < 3 else 1.0,
        active_weight=5.0,
        gate_loss_weight=0.001,
        gate_positive_weight=5.0,
        hard_gate_at_inference=True,
        inference_gate_probability_threshold=0.5,
        max_grad_norm=1.0,
        bound_radians=0.2,
        residual_features=288,
        original_actor_features=160,
        memory_handoff="reset",
        frozen_base=True,
        unique_training_rows=3810,
        rescue_cases=[0, 1, 4, 8, 9, 15],
        old_good_zero_cases=[2, 3, 5, 6, 7, 11, 12, 13],
        failed_body_only_zero_cases=[10, 14],
        rescue_rows=1474,
        zero_retention_rows=2336,
        minibatches_per_epoch=4,
        rescue_batch_sizes=[2, 2, 1, 1],
        zero_batch_sizes=[3, 3, 2, 2],
        DEV_episodes_used=0,
        runtime_case_lookup=False,
        loss_units="joint_residual_radians_squared_plus_gate_BCE",
    )
    check_pins(pins)
    a.output.mkdir()
    counts = dict(rescue_forward_rows=0, zero_forward_rows=0, optimizer_updates=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual import (
            ContactResidual,
            FrozenContactResidualActor,
        )
        from rosclaw_soccer.rsi.paid_live_distilled_actor import load_live_distilled_actor
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH

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
            raise ValueError("actual CUDA UUID differs")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = np.random.default_rng(config["seed"])
        policy = next(Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx"))
        scenes = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if len({digest(f) for f in scenes}) != 1:
            raise ValueError("one exact standard physical body required")
        scale = torch.from_numpy(
            PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
            .parameters["action_scale"]
            .copy()
        )

        def unique(h):
            paths = [Path(f) for f, ph in pins.items() if ph == h]
            if len(paths) != 1:
                raise ValueError("one exact source-bound artifact required")
            return paths[0]

        base_cp = unique(curated["base_checkpoint_hash"])
        base_rp = unique(curated["base_training_receipt_hash"])
        base, _ = load_live_distilled_actor(
            policy,
            checkpoint=base_cp,
            checkpoint_hash=digest(base_cp),
            training_receipt=base_rp,
            training_receipt_hash=digest(base_rp),
            scene_hash=digest(scenes[0]),
        )
        base.requires_grad_(False)
        original = {k: v.clone() for k, v in base.state_dict().items()}
        residual = ContactResidual()
        zero_actor = FrozenContactResidualActor(base, residual, scale).eval()
        preflight_rows = 0
        # The physical zero proof used the earlier ungated zero module. Replay
        # EVERY first/second observation with THIS gated zero module before any
        # gradient, proving identical actions and h/c on its physical histories.
        for world in reports["zero"]["candidates"][0]["continuous"]["results"]:
            for key in ("first", "second"):
                row = world[key]
                path = a.zero_root / row["raw_path"]
                if digest(path) != row["raw_hash"]:
                    raise ValueError("original physical zero history changed")
                pins[str(path)] = row["raw_hash"]
                with np.load(path, allow_pickle=False) as z, torch.no_grad():
                    for j, obs in enumerate(z["NN_obs"]):
                        state = tuple(
                            torch.from_numpy(z[k][j].copy()) for k in ("NN_h_before", "NN_c_before")
                        )
                        act, after = zero_actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                        if not np.array_equal(act[0, 0].numpy(), z["NN_action"][j]) or any(
                            not np.array_equal(x.numpy(), z[k][j])
                            for x, k in zip(after, ("NN_h_after", "NN_c_after"), strict=True)
                        ):
                            raise ValueError("gated zero module differs from exact physical proof")
                        preflight_rows += 1
        if preflight_rows != 7620:
            raise ValueError("complete32skill zero inference replay required")
        features, feature_rows = [], 0
        for c in curated["worlds"]:
            row = c["second"]
            path = Path(row["path"])
            if pins.get(str(path)) != row["raw_hash"]:
                raise ValueError("audited actual successor trajectory required")
            with np.load(path, allow_pickle=False) as z:
                obs, nominal, actual = [
                    z[k].copy() for k in ("NN_obs", "NN_nominal_action", "NN_action")
                ]
                state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                fs = []
                for j, item in enumerate(obs):
                    o = torch.from_numpy(item).reshape(1, 1, 160)
                    with torch.no_grad():
                        action, state, hidden = base.forward_with_features(o, state)
                        f = torch.cat([(o - base.mean) / base.denominator, hidden], dim=-1)[0, 0]
                    if not np.array_equal(action[0, 0].numpy(), nominal[j]) or any(
                        not np.array_equal(x.numpy(), z[k][j])
                        for x, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                    ):
                        raise ValueError(
                            "base features must be causally recomputed, not teacher h/c inputs"
                        )
                    fs.append(f)
                    feature_rows += 1
            target = torch.from_numpy(actual - nominal) * scale
            rescue = c["case"] in config["rescue_cases"]
            if target.abs().max() > 0.20001 or (not rescue and target.any()):
                raise ValueError("bounded teacher deltas and true zero retention targets required")
            features.append(
                dict(
                    case=c["case"],
                    rescue=rescue,
                    rows=len(obs),
                    features=torch.stack(fs).to("cuda:0"),
                    target=target.to("cuda:0"),
                    active=(target.abs().max(dim=1).values > 1e-6).to("cuda:0"),
                )
            )
        rescue = [x for x in features if x["rescue"]]
        retention = [x for x in features if not x["rescue"]]
        if (
            feature_rows != 3810
            or sum(x["rows"] for x in rescue) != 1474
            or sum(x["rows"] for x in retention) != 2336
        ):
            raise ValueError("exact6rescue+10zero measured curriculum required")
        residual = residual.to("cuda:0").train()
        optimizer = torch.optim.Adam(residual.parameters(), lr=config["lr"])
        write_once(
            a.output / "declaration.json",
            dict(
                source_head=a.expected_head,
                input_file_hashes=pins,
                config=config,
                base_checkpoint_hash=digest(base_cp),
                base_training_receipt_hash=digest(base_rp),
                zero_proof_hash=a.zero_hash,
                zero_audit_hash=a.zero_audit_hash,
                curation_hash=a.curation_hash,
                curation_audit_hash=a.curation_audit_hash,
                original_actor_hash=PUBLISHED_HASH,
                body_scene_hash=digest(scenes[0]),
                CPU_zero_neural_replay_rows=preflight_rows,
                CPU_feature_recompute_rows=feature_rows,
                base_frozen=True,
                teacher_hidden_states_copied_as_inputs=False,
                runtime_features="ownnormalized160+ownfrozenrecurrent128",
                worlds=curated["worlds"],
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
        compute = dict(
            UUID=visible,
            name=props.name,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            TF32=False,
            physics_steps=0,
            CUDA_control_qualified=False,
        )
        write_once(a.output / "compute.json", compute)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        def checkpoint(epoch):
            path = a.output / f"contact-residual-epoch{epoch}.pt"
            if path.exists():
                raise ValueError("checkpoint overwrite forbidden")
            state = dict(
                schema="soccer.paid_contact_residual.v1",
                epoch=epoch,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                counts=counts.copy(),
                base_checkpoint_hash=digest(base_cp),
                base_training_receipt_hash=digest(base_rp),
                body_scene_hash=digest(scenes[0]),
                original_actor_hash=PUBLISHED_HASH,
                residual=cpu(residual.state_dict()),
                optimizer=cpu(optimizer.state_dict()),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            )
            torch.save(state, path)
            if torch.load(path, map_location="cpu", weights_only=True)["schema"] != state["schema"]:
                raise ValueError("safe checkpoint reload failed")
            return digest(path)

        initial_hash = checkpoint(0)
        for epoch in range(1, a.epochs + 1):
            ro, zo = rng.permutation(6).tolist(), rng.permutation(10).tolist()
            losses = []
            for rs, zs in zip(
                ([0, 1], [2, 3], [4], [5]), ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9]), strict=True
            ):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for bank, ids, order, weight, key in (
                    (rescue, rs, ro, 1.0, "rescue_forward_rows"),
                    (retention, zs, zo, config["retention_weight"], "zero_forward_rows"),
                ):
                    for index in ids:
                        row = bank[order[index]]
                        pred, logits = residual.forward_with_gate(row["features"])
                        w = torch.where(row["active"], config["active_weight"], 1.0)
                        error = (pred - row["target"]).square().mean(dim=1)
                        gate = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits[:, 0],
                            row["active"].float(),
                            pos_weight=torch.tensor(
                                config["gate_positive_weight"], device="cuda:0"
                            ),
                        )
                        loss = (
                            weight
                            * ((error * w).sum() / w.sum() + config["gate_loss_weight"] * gate)
                            / len(ids)
                        )
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite residual loss")
                        loss.backward()
                        total += float(loss.detach())
                        counts[key] += row["rows"]
                torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                counts["optimizer_updates"] += 1
                losses.append(total)
            if (
                any(not torch.equal(v, base.state_dict()[k]) for k, v in original.items())
                or any(x.grad is not None for x in base.parameters())
                or any(not torch.isfinite(x).all() for x in residual.parameters())
            ):
                raise ValueError("base mutation/gradient or invalid residual")
            row = dict(
                epoch=epoch,
                rescue_order=ro,
                zero_order=zo,
                mean_loss=float(np.mean(losses)),
                counts=counts.copy(),
            )
            if epoch % 64 == 0 or epoch == a.epochs:
                row["checkpoint_hash"] = checkpoint(epoch)
            write_once(a.output / f"epoch{epoch}.json", row)
            if epoch % 16 == 0 or epoch == a.epochs:
                print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTACT_RESIDUAL_TRAINED_NOT_PHYSICALLY_EXAMINED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                config=config,
                counts=counts,
                compute=compute,
                initial_checkpoint_hash=initial_hash,
                final_checkpoint_hash=digest(a.output / f"contact-residual-epoch{a.epochs}.pt"),
                native_steps=0,
                CPU_zero_neural_replay_rows=preflight_rows,
                CPU_feature_recompute_rows=feature_rows,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
            ),
        )
    except Exception as error:
        write_once(a.output / "failed.json", dict(error=repr(error), counts=counts, native_steps=0))
        raise


if __name__ == "__main__":
    main()
