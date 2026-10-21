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
    for name in ("grown", "grown-primitive", "grown-neural"):
        p.add_argument("--" + name + "-root", type=Path)
        p.add_argument("--" + name + "-hash")
    p.add_argument("--variant", type=int, choices=range(4), required=True)
    p.add_argument("--epochs", type=int, choices=(1, 256), default=256)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    growth_args = [
        getattr(a, k + s)
        for k in ("grown", "grown_primitive", "grown_neural")
        for s in ("_root", "_hash")
    ]
    projected = any(x is not None for x in growth_args)
    if projected and not all(x is not None for x in growth_args):
        raise ValueError("all grown replay/primitive/neural commitments required")
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean frozen residual training source required")
    pins, reports = {}, {}
    for key in ("curation", "curation_audit", "zero", "zero_audit") + (
        ("grown", "grown_primitive", "grown_neural") if projected else ()
    ):
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
    original_worlds = curated["worlds"]
    if [x["case"] for x in curated["worlds"]] != list(range(16)):
        raise ValueError("complete unfiltered sixteen-case training bank required")
    grown_candidate = None
    if projected:
        g, gp, gn = [reports[k] for k in ("grown", "grown_primitive", "grown_neural")]
        if (
            gp["status"] != "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS"
            or gn["status"] != "PAID_GATED_CONTACT_ALL_NEURAL_PROPOSALS_EXACT"
            or gp["input_file_hashes"].get(str(a.grown_root / "complete.json")) != a.grown_hash
            or gn["input_file_hashes"].get(str(a.grown_root / "complete.json")) != a.grown_hash
            or gn["input_file_hashes"].get(str(a.grown_primitive_root / "complete.json"))
            != a.grown_primitive_hash
            or gn["own_neural_rows"] != g["actual_neural_rows"]
            or len(g["candidates"]) != 1
            or not g["contact_residual_trained"]
        ):
            raise ValueError("actual independently qualified grown replay required")
        grown_candidate = g["candidates"][0]
        if (
            not grown_candidate["full_continuous_skill_gate"]
            or grown_candidate["continuous"]["stable"] != 16
            or grown_candidate["continuous"]["second_quality"] != 12
        ):
            raise ValueError("this protocol requires the protected grown12 parent")
        new_worlds = []
        for original_world in curated["worlds"]:
            i = original_world["case"]
            c = dict(original_world, source_kind="original_audited_teacher_or_zero")
            if i in (1, 8, 9, 15):
                w = grown_candidate["continuous"]["results"][i]
                if not w["first_quality"] or not w["second_quality"] or not w["world_stable"]:
                    raise ValueError("grown positive labels require complete actual world success")
                c.update(
                    source_kind="grown_neural_success", role="retain_actually_learned_new_skill"
                )
                for key in ("first", "second"):
                    row = w[key]
                    path = a.grown_root / row["raw_path"]
                    if pins.get(str(path)) != row["raw_hash"]:
                        raise ValueError(
                            "grown whole-world trajectories must be independently sealed"
                        )
                    c[key] = dict(row, path=str(path))
            new_worlds.append(c)
        curated = dict(curated, worlds=new_worlds)
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
    if projected:
        check_pins(pins)
        from rosclaw_soccer.rsi.paid_contact_residual import residual_config

        config = residual_config(a.variant, projected=True)
        config["epochs"] = a.epochs
    check_pins(pins)
    a.output.mkdir()
    counts = dict(rescue_forward_rows=0, zero_forward_rows=0, optimizer_updates=0)
    projection = dict(
        calls=0, gate_evaluations=0, rejections=0, accepted_fractions={}, negative_support_rows=0
    )
    if projected:
        counts.update(hard_negative_gate_rows=0, projection_gate_rows=0, CPU_constraint_gate_rows=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual import (
            ContactResidual,
            FrozenContactResidualActor,
        )
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_gate_retention import project_contact_gate
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
        warm_cp_hash = warm_receipt_hash = None
        if projected:
            warm_cp_hash = grown_candidate["checkpoint_hash"]
            warm_receipt_hash = grown_candidate["training_receipt_hash"]
            warm_cp, warm_rp = unique(warm_cp_hash), unique(warm_receipt_hash)
            warm_state = torch.load(warm_cp, map_location="cpu", weights_only=True)
            if warm_state.get("schema") != "soccer.paid_contact_residual.v1":
                raise ValueError("bounded one-layer v1 plastic-head warmstart required")
            warm, _ = load_contact_residual_actor(
                policy,
                checkpoint=warm_cp,
                checkpoint_hash=warm_cp_hash,
                training_receipt=warm_rp,
                training_receipt_hash=warm_receipt_hash,
                scene_hash=digest(scenes[0]),
            )
            if any(not torch.equal(v, warm.base.state_dict()[k]) for k, v in original.items()):
                raise ValueError("grown learner must have the identical frozen base motor")
            residual.net.load_state_dict(warm.residual.net.state_dict(), strict=True)
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
                with np.load(path, allow_pickle=False) as archive:
                    z = {
                        k: archive[k].copy()
                        for k in (
                            "NN_obs",
                            "NN_action",
                            "NN_h_before",
                            "NN_c_before",
                            "NN_h_after",
                            "NN_c_after",
                        )
                    }
                with torch.no_grad():
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
            if c.get("source_kind") == "grown_neural_success":
                with (
                    np.load(c["first"]["path"], allow_pickle=False) as current,
                    np.load(
                        original_worlds[c["case"]]["first"]["path"], allow_pickle=False
                    ) as original_prefix,
                ):
                    if set(current.files) != set(original_prefix.files) or any(
                        not np.array_equal(current[k], original_prefix[k]) for k in current.files
                    ):
                        raise ValueError(
                            "grown replay must share EXACT original learned body prefix"
                        )
            row = c["second"]
            path = Path(row["path"])
            if pins.get(str(path)) != row["raw_hash"]:
                raise ValueError("audited actual successor trajectory required")
            with np.load(path, allow_pickle=False) as z:
                obs, nominal, actual = [
                    z[k].copy() for k in ("NN_obs", "NN_nominal_action", "NN_action")
                ]
                after_memory = {k: z[k].copy() for k in ("NN_h_after", "NN_c_after")}
                state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                fs = []
                recomputed_nominal = []
                for j, item in enumerate(obs):
                    o = torch.from_numpy(item).reshape(1, 1, 160)
                    with torch.no_grad():
                        action, state, hidden = base.forward_with_features(o, state)
                        f = torch.cat([(o - base.mean) / base.denominator, hidden], dim=-1)[0, 0]
                    if (
                        c.get("source_kind") != "grown_neural_success"
                        and not np.array_equal(action[0, 0].numpy(), nominal[j])
                    ) or any(
                        not np.array_equal(x.numpy(), after_memory[k][j])
                        for x, k in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                    ):
                        raise ValueError(
                            "base features must be causally recomputed, not teacher h/c inputs"
                        )
                    fs.append(f)
                    recomputed_nominal.append(action[0, 0].numpy().copy())
                    feature_rows += 1
            if c.get("source_kind") == "grown_neural_success" and not np.array_equal(
                actual, nominal
            ):
                raise ValueError(
                    "grown replay must be actual own neural actions, not another teacher mixture"
                )
            target = torch.from_numpy(actual - np.asarray(recomputed_nominal)) * scale
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
        negative_features = None
        if projected:
            negative_features = torch.cat([x["features"][~x["active"]] for x in features])
            projection["negative_support_rows"] = len(negative_features)
            if not 2336 <= len(negative_features) < 3810:
                raise ValueError("all zero targets plus actual inactive positive support required")
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
                **(
                    dict(
                        grown_exam_hash=a.grown_hash,
                        grown_primitive_hash=a.grown_primitive_hash,
                        grown_neural_hash=a.grown_neural_hash,
                        plastic_head_checkpoint_hash=warm_cp_hash,
                        plastic_head_training_receipt_hash=warm_receipt_hash,
                        negative_support_rows=len(negative_features),
                    )
                    if projected
                    else {}
                ),
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

        last_certificate = None

        def checkpoint(epoch):
            nonlocal last_certificate
            path = a.output / f"contact-residual-epoch{epoch}.pt"
            if path.exists():
                raise ValueError("checkpoint overwrite forbidden")
            certificate = None
            if projected:
                import copy

                with torch.no_grad():
                    cpu_gate = copy.deepcopy(residual.gate).to("cpu").eval()
                    maximum = float(cpu_gate(negative_features.cpu()).max())
                counts["CPU_constraint_gate_rows"] += len(negative_features)
                certificate = dict(
                    negative_support_rows=len(negative_features),
                    maximum_logit_cpu=maximum,
                    feasible=maximum <= -config["gate_retention_margin"] + 1e-5,
                    projection=projection.copy(),
                )
                if config["gate_projection"] and not certificate["feasible"]:
                    raise ValueError("projected model lost strict CPU negative-support feasibility")
                last_certificate = certificate
            state = dict(
                schema="soccer.paid_contact_residual.v2"
                if projected
                else "soccer.paid_contact_residual.v1",
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
                **(
                    dict(
                        retention_certificate=certificate,
                        plastic_head_checkpoint_hash=warm_cp_hash,
                        plastic_head_training_receipt_hash=warm_receipt_hash,
                    )
                    if projected
                    else {}
                ),
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
                before_gate = None
                if projected:
                    scores = residual.gate(negative_features)
                    hard = (
                        config["hard_negative_weight"]
                        * torch.relu(scores.max() + config["hard_negative_margin"]).square()
                    )
                    if not torch.isfinite(hard):
                        raise ValueError("nonfinite hard-negative loss")
                    hard.backward()
                    total += float(hard.detach())
                    counts["hard_negative_gate_rows"] += len(negative_features)
                    if config["gate_projection"]:
                        before_gate = {
                            k: p.detach().clone() for k, p in residual.gate.named_parameters()
                        }
                torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                counts["optimizer_updates"] += 1
                if projected and config["gate_projection"]:
                    proof = project_contact_gate(
                        residual.gate,
                        before_gate,
                        negative_features,
                        margin=config["gate_retention_margin"],
                        max_halvings=config["max_projection_halvings"],
                    )
                    projection["calls"] += 1
                    projection["gate_evaluations"] += proof["gate_evaluations"]
                    projection["rejections"] += int(proof["gate_update_rejected"])
                    key = str(proof["accepted_fraction"])
                    projection["accepted_fractions"][key] = (
                        projection["accepted_fractions"].get(key, 0) + 1
                    )
                    counts["projection_gate_rows"] += proof["gate_forward_rows"]
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
                **(dict(projection=projection.copy()) if projected else {}),
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
                **(dict(retention_certificate=last_certificate) if projected else {}),
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
