"""Distill CPU-verified composite teachers into an unassisted recurrent actor.

CUDA is ONLY a gradient-computation platform over CPU physical labels here.
This does not qualify CUDA control/physics. All candidates require independent
CPU neural-only closed-loop exams. External/derived weights remain CC-BY-NC.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_search_paid_foot_contact import contact_rank

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--variant", type=int, choices=range(4), required=True)
    p.add_argument("--epochs", type=int, choices=(1, 128, 256), default=256)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    teacher_path, audit_path = a.teacher_root / "complete.json", a.audit_root / "complete.json"
    if (
        a.output.exists()
        or digest(teacher_path) != a.teacher_hash
        or digest(audit_path) != a.audit_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen actual teacher/audit/trainer required")
    t, u = json.loads(teacher_path.read_text()), json.loads(audit_path.read_text())
    if (
        t["status"] != "PAID_CARTESIAN_CONTACT_TEACHER_RESEARCH_NOT_PROMOTED"
        or u["status"] != "PAID_CARTESIAN_RAW_SELECTION_AND_CLOSED_LOOP_AUDIT_PASSED"
        or u["input_file_hashes"].get(str(teacher_path)) != a.teacher_hash
        or u["all_selected_and_baseline_closedloop_exact"] is not True
    ):
        raise ValueError(
            "all selected CPU physical teachers must have independent closed-loop proof"
        )
    pins = {
        **t["input_file_hashes"],
        str(teacher_path): a.teacher_hash,
        str(audit_path): a.audit_hash,
    }
    data = []
    if len(t["selected"]) != 8 or len(t["baseline"]) != 16:
        raise ValueError("complete eight-source research cohort required")
    for selected in t["selected"]:
        if contact_rank(selected["results"])[:3] != (1, 2, 2):
            raise ValueError("all16 TRAIN teachers must be physically qualified, no cherry-picking")
        old = [r for r in t["baseline"] if r["source"] == selected["source"]]
        if len(old) != 2:
            raise ValueError("two full baseline rehearsal episodes per source required")
        for row, baseline in zip(selected["results"], old, strict=True):
            for r in (row, baseline):
                path = a.teacher_root / r["raw_path"]
                if not path.resolve().is_relative_to(a.teacher_root.resolve()):
                    raise ValueError("teacher data path escaped research root")
                pins[str(path)] = r["raw_hash"]
            data.append(
                dict(
                    source=selected["source"],
                    teacher=str(a.teacher_root / row["raw_path"]),
                    rehearsal=str(a.teacher_root / baseline["raw_path"]),
                )
            )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    variant = (
        dict(lr=0.0001, rehearsal=0.25, freeze_rnn=False),
        dict(lr=0.00003, rehearsal=0.25, freeze_rnn=False),
        dict(lr=0.0001, rehearsal=1.0, freeze_rnn=False),
        dict(lr=0.0003, rehearsal=0.25, freeze_rnn=True),
    )[a.variant]
    config = dict(
        **variant,
        epochs=a.epochs,
        minibatch_episodes=4,
        seed=20261021 + a.variant,
        active_weight=5.0,
        other_weight=1.0,
        max_grad_norm=1.0,
        full_episode_BPTT=True,
        teacher_hidden_states_not_inputs=True,
        normalizer_frozen=True,
        loss_units="joint_target_radians_squared",
        teacher_episodes=16,
        DEV_episodes_used=0,
    )
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            data=data,
            config=config,
            training_physics_labels="CPU native MuJoCo",
            CUDA_used_for="offline gradients only",
            CUDA_control_qualified=False,
            inference_exam_required="CPU, neural-only, no IK/latent intervention",
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    counts = dict(teacher_forward_rows=0, rehearsal_forward_rows=0, optimizer_updates=0)
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if (
            not visible.startswith("GPU-")
            or "," in visible
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.cuda.device_count() != 1
        ):
            raise ValueError("one UUID-isolated GPU and deterministic CUDA workspace required")
        props = torch.cuda.get_device_properties(0)
        if "GPU-" + str(props.uuid) != visible:
            raise ValueError("actual GPU UUID differs from declared training device")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = np.random.default_rng(config["seed"])
        paths = [Path(f) for f in pins if f.endswith("/ckp/policy_30000.onnx")]
        scenes = [Path(f) for f in pins if f.endswith("/cached-standard/declared-scene.xml")]
        if len(paths) != 1 or not scenes or len({digest(f) for f in scenes}) != 1:
            raise ValueError("one source actor and identical qualified standard physics required")
        model = PaidTrainableActor(paths[0], expected_hash=PUBLISHED_HASH).to("cuda:0")
        if config["freeze_rnn"]:
            for par in model.rnn.parameters():
                par.requires_grad_(False)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=config["lr"])
        normalization = {k: model.state_dict()[k].detach().clone() for k in ("mean", "denominator")}
        import onnx

        metadata = {
            m.key: m.value for m in onnx.load(paths[0], load_external_data=False).metadata_props
        }
        scale = torch.tensor(
            [float(x) for x in metadata["action_scale"].split(",")],
            device="cuda:0",
            dtype=torch.float32,
        )
        arrays = []
        for entry in data:
            with np.load(entry["teacher"], allow_pickle=False) as z:
                obs, action = z["NN_obs"].copy(), z["NN_action"].copy()
                active = np.any(z["PLAN_pose_offset"] != 0, axis=1)
            with np.load(entry["rehearsal"], allow_pickle=False) as z:
                robs, raction = z["NN_obs"].copy(), z["NN_action"].copy()
            if (
                obs.shape != robs.shape
                or obs.shape[1] != 160
                or action.shape != raction.shape
                or action.shape != (len(obs), 29)
                or not all(np.isfinite(x).all() for x in (obs, action, robs, raction))
            ):
                raise ValueError("complete paired finite160/29 full episodes required")
            arrays.append(
                dict(
                    obs=torch.from_numpy(obs).to("cuda:0"),
                    action=torch.from_numpy(action).to("cuda:0"),
                    robs=torch.from_numpy(robs).to("cuda:0"),
                    raction=torch.from_numpy(raction).to("cuda:0"),
                    active=torch.from_numpy(active).to("cuda:0"),
                )
            )
        compute = dict(
            UUID=visible,
            name=props.name,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            TF32=False,
            distributed_allreduce=False,
            physics_steps=0,
            CUDA_control_qualified=False,
        )
        write_once(a.output / "compute.json", compute)

        def checkpoint(epoch):
            path = a.output / f"distilled-epoch{epoch}.pt"
            if path.exists():
                raise ValueError("checkpoint overwrite forbidden")

            def cpu(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().clone()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [cpu(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(cpu(v) for v in value)
                return value

            state = dict(
                schema="soccer.paid_cartesian_distilled.v1",
                epoch=epoch,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                teacher_hash=a.teacher_hash,
                teacher_audit_hash=a.audit_hash,
                source_actor_hash=PUBLISHED_HASH,
                body_scene_hash=digest(scenes[0]),
                config=config,
                network=cpu(model.state_dict()),
                optimizer=cpu(optimizer.state_dict()),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                counts=counts.copy(),
                commercial_publicity_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            torch.save(state, path)
            if torch.load(path, map_location="cpu", weights_only=True)["schema"] != state["schema"]:
                raise ValueError("safe checkpoint reload failed")
            return digest(path)

        initial_hash = checkpoint(0)
        history = []
        for epoch in range(1, a.epochs + 1):
            order = rng.permutation(16).tolist()
            losses = []
            for start in range(0, 16, 4):
                optimizer.zero_grad(set_to_none=True)
                for i in order[start : start + 4]:
                    episode = arrays[i]
                    zero = (
                        torch.zeros(2, 1, 128, device="cuda:0"),
                        torch.zeros(2, 1, 128, device="cuda:0"),
                    )
                    predicted, _ = model(episode["obs"][:, None], zero)
                    counts["teacher_forward_rows"] += len(episode["obs"])
                    weight = torch.where(
                        episode["active"], config["active_weight"], config["other_weight"]
                    )
                    error = (predicted[:, 0] - episode["action"]) * scale
                    teacher_loss = (error.square() * weight[:, None]).sum() / (29 * weight.sum())
                    rehearsed, _ = model(episode["robs"][:, None], zero)
                    counts["rehearsal_forward_rows"] += len(episode["robs"])
                    mask = ~episode["active"]
                    rehearsal_loss = (
                        ((rehearsed[:, 0] - episode["raction"]) * scale).square() * mask[:, None]
                    ).sum() / (29 * mask.sum().clamp_min(1))
                    loss = teacher_loss + config["rehearsal"] * rehearsal_loss
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite supervised physical action loss")
                    (loss / 4).backward()
                    losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(
                    parameters, config["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                counts["optimizer_updates"] += 1
            if any(not torch.equal(model.state_dict()[k], v) for k, v in normalization.items()):
                raise ValueError("frozen normalization changed")
            row = dict(
                epoch=epoch,
                episode_order=order,
                mean_minibatch_training_loss=float(np.mean(losses)),
                counts=counts.copy(),
            )
            if epoch % 64 == 0 or epoch == a.epochs:
                row["checkpoint_hash"] = checkpoint(epoch)
            write_once(a.output / f"epoch{epoch}.json", row)
            history.append(row)
            if epoch % 16 == 0:
                print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CARTESIAN_DISTILLATION_TRAINED_NOT_PHYSICALLY_EXAMINED",
                input_file_hashes=pins,
                source_head=a.expected_head,
                config=config,
                history=history,
                initial_checkpoint_hash=initial_hash,
                final_checkpoint_hash=digest(a.output / f"distilled-epoch{a.epochs}.pt"),
                counts=counts,
                compute=compute,
                native_steps=0,
                CPU_neural_only_exam_required=True,
                goal_complete=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), counts=counts, native_steps=0, hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
