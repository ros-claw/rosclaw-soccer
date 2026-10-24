"""Train a bounded role-command MLP from verified pre-action native8 snapshots.

This is supervised high-level command learning, NOT joint-torque or online RL.
Retains the existing98->29 gait unchanged. Source repetitions are verified but
never duplicated into training data. Mirror courses are not independent exams.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "core-source-root"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-hash", "expected-head", "core-expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(926061, 926062, 926063), required=True)
    p.add_argument("--dagger-continuation", action="store_true")
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.plan.is_symlink() or digest(a.plan) != a.plan_hash:
        raise ValueError("new output and exact public-course plan required")
    plan = json.loads(a.plan.read_text())
    if (
        plan["schema"] != "soccer.native_role_command_training_plan.v1"
        or plan["source_head"] != a.expected_head
        or plan["optimizer_updates"] != 512
        or plan["train_courses"] != ["outbound-north", "incoming-north", "incoming-east"]
        or plan["validation_courses"] != ["incoming-south", "incoming-west"]
        or len(plan["sources"]) != 5
        or plan["hardware_authorized"] is not False
    ):
        raise ValueError("bounded predeclared command-only supervised curriculum required")
    pins = {str(a.plan): a.plan_hash}
    courses = {}
    for item in plan["sources"]:
        root = Path(item["root"])
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != item["hash"] or (root / "failed.json").exists():
            raise ValueError("complete exact source required; no failed-stage replacement")
        report = json.loads(path.read_text())
        path = root / "declaration.json"
        declaration = json.loads(path.read_text())
        pins[str(path)] = digest(path)
        if (
            declaration["protocol"] != "native8_measured_clearance_contrast.v5"
            or declaration["ball_course"] != item["course"]
            or item["course"] in courses
            or not report["fresh_reexecution_exact"]
            or len(report["records"]) != 4
        ):
            raise ValueError("complete distinct measured native8 clearance course required")
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source ancestry")
            pins[key] = value
        pins[str(root / "complete.json")] = item["hash"]
        for row in report["records"]:
            path = root / row["artifact"]["path"]
            if (
                path.parent.resolve() != root.resolve()
                or path.is_symlink()
                or digest(path) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined full course arrays required")
            pins[str(path)] = row["artifact"]["hash"]
            if row["arm"] == "role-foot-clearance" and not row["healthy_full_horizon"]:
                raise ValueError("unhealthy teacher not admitted")
            if row["arm"] == "role-foot-track" and not all(row["preceding_foot_identity"].values()):
                raise ValueError("unaltered predecessor identity required")
        row = next(
            row
            for row in report["records"]
            if row["arm"] == "role-foot-clearance" and row["repeat"] == 0
        )
        courses[item["course"]] = (root, row)
    if set(courses) != set(plan["train_courses"] + plan["validation_courses"]):
        raise ValueError("complete exact train/validation course partition required")
    parent = query = continuation = None
    if a.dagger_continuation:
        continuation = plan["continuations"][str(a.seed)]
        bound = {}
        for key in ("parent", "queries"):
            root = Path(continuation[key + "_root"])
            path = root / "complete.json"
            if path.is_symlink() or digest(path) != continuation[key + "_hash"]:
                raise ValueError("exact own parent and completed teacher queries required")
            bound[key] = json.loads(path.read_text())
            for file, value in bound[key]["input_file_hashes"].items():
                if file in pins and pins[file] != value:
                    raise ValueError("conflicting continuation ancestry")
                pins[file] = value
            pins[str(path)] = continuation[key + "_hash"]
        parent, query = bound["parent"], bound["queries"]
        if (
            parent["optimizer_updates"] != 512
            or parent["seed"] != a.seed
            or query["seed"] != a.seed
            or query["training_hash"] != continuation["parent_hash"]
            or query["original_teacher_labels_exact"] != 22000
            or query["counterfactual_learner_state_queries"] != 22000
            or query["status"]
            != "NATIVE_ROLE_DAGGER_LABELS_QUERIED_NOT_TRAINED_OR_PHYSICALLY_PROVEN"
        ):
            raise ValueError("own512-step parent and all validated DAgger queries required")
        for root, entries in (
            (Path(continuation["parent_root"]), [parent["policy"], parent["checkpoint"]]),
            (Path(continuation["queries_root"]), [query["dataset"]]),
        ):
            for item in entries:
                path = root / item["path"]
                if (
                    path.is_symlink()
                    or path.parent.resolve() != root.resolve()
                    or digest(path) != item["hash"]
                ):
                    raise ValueError("exact confined model/checkpoint/query data required")
                pins[str(path)] = item["hash"]
    elif "continuations" in plan:
        raise ValueError("continuation plan requires explicit DAgger mode")
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    import torch
    from torch import nn

    from rosclaw_soccer.rsi.native_role_command import (
        ROLE_COMMAND_SCHEMA,
        NativeRoleCommandPolicy,
        role_command_observation,
    )
    from rosclaw_soccer.rsi.native_team_intent import ROLES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    agents = [(team, role, team + "_" + role + "/") for team in ("red", "blue") for role in ROLES]
    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    gait = {
        k: v
        for k, v in pins.items()
        if k.endswith(("/velocity/v0/exported/policy.onnx", "/velocity/v0/params/deploy.yaml"))
    }
    if len(scenes) != 1 or len(gait) != 2:
        raise ValueError("unique exact scene and gait qualification required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=tuple(x[2] for x in agents))
    )
    metric = mujoco.MjData(model)
    layouts = {
        prefix: NativeVelocityLayout(
            model, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=8
        )
        for _, _, prefix in agents
    }
    binding = hash_json(
        dict(
            agents=[
                dict(team=t, role=r, body_hash=hash_json(layouts[prefix].profile()))
                for t, r, prefix in agents
            ],
            gait=gait,
        )
    )
    observations, commands, train, ids = [], [], [], []
    for course_index, (course, (root, row)) in enumerate(courses.items()):
        expected = [x["body_binding_hash"] for x in row["players"]]
        if expected != [hash_json(layouts[prefix].profile()) for _, _, prefix in agents]:
            raise ValueError("dataset body binding differs from actual compiled world")
        with np.load(root / row["artifact"]["path"]) as raw:
            raw_qpos, raw_qvel, raw_commands = (raw[key] for key in ("qpos", "qvel", "commands"))
            if len(raw_commands) != 600 or len(raw_qpos) != 6000:
                raise ValueError("complete12sec teacher required")
            for frame in range(50, 600):
                # State before command(frame), not state after its own effect.
                metric.qpos[:] = raw_qpos[frame * 10 - 1]
                metric.qvel[:] = raw_qvel[frame * 10 - 1]
                mujoco.mj_kinematics(model, metric)
                for i, (team, role, _prefix) in enumerate(agents):
                    observations.append(
                        role_command_observation(
                            metric,
                            metric,
                            layouts,
                            team=team,
                            role=role,
                            previous_body_command=raw_commands[frame - 1, i],
                        )
                    )
                    commands.append(raw_commands[frame, i].copy())
                    train.append(course in plan["train_courses"])
                    ids.append((course_index, frame, i))
    x, y, mask = np.array(observations), np.array(commands), np.array(train)
    if x.shape != (22000, 118) or int(mask.sum()) != 13200 or y.shape != (22000, 3):
        raise ValueError("complete deduplicated five-source pre-action dataset required")
    a.output.mkdir(parents=True)
    data_path = a.output / "dataset.npz"
    sample_kind = np.zeros(len(x), np.int8)
    qx = qy = qmask = None
    if query is not None:
        with np.load(Path(continuation["queries_root"]) / query["dataset"]["path"]) as saved:
            qx, qy, qmask, qids = (
                saved[k] for k in ("observations", "commands", "training", "sample_ids")
            )
        if (
            qx.shape != (22000, 118)
            or qy.shape != (22000, 3)
            or qmask.dtype != np.bool_
            or int(qmask.sum()) != 13200
            or not np.isfinite(qx).all()
            or not np.isfinite(qy).all()
        ):
            raise ValueError("complete finite on-policy state/teacher-label partition required")
        sx, sy, smask, sids = (
            np.concatenate((x, qx)),
            np.concatenate((y, qy)),
            np.concatenate((mask, qmask)),
            np.concatenate((np.array(ids), qids)),
        )
        sample_kind = np.concatenate((sample_kind, np.ones(len(qx), np.int8)))
    else:
        sx, sy, smask, sids = x, y, mask, np.array(ids)
    np.savez_compressed(
        data_path,
        observations=sx,
        commands=sy,
        training=smask,
        sample_ids=sids,
        sample_kind=sample_kind,
    )
    dataset_hash = digest(data_path)
    mean = x[mask].mean(axis=0)
    std = np.maximum(x[mask].std(axis=0), np.float32(0.02))
    normalized = np.clip((x - mean) / std, -10, 10)
    tx, ty = torch.from_numpy(normalized[mask]), torch.from_numpy(y[mask])
    vx, vy = torch.from_numpy(normalized[~mask]), torch.from_numpy(y[~mask])
    network = nn.Sequential(
        nn.Linear(118, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 3), nn.Tanh()
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    scale, bias = torch.tensor([0.75, 0.5, 1.0]), torch.tensor([0.25, 0, 0])
    qtx = qty = qvx = qvy = None
    if parent is not None:
        if parent["binding_hash"] != binding:
            raise ValueError("continued model must retain exact body/gait binding")
        checkpoint = torch.load(
            Path(continuation["parent_root"]) / parent["checkpoint"]["path"],
            map_location="cpu",
            weights_only=True,
        )
        if (
            checkpoint["updates"] != 512
            or checkpoint["seed"] != a.seed
            or not np.array_equal(mean, checkpoint["mean"].numpy())
            or not np.array_equal(std, checkpoint["std"].numpy())
        ):
            raise ValueError("original update history, seed and normalization required")
        network.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not all(float(v["step"]) == 512 for v in optimizer.state.values()):
            raise ValueError("all original Adam moments/step counters required")
        torch.set_rng_state(checkpoint["generator_state"])
        old_policy = NativeRoleCommandPolicy(
            Path(continuation["parent_root"]) / parent["policy"]["path"],
            expected_hash=parent["policy"]["hash"],
            expected_binding=binding,
        )
        with torch.no_grad():
            resumed = (network(torch.from_numpy(normalized[:64])) * scale + bias).numpy()
        if np.abs(resumed - np.stack([old_policy.propose(o) for o in x[:64]])).max() > 2e-6:
            raise ValueError("resumed Torch weights differ from the deployed parent")
        qnormalized = np.clip((qx - mean) / std, -10, 10)
        qtx, qty = torch.from_numpy(qnormalized[qmask]), torch.from_numpy(qy[qmask])
        qvx, qvy = torch.from_numpy(qnormalized[~qmask]), torch.from_numpy(qy[~qmask])

    def loss(inputs, labels):
        return (
            ((network(inputs) * scale + bias - labels) / torch.tensor([1.0, 0.5, 1.0])) ** 2
        ).mean()

    with torch.no_grad():
        before = float(loss(vx, vy))
        query_before = float(loss(qvx, qvy)) if qvx is not None else None
    for _ in range(512):
        indices = torch.randint(len(tx), (256,))
        optimizer.zero_grad(set_to_none=True)
        error = loss(tx[indices], ty[indices])
        if qtx is not None:
            qi = torch.randint(len(qtx), (256,))
            error = 0.5 * error + 0.5 * loss(qtx[qi], qty[qi])
        if not bool(torch.isfinite(error)):
            raise FloatingPointError("nonfinite command BC loss")
        error.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        after, training_loss = float(loss(vx, vy)), float(loss(tx, ty))
        query_after = float(loss(qvx, qvy)) if qvx is not None else None
    tensors = dict(mean=mean.tolist(), std=std.tolist())
    for i, index in enumerate((0, 2, 4)):
        tensors[f"w{i}"] = network[index].weight.detach().numpy().tolist()
        tensors[f"b{i}"] = network[index].bias.detach().numpy().tolist()
    value = dict(
        schema=ROLE_COMMAND_SCHEMA,
        binding_hash=binding,
        observation_dim=118,
        dataset_hash=dataset_hash,
        optimizer_updates=1024 if parent is not None else 512,
        seed=a.seed,
        action_semantics="ABSOLUTE_BODY_VELOCITY_COMMAND_BEFORE_CLEARANCE",
        training_kind="PUBLIC_COURSE_DAGGER_CONTINUATION"
        if parent is not None
        else "PUBLIC_COURSE_COACHED_BEHAVIOR_CLONING",
        hardware_authorized=False,
        tensors=tensors,
    )
    write_once(a.output / "command-policy.json", value)
    policy_hash = digest(a.output / "command-policy.json")
    policy = NativeRoleCommandPolicy(
        a.output / "command-policy.json", expected_hash=policy_hash, expected_binding=binding
    )
    with torch.no_grad():
        expected = (network(torch.from_numpy(normalized[:64])) * scale + bias).numpy()
    actual = np.stack([policy.propose(o) for o in x[:64]])
    maximum_difference = float(np.abs(actual - expected).max())
    if maximum_difference > 2e-6:
        raise ValueError("portable NumPy inference differs from trained Torch model")
    # Safe tensor-only checkpoint retains Adam for continued supervised updates.
    checkpoint = a.output / "resume.pt"
    torch.save(
        dict(
            model=network.state_dict(),
            optimizer=optimizer.state_dict(),
            seed=a.seed,
            updates=1024 if parent is not None else 512,
            mean=torch.from_numpy(mean),
            std=torch.from_numpy(std),
            generator_state=torch.get_rng_state(),
        ),
        checkpoint,
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_ROLE_COMMAND_DAGGER_TRAINED_NOT_PHYSICALLY_EVALUATED"
            if parent is not None
            else "NATIVE_ROLE_COMMAND_BC_TRAINED_NOT_PHYSICALLY_EVALUATED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_hash=a.plan_hash,
            policy=dict(path="command-policy.json", hash=policy_hash),
            binding_hash=binding,
            dataset=dict(path=data_path.name, hash=dataset_hash),
            unique_physical_source_courses=5,
            training_courses=plan["train_courses"],
            validation_courses=plan["validation_courses"],
            training_examples=int(smask.sum()),
            validation_examples=int((~smask).sum()),
            validation_is_mirror_related_not_independent_generalization=True,
            no_current_command_label_in_observation=True,
            seed=a.seed,
            actual_BC_actor_optimizer_updates=512,
            actual_SAC_actor_optimizer_updates=0,
            actual_critic_optimizer_updates=0,
            optimizer_updates=1024 if parent is not None else 512,
            actual_new_optimizer_updates=512,
            parent_training_hash=continuation["parent_hash"] if continuation else None,
            query_hash=continuation["queries_hash"] if continuation else None,
            original_Adam_and_RNG_and_normalization_retained=parent is not None,
            validation_scaled_mse_before=before,
            validation_scaled_mse_after=after,
            training_scaled_mse=training_loss,
            learner_state_validation_mse_before=query_before,
            learner_state_validation_mse_after=query_after,
            torch_numpy_max_abs_error=maximum_difference,
            torch_numpy_comparison_examples=64,
            retained_gait_weights_unchanged=True,
            fresh_physical_task_gain_not_claimed=True,
            checkpoint=dict(path=checkpoint.name, hash=digest(checkpoint)),
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
