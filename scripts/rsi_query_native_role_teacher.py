"""Validate coached labels then query that teacher on real learner-visited states.

No physics integration, action application, optimizer, or automatic promotion.
All118 observations must reconstruct exactly before accepting a label. Queries
are marked counterfactual; they do not inherit the original teacher's outcomes.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    path = a.source / "complete.json"
    if a.output.exists() or path.is_symlink() or digest(path) != a.source_hash:
        raise ValueError("new output and exact whole native family required")
    family = json.loads(path.read_text())
    if (
        family["status"] != "NATIVE_COMMAND_FAMILY_PHYSICALLY_MEASURED_NOT_PROMOTED"
        or len(family["results"]) != 5
    ):
        raise ValueError("complete five-course physical exam required")
    pins = {str(path): a.source_hash}
    stages = []
    for item in family["results"]:
        root = a.source / item["course"]
        path = root / "complete.json"
        if (
            root.parent.resolve() != a.source.resolve()
            or path.is_symlink()
            or digest(path) != item["source_hash"]
            or (root / "failed.json").exists()
        ):
            raise ValueError("exact confined completed physical stage required")
        report = json.loads(path.read_text())
        if not report["fresh_reexecution_exact"] or len(report["records"]) != 4:
            raise ValueError("all source and fresh executions required")
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting family ancestry")
            pins[key] = value
        pins[str(path)] = item["source_hash"]
        for row in report["records"]:
            raw = root / row["artifact"]["path"]
            if (
                raw.parent.resolve() != root.resolve()
                or raw.is_symlink()
                or digest(raw) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined physics arrays required")
            pins[str(raw)] = row["artifact"]["hash"]
        stages.append((item["course"], root, report))
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.native_role_command import role_command_observation
    from rosclaw_soccer.rsi.native_role_teacher import coached_native_role_label
    from rosclaw_soccer.rsi.native_team_intent import ROLES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique qualified shared scene required")
    agents = [(t, r, t + "_" + r + "/") for t in ("red", "blue") for r in ROLES]
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
    original_labels, queried_labels, examples, sample_ids, training = [], [], [], [], []
    a.output.mkdir(parents=True)
    # Validate ALL baseline labels first; do not publish learner labels before it.
    for arm in ("role-foot-clearance", "role-command-learned"):
        for course_index, (course, root, report) in enumerate(stages):
            row = next(r for r in report["records"] if r["arm"] == arm and r["repeat"] == 0)
            with np.load(root / row["artifact"]["path"]) as saved:
                raw_qpos, raw_qvel, raw_commands, observed = (
                    saved[k] for k in ("qpos", "qvel", "commands", "role_observations118")
                )
            course_labels = []
            for frame in range(50, len(raw_commands)):
                metric.qpos[:] = raw_qpos[frame * 10 - 1]
                metric.qvel[:] = raw_qvel[frame * 10 - 1]
                mujoco.mj_kinematics(model, metric)
                for i, (team, role, prefix) in enumerate(agents):
                    lo = layouts[prefix]
                    # Exactly the yaw when the previous body command was chosen,
                    # not current yaw (which has since turned in real physics).
                    quat = raw_qpos[(frame - 1) * 10 - 1, lo.rq + 3 : lo.rq + 7]
                    yaw = math.atan2(
                        2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                        1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
                    )
                    c, s = math.cos(yaw), math.sin(yaw)
                    previous = raw_commands[frame - 1, i]
                    world_previous = np.array(
                        [c * previous[0] - s * previous[1], s * previous[0] + c * previous[1]],
                        np.float64,
                    )
                    obs = role_command_observation(
                        metric,
                        metric,
                        layouts,
                        team=team,
                        role=role,
                        previous_body_command=previous,
                    )
                    if not np.array_equal(obs, observed[frame, i]):
                        raise ValueError("pre-action role observation reconstruction differs")
                    label = coached_native_role_label(
                        metric,
                        metric,
                        layouts,
                        team=team,
                        role=role,
                        previous_world_command=world_previous,
                    )
                    if arm == "role-foot-clearance":
                        if not np.array_equal(label, raw_commands[frame, i]):
                            raise ValueError(
                                f"coached label differs before DAgger: {course}/{frame}/{prefix}"
                            )
                        original_labels.append(label)
                    else:
                        examples.append(obs)
                        queried_labels.append(label)
                        sample_ids.append((course_index, frame, i))
                        training.append(
                            course in ("outbound-north", "incoming-north", "incoming-east")
                        )
                    course_labels.append(label)
            write_once(
                a.output / f"queried-{arm}-{course}.json",
                dict(
                    course=course,
                    arm=arm,
                    queries=len(course_labels),
                    original_teacher_labels_exact=arm == "role-foot-clearance",
                    new_labels_are_counterfactual=arm == "role-command-learned",
                    optimizer_updates=0,
                    native_steps=0,
                    hardware_authorized=False,
                ),
            )
    x, y, mask = np.array(examples), np.array(queried_labels), np.array(training)
    if (
        x.shape != (22000, 118)
        or y.shape != (22000, 3)
        or len(original_labels) != 22000
        or int(mask.sum()) != 13200
    ):
        raise ValueError("entire five-course baseline and learner datasets required")
    path = a.output / "counterfactual-dagger-dataset.npz"
    np.savez_compressed(
        path, observations=x, commands=y, training=mask, sample_ids=np.array(sample_ids)
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_ROLE_DAGGER_LABELS_QUERIED_NOT_TRAINED_OR_PHYSICALLY_PROVEN",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            seed=family["seed"],
            training_hash=family["training_hash"],
            original_teacher_labels_exact=22000,
            original_observations_reconstructed_exact=True,
            counterfactual_learner_state_queries=22000,
            dataset=dict(path=path.name, hash=digest(path)),
            training_examples=13200,
            validation_examples=8800,
            optimizer_updates=0,
            native_steps=0,
            counterfactual_action_outcomes_not_proven=True,
            private_fresh_accessed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
