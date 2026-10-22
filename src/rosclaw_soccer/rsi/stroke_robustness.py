"""Fixed paired teacher robustness bank; these are public TRAIN perturbations."""

import copy
from typing import Any

import numpy as np

CONTEXT_IDS = (4, 20, 21)


def robustness_courses(contexts: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    if set(contexts) != set(CONTEXT_IDS):
        raise ValueError("all three original source contexts required")
    courses: list[dict[str, Any]] = []
    for source in CONTEXT_IDS:
        origin = contexts[source]["case"]
        ball = np.asarray(origin["ball_world_initial"], dtype=np.float64)
        goal = np.asarray(origin["goal_world"], dtype=np.float64)
        velocity = np.asarray(origin["velocity_world_initial"], dtype=np.float64)
        if (
            ball.shape != (3,)
            or goal.shape != (3,)
            or velocity.shape != (2,)
            or not np.isfinite(np.r_[ball, goal, velocity]).all()
            or ball[2] != 0.11
            or origin["bank"] != "train"
            or origin["not_fresh"] is not True
        ):
            raise ValueError("finite regulation-ball consumed TRAIN origin required")
        heading = goal[:2] - ball[:2]
        norm = float(np.linalg.norm(heading))
        if norm < 0.1:
            raise ValueError("nondegenerate original goal direction required")
        forward = heading / norm
        side = np.array([-forward[1], forward[0]])
        for x, df in enumerate((-0.02, -0.01, 0.01, 0.02)):
            for y, ds in enumerate((-0.02, -0.01, 0.01, 0.02)):
                ctx = copy.deepcopy(contexts[source])
                point = ball.copy()
                point[:2] += df * forward + ds * side
                dvf = 0.03 if x % 2 else -0.03
                dvs = 0.03 if y % 2 else -0.03
                v = velocity + dvf * forward + dvs * side
                if not np.isfinite(np.r_[point, v]).all() or np.linalg.norm(v) > 0.6:
                    raise ValueError("entire perturbation bank must meet launch domain")
                ctx["case"]["ball_world_initial"] = point.tolist()
                ctx["case"]["velocity_world_initial"] = v.tolist()
                ctx.update(
                    context_id=source,
                    course_index=len(courses),
                    variant=4 * x + y,
                    perturbation=dict(
                        forward_m=df,
                        lateral_m=ds,
                        forward_velocity_mps=dvf,
                        lateral_velocity_mps=dvs,
                    ),
                )
                courses.append(ctx)
    return courses


def paired_robustness_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 96:
        raise ValueError("both arms of every one of48 courses required")
    summaries = {}
    for source_index, source in enumerate(CONTEXT_IDS):
        baseline, teacher = [], []
        for variant in range(16):
            index = 16 * source_index + variant
            pair = records[2 * index : 2 * index + 2]
            for r, arm in zip(pair, ("baseline", "teacher"), strict=True):
                if (
                    r["course_index"] != index
                    or r["context"] != source
                    or r["arm"] != arm
                    or type(r["healthy_quality"]) is not bool
                    or type(r["result"]["body_failed"]) is not bool
                    or r["assisted"] is not (arm == "teacher")
                    or r["PPO_actor_rows"] != 0
                ):
                    raise ValueError("complete ordered typed paired physical outcomes required")
            baseline.append(pair[0])
            teacher.append(pair[1])
        summaries[str(source)] = dict(
            courses=16,
            baseline_quality=sum(r["healthy_quality"] for r in baseline),
            teacher_quality=sum(r["healthy_quality"] for r in teacher),
            rescues=sum(
                not a["healthy_quality"] and b["healthy_quality"]
                for a, b in zip(baseline, teacher, strict=True)
            ),
            regressions=sum(
                a["healthy_quality"] and not b["healthy_quality"]
                for a, b in zip(baseline, teacher, strict=True)
            ),
            teacher_body_failures=sum(r["result"]["body_failed"] for r in teacher),
        )
    successes = sum(r["teacher_quality"] for r in summaries.values())
    net = sum(r["rescues"] - r["regressions"] for r in summaries.values())
    qualified = (
        successes >= 39
        and net > 0
        and all(
            r["teacher_quality"] >= 12
            and r["teacher_body_failures"] == 0
            and r["rescues"] >= r["regressions"]
            for r in summaries.values()
        )
    )
    return dict(
        contexts=summaries,
        courses=48,
        executions=96,
        teacher_successes=successes,
        paired_net_gain=net,
        teacher_locally_qualified=qualified,
        scope="public_TRAIN_goal_relative_position_velocity_grid",
        calibrated_probability=False,
        fresh_generalization_proven=False,
        student_learning_proven=False,
        promotion_authorized=False,
        goal_complete=False,
    )
