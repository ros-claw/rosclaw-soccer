"""Finite paired OLD/learned mean-policy TRAIN exam; never promotion evidence."""

from typing import Any


def exam_config(state: int, *, receiver_role_study: bool = False) -> dict[str, Any]:
    if type(receiver_role_study) is not bool:
        raise ValueError("explicit separate receiver-role protocol required")
    if type(state) is not int or state not in (25, 26):
        raise ValueError("two preregistered consumed TRAIN states required")
    cfg = dict(
        protocol="soccer.refinement_PPO_paired_mean_TRAIN12.v1",
        state_id=state,
        arms=["old", "learned", "old", "learned"],
        initial_conditions=3,
        worlds=12,
        maximum_native_steps=48000,
        maximum_actor_calls=9600,
        original_total_authority_rad=0.2,
        A_neural_recovery_controls=100,
        sampling=False,
        external_teacher=False,
        Fresh_cases_read=0,
        optimizer_updates=0,
    )
    if receiver_role_study:
        cfg.update(
            protocol="soccer.receiver_anchor_plane_paired_TRAIN12.v2",
            arms=["pursuit", "anchor_plane", "pursuit", "anchor_plane"],
            fixed_A_behavior=True,
            A_behavior_generation_index=2,
            A_behavior_cumulative_actor_steps=12,
            receiver_geometric_teacher=True,
            receiver_learned=False,
        )
    return cfg


def verify_exam(report: dict[str, Any], courses: list[dict[str, Any]]) -> None:
    role = report["config"].get("protocol") == "soccer.receiver_anchor_plane_paired_TRAIN12.v2"
    cfg = exam_config(report["config"]["state_id"], receiver_role_study=role)
    rows = report["records"]
    if report["config"] != cfg or len(rows) != 12 or len(courses) != 3:
        raise ValueError("all3 initial x2 arms x2 repeats required")
    totals = dict(native_steps=0, actor_calls=0)
    for i, r in enumerate(rows):
        if (
            r["index"] != i
            or r["arm"] != cfg["arms"][i % 4]
            or r["course"] != courses[i // 4]
            or r["course"]["assisted"] is not False
            or r["native_steps"] != 10 * r["controls"]
            or r["actor_calls"] != 2 * r["controls"]
            or not 1 <= r["controls"] <= 400
            or not 0 <= r["stand_controls"] <= 100
            or r["neural_actor_optimizer_updates"] != 0
            or r.get("receiver_intent_mode", "pursuit-v1")
            != ("anchor-plane-v2" if role and r["arm"] == "anchor_plane" else "pursuit-v1")
            or (not role and "receiver_intent_mode" in r)
            or r["termination"] not in ("BODY_FAILURE", "FULL_MOTION_PLUS100NN_RECOVERY")
            or (r["termination"] == "FULL_MOTION_PLUS100NN_RECOVERY" and r["stand_controls"] != 100)
            or any(k in r for k in ("sampling_seed", "feedback_variant", "student_prefix_controls"))
        ):
            raise ValueError("exact unassisted mean-policy own-native counters required")
        if i % 4 >= 2 and {k: v for k, v in r.items() if k not in ("index", "artifacts")} != {
            k: v for k, v in rows[i - 2].items() if k not in ("index", "artifacts")
        }:
            raise ValueError("deterministic same-arm repeats must match completely")
        for k in totals:
            totals[k] += r[k]
    if totals != report["counts"] or any(totals[k] > cfg["maximum_" + k] for k in totals):
        raise ValueError("all bounded charged clocks required")
