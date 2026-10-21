"""Read-only, independently reconstructed continuous kick/stand lifecycles."""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_balance_audit import audit_balance_recovery
from rosclaw_soccer.rsi.paid_continuity_audit import (
    audit_terminal,
    identical,
    require_integration_fields,
    snapshot_fields,
)
from rosclaw_soccer.rsi.paid_motor_schedule import lifecycle_motor_schedule
from rosclaw_soccer.rsi.paid_neural_stand_audit import audit_neural_stand


def audit_kick_entry(raw: dict[str, np.ndarray], entry: dict[str, np.ndarray]) -> None:
    """Legacy kick raw stores seven entry fields, not a ten-field snapshot.

    Full ten-field before/after handoff proof is checked separately. Never
    fabricate missing force/activation entry arrays in the legacy raw.
    """
    initial = snapshot_fields(raw, "initial_")
    keys = {"qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "time"}
    if set(initial) != keys:
        raise ValueError("exact seven-field legacy kick entry schema required")
    identical(initial, {key: entry[key] for key in keys})


def audit_ball_drop(
    auditor: Any, raw: dict[str, np.ndarray], entry: dict[str, np.ndarray], source: str
) -> dict[str, np.ndarray]:
    """Only the declared ball fixture may write state; no robot/time rewrite."""
    before = snapshot_fields(raw, "before_")
    require_integration_fields(before)
    identical(before, entry)
    q, _ = auditor.canonical(before["qpos"], before["qvel"])
    performed = bool(raw["performed"])
    if performed != bool(np.linalg.norm(q[36:38] - q[:2]) >= 2):
        raise ValueError("ball return eligibility differs")
    after = snapshot_fields(raw, "after_")
    if not performed:
        if after or set(raw) != {"performed", *("before_" + k for k in before)}:
            raise ValueError("unperformed fixture wrote state")
        return before
    require_integration_fields(after)
    x = auditor.layout
    restored = {k: v.copy() for k, v in after.items()}
    restored["qpos"][x.bq : x.bq + 7] = before["qpos"][x.bq : x.bq + 7]
    restored["qvel"][x.bv : x.bv + 6] = before["qvel"][x.bv : x.bv + 6]
    identical(restored, before)
    direction = before["mocap_pos"][auditor.goal_mocap_id, :2] - q[:2]
    direction /= np.linalg.norm(direction)
    sign = 1 if "left" in source else -1
    point = q[:2] + 0.8 * direction + sign * 0.15 * np.array([-direction[1], direction[0]])
    if (
        not np.array_equal(after["qpos"][x.bq : x.bq + 7], np.r_[point, 0.4, 1.0, 0, 0, 0])
        or after["qvel"][x.bv : x.bv + 6].any()
    ):
        raise ValueError("declared ball return location/velocity differs")
    return after


def audit_standing_lifecycle(
    auditor: Any,
    observer: Any,
    actor: Any,
    world: dict[str, Any],
    data: dict[str, dict[str, np.ndarray]],
    motion: dict[str, np.ndarray],
    first_prefix: dict[str, np.ndarray],
    *,
    variant: int,
    skills: int,
    count_inference: Any,
    motor_schedule: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    schedule = lifecycle_motor_schedule(variant, skills, motor_schedule)
    if (
        motor_schedule is None
        and "motor_schedule" in world
        or motor_schedule is not None
        and world.get("motor_schedule") != list(schedule)
    ):
        raise ValueError("motor schedule must match an explicit independent caller commitment")
    used: set[str] = set()
    native = neural = 0
    quality: list[bool] = []
    healthy: list[bool] = []
    certificates: list[bool] = []
    previous = np.zeros(29, np.float32)
    entry = None
    status = "STARTED"

    def block(key: str, row: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        used.add(key)
        value = data[key]
        if row is not None and row["raw_path"].rsplit("/", 1)[-1] != key + ".npz":
            raise ValueError("phase record does not name its own raw data")
        return value

    if "initial" in world:
        entry = block("world-initial", world["initial"])
        require_integration_fields(entry)

    for k in range(skills):
        if k >= len(world["kicks"]) or world["kicks"][k]["index"] != k:
            raise ValueError("actual ordered kick list required")
        kick = world["kicks"][k]
        row = kick["result"]
        z = block(f"kick{k}", row)
        if k == 0:
            identical(z, first_prefix)
        if entry is not None:
            audit_kick_entry(z, entry)
        checked = auditor.audit(
            z, row, horizon=len(motion["q"]), previous_action=previous, reference_prefix=True
        )
        native += checked["native_steps"]
        neural += checked["neural_rows"]
        quality.append(checked["quality"])
        model = actor if schedule[k] == "student" else actor.base
        state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
        goal = z["initial_mocap_pos"][auditor.goal_mocap_id]
        with torch.no_grad():
            for j in range(len(z["NN_action"])):
                pose = motion["q"][j - 1] if j else np.zeros(29, np.float32)
                obs = observer.observe(
                    z["native_qpos_before"][j], z["native_qvel_before"][j], pose, previous, goal
                )
                if j:
                    obs[29:58] = motion["v"][j - 1]
                    obs[61:64] = motion["w"][j - 1]
                if not np.array_equal(obs, z["NN_obs"][j]) or any(
                    not np.array_equal(t.numpy(), z[key][j])
                    for t, key in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
                ):
                    raise ValueError("kick actual body/reference/private memory differs")
                mean, state = model(torch.from_numpy(obs).reshape(1, 1, 160), state)
                count_inference()
                if not np.array_equal(mean[0, 0].numpy(), z["NN_action"][j]) or any(
                    not np.array_equal(t.numpy(), z[key][j])
                    for t, key in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
                ):
                    raise ValueError("own frozen kick network differs")
                previous = mean[0, 0].numpy().copy()
        terminal = block(f"kick-terminal{k}", kick["terminal"])
        audit_terminal(auditor, z, terminal)
        if entry is not None and any(
            not np.array_equal(terminal[key], entry[key])
            for key in ("mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "act")
        ):
            raise ValueError("kick changed external goal/force/activation state")
        if checked["body_failed"]:
            status = "KICK_BODY_FAILURE"
            break
        sr = world["stands"][k]
        if sr["index"] != k:
            raise ValueError("ordered standing phases required")
        stand = block(f"stand{k}", sr["result"])
        pose = motion["q"][-1]
        if variant:
            sc = audit_neural_stand(
                auditor,
                observer,
                actor.base,
                stand,
                sr["result"],
                entry=terminal,
                pose=pose,
                previous_action=previous,
                initial_memory=(state[0].numpy().copy(), state[1].numpy().copy()),
                start_control=0,
                maximum_controls=100,
                count_inference=count_inference,
            )
            previous = sc["previous_action"]
            memory = sc["state"]
        else:
            sc = audit_balance_recovery(
                auditor,
                stand,
                entry=terminal,
                anchor=z["target"][-1],
                anchor_action=previous,
                latent=np.zeros((9, 4)),
            )
            if any(sr["result"].get(key) != value for key, value in sc.items()):
                raise ValueError("fixed-target standing metrics differ")
            sc["stability_observed"] = sc["recovery_stability_observed"]
            sc["neural_rows"] = 0
        native += sc["native_steps"]
        neural += sc["neural_rows"]
        healthy.append(sc["healthy"])
        certificates.append(sc["stability_observed"])
        entry = snapshot_fields(stand, "terminal_")
        audit_terminal(auditor, stand, entry)
        if not sc["healthy"]:
            status = "STAND_BODY_FAILURE"
            break
        if k == skills - 1:
            status = "ALL_KICKS_AND_STANDS_EXECUTED"
            break
        # A rejected overlap has no state writes and no fabricated return rows.
        if len(world["boundaries"]) <= k:
            if world["status"] != "FIXTURE_REJECTED":
                raise ValueError("healthy continuation silently omitted")
            status = "FIXTURE_REJECTED"
            break
        boundary = world["boundaries"][k]
        prefix = f"between{k}-{k + 1}-"
        drop = block(prefix + "drop", boundary["drop"])
        entry = audit_ball_drop(auditor, drop, entry, row["source"])
        if bool(drop["performed"]):
            settle_row = boundary["settle"]
            settle = block(prefix + "settle", settle_row)
            if variant:
                sc = audit_neural_stand(
                    auditor,
                    observer,
                    actor.base,
                    settle,
                    settle_row,
                    entry=entry,
                    pose=pose,
                    previous_action=previous,
                    initial_memory=memory,
                    start_control=100,
                    maximum_controls=50,
                    count_inference=count_inference,
                )
                previous = sc["previous_action"]
                settled, good = sc["healthy"], settle_row["ball_settled"]
            else:
                sc = auditor.audit_hold(settle, entry, stand["target"][-1])
                sc["neural_rows"] = 0
                settled, good = not sc["body_failed"], sc["settled"]
                if settle_row["healthy"] != settled or settle_row["ball_settled"] != good:
                    raise ValueError("fixed-target settle metrics differ")
            native += sc["native_steps"]
            neural += sc["neural_rows"]
            entry = snapshot_fields(settle, "terminal_")
            audit_terminal(auditor, settle, entry)
            if not settled or not good:
                status = "SETTLE_BODY_FAILURE" if not settled else "BALL_NOT_SETTLED"
                break
        elif boundary["settle"] is not None:
            raise ValueError("unperformed fixture added settle physics")
        if boundary["boundary"] is None:
            if world["status"] != "HANDOFF_CONTACT_BLOCKED":
                raise ValueError("healthy continuation lost its handoff record")
            status = "HANDOFF_CONTACT_BLOCKED"
            break
        handoff = block(prefix + "boundary", boundary["boundary"])
        identical(snapshot_fields(handoff, "before_"), entry)
        identical(snapshot_fields(handoff, "after_"), entry)
        if not np.array_equal(handoff["previous_actual_action"], previous):
            raise ValueError("handoff erased actual last motor action")
    if (
        used != set(data)
        or len(world["kicks"]) != len(quality)
        or len(world["stands"]) != len(healthy)
        or world["quality_by_kick"] != quality
        or world["status"] != status
        or world["stable"] != (status == "ALL_KICKS_AND_STANDS_EXECUTED")
        or world["native_steps"] != native
        or world["neural_rows"] != neural
        or world["body_initializations"] != 1
        or world["body_resets"] != 0
        or world["variant"] != variant
        or world["requested_skills"] != skills
        or world["old_exam_result"] is not False
        or world["optimizer_updates"] != 0
        or world["team_game"] is not False
        or world["hardware_authorized"] is not False
        or not np.isclose(world["final_sim_time"], native * 0.002, rtol=0, atol=1e-10)
    ):
        raise ValueError("lifecycle branch/coverage/grading/accounting differs")
    return dict(
        native=native,
        neural=neural,
        stable=world["stable"],
        status=status,
        quality_by_kick=quality,
        stand_healthy=healthy,
        stand_certificates=certificates,
    )
