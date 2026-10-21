"""Independent body/NN/IK/native verification of one composite teacher branch."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.paid_continuity_audit import audit_terminal, identical, snapshot_fields
from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
from rosclaw_soccer.rsi.paid_neural_stand_audit import audit_neural_stand
from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_kick_entry


def audit_successor_contact(
    auditor: Any,
    observer: Any,
    replay: Any,
    actor: Any,
    record: dict[str, Any],
    *,
    read: Any,
    entry: dict[str, np.ndarray],
    previous: np.ndarray,
    motion: dict[str, np.ndarray],
    center: int,
    latent: np.ndarray,
    pure: bool,
    count_inference: Any,
) -> dict[str, Any]:
    import mujoco

    row = record["kick"]
    z = read(row)
    audit_kick_entry(z, entry)
    result = auditor.audit(
        z,
        row,
        horizon=len(motion["q"]),
        previous_action=previous,
        reference_prefix=True,
        intent=None if pure else (center, latent),
    )
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    model = auditor.layout.model
    kin = mujoco.MjData(model)
    foot = "left" if "left" in row["source"] else "right"
    intent = None if pure else FootTaskSpaceIntent(model, foot=foot, center=center, latent=latent)
    proxy = SimpleNamespace(
        env=SimpleNamespace(model=model, data=kin, pelvis_body_id=auditor.layout.pelvis),
        offset=auditor.offset,
        scale=auditor.scale,
        order=auditor.order,
        layout=auditor.layout,
    )
    goal = entry["mocap_pos"][auditor.goal_mocap_id]
    with torch.no_grad():
        for j in range(len(z["NN_action"])):
            obs = observer.observe(
                z["native_qpos_before"][j],
                z["native_qvel_before"][j],
                motion["q"][j - 1] if j else np.zeros(29, np.float32),
                previous,
                goal,
            )
            if j:
                obs[29:58], obs[61:64] = motion["v"][j - 1], motion["w"][j - 1]
            if not np.array_equal(obs, z["NN_obs"][j]) or any(
                not np.array_equal(t.numpy(), z[key][j])
                for t, key in zip(state, ("NN_h_before", "NN_c_before"), strict=True)
            ):
                raise ValueError("teacher branch actual observer/reference/private memory differs")
            mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
            count_inference()
            nominal = mean[0, 0].numpy().copy()
            if not np.array_equal(nominal, z["NN_nominal_action"][j]) or any(
                not np.array_equal(t.numpy(), z[key][j])
                for t, key in zip(state, ("NN_h_after", "NN_c_after"), strict=True)
            ):
                raise ValueError("teacher branch own frozen base differs")
            action = nominal
            if intent is not None:
                kin.qpos[:] = z["native_qpos_before"][j]
                kin.qvel[:] = z["native_qvel_before"][j]
                mujoco.mj_forward(model, kin)
                proxy.frame = j
                action, plan = intent(proxy, nominal)
                if any(
                    not np.array_equal(value, z["PLAN_" + key][j]) for key, value in plan.items()
                ):
                    raise ValueError("causal bounded scratch-only Cartesian IK differs")
            if not np.array_equal(action, z["NN_action"][j]):
                raise ValueError("actual composite teacher command differs")
            previous = action.copy()
    end = read(record["kick_terminal"])
    audit_terminal(auditor, z, end)
    for key in ("mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "act"):
        if not np.array_equal(entry[key], end[key]):
            raise ValueError("teacher modified external goal/force/activation")
    kick_replay = replay.replay(z, entry=entry, terminal=end)
    native, neural = result["native_steps"], result["neural_rows"]
    healthy = stable = False
    if result["body_failed"]:
        if record["stand"] is not None:
            raise ValueError("failed teacher body cannot start standing")
    else:
        sr = record["stand"]
        stand = read(sr)
        checked = audit_neural_stand(
            auditor,
            observer,
            actor,
            stand,
            sr,
            entry=end,
            pose=motion["q"][-1],
            previous_action=previous,
            initial_memory=(state[0].numpy(), state[1].numpy()),
            start_control=0,
            maximum_controls=100,
            count_inference=count_inference,
        )
        replay.replay(stand, entry=end, terminal=snapshot_fields(stand, "terminal_"))
        native += checked["native_steps"]
        neural += checked["neural_rows"]
        healthy, stable = checked["healthy"], checked["stability_observed"]
        identical(snapshot_fields(stand, "initial_"), end)
    return dict(
        native=native,
        neural=neural,
        stand_healthy=healthy,
        stand_stable=stable,
        quality=result["quality"],
        nonfoot_contacts=kick_replay["nonfoot_contacts"],
    )
