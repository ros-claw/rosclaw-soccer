"""Independent evidence reader for full-course recovery transfer trials."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_balance_audit import audit_balance_recovery
from rosclaw_soccer.rsi.paid_continuity_audit import (
    audit_practice_boundary,
    audit_terminal,
    identical,
    snapshot_fields,
)


def audit_recovery_trial(
    auditor: Any,
    world: dict[str, Any],
    *,
    data: dict[str, dict[str, np.ndarray]],
    source_blocks: dict[str, dict[str, np.ndarray]],
    first_boundary_status: str,
    motion: dict[str, np.ndarray],
    actor: Any,
    latent: np.ndarray,
) -> dict[str, Any]:
    used: set[str] = set()
    counts = dict(native=0, neural=0, own_neural=0)
    quality: list[bool] = []
    horizon = len(motion["q"])

    def block(key: str) -> dict[str, np.ndarray]:
        used.add(key)
        value = data[key]
        if key in source_blocks:
            identical(value, source_blocks[key])
        return value

    def skill(
        index: int,
        previous: np.ndarray,
        *,
        prefix: bool = False,
        entry: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
        row = world["skills"][index]
        if row["index"] != index:
            raise ValueError("actual ordered skill list required")
        z = block(f"skill{index}")
        if entry is not None:
            for key, value in entry.items():
                if "initial_" + key in z and not np.array_equal(z["initial_" + key], value):
                    raise ValueError("next motor reset physical integration state")
        result: dict[str, Any] = auditor.audit(
            z, row["result"], horizon=horizon, previous_action=previous, reference_prefix=prefix
        )
        terminal = block(f"terminal{index}")
        audit_terminal(auditor, z, terminal)
        counts["native"] += result["native_steps"]
        counts["neural"] += result["neural_rows"]
        quality.append(result["quality"])
        return z, terminal, result

    def boundary(
        index: int, prefix: str, previous: dict[str, np.ndarray], terminal: dict[str, np.ndarray]
    ) -> tuple[str, dict[str, np.ndarray] | None]:
        record = world["boundaries"][index]
        if index == 0 and record["status"] != first_boundary_status:
            raise ValueError("qualified first boundary status changed")
        if record["status"] == "FIXTURE_REJECTED":
            if any(k.startswith(prefix + "-") for k in data):
                raise ValueError("rejected fixture cannot mutate state")
            return record["status"], None
        result = audit_practice_boundary(
            auditor,
            previous=previous,
            terminal=terminal,
            drop=block(prefix + "-drop"),
            hold=block(prefix + "-hold") if prefix + "-hold" in data else None,
            boundary=block(prefix + "-boundary") if prefix + "-boundary" in data else None,
            source=world["skills"][0]["result"]["source"],
        )
        if result["status"] != record["status"]:
            raise ValueError("actual ball-return/settle boundary differs")
        counts["native"] += result["native_steps"]
        return result["status"], result["state"]

    first, terminal, checked = skill(0, np.zeros(29, np.float32))
    expected_skills, expected_boundaries = 1, 0
    rec_healthy = rec_cert = False
    if checked["body_failed"]:
        status = "FIRST_BODY_FAILURE"
    else:
        expected_boundaries = 1
        b, entry = boundary(0, "between0-1", first, terminal)
        if b != "READY":
            status = "FIRST_BOUNDARY_" + b
        else:
            expected_skills = 2
            second, terminal, checked = skill(1, first["NN_action"][-1], prefix=True, entry=entry)
            if checked["body_failed"]:
                status = "SECOND_REFERENCE_BODY_FAILURE"
            else:
                rec = block("recovery")
                checked_rec = audit_balance_recovery(
                    auditor,
                    rec,
                    entry=terminal,
                    anchor=second["target"][-1],
                    anchor_action=second["NN_action"][-1],
                    latent=latent,
                )
                if world["recovery"] is None or any(
                    world["recovery"].get(k) != v for k, v in checked_rec.items()
                ):
                    raise ValueError("actual recovery metrics differ")
                end = snapshot_fields(rec, "terminal_")
                audit_terminal(auditor, rec, end)
                for key in ("mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "act"):
                    if not np.array_equal(end[key], terminal[key]):
                        raise ValueError("recovery modified goal/forces/activation state")
                counts["native"] += checked_rec["native_steps"]
                rec_healthy, rec_cert = (
                    checked_rec["healthy"],
                    checked_rec["recovery_stability_observed"],
                )
                if not rec_healthy:
                    status = "RECOVERY_BODY_FAILURE"
                else:
                    expected_boundaries = 2
                    # Compatibility adapter carries a TEACHER's actually executed
                    # action. This does NOT claim recovery was neural inference.
                    proxy = dict(target=rec["target"], NN_action=rec["RECOVERY_actual_action"])
                    b, entry = boundary(1, "post-recovery", proxy, end)
                    if b != "READY":
                        status = "RECOVERY_CONTINUATION_" + b
                    else:
                        expected_skills = 3
                        third, terminal, checked = skill(
                            2, rec["RECOVERY_actual_action"][-1], entry=entry
                        )
                        import torch

                        n = len(third["NN_action"])
                        obs = third["NN_obs"]
                        if (
                            obs[0, :58].any()
                            or obs[0, 61:64].any()
                            or not np.array_equal(obs[1:, :29], motion["q"][: n - 1])
                            or not np.array_equal(obs[1:, 29:58], motion["v"][: n - 1])
                            or not np.array_equal(obs[1:, 61:64], motion["w"][: n - 1])
                        ):
                            raise ValueError("third reference clock/cache was not reset")
                        state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                        with torch.no_grad():
                            for j, x in enumerate(obs):
                                if any(
                                    not np.array_equal(t.numpy(), third[k][j])
                                    for t, k in zip(
                                        state, ("NN_h_before", "NN_c_before"), strict=True
                                    )
                                ):
                                    raise ValueError("third private neural memory differs")
                                action, state = actor(torch.from_numpy(x).reshape(1, 1, 160), state)
                                if not np.array_equal(
                                    action[0, 0].numpy(), third["NN_action"][j]
                                ) or any(
                                    not np.array_equal(t.numpy(), third[k][j])
                                    for t, k in zip(
                                        state, ("NN_h_after", "NN_c_after"), strict=True
                                    )
                                ):
                                    raise ValueError("third own neural output/memory differs")
                                counts["own_neural"] += 1
                        status = (
                            "THIRD_BODY_FAILURE"
                            if checked["body_failed"]
                            else "THREE_SKILLS_WITH_RECOVERY_EXECUTED"
                        )
    if (
        used != set(data)
        or (world["recovery"] is None) != ("recovery" not in used)
        or len(world["skills"]) != expected_skills
        or len(world["boundaries"]) != expected_boundaries
        or world["status"] != status
        or world["stable"] != (status == "THREE_SKILLS_WITH_RECOVERY_EXECUTED")
        or world["quality_by_skill"] != quality
        or world["body_initializations"] != 1
        or world["body_resets"] != 0
        or world["native_steps"] != counts["native"]
        or world["neural_rows"] != counts["neural"]
        or not np.isclose(world["final_sim_time"], counts["native"] * 0.002, rtol=0, atol=1e-10)
    ):
        raise ValueError("complete transfer branch, coverage, grading or accounting differs")
    return dict(
        **counts,
        quality_by_skill=quality,
        status=status,
        stable=world["stable"],
        recovery_healthy=rec_healthy,
        recovery_certificate=rec_cert,
    )
