"""Independently read native exit traces and rerun the frozen velocity NN."""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def closed_producer(unit, head, root):
    """Read manager-issued completion even after transient-unit GC; never retry."""
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.@-]{0,190}\.service", unit) is None:
        raise ValueError("one explicit read-only service name required")
    status = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "MainPID",
                "-p",
                "ActiveState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
            timeout=10,
        ).splitlines()
    )
    if (
        status.get("MainPID") != "0"
        or status.get("ActiveState") != "inactive"
        or status.get("Result") != "success"
        or status.get("ExecMainStatus") != "0"
    ):
        raise ValueError("producer must no longer be running and have no observed failure")
    raw = subprocess.check_output(
        ["journalctl", "--user", "-u", unit, "-o", "json", "-n", "128", "--no-pager"],
        text=True,
        timeout=10,
    )
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("bounded producer journal required")
    events = [json.loads(line) for line in raw.splitlines()]
    manager = [
        event
        for event in events
        if event.get("_COMM") == "systemd"
        and event.get("_TRANSPORT") == "journal"
        and event.get("USER_UNIT") == unit
        and event.get("JOB_TYPE") == "start"
    ]
    if not manager:
        raise ValueError("missing actual systemd producer completion history")
    done = manager[-1]
    if (
        done.get("JOB_RESULT") != "done"
        or done.get("MESSAGE_ID") != "39f53479d3a045ac8e11786248231fbf"
        or not done.get("USER_INVOCATION_ID")
        or type(done.get("MESSAGE")) is not str
        or not done["MESSAGE"].startswith("Finished ")
    ):
        raise ValueError("latest producer invocation did not complete successfully")
    argv = shlex.split(done["MESSAGE"][len("Finished ") :].removesuffix("."))
    for flag, expected in (
        ("--expected-head", head),
        ("--output", str(root)),
        ("-m", "scripts.rsi_run_neural_exit_pilot"),
    ):
        if (
            argv.count(flag) != 1
            or argv.index(flag) + 1 >= len(argv)
            or argv[argv.index(flag) + 1] != expected
        ):
            raise ValueError("journal completion is for another source or output")
    return dict(
        invocation_id=done["USER_INVOCATION_ID"],
        cursor=done["__CURSOR"],
        source_head=head,
        output_root=str(root),
        manager_completion_verified=True,
        producer_restarted=False,
    )


def verify_forecast_accounting(row):
    """These counters belong to old entry forecasts, not the new exit future."""
    counters = row["forecast_counters"]
    keys = (
        "forecast_calls",
        "attempted_native_steps",
        "integrated_native_steps",
        "future_nn_calls",
        "maximum_native_steps",
    )
    if any(type(counters.get(k)) is not int or counters[k] < 0 for k in keys):
        raise ValueError("complete integer entry-forecast accounting required")
    entry = row["entry_control"]
    horizon = 0 if entry is None else min(128, 360 - entry)
    if (
        counters["forecast_calls"] > 37
        or counters["attempted_native_steps"] != counters["forecast_calls"] * horizon * 10
        or counters["integrated_native_steps"] > counters["attempted_native_steps"]
        or counters["integrated_native_steps"] != 10 * counters["future_nn_calls"]
        or counters["maximum_native_steps"] != 37 * 1280
    ):
        raise ValueError("bounded forecast requests/integrations/NN calls disagree")
    return counters["attempted_native_steps"], counters["integrated_native_steps"]


def verify_velocity_history(row, arrays, *, layout, model, params, session):
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
        experimental_observation98,
    )
    from rosclaw_soccer.providers.g1.velocity_policy import observation98, phase_delta, target29

    n, exit_frame = row["controls"], row["exit_control"]
    forces = arrays["ball_forces"][:, 0]
    foot = np.flatnonzero((forces[:, :2] > 1).any(axis=1))
    trigger = None if not len(foot) else int(foot[0] // 10 + 1)
    expected_exit = trigger if trigger is not None and trigger < n else None
    if row.get("protect_passing_entry_prediction", False):
        from rosclaw_soccer.rsi.observed_velocity_continuation import exit_after_entry_abstention

        if not exit_after_entry_abstention(row["controller_events"]):
            expected_exit = None
    if exit_frame != expected_exit:
        raise ValueError("exit must follow first completed physical foot-contact control")
    data = mujoco.MjData(model)
    data.qpos[layout.rq : layout.rq + 7] = [0, 0, 0.793, 1, 0, 0, 0]
    data.qpos[layout.qi] = np.asarray(params.defaults)
    x, y, vx = row["course"]
    data.qpos[layout.bq : layout.bq + 7] = [x, y, 0.11, 1, 0, 0, 0]
    data.qvel[layout.bv] = vx
    data.qvel[layout.bv + 4] = vx / 0.11
    phase, history = phase_delta(), np.zeros(29, np.float32)
    codec = experimental_observation98 if row["experimental_fast_approach"] else observation98
    receipts = row["observed_velocity_continuation"]
    if len(receipts) != n or arrays["velocity_shadow_observations"].shape != (n, 98):
        raise ValueError("one shadow observation and actual-target history per physical control")
    for frame in range(n):
        if frame:
            data.qpos[:] = arrays["qpos"][frame * 10 - 1]
            data.qvel[:] = arrays["qvel"][frame * 10 - 1]
        q, v = layout.canonical(data)
        command = arrays["commands"][frame]
        obs, phase = codec(q, v, v[3:6], command, history, phase, params)
        stored = arrays["velocity_shadow_observations"][frame]
        # Angular-velocity body Jacobian arithmetic may differ at float roundoff.
        if not np.array_equal(obs[3:], stored[3:]) or not np.allclose(
            obs[:3], stored[:3], rtol=0, atol=1e-6
        ):
            raise ValueError("stored observation differs from actual body/phase/target history")
        output = session.run(["actions"], {"obs": stored[None]})[0]
        if output.shape != (1, 29) or output.dtype != np.float32:
            raise ValueError("frozen CPU NN action ABI differs")
        shadow = target29(output[0], params)
        actual = arrays["joint_targets"][frame]
        active = frame < row["entry_control"] or (exit_frame is not None and frame >= exit_frame)
        if arrays["velocity_control_active"][frame] != active:
            raise ValueError("wrong executed skill identity")
        if active and not np.array_equal(actual, shadow):
            raise ValueError("executed velocity target differs from original CPU NN")
        original = np.array_equal(actual, shadow)
        history = (
            output[0].copy()
            if original
            else (
                (actual - np.asarray(params.action_offset, np.float64))
                / np.asarray(params.action_scale, np.float64)
            ).astype(np.float32)
        )
        error = float(np.max(abs(target29(history, params) - actual)))
        receipt = receipts[frame]
        if (
            receipt["completed_control"] != frame + 1
            or receipt["original_shadow_action_retained"] != original
            or receipt["phase"] != float(phase)
            or error > 5e-7
            or receipt["maximum_target_decode_error_rad"] != error
            or receipt["maximum_encoded_action_magnitude"] != float(np.max(abs(history)))
            or not np.array_equal(np.array(receipt["actual_target_history"], np.float32), history)
            or receipt["phase_reset"]
            or receipt["physical_target_modified"]
            or receipt["hardware_authorized"]
        ):
            raise ValueError("actual-target history/codec/phase commitment differs")
        if exit_frame is not None and frame >= exit_frame:
            previous = arrays["commands"][frame - 1]
            expected = previous + np.clip(-previous, -0.05, 0.05)
            if not np.array_equal(command, expected):
                raise ValueError("zero-intent command rate or post-contact clock differs")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "backbone-model", "champion-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("pilot-hash", "pilot-unit", "producer-head", "expected-head"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--transfer18", action="store_true")
    parser.add_argument("--protected-audit", type=Path)
    parser.add_argument("--protected-audit-hash")
    a = parser.parse_args()
    from rosclaw_soccer.rsi.neural_exit_transfer import (
        EXIT_TRANSFER_INDICES,
        EXIT_TRANSFER_WITH_CONTROLS,
    )

    indices = EXIT_TRANSFER_WITH_CONTROLS if a.transfer18 else (0, 1, 2, 30)
    world_ceiling = 2 * len(indices)
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    terminal = closed_producer(a.pilot_unit, a.producer_head, a.pilot)
    cp = a.pilot / "complete.json"
    if a.output.exists() or digest(cp) != a.pilot_hash:
        raise ValueError("new readback root and exact completed producer required")
    pilot = json.loads(cp.read_text())
    if (
        pilot["status"] != "OBSERVED_NEURAL_EXIT_PILOT_COMPLETE_NOT_TEACHER_QUALIFIED"
        or pilot["source_head"] != a.producer_head
        or [r["course_index"] for r in pilot["results"]] != list(indices)
        or pilot["actual_worlds"] != world_ceiling
        or pilot["neural_optimizer_updates"] != 0
        or pilot.get("transfer18", False) != a.transfer18
        or (a.transfer18 and pilot.get("protect_passing_entry_prediction") is not True)
    ):
        raise ValueError("complete explicitly declared pilot/transfer scope required")
    pins = dict(pilot["input_file_hashes"])
    if a.transfer18:
        if (
            a.protected_audit is None
            or not a.protected_audit_hash
            or pins.get(str(a.protected_audit)) != a.protected_audit_hash
        ):
            raise ValueError("same producer-bound protected source audit required")
        from scripts.rsi_run_neural_exit_pilot import admit_protected_exit_source

        admit_protected_exit_source(a.protected_audit, a.protected_audit_hash, pins)
    elif a.protected_audit is not None or a.protected_audit_hash:
        raise ValueError("protected source binding is only for explicit transfer18")
    pins[str(cp)] = a.pilot_hash
    source_pins(pins)
    check_pins(pins)
    from scripts.rsi_teacher_physics_context import load_teacher_context

    context = load_teacher_context(pins, a.backbone_model)
    import numpy as np

    from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    assets = context["assets"]
    velocity = G1VelocityPolicy(
        assets["velocity_model"],
        assets["velocity_config"],
        expected_model_hash=assets["velocity_model_hash"],
        expected_config_hash=assets["velocity_config_hash"],
    )
    layout = context["layout"]
    champion = json.loads((a.champion_root / "complete.json").read_text())
    if pins.get(str(a.champion_root / "complete.json")) != digest(
        a.champion_root / "complete.json"
    ):
        raise ValueError("bound old champion identity differs")
    original_bank = None
    if a.transfer18:
        from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

        original_bank = search_course_bank(expanded_public=True)
        bank_hash = preflight_receiving_courses(tuple(original_bank[i] for i in indices))
        if pilot.get("course_bank_preflight_hash") != bank_hash:
            raise ValueError("complete fixed nominal transfer bank commitment differs")
    results, rescues, regressions = [], [], []
    steps = calls = forecast_requested = forecast_integrated = 0
    for item in pilot["results"]:
        index = item["course_index"]
        arms, outcomes, health = {}, {}, {}
        events = {}
        for arm in ("B", "C"):
            ref = item["arms"][arm]
            path = a.pilot / ref["path"]
            if path.parent != a.pilot / f"course-{index:02d}" or digest(path) != ref["hash"]:
                raise ValueError("bound actual arm record required")
            pins[str(path)] = ref["hash"]
            row = json.loads(path.read_text())
            requested, integrated = verify_forecast_accounting(row)
            forecast_requested += requested
            forecast_integrated += integrated
            if original_bank is not None and tuple(row["course"]) != original_bank[index].course:
                raise ValueError("actual transfer course differs from the preregistered bank")
            if (
                row["course_index"] != index
                or row["arm"] != arm
                or (
                    a.transfer18
                    and arm == "C"
                    and row.get("protect_passing_entry_prediction") is not True
                )
            ):
                raise ValueError("raw row scope or protected transfer mechanism differs")
            raw_path = path.parent / row["artifact"]["path"]
            if raw_path.parent != path.parent or digest(raw_path) != row["artifact"]["hash"]:
                raise ValueError("exact raw trajectory required")
            pins[str(raw_path)] = row["artifact"]["hash"]
            with np.load(raw_path, allow_pickle=False) as raw:
                arrays = {k: raw[k].copy() for k in raw.files}
            outcome = relabel_arrays(
                arrays, controls=row["controls"], ball_address=layout.bq, root_address=layout.rq
            )
            q = arrays["qpos"][:, layout.rq : layout.rq + 7]
            healthy = bool(
                row["controls"] == 360
                and (q[:, 2] >= 0.65).all()
                and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) >= 0.9).all()
            )
            if (
                hash_json(outcome) != hash_json(row["outcome"])
                or healthy != row["healthy_full_horizon"]
                or ref["task_pass"] != outcome["task_pass"]
                or ref["healthy"] != healthy
                or row["native_steps"] != row["controls"] * 10
                or (
                    abs(arrays["torque"][:, layout.ai]) > 0.85 * context["hard_limits"] + 1e-12
                ).any()
            ):
                raise ValueError("raw physical score/body/clock/torque differs")
            if arm == "B":
                old_ref = champion["results"][index]["arms"]["C"]
                old_path = a.champion_root / old_ref["record"]
                if pins.get(str(old_path)) != digest(old_path):
                    raise ValueError("old actual course identity differs")
                old = json.loads(old_path.read_text())
                old_raw = old_path.parent / old["artifact"]["path"]
                if pins.get(str(old_raw)) != digest(old_raw):
                    raise ValueError("old actual trajectory identity differs")
                with np.load(old_raw, allow_pickle=False) as expected:
                    if set(expected.files) != set(arrays) or any(
                        not np.array_equal(expected[k], v) for k, v in arrays.items()
                    ):
                        raise ValueError("baseline whole trajectory not exact")
            else:
                prefix = row["controls"] if row["exit_control"] is None else row["exit_control"]
                for key, previous in arms["B"].items():
                    n = prefix * 10 if key in ("qpos", "qvel", "torque", "ball_forces") else prefix
                    if not np.array_equal(previous[:n], arrays[key][:n]):
                        raise ValueError("pre-exit prefix differs")
                if row["controller_events"] != events["B"]:
                    raise ValueError("pre-exit planner changed")
                calls += verify_velocity_history(
                    row,
                    arrays,
                    layout=layout,
                    model=context["model"],
                    params=velocity.params,
                    session=velocity.session,
                )
            steps += row["native_steps"]
            arms[arm], outcomes[arm], health[arm] = arrays, outcome, healthy
            events[arm] = row["controller_events"]
        if not outcomes["B"]["task_pass"] and outcomes["C"]["task_pass"] and health["C"]:
            rescues.append(index)
        if any(outcomes["B"][k] and not outcomes["C"][k] for k in ("task_pass", "safe")) or (
            health["B"] and not health["C"]
        ):
            regressions.append(index)
        results.append(
            dict(course_index=index, B=outcomes["B"], C=outcomes["C"], C_healthy=health["C"])
        )
    if steps != pilot["actual_native_steps"] or steps > 3600 * world_ceiling:
        raise ValueError("actual physical accounting differs")
    if (
        forecast_requested != pilot["forecast_requested_native_steps"]
        or forecast_requested > world_ceiling * 37 * 1280
    ):
        raise ValueError("total entry-forecast budget differs")
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "complete.json",
        dict(
            status="NEURAL_EXIT_RAW_HISTORY_AND_CPU_NN_VERIFIED_NOT_TEACHER",
            source_head=a.expected_head,
            producer_terminal=terminal,
            transfer18=a.transfer18,
            transfer_rescues=[i for i in rescues if i in EXIT_TRANSFER_INDICES]
            if a.transfer18
            else [],
            transfer_research_gate_met=bool(
                a.transfer18
                and not regressions
                and all(r["C_healthy"] for r in results)
                and any(i in EXIT_TRANSFER_INDICES for i in rescues)
            ),
            input_file_hashes=pins,
            results=results,
            rescues=rescues,
            regressions=regressions,
            actual_native_steps=steps,
            forecast_requested_native_steps=forecast_requested,
            forecast_integrated_native_steps=forecast_integrated,
            cpu_velocity_nn_readback_calls=calls,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            qualified_for_expansion=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
