"""Whole-six independent recorded-control branch integration; no neural learning."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.recorded_feedback_replay import RecordedFeedbackReplay
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_impact_torque_authority_screen import equal
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg, *, early80=False):
    if (
        type(early80) is not bool
        or cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile")
        != (
            "six-recorded-early-feedback80-replay-v1"
            if early80
            else "six-recorded-feedback-replay-v1"
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("recorded_step_budget", 18000),
                ("branch_step_budget", 3360 if early80 else 1680),
                ("combined_step_budget", 21360 if early80 else 19680),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
                ("new_closed_loop_policy_trials", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "training_authorized",
                "teacher_qualified",
                "private_fresh_accessed",
            )
        )
    ):
        raise ValueError("exact bounded six-case19680-step recorded control replay required")


def prepare(path, *, early80=False):
    cfg = json.loads(path.read_text())
    require_scope(cfg, early80=early80)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        Path(__file__).resolve() != source / "scripts/rsi_replay_feedback_screen.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or root.exists()
        or type(cfg["deadline_unix"]) not in (int, float)
        or not np.isfinite(cfg["deadline_unix"])
        or time.time() >= cfg["deadline_unix"]
    ):
        raise ValueError("exact frozen source, bounded deadline and unused output required")
    require_terminal(cfg["screen_unit"])
    bind = cfg.get("bind_terminal_readback", False)
    if type(bind) is not bool or (
        bind and (not early80 or "readback_hash" in cfg or "readback_file_hash" in cfg)
    ):
        raise ValueError("explicit early80 terminal-only readback hash binding required")
    if bind:
        require_terminal(cfg["readback_unit"])
    pins = {str(path): digest(path)}
    pins.update(
        {
            str(p): digest(p)
            for folder in (source / "src", source / "scripts")
            for p in folder.rglob("*.py")
        }
    )

    def sealed(name, expected_file=None):
        path = Path(name)
        h = digest(path)
        if (
            path.is_symlink()
            or (expected_file is not None and h != expected_file)
            or (str(path) in pins and pins[str(path)] != h)
        ):
            raise ValueError("ordinary original authenticated evidence required")
        d = json.loads(path.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole original evidence seal differs")
        pins[str(path)] = h
        return d

    readback = sealed(cfg["readback"], None if bind else cfg["readback_file_hash"])
    if bind:
        if readback.get("source_head") != cfg["readback_source_head"]:
            raise ValueError("same declared terminal independent reader source required")
        cfg = dict(
            cfg,
            readback_hash=readback["report_hash"],
            readback_file_hash=pins[str(Path(cfg["readback"]))],
        )
    if (
        readback["schema"]
        != (
            "soccer.rsi.early_feedback_impact_screen_readback.v1"
            if early80
            else "soccer.rsi.feedback_impact_screen_readback.v1"
        )
        or readback["report_hash"] != cfg["readback_hash"]
        or readback["all_original_six_including_failures_retained"] is not True
        or len(readback["rows"]) != 6
        or readback["candidate_jacobian_bank_independently_regenerated"] is not True
        or any(
            readback[k] is not False
            for k in (
                "nonzero_branch_independent_physics_replayed",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "full_m0_gain_proven",
            )
        )
        or any(
            type(readback[k]) is not int or readback[k] != 0
            for k in ("new_physical_steps", "new_optimizer_updates")
        )
    ):
        raise ValueError("whole original non-promoting complete raw screen readback required")
    for name, h in readback["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("whole original screen source/model/body/raw proof changed")
        pins[name] = h
    screen_root = Path(cfg["screen_root"])

    def linked(name):
        if str(Path(name)) not in pins:
            raise ValueError("every predecessor must already belong to whole readback proof")
        return sealed(name)

    screen = linked(screen_root / "complete.json")
    declaration = linked(screen_root / "declaration.json")
    if (
        screen["report_hash"] != readback["complete_hash"]
        or screen["declaration_hash"] != declaration["report_hash"]
        or Path(declaration["config"]["output_root"]) != screen_root
        or (screen_root / "failure.json").exists()
    ):
        raise ValueError("original successful full screen required")
    stage = linked(declaration["config"]["stage_readback"])
    native = linked(stage["effective_cohort_spec"]["producer_complete"])
    if (
        len(native["results"]) != 6
        or len(screen["rows"]) != 6
        or native["report_hash"] != declaration["native_complete_hash"]
    ):
        raise ValueError("all six original physical cases including failures required")
    return cfg, root, pins, readback, screen, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--early80", action="store_true")
    args = parser.parse_args()
    cfg, root, pins, readback, screen, native = prepare(args.config, early80=args.early80)
    kind = "recorded_early_feedback_replay" if args.early80 else "recorded_feedback_replay"
    if args.preflight:
        print(
            "SIX_EARLY80_RECORDED_REPLAY_READY_NO_PHYSICS"
            if args.early80
            else "SIX_RECORDED_FEEDBACK_REPLAY_READY_NO_PHYSICS",
            flush=True,
        )
        return
    root.mkdir()
    declaration = dict(
        schema=f"soccer.rsi.{kind}_declaration.v1",
        config=cfg,
        readback_hash=readback["report_hash"],
        screen_complete_hash=screen["report_hash"],
        input_file_hashes=pins,
        recorded_next_targets_not_neural_recomputation=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(root / "declaration.json", declaration)
    attempted = integrated = 0
    replay = None
    current_case = current_branch = None
    rows = []

    def counts():
        return dict(
            recorded_attempted_steps=attempted,
            recorded_integrated_steps=integrated,
            branch_attempted_steps=0 if replay is None else replay.attempted_steps,
            branch_integrated_steps=0 if replay is None else replay.integrated_steps,
            branch_windows_completed=0 if replay is None else replay.completed_windows,
        )

    def bounded():
        if time.time() >= cfg["deadline_unix"] or attempted > 18000:
            raise RuntimeError("recorded replay deadline or original step budget exceeded")

    def arrays(w):
        if pins.get(w["path"]) != w["sha256"] or digest(w["path"]) != w["sha256"]:
            raise ValueError("whole prior authenticated trajectory required")
        with np.load(w["path"], allow_pickle=False) as archive:
            return dict(archive)

    try:
        world = native["results"][0]["witnesses"]["world"]
        if pins.get(world["path"]) != world["sha256"] or digest(world["path"]) != world["sha256"]:
            raise ValueError("original world required")
        model = mujoco.MjModel.from_binary_path(world["path"])
        physics = json.loads(
            Path(native["results"][0]["witnesses"]["commitment"]["path"]).read_text()
        )["physics"]
        replay = RecordedFeedbackReplay(
            model,
            world_hash=world["sha256"],
            kp=np.asarray(physics["canonical_pd_kp"], dtype=np.float64),
            kd=np.asarray(physics["canonical_pd_kd"], dtype=np.float64),
            maximum_steps=3360 if args.early80 else 1680,
            early80=args.early80,
        )

        def canonical(data):
            return np.concatenate(
                (data.qpos[:7], data.qpos[replay.qi], data.qpos[replay.bq : replay.bq + 7])
            ), np.concatenate(
                (data.qvel[:6], data.qvel[replay.vi], data.qvel[replay.bv : replay.bv + 6])
            )

        for index, (row, parent, checked) in enumerate(
            zip(screen["rows"], native["results"], readback["rows"], strict=True)
        ):
            bounded()
            current_case = index
            if (
                row["source_receipt_hash"] != parent["receipt_hash"]
                or row["source_receipt_hash"] != checked["source_receipt_hash"]
                or row["job"] != checked["job"]
            ):
                raise ValueError("every original case in source order required")
            commitment = json.loads(Path(parent["witnesses"]["commitment"]["path"]).read_text())
            if (
                commitment["physics"] != physics
                or parent["witnesses"]["world"]["sha256"] != world["sha256"]
            ):
                raise ValueError("same world and gains in all source cases required")
            trace = arrays(parent["witnesses"]["trace"])
            d = mujoco.MjData(model)
            q, v = trace["canonical_qpos"][0, 0], trace["canonical_qvel"][0, 0]
            d.qpos[:7], d.qpos[replay.qi], d.qpos[replay.bq : replay.bq + 7] = (
                q[:7],
                q[7:36],
                q[36:],
            )
            d.qvel[:6], d.qvel[replay.vi], d.qvel[replay.bv : replay.bv + 6] = (
                v[:6],
                v[6:35],
                v[35:],
            )
            mujoco.mj_forward(model, d)
            branches = []
            for frame in range(300):
                bounded()
                q, v = canonical(d)
                equal(q, trace["canonical_qpos"][frame, 0], "original source q")
                equal(v, trace["canonical_qvel"][frame, 0], "original source v")
                if frame == row["retrospective_frame"]:
                    for number, prediction in enumerate(row["predictions"]):
                        bounded()
                        current_branch = number
                        a = arrays(prediction)
                        result = replay.verify(
                            d,
                            first_target_rad=trace["joint_target_rad"][frame, 0],
                            recorded=a,
                            donor_hash=prediction["donor_hash"],
                        )
                        result.update(
                            index=number,
                            name=prediction["name"],
                            original_arrays_file_hash=prediction["sha256"],
                        )
                        result["report_hash"] = hash_json(result)
                        write_once(root / f"case-{index}-branch-{number}.json", result)
                        branches.append(result)
                        print("RECORDED_BRANCH_REPLAYED", index, number, flush=True)
                for step in range(10):
                    d.ctrl[replay.ai] = trace["torque_nm"][frame, 0, step]
                    attempted += 1
                    mujoco.mj_step(model, d)
                    integrated += 1
                    equal(
                        d.actuator_force[replay.ai],
                        trace["actual_actuator_force_nm"][frame, 0, step],
                        "original actual actuators",
                    )
                    equal(
                        d.xpos[replay.pelvis, 2],
                        trace["pelvis_z_per_substep_m"][frame, 0, step],
                        "original pelvis",
                    )
            if len(branches) != len(row["predictions"]):
                raise ValueError("all original raw branches must be physically replayed")
            result = dict(
                index=index,
                source_receipt_hash=parent["receipt_hash"],
                original_outcome=parent["outcome"],
                job=row["job"],
                branches=branches,
                original300_frames_reconstructed=True,
            )
            result["report_hash"] = hash_json(result)
            write_once(root / f"case-{index}-complete.json", result)
            rows.append(result)
        bounded()
        if (
            attempted != 18000
            or integrated != 18000
            or replay.attempted_steps != screen["forecast_attempted_steps"]
            or replay.integrated_steps != screen["forecast_integrated_steps"]
            or replay.completed_windows != readback["raw_branch_count"]
            or any(digest(path) != h for path, h in pins.items())
        ):
            raise ValueError("all bounded original integration counts and sources required")
        result = dict(
            schema=f"soccer.rsi.{kind}_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            **counts(),
            all_original_six_and_all_nonzero_branches_physically_replayed=True,
            recorded_next_targets_not_neural_recomputation=True,
            full_m0_gain_proven=False,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            new_closed_loop_policy_trials=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(root / "complete.json", result)
    except BaseException as error:
        if replay is not None:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(replay.model, spec))
            mujoco.mj_getState(replay.model, replay.scratch, state, spec)
            with (root / "failure-last-replay-state.npz").open("xb") as stream:
                np.savez_compressed(stream, integration_state=state)
        failure = dict(
            schema=f"soccer.rsi.{kind}_failure.v1",
            declaration_hash=declaration["report_hash"],
            current_case=current_case,
            current_branch=current_branch,
            error_type=type(error).__name__,
            error=str(error),
            **counts(),
            hardware_authorized=False,
            promotion_authorized=False,
        )
        failure["report_hash"] = hash_json(failure)
        write_once(root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
