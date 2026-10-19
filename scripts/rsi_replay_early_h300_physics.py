"""Whole-six recorded-target H300 physics only, not a causal neural/search audit."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("profile") != "six-recorded-early-h300-physics-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("physical_step_budget", 18000),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "automatic_retry",
                "independent_neural_or_search_qualification_claimed",
            )
        )
    ):
        raise ValueError("exact whole-six recorded-target no-learning physical diagnostic required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    require_scope(cfg)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        Path(__file__).resolve() != source / "scripts/rsi_replay_early_h300_physics.py"
        or root.exists()
        or not time.time() < cfg["deadline_unix"] <= time.time() + 18000
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("clean frozen exact entry, bounded deadline and unused output required")
    require_terminal(cfg["producer_unit"])
    require_terminal(cfg["readback_unit"])
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }
    pins[str(args.config)] = digest(args.config)

    def pin(name, expected=None):
        p = Path(name)
        h = digest(p)
        if (
            p.is_symlink()
            or (expected is not None and h != expected)
            or (str(p) in pins and pins[str(p)] != h)
        ):
            raise ValueError("whole ordinary immutable source evidence required")
        pins[str(p)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole evidence seal required")
        return d

    readback = sealed(cfg["readback_path"])
    complete = sealed(Path(cfg["producer_root"]) / "complete.json")
    declaration = sealed(Path(cfg["producer_root"]) / "declaration.json")
    if (
        readback["schema"] != "soccer.rsi.early_impact_h300_readback.v1"
        or readback["source_head"] != cfg["readback_source_head"]
        or readback["complete_hash"] != complete["report_hash"]
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(complete["rows"]) != 6
        or len(readback["rows"]) != 6
    ):
        raise ValueError("whole same six completed and independently read-back inputs required")
    for name, h in readback["input_file_hashes"].items():
        pin(name, h)
    stage = sealed(declaration["config"]["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    cases = []
    for i, (row, parent, checked) in enumerate(
        zip(complete["rows"], native["results"], readback["rows"], strict=True)
    ):
        if (
            row["index"] != i
            or row["source_receipt_hash"] != parent["receipt_hash"]
            or row["job"] != checked["job"]
            or checked["actual_outcome"] != row["causal_rollout_outcome"]
        ):
            raise ValueError("all original cases, including failures, in fixed order required")
        world = parent["witnesses"]["world"]
        commitment = parent["witnesses"]["commitment"]
        pin(world["path"], world["sha256"])
        pin(commitment["path"], commitment["sha256"])
        c = json.loads(Path(commitment["path"]).read_text())
        w = row["arrays"]
        pin(w["path"], w["sha256"])
        cases.append((row, world, c))
    if args.preflight:
        print("SIX_RECORDED_EARLY_H300_PHYSICS_READY_NO_NEW_PHYSICS", flush=True)
        return
    root.mkdir()
    d = dict(
        schema="soccer.rsi.recorded_early_h300_declaration.v1",
        config=cfg,
        input_file_hashes=pins,
        producer_complete_hash=complete["report_hash"],
        readback_hash=readback["report_hash"],
        recorded_targets_consumed=True,
        neural_policy_qualification_claimed=False,
    )
    d["report_hash"] = hash_json(d)
    write_once(root / "declaration.json", d)
    rows, current, audit = [], None, None
    try:
        import mujoco

        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

        for i, (row, world, c) in enumerate(cases):
            current, audit = i, None
            if time.time() >= cfg["deadline_unix"]:
                raise RuntimeError("fixed whole-six physical diagnostic deadline exhausted")
            model = mujoco.MjModel.from_binary_path(world["path"])
            physics = c["physics"]
            audit = RecordedEarlyEpisode(
                model,
                world_hash=world["sha256"],
                kp=np.asarray(physics["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(physics["canonical_pd_kd"], dtype=np.float64),
                course=tuple(c["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            with np.load(row["arrays"]["path"], allow_pickle=False) as a:
                review = audit.verify(dict(a))
            result = dict(index=i, source_case_hash=row["report_hash"], physical_review=review)
            result["report_hash"] = hash_json(result)
            write_once(root / f"case-{i}-complete.json", result)
            rows.append(result)
            print("RECORDED_EARLY_H300_PHYSICS_CASE_COMPLETE", i, flush=True)
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source/input changed during physical verification")
        result = dict(
            schema="soccer.rsi.recorded_early_h300_complete.v1",
            declaration_hash=d["report_hash"],
            rows=rows,
            actual_physical_steps=sum(r["physical_review"]["actual_physical_steps"] for r in rows),
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            independent_neural_or_search_qualified=False,
        )
        if result["actual_physical_steps"] != 18000:
            raise ValueError("whole six physical bodies required")
        result["report_hash"] = hash_json(result)
        write_once(root / "complete.json", result)
    except BaseException as error:
        failure = dict(
            current_case=current,
            completed_cases=len(rows),
            current_attempted_steps=0 if audit is None else audit.attempted_steps,
            current_integrated_steps=0 if audit is None else audit.integrated_steps,
            error_type=type(error).__name__,
            error=str(error),
            automatic_retry=False,
        )
        if audit is not None:
            with (root / "failure-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=audit.data.qpos, qvel=audit.data.qvel, ctrl=audit.data.ctrl
                )
        write_once(root / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
