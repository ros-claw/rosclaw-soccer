"""Read dense replay contact-exit diagnostics; no simulation, learning or promotion."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_exit_metrics import contact_exit_metrics
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-complete", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output and clean frozen source required")
    pins = {
        str(path): digest(path)
        for path in (Path(__file__), source / "src/rosclaw_soccer/physics/contact_exit_metrics.py")
    }

    def sealed(path):
        value = json.loads(Path(path).read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("original replay or declaration seal changed")
        pins[str(path)] = digest(path)
        return value

    rows, identities = [], set()
    for path in args.replay_complete:
        replay = sealed(path)
        declaration = sealed(path.parent / "declaration.json")
        if (
            replay["schema"] != "soccer.rsi.fixed_control_contact_replay_complete.v1"
            or replay["declaration_hash"] != declaration["report_hash"]
            or replay["integrated_substeps"] != 12000
            or replay["attempted_step_calls"] != 12000
            or len(replay["rows"]) != 4
            or declaration["mujoco_version"] != mujoco.__version__
        ):
            raise ValueError("whole original fixed four-replay declaration required")
        for name, h in declaration["input_file_hashes"].items():
            if digest(name) != h:
                raise ValueError("original replay source/input witness changed")
            pins[name] = h
        receipt = json.loads(Path(declaration["cases"][0]["receipt_path"]).read_text())
        model = mujoco.MjModel.from_binary_path(receipt["witnesses"]["world"]["path"])
        ball = int(model.body("ball").id)
        joint = int(model.body_jntadr[ball])
        dof = int(model.jnt_dofadr[joint])
        static = np.flatnonzero(model.geom_bodyid == 0)
        if (
            not np.array_equal(model.opt.gravity, [0, 0, -9.81])
            or np.any(model.dof_damping[dof : dof + 6] != 0)
            or np.any(model.body_ipos[ball] != 0)
            or len(static) != 1
            or model.geom_type[static[0]] != mujoco.mjtGeom.mjGEOM_PLANE
        ):
            raise ValueError(
                "free undamped ball COM at origin, gravity and floor-only world required"
            )
        for row in replay["rows"]:
            identity = row["source_receipt_hash"]
            if identity in identities:
                raise ValueError("duplicate endpoint cannot become another independent case")
            identities.add(identity)
            if sealed(path.parent / f"{row['case_id']}-complete.json") != row:
                raise ValueError("whole replay must bind exact individual result")
            witness = row["arrays"]
            if digest(witness["path"]) != witness["sha256"]:
                raise ValueError("dense replay arrays changed")
            pins[witness["path"]] = witness["sha256"]
            with np.load(witness["path"], allow_pickle=False) as archive:
                metrics = contact_exit_metrics(
                    dict(archive),
                    row["robot_contact_events_over_1n"],
                    mass_kg=float(model.body_mass[ball]),
                    dt_s=float(model.opt.timestep),
                )
            if metrics["maximum_linear_newton_residual_n"] > 1e-7:
                raise ValueError("translation force/momentum consistency failed")
            rows.append(
                dict(
                    case_id=row["case_id"],
                    source_receipt_hash=identity,
                    replay_result_hash=row["report_hash"],
                    original_outcome=row["original_outcome"],
                    metrics=metrics,
                )
            )
    if any(digest(name) != h for name, h in pins.items()):
        raise ValueError("source/input bytes changed during readback")
    result = dict(
        schema="soccer.rsi.contact_exit_metrics_readback.v1",
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        input_file_hashes=pins,
        rows=rows,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
