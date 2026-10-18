"""Zero-integration finite differences on fixed original G1 mesh-world poses."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.mesh_sphere_contact_state import measure_mesh_sphere_contact_state
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_foot_surface_gap import digest, surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--geometry-hash", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("clean frozen source and unused output required")
    report = json.loads(args.geometry_report.read_text())
    if report["report_hash"] != args.geometry_hash or report["report_hash"] != hash_json(
        {k: v for k, v in report.items() if k != "report_hash"}
    ):
        raise ValueError("exact sealed original geometry readback required")
    pins = dict(report["input_file_hashes"])
    pins[str(args.geometry_report)] = digest(args.geometry_report)
    pins[report["arrays"]["path"]] = report["arrays"]["sha256"]
    for folder in (root / "scripts", root / "src"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("original input bytes changed")
    config_paths = [
        p for p in pins if Path(p).name.startswith("rsi-mesh-raw4-foot-surface-gap-config-")
    ]
    if len(config_paths) != 1:
        raise ValueError("one original geometry declaration required")
    cfg = json.loads(Path(config_paths[0]).read_text())
    model = mujoco.MjModel.from_binary_path(cfg["compiled_world"])
    sphere = int(model.geom("ball_geom").id)
    feet = [int(model.geom(f"task_contact_surface_{i}").id) for i in range(2)]
    hulls = [surface(model, g) for g in feet]
    qi = [int(model.jnt_qposadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    vi = [int(model.jnt_dofadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    ball_joint = int(model.body_jntadr[model.geom_bodyid[sphere]])
    bq, bv = int(model.jnt_qposadr[ball_joint]), int(model.jnt_dofadr[ball_joint])
    epsilon = 1e-6
    rows = []
    steps = [1000, 1500, 1530, 1543, 1800, 1864]

    def forward(data):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    with np.load(report["arrays"]["path"], allow_pickle=False) as original_gaps:
        for case in cfg["cases"]:
            receipt = json.loads(Path(case["receipt"]).read_text())
            with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as trace:
                qs = trace["cerebellar_canonical_qpos_before_substep"].reshape(3000, 43)
                vs = trace["cerebellar_canonical_qvel_before_substep"].reshape(3000, 41)
            for step in steps:
                data = mujoco.MjData(model)
                data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = (
                    qs[step, :7],
                    qs[step, 7:36],
                    qs[step, 36:43],
                )
                data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = (
                    vs[step, :6],
                    vs[step, 6:35],
                    vs[step, 35:41],
                )
                forward(data)
                for side, (g, hull) in enumerate(zip(feet, hulls, strict=True)):
                    state = measure_mesh_sphere_contact_state(model, data, g, sphere, hull)
                    if (
                        abs(
                            state.gap_m
                            - original_gaps[case["name"] + "_independent_gap_m"][step, side]
                        )
                        > 1e-10
                    ):
                        raise ValueError("new measurement differs from original geometric gap")
                    numeric = []
                    for dof in range(model.nv):
                        values = []
                        for sign in (-1, 1):
                            probe = mujoco.MjData(model)
                            probe.qpos[:] = data.qpos
                            tangent = np.zeros(model.nv)
                            tangent[dof] = sign
                            mujoco.mj_integratePos(model, probe.qpos, tangent, epsilon)
                            forward(probe)
                            values.append(
                                measure_mesh_sphere_contact_state(
                                    model, probe, g, sphere, hull
                                ).gap_m
                            )
                        numeric.append((values[1] - values[0]) / (2 * epsilon))
                    error = float(np.max(np.abs(np.asarray(numeric) - state.gap_jacobian)))
                    if error > 2e-7:
                        raise ValueError("actual G1 surface gap differential failed numeric check")
                    rows.append(
                        dict(
                            case=case["name"],
                            step=step,
                            foot=side,
                            state=asdict(state),
                            max_jacobian_error=error,
                        )
                    )
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("frozen inputs changed")
    result = dict(
        schema="soccer.rsi.mesh_surface_differential_readback.v1",
        source_head=args.source_head,
        input_file_hashes=pins,
        rows=rows,
        fixed_substeps=steps,
        finite_difference_epsilon=epsilon,
        local_derivative_only=True,
        new_physical_steps=0,
        new_optimizer_updates=0,
        future_contact_qualified=False,
        controller_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
