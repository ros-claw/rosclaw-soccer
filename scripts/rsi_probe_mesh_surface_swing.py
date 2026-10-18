"""Fixed-pose material-point IK probe; no integration, force or skill claim."""

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.mesh_surface_swing import surface_strike_joint_delta
from rosclaw_soccer.rsi.taskspace_swing_probe import continuous_swing_joint_delta
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_foot_surface_gap import digest, surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--geometry-hash", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-adapter", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("clean frozen probe and unused output required")
    source = json.loads(args.geometry_report.read_text())
    if source["report_hash"] != args.geometry_hash or source["report_hash"] != hash_json(
        {k: v for k, v in source.items() if k != "report_hash"}
    ):
        raise ValueError("exact sealed source geometry required")
    pins = dict(source["input_file_hashes"])
    pins[str(args.geometry_report)] = digest(args.geometry_report)
    for folder in (root / "src", root / "scripts"):
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
    adapter = None
    if args.verify_adapter:
        from rosclaw_soccer.rsi.mesh_surface_swing_probe import G1MeshSurfaceSwingProbe

        adapter = G1MeshSurfaceSwingProbe(model, expected_world_hash=cfg["world_hash"])
    g, ball = int(model.geom("task_contact_surface_0").id), int(model.geom("ball_geom").id)
    body = int(model.geom_bodyid[g])
    # Explicit same left-leg mapping used by the native kernel, not right-foot mirroring.
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

    qi = [int(model.jnt_qposadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    vi = [int(model.jnt_dofadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES]
    limits = np.asarray([model.jnt_range[model.joint(n).id] for n in G1_DDS_JOINT_NAMES[:6]])
    bq = int(model.jnt_qposadr[model.body_jntadr[model.geom_bodyid[ball]]])
    hull = surface(model, g)
    steps, rows = [1400, 1450, 1500, 1530, 1543], []

    def forward(data):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    for case in cfg["cases"]:
        receipt = json.loads(Path(case["receipt"]).read_text())
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as trace:
            qs = trace["cerebellar_canonical_qpos_before_substep"].reshape(3000, 43)
        for step in steps:
            data = mujoco.MjData(model)
            data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = (
                qs[step, :7],
                qs[step, 7:36],
                qs[step, 36:43],
            )
            forward(data)
            center = data.geom_xpos[ball].copy()
            rotation = data.geom_xmat[g].reshape(3, 3)
            _, local = hull.nearest(rotation.T @ (center - data.geom_xpos[g]))
            point = data.geom_xpos[g] + rotation @ local
            jac, origin_jac = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, body)
            mujoco.mj_jacBody(model, data, origin_jac, None, body)
            target_point = center - np.array([float(model.geom_size[ball, 0]), 0, 0])
            baseline = data.qpos[qi[:6]].copy()
            variants = dict(
                body_origin=continuous_swing_joint_delta(
                    data.xpos[body].copy(),
                    center,
                    origin_jac[:, vi[:6]],
                    baseline,
                    limits,
                    forward_cap_m=0.08,
                    lateral_cap_m=0.15,
                    vertical_offset_m=0.04,
                ),
                mesh_surface=surface_strike_joint_delta(
                    point,
                    center,
                    jac[:, vi[:6]],
                    baseline,
                    limits,
                    np.array([1.0, 0.0, 0.0]),
                    ball_radius_m=float(model.geom_size[ball, 0]),
                    forward_cap_m=0.08,
                    lateral_cap_m=0.15,
                    vertical_cap_m=0.04,
                ),
            )
            before = float(np.linalg.norm(point - target_point))
            if adapter is not None:
                preserved = {
                    name: getattr(data, name).copy()
                    for name in (
                        "qpos",
                        "qvel",
                        "xpos",
                        "geom_xpos",
                        "ctrl",
                        "qacc",
                        "qacc_warmstart",
                    )
                }
                actual_delta, record = adapter.advance(
                    data,
                    frame=step // 10,
                    side=0,
                    baseline_target=baseline,
                    joint_limits=limits,
                    forward_cap_m=0.08,
                    lateral_cap_m=0.15,
                    vertical_cap_m=0.04,
                )
                expected_record = np.concatenate(
                    (
                        [step // 10, 0, 1],
                        point,
                        center,
                        jac[:, vi[:6]].ravel(),
                        baseline,
                        variants["mesh_surface"],
                    )
                )
                if not np.array_equal(actual_delta, variants["mesh_surface"]) or not np.array_equal(
                    record, expected_record
                ):
                    raise ValueError("scratch adapter differs from independent original-pose probe")
                if any(
                    not np.array_equal(getattr(data, name), old) for name, old in preserved.items()
                ):
                    raise ValueError("scratch adapter mutated caller state")
            for name, delta in variants.items():
                probe = mujoco.MjData(model)
                probe.qpos[:] = data.qpos
                probe.qpos[qi[:6]] += delta
                forward(probe)
                moved_point = probe.geom_xpos[g] + probe.geom_xmat[g].reshape(3, 3) @ local
                after = float(np.linalg.norm(moved_point - target_point))
                rows.append(
                    dict(
                        case=case["name"],
                        step=step,
                        method=name,
                        before_material_point_error_m=before,
                        after_material_point_error_m=after,
                        error_reduced=after < before,
                        delta_rad=delta.tolist(),
                    )
                )
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("frozen inputs changed")
    result = dict(
        schema="soccer.rsi.fixed_pose_surface_ik_probe.v1",
        source_head=args.source_head,
        input_file_hashes=pins,
        rows=rows,
        fixed_substeps=steps,
        scratch_adapter_verified=args.verify_adapter,
        pose_probe_baseline="actual measured joint pose, NOT Foundation target or PD dynamics",
        material_point_held_fixed_during_pose_probe=True,
        new_physical_steps=0,
        new_optimizer_updates=0,
        existing_task_labels_unchanged=True,
        strike_velocity_or_balance_qualified=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
