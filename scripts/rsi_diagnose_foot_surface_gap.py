"""Canonical-pose foot mesh / sphere gaps, zero integration or force solving.

Independent closest-triangle geometry cross-checks native CCD distances. The
existing physical task labels are never replaced by these geometric queries.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.physics.convex_surface_distance import ConvexSurfaceDistance
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def surface(model, geom):
    from scipy.spatial import ConvexHull

    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("actual compiled mesh contact surface required")
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh])
    vertices = np.asarray(
        model.mesh_vert[start : start + model.mesh_vertnum[mesh]], dtype=np.float64
    )
    hull = ConvexHull(vertices)
    return ConvexSurfaceDistance(
        vertices[hull.simplices], np.asarray(hull.equations, dtype=np.float64)
    )


def query(model, data, mesh, sphere, hull):
    center = data.geom_xpos[sphere]
    rotation = data.geom_xmat[mesh].reshape(3, 3)
    signed, local = hull.nearest(rotation.T @ (center - data.geom_xpos[mesh]))
    point = data.geom_xpos[mesh] + rotation @ local
    independent = signed - float(model.geom_size[sphere, 0])
    segment = np.zeros(6)
    native = float(mujoco.mj_geomDistance(model, data, mesh, sphere, 10.0, segment))
    return independent, native, point


def self_test():
    vertices = " ".join(
        str(v) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1) for v in (x, y, z)
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="cube" scale="2 1 .5" vertex="' + vertices + '"/></asset>'
        '<worldbody><geom name="mesh" type="mesh" mesh="cube" pos="1 0 0" '
        'quat=".7071067811865476 0 0 .7071067811865476"/>'
        '<geom name="sphere" type="sphere" size=".1" pos="2.5 0 0"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    g = int(model.geom("mesh").id)
    s = int(model.geom("sphere").id)
    hull = surface(model, g)
    checks = []
    for center, expected in (
        ([2.5, 0, 0], 0.4),
        ([1, 2.5, 0], 0.4),
        ([1, 0, 0.8], 0.2),
        ([2.05, 0, 0], -0.05),
        ([1, 0, 0.55], -0.05),
        ([1, 0, 0], -0.6),
    ):
        model.geom_pos[s] = center
        mujoco.mj_kinematics(model, data)
        independent, native, _ = query(model, data, g, s, hull)
        inside = independent < -float(model.geom_size[s, 0])
        if abs(independent - expected) > 1e-10:
            raise ValueError("scaled/rotated independent hull analytic check failed")
        if not inside and abs(native - independent) > 1e-6:
            raise ValueError("non-contained native CCD cross-check failed")
        checks.append(
            dict(
                center=center,
                expected_gap_m=expected,
                independent_gap_m=independent,
                native_ccd_gap_m=native,
                sphere_center_inside_hull=inside,
            )
        )
    result = dict(
        independent_analytic_geometry_qualified=True,
        native_containment_distance_qualified=False,
        checks=checks,
        new_physical_steps=0,
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.config is None:
        parser.error("--config required")
    cfg = json.loads(args.config.read_text())
    source = Path(cfg["source_root"])
    output = Path(cfg["output"])
    arrays_path = output.with_suffix(".npz")
    if (
        output.exists()
        or arrays_path.exists()
        or Path(__file__).resolve() != source / "scripts/rsi_diagnose_foot_surface_gap.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused output and exact frozen diagnostic source required")
    pins = {str(args.config): digest(args.config)}
    for path, expected in cfg["input_file_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError("original selected witness bytes changed")
        pins[path] = expected
    for relative in (
        "scripts/rsi_diagnose_foot_surface_gap.py",
        "src/rosclaw_soccer/physics/convex_surface_distance.py",
    ):
        path = source / relative
        pins[str(path)] = digest(path)
    pair = json.loads(Path(cfg["pair_readback"]).read_text())
    if (
        pair["report_hash"] != cfg["pair_readback_hash"]
        or pair["report_hash"] != hash_json({k: v for k, v in pair.items() if k != "report_hash"})
        or pair["paired_feedback_divergence"]["source_indices"] != [0, 24]
        or [case["source_index"] for case in cfg["cases"]] != [0, 24]
    ):
        raise ValueError("exact independently read raw4 regression pair required")
    if args.preflight:
        print("SURFACE_GAP_PREFLIGHT_ZERO_MODEL_ALLOCATIONS_OR_PHYSICS_STEPS", flush=True)
        return
    model = mujoco.MjModel.from_binary_path(cfg["compiled_world"])
    if digest(Path(cfg["compiled_world"])) != cfg["world_hash"] or (
        model.nq,
        model.nv,
        model.nu,
    ) != (43, 41, 29):
        raise ValueError("same original compiled G1 mesh world required")
    sphere = int(model.geom("ball_geom").id)
    if model.geom_type[sphere] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("real sphere required")
    feet = [
        int(model.geom(name).id) for name in ("task_contact_surface_0", "task_contact_surface_1")
    ]
    hulls = [surface(model, g) for g in feet]
    qi = np.asarray([model.jnt_qposadr[model.joint(n).id] for n in G1_DDS_JOINT_NAMES])
    ball_body = int(model.geom_bodyid[sphere])
    bq = int(model.jnt_qposadr[model.body_jntadr[ball_body]])
    if set(range(7)) | set(qi) | set(range(bq, bq + 7)) != set(range(43)):
        raise ValueError("complete canonical state mapping required")
    results, arrays = [], {}
    for item in cfg["cases"]:
        receipt = json.loads(Path(item["receipt"]).read_text())
        if receipt["receipt_hash"] != hash_json(
            {k: v for k, v in receipt.items() if k != "receipt_hash"}
        ):
            raise ValueError("original receipt seal differs")
        if (
            receipt["receipt_hash"] != pair["rows"][item["source_index"]]["receipt_hash"]
            or receipt["witnesses"]["world"]["sha256"] != cfg["world_hash"]
            or cfg["input_file_hashes"].get(receipt["witnesses"]["trace"]["path"])
            != receipt["witnesses"]["trace"]["sha256"]
        ):
            raise ValueError(
                "selected original pose trace must bind the exact paired receipt/world"
            )
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            q = archive["cerebellar_canonical_qpos_before_substep"].reshape(3000, 43)
            clock = archive["cerebellar_before_integration_substep_index"]
            forces = archive["force_n"].copy()
        if (
            q.dtype != np.float64
            or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, rtol=0, atol=1e-6)
            or not np.array_equal(clock.ravel(), np.arange(3000))
            or clock.dtype != np.int64
            or clock.shape != (300, 1, 10, 1)
        ):
            raise ValueError("whole actual canonical pre-integration pose clock required")
        gaps = np.zeros((3000, 2))
        native = np.zeros_like(gaps)
        points = np.zeros((3000, 2, 3))
        data = mujoco.MjData(model)
        for step in range(3000):
            data.qpos[:7] = q[step, :7]
            data.qpos[qi] = q[step, 7:36]
            data.qpos[bq : bq + 7] = q[step, 36:43]
            mujoco.mj_kinematics(model, data)
            for side, (g, hull) in enumerate(zip(feet, hulls, strict=True)):
                gaps[step, side], native[step, side], points[step, side] = query(
                    model, data, g, sphere, hull
                )
        arrays[item["name"] + "_independent_gap_m"] = gaps
        arrays[item["name"] + "_native_ccd_gap_m"] = native
        inside = gaps < -float(model.geom_size[sphere, 0])
        arrays[item["name"] + "_native_center_inside_hull_unqualified"] = inside
        arrays[item["name"] + "_closest_mesh_point_world_m"] = points
        results.append(
            dict(
                name=item["name"],
                receipt_hash=receipt["receipt_hash"],
                outcome=receipt["outcome"],
                robot_ball_peak_contact_n=float(forces.max()),
                minimum_surface_gap_by_foot_m=gaps.min(axis=0).tolist(),
                minimum_gap_substep_by_foot=np.argmin(gaps, axis=0).tolist(),
                original_contact_frame153_minimum_gap_by_foot_m=gaps[1530:1540]
                .min(axis=0)
                .tolist(),
                maximum_native_vs_independent_gap_error_m=float(np.max(np.abs(native - gaps))),
                native_query_center_inside_hull_count=int(inside.sum()),
                native_containment_distance_qualified=False,
                significant_native_geometry_disagreement_count=int(
                    np.count_nonzero(np.abs(native - gaps) > 1e-6)
                ),
            )
        )
        print("CANONICAL_FOOT_SURFACE_GAPS_RECONSTRUCTED", item["name"], flush=True)
    if any(digest(Path(p)) != h for p, h in pins.items()):
        raise ValueError("frozen inputs changed")
    with arrays_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    import scipy

    report = dict(
        schema="soccer.rsi.canonical_mesh_sphere_gap_diagnostic.v1",
        rows=results,
        input_file_hashes=pins,
        arrays=dict(path=str(arrays_path), sha256=digest(arrays_path)),
        mujoco_version=mujoco.__version__,
        scipy_version=scipy.__version__,
        nativeccd_enabled=not bool(
            model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        ),
        geometry_clock=(
            "canonical before each 2ms integration; scratch kinematics, not cached controller pose"
        ),
        independent_geometry=(
            "scipy convex hull + numpy closest triangles; not collision-force inference"
        ),
        whole_original_prefix_reaudited_here=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        original_task_labels_unchanged=True,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(output, report)
    print(report["report_hash"], flush=True)


if __name__ == "__main__":
    main()
