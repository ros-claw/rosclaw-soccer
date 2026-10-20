"""Read-only complete standard-motion numeric/FK qualification, no dynamics."""

import argparse
import ast
import subprocess
from pathlib import Path


def permutations(path):
    found = {}
    widths = {"_IL_TO_MJCF_JOINT": 29, "_IL_TO_MJCF_BODY": 30}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in widths:
                if target.id in found:
                    raise ValueError("single literal joint/body order declaration required")
                a = ast.literal_eval(node.value)
                if (
                    type(a) is not list
                    or any(type(i) is not int for i in a)
                    or sorted(a) != list(range(widths[target.id]))
                ):
                    raise ValueError("exact complete Isaac-to-native permutation required")
                found[target.id] = a
    if set(found) != set(widths):
        raise ValueError("both explicit motion-loader order tables required")
    return found["_IL_TO_MJCF_JOINT"], found["_IL_TO_MJCF_BODY"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-reference-head", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
        or subprocess.check_output(["git", "-C", str(a.reference_root), "status", "--porcelain"])
        or subprocess.check_output(
            ["git", "-C", str(a.reference_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != a.expected_reference_head
    ):
        raise ValueError("pristine frozen reference and qualifier with unused output required")
    root = a.reference_root
    motion_dir = root / "src/assets/soccer/motions/shooter"
    files = sorted(motion_dir.glob("soccer-standard-*.npz"))
    if len(files) != 10:
        raise ValueError("all ten predeclared standard motions required, no cherry-picking")
    loader = root / "src/tasks/soccer/mdp/shooter_commands.py"
    joint_order, body_order = permutations(loader)
    xml = root / "src/assets/robots/unitree_g1/xmls/g1.xml"
    pins = {
        str(f): digest(f)
        for dr in (
            Path.cwd() / "src",
            Path.cwd() / "scripts",
            root / "src/assets/robots/unitree_g1",
        )
        for f in dr.rglob("*")
        if f.is_file() and (dr == root / "src/assets/robots/unitree_g1" or f.suffix == ".py")
    }
    for f in [*files, loader, root / "LICENCE", root / "README.md", root / "setup.py"]:
        pins[str(f)] = digest(f)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

    model = mujoco.MjModel.from_xml_path(str(xml))
    joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if (model.nq, model.nv, model.nbody) != (36, 35, 31) or tuple(joints) != G1_DDS_JOINT_NAMES:
        raise ValueError("explicit native HG29 source skeleton and thirty moving bodies required")
    data = mujoco.MjData(model)
    ids = [model.joint(name).id for name in joints]
    limits = model.jnt_range[ids]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            reference_head=a.expected_reference_head,
            input_file_hashes=pins,
            motion_names=[f.name for f in files],
            frequency_hz=50,
            joint_permutation=joint_order,
            body_permutation=body_order,
            maximum_body_position_error_m=0.001,
            maximum_body_quaternion_sign_invariant_error=0.001,
            maximum_hard_joint_limit_excess_rad=1e-5,
            no_clipping_or_source_repair=True,
            new_native_steps=0,
            new_neural_calls=0,
            hardware_authorized=False,
        ),
    )
    rows = []
    try:
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            n = len(arrays["joint_pos"])
            shapes = dict(
                fps=(1,),
                joint_pos=(n, 29),
                joint_vel=(n, 29),
                body_pos_w=(n, 30, 3),
                body_quat_w=(n, 30, 4),
                body_lin_vel_w=(n, 30, 3),
                body_ang_vel_w=(n, 30, 3),
                kick_leg=(),
            )
            if (
                not 100 <= n <= 10000
                or set(arrays) != set(shapes)
                or any(arrays[k].shape != s for k, s in shapes.items())
                or arrays["fps"].dtype != np.int64
                or arrays["fps"][0] != 50
                or arrays["kick_leg"].dtype.kind != "U"
                or str(arrays["kick_leg"]) not in ("left", "right")
                or any(
                    arrays[k].dtype != np.float32 or not np.isfinite(arrays[k]).all()
                    for k in shapes
                    if k not in ("fps", "kick_leg")
                )
                or np.max(abs(np.linalg.norm(arrays["body_quat_w"], axis=-1) - 1)) > 1e-4
            ):
                raise ValueError("finite typed complete motion/frequency/body convention required")
            q = arrays["joint_pos"][:, joint_order].astype(np.float64)
            bp = arrays["body_pos_w"][:, body_order].astype(np.float64)
            bq = arrays["body_quat_w"][:, body_order].astype(np.float64)
            error_p, error_q = 0.0, 0.0
            for i in range(n):
                data.qpos[:3] = bp[i, 0]
                data.qpos[3:7] = bq[i, 0]
                data.qpos[7:36] = q[i]
                mujoco.mj_kinematics(model, data)
                error_p = max(error_p, float(np.linalg.norm(data.xpos[1:] - bp[i], axis=1).max()))
                errors = np.minimum(
                    np.linalg.norm(data.xquat[1:] - bq[i], axis=1),
                    np.linalg.norm(data.xquat[1:] + bq[i], axis=1),
                )
                error_q = max(error_q, float(errors.max()))
            excess = np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0)
            bad = [joints[j] for j in np.flatnonzero(excess.max(axis=0) > 1e-5)]
            row = dict(
                motion_name=f.name,
                motion_hash=pins[str(f)],
                frames=n,
                kick_leg=str(arrays["kick_leg"]),
                maximum_body_position_error_m=error_p,
                maximum_body_quaternion_sign_invariant_error=error_q,
                maximum_hard_joint_limit_excess_rad=float(excess.max()),
                outside_hard_joint_names=bad,
                numeric_fk_qualified=bool(error_p <= 0.001 and error_q <= 0.001 and not bad),
            )
            rows.append(row)
            print("MOTION_DATA_FK", row, flush=True)
        check_pins(pins)
        passed = all(r["numeric_fk_qualified"] for r in rows)
        write_once(
            a.output / "complete.json",
            dict(
                status="MJLAB_STANDARD_MOTION_NUMERIC_FK_QUALIFIED"
                if passed
                else "MJLAB_STANDARD_MOTION_NUMERIC_FK_REJECTED",
                source_head=a.expected_head,
                reference_head=a.expected_reference_head,
                input_file_hashes=pins,
                results=rows,
                source_motion_files=10,
                total_source_frames=sum(r["frames"] for r in rows),
                new_native_steps=0,
                new_neural_calls=0,
                new_optimizer_steps=0,
                mujoco_version=mujoco.__version__,
                static_kinematics_not_gpu_or_motor_qualification=True,
                source_license_scope_review_pending=True,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_sources=rows,
                new_native_steps=0,
                new_neural_calls=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
