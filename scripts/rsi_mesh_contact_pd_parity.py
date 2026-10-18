"""Ground-only paired G1 replay with current-joint PD, fixed recorded targets.

Unlike torque replay this closes joint feedback; Foundation targets remain
recorded. Neither successful tracking nor parity is an adaptive football skill.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1 import joint_contract
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim import contracts
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--candidate-world", type=Path, required=True)
    parser.add_argument("--candidate-world-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("unused diagnostic output required")
    receipt = json.loads(args.reference_receipt.read_text())
    if receipt["receipt_hash"] != hash_json(
        {k: v for k, v in receipt.items() if k != "receipt_hash"}
    ):
        raise ValueError("sealed original receipt required")
    witnesses = receipt["witnesses"]
    for witness in witnesses.values():
        if digest(Path(witness["path"])) != witness["sha256"]:
            raise ValueError("whole original physical witness bytes differ")
    if digest(args.candidate_world) != args.candidate_world_hash:
        raise ValueError("predeclared candidate world hash required")
    pins = {
        str(args.reference_receipt): digest(args.reference_receipt),
        str(args.candidate_world): digest(args.candidate_world),
        str(Path(__file__)): digest(Path(__file__)),
    }
    pins.update({w["path"]: w["sha256"] for w in witnesses.values()})
    for module in (joint_contract, contracts):
        pins[str(Path(module.__file__))] = digest(Path(module.__file__))
    commitment = json.loads(Path(witnesses["commitment"]["path"]).read_text())
    physics = commitment["physics"]
    if physics["canonical_joint_names"] != list(G1_DDS_JOINT_NAMES):
        raise ValueError("original named G1 PD contract required")
    kp, kd = (
        np.asarray(physics[key], dtype=np.float64) for key in ("canonical_pd_kp", "canonical_pd_kd")
    )
    if (
        kp.shape != (29,)
        or kd.shape != (29,)
        or not np.isfinite(kp).all()
        or not np.isfinite(kd).all()
    ):
        raise ValueError("finite 29-joint original gains required")
    if np.any(kp <= 0) or np.any(kd < 0):
        raise ValueError("original positive PD gains required")
    models = [
        mujoco.MjModel.from_binary_path(str(path))
        for path in (Path(witnesses["world"]["path"]), args.candidate_world)
    ]
    with np.load(witnesses["trace"]["path"], allow_pickle=False) as trace:
        qpos = trace["canonical_qpos"][0, 0].copy()
        qvel = trace["canonical_qvel"][0, 0].copy()
        targets = trace["joint_target_rad"][:, 0].copy()
    if qpos.shape != (43,) or qvel.shape != (41,) or targets.shape != (300, 29):
        raise ValueError("exact recorded G1 feedback diagnostic required")
    if not all(np.isfinite(v).all() for v in (qpos, qvel, targets)):
        raise ValueError("finite original states/targets required")
    bindings = []
    for model in models:
        if (model.nq, model.nv, model.nu) != (43, 41, 29) or model.opt.timestep != 0.002:
            raise ValueError("unchanged G1 dimensions and timestep required")
        joints = [int(model.joint(name).id) for name in G1_DDS_JOINT_NAMES]
        qi = np.asarray([model.jnt_qposadr[j] for j in joints])
        vi = np.asarray([model.jnt_dofadr[j] for j in joints])
        ai = [np.flatnonzero(model.actuator_trnid[:, 0] == j) for j in joints]
        if any(len(a) != 1 for a in ai):
            raise ValueError("one canonical actuator per named joint required")
        ai = [int(a[0]) for a in ai]
        bj = model.body_jntadr[model.body("ball").id]
        bq, bv = model.jnt_qposadr[bj], model.jnt_dofadr[bj]
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.qpos[bq : bq + 3] = (100, 100, 0.13)
        data.qvel[bv : bv + 6] = 0
        mujoco.mj_forward(model, data)
        bindings.append((model, data, qi, vi, ai))
    fields = ("qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_constraint", "actuator_force")
    stream_hash = hashlib.sha256()
    minimum = float("inf")
    maximum_joint_speed = 0.0
    for target in targets:
        for _ in range(10):
            for model, data, qi, vi, ai in bindings:
                torque = np.clip(
                    kp * (target - data.qpos[qi]) - kd * data.qvel[vi],
                    -G1_HARD_TORQUE_LIMITS,
                    G1_HARD_TORQUE_LIMITS,
                )
                data.ctrl[ai] = torque
                mujoco.mj_step(model, data)
                ball = int(model.geom("ball_geom").id)
                ball_body = int(model.body("ball").id)
                if any(
                    ball in c.geom
                    and any(model.geom_bodyid[g] not in (0, ball_body) for g in c.geom)
                    for c in data.contact[: data.ncon]
                ):
                    raise ValueError("unexpected player/ball contact in ground-only replay")
            for field in fields:
                arrays = [getattr(binding[1], field) for binding in bindings]
                if (
                    not all(np.isfinite(a).all() for a in arrays)
                    or arrays[0].tobytes() != arrays[1].tobytes()
                ):
                    raise ValueError("nonfinite state or old/new PD ground parity mismatch")
                stream_hash.update(arrays[0].tobytes())
            minimum = min(minimum, float(bindings[0][1].qpos[2]))
            maximum_joint_speed = max(
                maximum_joint_speed, float(np.max(np.abs(bindings[0][1].qvel[bindings[0][3]])))
            )
    if any(digest(Path(path)) != value for path, value in pins.items()):
        raise ValueError("source/input changed during diagnostic")
    result = dict(
        schema="soccer.physics.mesh_ground_pd_parity.v1",
        input_file_hashes=pins,
        original_receipt_hash=receipt["receipt_hash"],
        candidate_world_hash=args.candidate_world_hash,
        checked_fields=list(fields),
        byte_equal=True,
        common_state_hash=stream_hash.hexdigest(),
        new_physics_substeps=6000,
        worlds=2,
        original_pd_gains=True,
        targets_recomputed_by_foundation=False,
        adaptive_football_policy_rollouts=0,
        minimum_root_height_m=minimum,
        maximum_joint_speed_rad_s=maximum_joint_speed,
        new_optimizer_updates=0,
        football_skill_qualified=False,
        hardware_authorized=False,
        runner_source_hash=digest(Path(__file__)),
    )
    result["report_hash"] = hash_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
