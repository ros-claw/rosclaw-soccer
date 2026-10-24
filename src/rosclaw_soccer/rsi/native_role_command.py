"""Ball/role-conditioned command learning over a retained gait, SIM research only.

Uses pre-action snapshots, never the98-column gait observation (which includes
the current command label). A shared role-conditioned model is not eight
independently trained minds or an end-to-end joint-torque cerebellum.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_team_intent import ROLES

ROLE_COMMAND_SCHEMA = "soccer.native_role_command_mlp.v1"
ROLE_COMMAND_OBSERVATION_DIM = 118


def role_command_observation(
    data: Any,
    kinematics: Any,
    layouts: dict[str, Any],
    *,
    team: str,
    role: str,
    previous_body_command: np.ndarray,
) -> np.ndarray:
    """118 measured columns in actual own-body frame, before command selection.

    Own goal2, attack direction2, role4, ball position3/velocity3, root linear3/
    angular3, gravity3, foot positions6, joint position29/velocity29, previous
    applied command3, teammates3/opponents4 position2/relative-velocity2 each.
    """
    if team not in ("red", "blue") or role not in ROLES:
        raise ValueError("explicit native team and role required")
    prefix = team + "_" + role + "/"
    lo = layouts[prefix]
    m = lo.model
    rotation = kinematics.xmat[lo.pelvis].reshape(3, 3)
    root = data.qpos[lo.rq : lo.rq + 3]
    velocity = data.qvel[lo.rv : lo.rv + 3]
    sign = 1.0 if team == "red" else -1.0
    own_goal = rotation.T @ (np.array([-sign * 8, 0, root[2]]) - root)
    attack = rotation.T @ np.array([sign, 0, 0])
    identity = np.eye(4)[ROLES.index(role)]
    ball = rotation.T @ (data.qpos[lo.bq : lo.bq + 3] - root)
    ball_velocity = rotation.T @ data.qvel[lo.bv : lo.bv + 3]
    feet = [
        rotation.T @ (kinematics.xpos[m.body(prefix + side + "_ankle_roll_link").id] - root)
        for side in ("left", "right")
    ]
    previous = np.asarray(previous_body_command)
    if previous.shape != (3,):
        raise ValueError("previous actual3-command required, not current label")
    neighbors: list[np.ndarray] = []
    other_team = "blue" if team == "red" else "red"
    for neighbor_team in (team, other_team):
        for neighbor_role in ROLES:
            other_prefix = neighbor_team + "_" + neighbor_role + "/"
            if other_prefix == prefix:
                continue
            other = layouts[other_prefix]
            position = rotation.T @ (data.qpos[other.rq : other.rq + 3] - root)
            relative_velocity = rotation.T @ (data.qvel[other.rv : other.rv + 3] - velocity)
            neighbors.extend((position[:2], relative_velocity[:2]))
    observation = np.concatenate(
        [
            own_goal[:2],
            attack[:2],
            identity,
            ball,
            ball_velocity,
            rotation.T @ velocity,
            data.qvel[lo.rv + 3 : lo.rv + 6],
            rotation.T @ np.array([0, 0, -1]),
            *feet,
            data.qpos[lo.qi],
            data.qvel[lo.vi],
            previous,
            *neighbors,
        ]
    ).astype(np.float32)
    if observation.shape != (ROLE_COMMAND_OBSERVATION_DIM,) or not np.isfinite(observation).all():
        raise ValueError("finite118-column pre-action native observation required")
    return observation


class NativeRoleCommandPolicy:
    """Owned NumPy inference, no simulator/transport access or execution authority."""

    def __init__(self, path: Path, *, expected_hash: str, expected_binding: str) -> None:
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= 4 * 1024 * 1024
        ):
            raise ValueError("bounded ordinary command model required")
        payload = path.read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("exact command model required")
        value = json.loads(payload)
        if (
            value["schema"] != ROLE_COMMAND_SCHEMA
            or value["binding_hash"] != expected_binding
            or value["observation_dim"] != ROLE_COMMAND_OBSERVATION_DIM
            or value["action_semantics"] != "ABSOLUTE_BODY_VELOCITY_COMMAND_BEFORE_CLEARANCE"
            or value["hardware_authorized"] is not False
            or value["optimizer_updates"] != 512
            or value["training_kind"] != "PUBLIC_COURSE_COACHED_BEHAVIOR_CLONING"
        ):
            raise ValueError("explicit bound SIM command model semantics required")
        shapes = {
            "mean": (118,),
            "std": (118,),
            "w0": (128, 118),
            "b0": (128,),
            "w1": (128, 128),
            "b1": (128,),
            "w2": (3, 128),
            "b2": (3,),
        }
        self.arrays: dict[str, np.ndarray] = {}
        for key, shape in shapes.items():
            array = np.array(value["tensors"][key], np.float32, copy=True)
            if array.shape != shape or not np.isfinite(array).all() or (abs(array) > 1e5).any():
                raise ValueError("finite exact-shaped command weights required")
            array.flags.writeable = False
            self.arrays[key] = array
        if (self.arrays["std"] < 0.02).any():
            raise ValueError("positive trained observation scale required")
        self.metadata = {key: item for key, item in value.items() if key != "tensors"}

    def propose(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        if observation.shape != (118,) or not np.isfinite(observation).all():
            raise ValueError("finite pre-action command context required")
        a = self.arrays
        x = np.clip((observation.astype(np.float32) - a["mean"]) / a["std"], -10, 10)
        for layer in (0, 1):
            x = np.tanh(a[f"w{layer}"] @ x + a[f"b{layer}"])
        latent = np.tanh(a["w2"] @ x + a["b2"])
        command = latent * np.array([0.75, 0.5, 1], np.float32) + np.array([0.25, 0, 0], np.float32)
        if not np.isfinite(command).all():
            raise FloatingPointError("nonfinite command inference")
        return np.asarray(command, dtype=np.float32).copy()
