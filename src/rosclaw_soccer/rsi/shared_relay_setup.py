"""Pinned-asset SIM setup shared by relay collection and later auditing.

No import-time model allocation or physics. Initial placements happen once,
before the shared clock exists; no API here can reset an active world.
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


class SharedRelaySetup:
    def __init__(
        self, pins: dict[str, str], contexts: list[Any], *, bench: Any, actor: Any
    ) -> None:
        import mujoco

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

        self.pins, self.contexts = pins, contexts
        self.bench, self.actor = bench, actor
        xmls = [Path(p) for p in pins if p.endswith("/cached-standard/declared-scene.xml")]
        self.xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        dimensions = inspect_native_ball_dimensions(
            self.model, geom_id=int(self.model.geom("soccer_ball_geom").id)
        )
        if not dimensions.size_and_mass_in_ifab_range:
            raise ValueError("same declared standard ball dimensions required")
        self.ball_dimensions = asdict(dimensions)
        paths = [Path(p) for p in pins if p.endswith("/velocity/v0/exported/policy.onnx")]
        configs = [Path(p) for p in pins if p.endswith("/velocity/v0/params/deploy.yaml")]
        if len(paths) != 1 or len(configs) != 1:
            raise ValueError("one exact frozen velocity backend required")
        self.velocity_path, self.velocity_config = paths[0], configs[0]
        self.velocity_parameters = self.velocity().params
        self.episodes = {}
        for i in (1, 4, 20, 21):
            ep = bench.make(contexts[i]["case"])
            q, _ = ep.layout.canonical(ep.env.data)
            qw, x, y, z = q[3:7]
            yaw = np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z))
            ep.env.data.qpos[ep.layout.rq : ep.layout.rq + 2] -= contexts[i][
                "backoff_m"
            ] * np.array([np.cos(yaw), np.sin(yaw)])
            mujoco.mj_forward(ep.env.model, ep.env.data)
            self.episodes[i] = ep

    def velocity(self) -> Any:
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy

        return G1VelocityPolicy(
            self.velocity_path,
            self.velocity_config,
            expected_model_hash=self.pins[str(self.velocity_path)],
            expected_config_hash=self.pins[str(self.velocity_config)],
        )

    def prepare(
        self, context: int, lateral_m: float
    ) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
        import mujoco

        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.shared_native_motor_world import (
            NativePDParameters,
            SharedNativeMotorWorld,
        )

        ep = self.episodes[context]
        m = self.model
        layouts = {
            p: NativeVelocityLayout(m, robot_prefix=p, ball_body_name="soccer_ball", player_count=2)
            for p in ("a/", "b/")
        }
        q, v = ep.layout.canonical(ep.env.data)
        forward = ep.goal[:2] - q[36:38]
        forward /= np.linalg.norm(forward)
        lateral = np.array([-forward[1], forward[0]])
        home, anchor = (
            q[36:38] + 4 * forward + lateral_m * lateral,
            q[36:38] + 4 * forward + 0.12 * lateral,
        )
        d = mujoco.MjData(m)
        for prefix, layout in layouts.items():
            d.qpos[layout.rq : layout.rq + 7], d.qvel[layout.rv : layout.rv + 6] = q[:7], v[:6]
            d.qpos[layout.qi], d.qvel[layout.vi] = q[7:36], v[6:35]
            if prefix == "b/":
                d.qpos[layout.rq : layout.rq + 2] = home
                qw, x, y, z = q[3:7]
                d.qpos[layout.rq + 3 : layout.rq + 7] = [-z, -y, x, qw]
                d.qvel[layout.rv : layout.rv + 2] *= -1
        al = layouts["a/"]
        d.qpos[al.bq : al.bq + 7], d.qvel[al.bv : al.bv + 6] = q[36:], v[35:]
        mujoco.mj_forward(m, d)
        for c in d.contact:
            b1, b2 = (int(m.geom_bodyid[g]) for g in c.geom)
            if al.ball in (b1, b2) and (b2 if b1 == al.ball else b1) != 0:
                raise ValueError("initial ball/body overlap; entire relay bank rejected")
        vp = self.velocity_parameters
        world = SharedNativeMotorWorld(
            m,
            d,
            players={
                "a/": NativePDParameters(ep.kp, ep.kd, ep.effort),
                "b/": NativePDParameters(np.array(vp.stiffness), np.array(vp.damping), ep.effort),
            },
            ball_body_name="soccer_ball",
        )
        return ep, world, forward, home, anchor


def motion_segment(clip: Any, start: int) -> Any:
    if type(start) is not int or not 0 <= start < clip.length:
        raise ValueError("explicit actual source reference segment required")
    ang = clip.body_ang_vel_w[start:].copy()
    ang[..., :2] *= -1
    return SimpleNamespace(
        length=clip.length - start,
        joint_pos=clip.joint_pos[start:].copy(),
        joint_vel=clip.joint_vel[start:].copy(),
        body_ang_vel_w=ang,
    )


def receiving_target_for_player(world: Any, prefix: str) -> np.ndarray:
    q, _ = world.layouts[prefix].canonical(world.data)
    qw, x, y, z = q[3:7]
    yaw = np.arctan2(2 * (qw * z + x * y), 1 - 2 * (y * y + z * z))
    forward, lateral = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    point = q[:2] + 0.25 * forward + 0.12 * lateral
    return np.array([point[0], point[1], 0.11], dtype=np.float64)
