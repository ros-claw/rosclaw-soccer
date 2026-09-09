"""Positive-force contact observations with an explicit physics-rate ledger.

Contact groups are supplied by the caller (feet, palms, etc.). The observer
does not launch a ball, modify geometry, reset physics, or award promotion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


class PhysicsSampleClock:
    """A missed physical step invalidates the episode, not just that sample."""

    def __init__(self) -> None:
        self.episode = 0
        self.step = 0
        self.samples = 0
        self.valid = False

    def begin(self, episode: int, step: int) -> None:
        if (
            type(episode) is not int
            or type(step) is not int
            or episode <= self.episode
            or step < self.step
        ):
            raise ValueError("new explicit physics episode required")
        self.episode, self.step, self.samples, self.valid = episode, step, 0, True

    def observe(self, episode: int, step: int) -> None:
        if (
            not self.valid
            or type(episode) is not int
            or type(step) is not int
            or episode != self.episode
            or step != self.step + 1
        ):
            self.valid = False
            raise RuntimeError("contact evidence skipped/repeated a physics step or reset")
        self.step, self.samples = step, self.samples + 1


class G1VectorContactProbe:
    """Sample after *each* 2 ms motor step, never just each 50 Hz action."""

    def __init__(
        self,
        motor: Any,
        *,
        target_geoms: Sequence[int],
        groups: Mapping[str, Sequence[int]],
        minimum_normal_force_n: float = 1.0,
    ) -> None:
        import numpy as np
        import torch
        import warp as wp

        if (
            not math.isfinite(minimum_normal_force_n)
            or not 0 < minimum_normal_force_n <= 100
            or not 1 <= len(groups) <= 16
            or motor.config.physics_substeps != 1
        ):
            raise ValueError("bounded contact groups and single-step motor sampling required")

        def qualified_ids(values: Sequence[int]) -> list[int]:
            if (
                not values
                or any(type(i) is not int or not 0 <= i < motor.cpu_model.ngeom for i in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("unique existing physical geometry IDs required")
            return sorted(values)

        targets = qualified_ids(target_geoms)
        resolved = {}
        for name, values in groups.items():
            if not isinstance(name, str) or not name.isidentifier() or len(name) > 64:
                raise ValueError("bounded contact group identity required")
            resolved[name] = qualified_ids(values)
            if set(resolved[name]) & set(targets):
                raise ValueError("contact counterpart group overlaps target geometry")
        self.motor, self._torch = motor, torch
        self.clock = PhysicsSampleClock()
        self.minimum_force = minimum_normal_force_n
        self.configuration_hash = hash_json(
            {
                "target_geoms": targets,
                "groups": resolved,
                "minimum_normal_force_n": self.minimum_force,
            }
        )
        self._target = torch.tensor(targets, device=motor.device)
        self._groups = {n: torch.tensor(v, device=motor.device) for n, v in resolved.items()}
        self._geom = wp.to_torch(motor._data.contact.geom)
        self._world = wp.to_torch(motor._data.contact.worldid)
        self._count = wp.to_torch(motor._data.nacon)
        self._indices = wp.array(
            np.arange(len(self._geom), dtype=np.int32), dtype=wp.int32, device=str(motor.device)
        )
        self._forces = wp.zeros(len(self._geom), dtype=wp.spatial_vector, device=str(motor.device))
        self._force = wp.to_torch(self._forces)
        self._slots = torch.arange(len(self._geom), device=motor.device)

    def begin_episode(self) -> None:
        if not self.motor._ready:
            raise RuntimeError("contact probe requires initialized finite motor state")
        self.clock.begin(self.motor.episode_resets, self.motor.physics_steps)

    def sample(self) -> dict[str, Any]:
        import warp as wp

        self.clock.observe(self.motor.episode_resets, self.motor.physics_steps)
        torch, motor = self._torch, self.motor
        if not motor._ready:
            self.clock.valid = False
            raise RuntimeError("motor state invalidated contact evidence")
        stream = wp.stream_from_torch(torch.cuda.current_stream(motor.device))
        try:
            with wp.ScopedStream(stream):
                motor._mjw.contact_force(
                    motor._model, motor._data, self._indices, False, self._forces
                )
            normal = self._force[:, 0]
            valid = self._slots < self._count[0]
            if not bool(torch.isfinite(normal[valid]).all()) or bool(
                (
                    (self._world[valid] < 0)
                    | (self._world[valid] >= motor.config.environment_count)
                ).any()
            ):
                raise FloatingPointError("invalid physical contact force or world identity")
            active = valid & (normal > self.minimum_force)
            g0, g1 = self._geom.unbind(1)
            ball0, ball1 = torch.isin(g0, self._target), torch.isin(g1, self._target)
            result = {}
            for name, geoms in self._groups.items():
                counterpart = (ball0 & torch.isin(g1, geoms)) | (ball1 & torch.isin(g0, geoms))
                values = torch.where(active & counterpart, normal, 0)
                output = torch.zeros(motor.config.environment_count, device=motor.device)
                output.scatter_reduce_(
                    0,
                    self._world.long().clamp(0, motor.config.environment_count - 1),
                    values,
                    reduce="amax",
                )
                result[name] = output
            return result
        except Exception:
            self.clock.valid = False
            raise
