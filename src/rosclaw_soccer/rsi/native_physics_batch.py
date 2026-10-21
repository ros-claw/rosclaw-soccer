"""Experimental SIM_ONLY vector physics, with one native MjData per world.

Physics uses unmodified CPU MuJoCo, not float32 GPU contacts. This adapter does
not implement a reward, actor, policy promotion, or hardware transport. Shared
MjModel is read-only while native steps run in independent Python threads.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray


class NativePhysicsBatch:
    """Fixed-budget independent native integration; no active-world teleporting."""

    def __init__(self, model: Any, worlds: int, *, threads: int = 4) -> None:
        if type(worlds) is not int or worlds < 1 or type(threads) is not int or threads < 1:
            raise ValueError("positive integer world and thread counts required")
        if mujoco.get_mjcb_control() is not None:
            raise ValueError("external control callbacks are not allowed")
        self.model = model
        self.data = tuple(mujoco.MjData(model) for _ in range(worlds))
        self._pool = ThreadPoolExecutor(max_workers=min(worlds, threads))
        self._ready = False
        self._closed = False
        self.native_steps = 0

    def _array(self, value: Any, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"finite {name} of shape {shape} required")
        return result

    def restore(
        self,
        *,
        qpos: Any,
        qvel: Any,
        previous_ctrl: Any,
        warmstart: Any,
        time: Any,
    ) -> None:
        """Restore all worlds ONCE, including post-forward native solver seeds.

        Subsequent per-episode resets require a fresh batch. This diagnostic
        adapter must not overwrite an active trajectory or silently repair data.
        """
        if self._closed or self._ready:
            raise ValueError("fresh unused batch required for restoration")
        n, m = len(self.data), self.model
        q = self._array(qpos, (n, m.nq), "qpos")
        v = self._array(qvel, (n, m.nv), "qvel")
        ctrl = self._array(previous_ctrl, (n, m.nu), "previous ctrl")
        warm = self._array(warmstart, (n, m.nv), "warmstart")
        t = self._array(time, (n,), "time")
        if (t < 0).any():
            raise ValueError("nonnegative simulation time required")
        for i, d in enumerate(self.data):
            d.qpos[:] = q[i]
            d.qvel[:] = v[i]
            d.ctrl[:] = ctrl[i]
            d.qacc_warmstart[:] = warm[i]
            d.time = t[i]
            mujoco.mj_forward(m, d)
            d.qacc_warmstart[:] = warm[i]
        self._ready = True

    def advance(
        self,
        ctrl: Any,
        *,
        substeps: int,
        fields: tuple[str, ...] = ("qpos", "qvel"),
        world_ids: NDArray[np.int64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Integrate all worlds continuously, recording every declared substep.

        A worker error propagates; the batch cannot be reused after failure.
        Physics counters count actual completed calls even on failed attempts.
        """
        if self._closed or not self._ready:
            raise ValueError("restored live batch required")
        if type(substeps) is not int or substeps < 1:
            raise ValueError("positive integer substep budget required")
        allowed = {
            "qpos",
            "qvel",
            "qacc",
            "qfrc_actuator",
            "qfrc_bias",
            "qfrc_passive",
            "qfrc_constraint",
        }
        if not fields or len(set(fields)) != len(fields) or not set(fields) <= allowed:
            raise ValueError("unique declared physical fields required")
        targets = self._array(ctrl, (len(self.data), self.model.nu), "ctrl").copy()
        ids = np.arange(len(self.data), dtype=np.int64) if world_ids is None else world_ids
        if (
            not isinstance(ids, np.ndarray)
            or ids.dtype != np.int64
            or ids.ndim != 1
            or not len(ids)
            or len(set(ids.tolist())) != len(ids)
            or (ids < 0).any()
            or (ids >= len(self.data)).any()
        ):
            raise ValueError("unique nonempty explicit native world indices required")

        def step_world(i: int) -> tuple[dict[str, list[NDArray[np.float64]]], int, str | None]:
            d = self.data[i]
            rows: dict[str, list[NDArray[np.float64]]] = {k: [] for k in fields}
            done = 0
            try:
                d.ctrl[:] = targets[i]
                warnings = tuple(int(w.number) for w in d.warning)
                for _ in range(substeps):
                    mujoco.mj_step(self.model, d)
                    done += 1
                    if tuple(int(w.number) for w in d.warning) != warnings:
                        raise ValueError("native warning during physical integration")
                    for k in fields:
                        value = getattr(d, k).copy()
                        if not np.isfinite(value).all():
                            raise ValueError("nonfinite native physical state")
                        rows[k].append(value)
                return rows, done, None
            except Exception as exc:
                return rows, done, repr(exc)

        # Only selected live worlds are integrated. Unselected terminal states
        # and their simulation clocks remain untouched; output rows follow ids.
        results = list(self._pool.map(step_world, ids.tolist()))
        self.native_steps += sum(row[1] for row in results)
        errors = [
            dict(world=int(ids[i]), error=row[2])
            for i, row in enumerate(results)
            if row[2] is not None
        ]
        if errors:
            self._ready = False
            raise ValueError(f"native batch failed; no automatic retry: {errors}")
        return {k: np.array([row[0][k] for row in results], dtype=np.float64) for k in fields}

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._closed = True

    def __enter__(self) -> "NativePhysicsBatch":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
