"""Bounded, independent CPU SONIC inference for offline simulation learning.

No physics stepping, actuator transport, policy update or execution authority.
Unlike the Torch conversion, this retains the original ONNX numerical path.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController


class IndependentSonicCpuBatch:
    """Own preconstructed controllers exclusively until close.

    Read-only ONNX sessions may be shared, mutable histories/actions may not.
    Caller resets each simulated world before begin_episode and supplies its
    actual pre-control state at each consecutive 50 Hz tick. All worker tasks
    finish before results or errors return. A partial failure latches until an
    explicit new episode resets EVERY controller; no selective retry.
    """

    def __init__(self, controllers: Sequence[G1SonicRunupController], *, workers: int = 4):
        if (
            type(workers) is not int
            or not 1 <= workers <= 8
            or not 1 <= len(controllers) <= 32
            or any(not isinstance(c, G1SonicRunupController) for c in controllers)
        ):
            raise ValueError("1..32 SONIC controllers and 1..8 CPU workers required")
        self._controllers = tuple(controllers)
        self._assert_private()
        self._frames = min(c.config.execution_frames for c in controllers)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=min(workers, len(controllers)), thread_name_prefix="sonic-sim"
        )
        self._next_frame: int | None = None
        self._faulted = False
        self._closed = False

    def _assert_private(self) -> None:
        count = len(self._controllers)
        if (
            len({id(c) for c in self._controllers}) != count
            or len({id(c._history) for c in self._controllers}) != count
        ):
            raise ValueError("independent SONIC controllers and histories required")
        for index, controller in enumerate(self._controllers):
            if any(
                np.shares_memory(controller.action, other.action)
                for other in self._controllers[:index]
            ):
                raise ValueError("independent SONIC action buffers required")

    def _states(
        self, qpos: NDArray[np.floating], qvel: NDArray[np.floating]
    ) -> list[SimpleNamespace]:
        count = len(self._controllers)
        for value, width in ((qpos, 43), (qvel, 41)):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (count, width)
                or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(value).all()
                or (np.abs(value) > 1e6).any()
            ):
                raise ValueError("bounded finite native G1/ball batch required")
        if np.any(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1) > 1e-5):
            raise ValueError("normalized physical root quaternion required")
        return [
            SimpleNamespace(qpos=q.copy(), qvel=v.copy()) for q, v in zip(qpos, qvel, strict=True)
        ]

    def begin_episode(self, qpos: NDArray[np.floating], qvel: NDArray[np.floating]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SONIC batch is closed")
            self._faulted = True
            self._next_frame = None
            states = self._states(qpos, qvel)
            self._assert_private()
            for controller, state in zip(self._controllers, states, strict=True):
                controller.reset(state)
            self._assert_private()
            self._next_frame = 0
            self._faulted = False

    def step(
        self, frame: int, qpos: NDArray[np.floating], qvel: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        with self._lock:
            try:
                if self._closed or self._faulted or self._next_frame is None:
                    raise RuntimeError("begin a new open, unfaulted SONIC batch episode")
                if type(frame) is not int or frame != self._next_frame or frame >= self._frames:
                    raise ValueError("consecutive SONIC reference frame required")
                states = self._states(qpos, qvel)
                self._assert_private()

                def update(index: int) -> NDArray[np.float64]:
                    controller, state = self._controllers[index], states[index]
                    if frame:
                        controller.observe(state)
                    target = np.asarray(controller.update(state, frame), dtype=np.float64)
                    if target.shape != (29,) or not np.isfinite(target).all():
                        raise FloatingPointError("invalid SONIC target")
                    return target.copy()

                futures: list[Future[NDArray[np.float64]]] = []
                try:
                    for i in range(len(states)):
                        futures.append(self._pool.submit(update, i))
                finally:
                    # Even an executor submission failure must drain already
                    # accepted tasks before exposing a partially advanced batch.
                    wait(futures)
                result = np.stack([future.result() for future in futures])
                self._assert_private()
                self._next_frame = frame + 1
                result.flags.writeable = False
                return result
            except Exception:
                self._faulted = True
                raise

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pool.shutdown(wait=True)

    def __enter__(self) -> IndependentSonicCpuBatch:
        if self._closed:
            raise RuntimeError("SONIC batch is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
