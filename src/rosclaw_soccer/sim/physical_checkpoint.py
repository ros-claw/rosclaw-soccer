"""CPU MuJoCo integration checkpoints; controller memory is explicitly separate.

This primitive is not a complete receiving initial state. Callers must also
restore controller histories, RNGs, course clocks and policy bindings before
claiming paired replay. No pickle or hardware execution is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes


def compiled_model_hash(model: mujoco.MjModel) -> str:
    """Bind mutable compiled physics, not only the original XML filename."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hash_bytes(buffer.tobytes())


@dataclass(frozen=True)
class PhysicalCheckpoint:
    model_hash: str
    mujoco_version: str
    state_bytes: bytes
    state_hash: str

    @classmethod
    def capture(cls, model: mujoco.MjModel, data: mujoco.MjData) -> PhysicalCheckpoint:
        if data.model is not model:
            raise ValueError("data must belong to the supplied model")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        if not np.isfinite(state).all():
            raise ValueError("nonfinite integration state")
        payload = state.astype("<f8").tobytes()
        return cls(compiled_model_hash(model), mujoco.__version__, payload, hash_bytes(payload))

    def restore(self, model: mujoco.MjModel) -> mujoco.MjData:
        """Return fresh data, leaving any existing world untouched on rejection.

        Derived contacts and sensors are intentionally not refreshed here:
        mj_forward can change solver warm-start state. The next mj_step computes
        derived fields. A controller requiring pre-step sensors must explicitly
        qualify that refresh in its own replay test.
        """
        if self.mujoco_version != mujoco.__version__:
            raise ValueError("MuJoCo version mismatch")
        if self.model_hash != compiled_model_hash(model):
            raise ValueError("compiled physics mismatch")
        if type(self.state_bytes) is not bytes or hash_bytes(self.state_bytes) != self.state_hash:
            raise ValueError("integration state integrity mismatch")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if len(self.state_bytes) != 8 * mujoco.mj_stateSize(model, spec):
            raise ValueError("integration state size mismatch")
        state = np.frombuffer(self.state_bytes, dtype="<f8").astype(np.float64)
        if not np.isfinite(state).all():
            raise ValueError("nonfinite integration state")
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, state, spec)
        return data
