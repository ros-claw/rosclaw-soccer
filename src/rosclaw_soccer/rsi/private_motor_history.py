"""Explicit private motor state codecs; never infer policy choice from source logs.

Legacy two-tensor histories retain their exact representation. Option memory
has a separate typed contract and numeric trace fields, not fake LSTM channels.
"""

import numpy as np
import torch

from rosclaw_soccer.rsi.motor_option_state import MotorOptionState

MotorHistory = tuple[torch.Tensor, torch.Tensor] | MotorOptionState


def recurrent_history(
    state: MotorHistory, *, hidden_width: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(hidden_width) is not int or hidden_width not in (128, 256):
        raise ValueError("known private motor history width required")
    if type(state) is MotorOptionState:
        state.validate()
        if hidden_width != 256:
            raise ValueError("option history cannot reinterpret legacy recurrent width")
        return state.recurrent
    if (
        type(state) is not tuple
        or len(state) != 2
        or any(
            not isinstance(t, torch.Tensor)
            or t.shape != (2, 1, hidden_width)
            or t.dtype != torch.float32
            or t.device.type != "cpu"
            or not torch.isfinite(t).all()
            for t in state
        )
        or state[0].untyped_storage().data_ptr() == state[1].untyped_storage().data_ptr()
    ):
        raise ValueError("explicit finite private CPU h/c required")
    return state


def clone_history(state: MotorHistory, *, hidden_width: int = 256) -> MotorHistory:
    h, c = recurrent_history(state, hidden_width=hidden_width)
    if type(state) is MotorOptionState:
        return state.clone()
    return h.detach().clone(), c.detach().clone()


def history_trace(
    before: MotorHistory, after: MotorHistory, *, hidden_width: int = 256
) -> dict[str, np.ndarray]:
    bh, bc = recurrent_history(before, hidden_width=hidden_width)
    ah, ac = recurrent_history(after, hidden_width=hidden_width)
    if type(before) is not type(after):
        raise ValueError("private history representation cannot change mid-control")
    trace = dict(
        NN_h_before=bh.numpy().copy(),
        NN_c_before=bc.numpy().copy(),
        NN_h_after=ah.numpy().copy(),
        NN_c_after=ac.numpy().copy(),
    )
    if type(before) is MotorOptionState and type(after) is MotorOptionState:
        if (
            before.generation_hash != after.generation_hash
            or after.controls != before.controls + 1
            or after.option is None
            or (before.option is not None and after.option != before.option)
        ):
            raise ValueError("one own control must preserve model generation and option commitment")
        trace.update(
            NN_option_before=np.asarray(
                -1 if before.option is None else before.option, dtype=np.int64
            ),
            NN_option_after=np.asarray(after.option, dtype=np.int64),
            NN_option_controls_before=np.asarray(before.controls, dtype=np.int64),
            NN_option_controls_after=np.asarray(after.controls, dtype=np.int64),
            NN_option_generation=np.frombuffer(
                bytes.fromhex(before.generation_hash[7:]), dtype=np.uint8
            ).copy(),
        )
    return trace
