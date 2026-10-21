import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_trainable_actor import onnx_to_torch_gates


@pytest.mark.parametrize("width", [None, 160, 128])
def test_gate_order_is_iofc_to_ifgo_not_identity(width):
    shape = (512,) if width is None else (512, width)
    a = np.broadcast_to(
        np.repeat(np.arange(4, dtype=np.float32), 128).reshape(
            (512,) + (() if width is None else (1,))
        ),
        shape,
    ).copy()
    b = onnx_to_torch_gates(a)
    assert b.dtype == torch.float32
    assert [float(x.flatten()[0]) for x in b.chunk(4)] == [0, 2, 3, 1]
    a[:] = 7
    assert float(b.flatten()[0]) == 0


@pytest.mark.parametrize(
    "a", [np.zeros(511, np.float32), np.zeros(512, np.float64), np.full(512, np.nan, np.float32)]
)
def test_gate_conversion_rejects_bad_arrays(a):
    with pytest.raises(ValueError):
        onnx_to_torch_gates(a)
