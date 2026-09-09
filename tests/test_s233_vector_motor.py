from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.vector_motor import G1VectorMotorBatch, G1VectorMotorConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"environment_count": 0},
        {"environment_count": 4097},
        {"environment_count": True},
        {"device": "cpu"},
        {"device": "cuda"},
        {"device": "cuda:-1"},
        {"physics_substeps": 11},
        {"physics_substeps": 0},
        {"physics_substeps": True},
        {"guard_margin_rad": 0.03},
        {"guard_margin_rad": float("nan")},
        {"activation_ceiling": "REAL"},
    ],
)
def test_vector_motor_configuration_is_bounded(kwargs):
    with pytest.raises(ValueError):
        G1VectorMotorConfig(**kwargs)


def shell():
    torch = pytest.importorskip("torch")
    batch = object.__new__(G1VectorMotorBatch)
    batch._torch = torch
    batch.device = torch.device("cpu")
    batch.config = G1VectorMotorConfig(environment_count=2)
    batch._limits = torch.ones(29) * 10
    calls = []
    batch._step = lambda torque, pd: calls.append((torque, pd))
    return batch, calls


def test_direct_torque_has_no_pd_teacher_and_is_not_silently_clipped():
    batch, calls = shell()
    batch.step_torque(np.ones((2, 29)) * 0.5)
    assert calls[0][1] is None
    np.testing.assert_array_equal(calls[0][0], np.ones((2, 29)) * 5)
    for value in (np.ones((2, 29)) * 1.01, np.full((2, 29), float("nan")), np.zeros((2, 28))):
        with pytest.raises(ValueError):
            batch.step_torque(value)
    assert len(calls) == 1


def test_pd_teacher_is_separate_and_bounded():
    batch, calls = shell()
    q = np.zeros((2, 29))
    batch.step_pd(q, q + 40, q + 2)
    assert calls[0][0] is None
    assert calls[0][1] is not None
    with pytest.raises(ValueError):
        batch.step_pd(q, q + 301, q)
    assert len(calls) == 1


def test_execution_requires_explicit_reset():
    batch = object.__new__(G1VectorMotorBatch)
    batch._ready = False
    with pytest.raises(RuntimeError, match="reset"):
        batch._step(None, None)


def test_invalid_reset_does_not_mutate_or_enable_execution():
    batch, _ = shell()
    batch.cpu_model = SimpleNamespace(
        nq=43, nv=41, jnt_type=np.array([0]), jnt_qposadr=np.array([0])
    )
    batch._ready = False
    with pytest.raises(ValueError, match="normalized"):
        batch.reset(np.zeros((2, 43)), np.zeros((2, 41)))
    with pytest.raises(ValueError, match="finite"):
        batch.reset(np.full((2, 43), float("nan")), np.zeros((2, 41)))
    assert batch._ready is False


def test_graph_warmup_cannot_be_inserted_into_an_episode():
    batch = object.__new__(G1VectorMotorBatch)
    batch._ready = True
    batch.episode_resets = 1
    batch._physics_graph = None
    with pytest.raises(RuntimeError, match="first episode"):
        batch.prepare_physics_graph()
