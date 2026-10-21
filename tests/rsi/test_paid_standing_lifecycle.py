from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle


@pytest.mark.parametrize("variant,skills", [(False, 3), (3, 3), (-1, 3), (1, 2), (1, True)])
def test_only_predeclared_variants_and_horizons_can_allocate_worlds(variant, skills):
    with pytest.raises(ValueError):
        run_standing_lifecycle(None, {}, None, variant=variant, skills=skills, store=lambda *_: {})


@pytest.mark.parametrize("record_initial", [False, True])
@pytest.mark.parametrize("student_first", [False, True])
def test_first_body_failure_never_restarts_into_standing_or_another_kick(
    record_initial, student_first
):
    """Flow-only fixture, not native physics evidence."""
    data = SimpleNamespace(
        **{
            k: np.zeros(1)
            for k in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "mocap_pos",
                "mocap_quat",
                "qfrc_applied",
                "xfrc_applied",
                "act",
            )
        },
        time=0.0,
    )
    ep = SimpleNamespace(native_steps=0, env=SimpleNamespace(data=data))

    class Bench:
        neural_rows = 0

        @property
        def native_steps(self):
            return ep.native_steps

        def make(self, case):
            return ep

        def collect_current(self, ep, actor, *, stop_at_reference):
            assert stop_at_reference
            assert actor is selected
            ep.native_steps = 10
            data.time = 0.02
            self.neural_rows = 1
            return dict(NN_action=np.zeros((1, 29)), NN_nominal_action=np.zeros((1, 29))), dict(
                body_failed=True
            )

    captured = {}

    def store(key, raw):
        captured[key] = {k: v.copy() for k, v in raw.items()}
        return dict(raw_path=key, raw_hash="fixture")

    parent = SimpleNamespace(base=object())
    selected = parent if student_first else parent.base
    result = run_standing_lifecycle(
        Bench(),
        dict(source="motion-right.npz"),
        parent,
        variant=2 if student_first else 1,
        skills=3,
        store=store,
        record_initial=record_initial,
        motor_schedule=("student", "base", "base") if student_first else None,
    )
    assert result["status"] == "KICK_BODY_FAILURE" and len(result["kicks"]) == 1
    assert result["stands"] == result["boundaries"] == []
    assert not result["stable"] and result["body_resets"] == 0
    assert ("initial" in result) == record_initial
    if record_initial:
        assert len(captured["world-initial"]) == 10
        assert captured["world-initial"]["time"] == 0 and data.time == 0.02
