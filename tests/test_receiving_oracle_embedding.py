import numpy as np
import pytest

from rosclaw_soccer.training.receiving_oracle_embedding import embed_leg_oracle_in_body
from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)


def test_embedding_preserves_private_filter_and_admission_history():
    rng = np.random.default_rng(9217100)
    leg = ReceivingOracleSchedule(
        "blue.playmaker",
        "A0_leg12",
        30,
        20,
        tuple(tuple(float(v) for v in row) for row in rng.uniform(-1, 1, (4, 12))),
    )
    body = embed_leg_oracle_in_body(leg)
    assert body.substrate == "A1_body29" and body.agent_id == leg.agent_id
    assert body.start_frame == leg.start_frame and body.knot_frames == leg.knot_frames
    assert leg.contract_hash != body.contract_hash
    assert all(
        a[:12] == b and a[12:] == (0.0,) * 17 for a, b in zip(body.knots, leg.knots, strict=True)
    )
    leg_cursor, body_cursor = ReceivingOracleCursor(leg), ReceivingOracleCursor(body)
    predecessor = rng.uniform(-0.1, 0.1, 12)
    for frame in range(300):
        active = frame % 13 not in (0, 1, 2)
        a = leg_cursor.step(frame, active=active, predecessor=predecessor)
        b = body_cursor.step(frame, active=active, predecessor=predecessor)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b[:12])
            np.testing.assert_array_equal(b[12:], np.zeros(17))
            predecessor = a


@pytest.mark.parametrize("substrate", ["A1_body29", "A3_sonic_residual"])
def test_different_foundation_or_already_whole_body_cannot_be_silently_mapped(substrate):
    source = ReceivingOracleSchedule("red.finisher", substrate, 0, 20, ((0.0,) * 29,))
    with pytest.raises(ValueError):
        embed_leg_oracle_in_body(source)


def test_untyped_schedule_rejected():
    with pytest.raises(ValueError):
        embed_leg_oracle_in_body(None)
