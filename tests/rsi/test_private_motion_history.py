import numpy as np
import pytest

from rosclaw_soccer.rsi.private_motion_history import join_private_motion_history


def trace():
    sizes = dict(NN_obs=160, NN_action=29)
    rows = {k: np.zeros((110, n), np.float32) for k, n in sizes.items()}
    for prefix in ("h", "c"):
        rows[f"NN_{prefix}_before"] = np.broadcast_to(
            np.arange(110, dtype=np.float32)[:, None, None, None], (110, 2, 1, 256)
        ).copy()
        rows[f"NN_{prefix}_after"] = rows[f"NN_{prefix}_before"] + 1
    return {k: x[:10].copy() for k, x in rows.items()}, {k: x[10:].copy() for k, x in rows.items()}


def test_join_preserves_actual_recurrent_history_without_reset_or_padding():
    kick, stand = trace()
    result = join_private_motion_history(kick, stand)
    assert len(result["NN_obs"]) == 110
    assert result["NN_h_before"][10, 0, 0, 0] == 10
    result["NN_action"][0, 0] = 1
    assert not kick["NN_action"].any()


@pytest.mark.parametrize("fault", ["reset", "borrowed_prefix", "NaN", "short", "wrong_width"])
def test_incomplete_or_incoherent_witness_is_rejected(fault):
    kick, stand = trace()
    if fault == "reset":
        stand["NN_h_before"][0] = 0
    elif fault == "borrowed_prefix":
        kick["NN_c_before"][0] = 1
    elif fault == "NaN":
        kick["NN_obs"][0, 0] = np.nan
    elif fault == "short":
        stand = {k: x[:-1] for k, x in stand.items()}
    else:
        kick["NN_h_before"] = np.zeros((10, 2, 1, 128), np.float32)
    with pytest.raises(ValueError):
        join_private_motion_history(kick, stand)
