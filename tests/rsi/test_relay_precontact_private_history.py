import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_precontact_private_history import precontact_private_history


def traces():
    n = 3
    def hidden():
        return np.zeros((n, 2, 1, 256), dtype=np.float32)
    p = dict(
        control=np.arange(n),
        NN_h_before=hidden(),
        NN_c_before=hidden(),
        NN_h_after=hidden(),
        NN_c_after=hidden(),
        NN_obs=np.zeros((n, 160), np.float32),
        NN_action=np.zeros((n, 29), np.float32),
    )
    s = dict(
        control=np.arange(n),
        h_before=hidden(),
        c_before=hidden(),
        h_after=hidden(),
        c_after=hidden(),
        obs=np.zeros((n, 160), np.float32),
    )
    v = dict(
        control=np.arange(n),
        obs=np.zeros((n, 98), np.float32),
        action=np.zeros((n, 29), np.float32),
        phase=np.zeros(n, np.float32),
    )
    return p, v, s


def test_private_history_uses_only_committed_previous_action():
    p, v, s = traces()
    p["NN_action"][0] = 2
    p["NN_obs"][1, 125:154] = 2
    v["action"][0] = 3
    v["obs"][1, 69:98] = 3
    v["phase"][0] = 0.5
    result = precontact_private_history(p, v, s, control=1)
    assert np.all(result["passer_last_action"] == 2)
    assert np.all(result["receiver_velocity_last_action"] == 3)
    assert result["receiver_velocity_phase"] == 0.5


@pytest.mark.parametrize("defect", ["hidden", "action", "phase", "clock"])
def test_private_history_rejects_wrong_ownership_or_clock(defect):
    p, v, s = traces()
    if defect == "hidden":
        s["h_before"][1, 0, 0, 0] = 1
    elif defect == "action":
        v["obs"][1, 69] = 1
    elif defect == "phase":
        v["phase"][0] = 1
    else:
        p["control"][1] = 2
    with pytest.raises(ValueError):
        precontact_private_history(p, v, s, control=1)
