"""Extract and validate own-agent history at an observed simulation boundary."""

import numpy as np


def precontact_private_history(passer: dict, velocity: dict, shadow: dict, *, control: int) -> dict:
    """Use only prior executed actions and the boundary's existing own history.

    This does not construct a live controller or validate a counterfactual.
    """
    if type(control) is not int or control < 1:
        raise ValueError("positive actual boundary required")
    t = control
    for trace, control_key in ((passer, "control"), (velocity, "control"), (shadow, "control")):
        if (
            control_key not in trace
            or len(trace[control_key]) <= t
            or not np.array_equal(trace[control_key][: t + 1], np.arange(t + 1))
        ):
            raise ValueError("complete consecutive actual control histories required")

    def item(trace, key, shape):
        value = trace.get(key)
        if (
            not isinstance(value, np.ndarray)
            or value.shape[0] <= t
            or value.shape[1:] != shape
            or value.dtype != np.float32
            or not np.isfinite(value[: t + 1]).all()
        ):
            raise ValueError("finite complete private actor history required: " + key)
        return value

    ph = item(passer, "NN_h_before", (2, 1, 256))
    pc = item(passer, "NN_c_before", (2, 1, 256))
    pa = item(passer, "NN_h_after", (2, 1, 256))
    pb = item(passer, "NN_c_after", (2, 1, 256))
    po = item(passer, "NN_obs", (160,))
    px = item(passer, "NN_action", (29,))
    sh = item(shadow, "h_before", (2, 1, 256))
    sc = item(shadow, "c_before", (2, 1, 256))
    sa = item(shadow, "h_after", (2, 1, 256))
    sb = item(shadow, "c_after", (2, 1, 256))
    so = item(shadow, "obs", (160,))
    vo = item(velocity, "obs", (98,))
    va = item(velocity, "action", (29,))
    phase = item(velocity, "phase", ())
    if (
        not np.array_equal(ph[t], pa[t - 1])
        or not np.array_equal(pc[t], pb[t - 1])
        or not np.array_equal(sh[t], sa[t - 1])
        or not np.array_equal(sc[t], sb[t - 1])
        or not np.array_equal(po[t, 125:154], px[t - 1])
        or not np.array_equal(vo[t, 69:98], va[t - 1])
        or not 0 <= float(phase[t - 1]) < 1
    ):
        raise ValueError("own recurrent and actual previous-action continuity required")
    return dict(
        passer_h=ph[t].copy(),
        passer_c=pc[t].copy(),
        passer_last_action=px[t - 1].copy(),
        receiver_shadow_h=sh[t].copy(),
        receiver_shadow_c=sc[t].copy(),
        receiver_shadow_last_action=so[t, 125:154].copy(),
        receiver_velocity_last_action=va[t - 1].copy(),
        receiver_velocity_phase=phase[t - 1].copy(),
        control=np.array(t, dtype=np.int64),
    )
