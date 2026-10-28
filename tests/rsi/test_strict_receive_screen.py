import numpy as np

from rosclaw_soccer.rsi.strict_receive_screen import screen_receive


def history():
    f = np.zeros((200, 2, 3))
    q = np.zeros((200, 12))
    v = np.zeros((200, 12))
    failed = np.zeros((20, 2), dtype=bool)
    return f, q, v, failed


def test_screen_requires_actual_clean_foot_and_observed_tail():
    f, q, v, failed = history()
    kw = dict(ball_q=0, ball_v=0, root_q=3, root_v=3)
    assert screen_receive(f, q, v, failed, **kw)["status"] == "NO_PASSER_FOOT"
    f[10, 0, 0] = 20
    assert screen_receive(f, q, v, failed, **kw)["status"] == "NO_RECEIVER_FOOT"
    f[50, 1, 0] = 20
    assert screen_receive(f, q, v, failed, **kw)["screen_candidate"]
    f[51, 1, 2] = 2
    assert not screen_receive(f, q, v, failed, **kw)["screen_candidate"]
    f[51, 1, 2] = 0
    v[150, 0] = 0.6
    assert not screen_receive(f, q, v, failed, **kw)["screen_candidate"]
    f[50, 1, 0] = 0
    f[150, 1, 0] = 20
    assert screen_receive(f, q, v, failed, **kw)["status"] == "POST_CONTACT_CENSORED"
