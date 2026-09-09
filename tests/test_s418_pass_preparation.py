from dataclasses import replace

import pytest

from rosclaw_soccer.growth.pass_handoff import PassHandoff
from rosclaw_soccer.growth.pass_preparation import PassPreparationLedger, PassPreparationRequest
from rosclaw_soccer.sim.contracts import hash_json


def fixture():
    ledger = PassPreparationLedger("sha256:" + "a" * 64)
    previous = PassHandoff("blue.finisher", "blue.playmaker", 1.82, launch_target_xy=(5.0, 2.0))
    request = PassPreparationRequest("blue.finisher", "sha256:" + "b" * 64, 220, 4.4, (4.5, 2.1))
    arguments = dict(
        source="blue.finisher",
        receiver="blue.playmaker",
        accepted_target_xy=(4.5, 2.1),
        frame=220,
        time_sec=4.4,
        source_ready=True,
        receiver_ready=True,
        motor_target_valid=True,
        agreement_hash="sha256:" + "c" * 64,
        motor_target_hash="sha256:" + "d" * 64,
    )
    return ledger, previous, request, arguments


def test_fresh_agreement_binds_new_preparation_but_does_not_launch_or_grant_possession():
    ledger, previous, request, arguments = fixture()
    following, receipt = ledger.bind(previous, request, **arguments)
    assert previous.created_sec == 1.82 and following.created_sec == 4.4
    assert following.lifetime_sec == previous.lifetime_sec == 3.0
    assert following.flight_window_sec == previous.flight_window_sec == 3.0
    assert following.source_foot_contact_sec is None
    assert following.launch_target_xy == request.target_xy
    assert not following.can_activate(4.9, progressed=True)
    assert not receipt["hardware_authorized"]
    digest = receipt.pop("receipt_hash")
    assert digest == hash_json(receipt)
    launched = following.observe_directed_launch(
        agent_id=following.source,
        foot=True,
        force_n=2.0,
        time_sec=4.9,
        ball_xy=(2.0, 1.5),
        ball_velocity_xy=(0.8, 0.2),
    )
    assert launched.source_foot_contact_sec == 4.9
    assert launched.expired(7.9) and not launched.expired(7.89)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_ready", False),
        ("receiver_ready", False),
        ("motor_target_valid", False),
        ("frame", 221),
        ("time_sec", 4.5),
        ("receiver", "blue.defender"),
        ("source", "red.finisher"),
        ("accepted_target_xy", (4.6, 2.1)),
        ("agreement_hash", "unbound"),
        ("motor_target_hash", "unbound"),
    ],
)
def test_invalid_current_agreement_rejected_without_consuming_id(key, value):
    ledger, previous, request, arguments = fixture()
    with pytest.raises(ValueError):
        ledger.bind(previous, request, **(arguments | {key: value}))
    assert ledger.bind(previous, request, **arguments)[0].created_sec == 4.4


def test_replay_of_old_or_new_handoff_cannot_repeatedly_extend_preparation():
    ledger, previous, request, arguments = fixture()
    following, _ = ledger.bind(previous, request, **arguments)
    for old in (previous, following):
        for repeated in (request, replace(request, request_id="sha256:" + "e" * 64)):
            with pytest.raises(ValueError):
                ledger.bind(old, repeated, **arguments)


@pytest.mark.parametrize("change", ["launched", "expired", "interrupted"])
def test_existing_flight_or_invalid_offer_never_renewed(change):
    ledger, previous, request, arguments = fixture()
    if change == "launched":
        previous = replace(previous, source_foot_contact_sec=2.0)
    elif change == "expired":
        previous = replace(previous, created_sec=1.0)
    else:
        previous = replace(previous, interrupted=True)
    with pytest.raises(ValueError):
        ledger.bind(previous, request, **arguments)


@pytest.mark.parametrize(
    "change",
    [
        dict(frame=True),
        dict(time_sec=float("nan")),
        dict(target_xy=(float("inf"), 0.0)),
        dict(request_id="bad"),
    ],
)
def test_request_validation(change):
    _, _, request, _ = fixture()
    with pytest.raises(ValueError):
        replace(request, **change)
