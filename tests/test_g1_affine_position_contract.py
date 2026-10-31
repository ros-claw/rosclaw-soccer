from dataclasses import replace

import pytest

from rosclaw_soccer.providers.g1.affine_position_contract import G1AffinePositionContract
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def contract() -> G1AffinePositionContract:
    return G1AffinePositionContract(
        G1_DDS_JOINT_NAMES,
        tuple(i / 100 for i in range(29)),
        tuple((i + 1) / 100 for i in range(29)),
        (20.0,) * 29,
        (2.0,) * 29,
    )


def test_affine_map_and_explicit_bias() -> None:
    spec = contract()
    raw = tuple((i - 14) / 10 for i in range(29))
    bias = tuple(i / 1000 for i in range(29))
    proposal = spec.propose(raw, encoder_bias_rad=bias)
    assert proposal.target_rad == pytest.approx(
        tuple(spec.offset_rad[i] + spec.scale_rad[i] * raw[i] - bias[i] for i in range(29))
    )
    assert proposal.kp == spec.kp and proposal.kd == spec.kd
    assert spec.contract_hash == contract().contract_hash
    assert spec.contract_hash != replace(spec, kd=(3.0,) * 29).contract_hash


@pytest.mark.parametrize(
    "changes",
    [
        {"joint_names": tuple(reversed(G1_DDS_JOINT_NAMES))},
        {"joint_names": G1_DDS_JOINT_NAMES[:-1] + (G1_DDS_JOINT_NAMES[0],)},
        {"offset_rad": (float("nan"),) * 29},
        {"scale_rad": (0.0,) * 29},
        {"kp": (-1.0,) * 29},
        {"kd": (31.0,) * 29},
        {"activation_ceiling": "REAL"},
    ],
)
def test_bad_contract_fails_closed(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(contract(), **changes)


@pytest.mark.parametrize("bad", [(float("nan"),) * 29, (0.0,) * 28, [0.0] * 29, (101.0,) * 29])
def test_bad_actions_fail_closed(bad: object) -> None:
    with pytest.raises(ValueError):
        contract().propose(bad, encoder_bias_rad=(0.0,) * 29)  # type: ignore[arg-type]


def test_out_of_envelope_targets_are_rejected_not_clipped() -> None:
    with pytest.raises(ValueError):
        contract().propose((100.0,) * 29, encoder_bias_rad=(0.0,) * 29)
