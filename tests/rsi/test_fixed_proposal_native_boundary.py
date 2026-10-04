"""Reject invalid shared fixed means before allocating physical output."""

import pytest

from rosclaw_soccer.rsi.proposal_episode_decoder_factory import ProposalEpisodeDecoderFactory
from scripts.rsi_mujoco_motor_transfer import main


@pytest.mark.parametrize(
    "options,foreign,sampling,extended",
    [
        (
            ["--step-model", "absent.json", "--proposal-decoder", "owned_snapshot"],
            True,
            False,
            False,
        ),
        (["--proposal-decoder", "owned_snapshot"], False, False, False),
        (["--step-model", "absent.json"], False, False, False),
        (
            ["--step-model", "absent.json", "--proposal-decoder", "bounded_snapshot"],
            False,
            False,
            False,
        ),
        (
            [
                "--step-model",
                "absent.json",
                "--proposal-decoder",
                "owned_snapshot",
                "--cached-proposal-envelope",
            ],
            False,
            False,
            False,
        ),
        (
            [
                "--step-model",
                "absent.json",
                "--proposal-decoder",
                "owned_snapshot",
                "--body-response-bundle",
                "absent.json",
            ],
            False,
            False,
            False,
        ),
        (
            ["--step-model", "absent.json", "--proposal-decoder", "owned_snapshot"],
            False,
            True,
            False,
        ),
        (
            ["--step-model", "absent.json", "--proposal-decoder", "owned_snapshot"],
            False,
            False,
            True,
        ),
        (
            [
                "--step-model",
                "absent.json",
                "--proposal-decoder",
                "owned_snapshot",
                "--foundation-only",
            ],
            False,
            False,
            False,
        ),
    ],
)
def test_invalid_fixed_factory_declarations_rejected(
    tmp_path, options, foreign, sampling, extended
):
    output = tmp_path / "must-not-exist"
    factory = object() if foreign else object.__new__(ProposalEpisodeDecoderFactory)
    args = [
        "--scene",
        str(tmp_path / "absent.xml"),
        "--model-root",
        str(tmp_path / "absent-foundation"),
        "--late-swing-policy",
        str(tmp_path / "absent-policy.json"),
        "--output-root",
        str(output),
        "--seed",
        "0",
        "--lane",
        "0",
        *options,
    ]
    with pytest.raises(SystemExit) as exc:
        main(
            args,
            proposal_factory=factory,
            sampling_factory=object() if sampling else None,
            extended_factory=object() if extended else None,
        )
    assert exc.value.code == 2
    assert not output.exists()
