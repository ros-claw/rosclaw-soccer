"""Reject mixed or incomplete extended paths before any physical allocation."""

import pytest

from rosclaw_soccer.rsi.extended_proposal_episode_factory import ExtendedProposalEpisodeFactory
from scripts.rsi_mujoco_motor_transfer import main


@pytest.mark.parametrize(
    "options,foreign_factory,other_sampling_factory",
    [
        (["--step-model", "missing.json"], True, False),
        ([], False, False),
        (["--step-model", "missing.json", "--proposal-decoder", "bounded_snapshot"], False, False),
        (["--step-model", "missing.json", "--cached-proposal-envelope"], False, False),
        (["--step-model", "missing.json", "--body-response-bundle", "missing.json"], False, False),
        (["--step-model", "missing.json"], False, True),
        (["--step-model", "missing.json", "--foundation-only"], False, False),
    ],
)
def test_incomplete_or_mixed_factory_paths_rejected_without_output(
    tmp_path, options, foreign_factory, other_sampling_factory
):
    # Uninitialized exact-type sentinel is safe only because every declaration
    # must be rejected before reading a model or calling factory methods.
    factory = object() if foreign_factory else object.__new__(ExtendedProposalEpisodeFactory)
    output = tmp_path / "must-not-exist"
    args = [
        "--scene",
        str(tmp_path / "absent-scene.xml"),
        "--model-root",
        str(tmp_path / "absent-foundation"),
        "--late-swing-policy",
        str(tmp_path / "absent-policy.json"),
        "--seed",
        "0",
        "--lane",
        "0",
        "--output-root",
        str(output),
        *options,
    ]
    with pytest.raises(SystemExit) as exc:
        main(
            args,
            extended_factory=factory,
            sampling_factory=object() if other_sampling_factory else None,
        )
    assert exc.value.code == 2
    assert not output.exists()
