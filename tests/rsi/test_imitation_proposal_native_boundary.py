import pytest

from rosclaw_soccer.rsi.imitation_proposal_episode_factory import ImitationProposalEpisodeFactory
from scripts.rsi_mujoco_motor_transfer import main


@pytest.mark.parametrize(
    "options,foreign,other",
    [
        (["--step-model", "missing.json"], True, None),
        ([], False, None),
        (["--step-model", "missing.json", "--proposal-decoder", "owned_snapshot"], False, None),
        (["--step-model", "missing.json", "--cached-proposal-envelope"], False, None),
        (["--step-model", "missing.json", "--body-response-bundle", "missing.json"], False, None),
        (["--step-model", "missing.json", "--foundation-only"], False, None),
        (["--step-model", "missing.json"], False, "sampling_factory"),
        (["--step-model", "missing.json"], False, "proposal_factory"),
        (["--step-model", "missing.json"], False, "extended_factory"),
    ],
)
def test_incomplete_or_mixed_imitation_paths_rejected_without_output(
    tmp_path, options, foreign, other
):
    factory = object() if foreign else object.__new__(ImitationProposalEpisodeFactory)
    output = tmp_path / "must-not-exist"
    args = [
        "--scene",
        str(tmp_path / "absent.xml"),
        "--model-root",
        str(tmp_path / "absent"),
        "--late-swing-policy",
        str(tmp_path / "absent.json"),
        "--seed",
        "0",
        "--lane",
        "0",
        "--output-root",
        str(output),
        *options,
    ]
    with pytest.raises(SystemExit) as exc:
        main(args, imitation_factory=factory, **({other: object()} if other else {}))
    assert exc.value.code == 2
    assert not output.exists()
