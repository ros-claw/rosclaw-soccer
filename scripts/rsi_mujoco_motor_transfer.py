"""Actual CPU MuJoCo closed-loop transfer probe, not an Isaac evidence replay.

No robot transport or vendor SDK is imported. Asset differences remain explicit;
this diagnostic cannot authorize promotion or physical execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.sim.current_kinematic_observation import (
    observation_contract as make_observation_contract,
)


def main(
    argv: list[str] | None = None,
    *,
    sampling_factory: Any = None,
    extended_factory: Any = None,
    proposal_factory: Any = None,
    imitation_factory: Any = None,
    recurrent_factory: Any = None,
    recurrent_sampling_factory: Any = None,
    recurrent_clipped_factory: Any = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "model-root", "late-swing-policy", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--neural-model", type=Path)
    group.add_argument("--motor-policy", type=Path)
    group.add_argument("--step-model", type=Path)
    parser.add_argument("--body-response-bundle", type=Path)
    parser.add_argument("--cached-proposal-envelope", action="store_true")
    parser.add_argument("--consumed-bank", type=Path)
    parser.add_argument("--foundation-only", action="store_true")
    parser.add_argument("--compressed-report", action="store_true")
    parser.add_argument(
        "--record-foundation-observation",
        action="store_true",
        help="Record actual 994-input/29-action frozen foundation calls; not a trained policy",
    )
    parser.add_argument(
        "--shared-evidence",
        action="store_true",
        help="Opt-in lossless shared model/world storage for new compressed evidence only",
    )
    parser.add_argument(
        "--proposal-decoder",
        choices=("reference", "owned_snapshot", "bounded_snapshot"),
        default="reference",
        help="Opt-in owned numerical compilation; only sealed proposal models are accepted",
    )
    parser.add_argument(
        "--root-velocity-reference",
        choices=("body-com", "body-origin"),
        default="body-com",
        help="Explicit diagnostic input convention; default preserves historical COM observations",
    )
    parser.add_argument(
        "--observation-snapshot", choices=("cached", "current-kinematic"), default="cached"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lane", type=int, required=True, choices=range(16))
    args = parser.parse_args(argv)
    if recurrent_clipped_factory is not None:
        from rosclaw_soccer.rsi.recurrent_clipped_episode_factory import (
            RecurrentClippedEpisodeFactory,
        )

        if (
            type(recurrent_clipped_factory) is not RecurrentClippedEpisodeFactory
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                    recurrent_sampling_factory,
                )
            )
            or args.step_model is None
            or args.proposal_decoder != "reference"
            or args.cached_proposal_envelope
            or args.body_response_bundle
            or args.foundation_only
        ):
            parser.error("clipped factory requires only its explicit fixed SIM candidate")
    if recurrent_sampling_factory is not None:
        from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import (
            RecurrentSamplingEpisodeFactory,
        )

        if (
            type(recurrent_sampling_factory) is not RecurrentSamplingEpisodeFactory
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                    recurrent_factory,
                )
            )
            or args.step_model is None
            or args.proposal_decoder != "reference"
            or args.cached_proposal_envelope
            or args.body_response_bundle
        ):
            parser.error(
                "recurrent sampling factory requires only its explicit fixed SIM candidate"
            )
    if recurrent_factory is not None:
        from rosclaw_soccer.rsi.recurrent_success_episode_factory import (
            RecurrentSuccessEpisodeFactory,
        )

        if (
            type(recurrent_factory) is not RecurrentSuccessEpisodeFactory
            or any(
                factory is not None
                for factory in (
                    sampling_factory,
                    extended_factory,
                    proposal_factory,
                    imitation_factory,
                )
            )
            or args.step_model is None
            or args.proposal_decoder != "reference"
            or args.cached_proposal_envelope
            or args.body_response_bundle
        ):
            parser.error("sequence factory requires only its explicit fixed SIM candidate")
    if imitation_factory is not None:
        from rosclaw_soccer.rsi.imitation_proposal_episode_factory import (
            ImitationProposalEpisodeFactory,
        )

        if (
            type(imitation_factory) is not ImitationProposalEpisodeFactory
            or any(v is not None for v in (sampling_factory, extended_factory, proposal_factory))
            or args.step_model is None
            or args.proposal_decoder != "reference"
            or args.cached_proposal_envelope
            or args.body_response_bundle
        ):
            parser.error("imitation factory requires only its explicit bounded SIM candidate")
    if proposal_factory is not None:
        from rosclaw_soccer.rsi.proposal_episode_decoder_factory import (
            ProposalEpisodeDecoderFactory,
        )

        if (
            type(proposal_factory) is not ProposalEpisodeDecoderFactory
            or sampling_factory is not None
            or extended_factory is not None
            or args.step_model is None
            or args.proposal_decoder != "owned_snapshot"
            or args.cached_proposal_envelope
            or args.body_response_bundle
        ):
            parser.error("fixed proposal factory requires only its explicit owned snapshot")
    if extended_factory is not None:
        from rosclaw_soccer.rsi.extended_proposal_episode_factory import (
            ExtendedProposalEpisodeFactory,
        )

        if (
            type(extended_factory) is not ExtendedProposalEpisodeFactory
            or sampling_factory is not None
            or args.step_model is None
            or args.proposal_decoder != "reference"
            or args.cached_proposal_envelope
            or args.body_response_bundle
        ):
            parser.error("extended factory accepts only its explicit fixed SIM proposal")
    if args.cached_proposal_envelope:
        from rosclaw_soccer.rsi.owned_proposal_sampling_factory import (
            OwnedProposalSamplingEpisodeFactory,
        )

        if (
            type(sampling_factory) is not OwnedProposalSamplingEpisodeFactory
            or not args.step_model
            or args.proposal_decoder != "reference"
            or args.body_response_bundle
        ):
            parser.error(
                "cached envelope requires only the exact private proposal sampling factory"
            )
    make_observation_contract(args.root_velocity_reference, args.observation_snapshot)
    if args.foundation_only and (args.neural_model or args.motor_policy or args.step_model):
        parser.error("foundation-only cannot include a motor learning model")
    if args.proposal_decoder != "reference" and args.step_model is None:
        parser.error("owned proposal decoder requires an explicit sealed step-model")
    if args.body_response_bundle and (
        not args.step_model or not args.record_foundation_observation
    ):
        parser.error(
            "body response experiment requires a proposal parent and actual foundation capture"
        )
    if args.shared_evidence and not args.compressed_report:
        parser.error("shared evidence requires compressed-report")
    if sampling_factory is not None and (
        args.step_model is None or args.proposal_decoder != "reference"
    ):
        parser.error("shared smooth factory requires only an explicit reference sampling model")
    # This first diagnostic only accepts already consumed courses.
    consumed = {
        (20261177, 0),
        (20261227, 4),
        (20261282, 0),
        (20261360, 0),
        (20261378, 0),
        (20261440, 2),
        (20261446, 0),
        (20261095, 2),
        (20261097, 2),
        (20261146, 4),
        (20261148, 2),
        (20260975, 4),
    }
    consumed_bank_hash = None
    if args.consumed_bank is not None:
        # Artifact inspection must not import historical experiment scripts.
        bank = json.loads(args.consumed_bank.read_text())
        if (
            bank.get("report_hash")
            != hash_json({k: v for k, v in bank.items() if k != "report_hash"})
            or bank.get("schema") != "soccer.rsi.progressive_motor_learning_bank.v312"
            or bank.get("partition") != "TRAIN_CONSUMED"
            or bank.get("fresh_holdout_open_authorized") is not False
        ):
            parser.error("sealed consumed physical bank required for additional transfer cases")
        consumed |= {(r["seed"], r["lane"]) for r in bank["courses"]}
        consumed_bank_hash = bank["report_hash"]
    if (args.seed, args.lane) not in consumed:
        parser.error("transfer diagnostic cannot open a fresh course")
    from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode

    run_native_first_touch_episode(
        args,
        course=sample_training_courses(args.seed, 16)[args.lane],
        partition="CONSUMED_TRANSFER_DIAGNOSTIC",
        entry_source_path=Path(__file__),
        consumed_bank_hash=consumed_bank_hash,
        sampling_factory=sampling_factory,
        extended_factory=extended_factory,
        proposal_factory=proposal_factory,
        imitation_factory=imitation_factory,
        recurrent_factory=recurrent_factory,
        recurrent_sampling_factory=recurrent_sampling_factory,
        recurrent_clipped_factory=recurrent_clipped_factory,
    )


if __name__ == "__main__":
    main()
