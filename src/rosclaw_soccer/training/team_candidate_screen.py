"""Conservative paired development screen, never a policy-promotion gate."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw_soccer.training.team_episode_cut import TeamEpisodeCut


@dataclass(frozen=True)
class TeamScreenArm:
    """Evidence for one arm: complete sampled cut and full-horizon latch."""

    cut: TeamEpisodeCut
    first_unsafe_substep: int | None
    observed_substeps: int
    expected_substeps: int
    controlled_windows: int

    def __post_init__(self) -> None:
        if not isinstance(self.cut, TeamEpisodeCut) or not self.cut.sampled_only:
            raise ValueError("sampled team cut required")
        for name in ("observed_substeps", "expected_substeps", "controlled_windows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.expected_substeps == 0 or self.observed_substeps != self.expected_substeps:
            raise ValueError("full-horizon substep evidence required")
        unsafe = self.first_unsafe_substep
        if unsafe is not None and (
            type(unsafe) is not int or not 1 <= unsafe <= self.expected_substeps
        ):
            raise ValueError("invalid substep latch")
        for frame in (
            self.cut.first_safety_frame,
            self.cut.first_exit_frame,
            self.cut.first_terminal_frame,
        ):
            if frame is not None and (type(frame) is not int or frame < 0):
                raise ValueError("invalid sampled frame")
        expected = min(
            (f for f in (self.cut.first_safety_frame, self.cut.first_exit_frame) if f is not None),
            default=None,
        )
        if self.cut.first_terminal_frame != expected:
            raise ValueError("inconsistent sampled cut")


@dataclass(frozen=True)
class TeamCandidateScreen:
    sampled_safety_regression: bool
    substep_latch_regression: bool
    controlled_window_gain: int
    eligible_for_fresh_validation: bool
    promotion_authorized: bool = False


def _before_exit(cut: TeamEpisodeCut) -> int | None:
    safety, exit_frame = cut.first_safety_frame, cut.first_exit_frame
    return safety if safety is not None and (exit_frame is None or safety <= exit_frame) else None


def screen_team_candidate(parent: TeamScreenArm, candidate: TeamScreenArm) -> TeamCandidateScreen:
    """Screen matched arms; improvement is needed, but never proves generalization.

    Sampled windows may overlap and this screen does not establish statistical
    significance. The caller must verify matching scenarios, evidence hashes,
    replay and independent evaluation separately.
    """
    if not isinstance(parent, TeamScreenArm) or not isinstance(candidate, TeamScreenArm):
        raise ValueError("validated paired arms required")
    if parent.expected_substeps != candidate.expected_substeps:
        raise ValueError("matched horizons required")
    ps, cs = _before_exit(parent.cut), _before_exit(candidate.cut)
    sampled_regression = cs is not None and (ps is None or cs < ps)
    pl, cl = parent.first_unsafe_substep, candidate.first_unsafe_substep
    latch_regression = cl is not None and (pl is None or cl < pl)
    gain = candidate.controlled_windows - parent.controlled_windows
    return TeamCandidateScreen(
        sampled_regression,
        latch_regression,
        gain,
        not sampled_regression and not latch_regression and gain > 0,
    )
