"""Exclusive, simulation-only arbitration for native and learned motor owners.

This state machine does not decide when a skill should be admitted, call a
policy, or command a robot. The caller must make the transition before either
backend advances and must not apply an output that lacks the matching grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MotorOwner(Enum):
    """The only possible motor owners for one simulated player."""

    NONE = "none"
    NATIVE = "native"
    LEARNED = "learned"
    FAULT = "fault"


@dataclass(frozen=True)
class MotorHandoffDecision:
    frame: int
    owner: MotorOwner
    claim_deferred: bool
    activation_ceiling: str = "SIM_ONLY"


class ExclusiveMotorHandoff:
    """Explicit per-player lifetime; a fault is latched until a new instance.

    A learned owner retains exclusivity until normal release. Native activity
    discovered while that owner is live is a contract violation, not a reason
    to silently choose one target. A deferred claimant has not started and
    must not advance its recurrent state. All events apply to one frame only.
    """

    def __init__(self) -> None:
        self._next_frame = 0
        self._learned_owned = False
        self._faulted = False

    @property
    def learned_owned(self) -> bool:
        return self._learned_owned

    def latch_fault(self) -> None:
        """Stop future ownership after a provider failure in the current frame."""
        self._faulted = True
        self._learned_owned = False

    def advance(
        self,
        *,
        frame: int,
        native_active: bool,
        learned_claim: bool,
        learned_release: bool,
        learned_fault: bool,
    ) -> MotorHandoffDecision:
        if (
            type(frame) is not int
            or frame != self._next_frame
            or any(
                type(v) is not bool
                for v in (native_active, learned_claim, learned_release, learned_fault)
            )
        ):
            raise ValueError("consecutive frame and explicit boolean events required")
        if self._faulted:
            raise ValueError("motor handoff remains faulted")
        if (learned_release and not self._learned_owned) or (learned_release and learned_claim):
            raise ValueError("only the current learned owner may release without a new claim")
        if native_active and self._learned_owned:
            self._faulted = True
            raise ValueError("native motor overlaps active learned owner")
        self._next_frame += 1
        if learned_fault:
            self._faulted = True
            self._learned_owned = False
            return MotorHandoffDecision(frame, MotorOwner.FAULT, False)
        if learned_release:
            self._learned_owned = False
            # A release takes effect after this frame's ownership decision;
            # native admission can resume on the following frame only.
            return MotorHandoffDecision(frame, MotorOwner.NONE, False)
        if self._learned_owned:
            return MotorHandoffDecision(frame, MotorOwner.LEARNED, False)
        if native_active:
            return MotorHandoffDecision(frame, MotorOwner.NATIVE, learned_claim)
        if learned_claim:
            self._learned_owned = True
            return MotorHandoffDecision(frame, MotorOwner.LEARNED, False)
        return MotorHandoffDecision(frame, MotorOwner.NONE, False)
