"""Fixed public source split declared before any new policy optimization."""

import re
from dataclasses import dataclass
from typing import Any

TRAIN_MOTION_GLOB = "soccer-standard-00[12345689]_*.npz"
TRAIN_MOTION_NAMES = (
    "soccer-standard-001_right.npz",
    "soccer-standard-002_left.npz",
    "soccer-standard-003_left.npz",
    "soccer-standard-004_right.npz",
    "soccer-standard-005_right.npz",
    "soccer-standard-006_right.npz",
    "soccer-standard-008_left.npz",
    "soccer-standard-009_right.npz",
)
DEVELOPMENT_MOTION_NAMES = ("soccer-standard-007_left.npz", "soccer-standard-010_right.npz")


def motion_checkpoint_filename(report: dict[str, Any]) -> str:
    name = report.get("checkpoint_file")
    if name is None:
        iterations = report.get("iterations")
        if type(iterations) is not int or not 1 <= iterations <= 100000:
            raise ValueError("explicit bounded checkpoint iteration required")
        name = f"learner-update{iterations}.pt"
    if type(name) is not str or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}\.pt", name) is None:
        raise ValueError("checkpoint must be an ordinary local artifact basename")
    return name


def motion_names_for_pattern(pattern: str) -> tuple[str, ...]:
    if type(pattern) is not str:
        raise ValueError("declared source pattern required")
    if pattern == TRAIN_MOTION_GLOB:
        return TRAIN_MOTION_NAMES
    if pattern in DEVELOPMENT_MOTION_NAMES:
        return (pattern,)
    raise ValueError("fixed eight-source training or one explicit development source required")


@dataclass(frozen=True)
class MotionLearningBudget:
    """Bounded single-GPU motion-prior research, never a football certificate."""

    mode: str = "smoke"
    num_envs: int = 32
    iterations: int = 2
    seed: int = 501

    def __post_init__(self) -> None:
        if self.mode not in ("smoke", "course") or any(
            type(v) is not int for v in (self.num_envs, self.iterations, self.seed)
        ):
            raise ValueError("explicit mode and integer learning budget required")
        if not 0 <= self.seed < 2**31:
            raise ValueError("bounded integer seed required")
        if self.mode == "smoke":
            if (self.num_envs, self.iterations, self.seed) != (32, 2, 501):
                raise ValueError("smoke is the frozen 32-env two-update seed501 protocol")
        elif not (32 <= self.num_envs <= 512 and 3 <= self.iterations <= 200):
            raise ValueError("course requires 32..512 environments and 3..200 iterations")

    @property
    def training_control_steps(self) -> int:
        return self.iterations * 24

    @property
    def gpu_world_substeps(self) -> int:
        return (8 + self.training_control_steps) * 10 * self.num_envs


def require_completed_motion_smoke(report: dict[str, Any]) -> None:
    """Only a real bounded learner smoke can unlock a longer course.

    The caller still must authenticate the receipt hash, original unit exit,
    dependency source and data pins. No optimizer or skill claim follows from
    calling this structural check alone.
    """
    if report.get("status") != "MJLAB_MOTION_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED":
        raise ValueError("actual motion GPU PPO smoke receipt required")
    for key, expected in (
        ("gpu_step_calls", 560),
        ("gpu_world_substeps", 17920),
        ("policy_control_samples", 1536),
        ("new_cpu_native_steps", 0),
    ):
        if type(report.get(key)) is not int or report[key] != expected:
            raise ValueError("complete original smoke physics/sample accounting required")
    steps = report.get("new_optimizer_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError("completed optimizer updates required")
    counts = report.get("optimizer_counts")
    if (
        not isinstance(counts, list)
        or not counts
        or any(
            not isinstance(c, dict)
            or type(c.get("completed")) is not int
            or type(c.get("attempted")) is not int
            or c["completed"] <= 0
            or c["completed"] != c["attempted"]
            for c in counts
        )
        or sum(c["completed"] for c in counts) != steps
    ):
        raise ValueError("actual optimizer step accounting required")
    for part in ("changed_actor_weights", "changed_critic_weights"):
        names = report.get(part)
        if (
            not isinstance(names, list)
            or not names
            or any(type(n) is not str or not n.endswith((".weight", ".bias")) for n in names)
        ):
            raise ValueError("both neural MLPs must actually change")
    if report.get("weights_only_checkpoint_reload") is not True or any(
        report.get(k) is not False
        for k in (
            "football_skill_proven",
            "goal_complete",
            "hardware_authorized",
            "promotion_authorized",
        )
    ):
        raise ValueError("safe checkpoint and explicit non-promotion scope required")


def require_completed_stable_motion_smoke(report: dict[str, Any]) -> None:
    """Different protocol: never inherit a from-scratch smoke certificate."""
    if report.get("status") != "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED":
        raise ValueError("actual stable-prior rehearsal smoke required")
    # Reuse accounting and non-promotion checks, not the old policy identity.
    accounting = dict(report, status="MJLAB_MOTION_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED")
    require_completed_motion_smoke(accounting)
    if type(report.get("native_rehearsal")) is not dict:
        raise ValueError("complete native rehearsal protocol required")
    if (
        report.get("stable_prior_rehearsal") is not True
        or report.get("new_optimizer_steps") != 42
        or report.get("iterations") != 2
        or report.get("num_envs") != 32
        or report.get("seed") != 501
        or report.get("native_rehearsal")
        != dict(
            bank_hash=report.get("native_rehearsal", {}).get("bank_hash"),
            attempted_updates=2,
            updates=2,
            batch_size=256,
            rows=512,
            weight=1.0,
        )
    ):
        raise ValueError("complete two-iteration stable prior and rehearsal protocol required")
    bank_hash = report["native_rehearsal"]["bank_hash"]
    if type(bank_hash) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", bank_hash) is None:
        raise ValueError("immutable native rehearsal bank identity required")
