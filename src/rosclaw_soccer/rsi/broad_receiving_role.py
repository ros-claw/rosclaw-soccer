"""Fixed sixteen-state TRAIN teacher coverage probe, not another seed sweep."""

import copy
from typing import Any

STATES = tuple(10 * c + k for c in range(4) for k in (0, 4, 5, 9))
MODES = ("pursuit-v1", "projected-capture-v3")
SOURCE = "BROAD_RECEIVER16_PAIRED_ROLE_SOURCE_PENDING_OWN"
OWN = "BROAD_RECEIVER16_PAIRED_ROLE_ALL_OWN_MATCH_NOT_TRAINED"


def select_courses(courses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(courses) != 80
        or {(c["state_id"], c["task"]) for c in courses}
        != {(s, t) for s in range(40) for t in (0, 1)}
        or any(
            type(c["state_id"]) is not int
            or type(c["task"]) is not int
            or c["context"] != (1, 4, 20, 21)[c["state_id"] // 10]
            or c["assisted"] is not False
            for c in courses
        )
    ):
        raise ValueError("all80 unchanged unassisted consumed TRAIN courses required")
    by_state = {c["state_id"]: c for c in courses if c["task"] == 0}
    return [copy.deepcopy(by_state[s]) for s in STATES]


def config() -> dict[str, Any]:
    return dict(
        protocol="soccer.broad_receiver16_paired_roles.v1",
        states=list(STATES),
        goal_task=0,
        modes=list(MODES),
        independent_physical_initials=16,
        reference_contexts=[1, 4, 20, 21],
        maximum_worlds_per_side=32,
        maximum_native_steps_per_side=128000,
        maximum_neural_calls_per_side=25600,
        formal_controls=300,
        recovery_controls=100,
        A_fixed_generation=2,
        A_fixed_actor_steps=12,
        sampling_profile="contact-v1",
        sampling_include_recovery=True,
        one_private_noise_seed_per_initial=True,
        repeats_or_replay_are_new_initials=False,
        optimizer_updates=0,
        Fresh_verified=False,
        goal_complete=False,
    )


def sampling_seed(state: int) -> int:
    if type(state) is not int or state not in STATES:
        raise ValueError("predeclared broad physical state required")
    return 908023 + 137 * state
