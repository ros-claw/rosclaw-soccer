"""Separate local pass Fresh protocol; NOT the consumed one-body Fresh56.

Sample only after all policies/eligibility freeze. New initial ball positions
have a nonzero forward component; consumed TRAIN offsets were lateral only.
No physics, teacher policy, fitting or admission is performed by this module.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.pass_transfer_curriculum import initial_ball_offset
from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

PROTOCOL = "soccer.pass_near_2D48_sealed.v1"
CONTEXTS = (1, 4, 20, 21)


def pass_fresh_courses(
    templates: list[dict[str, Any]], directions: dict[int, np.ndarray], seed: int
) -> list[dict[str, Any]]:
    """48 distinct states /96 goal tasks, balanced contexts, quadrants and sides."""
    if (
        type(seed) is not int
        or not 0 <= seed < 2**64
        or len(templates) != 80
        or set(directions) != set(CONTEXTS)
        or any(
            not isinstance(f, np.ndarray)
            or f.shape != (2,)
            or not np.isfinite(f).all()
            or abs(np.linalg.norm(f) - 1) > 1e-10
            for f in directions.values()
        )
    ):
        raise ValueError(
            "whole consumed task templates, unit directions and explicit uint64 required"
        )
    if {(c["state_id"], c["task"]) for c in templates} != {
        (i, j) for i in range(40) for j in (0, 1)
    }:
        raise ValueError("all original80 matched TRAIN template identities required")
    if any(c["assisted"] is not False for c in templates):
        raise ValueError("unassisted physical task templates required")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for context in CONTEXTS:
        forward = directions[context]
        lateral = np.array([-forward[1], forward[0]])
        choices = {}
        for task in (0, 1):
            group = [c for c in templates if c["context"] == context and c["task"] == task]
            if len(group) != 10 or len({tuple(c["target"]) for c in group}) != 1:
                raise ValueError("unchanged per-context goal contracts required")
            choices[task] = group[0]
        for quadrant, (sf, sl) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            for repeat in range(3):
                df = sf * float(rng.uniform(0.005, 0.025))
                dl = sl * float(rng.uniform(0.005, 0.04))
                offset = initial_ball_offset(df * forward + dl * lateral)
                assert offset is not None
                # Analytic disjointness: TRAIN had zero forward component.
                if abs(float(offset @ forward)) < 0.004999999:
                    raise ValueError("new Fresh state must be outside the consumed lateral line")
                state_id = len(rows) // 2
                side = -0.55 if (3 * quadrant + repeat) % 2 == 0 else 0.55
                for task in (0, 1):
                    source = choices[task]
                    rows.append(
                        dict(
                            state_id=state_id,
                            context=context,
                            task=task,
                            side=side,
                            foot=source["foot"],
                            center=source["center"],
                            target=list(source["target"]),
                            latent=[[0.0] * 6 for _ in range(2)],
                            assisted=False,
                            source_teacher_qualified=False,
                            ball_planar_offset_world=offset.tolist(),
                            bank="FRESH",
                            not_fresh=False,
                            fresh_protocol=PROTOCOL,
                        )
                    )
    identities = {(c["context"], c["side"], tuple(c["ball_planar_offset_world"])) for c in rows}
    if len(identities) != 48:
        raise ValueError("all48 genuinely distinct initial states required; no replacement")
    return rows


def pass_fresh_screen(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Paired screen, clustered by physical initial state, never adjacent frames.

    Local pass screen only. Existing consumed-TRAIN regressions and independent
    critical old17 retention remain separate mandatory reported evidence.
    """

    def index(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
        if len(rows) != 96:
            raise ValueError("ALL96 goal trials required, not only successful cases")
        out = {}
        for row in rows:
            c = row["course"]
            key = (c["state_id"], c["task"])
            if (
                type(c["state_id"]) is not int
                or not 0 <= c["state_id"] < 48
                or type(c["task"]) is not int
                or c["task"] not in (0, 1)
                or type(c["context"]) is not int
                or c["context"] not in CONTEXTS
                or c["context"] != CONTEXTS[c["state_id"] // 12]
                or c["bank"] != "FRESH"
                or c["not_fresh"] is not False
                or c["fresh_protocol"] != PROTOCOL
                or c["assisted"] is not False
                or key in out
            ):
                raise ValueError("unique whole unassisted sealed task identities required")
            out[key] = row
        if set(out) != {(i, j) for i in range(48) for j in (0, 1)}:
            raise ValueError("complete balanced48x2 Fresh required")
        for state in range(48):
            pair = [
                {k: v for k, v in out[state, t]["course"].items() if k not in ("task", "target")}
                for t in (0, 1)
            ]
            if pair[0] != pair[1]:
                raise ValueError("two goals cannot masquerade as one physical initial state")
        return out

    old, new = index(baseline), index(candidate)
    comparison = []
    for key in sorted(old):
        a, b = old[key], new[key]
        if a["course"] != b["course"]:
            raise ValueError("candidate must execute identical whole sealed inputs")
        x, y = a["outcome"], b["outcome"]
        before, after = controlled_pass_label(x), controlled_pass_label(y)
        comparison.append(
            dict(
                state_id=key[0],
                task=key[1],
                context=a["course"]["context"],
                baseline=before,
                candidate=after,
                rescue=after and not before,
                regression=before and not after,
                new_body_failure=x["whole_body_healthy"] and not y["whole_body_healthy"],
                new_nonfoot=x["passer_clean_foot"] and not y["passer_clean_foot"],
                new_B_excluded=not x["B_excluded_contact"] and y["B_excluded_contact"],
            )
        )
    context_net = {}
    clusters = []
    for ctx in CONTEXTS:
        group = [r for r in comparison if r["context"] == ctx]
        states = sorted({r["state_id"] for r in group})
        if len(group) != 24 or len(states) != 12:
            raise ValueError("ALL4 balanced motion contexts required")
        # Both goals of a state must belong to the same context.
        if any(sum(r["state_id"] == i for r in group) != 2 for i in states):
            raise ValueError("two paired goal tasks per same physical state required")
        values = np.array(
            [
                sum(int(r["candidate"]) - int(r["baseline"]) for r in group if r["state_id"] == i)
                for i in states
            ],
            dtype=np.float64,
        )
        context_net[str(ctx)] = int(values.sum())
        clusters.append(values)
    rng = np.random.default_rng(2026102348)
    draws = sum(values[rng.integers(0, 12, size=(4096, 12))].sum(1) for values in clusters) / 96
    low, high = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    rescues = sum(r["rescue"] for r in comparison)
    regressions = sum(r["regression"] for r in comparison)
    unsafe = {
        k: sum(r[k] for r in comparison)
        for k in ("new_body_failure", "new_nonfoot", "new_B_excluded")
    }
    passed = (
        rescues - regressions >= 8
        and min(context_net.values()) >= 0
        and low > 0
        and not any(unsafe.values())
    )
    return dict(
        protocol=PROTOCOL,
        baseline_controlled=sum(r["baseline"] for r in comparison),
        candidate_controlled=sum(r["candidate"] for r in comparison),
        rescues=rescues,
        regressions=regressions,
        context_net=context_net,
        **unsafe,
        paired_state_stratified_bootstrap95=[low, high],
        independent_initial_states=48,
        goal_trials=96,
        bootstrap_replicates=4096,
        local_pass_screen_passed=passed,
        screen_rule=(
            "net>=8; every_context_net>=0; state_bootstrap95_lower>0; "
            "zero_new_falls/nonfoot/B_excluded"
        ),
        zero_global_forgetting_proven=False,
        critical_old_retention_included_here=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
