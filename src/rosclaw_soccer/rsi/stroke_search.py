"""Finite two-condition teacher CEM math, never learned motor credit."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motor_option_credit import contact_skill_return

SEED = 55023
GENERATIONS = 2
CANDIDATES = 16
CONTEXTS = (4, 20)
CONTROL = 21
WORLDS = 72


def checked_latent(value: Any) -> np.ndarray:
    z = np.asarray(value, dtype=np.float64)
    if z.shape != (2, 6) or not np.isfinite(z).all() or (abs(z) > 1).any():
        raise ValueError("bounded twelve-dimensional coherent stroke required")
    return z.copy()


def sample_generation(
    rng: np.random.Generator, mean: np.ndarray, std: np.ndarray
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    mean = checked_latent(mean)
    std = np.asarray(std, dtype=np.float64)
    if std.shape != (2, 6) or not np.isfinite(std).all() or (std < 0.1).any() or (std > 0.4).any():
        raise ValueError("preregistered bounded CEM spread required")
    accepted, draws = [], []
    for _ in range(4096):
        z = rng.normal(mean, std)
        valid = bool(np.isfinite(z).all() and (abs(z) <= 1).all())
        draws.append(dict(latent=z.tolist(), accepted=valid))
        if valid:
            accepted.append(z)
        if len(accepted) == CANDIDATES:
            return accepted, draws
    raise ValueError("bounded CEM rejection budget exhausted; no clipping/replacement")


def pair_rank(rows: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    if len(rows) != 2 or [r["context"] for r in rows] != list(CONTEXTS):
        raise ValueError("both fixed physical conditions in declared order required")
    return (
        int(all(not r["result"]["body_failed"] for r in rows)),
        sum(r["healthy_quality"] for r in rows),
        sum(not r["result"]["body_failed"] and not r["result"]["forbidden"] for r in rows),
        sum(contact_skill_return(r["result"]) for r in rows),
    )


def update_elites(
    points: list[np.ndarray], ranks: list[tuple[int, int, int, float]]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if len(points) != 17 or len(ranks) != 17:
        raise ValueError("one retained best plus sixteen actual paired trials required")
    values = np.stack([checked_latent(p) for p in points])
    elite = sorted(range(17), key=lambda i: (ranks[i], -i), reverse=True)[:4]
    return values[elite].mean(0), np.clip(values[elite].std(0), 0.1, 0.4), elite


def verify_search_selection(report: dict[str, Any]) -> None:
    """Recreate all proposals and selection; physical grades need own replay too."""
    rows: list[dict[str, Any]] = report["records"]
    if len(rows) != WORLDS or len(report["generations"]) != GENERATIONS:
        raise ValueError("all declared search, repeat and control worlds required")
    cursor = 0

    def row(tag: str, context: int, z: np.ndarray) -> dict[str, Any]:
        nonlocal cursor
        r = rows[cursor]
        cursor += 1
        if (
            r["tag"] != tag
            or r["context"] != context
            or not np.array_equal(checked_latent(r["latent"]), z)
            or r["assisted"] is not True
            or r["PPO_actor_rows"] != 0
        ):
            raise ValueError("every ordered physical proposal must match the CEM declaration")
        return r

    def pair(tag: str, z: np.ndarray) -> dict[str, Any]:
        rr = [row(f"{tag}-context{i}", i, z) for i in CONTEXTS]
        return dict(tag=tag, latent=z, rank=pair_rank(rr), rows=rr)

    incumbent = pair("baseline", np.zeros((2, 6)))
    rng = np.random.default_rng(SEED)
    mean, std = np.zeros((2, 6)), np.full((2, 6), 0.4)
    for g in range(GENERATIONS):
        samples, draws = sample_generation(rng, mean, std)
        values = [incumbent] + [pair(f"g{g}-candidate{k}", z) for k, z in enumerate(samples)]
        ranks = [v["rank"] for v in values]
        updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
        best = max(range(len(values)), key=lambda k: (ranks[k], -k))
        expected = dict(
            generation=g,
            mean_before=mean.tolist(),
            std_before=std.tolist(),
            draws=draws,
            tags=[v["tag"] for v in values],
            ranks=[list(v) for v in ranks],
            elite=elite,
            mean_after=updated.tolist(),
            std_after=spread.tolist(),
            selected=values[best]["tag"],
        )
        if expected != report["generations"][g]:
            raise ValueError("own CEM resampling or elite/selection differs")
        incumbent = values[best]
        mean, std = updated, spread
    for i in CONTEXTS:
        for repeat in range(2):
            row(f"selected-context{i}-repeat{repeat}", i, incumbent["latent"])
    control = [
        row(f"control-context{CONTROL}-repeat{k}", CONTROL, incumbent["latent"]) for k in range(2)
    ]
    if (
        cursor != WORLDS
        or report["selected_tag"] != incumbent["tag"]
        or report["selected_latent"] != incumbent["latent"].tolist()
        or report["selected_rank"] != list(incumbent["rank"])
        or report["selected_teacher_successes"]
        != sum(r["healthy_quality"] for r in incumbent["rows"])
        or report["unsearched_control_quality"] != control[0]["healthy_quality"]
    ):
        raise ValueError("complete paired teacher selection accounting required")
