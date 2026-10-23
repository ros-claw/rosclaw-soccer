"""Synthetic CEM transcripts only, not physical teacher qualification."""

from copy import deepcopy

import numpy as np
import pytest

from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites
from scripts.rsi_self_teacher_search_common import calibrations, compare_common, rank, verify_search


def outcome(received=True, error=0.4):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_legal_functional_receive_before_deadline=received,
        B_excluded_contact=False,
        combined_precision_and_functional_receive=False,
        incoming_arrival_speed_mps=3.2,
        incoming=dict(
            A_first_foot_strike_substep=100,
            B_first_ball_contact_substep=200,
            incoming_target_error_3d_before_B_contact_m=error,
        ),
    )


def transcript():
    state = 20
    cal = dict(records=[])
    report = dict(state_id=state, records=[], generations=[], selections=[])
    for task in (0, 1):
        c = dict(state_id=state, task=task, context=20, assisted=True, target=[0, task * 0.2, 0.11])
        for prefix in (0, 80):
            cal["records"].append(
                dict(
                    course=c | dict(latent=[[0.1 if prefix else 0.0] * 6 for _ in range(2)]),
                    student_prefix_controls=prefix,
                    outcome=outcome(bool(prefix)),
                )
            )
        old = calibrations(cal, state, task)
        values = [
            dict(
                tag=f"cal-t{task}-p{r['student_prefix_controls']}",
                row=r,
                latent=np.array(r["course"]["latent"], np.float64),
            )
            for r in old
        ]
        incumbent = max(enumerate(values), key=lambda kv: (rank(kv[1]["row"]), -kv[0]))[1]
        mean = incumbent["latent"].copy()
        std = np.full((2, 6), 0.4)
        rng = np.random.default_rng(146023 + state * 101 + task * 7)
        course = old[1]["course"]

        def record(z, tag, o=None, course=course, task=task):
            row = dict(
                index=len(report["records"]),
                course=course | dict(latent=z.tolist()),
                tag=tag,
                search_task=task,
                prefix_identity_fields=38,
                neural_actor_optimizer_updates=0,
                outcome=outcome(False) if o is None else deepcopy(o),
            )
            report["records"].append(row)
            return dict(tag=tag, row=row, latent=z)

        for g in range(2):
            points, draws = sample_generation(rng, mean, std)
            values = [incumbent] + [record(z, f"t{task}-g{g}-c{k}") for k, z in enumerate(points)]
            ranks = [rank(v["row"]) for v in values]
            updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
            incumbent = values[max(range(17), key=lambda k: (ranks[k], -k))]
            report["generations"].append(
                dict(
                    task=task,
                    generation=g,
                    mean_before=mean.tolist(),
                    std_before=std.tolist(),
                    draws=draws,
                    tags=[v["tag"] for v in values],
                    ranks=[list(v) for v in ranks],
                    elite=elite,
                    mean_after=updated.tolist(),
                    std_after=spread.tolist(),
                    selected=incumbent["tag"],
                )
            )
            mean, std = updated, spread
        for repeat in range(2):
            record(incumbent["latent"], f"t{task}-repeat{repeat}", incumbent["row"]["outcome"])
        report["selections"].append(
            dict(
                task=task,
                selected=incumbent["tag"],
                latent=incumbent["latent"].tolist(),
                rank=list(rank(incumbent["row"])),
                outcome=incumbent["row"]["outcome"],
            )
        )
    return report, cal


def test_all68_world_search_with4_inherited_calibrations_reconstructs_exactly():
    report, cal = transcript()
    verify_search(report, cal)
    assert len(report["records"]) == 68


@pytest.mark.parametrize(
    "fault", ["missing", "latent", "rank", "draw", "elite", "selection", "repeat", "prefix"]
)
def test_no_cherry_picked_or_modified_search_transcript_is_admitted(fault):
    r, c = transcript()
    if fault == "missing":
        r["records"].pop()
    elif fault == "latent":
        r["records"][0]["course"]["latent"][0][0] += 0.001
    elif fault == "rank":
        r["generations"][0]["ranks"][0][0] = 0
    elif fault == "draw":
        r["generations"][0]["draws"][0]["accepted"] = not r["generations"][0]["draws"][0][
            "accepted"
        ]
    elif fault == "elite":
        r["generations"][0]["elite"] = [1, 2, 3, 4]
    elif fault == "selection":
        r["selections"][0]["selected"] = "invented"
    elif fault == "repeat":
        r["records"][32]["outcome"]["B_legal_functional_receive_before_deadline"] = False
    else:
        r["records"][0]["prefix_identity_fields"] = 0
    with pytest.raises(ValueError):
        verify_search(r, c)


def test_contact_safety_and_actual_receiving_precede_dense_point_error():
    safe = outcome(True, error=0.9)
    unsafe = outcome(True, error=0.01)
    unsafe["passer_clean_foot"] = False
    assert rank(dict(outcome=safe)) > rank(dict(outcome=unsafe))
    missed = outcome(False, error=0.001)
    assert rank(dict(outcome=safe)) > rank(dict(outcome=missed))
    missed["incoming_arrival_speed_mps"] = float("nan")
    with pytest.raises(ValueError):
        rank(dict(outcome=missed))


@pytest.mark.parametrize("fault", [None, "drop_actual_NN", "different_actual_value"])
def test_calibration_comparison_omits_only_redundant_shadow_fields(tmp_path, fault):
    from scripts.rsi_native_context_recovery_common import digest

    roots = [tmp_path / "new", tmp_path / "old"]
    rows = []
    for which, root in enumerate(roots):
        root.mkdir()
        art = {}
        for key in ("initial", "physical", "A", "B"):
            fields = dict(value=np.zeros(100, np.float32))
            if key == "A":
                fields["NN_obs"] = np.zeros((100, 160), np.float32)
                if which:
                    fields["STUDENT_NN_obs"] = np.ones((100, 160), np.float32)
                elif fault == "drop_actual_NN":
                    fields.pop("NN_obs")
                elif fault == "different_actual_value":
                    fields["NN_obs"][0, 0] = 1
            p = root / (key + ".npz")
            np.savez_compressed(p, **fields)
            art[key] = dict(path=p.name, hash=digest(p))
        rows.append(dict(artifacts=art))
    if fault is None:
        assert compare_common(roots[0], rows[0], roots[1], rows[1]) == 5
    else:
        with pytest.raises(ValueError):
            compare_common(roots[0], rows[0], roots[1], rows[1])
