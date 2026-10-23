"""Admission/setup for bounded PUBLIC TRAIN search on the WHOLE learned motor."""

import json
import math

from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import receipt, source_pins


def admit(calibration_root, own_root, review_root, calibration_head):
    paths = [
        calibration_root / "complete.json",
        own_root / "complete.json",
        review_root / "complete.json",
        calibration_root / "declaration.json",
        calibration_root / "preflight.json",
        calibration_root / "scene.xml",
    ]
    if any(p.is_symlink() for p in paths) or any(
        (r / "failed.json").exists() for r in (calibration_root, own_root, review_root)
    ):
        raise ValueError(
            "complete ordinary calibration, own replay and failed-pilot review required"
        )
    s, o, r, d, pf = [json.loads(p.read_text()) for p in paths[:5]]
    if (
        s["source_head"] != calibration_head
        or o["source_head"] != calibration_head
        or r["source_head"] != calibration_head
        or s["status"] != "STUDENT_PASS_CONTINUATION_SHARD_COLLECTED_PENDING_WHOLE_OWN_REPLAY"
        or o["status"] != "STUDENT_PASS_CONTINUATION_SHARD_WHOLE_OWN_NATIVE_REPLAY_MATCH"
        or r["status"] != "ADAPTER_ACTUAL_STATE_PILOT_REVIEWED_NOT_TRAINED_OR_PROMOTED"
        or r["pilot_research_screen_passed"] is not False
        or o["producer_hash"] != digest(paths[0])
        or s["declaration_hash"] != digest(paths[3])
        or s["preflight_hash"] != digest(paths[4])
        or s["scene_hash"] != digest(paths[5])
        or o["counts"] != s["counts"]
        or len(s["records"]) != 40
        or o["records"]
        != [{k: v for k, v in row.items() if k != "artifacts"} for row in s["records"]]
        or d["protocol"] != "soccer.adapter_self_based_stroke_TRAIN80.v1"
        or d["teacher_basis"] != "student"
        or d["student_seed"] != 105025
        or d["student_kind"] != "consolidated-adapter-v1"
        or d["preflight_only"] is not False
        or d["teacher_training_hash"] != d["student_training_hash"]
        or any(r["input_file_hashes"].get(str(p)) != digest(p) for p in paths[:2])
        or any(
            v[k] is not False
            for v in (s, o, r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError(
            "exact whole learned-motor calibration and rejected research pilot required"
        )
    pins = dict(o["input_file_hashes"])
    pins.update({str(p): digest(p) for p in paths})
    source_pins(pins)
    check_pins(pins)
    _, bank = receipt(pins, d["bank_hash"])
    return s, d, pf, pins, bank


def open_setup(d, pf, pins, bank, scene):
    import numpy as np

    from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.native_pass_courses import NativePassCourse
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_stroke_search_common import open_motor

    contexts = (
        {int(k): v for k, v in bank["contexts"].items()}
        if isinstance(bank["contexts"], dict)
        else dict(enumerate(bank["contexts"]))
    )
    bench, actor = open_motor(
        pins,
        d["student_training_hash"],
        {i: contexts[i] for i in (1, 4, 20, 21)},
        checkpoint_kind=d["student_kind"],
    )
    setup = SharedRelaySetup(pins, contexts, bench=bench, actor=actor)
    if setup.xml != scene.read_text():
        raise ValueError("unchanged calibrated standard two-body scene required")
    typed = []
    for c in d["global_courses"][::2]:
        _, w, fwd, home, _ = setup.prepare(
            c["context"],
            c["side"],
            ball_planar_offset_world=np.array(c["ball_planar_offset_world"]),
        )
        q, _ = w.layouts["a/"].canonical(w.data)
        rel = home - q[36:38]
        lat = np.array([-fwd[1], fwd[0]])
        typed.append(
            NativePassCourse(
                c["state_id"],
                float(rel @ fwd),
                float(rel @ lat),
                hash_json(dict(qpos=w.data.qpos.tolist(), qvel=w.data.qvel.tolist())),
            )
        )
        if w.native_steps:
            raise ValueError("all40 preflight before search required")
    if (
        len(typed) != 40
        or preflight_receiving_courses(tuple(typed)) != pf["course_binding"]
        or bench.native_steps
    ):
        raise ValueError("same whole40 compiled TRAIN initial states required")
    return setup, actor


def rank(row):
    from rosclaw_soccer.rsi.pass_transfer_learning import controlled_pass_label

    o = row["outcome"]
    err = o["incoming"]["incoming_target_error_3d_before_B_contact_m"]
    speed = o["incoming_arrival_speed_mps"]
    if type(o["combined_precision_and_functional_receive"]) is not bool or any(
        v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0)
        for v in (err, speed)
    ):
        raise ValueError("finite nonnegative actual delivery metrics and typed composite required")
    score = -1e6 if err is None or speed is None else -err - 0.03 * max(0.0, speed - 3.0) ** 2
    return (
        int(
            o["whole_body_healthy"]
            and o["passer_full100NNrecovery"]
            and not o["B_excluded_contact"]
        ),
        int(o["passer_clean_foot"]),
        int(controlled_pass_label(o)),
        float(score + 1000 * o["combined_precision_and_functional_receive"]),
    )


def compare_common(root, row, oldroot, old, *, prefix=None):
    """All actual one-motor primitives, allowing old redundant shadow fields."""
    import numpy as np

    fields = 0
    for k in ("initial", "physical", "A", "B"):
        p, q = [r / v["artifacts"][k]["path"] for r, v in ((root, row), (oldroot, old))]
        if digest(q) != old["artifacts"][k]["hash"]:
            raise ValueError("bound calibrated primitives required")
        with np.load(p, allow_pickle=False) as x, np.load(q, allow_pickle=False) as y:
            shadow = {
                key
                for key in y.files
                if k == "A"
                and (
                    key.startswith(("STUDENT_", "TEACHER_", "teacher_advice_"))
                    or key == "student_executed"
                )
            }
            if set(x.files) != set(y.files) - shadow:
                raise ValueError("only redundant old shadowNN fields may be omitted")
            for key in x.files:
                a, b = x[key], y[key]
                if prefix is not None and k != "initial":
                    a, b = a[:prefix], b[:prefix]
                if a.dtype != b.dtype or not np.array_equal(a, b):
                    raise ValueError(
                        "actual calibrated prefix/whole identity differs: " + k + "." + key
                    )
                fields += 1
    return fields


def calibrations(s, state, task):
    rows = [
        r for r in s["records"] if r["course"]["state_id"] == state and r["course"]["task"] == task
    ]
    if len(rows) != 2 or [r["student_prefix_controls"] for r in rows] != [0, 80]:
        raise ValueError("both whole physically calibrated source arms required")
    if (
        any(r["course"]["context"] != 20 or r["course"]["assisted"] is not True for r in rows)
        or rows[0]["course"]["latent"] != [[0.0] * 6 for _ in range(2)]
        or {k: v for k, v in rows[0]["course"].items() if k != "latent"}
        != {k: v for k, v in rows[1]["course"].items() if k != "latent"}
    ):
        raise ValueError("exact current-student zero/pulse calibrated task pair required")
    return rows


def verify_search(report, calibration):
    """Reconstruct every proposal/elite/selection from ALL actual outcome rows."""
    import numpy as np

    from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites

    state = report["state_id"]
    if (
        state not in (20, 23, 26, 29)
        or len(report["records"]) != 68
        or len(report["generations"]) != 4
        or len(report["selections"]) != 2
    ):
        raise ValueError("whole predeclared four-state search shard required")
    cursor = 0

    def take(course, z, tag, task):
        nonlocal cursor
        row = report["records"][cursor]
        if (
            row["index"] != cursor
            or row["tag"] != tag
            or row["search_task"] != task
            or row["course"] != course | dict(latent=z.tolist())
            or type(row["prefix_identity_fields"]) is not int
            or row["prefix_identity_fields"] <= 0
            or row["neural_actor_optimizer_updates"] != 0
        ):
            raise ValueError("every ordered physical proposal must match declared CEM")
        cursor += 1
        return dict(tag=tag, row=row, latent=z)

    for task in (0, 1):
        old = calibrations(calibration, state, task)
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
        for g in range(2):
            samples, draws = sample_generation(rng, mean, std)
            values = [incumbent] + [
                take(course, z, f"t{task}-g{g}-c{k}", task) for k, z in enumerate(samples)
            ]
            ranks = [rank(v["row"]) for v in values]
            updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
            best = max(range(17), key=lambda k: (ranks[k], -k))
            incumbent = values[best]
            expected = dict(
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
            if report["generations"][2 * task + g] != expected:
                raise ValueError("CEM random draws, ranks, elites or selection changed")
            mean, std = updated, spread
        for repeat in range(2):
            current = take(course, incumbent["latent"], f"t{task}-repeat{repeat}", task)
            if current["row"]["outcome"] != incumbent["row"]["outcome"]:
                raise ValueError("selected outcome failed whole repetition")
        selected = dict(
            task=task,
            selected=incumbent["tag"],
            latent=incumbent["latent"].tolist(),
            rank=list(rank(incumbent["row"])),
            outcome=incumbent["row"]["outcome"],
        )
        if report["selections"][task] != selected:
            raise ValueError("complete final selection differs")
    if cursor != 68:
        raise ValueError("all68 new worlds required")
