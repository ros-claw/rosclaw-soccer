"""New sealed96 near cases: frozen anchor and ALL three categorical endpoints.

Not the canonical far RSI/private Fresh bank. No training, retries, actor-input
labels or outcome-dependent case replacement. Every arm is actually integrated
and independently replays its current neural policy/muscle/native physics.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("declare", "run"), required=True)
    for k in ("inputs", "core-source", "review", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-inputs-hash", "expected-review-hash", "review-unit"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--expected-plan-hash")
    p.add_argument("--indices", type=int, nargs="+")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    wait_for_terminal_collection(a.review_unit)
    if digest(a.review / "complete.json") != a.expected_review_hash:
        raise ValueError("fixed completed three-seed review required")
    review = json.loads((a.review / "complete.json").read_text())
    if review["status"] != "ALL_THREE_CATEGORICAL_SEEDS_NATIVE_PRIMITIVES_AUTHENTICATED" or not any(
        r["necessary_research_growth"] for r in review["summaries"].values()
    ):
        raise ValueError("at least one actual whole40 research gain required before new exam")
    import subprocess

    sources = [
        Path(k).parent.parent
        for k in review["input_file_hashes"]
        if k.endswith("/scripts/rsi_review_categorical_motion.py")
    ]
    if (
        len(sources) != 1
        or subprocess.check_output(
            ["git", "-C", str(sources[0]), "rev-parse", "HEAD"], text=True
        ).strip()
        != review["source_head"]
    ):
        raise ValueError("one immutable reviewed inference source required")
    for folder in ("src", "scripts"):
        for old in (sources[0] / folder).rglob("*.py"):
            rel = old.relative_to(sources[0])
            if str(rel) != "src/rosclaw_soccer/training/receiving_course_preflight.py" and digest(
                old
            ) != digest(Path.cwd() / rel):
                raise ValueError("new population cannot change frozen inference: " + str(rel))
    for k, h in review["input_file_hashes"].items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting frozen research lineage")
        pins[k] = h
    pins[str(a.review / "complete.json")] = a.expected_review_hash
    # Explicitly retain all three fixed endpoints, including rejected seed29.
    roster = []
    for seed in (17, 29, 43):
        roots = [
            Path(k).parent
            for k in pins
            if k.endswith(f"/rsi-categorical-motion-seed{seed}-20261020/complete.json")
        ]
        if len(roots) != 1:
            raise ValueError("one already-frozen endpoint per seed required")
        c = json.loads((roots[0] / "complete.json").read_text())
        if (
            digest(roots[0] / "candidate.pt") != c["checkpoint_hash"]
            or c["optimizer_steps"] != 1000
        ):
            raise ValueError("unchanged fixed categorical endpoint required")
        roster.append(
            dict(
                variant=f"seed{seed}",
                path=str(roots[0] / "candidate.pt"),
                hash=c["checkpoint_hash"],
                anchor_hash=c["anchor_hash"],
            )
        )
    if len({r["anchor_hash"] for r in roster}) != 1:
        raise ValueError("common matched anchor required")
    anchors = [
        Path(k) for k, h in pins.items() if h == roster[0]["anchor_hash"] and k.endswith(".pt")
    ]
    if len(anchors) != 1:
        raise ValueError("exact shared research anchor required")
    roster.insert(0, dict(variant="anchor", path=str(anchors[0]), hash=roster[0]["anchor_hash"]))
    from rosclaw_soccer.training.categorical_motion_holdout import (
        categorical_motion_holdout_courses,
    )
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    courses = categorical_motion_holdout_courses()
    preflight = preflight_receiving_courses(courses)
    # Original40 and previously consumed192 are not reused as new cases.
    if {c.course for c in courses} & {tuple(r["course"]["course"]) for r in bd["rows"]}:
        raise ValueError("new population overlaps original development states")
    table = json.loads(json.dumps([asdict(c) for c in courses]))
    check_pins(pins)
    if a.mode == "declare":
        if a.plan or a.indices:
            raise ValueError("declaration contains whole96 only")
        a.output.mkdir(parents=True)
        write_once(a.output / "courses.json", dict(courses=table, preflight_hash=preflight))
        write_once(
            a.output / "complete.json",
            dict(
                status="CATEGORICAL_POSTFREEZE96_DECLARED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                review_hash=a.expected_review_hash,
                preflight_hash=preflight,
                courses_hash=digest(a.output / "courses.json"),
                roster=roster,
                planned_native_steps=2304000,
                independent_contexts=96,
                body_and_world_unchanged=True,
                original_far_or_private_fresh=False,
                optimizer_steps=0,
                new_native_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        return
    if (
        not a.plan
        or not a.expected_plan_hash
        or digest(a.plan / "complete.json") != a.expected_plan_hash
        or not a.indices
        or len(set(a.indices)) != len(a.indices)
        or any(i not in range(96) for i in a.indices)
    ):
        raise ValueError("sealed full96 plan and unique declared worker subset required")
    plan = json.loads((a.plan / "complete.json").read_text())
    if (
        plan["source_head"] != a.expected_head
        or plan["roster"] != roster
        or plan["preflight_hash"] != preflight
        or digest(a.plan / "courses.json") != plan["courses_hash"]
        or json.loads((a.plan / "courses.json").read_text())["courses"] != table
    ):
        raise ValueError("frozen new cases/source/models differ")
    for name in ("complete.json", "courses.json"):
        pins[str(a.plan / name)] = digest(a.plan / name)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            plan_hash=a.expected_plan_hash,
            preflight_hash=preflight,
            indices=a.indices,
            roster=roster,
            shared_prefix_neural_math=True,
            prefix_physically_reexecuted_each_arm=True,
            independent_replay_recomputes_student_not_prefix_networks=True,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        result = run(a, spec, pins, courses, roster)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CATEGORICAL_NEAR96_FOUR_ARM_NATIVE_VERIFIED",
                source_head=a.expected_head,
                results=result,
                new_native_steps=24000 * len(result),
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                automatic_retry_authorized=False,
                partial_charge_requires_preserved_receipts=True,
            ),
        )
        raise


def run(a, spec, pins, courses, roster):
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_navigation import (
        G1SonicNavigation,
        SonicNavigationConfig,
    )
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.categorical_motion import create_categorical_motion
    from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
    from rosclaw_soccer.rsi.foundation_motor_codec import FoundationMotorCodec
    from rosclaw_soccer.rsi.full_entry_forecast import nominal_prefix20, score_full_entry
    from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree
    from rosclaw_soccer.rsi.recorded_entry_review import replay_prefix
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_motion_student_rollout import replay_motion_student, rollout_motion_student
    from scripts.rsi_motor_student_rollout import PHYSICAL_FIELDS, exact_array
    from scripts.rsi_native_context_recovery_common import digest
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    class DeterministicAnchor:
        def __init__(self, parent):
            self.parent = parent

        def reset(self):
            self.parent.sample_calls = 0

        def __call__(self, x, hidden):
            row = self.parent.sample(x, hidden, generator=None, stochastic=False)
            return row["token"], row["value"], row["hidden"]

        def network_call_counts(self):
            return self.parent.network_call_counts()

    parents = [
        load_model(spec, Path(roster[0]["path"])).eval().requires_grad_(False) for _ in range(4)
    ]
    models = [DeterministicAnchor(parents[0])]
    for r, parent in zip(roster[1:], parents[1:], strict=True):
        before = {k: v.clone() for k, v in parent.state_dict().items()}
        model = create_categorical_motion(parent)
        cp = torch.load(r["path"], map_location="cpu", weights_only=True)
        if (
            cp["model_kind"] != "categorical_fsq32_motion64"
            or cp["anchor_hash"] != roster[0]["hash"]
            or cp["world_hash"] != digest(Path(spec["world"]))
        ):
            raise ValueError("frozen categorical execution identity required")
        model.load_state_dict(cp["state_dict"], strict=True)
        for k, v in parent.state_dict().items():
            exact_tensor_tree(v, before[k], path=k)
        models.append(model.eval().requires_grad_(False))
    world = mujoco.MjModel.from_binary_path(spec["world"])
    model_root = Path(spec["model_root"])
    backend = G1SonicRunupController(
        model_root, G1SonicRunupConfig(model_variant="low_latency"), inference_threads=1
    )
    priors = [Path(k) for k in pins if Path(k).name == "entry-affordance.json"]
    if len(priors) != 1:
        raise ValueError("one original prefix prior required")
    stored = json.loads(priors[0].read_text())
    prior = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if prior.contract_hash != stored["contract_hash"]:
        raise ValueError("prefix contract differs")

    class CheckedPrefixBody:
        def __init__(self, actual, prefix):
            self.actual, self.prefix = actual, prefix

        def __getattr__(self, k):
            return getattr(self.actual, k)

        def step(self, frame, target):
            row = self.actual.step(frame, target)
            for k in PHYSICAL_FIELDS:
                exact_array(row[k], self.prefix[k][frame : frame + 1])
            return row

    def reset(model):
        if isinstance(model, DeterministicAnchor):
            model.reset()
        else:
            model.calls = 0

    result = []
    for i in a.indices:
        course = courses[i]
        folder = a.output / f"course{i:03d}"
        folder.mkdir()

        def body(course=course):
            return G1TargetEpisode(
                world,
                world_hash=digest(Path(spec["world"])),
                kp=backend.kp.copy(),
                kd=backend.kd.copy(),
                course=course.course,
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )

        first = body()
        nav = G1SonicNavigation(
            model_root,
            course.agent_id,
            SonicNavigationConfig(
                maximum_frames=300,
                model_variant="low_latency",
                inference_threads=1,
                record_default_planner_events=True,
            ),
        )
        try:
            prefix = nominal_prefix20(nav, first, prior)
        except Exception as error:
            write_once(
                folder / "failure-prefix.json",
                dict(
                    error=repr(error),
                    native_steps=first.integrated_steps,
                    attempted_steps=first.attempted_steps,
                    automatic_retry_authorized=False,
                ),
            )
            raise
        del nav
        variants = []
        for r, model in zip(roster, models, strict=True):
            part = folder / r["variant"]
            part.mkdir()
            actual = first if r["variant"] == "anchor" else body()
            progress = {}
            independent = None
            try:
                if r["variant"] != "anchor":
                    replay_prefix(CheckedPrefixBody(actual, prefix), prefix, prior)
                reset(model)
                trace = rollout_motion_student(
                    model, FoundationMotorCodec(backend), actual, prefix, progress=progress
                )
                np.savez_compressed(part / "native.npz", **trace)
                reset(model)
                independent = body()
                proof = replay_motion_student(
                    model, FoundationMotorCodec(backend), independent, trace
                )
                write_once(part / "replay.json", proof)
            except Exception as error:
                rows = progress.get("rows", [])
                if rows:
                    np.savez_compressed(
                        part / "partial.npz",
                        **{k: np.concatenate([q[k] for q in rows]) for k in rows[0]},
                    )
                write_once(
                    part / "failure.json",
                    dict(
                        error=repr(error),
                        original_native_steps=actual.integrated_steps,
                        original_attempted_steps=actual.attempted_steps,
                        independent_native_steps=independent.integrated_steps
                        if independent is not None
                        else 0,
                        automatic_retry_authorized=False,
                    ),
                )
                raise
            score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
            pd = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
            row = dict(
                variant=r["variant"],
                checkpoint_hash=r["hash"],
                score=score,
                prefix_diagnostics=pd,
                causal_qualified=bool(
                    score["qualified"]
                    and pd["prefix_physical_safe"]
                    and pd["post_boundary_control_can_cause_first_touch"]
                ),
                native_hash=digest(part / "native.npz"),
                native_steps=actual.integrated_steps + independent.integrated_steps,
                replay=proof,
            )
            if row["native_steps"] != 6000:
                raise ValueError("complete native and independent replay required")
            write_once(part / "complete.json", row)
            variants.append(row)
        row = dict(course=asdict(course), variants=variants, new_native_steps=24000)
        write_once(folder / "complete.json", row)
        write_once(a.output / f"progress{i:03d}.json", row)
        result.append(row)
        print(
            "CATEGORICAL_NEAR96_FOUR_ARM", i, [v["causal_qualified"] for v in variants], flush=True
        )
    return result


if __name__ == "__main__":
    main()
