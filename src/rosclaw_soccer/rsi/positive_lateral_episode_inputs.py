"""Authenticate prospective source views for actual qualified body episodes.

No simulator, learner or authority. Registry rows without a sampling view are
metadata fillers, never executable examples or extra independent scenes.
"""

from pathlib import Path

from rosclaw_soccer.rsi.portfolio_physical_study import logical_document
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import (
    PublicPlanningContext,
    validate_context_source,
)
from rosclaw_soccer.sim.contracts import hash_json


def episode_inputs(registry, *, expected_registry_hash, behavior_model_hash, digest, pins):
    """Reopen exact prepared assets; full numerical policy audit stays downstream."""
    if (
        registry.get("schema") != "soccer.rsi.positive_lateral_sampling_registry.v1"
        or registry.get("report_hash") != expected_registry_hash
        or expected_registry_hash
        != hash_json({k: v for k, v in registry.items() if k != "report_hash"})
        or any(
            registry.get(k) is not False
            for k in (
                "execution_authorized",
                "dataset_admission_authorized",
                "teacher_qualified",
                "fresh_exam",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            type(registry.get(k)) is not int or registry[k] != 0
            for k in (
                "new_nn_model_allocations",
                "new_neural_inferences",
                "new_physical_steps",
                "new_optimizer_updates",
            )
        )
        or registry.get("source_view_count") != 8
        or registry.get("metadata_registry_rows") != 96
        or registry.get("source_registry_fillers_are_not_physical_trials") is not True
        or registry.get("all_old_receding_failures_retained_not_relabelled") is not True
    ):
        raise ValueError("whole original metadata-only registry required")
    scope = build_scope(behavior_model_hash)
    if registry.get("scope") != scope or registry.get("scope_hash") != scope["scope_hash"]:
        raise ValueError("unchanged metadata sampling domain and first-match rule required")
    for name, h in registry["input_file_hashes"].items():
        if digest(Path(name)) != h:
            raise ValueError("original registry input/source bytes changed")
        pins[name] = h
    mean = registry["shared_mean"]
    if mean["model_hash"] != behavior_model_hash or digest(Path(mean["path"])) != mean["sha256"]:
        raise ValueError("unchanged whole mean payload bytes and behavior identity required")
    pins[mean["path"]] = mean["sha256"]
    plan = PublicFirstTouchTrainingPlan(
        tuple(scope["source_plan"]["seeds"]), behavior_model_hash, 19
    )
    rows = registry["records"]
    if (
        plan.contract() != scope["source_plan"]
        or len(rows) != 96
        or [r["draw"] for r in rows] != scope["source_plan"]["jobs"]
        or any(r.get("metadata_only_no_physical_trajectory") is not True for r in rows)
    ):
        raise ValueError("entire 96-row source registry order required; not a collected bank")
    contexts = tuple(
        PublicPlanningContext(
            c["context_index"],
            c["source_group"],
            c["source_context_id"],
            c["source_seed"],
            c["source_lane"],
            c["sampling_seed"],
            tuple(c["course"]),
            c["perturbation_index"],
        )
        for c in scope["contexts"]
    )
    groups = [c.source_group for c in contexts]
    if registry["declared_physical_source_groups"] != groups:
        raise ValueError("all eight predeclared physical source groups required")
    for context in contexts:
        validate_context_source(context, plan, group=context.source_group)
        witness = rows[context.source_group]["witnesses"]["sampling_view"]
        path = Path(witness["path"])
        if path.parent.name != "models" or digest(path) != witness["sha256"]:
            raise ValueError("byte-pinned qualified sampling-view location required")
        view = logical_document(path)
        if (
            view.get("schema") != "soccer.rsi.recurrent_motor_sampling.v1"
            or type(view.get("seed")) is not int
            or view["seed"] != context.sampling_seed
            or view.get("mean_model") != {"$rosclaw_shared_payload": mean["payload_hash"]}
            or view.get("std_raw") != 0.1
            or view.get("rho") != 0.9
            or view.get("training_only") is not True
            or view.get("activation_ceiling") != "SIM_ONLY"
            or any(
                view.get(k) is not False
                for k in (
                    "promotion_authorized",
                    "runtime_execution_authorized",
                    "hardware_authorized",
                    "fresh_holdout_open_authorized",
                )
            )
        ):
            raise ValueError("unchanged whole-mean binding and original sampling law required")
        pins[str(path)] = witness["sha256"]
    if {r["draw"]["group"] for r in rows if r["witnesses"]} != set(groups):
        raise ValueError("registry fillers cannot become physical episodes")
    return plan, dict(records=rows), contexts


def prospective_split(scope):
    """Future supervised split by source seed BEFORE collecting new outcomes.

    This is public development validation, not a private exam or global ancestry
    separation claim. Both perturbations and all controller families stay together.
    """
    seeds = scope["source_seed_search"]["selected_seeds"]
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("four distinct prospective source seeds required")
    rows = [
        dict(
            source_seed=c["source_seed"],
            group=c["source_group"],
            context_hash=c["context_hash"],
            partition="train" if c["source_seed"] in seeds[:3] else "validation",
        )
        for c in scope["contexts"]
    ]
    if len(rows) != 8 or any(sum(r["source_seed"] == seed for r in rows) != 2 for seed in seeds):
        raise ValueError("exactly two perturbations in each source cluster required")
    value = dict(
        schema="soccer.rsi.positive_lateral_prospective_split.v1",
        source_plan_hash=scope["source_plan"]["plan_hash"],
        rows=rows,
        train_source_seeds=seeds[:3],
        validation_source_seeds=seeds[3:],
        training_only_normalization_required=True,
        controller_pairs_share_partition=True,
        global_ancestry_separation_proven=False,
        fresh_exam=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, split_hash=hash_json(value))
