"""Bind an initial-state portfolio to qualified SIM episodes, never admission."""

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.first_touch_outcome import outcome_contract
from rosclaw_soccer.rsi.initial_context_control_portfolio import InitialContextControlPortfolio
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState
from rosclaw_soccer.rsi.postcontact_navigation_probe import PostcontactNavigationProbe
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

PROFILE = "initial_context_portfolio_public40"
GROUPS = list(range(0, 160, 4))


def validate_option(candidate: Any) -> dict[str, Any] | None:
    if candidate is None:
        return None
    if (
        type(candidate) is not dict
        or set(candidate) not in ({"parameters", "swing"}, {"parameters", "swing", "postcontact"})
        or type(candidate["parameters"]) is not dict
        or set(candidate["parameters"]) != set(asdict(BoundedFirstTouchControl()))
        or candidate["swing"] not in ("legacy", "bounded_contact_diagnostic")
    ):
        raise ValueError("only original or bounded SIM control options allowed")
    BoundedFirstTouchControl(**candidate["parameters"])
    if "postcontact" in candidate:
        if (
            candidate["swing"] != "bounded_contact_diagnostic"
            or type(candidate["postcontact"]) is not dict
            or set(candidate["postcontact"]) != set(asdict(PostcontactNavigationProbe()))
        ):
            raise ValueError("postcontact control requires explicit diagnostic swing")
        PostcontactNavigationProbe(**candidate["postcontact"])
    return copy.deepcopy(candidate)


def logical_document(path: Path) -> dict[str, Any]:
    """Read only authenticated envelope metadata; do not claim full restoration.

    The caller must authenticate the file bytes. Actual qualification still
    restores/verifies the complete parent and all original numerical seals.
    """
    value = load_json_artifact(path)
    if value.get("schema") != "rosclaw.growth.shared_json_payload.v1":
        raise ValueError("explicit shared envelope required")
    if value.get("envelope_hash") != hash_json(
        {k: v for k, v in value.items() if k != "envelope_hash"}
    ):
        raise ValueError("initial metadata envelope integrity failure")
    return dict(value["stripped_document"])


def prefix_query(
    model: InitialContextControlPortfolio,
    options: tuple[dict[str, Any] | None, ...],
    trace: Any,
    commitment: dict[str, Any],
    view: dict[str, Any],
    *,
    expected_binding: dict[str, Any],
) -> dict[str, Any]:
    """Use frame0 only. Later trajectory/label rows are not policy inputs."""
    rows = {}
    for key, width in (
        ("canonical_qpos", 43),
        ("canonical_qvel", 41),
        ("root_pose_xyzw_m", 7),
        ("ball_position_before_step_m", 3),
        ("ball_linear_velocity_before_step_m_s", 3),
    ):
        raw = np.asarray(trace[key])
        if raw.ndim != 3 or len(raw) < 1 or raw.shape[1:] != (1, width):
            raise ValueError("complete measured initial state layout required")
        if raw.dtype.kind not in "fi" or not np.isfinite(raw[0]).all():
            raise ValueError("finite measured initial state required")
        rows[key] = raw[0, 0]
    if hash_json(expected_binding) != model.execution_binding_hash:
        raise ValueError("independent initial robot declaration differs from model binding")
    ball = rows["ball_position_before_step_m"]
    velocity = rows["ball_linear_velocity_before_step_m_s"]
    course = (float(ball[0]), float(ball[1]), float(velocity[0]))
    initial = NativeFirstTouchInitialState(
        course, np.asarray(expected_binding["initial_robot_qpos"][7:36], dtype=np.float64)
    )
    initial.verify(list(course), rows["canonical_qpos"], rows["canonical_qvel"])
    canonical = rows["canonical_qpos"]
    if (
        not np.array_equal(rows["root_pose_xyzw_m"], canonical[[0, 1, 2, 4, 5, 6, 3]])
        or not np.array_equal(ball, canonical[36:39])
        or not np.array_equal(velocity, rows["canonical_qvel"][35:38])
    ):
        raise ValueError("initial body/ball observations differ from canonical state")
    mean_hash = commitment["recurrent_sampling_factory"]["complete_canonical_mean_hash"]
    if view["model_hash"] != commitment["step_model_hash"] or view.get("mean_model") != {
        "$rosclaw_shared_payload": mean_hash
    }:
        raise ValueError("initial complete sampling view identity mismatch")
    binding = dict(
        initial_robot_qpos=rows["canonical_qpos"][:36].tolist(),
        initial_robot_qvel=rows["canonical_qvel"][:35].tolist(),
        world_hash=commitment["compiled_model_hash"],
        canonical_mean_hash=mean_hash,
        foundation_hash=commitment["foundation_hash"],
        late_actor_hash=commitment["late_actor_hash"],
        physics=commitment["physics"],
        outcome_contract_hash=outcome_contract()["contract_hash"],
        sampling_law={
            k: v for k, v in view.items() if k not in ("mean_model", "seed", "model_hash")
        },
    )
    gap = rows["ball_position_before_step_m"] - rows["root_pose_xyzw_m"][:3]
    context = (float(gap[0]), float(gap[1]), float(rows["ball_linear_velocity_before_step_m_s"][0]))
    query = model.query(context, execution_binding_hash=hash_json(binding))
    if query["status"] != "SIM_RESEARCH_OPTION":
        raise ValueError(
            "all declared portfolio initial states must be supported: " + query["reason"]
        )
    index = query["option_index"]
    if (
        len(options) != len(model.option_hashes)
        or hash_json(options[index]) != query["option_hash"]
    ):
        raise ValueError("selected control content differs from fixed model")
    return dict(
        context=list(context),
        execution_binding_hash=hash_json(binding),
        model_query=query,
        candidate=validate_option(options[index]),
        future_outcomes_read=False,
    )


@dataclass(frozen=True)
class PreparedPortfolio:
    model: InitialContextControlPortfolio
    options: tuple[dict[str, Any] | None, ...]
    fit_report_hash: str
    execution_binding: dict[str, Any]
    queries: dict[int, dict[str, Any]]


def prepare_portfolio(
    p: Any, manifest: dict[str, Any], config: dict[str, Any]
) -> PreparedPortfolio:
    model_path, fit_path = (
        Path(config["portfolio_model_path"]),
        Path(config["portfolio_fit_report_path"]),
    )
    pins = {str(path): p.digest(path) for path in (model_path, fit_path)}
    fit = p.sealed(fit_path)
    model = InitialContextControlPortfolio.from_dict(load_json_artifact(model_path))
    arms = config["portfolio_control_arm_count"]
    if (
        fit.get("schema") != "soccer.rsi.initial_context_portfolio_training_fit.v1"
        or fit.get("report_hash") != config["portfolio_fit_report_hash"]
        or model.model_hash != config["portfolio_model_hash"]
        or model.model_hash != fit.get("model_hash")
        or hash_json(model.to_dict()) != hash_json(fit.get("model"))
        or fit.get("training_contexts") != 16
        or fit.get("complete_control_arms") != arms
        or fit.get("source_option_context_rows") != 16 * arms
        or fit.get("complete_static_sampling_law_bound") is not True
        or fit.get("original_A_full_sampling_model_identity_matched") is not True
        or any(
            fit.get(key) is not False
            for key in ("teacher_qualified", "promotion_authorized", "hardware_authorized")
        )
        or fit.get("new_physics_executions") != 0
        or fit.get("neural_optimizer_updates") != 0
    ):
        raise ValueError("exact complete non-authorizing consumed-public fit required")
    fitted_source = Path(config["portfolio_fit_source_root"])
    if fit["source_head"] != config["portfolio_fit_source_head"]:
        raise ValueError("exact portfolio algorithm source required")
    relative = "src/rosclaw_soccer/rsi/initial_context_control_portfolio.py"
    source_path = fitted_source / relative
    expected_source = fit["input_file_hashes"][str(source_path)]
    if (
        p.digest(source_path) != expected_source
        or p.digest(Path(config["source_root"]) / relative) != expected_source
    ):
        raise ValueError("fitting and executing portfolio algorithms differ")
    p.HEADS[str(fitted_source)] = config["portfolio_fit_source_head"]
    pins[str(source_path)] = expected_source
    options = tuple(validate_option(row["candidate"]) for row in fit["options"])
    if (
        tuple(hash_json(option) for option in options) != model.option_hashes
        or len(options) != arms
    ):
        raise ValueError("complete ordered control content required")
    queries = {}
    for group in GROUPS:
        row = manifest["records"][group]
        trace_witness, view_witness = (row["witnesses"][key] for key in ("trace", "sampling_view"))
        for witness in (trace_witness, view_witness):
            path = Path(witness["path"])
            if p.digest(path) != witness["sha256"]:
                raise ValueError("initial portfolio source identity mismatch")
            pins[str(path)] = witness["sha256"]
        folder = Path(trace_witness["path"]).parent
        cp, world = folder / "commitment.json", folder / "compiled_model.mjb"
        commitment = load_json_artifact(cp)
        if p.digest(world) != commitment["compiled_model_hash"]:
            raise ValueError("initial compiled world differs from commitment")
        pins.update({str(cp): p.digest(cp), str(world): p.digest(world)})
        with np.load(trace_witness["path"], allow_pickle=False) as trace:
            query = prefix_query(
                model,
                options,
                trace,
                commitment,
                logical_document(Path(view_witness["path"])),
                expected_binding=fit["execution_binding"],
            )
        if query["context"] != row["draw"]["course"]:
            raise ValueError("declared public course differs from measured initial context")
        if query["model_query"]["training_context_match"] is not (group < 64):
            raise ValueError("declared fitted/public-not-fitted split differs from actual contexts")
        queries[group] = dict(
            schema="soccer.rsi.initial_portfolio_selection.v1",
            group=group,
            **query,
            fit_report_hash=fit["report_hash"],
            model_hash=model.model_hash,
            evaluation_partition="training_reexecution"
            if group < 64
            else "public_not_fitted_state",
            source_initial_trace_witness=trace_witness,
            source_sampling_view_witness=view_witness,
            selected_before_rollout=True,
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
    if len({tuple(queries[g]["context"]) for g in GROUPS}) != 40:
        raise ValueError("forty distinct initial states required")
    for location, expected in pins.items():
        if p.digest(Path(location)) != expected:
            raise ValueError("portfolio preflight inputs changed")
    p.add_pins(pins)
    p.stable()
    return PreparedPortfolio(model, options, fit["report_hash"], fit["execution_binding"], queries)


def publish_selection(folder: Path, selection: dict[str, Any]) -> dict[str, str]:
    from scripts.rsi_atomic_artifacts import write_once

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "portfolio-selection.json"
    write_once(path, dict(selection, report_hash=hash_json(selection)))
    return dict(path=str(path), sha256=hash_bytes(path.read_bytes()))


def verify_selected_receipt(
    receipt: dict[str, Any], selection: dict[str, Any], portfolio: PreparedPortfolio, root: Path
) -> None:
    witness = receipt["portfolio_selection_witness"]
    path = Path(witness["path"])
    if (
        not path.resolve().is_relative_to(root.resolve())
        or hash_bytes(path.read_bytes()) != witness["sha256"]
    ):
        raise ValueError("portfolio selection witness changed or escaped experiment")
    actual = json.loads(path.read_text())
    expected = dict(selection, report_hash=hash_json(selection))
    if (
        actual != expected
        or receipt["candidate"] != selection["candidate"]
        or receipt["group"] != selection["group"]
    ):
        raise ValueError("actual control differs from committed portfolio selection")
    cp = Path(receipt["witnesses"]["commitment"]["path"])
    report = logical_document(Path(receipt["witnesses"]["report"]["path"]))
    view = report["executed_motor_policy"]["step_motor_proof"]["model"]
    try:
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as trace:
            query = prefix_query(
                portfolio.model,
                portfolio.options,
                trace,
                load_json_artifact(cp),
                view,
                expected_binding=portfolio.execution_binding,
            )
    except ValueError as error:
        raise ValueError(
            "actual initial state/model binding differs from declared query"
        ) from error
    if any(query[key] != selection[key] for key in query):
        raise ValueError("actual initial state/model choice differs from pre-execution query")
