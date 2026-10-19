"""New native-request boundary and raw capture; no actual G1 teacher claim."""

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe
from rosclaw_soccer.rsi.impact_feedback_teacher import ImpactCandidate
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.native_impact_feedback import (
    NativeImpactFeedback,
    close,
    require_native_impact_request,
)
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.rsi.recurrent_sampling_episode_factory import RecurrentSamplingEpisodeFactory
from tests.rsi.test_feedback_impact_score import prediction
from tests.rsi.test_swing_release_default_arithmetic import OffPath


def request():
    signature = inspect.signature(run_native_first_touch_episode)
    v = {
        k: None if p.default is inspect.Parameter.empty else p.default
        for k, p in signature.parameters.items()
    }
    v.update(
        args=SimpleNamespace(
            record_foundation_observation=True,
            foundation_only=False,
            proposal_decoder="reference",
            step_model=Path("unused"),
            body_response_bundle=None,
            cached_proposal_envelope=False,
            neural_model=None,
            motor_policy=None,
            observation_snapshot="cached",
            root_velocity_reference="body-com",
        ),
        partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
        approach_mode="bounded_approach_diagnostic",
        swing_mode="bounded_contact_diagnostic",
        control_profile=BoundedFirstTouchControl(),
        recurrent_sampling_factory=object.__new__(RecurrentSamplingEpisodeFactory),
        original_ik_planner_pulse=OriginalIkPlannerPulse(0.08),
        foot_heading_control=FootHeadingContactProbe(False),
        mesh_task_contact_world_hash="unused",
        mesh_world_review=object(),
        generated_seed_episode_binding=True,
        record_default_planner_events=True,
        use_impact_feedback_teacher=True,
    )
    return v


def test_exact_captured_unmixed_request():
    require_native_impact_request(request())


@pytest.mark.parametrize(
    "key,value",
    [
        ("partition", "FRESH"),
        ("approach_mode", "legacy"),
        ("swing_mode", "legacy"),
        ("control_profile", None),
        ("recurrent_sampling_factory", None),
        ("training_plan", object()),
        ("training_group", 28),
        ("consumed_bank_hash", "consumed"),
        ("original_ik_planner_pulse", OriginalIkPlannerPulse(0.0)),
        ("foot_heading_control", None),
        ("mesh_task_contact_world_hash", None),
        ("mesh_world_review", None),
        ("generated_seed_episode_binding", False),
        ("record_default_planner_events", False),
        ("use_impact_feedback_teacher", 1),
    ],
)
def test_no_weakened_boundary(key, value):
    v = request()
    v[key] = value
    with pytest.raises(ValueError):
        require_native_impact_request(v)


@pytest.mark.parametrize(
    "key",
    [
        "sampling_factory",
        "extended_factory",
        "proposal_factory",
        "imitation_factory",
        "recurrent_factory",
        "recurrent_clipped_factory",
        "postcontact_profile",
        "locomotion_mode_probe",
        "reference_height_probe_m",
        "cerebellar_model",
        "cerebellar_sampling_seed",
        "reference_student_model",
        "planner_navigation_pulse",
        "foundation_approach_probe",
        "suppress_postcontact_swing_release",
        "use_contact_exit_torque_mpc",
        "use_directional_contact_compliance",
        "use_mesh_surface_swing",
    ],
)
def test_no_mixed_control_or_probe(key):
    v = request()
    v[key] = True
    with pytest.raises(ValueError):
        require_native_impact_request(v)


@pytest.mark.parametrize(
    "key,value",
    [
        ("record_foundation_observation", False),
        ("foundation_only", True),
        ("proposal_decoder", "legacy"),
        ("step_model", None),
        ("body_response_bundle", {}),
        ("cached_proposal_envelope", True),
        ("neural_model", "other"),
        ("motor_policy", "other"),
        ("observation_snapshot", "current-kinematic"),
        ("root_velocity_reference", "origin"),
    ],
)
def test_no_different_observation_or_actor(key, value):
    v = request()
    setattr(v["args"], key, value)
    with pytest.raises(ValueError):
        require_native_impact_request(v)


def test_unknown_future_control_parameter_is_rejected():
    v = request()
    v["extra_control"] = False
    with pytest.raises(ValueError):
        require_native_impact_request(v)
    v = request()
    del v["args"]
    with pytest.raises(ValueError):
        require_native_impact_request(v)


def test_new_default_true_cannot_be_hidden_by_off_projection():
    node = next(
        n
        for n in ast.parse(Path(inspect.getfile(run_native_first_touch_episode)).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "run_native_first_touch_episode"
    )
    i = [a.arg for a in node.args.kwonlyargs].index("use_impact_feedback_teacher")
    node.args.kw_defaults[i] = ast.Constant(value=True)
    with pytest.raises(AssertionError):
        OffPath().visit(node)


def test_native_bad_flag_fails_before_model_or_output_allocation():
    with pytest.raises(ValueError, match="boolean"):
        run_native_first_touch_episode(
            SimpleNamespace(),
            course=(0, 0, 0),
            partition="INVALID",
            entry_source_path=Path("unused"),
            consumed_bank_hash=None,
            use_impact_feedback_teacher=1,
        )


def test_raw_candidates_are_saved_with_neural_targets_before_return(tmp_path):
    adapter = object.__new__(NativeImpactFeedback)
    adapter.root = tmp_path
    f = prediction()
    candidate = ImpactCandidate(
        "synthetic", np.zeros(29), f, {}, {"decoder_input": np.zeros((1, 994), dtype=np.float32)}
    )
    witnesses = adapter._save(30, [candidate])
    with np.load(witnesses[0]["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["canonical_qvel_after_substep"], f.arrays["canonical_qvel_after_substep"]
        )
        assert archive["offset_nm"].shape == (29,)
        assert archive["neural_next_decoder_input"].shape == (1, 994)
    with pytest.raises(FileExistsError):
        adapter._save(30, [candidate])


def test_mirror_comparison_rejects_nonfinite_shapes_and_mismatch():
    close(np.zeros(29), np.zeros(29), "synthetic")
    for value in (np.zeros(28), np.ones(29), np.full(29, np.nan)):
        with pytest.raises(ValueError):
            close(value, np.zeros(29), "synthetic")


def test_failed_mirror_keeps_native_prefix_state_and_budget(tmp_path):
    from tests.physics.test_g1_feedback_forecast import world

    model, data, forecaster = world()
    adapter = object.__new__(NativeImpactFeedback)
    adapter.root, adapter.model = tmp_path, model
    adapter.ik = SimpleNamespace(before_step=lambda *a: np.zeros(29))
    adapter.teacher = SimpleNamespace(last_frame=-1, last_candidates=[], forecaster=forecaster)
    with pytest.raises(ValueError, match="mirror differs"):
        adapter.before_frame(
            data,
            frame=30,
            navigation_target=np.zeros(29),
            native_increment=np.ones(29),
            target=np.zeros(29),
            previous=np.zeros(12),
            tracker=None,
            decoder=None,
            policy={},
            history={"canonical_qpos": [data.qpos.copy()]},
        )
    failure = json.loads((tmp_path / "frame-030-failure.json").read_text())
    assert failure["forecast_attempted_steps"] == 0
    assert failure["native_substeps_before_failed_decision"] == 300
    with np.load(tmp_path / "frame-030-failure-prefix.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["canonical_qpos"][0], data.qpos)
        assert "native_integration_state" in a


def test_legacy_cpu_audit_cannot_implicitly_qualify_new_teacher_even_if_offsets_zero(tmp_path):
    import mujoco

    from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once

    source = Path(__file__)
    snapshot, trace = tmp_path / "compiled_model.mjb", tmp_path / "physical_trace.npz"
    snapshot.write_bytes(b"must reject before parsing model")
    trace.write_bytes(b"must reject before parsing trace")
    c = dict(
        source_hash=hash_bytes(source.read_bytes()),
        physical_trace_hash=hash_bytes(trace.read_bytes()),
        compiled_model_hash=hash_bytes(snapshot.read_bytes()),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
        physics=dict(engine="MuJoCo", device="cpu", version=mujoco.__version__),
        impact_feedback_teacher={"teacher_qualified": False},
    )
    write_once(tmp_path / "commitment.json", c)
    report = dict(c, commitment_hash=hash_json(c))
    report["report_hash"] = hash_json(report)
    write_once(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="distinct causal search audit"):
        audit_cpu_transfer(tmp_path, source)
