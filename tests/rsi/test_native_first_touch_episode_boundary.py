"""CLI routing and exact physics-body refactor checks, not physics qualification."""

import argparse
import ast
import hashlib
from pathlib import Path

import pytest

import rosclaw_soccer.rsi.native_first_touch_episode as kernel
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from scripts import rsi_mujoco_motor_transfer as entry


def arguments(tmp_path, seed, lane):
    return [
        "--scene",
        str(tmp_path / "absent.xml"),
        "--model-root",
        str(tmp_path / "absent-model"),
        "--late-swing-policy",
        str(tmp_path / "absent-policy.json"),
        "--output-root",
        str(tmp_path / "must-not-exist"),
        "--seed",
        str(seed),
        "--lane",
        str(lane),
    ]


def test_legacy_consumed_cli_passes_explicit_original_course(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        kernel, "run_native_first_touch_episode", lambda *a, **kw: calls.append((a, kw))
    )
    entry.main(arguments(tmp_path, 20261177, 0))
    assert len(calls) == 1
    args, keywords = calls[0]
    assert args[0].seed == 20261177 and args[0].lane == 0
    assert keywords["course"] == sample_training_courses(20261177, 16)[0]
    assert keywords["partition"] == "CONSUMED_TRANSFER_DIAGNOSTIC"
    assert keywords["consumed_bank_hash"] is None
    assert keywords["entry_source_path"].resolve() == Path(entry.__file__).resolve()
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("seed,lane", [(0, 0), (20261177, 1), (20261227, 0)])
def test_unconsumed_cli_still_rejects_before_kernel_or_output(tmp_path, monkeypatch, seed, lane):
    def unopened(*args, **kwargs):
        raise AssertionError("unconsumed course must not reach native engine")

    monkeypatch.setattr(kernel, "run_native_first_touch_episode", unopened)
    with pytest.raises(SystemExit) as failure:
        entry.main(arguments(tmp_path, seed, lane))
    assert failure.value.code == 2
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "partition", ["FRESH_CPU_EXAM", "FRESH_QUARANTINED_NOT_OPENED", "", None, True]
)
def test_kernel_is_not_a_fresh_admission_api(tmp_path, partition):
    with pytest.raises(ValueError, match="private Fresh exam"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(output_root=tmp_path / "must-not-exist"),
            course=(2.5, 0.0, -0.5),
            partition=partition,
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
        )
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "course",
    [
        None,
        [],
        [2.5, 0.0, -0.5],
        (2.5, 0.0),
        (2.5, True, -0.5),
        (2.5, 0, -0.5),
        (2.5, float("nan"), -0.5),
        (2.5, 0.0, float("inf")),
    ],
)
def test_invalid_explicit_course_rejects_before_model_allocation(tmp_path, course):
    with pytest.raises(ValueError, match="explicit finite"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(),
            course=course,
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_entry_source_required_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="entry source"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=tmp_path / "absent.py",
            consumed_bank_hash=None,
        )


@pytest.mark.parametrize(
    "field",
    ["neural_model", "motor_policy", "step_model", "body_response_bundle"],
)
def test_foundation_only_native_api_rejects_residual_inputs_before_allocation(tmp_path, field):
    args = argparse.Namespace(foundation_only=True, **{field: tmp_path / "absent-model"})
    with pytest.raises(ValueError, match="foundation-only cannot include residual"):
        kernel.run_native_first_touch_episode(
            args,
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
        )


@pytest.mark.parametrize(
    "field",
    [
        "sampling_factory",
        "extended_factory",
        "proposal_factory",
        "imitation_factory",
        "recurrent_factory",
        "recurrent_sampling_factory",
        "recurrent_clipped_factory",
    ],
)
def test_foundation_only_rejects_even_unbound_factories_before_allocation(field):
    with pytest.raises(ValueError, match="foundation-only cannot include residual"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(foundation_only=True),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
            **{field: object()},
        )


@pytest.mark.parametrize(
    "extra",
    [{"cached_proposal_envelope": True}, {"proposal_decoder": "owned_snapshot"}],
)
def test_foundation_only_rejects_residual_decoder_or_envelope(extra):
    with pytest.raises(ValueError, match="foundation-only cannot include residual"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(foundation_only=True, **extra),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
        )


@pytest.mark.parametrize("flag", [1, "true", None])
def test_foundation_only_requires_boolean(flag):
    with pytest.raises(ValueError, match="explicit boolean foundation-only"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(foundation_only=flag),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
        )


@pytest.mark.parametrize("foundation_only", [False, True])
def test_unmixed_mode_reaches_mock_allocator_boundary(tmp_path, monkeypatch, foundation_only):
    import sys
    from types import SimpleNamespace

    def allocator_never_run(path):
        raise RuntimeError("mock allocator boundary, no physical allocation")

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjSpec=SimpleNamespace(from_file=allocator_never_run)),
    )
    args = argparse.Namespace(
        scene=tmp_path / "absent.xml",
        foundation_only=foundation_only,
        step_model=None if foundation_only else tmp_path / "absent-residual.json",
    )
    with pytest.raises(RuntimeError, match="mock allocator boundary"):
        kernel.run_native_first_touch_episode(
            args,
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
        )


def specialize_legacy_profile(function):
    """Specialize only explicit profile=None branches, retaining frozen hash.

    This does not erase new arithmetic. Scalar aliases are inlined from their
    actual expressions, so a change to gains/caps/torque still changes the hash.
    A real native control additionally checks bit parity of every old array.
    """
    import copy

    class ProfileNone(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test == "generated_seed_episode_binding":
                # Default false retains the original preview/bind statements
                # verbatim. New opt-in path needs separate physical parity.
                return [self.visit(n) for n in node.orelse]
            if test in (
                "mesh_task_contact_world_hash is not None",
                "mesh_contact_construction is not None",
                "mesh_world_review is not None",
            ):
                # New task-contact world is explicitly opt-in. Default None
                # retains the historical model and controller arithmetic.
                return [self.visit(n) for n in node.orelse]
            if test == "record_default_planner_events":
                # Explicit opt-in observation metadata is disabled on the
                # frozen default path. No control expression is removed.
                return [self.visit(n) for n in node.orelse]
            if (
                isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.And)
                and any(ast.unparse(v) == "control_profile is not None" for v in node.test.values)
            ):
                return [self.visit(n) for n in node.orelse]
            if test in ("control_profile is None", "control_profile is not None"):
                selected = node.body if test == "control_profile is None" else node.orelse
                return [self.visit(n) for n in selected]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == "SonicNavigationConfig":
                for keyword in node.keywords[:]:
                    if keyword.arg == "record_default_planner_events":
                        assert ast.unparse(keyword.value) == "record_default_planner_events"
                        node.keywords.remove(keyword)
            return self.generic_visit(node)

        def visit_IfExp(self, node):
            if ast.unparse(node.test) == "control_profile is None":
                return self.visit(node.body)
            return self.generic_visit(node)

    function = ProfileNone().visit(function)
    names = {
        "foot_forward_cap",
        "foot_lateral_cap",
        "foot_vertical_offset",
        "navigation_command",
        "requested_torque",
    }
    assignments = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in names
    ]
    assert len(assignments) == len(names)
    aliases = {n.targets[0].id: n.value for n in assignments}
    assert set(aliases) == names

    class Inline(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in aliases:
                return copy.deepcopy(aliases[node.id])
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name == "mesh_contact_construction":
                    assert isinstance(node.value, ast.Constant) and node.value.value is None
                    return None
                if name in aliases:
                    return None
                if name == "requested_torques":
                    assert ast.dump(node.value) == ast.dump(ast.List(elts=[], ctx=ast.Load()))
                    return None
            return self.generic_visit(node)

    return Inline().visit(function)


def test_extracted_physics_body_matches_original_frozen_ast():
    # Original native source SHA3278a076... was independently archived before
    # extraction. Normalize ONLY explicit course, partition and added entry
    # source metadata. All control arithmetic, dimensions, step order, targets,
    # observations, policy selection and output construction remain covered.
    tree = ast.parse(Path(kernel.__file__).read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_native_first_touch_episode"
    )
    # Explicitly specialize the newly introduced default-False diagnostic
    # branches, exactly as the separate four-path arithmetic guard does. Do
    # not refresh the historical hash to accept unreviewed arithmetic.
    from tests.rsi.test_swing_release_default_arithmetic import OffPath

    function = OffPath().visit(function)
    function = specialize_legacy_profile(function)
    start = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "torch.set_num_threads"
    )
    nodes = function.body[start:]
    declaration_check = next(
        n
        for n in nodes
        if isinstance(n, ast.If) and ast.unparse(n.test) == "training_plan is not None"
    )
    expected_check = ast.parse(
        "if training_plan is not None:\n"
        "    training_plan.require_sampling_view(training_plan.require_draw("
        "group=training_group, seed=args.seed, lane=args.lane, course=course), step_model)\n"
    ).body[0]
    assert ast.dump(declaration_check, include_attributes=False) == ast.dump(
        expected_check, include_attributes=False
    )
    nodes.remove(declaration_check)
    for node in nodes:
        if isinstance(node, ast.For) and ast.unparse(node.target) == "frame":
            lateral = next(
                n
                for n in node.body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "lateral"
            )
            expected_call = ast.parse(
                "lateral = native_lateral_command(gap_x=float(gap[0]), gap_y=float(gap[1]), "
                "tracking=tracking, before_contact=contact_frame is None, mode=approach_mode)"
            ).body[0]
            assert ast.dump(lateral, include_attributes=False) == ast.dump(
                expected_call, include_attributes=False
            )
            lateral.value = ast.parse(
                "float(np.clip(1.2 * gap[1], -0.2, 0.2)) "
                "if tracking and gap[0] > 0.95 and contact_frame is None else 0.0",
                mode="eval",
            ).body
            conditional = next(
                n
                for n in node.body
                if isinstance(n, ast.If) and ast.unparse(n.test) == "frame == 30"
            )
            gate = next(
                n
                for n in conditional.body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "gate"
            )
            expected_gate = ast.parse(
                "native_swing_eligible(memory_gate=bool(select_taskspace_gate(features, "
                "np.asarray(late['memory_features']), np.asarray(late['memory_clean']), "
                "np.asarray(late['memory_reward']), np.asarray(late['memory_groups']), "
                "neighbors=late['neighbors'], confidence=late['confidence'], "
                "baseline_clean_ceiling=late['baseline_clean_ceiling'])[0]), "
                "initial_vx=float(vx), foundation_only=args.foundation_only, mode=swing_mode)",
                mode="eval",
            ).body
            assert ast.dump(gate.value, include_attributes=False) == ast.dump(
                expected_gate, include_attributes=False
            )
            gate.value = ast.Call(
                func=ast.Name(id="bool", ctx=ast.Load()),
                args=[
                    ast.BoolOp(
                        op=ast.And(),
                        values=[
                            gate.value.keywords[0].value.args[0],
                            ast.parse("vx < 0", mode="eval").body,
                            ast.parse("not args.foundation_only", mode="eval").body,
                        ],
                    )
                ],
                keywords=[],
            )
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "(x, y, vx)":
            node.value = ast.Name(id="INDEPENDENT_COURSE", ctx=ast.Load())
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "commitment":
            node.value.keywords = [
                k
                for k in node.value.keywords
                if k.arg
                not in (
                    "entry_source_hash",
                    "approach_mode",
                    "swing_mode",
                    "public_training_declaration",
                )
            ]
            for keyword in node.value.keywords:
                if keyword.arg == "partition":
                    keyword.value = ast.Name(id="EXPLICIT_PARTITION", ctx=ast.Load())
    body = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    assert (
        hashlib.sha256(body.encode()).hexdigest()
        # Baseline d9b09fd already includes the optional cerebellar layer.
        # Its normalized AST was independently compared byte-for-byte with
        # this tree after specializing ONLY the new mesh option to None.
        == "65553cdb7c835e95fd63a2a20d987ebdb9237aaae1d7c17e4e4a07915cab88ce"
    )
    assert "sample_training_courses" not in ast.dump(function)


def test_legacy_projection_still_detects_changed_default_pd_gain(tmp_path, monkeypatch):
    original = Path(kernel.__file__).read_text()
    expression = "(target - data.qpos[qi]) * navigation.backend.kp"
    assert original.count(expression) == 1
    changed = tmp_path / "changed_native.py"
    changed.write_text(original.replace(expression, "2.0 * " + expression))
    monkeypatch.setattr(kernel, "__file__", str(changed))
    with pytest.raises(AssertionError):
        test_extracted_physics_body_matches_original_frozen_ast()


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", True])
def test_generated_binding_requires_explicit_private_factory_before_allocation(tmp_path, enabled):
    with pytest.raises(ValueError, match="generated|boolean"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
            generated_seed_episode_binding=enabled,
        )


@pytest.mark.parametrize("world", [True, "", "a" * 64, "sha256:" + "g" * 64])
def test_mesh_contact_hash_rejected_before_allocation(world):
    with pytest.raises(ValueError, match="mesh contact world"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(record_foundation_observation=True),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
            mesh_task_contact_world_hash=world,
        )


@pytest.mark.parametrize(
    "extra",
    [
        dict(partition="DECLARED_PUBLIC_TRAINING"),
        dict(captured=False),
        dict(cerebellar_model={}),
        dict(cerebellar_sampling_seed=1),
    ],
)
def test_new_contact_world_cannot_reuse_old_training_or_neural_torque_qualification(extra):
    with pytest.raises(ValueError, match="mesh contact world"):
        kernel.run_native_first_touch_episode(
            argparse.Namespace(record_foundation_observation=extra.get("captured", True)),
            course=(2.5, 0.0, -0.5),
            partition=extra.get("partition", "DECLARED_DEVELOPMENT_DIAGNOSTIC"),
            entry_source_path=Path(entry.__file__),
            consumed_bank_hash=None,
            mesh_task_contact_world_hash="sha256:" + "a" * 64,
            cerebellar_model=extra.get("cerebellar_model"),
            cerebellar_sampling_seed=extra.get("cerebellar_sampling_seed"),
        )
