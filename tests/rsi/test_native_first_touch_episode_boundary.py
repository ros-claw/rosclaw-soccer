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
    start = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "torch.set_num_threads"
    )
    nodes = function.body[start:]
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
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "(x, y, vx)":
            node.value = ast.Name(id="INDEPENDENT_COURSE", ctx=ast.Load())
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "commitment":
            node.value.keywords = [
                k
                for k in node.value.keywords
                if k.arg not in ("entry_source_hash", "approach_mode")
            ]
            for keyword in node.value.keywords:
                if keyword.arg == "partition":
                    keyword.value = ast.Name(id="EXPLICIT_PARTITION", ctx=ast.Load())
    body = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    assert (
        hashlib.sha256(body.encode()).hexdigest()
        == "b49c6c901799b82c56e3a0bb663ffd7e30720f6ac5329e85b675ce2d99c012c2"
    )
    assert "sample_training_courses" not in ast.dump(function)
