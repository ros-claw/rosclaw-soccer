"""AST projection of the OFF path, not a physical noninterference certificate."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

FLAG = "suppress_postcontact_swing_release"
MPC_FLAG = "use_contact_exit_torque_mpc"
COMPLIANCE_FLAG = "use_directional_contact_compliance"
SURFACE_FLAG = "use_mesh_surface_swing"
STUDENT_FLAG = "reference_student_model"
PULSE_FLAG = "planner_navigation_pulse"
IK_PULSE_FLAG = "original_ik_planner_pulse"
HEADING_FLAG = "foot_heading_control"


def canonical(node):
    if isinstance(node, ast.AST):
        return [
            type(node).__name__,
            [(k, canonical(v)) for k, v in ast.iter_fields(node) if k != "type_params" or v],
        ]
    if isinstance(node, list):
        return [canonical(n) for n in node]
    return node


def expression(source):
    return ast.dump(ast.parse(source, mode="eval").body, include_attributes=False)


class OffPath(ast.NodeTransformer):
    def visit_arguments(self, node):
        for argument, default in zip(node.kwonlyargs, node.kw_defaults, strict=True):
            if argument.arg in {STUDENT_FLAG, PULSE_FLAG, IK_PULSE_FLAG, HEADING_FLAG}:
                assert isinstance(default, ast.Constant) and default.value is None
        pairs = [
            (a, d)
            for a, d in zip(node.kwonlyargs, node.kw_defaults, strict=True)
            if a.arg
            not in {
                FLAG,
                MPC_FLAG,
                COMPLIANCE_FLAG,
                SURFACE_FLAG,
                STUDENT_FLAG,
                PULSE_FLAG,
                IK_PULSE_FLAG,
                HEADING_FLAG,
            }
        ]
        node.kwonlyargs = [p[0] for p in pairs]
        node.kw_defaults = [p[1] for p in pairs]
        return self.generic_visit(node)

    def visit_If(self, node):
        known = {
            "foot_heading_control is not None",
            "('foot_heading_contact_probe' in report) != (foot_heading_control is not None)",
            "'foot_heading_contact_probe' in commitment",
            "'foot_heading_contact_probe' in commitment or "
            "any(name.startswith('foot_heading_') for name in trace)",
            "original_ik_planner_pulse is not None",
            "'original_ik_pre_planner_pulse' in commitment",
            "('original_ik_pre_planner_pulse' in report) != "
            "(original_ik_planner_pulse is not None)",
            "'original_ik_pre_planner_pulse' in commitment or "
            "any(name.startswith('ik_planner_pulse_') for name in trace)",
            "planner_navigation_pulse is not None",
            "'pre_planner_navigation_pulse' in commitment",
            "('pre_planner_navigation_pulse' in report) != (planner_navigation_pulse is not None)",
            "'pre_planner_navigation_pulse' in commitment or "
            "any(name.startswith('planner_pulse_') for name in trace)",
            "reference_student_model is not None",
            "control_profile is not None and reference_student_model is not None",
            "'reference_student_model' in commitment",
            "('reference_student_model' in report) != (reference_student_model is not None)",
            "'reference_student_model' in commitment or "
            "any(name.startswith('reference_student_') for name in trace)",
            FLAG,
            f"type({FLAG}) is not bool",
            f"('swing_release_ablation' in report) != {FLAG}",
            "'swing_release_ablation' in commitment",
            MPC_FLAG,
            f"type({MPC_FLAG}) is not bool",
            f"('contact_exit_torque_mpc' in report) != {MPC_FLAG}",
            "'contact_exit_torque_mpc' in commitment",
            COMPLIANCE_FLAG,
            f"type({COMPLIANCE_FLAG}) is not bool",
            f"('directional_contact_compliance' in report) != {COMPLIANCE_FLAG}",
            "'directional_contact_compliance' in commitment",
            SURFACE_FLAG,
            f"type({SURFACE_FLAG}) is not bool",
            f"('mesh_surface_swing' in report) != {SURFACE_FLAG}",
            "'mesh_surface_swing' in commitment",
            "('mesh_surface_swing' in commitment) != "
            "any(name.startswith('mesh_surface_') for name in trace)",
        }
        if ast.dump(node.test, include_attributes=False) in {expression(s) for s in known}:
            result = []
            for child in node.orelse:
                visited = self.visit(child)
                if isinstance(visited, list):
                    result.extend(visited)
                elif visited is not None:
                    result.append(visited)
            return result
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [
            k
            for k in node.keywords
            if not (
                k.arg
                in {
                    FLAG,
                    MPC_FLAG,
                    COMPLIANCE_FLAG,
                    SURFACE_FLAG,
                    STUDENT_FLAG,
                    PULSE_FLAG,
                    IK_PULSE_FLAG,
                    HEADING_FLAG,
                }
                and isinstance(k.value, ast.Name)
                and k.value.id == k.arg
            )
        ]
        return self.generic_visit(node)


@pytest.mark.parametrize(
    "filename,function,expected",
    [
        (
            "native_first_touch_episode.py",
            "run_native_first_touch_episode",
            "612f6b5d7564e0f0435f9f87fe0c3e9702bee64b8ff5b7542162b9d9d6f19311",
        ),
        (
            "first_touch_campaign_worker.py",
            "qualified_episode",
            "bb0330d7f93110b930be45c177a696188e2cdc9492b4018aa793e29aed4df31c",
        ),
        (
            "cpu_motor_transfer_evidence.py",
            "validate_cpu_foundation_capture",
            "5786657a2b152aa467aaaf08b0878c9bed7198a7fc7e1d088db5215393b78f58",
        ),
        (
            "cpu_motor_transfer_evidence.py",
            "audit_cpu_transfer",
            "9ac9a5c8052d4a143e95e635b8df94ab609e64f8c8b2512d5d1cbe67d9ab428d",
        ),
    ],
)
def test_off_path_retains_original_execution_arithmetic(filename, function, expected):
    root = Path(__file__).resolve().parents[2] / "src/rosclaw_soccer/rsi"
    node = next(
        n
        for n in ast.parse((root / filename).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == function
    )
    projected = OffPath().visit(node)
    encoded = json.dumps(
        canonical(projected), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(encoded.encode()).hexdigest() == expected
