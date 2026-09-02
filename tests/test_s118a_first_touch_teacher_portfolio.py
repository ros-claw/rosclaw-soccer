from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.training import first_touch_teacher_portfolio as portfolio

_DIRECTORIES = (
    "3.3.3.1.Short_Pass",
    "3.3.3.2.Long_Pass",
    "3.3.3.3.Shooting",
    "3.3.3.4.Ball_Control",
    "3.3.3.5.Others",
)


def _header() -> tuple[str, ...]:
    return (
        "root_pos_x(m)",
        "root_pos_y(m)",
        "root_pos_z(m)",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
        *(f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES),
    )


def _write_clip(root: Path, directory: str, *, foot: str, suffix: int) -> Path:
    category_root = root / "extracted_s118" / "3.3.3.Football" / directory
    category_root.mkdir(parents=True, exist_ok=True)
    prefix = directory.split(".")[-1]
    while True:
        path = category_root / f"BGI_{prefix}_{suffix:05d}.csv"
        relative = path.relative_to(root).as_posix()
        if portfolio._split_for_path(relative) == "train":  # noqa: SLF001
            break
        suffix += 1
    values = np.zeros((32, len(_header())), dtype=np.float64)
    values[:, 2] = 0.75
    values[:, 3] = 1.0
    phase = np.linspace(0.0, 2.0 * np.pi, values.shape[0])
    leg_start = 7 if foot == "left" else 13
    values[:, leg_start : leg_start + 6] = 0.08 * np.sin(phase)[:, None]
    np.savetxt(path, values, delimiter=",", header=",".join(_header()), comments="")
    return path


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "MotionDecode"
    root.mkdir()
    (root / "README.md").write_text("Unitree G1 data at 120 Hz\n", encoding="utf-8")
    (root / "LICENSE.md").write_text(
        "non-commercial research; attribution required\n",
        encoding="utf-8",
    )
    archive = root / "samples" / "3.3.Ball_Game_Interaction" / "3.3.3.Football.rar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"synthetic-rar-fixture")
    for directory in _DIRECTORIES:
        _write_clip(root, directory, foot="right", suffix=1)
    for index, directory in enumerate(("3.3.3.1.Short_Pass", "3.3.3.4.Ball_Control")):
        _write_clip(root, directory, foot="left", suffix=100 + index)
    return root


def test_motiondecode_audit_keeps_retention_metrics_sealed(tmp_path: Path) -> None:
    report = portfolio.audit_motiondecode_football(
        _dataset(tmp_path),
        selected_per_category_and_foot=1,
        minimum_total_clips=7,
    )

    assert report["total_clips"] == 7
    assert report["license_scope"] == "NON_COMMERCIAL_RESEARCH_ONLY"
    assert report["retention_metrics_accessed"] is False
    assert report["authority"] == "KINEMATIC_STYLE_ONLY"
    assert len(report["selected_style_teachers"]) == 4
    assert {item["active_foot"] for item in report["selected_style_teachers"]} == {
        "left",
        "right",
    }


def test_motiondecode_audit_rejects_joint_header_drift(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    clip = next((root / "extracted_s118" / "3.3.3.Football").rglob("*.csv"))
    text = clip.read_text(encoding="utf-8")
    clip.write_text(text.replace("dof_left_hip_pitch_joint(rad)", "wrong_joint"), encoding="utf-8")

    with pytest.raises(ValueError, match="joint header mismatch"):
        portfolio.audit_motiondecode_football(
            root,
            selected_per_category_and_foot=1,
            minimum_total_clips=7,
        )
