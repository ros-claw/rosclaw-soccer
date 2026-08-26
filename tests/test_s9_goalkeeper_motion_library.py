from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.skills.goalkeeper_v2.motion_library import (
    GoalkeeperMotionFamily,
    build_motiondecode_goalkeeper_library,
    load_goalkeeper_motion_library,
    load_motion_clip_frames,
)

_HASH = "sha256:" + "1" * 64


def _source(
    root: Path,
    marker: str,
    name: str,
    *,
    energy: float,
    low: bool = False,
    recovery_middle: bool = False,
    recovery_posture: str | None = None,
) -> None:
    directory = root / "samples" / marker
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    header = (
        "root_pos_x(m),root_pos_y(m),root_pos_z(m),root_rot_w,root_rot_x,"
        "root_rot_y,root_rot_z,"
        + ",".join(f"dof_{joint}(rad)" for joint in G1_DDS_JOINT_NAMES)
        + "\n"
    )
    rows = []
    frame_count = 300 if recovery_middle else 20
    for frame in range(frame_count):
        phase = frame / (frame_count - 1)
        if recovery_middle and frame < 100:
            height = 0.77
        elif recovery_middle and frame < 130:
            height = 0.77 - 0.67 * ((frame - 100) / 29.0)
        elif recovery_middle and frame < 200:
            height = 0.1 + 0.67 * ((frame - 130) / 69.0)
        elif recovery_middle:
            height = 0.77
        else:
            height = 0.1 + 0.67 * phase if low else 0.77 + 0.02 * np.sin(phase * np.pi)
        joints = energy * np.sin(phase * np.pi) * np.linspace(0.1, 1.0, 29)
        recovery_quaternion = {
            "PRONE": (2**-0.5, 0.0, 2**-0.5, 0.0),
            "SUPINE": (2**-0.5, 0.0, -(2**-0.5), 0.0),
            "LEFT_SIDE": (2**-0.5, -(2**-0.5), 0.0, 0.0),
        }.get(recovery_posture, (1.0, 0.0, 0.0, 0.0))
        quaternion = (
            recovery_quaternion
            if recovery_middle and 100 <= frame < 170
            else (1.0, 0.0, 0.0, 0.0)
        )
        row = (0.0, 0.02 * phase, height, *quaternion, *joints)
        rows.append(",".join(str(float(value)) for value in row))
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _dataset(root: Path) -> None:
    root.mkdir()
    (root / "LICENSE.md").write_text(
        "academic research; non-commercial; retain attribution\n", encoding="utf-8"
    )
    (root / "README.md").write_text("Unitree G1 motions at 120 Hz\n", encoding="utf-8")
    for index in range(2):
        _source(
            root,
            "Single_Leg_Standing",
            f"balance_{index}.csv",
            energy=0.1 + index * 0.1,
        )
        _source(
            root,
            "Lateral_Walking",
            f"lateral_{index}.csv",
            energy=0.2 + index * 0.1,
        )
    for index in range(3):
        _source(
            root,
            "Catching_Action",
            f"catch_{index}.csv",
            energy=0.3 + index * 0.1,
            low=index == 0,
        )
        _source(
            root,
            "Lie_Down_to_Stand",
            f"recovery_{index}.csv",
            energy=0.3,
            low=True,
            recovery_middle=True,
            recovery_posture=("PRONE", "SUPINE", "LEFT_SIDE")[index],
        )


def test_motion_library_is_conditioned_content_bound_and_proxy_honest(tmp_path: Path) -> None:
    root = tmp_path / "motiondecode"
    _dataset(root)
    output = tmp_path / "evidence" / "library.json"

    library = build_motiondecode_goalkeeper_library(
        dataset_root=root,
        output_path=output,
        source_checkout=tmp_path / "checkout",
        body_hash=_HASH,
    )

    assert set(GoalkeeperMotionFamily) == {clip.family for clip in library.clips}
    assert library.contains_only_proxy_motion
    assert not library.human_goalkeeper_claim_allowed
    assert not library.commercial_use_allowed
    assert library.clips_for(task="save", region="upper_left")[0].family is (
        GoalkeeperMotionFamily.HIGH_REACH_LEFT
    )
    assert library.clips_for(task="recovery", region="lower_right")[0].family is (
        GoalkeeperMotionFamily.RECOVERY
    )
    recovery = library.clips_for(task="recovery", region="center")[0]
    assert 125 <= recovery.segment_start_frame <= 135
    assert recovery.segment_end_frame < recovery.frame_count
    recovery_postures = {
        clip.recovery_posture
        for clip in library.clips_for(task="recovery", region="center")
    }
    assert recovery_postures == {
        "LEFT_SIDE",
        "PRONE",
        "SUPINE",
    }
    assert load_goalkeeper_motion_library(output, dataset_root=root) == library


def test_motion_library_mirror_swaps_left_and_right(tmp_path: Path) -> None:
    root = tmp_path / "motiondecode"
    _dataset(root)
    library = build_motiondecode_goalkeeper_library(
        dataset_root=root,
        output_path=tmp_path / "library.json",
        source_checkout=tmp_path / "checkout",
        body_hash=_HASH,
    )
    left = library.clips_for(task="shuffle", region="upper_left")[0]
    right = library.clips_for(task="shuffle", region="upper_right")[0]
    left_q, _ = load_motion_clip_frames(dataset_root=root, clip=left)
    right_q, _ = load_motion_clip_frames(dataset_root=root, clip=right)

    np.testing.assert_allclose(right_q[:, 0], left_q[:, 6])
    np.testing.assert_allclose(right_q[:, 6], left_q[:, 0])
    np.testing.assert_allclose(right_q[:, 1], -left_q[:, 7])


def test_motion_library_rejects_tampering_or_missing_license(tmp_path: Path) -> None:
    root = tmp_path / "motiondecode"
    _dataset(root)
    output = tmp_path / "library.json"
    build_motiondecode_goalkeeper_library(
        dataset_root=root,
        output_path=output,
        source_checkout=tmp_path / "checkout",
        body_hash=_HASH,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["clips"][0]["quality_score"] = 999.0
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_goalkeeper_motion_library(output)

    broken = tmp_path / "broken"
    _dataset(broken)
    (broken / "LICENSE.md").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty"):
        build_motiondecode_goalkeeper_library(
            dataset_root=broken,
            output_path=tmp_path / "broken.json",
            source_checkout=tmp_path / "checkout",
            body_hash=_HASH,
        )
