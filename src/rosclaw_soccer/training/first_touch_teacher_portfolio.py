"""Build a content-bound, read-only First Touch teacher portfolio.

MotionDecode supplies G1 kinematic style, PAiD supplies a moving-ball task
reference, and RoboNaldo remains the frozen whole-body execution prior.  None
of these assets may promote itself or command hardware.  Raw data and the
generated report must remain outside the source checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_ROOT_COLUMNS = (
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
)
_EXPECTED_COLUMNS = _ROOT_COLUMNS + tuple(f"dof_{name}(rad)" for name in G1_DDS_JOINT_NAMES)
_CATEGORY_BY_DIRECTORY = {
    "3.3.3.1.Short_Pass": "short_pass",
    "3.3.3.2.Long_Pass": "long_pass",
    "3.3.3.3.Shooting": "shooting",
    "3.3.3.4.Ball_Control": "ball_control",
    "3.3.3.5.Others": "others",
}
_RELEVANT_CATEGORIES = frozenset(("ball_control", "short_pass"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_head_optional(checkout: Path) -> str | None:
    try:
        return _git_head(checkout)
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class MotionDecodeClip:
    relative_path: str
    source_hash: str
    category: str
    split: str
    frame_count: int
    fps: float = 120.0

    def __post_init__(self) -> None:
        if (
            not self.relative_path
            or not self.source_hash.startswith("sha256:")
            or self.category not in _CATEGORY_BY_DIRECTORY.values()
            or self.split not in {"train", "retention"}
            or self.frame_count < 2
            or self.fps != 120.0
        ):
            raise ValueError("MotionDecode clip contract is invalid")


@dataclass(frozen=True)
class FirstTouchStyleTeacher:
    relative_path: str
    source_hash: str
    category: str
    active_foot: str
    frame_count: int
    reference_frame: int
    active_leg_p99_rad_s: float
    support_leg_p95_rad_s: float
    waist_p95_rad_s: float
    minimum_root_height_m: float
    quaternion_norm_maximum_error: float
    selection_score: float

    def __post_init__(self) -> None:
        values = (
            self.active_leg_p99_rad_s,
            self.support_leg_p95_rad_s,
            self.waist_p95_rad_s,
            self.minimum_root_height_m,
            self.quaternion_norm_maximum_error,
            self.selection_score,
        )
        if (
            not self.relative_path
            or not self.source_hash.startswith("sha256:")
            or self.category not in _RELEVANT_CATEGORIES
            or self.active_foot not in {"left", "right"}
            or not 0 <= self.reference_frame < self.frame_count
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
        ):
            raise ValueError("First Touch style teacher contract is invalid")


def _split_for_path(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode()).digest()
    return "retention" if digest[0] % 5 == 0 else "train"


def _frame_count_and_header(path: Path) -> tuple[int, tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"MotionDecode clip is empty: {path}") from exc
        frame_count = sum(1 for row in reader if row)
    return frame_count, header


def _style_teacher(path: Path, clip: MotionDecodeClip) -> FirstTouchStyleTeacher:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (clip.frame_count, len(_EXPECTED_COLUMNS)) or not np.isfinite(values).all():
        raise ValueError(f"MotionDecode clip has invalid numeric data: {path}")
    quaternion_error = float(np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1.0)))
    if quaternion_error > 0.03:
        raise ValueError(f"MotionDecode clip has non-unit root quaternions: {path}")
    joint_velocity = np.gradient(values[:, 7:], 1.0 / clip.fps, axis=0)
    left = np.linalg.norm(joint_velocity[:, 0:6], axis=1)
    right = np.linalg.norm(joint_velocity[:, 6:12], axis=1)
    # One-sided gradients at clip boundaries often encode a cut, not an
    # athletic event.  Keep a contact-sized context window on both sides so a
    # selected reference can later provide approach and successor targets.
    margin = min(round(0.25 * clip.fps), max(1, (clip.frame_count - 3) // 4))
    interior = slice(margin, clip.frame_count - margin)
    left_peak = float(np.percentile(left[interior], 99.0))
    right_peak = float(np.percentile(right[interior], 99.0))
    active_foot = "left" if left_peak > right_peak else "right"
    active = left if active_foot == "left" else right
    support = right if active_foot == "left" else left
    reference_frame = margin + int(np.argmax(active[interior]))
    active_p99 = float(np.percentile(active[interior], 99.0))
    support_p95 = float(np.percentile(support[interior], 95.0))
    waist_p95 = float(np.percentile(np.linalg.norm(joint_velocity[interior, 12:15], axis=1), 95.0))
    minimum_height = float(np.min(values[:, 2]))
    score = active_p99 / (1.0 + support_p95 + waist_p95)
    return FirstTouchStyleTeacher(
        relative_path=clip.relative_path,
        source_hash=clip.source_hash,
        category=clip.category,
        active_foot=active_foot,
        frame_count=clip.frame_count,
        reference_frame=reference_frame,
        active_leg_p99_rad_s=active_p99,
        support_leg_p95_rad_s=support_p95,
        waist_p95_rad_s=waist_p95,
        minimum_root_height_m=minimum_height,
        quaternion_norm_maximum_error=quaternion_error,
        selection_score=score,
    )


def audit_motiondecode_football(
    dataset_root: Path,
    *,
    selected_per_category_and_foot: int = 6,
    minimum_total_clips: int = 1000,
) -> dict[str, Any]:
    """Audit the complete football archive and select train-only style teachers."""

    root = dataset_root.expanduser().resolve()
    readme = root / "README.md"
    license_path = root / "LICENSE.md"
    archive = root / "samples" / "3.3.Ball_Game_Interaction" / "3.3.3.Football.rar"
    football_root = root / "extracted_s118" / "3.3.3.Football"
    if (
        not readme.is_file()
        or not license_path.is_file()
        or not archive.is_file()
        or not football_root.is_dir()
    ):
        raise ValueError(
            "MotionDecode README, license, football archive, and extracted root are required"
        )
    readme_text = readme.read_text(encoding="utf-8")
    license_text = license_path.read_text(encoding="utf-8")
    if "120 Hz" not in readme_text or "Unitree G1" not in readme_text:
        raise ValueError("MotionDecode source lacks its declared G1/120 Hz contract")
    if "non-commercial" not in license_text or "attribution" not in license_text:
        raise ValueError("MotionDecode non-commercial attribution terms are not verifiable")
    if not 1 <= selected_per_category_and_foot <= 16 or minimum_total_clips < 1:
        raise ValueError("MotionDecode audit selection bounds are invalid")

    clips: list[MotionDecodeClip] = []
    style_candidates: list[FirstTouchStyleTeacher] = []
    for directory_name, category in _CATEGORY_BY_DIRECTORY.items():
        category_root = football_root / directory_name
        if not category_root.is_dir():
            raise ValueError(f"MotionDecode football category is missing: {directory_name}")
        for path in sorted(category_root.glob("*.csv")):
            frame_count, header = _frame_count_and_header(path)
            if header != _EXPECTED_COLUMNS:
                raise ValueError(f"MotionDecode G1 joint header mismatch: {path}")
            relative_path = path.relative_to(root).as_posix()
            clip = MotionDecodeClip(
                relative_path=relative_path,
                source_hash=_hash_file(path),
                category=category,
                split=_split_for_path(relative_path),
                frame_count=frame_count,
            )
            clips.append(clip)
            if category in _RELEVANT_CATEGORIES and clip.split == "train":
                style_candidates.append(_style_teacher(path, clip))
    if len(clips) < minimum_total_clips:
        raise ValueError("MotionDecode football extraction is incomplete")
    if len({clip.relative_path for clip in clips}) != len(clips):
        raise ValueError("MotionDecode football inventory contains duplicate paths")

    selected: list[FirstTouchStyleTeacher] = []
    for category in sorted(_RELEVANT_CATEGORIES):
        for foot in ("left", "right"):
            pool = [
                item
                for item in style_candidates
                if item.category == category
                and item.active_foot == foot
                and item.minimum_root_height_m >= 0.62
                and 0.50 <= item.active_leg_p99_rad_s <= 30.0
                and item.support_leg_p95_rad_s <= 8.0
                and item.waist_p95_rad_s <= 4.0
            ]
            pool.sort(key=lambda item: (-item.selection_score, item.source_hash))
            if len(pool) < selected_per_category_and_foot:
                raise ValueError(f"MotionDecode lacks qualified {category}/{foot} style teachers")
            selected.extend(pool[:selected_per_category_and_foot])

    category_counts = {
        category: sum(clip.category == category for clip in clips)
        for category in _CATEGORY_BY_DIRECTORY.values()
    }
    split_counts = {
        split: sum(clip.split == split for clip in clips) for split in ("train", "retention")
    }
    inventory = [asdict(clip) for clip in clips]
    return {
        "dataset_id": "MotionDecode",
        "dataset_revision": _git_head_optional(root),
        "readme_hash": _hash_file(readme),
        "license_hash": _hash_file(license_path),
        "football_archive_hash": _hash_file(archive),
        "license_scope": "NON_COMMERCIAL_RESEARCH_ONLY",
        "attribution_required": True,
        "fps": 120.0,
        "joint_names": list(G1_DDS_JOINT_NAMES),
        "total_clips": len(clips),
        "total_frames": sum(clip.frame_count for clip in clips),
        "total_duration_sec": sum(clip.frame_count for clip in clips) / 120.0,
        "category_counts": category_counts,
        "split_counts": split_counts,
        "split_rule": "sha256(relative_path)[0] % 5 == 0 -> retention",
        "retention_metrics_accessed": False,
        "inventory_hash": hash_json(inventory),
        "selected_style_teachers": [asdict(item) for item in selected],
        "selected_style_teacher_hash": hash_json([asdict(item) for item in selected]),
        "task_labels_present": False,
        "ball_state_present": False,
        "contact_labels_present": False,
        "authority": "KINEMATIC_STYLE_ONLY",
    }


def build_first_touch_teacher_portfolio(
    *,
    motiondecode_root: Path,
    paid_root: Path,
    paid_audit_report: Path,
    robonaldo_asset_root: Path,
    source_checkout: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind complementary teacher roles without granting any motor authority."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("teacher portfolio output must be new and outside the checkout")
    paid = paid_root.expanduser().resolve()
    paid_policy = paid / "ckp" / "policy_30000.onnx"
    paid_license = paid / "LICENSE.md"
    audit_path = paid_audit_report.expanduser().resolve()
    if not paid_policy.is_file() or not paid_license.is_file() or not audit_path.is_file():
        raise ValueError("PAiD policy, license, and moving-ball audit are required")
    paid_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_metrics = {"num_trials", "first_contact_rate", "success_goal_rate"}
    if not required_metrics.issubset(paid_audit):
        raise ValueError("PAiD moving-ball audit lacks required metrics")
    qualification = qualify_g1_assets(robonaldo_asset_root)
    qualification.require_eligible()
    motiondecode = audit_motiondecode_football(motiondecode_root)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.first_touch_teacher_portfolio.v1",
        "task": "soccer.first_touch",
        "motiondecode": motiondecode,
        "paid": {
            "repository_commit": _git_head_optional(paid),
            "policy_hash": _hash_file(paid_policy),
            "license_hash": _hash_file(paid_license),
            "license_scope": "CC_BY_NC_4_0_RESEARCH_ONLY",
            "audit_report_hash": _hash_file(audit_path),
            "audit_metrics": {
                key: paid_audit[key]
                for key in ("num_trials", "first_contact_rate", "success_goal_rate")
            },
            "authority": "MOVING_BALL_TASK_TEACHER_ONLY",
        },
        "robonaldo": {
            "body_hash": qualification.body_hash,
            "kick_prior_hash": qualification.kick_prior_hash,
            "motion_hash": qualification.motion_hash,
            "backend_commit": qualification.backend_commit,
            "authority": "FROZEN_WHOLE_BODY_PRIOR",
        },
        "role_separation": {
            "MotionDecode": "pose and coordination style; no ball/contact truth",
            "PAiD": "moving-ball task proposal; not a First Touch champion",
            "RoboNaldo": "qualified execution prior; frozen during acquisition",
        },
        "provenance": {
            "source_commit": _git_head(checkout),
            "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
        },
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "promotion_eligible": False,
            "commercial_promotion_eligible": False,
            "raw_data_copied_into_checkout": False,
            "hardware_command_sent": False,
        },
    }
    report["portfolio_hash"] = hash_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motiondecode-root", type=Path, required=True)
    parser.add_argument("--paid-root", type=Path, required=True)
    parser.add_argument("--paid-audit-report", type=Path, required=True)
    parser.add_argument("--robonaldo-asset-root", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_first_touch_teacher_portfolio(
        motiondecode_root=args.motiondecode_root,
        paid_root=args.paid_root,
        paid_audit_report=args.paid_audit_report,
        robonaldo_asset_root=args.robonaldo_asset_root,
        source_checkout=args.source_checkout,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "FirstTouchStyleTeacher",
    "MotionDecodeClip",
    "audit_motiondecode_football",
    "build_first_touch_teacher_portfolio",
]
