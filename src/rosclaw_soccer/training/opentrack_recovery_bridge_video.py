"""Render a non-evidentiary showcase of frozen recovery-teacher routes."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    OpenTrackRecoveryBridgeExamConfig,
    _atomic_json,
    _file_hash,
    _run_bridge_trial,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _trial_from_dict,
    _verified_development_report,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatch,
    RecoveryEntryMatcher,
    RecoveryEntrySearchConfig,
)

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _select_showcase_routes(
    *,
    routes: tuple[RecoveryBridgeTrial, ...],
    snapshots: dict[str, RecoverySnapshot],
    maximum_cases: int,
) -> tuple[RecoveryBridgeTrial, ...]:
    if not 1 <= maximum_cases <= 6:
        raise ValueError("recovery showcase case count is invalid")
    by_hash = {route.snapshot_hash: route for route in routes}
    if set(by_hash) != set(snapshots):
        raise ValueError("recovery showcase routes and snapshots differ")
    ranked: list[RecoveryBridgeTrial] = []
    side = sorted(
        (
            route
            for route in routes
            if snapshots[route.snapshot_hash].posture_cluster in {"LEFT_SIDE", "RIGHT_SIDE"}
        ),
        key=lambda item: item.trial_hash,
    )
    if side:
        ranked.append(side[0])
    ranked.extend(
        sorted(
            routes,
            key=lambda item: (
                -item.time_dilation,
                -item.peak_root_angular_speed_rad_s,
                item.trial_hash,
            ),
        )
    )
    ranked.extend(
        sorted(
            routes,
            key=lambda item: (-item.peak_root_angular_speed_rad_s, item.trial_hash),
        )
    )
    selected: list[RecoveryBridgeTrial] = []
    for route in ranked:
        if route.snapshot_hash in {item.snapshot_hash for item in selected}:
            continue
        selected.append(route)
        if len(selected) == maximum_cases:
            break
    return tuple(selected)


def render_opentrack_recovery_bridge_video(
    *,
    opentrack_root: Path,
    teacher_policy_path: Path,
    teacher_config_path: Path,
    motion_paths: tuple[Path, ...],
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    output_dir: Path,
    maximum_cases: int = 3,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    """Render selected exact-state routes; pixels never participate in promotion."""

    root = opentrack_root.expanduser().resolve()
    policy_path = teacher_policy_path.expanduser().resolve()
    teacher_configuration_path = teacher_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if (
        not root.is_dir()
        or any(
            not path.is_file()
            for path in (
                policy_path,
                teacher_configuration_path,
                snapshot_path,
                development_path,
            )
        )
        or not _DATASET_ID.fullmatch(motion_dataset_id)
    ):
        raise FileNotFoundError("recovery showcase inputs are incomplete")
    if target.exists() or target == root or root in target.parents:
        raise ValueError("recovery showcase refuses to overwrite or enter source checkout")
    if width < 640 or height < 480 or width % 2 or height % 2:
        raise ValueError("recovery showcase resolution is invalid")
    expected_motion_root = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1"
    )
    resolved_motions = tuple(path.expanduser().resolve() for path in motion_paths)
    if not resolved_motions or any(
        path.parent != expected_motion_root for path in resolved_motions
    ):
        raise ValueError("showcase motions must belong to the declared dataset")

    development = _verified_development_report(development_path)
    if (
        development["teacher_policy_hash"] != _file_hash(policy_path)
        or development["teacher_config_hash"] != _file_hash(teacher_configuration_path)
        or development["snapshot_manifest_hash"] != _file_hash(snapshot_path)
    ):
        raise ValueError("recovery showcase inputs differ from development evidence")
    search_config = RecoveryEntrySearchConfig(**development["search_config"])
    matcher = RecoveryEntryMatcher.from_paths(resolved_motions, config=search_config)
    if matcher.library_hash != development["reference_library_hash"]:
        raise ValueError("recovery showcase reference library differs from development")
    exam_payload = dict(development["exam_config"])
    exam_payload["time_dilations"] = tuple(exam_payload["time_dilations"])
    exam_config = OpenTrackRecoveryBridgeExamConfig(**exam_payload)
    selected_payload = development["post_skill_transfer"]["development_schedule"][
        "selected_trials"
    ]
    routes = tuple(_trial_from_dict(item) for item in selected_payload)
    snapshots = {
        item.snapshot_hash: item for item in load_recovery_snapshot_corpus(snapshot_path)
    }
    selected = _select_showcase_routes(
        routes=routes, snapshots=snapshots, maximum_cases=maximum_cases
    )

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco = importlib.import_module("mujoco")
    tmj = importlib.import_module("track_mj")
    importlib.import_module(
        "track_mj.envs.g1_tracking.play.play_g1_env_tracking_general"
    )
    ort = importlib.import_module("onnxruntime")
    imageio = importlib.import_module("imageio.v2")
    image_module = importlib.import_module("PIL.Image")
    image_draw = importlib.import_module("PIL.ImageDraw")
    image_font = importlib.import_module("PIL.ImageFont")
    teacher_payload = json.loads(teacher_configuration_path.read_text(encoding="utf-8"))
    if not isinstance(teacher_payload, dict) or not isinstance(
        teacher_payload.get("env_config"), dict
    ):
        raise ValueError("OpenTrack teacher config has no environment contract")
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    if (
        tuple(item.name for item in session.get_inputs()) != ("obs",)
        or tuple(item.name for item in session.get_outputs())
        != ("continuous_actions",)
    ):
        raise ValueError("OpenTrack recovery teacher IO is incompatible")

    try:
        title_font = image_font.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27
        )
        body_font = image_font.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except OSError:
        title_font = image_font.load_default()
        body_font = image_font.load_default()

    def annotate(
        frame: np.ndarray[Any, np.dtype[np.uint8]],
        *,
        title: str,
        detail: str,
        footer: str,
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        canvas = image_module.fromarray(frame)
        draw = image_draw.Draw(canvas, "RGBA")
        draw.rectangle((0, 0, width, 80), fill=(5, 12, 24, 205))
        draw.rectangle((0, height - 42, width, height), fill=(5, 12, 24, 205))
        draw.text((22, 12), title, font=title_font, fill=(245, 250, 255, 255))
        draw.text((22, 48), detail, font=body_font, fill=(120, 220, 255, 255))
        draw.text((22, height - 33), footer, font=body_font, fill=(255, 194, 90, 255))
        return np.asarray(canvas, dtype=np.uint8)

    def make_env(match: RecoveryEntryMatch) -> Any:
        environment_config = copy.deepcopy(
            tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
        )
        environment_config.update(teacher_payload["env_config"])
        environment_config.reference_traj_config.name = {
            motion_dataset_id: [match.motion_id]
        }
        environment_config.reference_traj_config.random_start = False
        environment_config.reference_traj_config.fixed_start_frame = match.entry_frame
        environment_class = tmj.registry.get(
            "G1TrackingGeneral", "tracking_play_env_class"
        )
        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            return environment_class(
                config=environment_config,
                play_ref_motion=False,
                use_viewer=False,
                use_renderer=False,
                exp_name="rosclaw-s51-recovery-bridge-video",
            )
        finally:
            os.chdir(previous_directory)

    target.mkdir(parents=True)
    video_path = target / "s51-recovery-teacher-bridge-showcase.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=25, codec="libx264", quality=8, macro_block_size=1
    )
    rendered: list[dict[str, Any]] = []
    try:
        for case_index, route in enumerate(selected, 1):
            snapshot = snapshots[route.snapshot_hash]
            case_title = (
                f"Case {case_index}/{len(selected)} - {snapshot.posture_cluster} recovery"
            )
            case_detail = (
                f"{route.match.motion_id} frame {route.match.entry_frame} | "
                f"phase {route.time_dilation}x"
            )
            intro = np.full((height, width, 3), (10, 20, 35), dtype=np.uint8)
            intro = annotate(
                intro,
                title=case_title,
                detail=case_detail,
                footer="SIM_ONLY | privileged teacher | pixels are not promotion evidence",
            )
            for _ in range(30):
                writer.append_data(intro)

            environment = make_env(route.match)
            environment.mj_model.vis.global_.offwidth = width
            environment.mj_model.vis.global_.offheight = height
            renderer = mujoco.Renderer(environment.mj_model, height=height, width=width)
            rendered_frames = 0

            def capture(
                env: Any,
                step: int,
                handoff: bool,
                active_renderer: Any = renderer,
                active_title: str = case_title,
                active_detail: str = case_detail,
            ) -> None:
                nonlocal rendered_frames
                if step % 2:
                    return
                active_renderer.update_scene(env.mj_data, camera=0)
                frame = active_renderer.render()
                elapsed = (step + 1) * env.dt
                detail = "READY HOLD" if handoff else active_detail
                writer.append_data(
                    annotate(
                        frame,
                        title=active_title,
                        detail=f"t={elapsed:05.2f}s | {detail}",
                        footer=(
                            "SIM_ONLY | privileged reference teacher | no reset inside case"
                        ),
                    )
                )
                rendered_frames += 1

            try:
                observed, trace = _run_bridge_trial(
                    env=environment,
                    session=session,
                    snapshot=snapshot,
                    snapshot_hash=snapshot.snapshot_hash,
                    match=route.match,
                    teacher_policy_hash=_file_hash(policy_path),
                    time_dilation=route.time_dilation,
                    config=exam_config,
                    mujoco=mujoco,
                    frame_callback=capture,
                )
            finally:
                renderer.close()
                environment.close()
            result_frame = np.full(
                (height, width, 3),
                (10, 50, 35) if observed.succeeded else (70, 15, 20),
                dtype=np.uint8,
            )
            result_frame = annotate(
                result_frame,
                title="RECOVERED TO STABLE READY" if observed.succeeded else "RECOVERY FAILED",
                detail=(
                    f"stable={observed.final_stable_sec:.2f}s | "
                    f"peak angular={observed.peak_root_angular_speed_rad_s:.2f} rad/s"
                ),
                footer="Training bridge result - not a deployable proprioceptive policy",
            )
            for _ in range(30):
                writer.append_data(result_frame)
            rendered.append(
                {
                    "case_index": case_index,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "environment_index": snapshot.environment_index,
                    "posture_cluster": snapshot.posture_cluster,
                    "fixed_development_trial_hash": route.trial_hash,
                    "observed_trial": observed.to_dict()
                    | {"trial_hash": observed.trial_hash},
                    "trace_summary": trace,
                    "rendered_frame_count": rendered_frames + 60,
                }
            )
    finally:
        writer.close()

    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_recovery_bridge_video.v1",
        "development_report_hash": development["report_hash"],
        "teacher_policy_hash": _file_hash(policy_path),
        "teacher_config_hash": _file_hash(teacher_configuration_path),
        "reference_library_hash": matcher.library_hash,
        "selected_cases": rendered,
        "video_file": video_path.name,
        "video_hash": _file_hash(video_path),
        "resolution": [width, height],
        "fps": 25,
        "render_backend": "mujoco_cpu_egl",
        "pixels_used_for_promotion": False,
        "promotion_eligible": False,
        "claim_boundary": "VISUALIZATION_ONLY_NOT_EVIDENCE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-policy", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--motion-path", required=True, action="append", type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-cases", default=3, type=int)
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=720, type=int)
    args = parser.parse_args()
    report = render_opentrack_recovery_bridge_video(
        opentrack_root=args.opentrack_root,
        teacher_policy_path=args.teacher_policy,
        teacher_config_path=args.teacher_config,
        motion_paths=tuple(args.motion_path),
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        output_dir=args.output_dir,
        maximum_cases=args.maximum_cases,
        width=args.width,
        height=args.height,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "video_hash": report["video_hash"],
                "selected_cases": len(report["selected_cases"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_opentrack_recovery_bridge_video"]
