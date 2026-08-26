"""Evidence-downstream video for a passer, shooter and reactive goalkeeper."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.evidence.three_player import ThreePlayerEvidenceBundle
from rosclaw_soccer.evidence.three_player import (
    validate_three_player_evidence as _validate_evidence,
)
from rosclaw_soccer.media.trajectory_render import append_sphere, escape_filtergraph_option
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_three_player_stadium_model,
)

Array: TypeAlias = NDArray[np.generic]
_RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080)}


@dataclass(frozen=True)
class ThreePlayerVideoClip:
    clip_id: str
    title: str
    frame_count: int
    duration_sec: float
    playback_kind: str


@dataclass(frozen=True)
class ThreePlayerVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str
    renderer_hash: str
    fps: int
    width: int
    height: int
    frame_count: int
    duration_sec: float
    codec_name: str
    python_version: str
    numpy_version: str
    mujoco_version: str
    ffmpeg_version: str
    ffprobe_version: str
    clips: tuple[ThreePlayerVideoClip, ...]
    goal_contract: str
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    simultaneous_three_body_physics: bool = True
    shared_ball_state: bool = True
    source_evidence_passed: bool = True
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.three_player_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "clips": [asdict(clip) for clip in self.clips]}


@dataclass(frozen=True)
class _Frame:
    simulation_time_sec: float
    view: str


def render_three_player_showcase_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    resolution: str = "1080p",
    allow_rejected_candidate: bool = False,
) -> ThreePlayerVideoResult:
    """Render one strict shared-world trajectory plus auditable replays."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("three-player video must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("three-player video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("three-player video fps must be in [10, 60]")
    try:
        width, height = _RESOLUTIONS[resolution]
    except KeyError as error:
        raise ValueError("three-player video resolution must be 720p or 1080p") from error
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("three-player video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for three-player video export")

    bundle = _validate_evidence(
        evidence_path,
        source_checkout=checkout,
        allow_development_candidate=allow_rejected_candidate,
    )
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != bundle.report.get("body_hash"):
            raise ValueError("three-player video Body hash does not match evidence")
        import mujoco

        goal = G1TrainingGoalSpec(**dict(bundle.request["goal_spec"]))
        passer_origin = _xyz(bundle.request.get("passer_origin_m"), "passer origin")
        keeper_config = bundle.request.get("goalkeeper_config")
        if not isinstance(keeper_config, dict):
            raise ValueError("three-player goalkeeper config is missing")
        keeper_depth = _number(keeper_config.get("depth_from_goal_line_m"), "keeper depth")
        model = build_g1_three_player_stadium_model(
            asset_root.expanduser().resolve(),
            passer_origin_m=passer_origin,
            goalkeeper_origin_m=(goal.plane_x_m - keeper_depth, 0.0, 0.0),
            spec=goal,
        )
        _configure_offscreen_framebuffer(model, width=width, height=height)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        timelines, clips = _timelines(bundle, fps)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="rosclaw-soccer-three-g1-") as temp:
                labels = _write_labels(Path(temp), bundle, goal)
                process = subprocess.Popen(
                    _ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=output,
                        fps=fps,
                        width=width,
                        height=height,
                        labels=labels,
                        clips=clips,
                        source_evidence_passed=bundle.report.get("passed") is True,
                        candidate_promoted=(
                            bundle.report.get("promotion_status")
                            not in {
                                "REJECTED_DEVELOPMENT",
                                "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED",
                            }
                        ),
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("three-player ffmpeg raw-video pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        renderer=renderer,
                        bundle=bundle,
                        timelines=timelines,
                        stream=cast(BinaryIO, process.stdin),
                    )
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                code = process.wait()
                if code:
                    raise RuntimeError(f"three-player ffmpeg failed ({code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    expected_frames = sum(clip.frame_count for clip in clips)
    expected_duration = sum(clip.duration_sec for clip in clips)
    probe = _probe_video(ffprobe, output)
    if (probe["width"], probe["height"]) != (width, height):
        raise RuntimeError("encoded three-player video dimensions do not match the request")
    if probe["fps"] != fps or probe["frame_count"] != expected_frames:
        raise RuntimeError("encoded three-player video timeline does not match rendered frames")
    if abs(probe["duration_sec"] - expected_duration) > 1.0 / fps + 1e-6:
        raise RuntimeError("encoded three-player video duration does not match rendered frames")

    value = ThreePlayerVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        evidence_hash=bundle.evidence_hash,
        request_hash=bundle.request_hash,
        trajectory_hash=bundle.trajectory_hash,
        trajectory_digest=bundle.trajectory_digest,
        renderer_hash=_renderer_hash(),
        fps=fps,
        width=width,
        height=height,
        frame_count=expected_frames,
        duration_sec=probe["duration_sec"],
        codec_name=str(probe["codec_name"]),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        mujoco_version=str(mujoco.__version__),
        ffmpeg_version=_tool_version(ffmpeg),
        ffprobe_version=_tool_version(ffprobe),
        clips=clips,
        goal_contract=_goal_contract(goal),
        source_evidence_passed=bundle.report.get("passed") is True,
    )
    manifest.write_text(
        json.dumps(value.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _timelines(
    bundle: ThreePlayerEvidenceBundle, fps: int
) -> tuple[tuple[tuple[_Frame, ...], ...], tuple[ThreePlayerVideoClip, ...]]:
    result = bundle.report["result"]
    pass_time = _number(result.get("pass_contact_time_sec"), "pass contact time")
    shot_time = _number(result.get("shot_contact_time_sec"), "shot contact time")
    goalkeeper_save = result.get("goalkeeper_save_observed") is True
    outcome_time = (
        _number(result.get("goalkeeper_ball_contact_time_sec"), "goalkeeper contact time")
        if goalkeeper_save
        else shot_time
    )
    start = float(bundle.trajectory["time"][0])
    end = float(bundle.trajectory["time"][-1])
    intro = tuple(_Frame(start, "wide") for _ in range(round(1.5 * fps)))
    continuous = (
        *_segment(start, pass_time - 0.75, 1.0, "wide", fps),
        *_segment(pass_time - 0.75, pass_time + 0.35, 1.0, "pass", fps),
        *_segment(pass_time + 0.35, shot_time - 0.45, 1.0, "roll", fps),
        *_segment(shot_time - 0.45, outcome_time + 1.30, 1.0, "goal_field", fps),
        *_segment(outcome_time + 1.30, end, 1.0, "recovery_wide", fps),
    )
    pass_replay = _segment(pass_time - 0.80, pass_time + 1.25, 0.45, "pass_close", fps)
    goal_replay = _segment(shot_time - 0.70, outcome_time + 1.20, 0.38, "goal_front", fps)
    recovery_mid = shot_time + 0.5 * (end - shot_time)
    shooter_recovery = _segment(shot_time + 1.25, recovery_mid, 0.72, "recovery_shooter", fps)
    passer_recovery = _segment(recovery_mid, end, 0.72, "recovery_passer", fps)
    finale = tuple(_Frame(end, "wide") for _ in range(round(2.0 * fps)))
    timelines = (
        intro,
        continuous,
        pass_replay,
        goal_replay,
        shooter_recovery,
        passer_recovery,
        finale,
    )
    specs = (
        ("01-intro", "THREE G1 SHARED-WORLD BASELINE", "VERIFIED_POSE_HOLD"),
        (
            "02-continuous",
            "PASS → SHOT → SAVE" if goalkeeper_save else "PASS → FINISH → RECOVERY",
            "STRICT_PHYSICS_REPLAY",
        ),
        ("03-pass", "MEASURED ROLLING PASS", "INTERPOLATED_SLOW_MOTION_REPLAY"),
        (
            "04-goal",
            (
                "LEARNED PROPRIOCEPTIVE GOALKEEPER SAVE"
                if goalkeeper_save
                else "SHOT AND PROPRIOCEPTIVE GOALKEEPER"
            ),
            "INTERPOLATED_SLOW_MOTION_REPLAY",
        ),
        ("05-shooter-recovery", "SHOOTER RECOVERY", "INTERPOLATED_REVIEW_REPLAY"),
        ("06-passer-recovery", "PASSER RECOVERY", "INTERPOLATED_REVIEW_REPLAY"),
        ("07-scorecard", "STRICT SHARED-WORLD SCORECARD", "VERIFIED_FINAL_POSE_HOLD"),
    )
    clips = tuple(
        ThreePlayerVideoClip(
            clip_id=clip_id,
            title=title,
            frame_count=len(timeline),
            duration_sec=len(timeline) / fps,
            playback_kind=kind,
        )
        for timeline, (clip_id, title, kind) in zip(timelines, specs, strict=True)
    )
    return timelines, clips


def _segment(start: float, end: float, speed: float, view: str, fps: int) -> tuple[_Frame, ...]:
    if end <= start or speed <= 0.0:
        raise ValueError("three-player video segment is invalid")
    count = max(1, int(math.ceil((end - start) / speed * fps)))
    return tuple(_Frame(min(end, start + index / fps * speed), view) for index in range(count))


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    bundle: ThreePlayerEvidenceBundle,
    timelines: tuple[tuple[_Frame, ...], ...],
    stream: BinaryIO,
) -> None:
    ball_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    joint_qpos = {
        role: _joint_qpos(mujoco, model, prefix)
        for role, prefix in (("shooter", ""), ("passer", "passer_"), ("goalkeeper", "goalkeeper_"))
    }
    free_qpos = {"shooter": 0}
    for role, prefix in (("passer", "passer_"), ("goalkeeper", "goalkeeper_")):
        joint = _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
        free_qpos[role] = int(model.jnt_qposadr[joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    target = np.asarray(bundle.request["physical_scoring_target_m"], dtype=np.float64)
    result = bundle.report["result"]
    if result.get("goal_crossing_y_m") is not None:
        actual = (
            float(bundle.request["goal_spec"]["plane_x_m"]),
            float(result["goal_crossing_y_m"]),
            float(result["goal_crossing_z_m"]),
        )
    else:
        contact_time = _number(
            result.get("goalkeeper_ball_contact_time_sec"), "goalkeeper contact time"
        )
        contact_index = int(np.searchsorted(bundle.trajectory["time"], contact_time, side="left"))
        contact = bundle.trajectory["ball_pose"][contact_index, :3]
        actual = (
            float(contact[0]),
            float(contact[1]),
            float(contact[2]),
        )
    for timeline in timelines:
        for frame in timeline:
            poses = _sample(bundle.trajectory, frame.simulation_time_sec)
            data.qpos[:] = model.qpos0
            for role in ("shooter", "passer", "goalkeeper"):
                data.qpos[free_qpos[role] : free_qpos[role] + 7] = poses[f"{role}_pelvis_pose"]
                data.qpos[joint_qpos[role]] = poses[f"{role}_joint_position"]
            data.qpos[ball_qpos : ball_qpos + 7] = poses["ball_pose"]
            mujoco.mj_forward(model, data)
            _set_camera(camera, frame.view, poses, bundle)
            renderer.update_scene(data, camera=camera)
            _add_markers(
                mujoco,
                renderer.scene,
                bundle.trajectory,
                target,
                actual,
                int(poses["index"]),
            )
            stream.write(np.ascontiguousarray(renderer.render().copy()).tobytes())


def _sample(trajectory: dict[str, Array], simulation_time: float) -> dict[str, Array | int]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    upper = int(np.searchsorted(time, simulation_time, side="right"))
    if upper <= 0:
        lower = upper = 0
        ratio = 0.0
    elif upper >= len(time):
        lower = upper = len(time) - 1
        ratio = 0.0
    else:
        lower = upper - 1
        ratio = float((simulation_time - time[lower]) / (time[upper] - time[lower]))
    value: dict[str, Array | int] = {"index": upper if ratio >= 0.5 else lower}
    for role in ("passer", "shooter", "goalkeeper"):
        value[f"{role}_pelvis_pose"] = _pose(
            trajectory[f"{role}_pelvis_pose"][lower],
            trajectory[f"{role}_pelvis_pose"][upper],
            ratio,
        )
        value[f"{role}_joint_position"] = _lerp(
            trajectory[f"{role}_joint_position"][lower],
            trajectory[f"{role}_joint_position"][upper],
            ratio,
        )
    value["ball_pose"] = _pose(
        trajectory["ball_pose"][lower], trajectory["ball_pose"][upper], ratio
    )
    return value


def _set_camera(
    camera: Any,
    view: str,
    poses: dict[str, Array | int],
    bundle: ThreePlayerEvidenceBundle,
) -> None:
    ball = cast(Array, poses["ball_pose"])
    shooter = cast(Array, poses["shooter_pelvis_pose"])
    passer = cast(Array, poses["passer_pelvis_pose"])
    keeper = cast(Array, poses["goalkeeper_pelvis_pose"])
    goal_x = float(bundle.request["goal_spec"]["plane_x_m"])
    if view == "wide":
        camera.lookat[:] = (3.55, 0.0, 0.72)
        camera.distance, camera.azimuth, camera.elevation = 10.0, 91.0, -11.0
    elif view == "pass":
        camera.lookat[:] = 0.5 * (
            np.asarray(passer[:3], dtype=np.float64) + np.asarray(shooter[:3], dtype=np.float64)
        )
        camera.lookat[2] = 0.68
        camera.distance, camera.azimuth, camera.elevation = 6.15, 92.0, -8.0
    elif view == "pass_close":
        camera.lookat[:] = ball[:3] + np.asarray((0.0, 0.0, 0.40))
        camera.distance, camera.azimuth, camera.elevation = 4.65, 116.0, -6.0
    elif view == "roll":
        camera.lookat[:] = ball[:3] + np.asarray((0.0, 0.0, 0.48))
        camera.distance, camera.azimuth, camera.elevation = 5.7, 92.0, -7.0
    elif view == "goal_front":
        camera.lookat[:] = (goal_x, 0.0, 0.92)
        camera.distance, camera.azimuth, camera.elevation = 7.5, 180.0, -6.0
    elif view == "goal_field":
        # Pull the continuous shot view toward the goal centre.  A 105-degree
        # sideline angle placed the near upright across the ball/keeper path.
        camera.lookat[:] = (goal_x - 1.35, 0.20, 0.76)
        camera.distance, camera.azimuth, camera.elevation = 7.25, 142.0, -7.0
    elif view == "recovery_shooter":
        camera.lookat[:] = shooter[:3]
        camera.lookat[2] = 0.72
        camera.distance, camera.azimuth, camera.elevation = 3.7, 100.0, -7.0
    elif view == "recovery_passer":
        camera.lookat[:] = passer[:3]
        camera.lookat[2] = 0.72
        camera.distance, camera.azimuth, camera.elevation = 3.7, 82.0, -7.0
    else:
        camera.lookat[:] = 0.5 * (
            np.asarray(shooter[:3], dtype=np.float64) + np.asarray(keeper[:3], dtype=np.float64)
        )
        camera.lookat[2] = 0.74
        camera.distance, camera.azimuth, camera.elevation = 7.4, 97.0, -8.0


def _add_markers(
    mujoco: Any,
    scene: Any,
    trajectory: dict[str, Array],
    target: Array,
    actual: tuple[float, float, float],
    index: int,
) -> None:
    for angle in np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False):
        position = target + np.asarray(
            (0.02, 0.10 * math.cos(angle), 0.10 * math.sin(angle)), dtype=np.float64
        )
        append_sphere(mujoco, scene, position, 0.012, (0.16, 1.0, 0.38, 0.90))
    append_sphere(mujoco, scene, target, 0.023, (1.0, 0.82, 0.18, 0.96))
    for dy, dz in ((-0.035, -0.035), (-0.035, 0.035), (0.035, -0.035), (0.035, 0.035)):
        append_sphere(
            mujoco,
            scene,
            np.asarray((actual[0] + 0.025, actual[1] + dy, actual[2] + dz)),
            0.011,
            (0.20, 0.78, 1.0, 0.98),
        )
    start = max(0, index - 90)
    indices = np.linspace(start, index, min(24, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.03, 0.58, len(indices)), strict=True):
        append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["ball_pose"][trail_index, :3]),
            0.026,
            (0.20, 0.78, 1.0, float(alpha)),
        )


def _write_labels(
    root: Path,
    bundle: ThreePlayerEvidenceBundle,
    goal: G1TrainingGoalSpec,
) -> tuple[Path, ...]:
    result = bundle.report["result"]
    regulation = _goal_contract(goal).startswith("REGULATION_")
    goal_label = f"{goal.width_m:.2f}×{goal.height_m:.2f} m " + (
        "REGULATION GOAL" if regulation else "TRAINING GOAL · NOT REGULATION GOAL"
    )
    goalkeeper_save = result.get("goalkeeper_save_observed") is True
    schema = bundle.report.get("schema_version")
    imitation = schema in {
        "rosclaw_soccer.g1_imitation_evidence.v1",
        "rosclaw_soccer.g1_composite_imitation_evidence.v1",
        "rosclaw_soccer.g1_agility_evidence.v1",
        "rosclaw_soccer.g1_follow_through_evidence.v1",
    }
    composite_imitation = schema == "rosclaw_soccer.g1_composite_imitation_evidence.v1"
    agility_growth = schema == "rosclaw_soccer.g1_agility_evidence.v1"
    follow_through_growth = schema == "rosclaw_soccer.g1_follow_through_evidence.v1"
    parent_follow = bundle.report.get("parent_follow_through", {})
    candidate_follow = bundle.report.get("candidate_follow_through", {})
    arm_excursion_gain = _metric_gain(
        parent_follow,
        candidate_follow,
        "arm_excursion_rms_rad",
    )
    motion_energy_gain = _metric_gain(
        parent_follow,
        candidate_follow,
        "upper_body_motion_energy",
    )
    imitation_naturalness = bundle.report.get("candidate_naturalness", {})
    backward_peak = float(imitation_naturalness.get("post_contact_peak_backward_velocity_mps", 0.0))
    outcome_metric = (
        (
            f"SAVE {float(bundle.report['shot_to_block_distance_m']):.2f} m AFTER SHOT · "
            f"MIN PELVIS {float(result['goalkeeper_min_pelvis_height_m']):.3f} m"
        )
        if goalkeeper_save
        else (
            f"SHOT {float(bundle.report['shot_distance_m']):.2f} m / "
            f"{float(result['target_error_m']) * 1000:.1f} mm"
        )
    )
    outcome_replay = (
        (
            f"TRUE MUJOCO CONTACT · BALL {float(result['shot_peak_ball_speed_mps']):.2f} m/s · "
            "NO GOAL · SAFE CROSS-STEP BLOCK"
        )
        if goalkeeper_save
        else (
            f"SHOT {float(result['shot_peak_ball_speed_mps']):.2f} m/s · "
            f"ERROR {float(result['target_error_m']):.4f} m / LIMIT 0.10 m · "
            "PROPRIOCEPTIVE KEEPER"
        )
    )
    values = (
        (
            f"PASS {float(bundle.report['pass_distance_m']):.2f} m / "
            f"{float(result['pass_delivery_error_m']) * 100:.1f} cm  ·  "
            f"{outcome_metric}  ·  "
            f"KEEPER {float(result['goalkeeper_lateral_displacement_m']):.2f} m"
        ),
        (
            "PASS → SHOT → SAVE · ONE BALL · ONE CPU MUJOCO WORLD"
            if goalkeeper_save
            else "PASS → FINISH → RECOVERY · ONE BALL · ONE CPU MUJOCO WORLD"
        ),
        (
            f"ROLL {float(bundle.report['pass_speed_start_mps']):.2f}→"
            f"{float(bundle.report['pass_speed_end_mps']):.2f} m/s · "
            f"POSITIVE SPEED JUMPS {int(bundle.report['pass_speed_positive_step_count'])}"
        ),
        (
            (
                "VISIBLE FOLLOW-THROUGH · MOSAIC SOCCER TEACHER · 9/9 LOCAL BASIN"
                if follow_through_growth
                else "JOINT-GROUP AGILITY · MOTIONDECODE + OMNICONTACT · LOCAL BASIN GATE"
                if agility_growth
                else "MOTIONDECODE WHOLE BODY + OMNICONTACT TRAIN CONTACT · BOUNDED PD RESIDUAL"
                if composite_imitation
                else "MOTIONDECODE WHOLE-BODY POSITION + VELOCITY TEACHER · BOUNDED PD RESIDUAL"
            )
            if imitation
            else outcome_replay
        ),
        (
            (
                f"ARM EXCURSION +{100.0 * arm_excursion_gain:.1f}% · "
                f"FOLLOW-THROUGH ENERGY +{100.0 * motion_energy_gain:.1f}% · "
                "ENDPOINT-NEUTRAL"
                if follow_through_growth
                else "IMITATION RECOVERY · "
                "BACKWARD PEAK "
                f"{backward_peak:.3f} m/s · "
                f"ROLL {float(result['shooter_roll_peak_rad']):.3f} rad"
            )
            if imitation
            else (
                "SHOOTER RECOVERY · NO FALL · "
                f"TAIL WOBBLE {float(result['shooter_tail_wobble_index']):.5f} · "
                f"MIN PELVIS {float(result['shooter_min_pelvis_height_m']):.3f} m"
            )
        ),
        (
            "PASSER RECOVERY · NO FALL · "
            f"TAIL WOBBLE {float(result['passer_tail_wobble_index']):.5f} · "
            f"MIN PELVIS {float(result['passer_min_pelvis_height_m']):.3f} m"
        ),
        (
            f"STRICT REPLAY · SAFE SAVE DEVELOPMENT GATE · {goal_label}"
            if goalkeeper_save
            else f"STRICT REPLAY · IMITATION GROWTH GATE · {goal_label}"
            if imitation
            else f"STRICT REPLAY · KEEPER DEVELOPMENT CANDIDATE · {goal_label}"
            if bundle.report.get("passed") is not True
            else f"STRICT REPLAY PASS · {goal_label}"
        ),
    )
    paths = tuple(root / f"label-{index}.txt" for index in range(len(values)))
    for path, value in zip(paths, values, strict=True):
        path.write_text(value, encoding="utf-8")
    return paths


def _metric_gain(parent: Any, candidate: Any, name: str) -> float:
    if not isinstance(parent, dict) or not isinstance(candidate, dict):
        return 0.0
    parent_value = float(parent.get(name, 0.0))
    candidate_value = float(candidate.get(name, 0.0))
    return (candidate_value - parent_value) / max(parent_value, 1e-9)


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    width: int,
    height: int,
    labels: tuple[Path, ...],
    clips: tuple[ThreePlayerVideoClip, ...],
    source_evidence_passed: bool,
    candidate_promoted: bool,
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    title = escape_filtergraph_option(
        "ROSClaw Soccer · VISIBLE FOLLOW-THROUGH GROWTH"
        if "follow" in output.stem.lower()
        else "ROSClaw Soccer · JOINT-GROUP AGILITY GROWTH"
        if "agility" in output.stem.lower()
        else "ROSClaw Soccer · COMPOSITE IMITATION GROWTH"
        if "composite" in output.stem.lower()
        else "ROSClaw Soccer · MOTIONDECODE IMITATION GROWTH"
        if "imitation" in output.stem.lower()
        else "ROSClaw Soccer · THREE G1 SHARED-WORLD RELAY"
    )
    footer = escape_filtergraph_option(
        (
            "PASSING STRICT EVIDENCE"
            if source_evidence_passed and candidate_promoted
            else "PASSED DEVELOPMENT GATE · NOT PROMOTED"
            if source_evidence_passed
            else "REJECTED DEVELOPMENT CANDIDATE · NOT PROMOTED"
        )
        + " · CPU MUJOCO · SIM ONLY · NO PIXEL SCORING"
    )
    scale = height / 720.0
    left = round(30 * scale)
    filters = [
        f"drawbox=x=0:y=0:w=iw:h={round(118 * scale)}:color=0x030711@0.84:t=fill",
        f"drawbox=x=0:y=h-{round(64 * scale)}:w=iw:h={round(64 * scale)}:"
        "color=0x030711@0.84:t=fill",
        f"drawtext={font_option}text={title}:expansion=none:x={left}:y={round(13 * scale)}:"
        f"fontsize={round(32 * scale)}:fontcolor=white",
        f"drawtext={font_option}text={footer}:expansion=none:x={left}:"
        f"y=h-{round(42 * scale)}:fontsize={round(19 * scale)}:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for label, clip in zip(labels, clips, strict=True):
        end = offset + clip.duration_sec
        filters.append(
            f"drawtext={font_option}textfile={escape_filtergraph_option(str(label))}:"
            f"expansion=none:x={left}:y={round(62 * scale)}:fontsize={round(20 * scale)}:"
            f"fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'"
        )
        offset = end
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _configure_offscreen_framebuffer(model: Any, *, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)


def _goal_contract(goal: G1TrainingGoalSpec) -> str:
    if abs(goal.width_m - 7.32) <= 1e-9 and abs(goal.height_m - 2.44) <= 1e-9:
        return "REGULATION_7.32X2.44M_GOAL"
    return f"TRAINING_{goal.width_m:.2f}X{goal.height_m:.2f}M_GOAL"


def _joint_qpos(mujoco: Any, model: Any, prefix: str) -> Array:
    return np.asarray(
        [
            model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint_name)]
            for joint_name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )


def _id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"three-player model is missing {name}")
    return value


def _pose(left: Array, right: Array, ratio: float) -> Array:
    result: NDArray[np.float64] = np.empty(7, dtype=np.float64)
    result[:3] = _lerp(left[:3], right[:3], ratio)
    result[3:] = _slerp(left[3:], right[3:], ratio)
    return result


def _lerp(left: Array, right: Array, ratio: float) -> Array:
    start = np.asarray(left, dtype=np.float64)
    return start + ratio * (np.asarray(right, dtype=np.float64) - start)


def _slerp(left: Array, right: Array, ratio: float) -> Array:
    start = np.asarray(left, dtype=np.float64)
    end = np.asarray(right, dtype=np.float64)
    start /= np.linalg.norm(start)
    end /= np.linalg.norm(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + ratio * (end - start)
        return np.asarray(result / np.linalg.norm(result), dtype=np.float64)
    angle = float(np.arccos(dot))
    scale = float(np.sin(angle))
    return np.asarray(
        np.sin((1.0 - ratio) * angle) / scale * start + np.sin(ratio * angle) / scale * end,
        dtype=np.float64,
    )


def _xyz(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"three-player {label} is invalid")
    return (
        _number(value[0], label),
        _number(value[1], label),
        _number(value[2], label),
    )


def _number(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"three-player {label} is invalid")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"three-player {label} is non-finite")
    return converted


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _renderer_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).resolve().parents[1] / "evidence/three_player.py",
        Path(__file__).resolve().parents[1] / "world/field.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    process = subprocess.run(
        (
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    value = json.loads(process.stdout)
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise RuntimeError("ffprobe did not return exactly one three-player video stream")
    stream = streams[0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    return {
        "codec_name": str(stream["codec_name"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(round(int(numerator) / int(denominator))),
        "frame_count": int(stream["nb_read_frames"]),
        "duration_sec": float(value["format"]["duration"]),
    }


def _tool_version(executable: str) -> str:
    result = subprocess.run(
        (executable, "-version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    first = result.stdout.splitlines()[0] if result.stdout else ""
    if not first or len(first) > 240:
        raise RuntimeError("media tool returned an invalid version string")
    return first


__all__ = ["ThreePlayerVideoResult", "render_three_player_showcase_video"]
