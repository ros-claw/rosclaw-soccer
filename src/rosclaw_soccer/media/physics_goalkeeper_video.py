"""Evidence-downstream video for a promoted physics-trained goalkeeper."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_cpu_exam import _CombatTeacherRuntime, _run_episode
from rosclaw_soccer.training.goalkeeper_mjwarp import goalkeeper_world_config
from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    _build_actor_critic,
    _load_actor_critic_state,
)


@dataclass(frozen=True)
class PhysicsGoalkeeperVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    checkpoint_hash: str
    cpu_exam_report_hash: str
    cpu_exam_file_hash: str
    selected_seeds: tuple[int, ...]
    selected_second_save_count: int
    frame_count: int
    fps: int
    width: int
    height: int
    duration_sec: float
    visualization_only: bool = True
    pixels_used_for_scoring: bool = False
    promoted_sim_only: bool = True
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.physics_goalkeeper_video.v4"


def render_physics_goalkeeper_champion_video(
    *,
    checkpoint_path: Path,
    cpu_exam_path: Path,
    asset_root: Path,
    locomotion_policy_path: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    segment_count: int = 4,
    allow_rejected_development: bool = False,
    champion_decision_path: Path | None = None,
) -> PhysicsGoalkeeperVideoResult:
    """Render selected safe CPU trajectories without rescoring pixels.

    Rejected candidates require an explicit opt-in and receive a permanent
    development/not-promoted label.  This preserves useful failure analysis
    without letting selected pixels become promotion evidence.
    """

    import torch
    from torch import nn

    # MuJoCo selects its OpenGL platform at import time.  Headless rendering
    # must therefore choose EGL before importing mujoco; doing this after the
    # import silently selects GLFW and fails on hosts without DISPLAY.
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from rosclaw_soccer.world.field import build_g1_stadium_model

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    checkpoint_file = checkpoint_path.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("physics goalkeeper video must remain outside the source checkout")
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("physics goalkeeper video requires a new .mp4 path")
    if not 10 <= fps <= 60 or width < 1280 or height < 720 or not 2 <= segment_count <= 6:
        raise ValueError("physics goalkeeper video dimensions or segment count are invalid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for physics goalkeeper video")
    exam = json.loads(cpu_exam_path.expanduser().resolve().read_text(encoding="utf-8"))
    promoted = bool(exam.get("passed") and exam.get("promotion_status") == "PROMOTED_SIM_ONLY")
    champion_status = "BASELINE_EXAM_ONLY"
    if champion_decision_path is not None:
        champion_decision = json.loads(
            champion_decision_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        decision = dict(champion_decision.get("decision", {}))
        if decision.get("candidate_artifact_hash") != hash_bytes(checkpoint_file.read_bytes()):
            raise ValueError("goalkeeper Champion decision does not match checkpoint")
        champion_status = str(decision.get("status", ""))
        if champion_status not in {"REPLACE_CHAMPION", "RETAIN_PARENT_ARCHIVE_CANDIDATE"}:
            raise ValueError("goalkeeper Champion decision status is invalid")
    if not promoted and not allow_rejected_development:
        raise ValueError("physics goalkeeper video requires a promoted SIM_ONLY CPU exam")
    if not promoted and not str(exam.get("promotion_status", "")).startswith("REJECTED"):
        raise ValueError("development video requires an explicitly rejected CPU exam")
    if exam.get("checkpoint_hash") != hash_bytes(checkpoint_file.read_bytes()):
        raise ValueError("physics goalkeeper checkpoint does not match CPU exam")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    training_config = checkpoint.get("training_config", {})
    if not isinstance(training_config, dict):
        raise ValueError("physics goalkeeper video training config is invalid")
    combat_teacher = None
    combat_checkout = training_config.get("combat_teacher_checkout")
    combat_checkpoint = training_config.get("combat_teacher_checkpoint")
    if (combat_checkout is None) != (combat_checkpoint is None):
        raise ValueError("physics goalkeeper video combat provenance is incomplete")
    if combat_checkout is not None:
        from rosclaw_soccer.training.goalkeeper_combat_teacher import (
            OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE,
            OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
            load_official_goalkeeper_teacher,
        )
        from rosclaw_soccer.training.goalkeeper_mobility_option import (
            GoalkeeperMobilityOptionConfig,
        )

        teacher, teacher_report = load_official_goalkeeper_teacher(
            checkout=Path(str(combat_checkout)),
            checkpoint=Path(str(combat_checkpoint)),
            device=torch.device("cpu"),
        )
        mobility_option_enabled = bool(training_config.get("mobility_option_enabled", False))
        mobility_residual_plasticity_scale = float(
            training_config.get("mobility_residual_plasticity_scale", 0.0)
        )
        raw_waist_plasticity = training_config.get("mobility_waist_residual_plasticity_scale")
        raw_arm_plasticity = training_config.get("mobility_arm_residual_plasticity_scale")
        mobility_config = GoalkeeperMobilityOptionConfig(
            lateral_command_limit=float(
                training_config.get("mobility_lateral_command_limit", 0.75)
            ),
            recovery_command_limit=float(
                training_config.get("mobility_recovery_command_limit", 0.55)
            ),
            residual_plasticity_scale=mobility_residual_plasticity_scale,
            waist_residual_plasticity_scale=(
                None if raw_waist_plasticity is None else float(raw_waist_plasticity)
            ),
            arm_residual_plasticity_scale=(
                None if raw_arm_plasticity is None else float(raw_arm_plasticity)
            ),
            teacher_lower_body_scale=float(
                training_config.get("mobility_teacher_lower_body_scale", 0.25)
            ),
            teacher_waist_scale=float(training_config.get("mobility_teacher_waist_scale", 0.80)),
            teacher_arm_scale=float(training_config.get("mobility_teacher_arm_scale", 1.00)),
            predictive_teacher_gate_floor=float(
                training_config.get("mobility_predictive_teacher_gate_floor", 0.0)
            ),
            teacher_lower_body_target_step_rad=float(
                training_config.get("mobility_teacher_lower_body_target_step_rad", 0.08)
            ),
            teacher_lower_body_target_filter_fraction=float(
                training_config.get(
                    "mobility_teacher_lower_body_target_filter_fraction", 0.35
                )
            ),
            teacher_waist_target_step_rad=float(
                training_config.get("mobility_teacher_waist_target_step_rad", 0.05)
            ),
            teacher_waist_target_filter_fraction=float(
                training_config.get("mobility_teacher_waist_target_filter_fraction", 0.25)
            ),
            teacher_arm_target_step_rad=float(
                training_config.get("mobility_teacher_arm_target_step_rad", 0.045)
            ),
            teacher_arm_target_filter_fraction=float(
                training_config.get("mobility_teacher_arm_target_filter_fraction", 0.15)
            ),
            counter_rotation_enabled=bool(
                training_config.get("mobility_counter_rotation_enabled", False)
            ),
            anticipatory_arm_reach_enabled=bool(
                training_config.get("mobility_anticipatory_arm_reach_enabled", False)
            ),
            predictive_teacher_warmstart_enabled=bool(
                training_config.get("mobility_predictive_teacher_warmstart_enabled", False)
            ),
            teacher_recovery_latch_enabled=bool(
                training_config.get("mobility_teacher_recovery_latch_enabled", False)
            ),
            teacher_recovery_hold_sec=float(
                training_config.get("mobility_teacher_recovery_hold_sec", 0.24)
            ),
            teacher_recovery_decay_sec=float(
                training_config.get("mobility_teacher_recovery_decay_sec", 0.60)
            ),
            lateral_velocity_guard_enabled=bool(
                training_config.get("mobility_lateral_velocity_guard_enabled", False)
            ),
            substep_upper_body_guard_enabled=bool(
                training_config.get("mobility_substep_upper_body_guard_enabled", False)
            ),
            substep_upper_body_guard_onset_rad_s=float(
                training_config.get("mobility_substep_upper_body_guard_onset_rad_s", 1.80)
            ),
            substep_upper_body_guard_ceiling_rad_s=float(
                training_config.get("mobility_substep_upper_body_guard_ceiling_rad_s", 2.80)
            ),
            substep_upper_body_minimum_position_scale=float(
                training_config.get("mobility_substep_upper_body_minimum_position_scale", 0.05)
            ),
        )
        combat_teacher = _CombatTeacherRuntime(
            teacher=teacher,
            report=teacher_report,
            maximum_blend=float(training_config["maximum_combat_teacher_blend"]),
            default_qpos=np.asarray(OFFICIAL_GOALKEEPER_DEFAULT_QPOS, dtype=np.float64),
            joint_group_scale=np.asarray(
                mobility_config.teacher_group_scale
                if mobility_option_enabled
                else OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE,
                dtype=np.float64,
            ),
            mobility_option_enabled=mobility_option_enabled,
            mobility_option_config=mobility_config,
            intercept_conditioning_enabled=bool(
                training_config.get("combat_teacher_intercept_conditioning_enabled", False)
            ),
        )
    candidate = _build_actor_critic(
        torch,
        nn,
        int(checkpoint["observation_size"]),
        int(checkpoint["action_size"]),
        int(checkpoint["hidden_size"]),
    )
    _load_actor_critic_state(candidate, checkpoint["state_dict"])
    candidate.eval()
    locomotion = torch.jit.load(
        str(locomotion_policy_path.expanduser().resolve()), map_location="cpu"
    )
    locomotion.eval()
    root = asset_root.expanduser().resolve()
    model = build_g1_stadium_model(root)
    data = mujoco.MjData(model)
    exam_world = dict(exam.get("world_config", {}))
    profile = str(exam_world.get("difficulty_profile", "standard"))
    if profile not in {"standard", "match", "advanced", "elite"}:
        raise ValueError("physics goalkeeper video exam difficulty profile is invalid")
    world_config = goalkeeper_world_config(
        difficulty_profile=profile,  # type: ignore[arg-type]
        environment_count=1,
        second_shot_probability=exam_world.get("second_shot_probability", 0.75),
        shot_intent_cue_enabled=bool(exam_world.get("shot_intent_cue_enabled", False)),
        hard_shot_fraction=exam_world.get("hard_shot_fraction", 0.0),
        hard_shot_height_mode=exam_world.get("hard_shot_height_mode", "high"),
        hard_shot_flight_time_range_sec=(
            None
            if exam_world.get("hard_shot_flight_time_range_sec") is None
            else tuple(exam_world["hard_shot_flight_time_range_sec"])
        ),
    )
    if world_config.config_hash != exam.get("world_config_hash"):
        raise ValueError("physics goalkeeper video world config does not match CPU exam")
    successful: list[tuple[int, Any]] = []
    fallback: list[tuple[int, Any]] = []
    for seed in tuple(int(value) for value in exam["seeds"]):
        result = _run_episode(
            model=model,
            data=data,
            locomotion=locomotion,
            policy=candidate,
            seed=seed,
            world_config=world_config,
            combat_teacher=combat_teacher,
            record_trajectory=True,
        )
        if result.failed or result.joint_limit_violation or not result.finite_state:
            continue
        # A visually selected goalkeeper clip must contain anatomical hand
        # involvement.  Torso/hip blocks remain valid evaluation episodes but
        # may not masquerade as a dive-save demonstration.
        if result.second_save and (result.first_hand_save or result.second_hand_save):
            successful.append((seed, result))
        elif result.first_save and result.recovered:
            # Diversity fallback: retain safe high/low, left/right body blocks
            # so a sparse hand-contact set cannot collapse every promotional
            # segment onto the same low corner. Labels remain outcome-honest.
            fallback.append((seed, result))
    selected = _select_diverse(successful, fallback, segment_count)
    if len(selected) != segment_count:
        raise RuntimeError("CPU exam does not contain enough safe successful video segments")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".json")
    renderer: Any | None = None
    try:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        render_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        frames_per_segment = int(round(world_config.episode_duration_sec * fps))
        labels = tuple(_label(seed, result) for seed, result in selected)
        process = subprocess.Popen(
            _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                fps=fps,
                width=width,
                height=height,
                labels=labels,
                frames_per_segment=frames_per_segment,
                promoted=promoted,
                champion_status=champion_status,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("physics goalkeeper ffmpeg pipe is unavailable")
        try:
            for _, result in selected:
                if result.trajectory is None:
                    raise RuntimeError("selected goalkeeper trajectory is unavailable")
                _write_trajectory(
                    mujoco=mujoco,
                    model=model,
                    data=render_data,
                    renderer=renderer,
                    trajectory=result.trajectory,
                    fps=fps,
                    frame_count=frames_per_segment,
                    stream=cast(BinaryIO, process.stdin),
                )
        except BaseException:
            process.stdin.close()
            process.kill()
            process.wait()
            raise
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"physics goalkeeper ffmpeg failed: {stderr[-3000:]}")
    finally:
        if renderer is not None:
            renderer.close()
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    video_result = PhysicsGoalkeeperVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_file_hash(output),
        checkpoint_hash=hash_bytes(checkpoint_file.read_bytes()),
        cpu_exam_report_hash=str(exam["report_hash"]),
        cpu_exam_file_hash=hash_bytes(cpu_exam_path.expanduser().resolve().read_bytes()),
        selected_seeds=tuple(seed for seed, _ in selected),
        selected_second_save_count=sum(item.second_save for _, item in selected),
        frame_count=segment_count * frames_per_segment,
        fps=fps,
        width=width,
        height=height,
        duration_sec=segment_count * frames_per_segment / fps,
        promoted_sim_only=promoted,
    )
    payload = asdict(video_result)
    payload["selected_seeds"] = list(video_result.selected_seeds)
    payload["labels"] = list(labels)
    payload["cpu_exam_passed"] = promoted
    payload["cpu_exam_reasons"] = list(exam.get("reasons", ()))
    payload["development_only"] = not promoted
    payload["external_combat_teacher"] = None if combat_teacher is None else combat_teacher.report
    payload["champion_status"] = champion_status
    payload["selected_rollout_metrics"] = [
        {
            "seed": seed,
            "first_save": bool(item.first_save),
            "first_hand_save": bool(item.first_hand_save),
            "recovered": bool(item.recovered),
            "second_save": bool(item.second_save),
            "second_hand_save": bool(item.second_hand_save),
            "minimum_pelvis_height_m": float(item.minimum_pelvis_height_m),
            "maximum_root_speed_mps": float(item.maximum_root_speed_mps),
            "maximum_root_angular_speed_rad_s": float(item.maximum_root_angular_speed_rad_s),
            "maximum_lateral_displacement_m": float(item.maximum_lateral_displacement_m),
            "maximum_lateral_speed_mps": float(item.maximum_lateral_speed_mps),
            "maximum_hand_displacement_m": float(item.maximum_hand_displacement_m),
            "maximum_hand_speed_mps": float(item.maximum_hand_speed_mps),
            "second_release_lateral_error_m": float(item.second_release_lateral_error_m),
            "joint_guard_active_fraction": float(item.joint_guard_active_fraction),
        }
        for seed, item in selected
    ]
    payload["implementation_hash"] = hash_bytes(Path(__file__).read_bytes())
    payload["manifest_hash"] = hash_json(payload)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return video_result


def _select_diverse(
    successful: list[tuple[int, Any]],
    fallback: list[tuple[int, Any]],
    count: int,
) -> list[tuple[int, Any]]:
    ranked = sorted(successful, key=_video_rank)
    ranked_fallback = sorted(fallback, key=_video_rank)
    pool = ranked + ranked_fallback
    selected: list[tuple[int, Any]] = []
    categories: set[tuple[bool, bool]] = set()
    for item in pool:
        trajectory = item[1].trajectory
        if trajectory is None:
            continue
        target = np.asarray(trajectory["first_target"])
        category = (bool(target[1] >= 0.0), bool(target[2] >= 0.85))
        remaining_categories = 4 - len(categories)
        if category in categories and len(selected) + remaining_categories <= count:
            continue
        selected.append(item)
        categories.add(category)
        if len(selected) == count:
            break
    if len(selected) < count:
        used = {seed for seed, _ in selected}
        selected.extend(item for item in pool if item[0] not in used)
    return selected[:count]


def _video_rank(item: tuple[int, Any]) -> tuple[float, float, float, float]:
    """Prefer real glove saves with visible, stable upper-body motion."""

    trajectory = item[1].trajectory
    arm_excursion = 0.0
    if trajectory is not None:
        upper = np.asarray(trajectory["qpos"], dtype=np.float64)[:, 19:36]
        arm_excursion = float(np.max(np.linalg.norm(upper - upper[0], axis=1)))
    hand_save_count = int(item[1].first_hand_save) + int(item[1].second_hand_save)
    return (
        -float(hand_save_count),
        -arm_excursion,
        float(item[1].maximum_root_angular_speed_rad_s),
        -float(item[1].minimum_pelvis_height_m),
    )


def _label(seed: int, result: Any) -> str:
    if result.trajectory is None:
        return f"CPU SEED {seed}"
    first = np.asarray(result.trajectory["first_target"])
    side = "LEFT" if first[1] >= 0.0 else "RIGHT"
    level = "HIGH" if first[2] >= 0.85 else "LOW"
    if result.first_hand_save and result.second_hand_save:
        feat = "DOUBLE GLOVE SAVE + RECENTER"
    elif result.first_hand_save or result.second_hand_save:
        feat = "GLOVE SAVE + RECENTER"
    else:
        feat = "BODY BLOCK + RECOVER"
    return f"{level} {side} · {feat} · CPU SEED {seed}"


def _write_trajectory(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    trajectory: dict[str, np.ndarray],
    fps: int,
    frame_count: int,
    stream: BinaryIO,
) -> None:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Keep the complete regulation-size goal legible, but frame the keeper
    # closely enough that reach, impact absorption, and recovery remain
    # inspectable.  The subtle broadcast-camera drift is visual only: qpos is
    # always replayed from the passed CPU exam and pixels never affect scores.
    camera.lookat[:] = (4.30, 0.0, 0.88)
    camera.elevation = -8.0
    for frame in range(frame_count):
        timestamp = min(float(time[-1]), (frame + 1) / fps)
        sample = min(len(time) - 1, int(np.searchsorted(time, timestamp, side="left")))
        progress = frame / max(1, frame_count - 1)
        camera.distance = 3.85 - 0.30 * np.sin(np.pi * progress)
        camera.azimuth = 153.0 + 4.0 * np.sin(2.0 * np.pi * progress)
        data.qpos[:] = qpos[sample]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        stream.write(np.ascontiguousarray(renderer.render()).tobytes())


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    width: int,
    height: int,
    labels: tuple[str, ...],
    frames_per_segment: int,
    promoted: bool,
    champion_status: str,
) -> list[str]:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if champion_status == "REPLACE_CHAMPION":
        status = "4x A6000  |  REPLACES CHAMPION  |  SIM ONLY"
    elif champion_status == "RETAIN_PARENT_ARCHIVE_CANDIDATE":
        status = "SAFE BRANCH  |  PARENT CHAMPION RETAINED  |  SIM ONLY"
    elif promoted:
        status = "4x A6000  |  BASELINE EXAM PASSED  |  CHAMPION UNDECIDED  |  SIM ONLY"
    else:
        status = "DEVELOPMENT  |  CPU EXAM REJECTED  |  NOT PROMOTED  |  SIM ONLY"
    status_color = "0x66E0FF" if promoted else "0xFFB347"
    filters = [
        "drawbox=x=0:y=0:w=iw:h=104:color=black@0.58:t=fill",
        (
            f"drawtext=fontfile={font}:text='ROSClaw G1 · PHYSICS RL GOALKEEPER':"
            "fontcolor=white:fontsize=40:x=54:y=20"
        ),
        (
            f"drawtext=fontfile={font}:text='{status}':"
            f"fontcolor={status_color}:fontsize=24:x=56:y=66"
        ),
    ]
    segment_sec = frames_per_segment / fps
    for index, label in enumerate(labels):
        start = index * segment_sec
        end = (index + 1) * segment_sec
        safe = label.replace("'", "").replace(":", "-")
        filters.append(
            f"drawtext=fontfile={font}:text='{safe}':fontcolor=white:fontsize=30:"
            f"x=(w-text_w)/2:y=h-72:box=1:boxcolor=black@0.55:boxborderw=14:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )
    filters.append("fade=t=in:st=0:d=0.35")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = ["PhysicsGoalkeeperVideoResult", "render_physics_goalkeeper_champion_video"]
