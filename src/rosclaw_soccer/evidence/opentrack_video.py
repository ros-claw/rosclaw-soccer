"""High-resolution physical video renderer for a gated OpenTrack residual."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.evidence.opentrack_exam import (
    OpenTrackEpisodeSpec,
    load_opentrack_exam_plan,
)
from rosclaw_soccer.sim.contracts import hash_json


def render_opentrack_residual_videos(
    *,
    opentrack_root: Path,
    candidate_policy_path: Path,
    parent_policy_path: Path,
    source_config_path: Path,
    episodes: tuple[OpenTrackEpisodeSpec, ...],
    residual_scale: float,
    output_dir: Path,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    """Render policy/reference split screens from the same CPU-MuJoCo rollout."""

    root = opentrack_root.expanduser().resolve()
    candidate_path = candidate_policy_path.expanduser().resolve()
    parent_path = parent_policy_path.expanduser().resolve()
    config_path = source_config_path.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    if not root.is_dir() or not all(
        path.is_file() for path in (candidate_path, parent_path, config_path)
    ):
        raise FileNotFoundError("OpenTrack video inputs do not exist")
    if target == root or root in target.parents:
        raise ValueError("OpenTrack physical videos must remain outside the source checkout")
    if target.exists():
        raise ValueError("OpenTrack video renderer refuses to overwrite an output directory")
    if not episodes or len({item.episode_id for item in episodes}) != len(episodes):
        raise ValueError("OpenTrack video episodes must be a non-empty unique set")
    if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
        raise ValueError("OpenTrack video residual scale must be in [0, 1]")
    if width < 640 or height < 480:
        raise ValueError("OpenTrack evidence video resolution is too small")
    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking_adapter.play.play_g1_env_tracking_general")
    ort = importlib.import_module("onnxruntime")
    mujoco = importlib.import_module("mujoco")
    imageio = importlib.import_module("imageio.v2")
    candidate = ort.InferenceSession(str(candidate_path), providers=["CPUExecutionProvider"])
    parent = ort.InferenceSession(str(parent_path), providers=["CPUExecutionProvider"])
    if {item.name for item in candidate.get_inputs()} != {"obs", "history"}:
        raise ValueError("OpenTrack video candidate has no history adapter input")
    if tuple(item.name for item in parent.get_inputs()) != ("obs",):
        raise ValueError("OpenTrack video parent has an incompatible input")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("env_config"), dict):
        raise ValueError("OpenTrack video source config has no env_config")

    target.mkdir(parents=True)
    rendered: list[dict[str, Any]] = []
    old_cwd = Path.cwd()
    try:
        os.chdir(root)
        for episode in episodes:
            env_cfg = copy.deepcopy(
                tmj.registry.get("G1TrackingGeneral", "tracking_adapter_config").env_config
            )
            env_cfg.update(config["env_config"])
            env_cfg.reference_traj_config.name = {episode.dataset_id: [episode.motion_id]}
            env_cfg.reference_traj_config.random_start = False
            env_cfg.reference_traj_config.fixed_start_frame = episode.start_frame
            env_class = tmj.registry.get("G1TrackingGeneral", "tracking_adapter_play_env_class")
            env = env_class(
                config=env_cfg,
                play_ref_motion=False,
                use_viewer=False,
                use_renderer=False,
                exp_name="sealed-residual-video",
            )
            video_path = target / f"{episode.episode_id}.mp4"
            try:
                state = env.reset()
                env.mj_model.vis.global_.offwidth = width
                env.mj_model.vis.global_.offheight = height
                sim_renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
                ref_renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
                writer = imageio.get_writer(
                    str(video_path), fps=50, codec="libx264", quality=8, macro_block_size=1
                )
                available = int(env.th.len_trajectory(0)) - episode.start_frame - 2
                steps = min(episode.max_steps, available)
                try:
                    for _ in range(steps):
                        obs = np.asarray(state.obs["state"], dtype=np.float32).reshape(1, -1)
                        history_len = int(env_cfg.history_len)
                        history = np.asarray(state.obs["history_state"], dtype=np.float32).reshape(
                            history_len, -1
                        )
                        history = history.swapaxes(-1, -2)[None, ...]
                        candidate_action = candidate.run(
                            ["continuous_actions"], {"obs": obs, "history": history}
                        )[0][0]
                        parent_action = parent.run(["continuous_actions"], {"obs": obs})[0][0]
                        action = parent_action + residual_scale * (candidate_action - parent_action)
                        state = env.step(state, action)
                        sim_renderer.update_scene(env.mj_data, camera=0)
                        ref_renderer.update_scene(env.ref_mj_data, camera=0)
                        writer.append_data(
                            np.concatenate([sim_renderer.render(), ref_renderer.render()], axis=1)
                        )
                finally:
                    writer.close()
                    sim_renderer.close()
                    ref_renderer.close()
            finally:
                env.close()
            rendered.append(
                {
                    "episode_id": episode.episode_id,
                    "suite_id": episode.suite_id,
                    "video_file": video_path.name,
                    "video_hash": _file_hash(video_path),
                    "frame_count": steps,
                    "duration_sec": steps / 50.0,
                    "resolution": [width * 2, height],
                }
            )
    finally:
        os.chdir(old_cwd)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_residual_video.v1",
        "candidate_policy_hash": _file_hash(candidate_path),
        "parent_policy_hash": _file_hash(parent_path),
        "source_config_hash": _file_hash(config_path),
        "residual_scale": residual_scale,
        "render_backend": "mujoco_cpu_egl",
        "videos": rendered,
        "pixels_used_for_promotion": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a gated OpenTrack residual")
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--candidate-policy-path", required=True, type=Path)
    parser.add_argument("--parent-policy-path", required=True, type=Path)
    parser.add_argument("--source-config-path", required=True, type=Path)
    parser.add_argument("--plan-path", required=True, type=Path)
    parser.add_argument("--episode-id", required=True, action="append")
    parser.add_argument("--residual-scale", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=720, type=int)
    args = parser.parse_args()
    plan = load_opentrack_exam_plan(args.plan_path)
    selected = tuple(item for item in plan.episodes if item.episode_id in args.episode_id)
    if {item.episode_id for item in selected} != set(args.episode_id):
        raise ValueError("OpenTrack video requested an unknown episode")
    report = render_opentrack_residual_videos(
        opentrack_root=args.opentrack_root,
        candidate_policy_path=args.candidate_policy_path,
        parent_policy_path=args.parent_policy_path,
        source_config_path=args.source_config_path,
        episodes=selected,
        residual_scale=args.residual_scale,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({"report_hash": report["report_hash"], "videos": len(selected)}))


if __name__ == "__main__":
    main()


__all__ = ["render_opentrack_residual_videos"]
