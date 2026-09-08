"""Test the actual kick network under rigid pitch-frame changes, without motion."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.mujoco_primitives import load_robonaldo
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def verify(*, asset_root: Path, output: Path, warmstart: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    state_type, output_type, kick_type, _ = load_robonaldo(qualification.asset_root)
    observations, actions = [], []
    yaws = (0.0, math.pi, math.pi / 2, -math.pi / 2, 0.7)
    for yaw in yaws:
        state, policy_output = state_type(29), output_type(29)
        with contextlib.redirect_stdout(io.StringIO()):
            policy = kick_type(state, policy_output)
        rotation = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        origin = np.asarray([3 * math.sin(yaw), 2 * math.cos(yaw), 0.0])
        quaternion = np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        state.q = policy.default_q_mj.copy()
        state.dq = np.zeros(29)
        state.pelvis_quat_w = quaternion.copy()
        state.torso_quat_w = quaternion.copy()
        state.pelvis_pos_w = origin + rotation @ np.asarray([0.0, 0.0, 0.75])
        state.torso_pos_w = origin + rotation @ np.asarray([0.0, 0.0, 1.0])
        state.ball_pos_w = origin + rotation @ np.asarray([0.55, -0.12, 0.115])
        state.root_ang_vel_b = np.asarray([0.03, -0.01, 0.02])
        state.ball_valid = True
        policy.target_pos_w = origin + rotation @ np.asarray([5.0, 0.5, 1.2])
        with contextlib.redirect_stdout(io.StringIO()):
            policy.enter()
        if warmstart:
            from rosclaw_soccer.providers.g1.kick_warmstart import prepare_kick_handoff

            policy.time_step = policy.WARMUP_STEPS + 205
            prepare_kick_handoff(policy, entry_frame=205)
        obs_rows, action_rows = [], []
        for frame in (205, 206, 207, 208, 209, 215, 225):
            policy.time_step = policy.WARMUP_STEPS + frame
            obs = policy._build_obs()
            if obs.shape != (547,) or not np.all(np.isfinite(obs)):
                raise ValueError("kick observation contract changed")
            action = policy.ort_session.run(["actions"], {"obs": obs[None, :]})[0][0]
            if action.shape != (29,) or not np.all(np.isfinite(action)):
                raise ValueError("kick action contract changed")
            policy.last_action_il = np.clip(
                action, policy.action_clip_lo_il, policy.action_clip_hi_il
            )
            obs_rows.append(obs.copy())
            action_rows.append(action.copy())
        observations.append(np.asarray(obs_rows))
        actions.append(np.asarray(action_rows))
    obs_error = float(np.max(np.abs(np.asarray(observations) - observations[0])))
    action_error = float(np.max(np.abs(np.asarray(actions) - actions[0])))
    result = {
        "schema_version": "rosclaw_soccer.kick_frame_contract.v1",
        "yaws_rad": yaws,
        "observation_warmstart": warmstart,
        "observation_max_absolute_error": obs_error,
        "action_max_absolute_error": action_error,
        "tolerance": 1e-5,
        "status": "PASS" if max(obs_error, action_error) <= 1e-5 else "FAIL",
        "qualification": qualification.to_dict(),
        "physics_stepped": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "claim": "network_frame_equivariance_only_not_bilateral_kick_success",
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / "network.npz", observations=observations, actions=actions)
    result["network_trace_hash"] = hash_bytes((output / "network.npz").read_bytes())
    result["report_hash"] = hash_json(result)
    (output / "contract.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmstart", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(verify(asset_root=args.asset_root, output=args.output, warmstart=args.warmstart))
    )
