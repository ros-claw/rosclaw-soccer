"""Deterministic SIM-only closed-loop audit for public SONIC variants."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_runup import (
    G1SonicModelVariant,
    G1SonicRunupConfig,
    G1SonicRunupController,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1SonicVariantTrial:
    model_variant: G1SonicModelVariant
    qualification_hash: str
    reference_digest: str
    finite_state: bool
    final_forward_position_m: float
    minimum_pelvis_height_m: float
    maximum_root_angular_speed_rad_s: float
    maximum_absolute_joint_position_rad: float
    physics_steps: int


def audit_g1_sonic_variants(
    *,
    model_root: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    control_frames: int = 100,
) -> dict[str, Any]:
    """Run both frozen neural policies through identical native MuJoCo physics."""

    import mujoco

    from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents or output.exists():
        raise ValueError("SONIC audit requires a new output outside the source checkout")
    if not 50 <= control_frames <= 150:
        raise ValueError("SONIC audit control frames must be in [50, 150]")
    root = asset_root.expanduser().resolve()
    model = build_g1_stadium_model(root)
    trials: list[G1SonicVariantTrial] = []
    for variant in ("low_latency", "sonic_v1_1"):
        data = mujoco.MjData(model)
        data.qpos[:7] = (0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0)
        controller = G1SonicRunupController(
            model_root,
            G1SonicRunupConfig(execution_duration_sec=3.0, model_variant=variant),
        )
        data.qpos[7:36] = controller.default_angles
        mujoco.mj_forward(model, data)
        controller.reset(data)
        minimum_pelvis = float(data.qpos[2])
        maximum_root_angular = 0.0
        maximum_joint = float(np.max(np.abs(data.qpos[7:36])))
        for frame in range(control_frames):
            controller.update(data, frame)
            for _ in range(10):
                data.ctrl[:] = controller.torque(data)
                mujoco.mj_step(model, data)
            controller.observe(data)
            minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
            maximum_root_angular = max(
                maximum_root_angular,
                float(np.linalg.norm(data.qvel[3:6])),
            )
            maximum_joint = max(maximum_joint, float(np.max(np.abs(data.qpos[7:36]))))
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        )
        trials.append(
            G1SonicVariantTrial(
                model_variant=variant,
                qualification_hash=controller.qualification.qualification_hash,
                reference_digest=controller.reference_digest,
                finite_state=finite,
                final_forward_position_m=float(data.qpos[0]),
                minimum_pelvis_height_m=minimum_pelvis,
                maximum_root_angular_speed_rad_s=maximum_root_angular,
                maximum_absolute_joint_position_rad=maximum_joint,
                physics_steps=control_frames * 10,
            )
        )
    low, current = trials
    passed = bool(
        all(item.finite_state for item in trials)
        and min(item.minimum_pelvis_height_m for item in trials) >= 0.60
        and max(item.maximum_root_angular_speed_rad_s for item in trials) <= 3.50
        and all(
            math.isfinite(value)
            for item in trials
            for value in (
                item.final_forward_position_m,
                item.minimum_pelvis_height_m,
                item.maximum_root_angular_speed_rad_s,
                item.maximum_absolute_joint_position_rad,
            )
        )
    )
    payload: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.g1_sonic_variant_audit.v1",
        "physics_backend": "mujoco_cpu",
        "physics_scene_hash": g1_stadium_scene_hash(root),
        "model_root_name": model_root.expanduser().resolve().name,
        "control_frames": control_frames,
        "trials": [asdict(item) for item in trials],
        "sonic_v1_1_stability_deltas": {
            "minimum_pelvis_height_m": (
                current.minimum_pelvis_height_m - low.minimum_pelvis_height_m
            ),
            "maximum_root_angular_speed_rad_s": (
                current.maximum_root_angular_speed_rad_s - low.maximum_root_angular_speed_rad_s
            ),
        },
        "comparison_interpretation": (
            "MEASURED_DELTA_ONLY_NO_VARIANT_PROMOTION; fixed-horizon results do not "
            "authorize replacing a qualified run-up policy"
        ),
        "passed": passed,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    payload["report_hash"] = hash_json(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


__all__ = ["G1SonicVariantTrial", "audit_g1_sonic_variants"]
