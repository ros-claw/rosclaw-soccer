"""Check CFtrack codec/gain compatibility on replayed two-player boundaries."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.cftrack_transfer_preflight import compare_cftrack_transfer
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "audit", "cf-config", "velocity-config", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    for flag in (
        "audit-hash",
        "cf-config-hash",
        "velocity-config-hash",
        "scene-hash",
        "expected-head",
    ):
        p.add_argument("--" + flag, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source, _, source_hash = report(
        a.source,
        a.audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_ANKLE_FREEZE16_PENDING_OWN_REPLAY",
        worlds=16,
    )
    if any(
        path.is_symlink() or digest(path) != expected
        for path, expected in (
            (a.cf_config, a.cf_config_hash),
            (a.velocity_config, a.velocity_config_hash),
        )
    ):
        raise ValueError("exact pinned reference and executed velocity configs required")
    if source["input_file_hashes"].get(str(a.velocity_config)) != a.velocity_config_hash:
        raise ValueError("velocity gain config must be bound to actual source")
    cf, velocity = (yaml.safe_load(x.read_text()) for x in (a.cf_config, a.velocity_config))
    lab2mj = np.asarray(cf["lab2mj"], dtype=np.int64)
    mj2lab = np.asarray(cf["mj2lab"], dtype=np.int64)
    scene = a.source / "scene.xml"
    scene_hash = digest(scene)
    if scene.is_symlink() or scene_hash != a.scene_hash:
        raise ValueError("exact independently reviewed scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for row in source["records"]:
        boundary = row["controls"] if row["handoff"] is None else row["handoff"]["control"]
        if boundary < 1:
            raise ValueError("observed pre-boundary target required")
        with np.load(bind(a.source, row, "physics"), allow_pickle=False) as z:
            q, v = z["qpos"][boundary - 1, -1], z["qvel"][boundary - 1, -1]
            previous = z["target_b"][boundary - 1]
            info = compare_cftrack_transfer(
                previous_target=previous,
                joint_position=q[layout.qi],
                joint_velocity=v[layout.vi],
                old_stiffness=np.asarray(velocity["stiffness"], dtype=np.float64),
                old_damping=np.asarray(velocity["damping"], dtype=np.float64),
                cf_stiffness=np.asarray(cf["kp_lab"], dtype=np.float64)[lab2mj],
                cf_damping=np.asarray(cf["kd_lab"], dtype=np.float64)[lab2mj],
                default_lab=np.asarray(cf["default_angles_lab"], dtype=np.float32),
                scale_lab=np.asarray(cf["action_scale_lab"], dtype=np.float32),
                mj2lab=mj2lab,
                lab2mj=lab2mj,
            )
        info["limiting_joint"] = G1_DDS_JOINT_NAMES[info["limiting_joint_index"]]
        rows.append(
            dict(
                index=row["index"],
                course=row["course"],
                boundary_control=boundary,
                boundary_kind="existing_handoff"
                if row["handoff"] is not None
                else "observed_end_not_handoff",
                body_or_ball_reset=False,
                **info,
            )
        )
    if digest(scene) != scene_hash or any(
        digest(path) != expected
        for path, expected in (
            (a.cf_config, a.cf_config_hash),
            (a.velocity_config, a.velocity_config_hash),
        )
    ):
        raise ValueError("qualified model or configuration changed")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="CFTRACK_SHARED_BOUNDARY_DIAGNOSTIC_ONLY",
            source_hash=source_hash,
            audit_hash=a.audit_hash,
            scene_hash=scene_hash,
            cf_config_hash=a.cf_config_hash,
            velocity_config_hash=a.velocity_config_hash,
            code_head=a.expected_head,
            new_native_steps=0,
            new_policy_calls=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
            rows=rows,
        ),
    )


if __name__ == "__main__":
    main()
