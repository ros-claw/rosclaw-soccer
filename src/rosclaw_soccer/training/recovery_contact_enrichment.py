"""CPU-MuJoCo contact enrichment for exact recovery failure memory.

MJX failure archives preserve policy context but their v2 schema predates
contact topology.  This module replays every exact G1 qpos/qvel through the
content-bound CPU MuJoCo model and emits an external, immutable diagnostic
report.  It never trains, promotes, renders, or authorizes hardware.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx import (
    compiled_mujoco_model_contract,
    validate_recovery_mjx_failure_state_manifest,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _body_contact_class(body_name: str) -> str:
    lowered = body_name.lower()
    if "ankle" in lowered or "foot" in lowered:
        return "FOOT"
    if "knee" in lowered:
        return "KNEE"
    if "wrist" in lowered or "hand" in lowered:
        return "HAND"
    if "elbow" in lowered or "shoulder" in lowered or "arm" in lowered:
        return "ARM"
    if any(token in lowered for token in ("pelvis", "waist", "torso")):
        return "TRUNK"
    return "OTHER_BODY"


def _name(mujoco: Any, model: Any, object_type: Any, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    return str(value) if value else f"unnamed_{index}"


def _support_aabb(
    *,
    support_xy: list[tuple[float, float]],
    com_xy: tuple[float, float],
) -> dict[str, Any]:
    if not support_xy:
        return {
            "point_count": 0,
            "minimum_xy_m": None,
            "maximum_xy_m": None,
            "centroid_xy_m": None,
            "com_aabb_signed_margin_m": None,
        }
    points = np.asarray(support_xy, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    centroid = points.mean(axis=0)
    com = np.asarray(com_xy, dtype=np.float64)
    signed_margin = float(
        min(
            com[0] - minimum[0],
            maximum[0] - com[0],
            com[1] - minimum[1],
            maximum[1] - com[1],
        )
    )
    return {
        "point_count": len(support_xy),
        "minimum_xy_m": [float(value) for value in minimum],
        "maximum_xy_m": [float(value) for value in maximum],
        "centroid_xy_m": [float(value) for value in centroid],
        "com_aabb_signed_margin_m": signed_margin,
    }


def enrich_recovery_failure_bank_contacts(
    *,
    failure_state_manifest_path: Path,
    scene_xml_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
) -> dict[str, Any]:
    """Enrich one exact failure bank with CPU-MuJoCo contact topology."""

    import mujoco

    manifest_path = failure_state_manifest_path.expanduser().resolve()
    scene_path = scene_xml_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if (
        not scene_path.is_file()
        or destination.exists()
        or destination == checkout
        or checkout in destination.parents
    ):
        raise ValueError("recovery contact-enrichment paths are invalid")
    manifest = validate_recovery_mjx_failure_state_manifest(manifest_path)
    archive_path = manifest_path.parent / str(manifest["state_archive"])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model_contract = compiled_mujoco_model_contract(model)
    if model_contract != manifest.get("compiled_model_contract"):
        raise ValueError("contact-enrichment model differs from the failure bank")
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("contact enrichment requires canonical G1 29-DoF physics")
    free_joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if free_joints.size != 1:
        raise ValueError("contact enrichment requires exactly one floating G1 root")
    root_body = int(model.jnt_bodyid[int(free_joints[0])])

    with np.load(archive_path, allow_pickle=False) as archive:
        qpos = np.array(archive["qpos"], dtype=np.float64, copy=True)
        qvel = np.array(archive["qvel"], dtype=np.float64, copy=True)
    state_count = int(manifest["collected_state_count"])
    selection_rows = manifest.get("selection_rows")
    if (
        qpos.shape != (state_count, 36)
        or qvel.shape != (state_count, 35)
        or not isinstance(selection_rows, list)
        or len(selection_rows) != state_count
    ):
        raise ValueError("contact enrichment failure-bank state contract is invalid")

    rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    for state_index in range(state_count):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[state_index]
        data.qvel[:] = qvel[state_index]
        mujoco.mj_forward(model, data)
        robot_com = np.asarray(data.subtree_com[root_body], dtype=np.float64)
        contacts: list[dict[str, Any]] = []
        ground_classes: set[str] = set()
        grounded_foot_sides: set[str] = set()
        support_xy: list[tuple[float, float]] = []
        vertical_force_total = 0.0
        center_of_pressure_numerator: NDArray[np.float64] = np.zeros(
            2, dtype=np.float64
        )
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            geom1_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            geom2_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            body1_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body2)
            force: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            normal_force = max(float(force[0]), 0.0)
            position = np.asarray(contact.pos, dtype=np.float64)
            penetrating = bool(float(contact.dist) <= 0.0)
            world_body = body1 == 0 or body2 == 0
            robot_body_name = body2_name if body1 == 0 else body1_name
            body_class = _body_contact_class(robot_body_name) if world_body else "SELF_OR_OBJECT"
            if world_body and penetrating:
                ground_classes.add(body_class)
                lowered_robot_body = robot_body_name.lower()
                if body_class == "FOOT" and "left" in lowered_robot_body:
                    grounded_foot_sides.add("LEFT")
                if body_class == "FOOT" and "right" in lowered_robot_body:
                    grounded_foot_sides.add("RIGHT")
                support_xy.append((float(position[0]), float(position[1])))
                vertical_force_total += normal_force
                center_of_pressure_numerator += normal_force * position[:2]
            contacts.append(
                {
                    "contact_index": contact_index,
                    "geom_pair": [geom1_name, geom2_name],
                    "body_pair": [body1_name, body2_name],
                    "robot_body_class": body_class,
                    "distance_m": float(contact.dist),
                    "penetrating": penetrating,
                    "position_xyz_m": [float(value) for value in position],
                    "normal_force_n": normal_force,
                }
            )
        class_counts.update(ground_classes)
        topology = "+".join(sorted(ground_classes)) if ground_classes else "NO_GROUND_CONTACT"
        topology_counts[topology] += 1
        center_of_pressure = (
            [float(value) for value in center_of_pressure_numerator / vertical_force_total]
            if vertical_force_total > 1.0e-6
            else None
        )
        root_quaternion = qpos[state_index, 3:7]
        quaternion_norm = float(np.linalg.norm(root_quaternion))
        if not math.isfinite(quaternion_norm) or quaternion_norm < 1.0e-8:
            raise ValueError("contact enrichment encountered an invalid root quaternion")
        root_quaternion = root_quaternion / quaternion_norm
        upright = float(2.0 * (root_quaternion[0] ** 2 + root_quaternion[3] ** 2) - 1.0)
        rows.append(
            {
                "state_index": state_index,
                "state_identity": selection_rows[state_index]["state_identity"],
                "posture_proxy": selection_rows[state_index].get("posture_proxy"),
                "event_window_proxy": selection_rows[state_index].get("event_window_proxy"),
                "angular_momentum_proxy": selection_rows[state_index].get("angular_momentum_proxy"),
                "pelvis_height_m": float(qpos[state_index, 2]),
                "upright_projection": upright,
                "root_linear_speed_mps": float(np.linalg.norm(qvel[state_index, :3])),
                "root_angular_speed_rad_s": float(np.linalg.norm(qvel[state_index, 3:6])),
                "robot_com_xyz_m": [float(value) for value in robot_com],
                "contact_count": len(contacts),
                "penetrating_contact_count": sum(
                    contact["penetrating"] is True for contact in contacts
                ),
                "ground_contact_classes": sorted(ground_classes),
                "grounded_foot_sides": sorted(grounded_foot_sides),
                "ground_contact_topology": topology,
                "bilateral_foot_support_proxy": grounded_foot_sides == {"LEFT", "RIGHT"},
                "nonfoot_ground_contact": any(value != "FOOT" for value in ground_classes),
                "support_aabb": _support_aabb(
                    support_xy=support_xy,
                    com_xy=(float(robot_com[0]), float(robot_com[1])),
                ),
                "normal_force_total_n": vertical_force_total,
                "center_of_pressure_xy_m": center_of_pressure,
                "contacts": contacts,
            }
        )

    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_contact_enrichment.v1",
        "failure_state_manifest_hash": manifest["report_hash"],
        "failure_state_manifest_file_hash": hash_bytes(manifest_path.read_bytes()),
        "failure_state_archive_hash": manifest["state_archive_hash"],
        "scene_xml_hash": hash_bytes(scene_path.read_bytes()),
        "compiled_model_contract": model_contract,
        "state_count": state_count,
        "state_rows": rows,
        "contact_topology_counts": dict(sorted(topology_counts.items())),
        "ground_contact_class_state_counts": dict(sorted(class_counts.items())),
        "contact_enrichment_complete": True,
        "contact_truth_backend": "CPU_MUJOCO_FORWARD",
        "source_bank_posture_stratification_complete": bool(
            manifest.get("stratification_complete")
        ),
        "claim_boundary": "EXACT_STATE_STATIC_CONTACT_ENRICHMENT_NOT_POST_DIVE_ROLLOUT",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    _atomic_json(destination / "contact-enrichment.json", report)
    return report


def validate_recovery_contact_enrichment_report(path: Path) -> dict[str, Any]:
    """Validate a content-bound CPU contact-enrichment report."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery contact-enrichment report is invalid")
    declared = payload.pop("report_hash", None)
    rows = payload.get("state_rows")
    state_count = payload.get("state_count")
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_contact_enrichment.v1"
        or declared != hash_json(payload)
        or not isinstance(state_count, int)
        or isinstance(state_count, bool)
        or state_count <= 0
        or not isinstance(rows, list)
        or len(rows) != state_count
        or [row.get("state_index") for row in rows if isinstance(row, dict)]
        != list(range(state_count))
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("state_identity"), str)
            or not row["state_identity"].startswith("sha256:")
            or not isinstance(row.get("contacts"), list)
            or not isinstance(row.get("ground_contact_classes"), list)
            or not isinstance(row.get("grounded_foot_sides"), list)
            or not isinstance(row.get("support_aabb"), dict)
            for row in rows
        )
        or payload.get("contact_enrichment_complete") is not True
        or payload.get("contact_truth_backend") != "CPU_MUJOCO_FORWARD"
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery contact-enrichment report is invalid")
    payload["report_hash"] = declared
    return payload


__all__ = [
    "enrich_recovery_failure_bank_contacts",
    "validate_recovery_contact_enrichment_report",
]
