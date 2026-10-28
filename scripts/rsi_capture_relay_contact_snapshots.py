"""Qualify retrospective native snapshots for later SIM-only contact research.

Recorded torque replay is not an independent neural rollout or a policy. It
only qualifies whether archived trajectories can reconstruct integration state.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.relay_snapshot_cases import select_relay_snapshot_cases
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "audit-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--nondefault", action="store_true")
    p.add_argument("--contexts", type=int, nargs=4, default=(2, 3, 5, 7))
    a = p.parse_args()
    clean(a.expected_head)
    root, audit = a.source / "source", a.source / "audit/complete.json"
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external snapshot output required")
    source, _, source_hash = report(
        root,
        audit,
        a.audit_hash,
        "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY",
        worlds=32,
    )
    if source_hash != a.source_hash:
        raise ValueError("exact independently replayed source required")
    scene = root / "scene.xml"
    xmls = [
        Path(x)
        for x in source["input_file_hashes"]
        if x.endswith("/cached-standard/declared-scene.xml")
    ]
    if (
        len(xmls) != 1
        or digest(xmls[0]) != source["input_file_hashes"][str(xmls[0])]
        or scene.is_symlink()
        or scene.read_text() != shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
    ):
        raise ValueError("exact audited shared-world scene required")
    scene_hash = digest(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nq != 79 or model.nv != 76 or model.nu != 58 or model.opt.timestep != 0.002:
        raise ValueError("expected native two-player standard-ball world required")
    candidates = select_relay_snapshot_cases(
        source["records"], nondefault=a.nondefault, contexts=tuple(a.contexts)
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            audit_hash=a.audit_hash,
            scene_hash=scene_hash,
            capture_control=140,
            contexts=list(a.contexts),
            verification_controls=1,
            retrospective_development_selection=True,
            nondefault_development_expansion=a.nondefault,
            saved_torque_is_not_policy=True,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    rows = []
    try:
        for r in candidates:
            initial, path = bind(root, r, "initial"), bind(root, r, "physics")
            if initial is None or path is None:
                raise ValueError("bound original actual initial and physics artifacts required")
            with (
                np.load(initial, allow_pickle=False) as ini,
                np.load(path, allow_pickle=False) as raw,
            ):
                if (
                    len(raw["qpos"]) <= 140
                    or np.any(raw["ball_forces"][:140, :, 1, :] > 1)
                    or not np.any(raw["ball_forces"][:140, :, 0, :2] > 10)
                ):
                    raise ValueError(
                        "full original post-pass pre-receiver-contact boundary required"
                    )
                state = mujoco.MjData(model)
                state.qpos[:] = ini["qpos"]
                state.qvel[:] = ini["qvel"]
                for i in range(140):
                    mujoco.mj_forward(model, state)
                    for j in range(10):
                        state.ctrl[:] = raw["torque"][i, j]
                        mujoco.mj_step(model, state)
                        if not (
                            np.array_equal(state.qpos, raw["qpos"][i, j])
                            and np.array_equal(state.qvel, raw["qvel"][i, j])
                        ):
                            raise ValueError(f"exact original torque replay diverged at {i}:{j}")
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                snapshot = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
                mujoco.mj_getState(model, state, snapshot, spec)
                restored = mujoco.MjData(model)
                mujoco.mj_setState(model, restored, snapshot, spec)
                mujoco.mj_forward(model, restored)
                for j in range(10):
                    restored.ctrl[:] = raw["torque"][140, j]
                    mujoco.mj_step(model, restored)
                    if not (
                        np.array_equal(restored.qpos, raw["qpos"][140, j])
                        and np.array_equal(restored.qvel, raw["qvel"][140, j])
                    ):
                        raise ValueError(f"restored integration state diverged at 140:{j}")
                if (
                    digest(initial) != r["artifacts"]["initial"]["hash"]
                    or digest(path) != r["artifacts"]["physics"]["hash"]
                ):
                    raise ValueError("actual trace changed during reconstruction")
                out = a.output / f"case-{r['index']}.npz"
                np.savez_compressed(out, state=snapshot)
                rows.append(
                    dict(
                        index=r["index"],
                        context=r["course"]["context"],
                        side=r["course"]["lateral_m"],
                        receiver_lateral_offset_m=r["course"]["arm"]["receiver_lateral_offset_m"],
                        original_physics_hash=digest(path),
                        snapshot=out.name,
                        snapshot_hash=digest(out),
                        replayed_steps=1400,
                        restored_validation_steps=10,
                    )
                )
                print(json.dumps({"index": r["index"], "qualified": len(rows)}), flush=True)
        if (
            digest(root / "complete.json") != a.source_hash
            or digest(audit) != a.audit_hash
            or digest(scene) != scene_hash
        ):
            raise ValueError("source changed during snapshot reconstruction")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
                    if a.nondefault
                    else "RELAY_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY"
                ),
                source_head=a.expected_head,
                source_hash=a.source_hash,
                audit_hash=a.audit_hash,
                scene_hash=scene_hash,
                contexts=list(a.contexts),
                rows=rows,
                independent_fresh=False,
                saved_torque_is_not_policy=True,
                optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="RELAY_PRECONTACT_SNAPSHOT_QUALIFICATION_FAILED",
                source_head=a.expected_head,
                completed=len(rows),
                error_type=type(error).__name__,
                error=str(error),
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
