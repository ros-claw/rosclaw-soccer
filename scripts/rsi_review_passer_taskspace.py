"""Review one pinned paired task-space passer study; no new physics or training."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.pass_impulse_window import measure_first_pass_impulse
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def summarize(
    raw: dict, model, a, b, enabled: bool, commands: int, active: int, *, tempo: bool = False
) -> dict:
    torque = raw["torque"]
    parent = raw["parent_torque"] if not tempo else torque
    delta = raw["applied_torque_change"] if not tempo else np.zeros_like(torque)
    f = raw["ball_forces"].reshape(-1, 2, 3)
    q = raw["qpos"].reshape(-1, model.nq)
    v = raw["qvel"].reshape(-1, model.nv)
    n = len(torque)
    if (
        torque.shape != (n, 10, model.nu)
        or parent.shape != torque.shape
        or delta.shape != torque.shape
        or f.shape != (n * 10, 2, 3)
        or q.shape != (n * 10, model.nq)
        or v.shape != (n * 10, model.nv)
        or raw["body_failed"].shape != (n, 2)
        or not all(np.isfinite(x).all() for x in (torque, parent, delta, f, q, v))
        or not np.array_equal(torque - parent, delta)
        or np.any(delta[:, :, b.ai])
        or np.any(delta[:80])
        or (not enabled and np.any(delta))
        or commands != (max(n - 80, 0) * 10 if enabled and not tempo else 0)
        or not 0 <= active <= commands
    ):
        raise ValueError("complete own-passer-only actual motor accounting required")
    changed = int(np.any(delta[:, :, a.ai] != 0, axis=2).sum())
    if changed > active or np.max(np.abs(delta)) > 14.000001:
        raise ValueError("bounded active strike teacher required")
    impact = measure_first_pass_impulse(f, v[:, a.bv : a.bv + 3])
    start = impact["first_strong"]
    foot = (
        np.flatnonzero(np.max(f[start:, 1, :2], axis=1) > 10.0) + start
        if start is not None
        else np.array([], dtype=int)
    )
    nonfoot = (
        np.flatnonzero(f[start:, 1, 2] > 1.0) + start
        if start is not None
        else np.array([], dtype=int)
    )
    first_foot = int(foot[0]) if len(foot) else None
    first_nonfoot = int(nonfoot[0]) if len(nonfoot) else None
    foot_first = first_foot is not None and (first_nonfoot is None or first_foot < first_nonfoot)
    post = None
    if foot_first and first_foot + 100 < len(f):
        i = first_foot + 100
        post = dict(
            nonfoot_within_0p2s=bool(np.any(f[first_foot : i + 1, 1, 2] > 1)),
            ball_root_distance_m=float(
                np.linalg.norm(q[i, b.bq : b.bq + 3] - q[i, b.rq : b.rq + 3])
            ),
            relative_speed_mps=float(np.linalg.norm(v[i, b.bv : b.bv + 3] - v[i, b.rv : b.rv + 3])),
        )
    return dict(
        controls=n,
        impulse=impact,
        changed_native_commands=changed if not tempo else None,
        max_abs_delta_nm=float(np.max(np.abs(delta))) if not tempo else None,
        receiver_first_foot_step=first_foot,
        receiver_first_nonfoot_step=first_nonfoot,
        receiver_foot_first=foot_first,
        receiver_post_0p2s=post,
        final_body_failure=bool(raw["body_failed"][-1].any()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("physical-head", "scene-hash", "expected-head", "unit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    args = p.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or args.unit
        not in (
            "rosclaw-passer-taskspace-strike-b0-20261029.service",
            "rosclaw-passer-reference-tempo-b0-20261029.service",
        )
        or not 1 <= args.timeout_sec <= 7200
    ):
        raise ValueError("exact physical unit and unused external output required")
    receipt, scene = args.source / "complete.json", args.source / "scene.xml"
    deadline = time.monotonic() + args.timeout_sec
    while not receipt.is_file():
        state = subprocess.run(
            ["systemctl", "--user", "show", args.unit, "-p", "ActiveState"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if state not in ("ActiveState=active", "ActiveState=activating"):
            raise ValueError("physical source ended without complete receipt")
        if time.monotonic() >= deadline:
            raise TimeoutError("observer expired; never restart live source")
        time.sleep(15)
    if receipt.is_symlink() or scene.is_symlink() or digest(scene) != args.scene_hash:
        raise ValueError("exact physical scene required")
    source_hash = digest(receipt)
    source = json.loads(receipt.read_text())
    tempo = args.unit == "rosclaw-passer-reference-tempo-b0-20261029.service"
    if (
        source.get("status")
        != (
            "NEURAL_RELAY_PASSER_REFERENCE_TEMPO16_PENDING_OWN_REPLAY"
            if tempo
            else "NEURAL_RELAY_PASSER_STRIKE_TEACHER16_PENDING_OWN_REPLAY"
        )
        or source.get("source_head") != args.physical_head
        or len(source.get("records", [])) != 16
        or any(
            source.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("complete exact unpromoted paired physical source required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    a, b = [
        NativeVelocityLayout(model, robot_prefix=x, ball_body_name="soccer_ball", player_count=2)
        for x in ("a/", "b/")
    ]
    cases = {}
    for record in source["records"]:
        course = record["course"]
        if tempo:
            rate = course["arm"]["passer_reference_rate"]
            if type(rate) is not float or rate not in (1.0, 0.8):
                raise ValueError("exact preregistered tempo arms required")
            enabled = rate == 0.8
        else:
            enabled = course["arm"]["passer_strike_teacher"]
        if type(enabled) is not bool:
            raise ValueError("explicit paired strike arm required")
        identity = (course["context"], course["lateral_m"])
        item = record["artifacts"]["physics"]
        path = args.source / item["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(args.source.resolve())
            or digest(path) != item["hash"]
        ):
            raise ValueError("confined exact physical trajectory required")
        with np.load(path, allow_pickle=False) as z:
            raw = {
                k: z[k].copy()
                for k in (
                    "torque",
                    "ball_forces",
                    "qpos",
                    "qvel",
                    "body_failed",
                )
            }
            if not tempo:
                raw.update({k: z[k].copy() for k in ("parent_torque", "applied_torque_change")})
        if digest(path) != item["hash"] or enabled in cases.setdefault(identity, {}):
            raise ValueError("unchanged unique matched trajectory required")
        cases[identity][enabled] = (
            record,
            raw,
            summarize(
                raw,
                model,
                a,
                b,
                enabled,
                record["passer_strike_teacher_commands"],
                record["passer_strike_teacher_active"],
                tempo=tempo,
            ),
        )
    if len(cases) != 8 or any(set(pair) != {False, True} for pair in cases.values()):
        raise ValueError("eight complete paired identities required")
    rows = []
    for (context, side), pair in sorted(cases.items()):
        base, cand = pair[False], pair[True]
        if not all(
            np.array_equal(base[1][k][0], cand[1][k][0])
            for k in ("qpos", "qvel", "torque", "ball_forces")
        ):
            raise ValueError("same initial physical control required")
        common = min(base[2]["controls"], cand[2]["controls"])
        rows.append(
            dict(
                context=context,
                side=side,
                common_controls=common,
                baseline=base[2],
                candidate=cand[2],
                baseline_common_body_healthy=not bool(base[1]["body_failed"][:common].any()),
                candidate_common_body_healthy=not bool(cand[1]["body_failed"][:common].any()),
                baseline_handoff=base[0]["result"]["handoff_executed"],
                candidate_handoff=cand[0]["result"]["handoff_executed"],
                paired_passer_torque_changed_substeps=int(
                    np.any(
                        base[1]["torque"][:common, :, a.ai] != cand[1]["torque"][:common, :, a.ai],
                        axis=2,
                    ).sum()
                )
                if tempo
                else None,
            )
        )
    if digest(receipt) != source_hash or digest(scene) != args.scene_hash:
        raise ValueError("source changed during review")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status=(
                "PASSER_TEMPO_PAIRED_PHYSICAL_DIAGNOSIS_NOT_SUCCESS"
                if tempo
                else "PASSER_STRIKE_PAIRED_PHYSICAL_DIAGNOSIS_NOT_SUCCESS"
            ),
            tempo_reference_not_learned=tempo,
            source_hash=source_hash,
            source_head=args.expected_head,
            rows=rows,
            independently_replayed=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
