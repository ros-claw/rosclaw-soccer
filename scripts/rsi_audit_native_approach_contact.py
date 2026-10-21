"""Read-only independent raw audit of the new approach/contact/recovery probe.

Reconstruct measured clocks, safety, terminal stop and outcomes. For normal
probes also compare EVERY pre-handoff sample to original full-frame native
foundation rollouts, not just the first24 controls. Zero integration/inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco

    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hash", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    receipt = a.root / "complete.json"
    if a.output.exists() or digest(receipt) != a.hash:
        raise ValueError("new output and exact completed probe required")
    r = json.loads(receipt.read_text())
    if r["status"] != "NATIVE_APPROACH_MEASURED_CONTACT_RECOVERY_PROBE_NOT_LEARNER" or any(
        r[k] is not True
        for k in (
            "continuous_handoff_exact",
            "strict_replays_exact",
            "full_repeats_exact",
            "stopped_worlds_frozen",
        )
    ):
        raise ValueError("passing actual probe required")
    declaration = a.root / "declaration.json"
    d = json.loads(declaration.read_text())
    pins = {**r["input_file_hashes"], str(receipt): a.hash, str(declaration): digest(declaration)}
    for file in (
        Path(__file__).resolve(),
        Path("src/rosclaw_soccer/rsi/native_contact_window.py").resolve(),
    ):
        pins[str(file)] = digest(file)

    def pinned(suffix):
        found = [Path(f) for f in pins if f.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("exact source input required: " + suffix)
        return found[0]

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    check_pins(pins)
    bank = load(pinned("/precontact-resets.npz"))
    model = mujoco.MjModel.from_binary_path(
        str(pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb"))
    )
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    plans = [
        load(
            pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + n.removesuffix(".npz")
                + "-plan.npz"
            )
        )
        for n in TRAIN_MOTION_NAMES
    ]
    qi = [*range(layout.rq, layout.rq + 7), *layout.qi, *range(layout.bq, layout.bq + 7)]
    vi = [*range(layout.rv, layout.rv + 6), *layout.vi, *range(layout.bv, layout.bv + 6)]
    normal = d.get("fault_injection") is None
    full_baselines = {}
    seen, first, rows = set(), {}, []
    live = head_prefix = 0
    for result in r["results"]:
        lead, repeat = result["lead"], result["repeat"]
        if (lead, repeat) in seen or lead not in d["leads"] or repeat not in (0, 1):
            raise ValueError("unexpected/duplicate declared course repeat")
        seen.add((lead, repeat))
        path = a.root / f"lead{lead}-repeat{repeat}.npz"
        h = r["raw_hashes"][path.name]
        if digest(path) != h:
            raise ValueError("raw probe changed")
        pins[str(path)] = h
        z = load(path)
        if repeat == 0:
            first[lead] = z
        else:
            old = first.pop(lead)
            if set(old) != set(z) or any(not np.array_equal(old[k], z[k]) for k in old):
                raise ValueError("whole neural/physics repeat not exact")
        if len(result["worlds"]) != d["worlds"] or z["bank_index"].shape != (d["worlds"],):
            raise ValueError("complete declared worlds required")
        reconstructed = []
        for w, result_world in enumerate(result["worlds"]):
            index = int(z["bank_index"][w])
            if (
                bank["lead_controls"][index] != lead
                or result_world["world"] != w
                or int(bank["source_id"][index]) != w % 8
            ):
                raise ValueError("explicit balanced TRAIN identity required")
            chunks = {key: [] for key in ("qpos", "qvel", "forces")}
            counts = {}
            for part in ("head_", "tail_"):
                if part + "valid" not in z:
                    counts[part] = 0
                    continue
                mask = z[part + "valid"][:, w]
                count = int(mask.sum())
                counts[part] = count
                if mask.dtype != np.bool_ or not np.array_equal(mask, np.arange(len(mask)) < count):
                    raise ValueError("noncontiguous actual execution")
                for key, width in (("qpos", 43), ("qvel", 41), ("forces", 3)):
                    if count:
                        chunks[key].append(z[part + key][:count, w].reshape(-1, width))
                    if (
                        key != "forces"
                        and 0 < count < len(mask)
                        and not np.array_equal(
                            z[part + key][count:, w],
                            np.broadcast_to(
                                z[part + key][count - 1, w, -1], z[part + key][count:, w].shape
                            ),
                        )
                    ):
                        raise ValueError("stopped body state continued changing")
                for key in ("observation", "phase_before"):
                    if 0 < count < len(mask) and not np.array_equal(
                        z[part + key][count:, w],
                        np.broadcast_to(z[part + key][count, w], z[part + key][count:, w].shape),
                    ):
                        raise ValueError("stopped observation/clock continued changing")
            hc, tc = counts["head_"], counts["tail_"]
            if hc < 1:
                raise ValueError("head must physically execute")
            q, v, f = (np.concatenate(chunks[k]) for k in ("qpos", "qvel", "forces"))
            q, v = q[:, qi], v[:, vi]
            first_hits = np.flatnonzero(
                (z["head_forces"][:hc, w, :, :2] > 10).any(axis=-1).reshape(-1)
            )
            contact = int(first_hits[0] // 10) if len(first_hits) else -1
            frame = int(bank["reference_frame"][index])
            # Reconstruct remaining source length from its pinned plan, not row metadata.
            horizon = len(plans[int(bank["source_id"][index])]["joint"]) - frame
            headq = z["head_qpos"][:hc, w]
            bodybad = (
                (headq[:, :, layout.rq + 2] < 0.65)
                | (
                    1 - 2 * (headq[:, :, layout.rq + 4] ** 2 + headq[:, :, layout.rq + 5] ** 2)
                    < 0.9
                )
            ).any(axis=1)
            bad = bodybad | z["head_guards"][:hc, w].any(axis=1)
            head_failed = bool(bad.any())
            expected_hc = min(horizon, contact + 15) if contact >= 0 else horizon
            if (
                (head_failed and not np.array_equal(np.flatnonzero(bad), [hc - 1]))
                or (not head_failed and hc != expected_hc)
                or not np.array_equal(z["head_terminated"][:hc, w], bad)
                or not np.array_equal(
                    np.flatnonzero(z["head_truncated"][:, w]), [] if head_failed else [hc - 1]
                )
            ):
                raise ValueError("head ignored measured boundary or body failure")
            if not np.array_equal(z["head_frame_before"][:hc, w], np.arange(frame, frame + hc)):
                raise ValueError("reference frame repeated or extended")
            if not np.array_equal(
                z["head_first_contact_control"][:, w],
                np.where(np.arange(len(z["head_valid"])) >= contact, contact, -1)
                if contact >= 0
                else np.full(len(z["head_valid"]), -1),
            ):
                raise ValueError("contact history used a future label")
            if head_failed != (tc == 0):
                raise ValueError("tail must follow exactly completed nonfailed heads")
            if tc:
                tailq = z["tail_qpos"][:tc, w]
                badtail = (
                    (tailq[:, :, layout.rq + 2] < 0.65)
                    | (
                        1 - 2 * (tailq[:, :, layout.rq + 4] ** 2 + tailq[:, :, layout.rq + 5] ** 2)
                        < 0.9
                    )
                ).any(axis=1)
                failed_tail = bool(badtail.any())
                if (
                    (failed_tail and not np.array_equal(np.flatnonzero(badtail), [tc - 1]))
                    or (not failed_tail and tc != 60)
                    or not np.array_equal(z["tail_terminated"][:tc, w], badtail)
                    or not np.array_equal(
                        np.flatnonzero(z["tail_completed"][:, w]), [] if failed_tail else [tc - 1]
                    )
                    or z["tail_observation_before"][:tc, w, 98:].any()
                    or z["tail_observation_before"][:tc, w, 6:9].any()
                ):
                    raise ValueError("tail failed to stop or used reference/ball commands")
            else:
                failed_tail = False
            actual = dict(
                world=w,
                bank_index=index,
                head_controls=hc,
                first_contact_control=contact,
                reference_horizon=horizon,
                head_failed=head_failed,
                tail_failed=head_failed or failed_tail,
                tail_controls=tc,
                forbidden=bool((f[:, 2] > 1).any()),
                stable=bool(
                    q[:, 2].min() >= 0.65 and (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)).min() >= 0.9
                ),
                postcontact=native_contact_window(
                    force=f[:, :2],
                    canonical_qpos=q,
                    canonical_qvel=v,
                    goal=bank["goal_world"][index],
                    intended_foot=int(bank["expected_foot"][index]),
                ),
            )
            if actual != result_world:
                raise ValueError("outcome or clock differs from independent raw reconstruction")
            if normal:
                world = int(bank["world"][index])
                if world not in full_baselines:
                    full_baselines[world] = load(
                        pinned(
                            f"/rsi-body-bound-contact-episodes-20261021/world{world}-repeat0.npz"
                        )
                    )
                base = full_baselines[world]
                pairs = [
                    (
                        z["head_qpos"][:hc, w][:, :, qi],
                        base["canonical_qpos_after"][frame : frame + hc],
                    ),
                    (
                        z["head_qvel"][:hc, w][:, :, vi],
                        base["canonical_qvel_after"][frame : frame + hc],
                    ),
                    (z["head_forces"][:hc, w, :, :2], base["foot_force"][frame : frame + hc]),
                    (z["head_forces"][:hc, w, :, 2], base["nonfoot_force"][frame : frame + hc]),
                ]
                pairs += [
                    (z["head_" + k][:hc, w], base[b][frame : frame + hc])
                    for k, b in (
                        ("observation_before", "observation"),
                        ("phase_before", "phase_before"),
                        ("action", "action"),
                        ("target", "target_rad"),
                    )
                ]
                if any(not np.array_equal(x, y) for x, y in pairs):
                    raise ValueError(
                        "long reference prefix differs from original frame-zero native run"
                    )
                head_prefix += hc
            reconstructed.append(actual)
            live += (hc + tc) * 10
        rows.append(dict(lead=lead, repeat=repeat, worlds=reconstructed))
    if seen != {(lead, rep) for lead in d["leads"] for rep in (0, 1)} or first:
        raise ValueError("missing declared lead/repeat")
    if (
        live != r["live_native_steps"]
        or live != r["replay_native_steps"]
        or live != r["neural_rows"] * 10
    ):
        raise ValueError("physical/replay/inference accounting differs")
    check_pins(pins)
    write_once(
        a.output,
        dict(
            schema="soccer.rsi.native_approach_independent_audit.v1",
            input_file_hashes=pins,
            results=rows,
            raw_files=len(seen),
            original_full_reference_prefix_controls_exact=head_prefix if normal else None,
            measured_handoff_and_body_failure_exact=True,
            whole_repeats_exact=True,
            new_physics_steps=0,
            new_neural_rows=0,
            optimizer_updates=0,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
